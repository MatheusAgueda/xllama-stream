// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

// Built only with the llama.cpp backend (it links ggml); the ORT-only UWP
// variant compiles this file to nothing.
#ifdef XLLAMA_USE_LLAMA

    #include "xllama/ggml_d3d12.h"

    #include "ggml-cpu.h"

    #include <algorithm>
    #include <cmath>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <ctime>
    #include <limits>
    #include <string>
    #include <vector>

namespace xllama {

// --- Pure rules (host-tested) ---

bool d3d12_weight_type_supported(ggml_type t) {
    return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K ||
           t == GGML_TYPE_IQ3_S || t == GGML_TYPE_IQ4_XS || t == GGML_TYPE_Q5_K ||
           t == GGML_TYPE_F32;
}

D3d12Dispatch d3d12_mm_dispatch(std::int64_t n, std::int64_t ncols) {
    D3d12Dispatch d;
    if (n <= 0 || ncols < 0)
        return d;
    if (ncols == 0) { // no output rows in this ubatch: a no-op, still ours
        d.ok = true;
        return d;
    }
    const std::int64_t gx = (n + kD3d12MmvRows - 1) / kD3d12MmvRows;
    d.ok = gx <= kD3d12MaxGroups && ncols <= kD3d12MaxGroups;
    if (d.ok) {
        d.groups_x = static_cast<std::uint32_t>(gx);
        d.groups_y = static_cast<std::uint32_t>(ncols);
    }
    return d;
}

int d3d12_mm_threads(std::int64_t k) {
    return k / kD3d12Chunk >= kD3d12LongKChunks ? kD3d12MmvThreadsLong : kD3d12MmvThreadsShort;
}

namespace {
std::uint32_t floor_pow2(std::uint32_t v) {
    std::uint32_t p = 1;
    while (p * 2u <= v)
        p *= 2u;
    return p;
}
} // namespace

D3d12Geom d3d12_geom_for(const D3d12GeomReq& req, std::uint32_t nchunk, std::uint32_t threads) {
    D3d12Geom g;
    const std::uint32_t lanes = std::max(threads / 16u, 1u);
    std::uint32_t rsl = req.rsl;
    if (rsl == 0) // auto: only the lanes a short K leaves idle
        rsl = nchunk > 0 && nchunk < lanes ? lanes / nchunk : 1u;
    g.rsl = std::min(floor_pow2(std::max(rsl, 1u)), floor_pow2(lanes));
    g.rt = std::clamp(req.rt, 1u, kD3d12GeomMaxRt);
    return g;
}

std::uint32_t d3d12_geom_parse(const char* s, std::uint32_t fallback) {
    if (!s || !*s)
        return fallback;
    if (std::strcmp(s, "auto") == 0)
        return 0;
    char* end = nullptr;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (end == s || *end != '\0' || v > 0xffu)
        return fallback;
    return static_cast<std::uint32_t>(v);
}

bool d3d12_mm_supported(const D3d12MatmulDesc& d) {
    return d3d12_weight_type_supported(d.src0_type) && d.src0_in_weight_buffer &&
           d.src0_contiguous && d.src1_contiguous && d.src1_type == GGML_TYPE_F32 &&
           d.dst_type == GGML_TYPE_F32 && d.ne00 > 0 && d.ne00 % kD3d12Chunk == 0 &&
           d.ne00 == d.ne10 && d.ne02 == 1 && d.ne03 == 1 && d.ne12 == 1 && d.ne13 == 1 &&
           d3d12_mm_dispatch(d.ne01, d.ne11).ok;
}

bool d3d12_get_rows_type_supported(ggml_type t) {
    return t == GGML_TYPE_Q6_K;
}

bool d3d12_get_rows_supported(const D3d12GetRowsDesc& d) {
    return d3d12_get_rows_type_supported(d.src0_type) && d.src0_in_weight_buffer &&
           d.src0_contiguous && d.src1_contiguous && d.src1_type == GGML_TYPE_I32 &&
           d.dst_type == GGML_TYPE_F32 && d.ne00 > 0 && d.ne00 % kD3d12Chunk == 0 && d.ne02 == 1 &&
           d.ne03 == 1 && d.ne10 >= 0 && d.ne10 <= kD3d12MaxGroups && d.ne11 == 1 && d.ne12 == 1 &&
           d.ne00 / kD3d12Chunk <= kD3d12MaxGroups;
}

bool d3d12_place_tied_embedding(ggml_type embd_type, bool has_output_weight) {
    return !has_output_weight && d3d12_get_rows_type_supported(embd_type);
}

namespace {
bool g_output_on_cpu = false;
} // namespace

void gguf_set_output_on_cpu(bool on) {
    g_output_on_cpu = on;
}

bool gguf_output_on_cpu() {
    return g_output_on_cpu;
}

// --- Host emulation of shaders/ggml_d3d12_mmv_*.hlsl ---

namespace {

// Little-endian loads with the shaders' 2-byte-aligned dword trick.
std::uint32_t ld32(const std::uint8_t* w, std::uint32_t a) {
    auto dw = [&](std::uint32_t at) {
        std::uint32_t v;
        std::memcpy(&v, w + at, 4);
        return v;
    };
    const std::uint32_t b = a & ~3u;
    const std::uint32_t lo = dw(b);
    if ((a & 3u) == 0u)
        return lo;
    return (lo >> 16) | (dw(b + 4u) << 16);
}

std::uint32_t ld16(const std::uint8_t* w, std::uint32_t a) {
    std::uint32_t v;
    std::memcpy(&v, w + (a & ~3u), 4);
    return (a & 2u) != 0u ? (v >> 16) : (v & 0xffffu);
}

float h2f(std::uint32_t h) {
    return ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(h));
}

std::int32_t nib(std::uint32_t word, std::uint32_t i, std::uint32_t shift) {
    return static_cast<std::int32_t>((word >> (8u * i + shift)) & 0xFu);
}

std::int32_t sbyte(std::uint32_t word, std::uint32_t i) {
    return static_cast<std::int8_t>((word >> (8u * i)) & 0xffu);
}

void q4k_scale_min(std::uint32_t j, const std::uint8_t* sc, std::int32_t* d, std::int32_t* m) {
    std::uint32_t dd, mm;
    if (j < 4u) {
        dd = sc[j] & 63u;
        mm = sc[j + 4u] & 63u;
    } else {
        dd = (sc[j + 4u] & 0xFu) | ((sc[j - 4u] >> 6) << 4);
        mm = (sc[j + 4u] >> 4) | ((sc[j] >> 6) << 4);
    }
    *d = static_cast<std::int32_t>(dd);
    *m = static_cast<std::int32_t>(mm);
}

constexpr std::uint32_t kIq3sGrid[512] = {
    #include "../../shaders/ggml_d3d12_iq3s_grid.inc"
};

// shaders/ggml_d3d12_mmv_iq3_s.hlsl signed4: negate the bytes selected by s4.
std::uint32_t iq3s_signed4(std::uint32_t g, std::uint32_t s4) {
    const std::uint32_t m = ((s4 * 0x00204081u) & 0x01010101u) * 0xffu;
    return (g ^ m) + (m & 0x01010101u);
}

// One thread's contribution for one row and one 256-element chunk. `xq` is
// one activation column in the vec_dot type's ggml blocks (block_q8_0 for
// Q4_0, block_q8_K for Q4_K / Q6_K / IQ3_S). Integer sums are exact; the float
// terms follow ggml's vec_dot (#312).
float thread_chunk(ggml_type t, const std::uint8_t* w, std::uint32_t row_off, std::uint32_t blk,
                   std::uint32_t itid, const std::uint8_t* xq) {
    if (t == GGML_TYPE_F32) { // shaders/ggml_d3d12_mmv_f32.hlsl: f32 activations as they are
        float acc = 0.f;
        for (std::uint32_t j = 0; j < 4; ++j) {
            const std::uint32_t e = (blk * 256u + 64u * j + 4u * itid) * 4u;
            float wv[4], xv[4];
            std::memcpy(wv, w + row_off + e, sizeof(wv));
            std::memcpy(xv, xq + e, sizeof(xv));
            float s = wv[0] * xv[0];
            s += wv[1] * xv[1];
            s += wv[2] * xv[2];
            s += wv[3] * xv[3];
            acc += s;
        }
        return acc;
    }
    if (t == GGML_TYPE_Q4_0) {
        const std::uint32_t b = itid >> 1, h = itid & 1u;
        const std::uint8_t* xb = xq + (blk * 8u + b) * 34u; // block_q8_0
        const float dx = h2f(ld16(xb, 0));
        const auto* qx = reinterpret_cast<const std::int8_t*>(xb + 2);
        const std::uint32_t bb = row_off + (blk * 8u + b) * 18u;
        const float d = h2f(ld16(w, bb));
        const std::uint32_t q0 = ld32(w, bb + 2u + 8u * h), q1 = ld32(w, bb + 6u + 8u * h);
        std::int32_t sumi = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            const std::uint32_t lo = 8u * h + i, hi = 16u + 8u * h + i;
            sumi += (nib(q0, i, 0) - 8) * qx[lo] + (nib(q1, i, 0) - 8) * qx[lo + 4] +
                    (nib(q0, i, 4) - 8) * qx[hi] + (nib(q1, i, 4) - 8) * qx[hi + 4];
        }
        return (d * dx) * static_cast<float>(sumi);
    }
    float dy;
    std::memcpy(&dy, xq + blk * 292u, 4); // block_q8_K: float d; int8 qs[256]; ...
    const auto* qx = reinterpret_cast<const std::int8_t*>(xq + blk * 292u + 4u);
    if (t == GGML_TYPE_IQ3_S) {
        const std::uint32_t ib32 = itid >> 1, h = itid & 1u, e = 32u * ib32 + 16u * h;
        const std::uint32_t bb = row_off + blk * 110u;
        const float d = h2f(ld16(w, bb));
        const std::uint32_t qs = ld32(w, bb + 2u + 8u * ib32 + 4u * h);
        const std::uint32_t hb = (w[bb + 66u + ib32] >> (4u * h)) & 0xFu;
        const std::uint32_t sg = ld16(w, bb + 74u + 4u * ib32 + 2u * h);
        const std::int32_t ls = 2 * static_cast<std::int32_t>(
                                        (w[bb + 106u + (ib32 >> 1)] >> (4u * (ib32 & 1u))) & 0xFu) +
                                1;
        std::int32_t sumi = 0;
        for (std::uint32_t k = 0; k < 4; ++k) {
            const std::uint32_t g =
                iq3s_signed4(kIq3sGrid[((qs >> (8u * k)) & 0xFFu) | (((hb >> k) & 1u) << 8)],
                             (sg >> (4u * k)) & 0xFu);
            for (std::uint32_t j = 0; j < 4; ++j)
                sumi += sbyte(g, j) * qx[e + 4u * k + j];
        }
        return (d * dy) * static_cast<float>(ls * sumi);
    }
    if (t == GGML_TYPE_Q4_K) {
        const std::uint32_t il = itid >> 2, ir = itid & 3u;
        const std::uint32_t e_lo = il * 64u + ir * 8u, e_hi = e_lo + 32u;
        const std::uint32_t bb = row_off + blk * 144u;
        const float d = h2f(ld16(w, bb)), dmin = h2f(ld16(w, bb + 2u));
        std::int32_t sc1, m1, sc2, m2;
        q4k_scale_min(2u * il, w + bb + 4u, &sc1, &m1);
        q4k_scale_min(2u * il + 1u, w + bb + 4u, &sc2, &m2);
        const std::uint32_t qa = ld32(w, bb + 16u + il * 32u + ir * 8u);
        const std::uint32_t qb = ld32(w, bb + 20u + il * 32u + ir * 8u);
        std::int32_t lo = 0, hi = 0, sl = 0, sh = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            lo += nib(qa, i, 0) * qx[e_lo + i] + nib(qb, i, 0) * qx[e_lo + 4 + i];
            hi += nib(qa, i, 4) * qx[e_hi + i] + nib(qb, i, 4) * qx[e_hi + 4 + i];
            sl += qx[e_lo + i] + qx[e_lo + 4 + i];
            sh += qx[e_hi + i] + qx[e_hi + 4 + i];
        }
        return (d * dy) * static_cast<float>(sc1 * lo + sc2 * hi) -
               (dmin * dy) * static_cast<float>(m1 * sl + m2 * sh);
    }
    if (t == GGML_TYPE_Q5_K) {
        const std::uint32_t il = itid >> 2, ir = itid & 3u;
        const std::uint32_t e_lo = il * 64u + ir * 8u, e_hi = e_lo + 32u;
        const std::uint32_t bb = row_off + blk * 176u;
        const float d = h2f(ld16(w, bb)), dmin = h2f(ld16(w, bb + 2u));
        std::int32_t sc1, m1, sc2, m2;
        q4k_scale_min(2u * il, w + bb + 4u, &sc1, &m1);
        q4k_scale_min(2u * il + 1u, w + bb + 4u, &sc2, &m2);
        const std::uint32_t qa = ld32(w, bb + 48u + il * 32u + ir * 8u);
        const std::uint32_t qb = ld32(w, bb + 52u + il * 32u + ir * 8u);
        const std::uint32_t ha = ld32(w, bb + 16u + ir * 8u), hb = ld32(w, bb + 20u + ir * 8u);
        auto q5 = [](std::uint32_t q, std::uint32_t qh, std::uint32_t i, std::uint32_t sh,
                     std::uint32_t hs) {
            return static_cast<std::int32_t>(((q >> (8u * i + sh)) & 0xFu) |
                                             (((qh >> (8u * i + hs)) & 1u) << 4));
        };
        std::int32_t lo = 0, hi = 0, sl = 0, sh = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            lo += q5(qa, ha, i, 0, 2u * il) * qx[e_lo + i] +
                  q5(qb, hb, i, 0, 2u * il) * qx[e_lo + 4 + i];
            hi += q5(qa, ha, i, 4, 2u * il + 1u) * qx[e_hi + i] +
                  q5(qb, hb, i, 4, 2u * il + 1u) * qx[e_hi + 4 + i];
            sl += qx[e_lo + i] + qx[e_lo + 4 + i];
            sh += qx[e_hi + i] + qx[e_hi + 4 + i];
        }
        return (d * dy) * static_cast<float>(sc1 * lo + sc2 * hi) -
               (dmin * dy) * static_cast<float>(m1 * sl + m2 * sh);
    }
    if (t == GGML_TYPE_IQ4_XS) {
        static constexpr std::int32_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                                1,    13,   25,  38,  53,  69,  89,  113};
        const std::uint32_t ib = itid >> 1, h = itid & 1u, e_lo = 32u * ib + 8u * h;
        const std::uint32_t bb = row_off + blk * 136u;
        const std::uint32_t hdr0 = ld32(w, bb), hdr1 = ld32(w, bb + 4u);
        const float d = h2f(hdr0 & 0xffffu);
        const std::int32_t ls = static_cast<std::int32_t>(((hdr1 >> (4u * ib)) & 0xFu) |
                                                          (((hdr0 >> (16u + 2u * ib)) & 3u) << 4)) -
                                32;
        const std::uint32_t qa = ld32(w, bb + 8u + 16u * ib + 8u * h);
        const std::uint32_t qb = ld32(w, bb + 12u + 16u * ib + 8u * h);
        std::int32_t sumi = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            sumi += kv[nib(qa, i, 0)] * qx[e_lo + i] + kv[nib(qb, i, 0)] * qx[e_lo + 4 + i] +
                    kv[nib(qa, i, 4)] * qx[e_lo + 16 + i] + kv[nib(qb, i, 4)] * qx[e_lo + 20 + i];
        }
        return (d * dy) * static_cast<float>(ls * sumi);
    }
    // Q6_K
    const std::uint32_t v = itid >> 3, l0 = 4u * (itid & 7u), is = l0 >> 4;
    const std::uint32_t e = 128u * v + l0;
    const std::uint32_t bb = row_off + blk * 210u;
    const std::uint32_t qla = ld32(w, bb + 64u * v + l0);
    const std::uint32_t qlb = ld32(w, bb + 64u * v + 32u + l0);
    const std::uint32_t qh = ld32(w, bb + 128u + 32u * v + l0);
    const std::uint32_t s0 = ld32(w, bb + 192u + 8u * v), s1 = ld32(w, bb + 196u + 8u * v);
    const float d = h2f(ld16(w, bb + 208u));
    auto q6 = [](std::uint32_t ql, std::uint32_t qhw, std::uint32_t i, std::uint32_t qs,
                 std::uint32_t hs) {
        const std::uint32_t lo4 = (ql >> (8u * i + qs)) & 0xFu;
        const std::uint32_t hi2 = (qhw >> (8u * i + hs)) & 3u;
        return static_cast<std::int32_t>(lo4 | (hi2 << 4)) - 32;
    };
    std::int32_t t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    for (std::uint32_t i = 0; i < 4; ++i) {
        t1 += q6(qla, qh, i, 0, 0) * qx[e + i];
        t2 += q6(qlb, qh, i, 0, 2) * qx[e + 32 + i];
        t3 += q6(qla, qh, i, 4, 4) * qx[e + 64 + i];
        t4 += q6(qlb, qh, i, 4, 6) * qx[e + 96 + i];
    }
    const std::int32_t tsum =
        sbyte(s0, is) * t1 + sbyte(s0, is + 2) * t2 + sbyte(s1, is) * t3 + sbyte(s1, is + 2) * t4;
    return (d * dy) * static_cast<float>(tsum);
}

} // namespace

ggml_type d3d12_activation_type(ggml_type weight) {
    if (weight == GGML_TYPE_F32)
        return GGML_TYPE_F32; // the kernel reads the f32 activations as they are
    return weight == GGML_TYPE_Q4_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q8_K;
}

void d3d12_quantize_activations(ggml_type weight, const float* x, std::size_t x_stride, int k,
                                int ncols, std::uint8_t* out) {
    const ggml_type at = d3d12_activation_type(weight);
    const auto from_float = ggml_get_type_traits_cpu(at)->from_float;
    const std::size_t row = ggml_row_size(at, k);
    for (int c = 0; c < ncols; ++c)
        from_float(x + static_cast<std::size_t>(c) * x_stride, out + c * row, k);
}

void d3d12_mmv_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes, const float* x,
                       std::size_t x_stride, float* y, std::size_t y_stride, int n, int k,
                       int ncols) {
    if (!w || !x || !y || n <= 0 || k <= 0 || k % kD3d12Chunk != 0 || ncols <= 0 ||
        !d3d12_weight_type_supported(t))
        return;
    const std::size_t xq_row = ggml_row_size(d3d12_activation_type(t), k);
    std::vector<std::uint8_t> xq(xq_row * static_cast<std::size_t>(ncols) + 4);
    d3d12_quantize_activations(t, x, x_stride, k, ncols, xq.data());
    const std::uint32_t nchunk = static_cast<std::uint32_t>(k / kD3d12Chunk);
    const std::uint32_t threads = static_cast<std::uint32_t>(d3d12_mm_threads(k));
    const std::uint32_t in_flight = threads / 16u;
    float acc[kD3d12MmvThreadsLong];
    for (int c = 0; c < ncols; ++c) {
        const std::uint8_t* xc = xq.data() + static_cast<std::size_t>(c) * xq_row;
        for (int row = 0; row < n; ++row) {
            const std::uint32_t row_off = static_cast<std::uint32_t>(row * w_row_bytes);
            for (std::uint32_t tid = 0; tid < threads; ++tid) {
                float a = 0.f;
                for (std::uint32_t blk = tid >> 4; blk < nchunk; blk += in_flight)
                    a += thread_chunk(t, w, row_off, blk, tid & 15u, xc);
                acc[tid] = a;
            }
            for (int stride = static_cast<int>(threads) / 2; stride > 0; stride >>= 1)
                for (int i = 0; i < stride; ++i)
                    acc[i] += acc[i + stride];
            y[static_cast<std::size_t>(c) * y_stride + row] = acc[0];
        }
    }
}

void d3d12_get_rows_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes,
                            const std::int32_t* ids, int n_ids, int k, float* y,
                            std::size_t y_stride) {
    if (!w || !ids || !y || n_ids <= 0 || k <= 0 || k % kD3d12Chunk != 0 ||
        !d3d12_get_rows_type_supported(t))
        return;
    auto ld8 = [&](std::uint32_t a) { return (ld32(w, a & ~3u) >> ((a & 3u) * 8u)) & 0xffu; };
    // One thread group per (256-element chunk, id); thread tid = 32v + l writes
    // elements 128v + l + {0, 32, 64, 96}, as dequantize_row_q6_K's inner loop.
    for (int i = 0; i < n_ids; ++i) {
        for (std::uint32_t blk = 0; blk < static_cast<std::uint32_t>(k / kD3d12Chunk); ++blk) {
            const std::uint32_t bb = static_cast<std::uint32_t>(ids[i] * w_row_bytes) + blk * 210u;
            const float d = h2f(ld16(w, bb + 208u));
            for (std::uint32_t tid = 0; tid < 64u; ++tid) {
                const std::uint32_t v = tid >> 5, l = tid & 31u, is = l >> 4;
                const std::uint32_t qla = ld8(bb + 64u * v + l), qlb = ld8(bb + 64u * v + l + 32u);
                const std::uint32_t qh = ld8(bb + 128u + 32u * v + l);
                auto sc = [&](std::uint32_t j) {
                    return static_cast<float>(
                        static_cast<std::int8_t>(ld8(bb + 192u + 8u * v + is + j)));
                };
                const int q1 = static_cast<int>((qla & 0xFu) | (((qh >> 0) & 3u) << 4)) - 32;
                const int q2 = static_cast<int>((qlb & 0xFu) | (((qh >> 2) & 3u) << 4)) - 32;
                const int q3 = static_cast<int>((qla >> 4) | (((qh >> 4) & 3u) << 4)) - 32;
                const int q4 = static_cast<int>((qlb >> 4) | (((qh >> 6) & 3u) << 4)) - 32;
                float* out = y + static_cast<std::size_t>(i) * y_stride + blk * 256u + 128u * v + l;
                out[0] = d * sc(0) * static_cast<float>(q1);
                out[32] = d * sc(2) * static_cast<float>(q2);
                out[64] = d * sc(4) * static_cast<float>(q3);
                out[96] = d * sc(6) * static_cast<float>(q4);
            }
        }
    }
}

// --- f32 ops between the matmuls (shaders/ggml_d3d12_elt.hlsl) ---

bool d3d12_elt_rows_supported(std::int64_t ne0, std::int64_t ne1, std::int64_t ne2,
                              std::int64_t ne3) {
    // The kernels address rows with 32-bit byte offsets from the tensor start.
    return ne0 >= kD3d12EltMinRow && ne1 >= 1 && ne1 <= kD3d12MaxGroups && ne2 == 1 && ne3 == 1 &&
           (ne0 + 255) / 256 <= kD3d12MaxGroups && ne0 * ne1 <= (std::int64_t(1) << 28);
}

bool d3d12_norm_weight_name(const char* name) {
    if (!name)
        return false;
    if (std::strcmp(name, "output_norm.weight") == 0)
        return true;
    if (std::strncmp(name, "blk.", 4) != 0)
        return false;
    const char* p = name + 4;
    const char* digits = p;
    while (*p >= '0' && *p <= '9')
        ++p;
    if (p == digits || *p != '.')
        return false;
    ++p;
    return std::strcmp(p, "attn_norm.weight") == 0 ||
           std::strcmp(p, "post_attention_norm.weight") == 0 ||
           std::strcmp(p, "ffn_norm.weight") == 0;
}

float d3d12_div_rn(std::uint32_t n, float b) {
    std::uint32_t bb;
    std::memcpy(&bb, &b, 4);
    bb &= 0x7fffffffu;
    const std::uint32_t eb = bb >> 23;
    auto bits = [](std::uint32_t u) {
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    };
    if (eb == 0u || n == 0u)
        return eb == 0u ? bits(0x7f800000u) : 0.0f;
    const std::uint32_t mb = (bb & 0x7fffffu) | 0x800000u;
    std::uint32_t s = 0;
    while ((n << s) < 0x800000u)
        ++s;
    std::uint32_t r = n << s, q = 0;
    for (int i = 0; i < 26; ++i) {
        q <<= 1;
        if (r >= mb) {
            r -= mb;
            q |= 1u;
        }
        r <<= 1;
    }
    int e = 150 - static_cast<int>(s) - static_cast<int>(eb);
    std::uint32_t m, guard;
    bool sticky = r != 0u;
    if (q >= (1u << 25)) {
        m = q >> 2;
        guard = (q >> 1) & 1u;
        sticky = sticky || (q & 1u) != 0u;
    } else {
        m = q >> 1;
        guard = q & 1u;
        e -= 1;
    }
    if (guard != 0u && (sticky || (m & 1u) != 0u)) {
        m += 1u;
        if (m == (1u << 24)) {
            m >>= 1;
            e += 1;
        }
    }
    if (e > 127)
        return bits(0x7f800000u);
    if (e < -126)
        return 0.0f;
    return bits((static_cast<std::uint32_t>(e + 127) << 23) | (m & 0x7fffffu));
}

void d3d12_quant_q8k_emulate(const float* x, int k, std::uint8_t* out) {
    if (!x || !out || k <= 0 || k % kD3d12Chunk != 0)
        return;
    for (int blk = 0; blk < k / kD3d12Chunk; ++blk) {
        const float* xb = x + blk * kD3d12Chunk;
        std::uint8_t* ob = out + static_cast<std::size_t>(blk) * 292u;
        float s_ax[64], s_v[64];
        std::uint32_t s_i[64];
        for (std::uint32_t tid = 0; tid < 64u; ++tid) {
            float ax = 0.0f, v = 0.0f;
            std::uint32_t idx = 4u * tid;
            for (std::uint32_t j = 0; j < 4u; ++j) {
                const float a = std::fabs(xb[4u * tid + j]);
                if (a > ax) {
                    ax = a;
                    v = xb[4u * tid + j];
                    idx = 4u * tid + j;
                }
            }
            s_ax[tid] = ax;
            s_v[tid] = v;
            s_i[tid] = idx;
        }
        for (std::uint32_t stride = 32u; stride > 0u; stride >>= 1u)
            for (std::uint32_t tid = 0; tid < stride; ++tid) {
                const float o = s_ax[tid + stride];
                if (o > s_ax[tid] || (o == s_ax[tid] && s_i[tid + stride] < s_i[tid])) {
                    s_ax[tid] = o;
                    s_v[tid] = s_v[tid + stride];
                    s_i[tid] = s_i[tid + stride];
                }
            }
        float d = 0.0f;
        std::int8_t qs[256] = {};
        if (s_ax[0] != 0.0f) {
            const float v = s_v[0];
            const float ais = d3d12_div_rn(127u, v);
            const float iscale = v > 0.0f ? -ais : ais;
            const float dm = d3d12_div_rn(1u, ais);
            d = v > 0.0f ? -dm : dm;
            for (int j = 0; j < 256; ++j) {
                const float p = iscale * xb[j];
                qs[j] =
                    static_cast<std::int8_t>(std::min(127, static_cast<int>(std::nearbyint(p))));
            }
        }
        std::memcpy(ob, &d, 4);
        std::memcpy(ob + 4, qs, 256);
        for (int j = 0; j < 16; ++j) {
            int sum = 0;
            for (int i = 0; i < 16; ++i)
                sum += qs[16 * j + i];
            const std::int16_t s16 = static_cast<std::int16_t>(sum);
            std::memcpy(ob + 260 + 2 * j, &s16, 2);
        }
    }
}

// --- Selftest CSV ---

const char* d3d12_selftest_csv_header() {
    return "type,n,k,ncols,rel_err,gpu_ms,packed_gbs,ok,d3d12_ran,host,date,error\n";
}

std::string format_d3d12_selftest_row(const D3d12SelftestRow& r, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    std::string err = r.error.empty() ? "-" : r.error;
    for (char& ch : err)
        if (ch == ',' || ch == '\n' || ch == '\r')
            ch = ' ';
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s,%d,%d,%d,%.3g,%.4f,%.2f,%d,%d,%s,%s,%s\n", r.type.c_str(),
                  r.n, r.k, r.ncols, r.rel_err, r.gpu_ms, r.packed_gbs, r.ok ? 1 : 0,
                  r.d3d12_ran ? 1 : 0, host_label ? host_label : "unknown", date_buf, err.c_str());
    return buf;
}

} // namespace xllama

    #if !defined(_WIN32)

namespace xllama {

bool ggml_d3d12_register() {
    return false;
}

ggml_backend_buffer_type_t ggml_d3d12_weights_buft() {
    return nullptr;
}

void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    D3d12SelftestRow r;
    r.type = "-";
    r.error = "d3d12 unavailable on this platform";
    out->push_back(std::move(r));
}

void ggml_d3d12_set_weights_max_buffer(std::size_t) {}

void run_d3d12_memprobe(const std::function<void(const std::string&)>& emit) {
    if (emit)
        emit("d3d12 unavailable on this platform");
}

} // namespace xllama

    #else // _WIN32 — the backend itself

        #include <chrono>
        #include <cstdint>
        #include <cstdio>
        #include <cstdlib>
        #include <iterator>
        #include <map>
        #include <memory>
        #include <mutex>
        #include <random>
        #include <tuple>
        #include <unordered_map>
        #include <unordered_set>
        #include <vector>

        #include <dxgi1_4.h>

        #include "d3d12_compute.h"
        #include "ggml-alloc.h"
        #include "ggml-backend-impl.h"
        #include "ggml-backend.h"
        #include "ggml-impl.h"
        #include "ggml_d3d12_elt_binary_dxil.h"
        #include "ggml_d3d12_elt_quant_q8_k_dxil.h"
        #include "ggml_d3d12_elt_rms_norm_dxil.h"
        #include "ggml_d3d12_get_rows_q6_k_dxil.h"
        #include "ggml_d3d12_mmid_emu.h"
        #include "ggml_d3d12_mmid_iq3_s_t128_dxil.h"
        #include "ggml_d3d12_mmid_iq3_s_t64_dxil.h"
        #include "ggml_d3d12_mmid_iq4_xs_t128_dxil.h"
        #include "ggml_d3d12_mmid_iq4_xs_t64_dxil.h"
        #include "ggml_d3d12_mmv_f32_t128_dxil.h"
        #include "ggml_d3d12_mmv_f32_t64_dxil.h"
        #include "ggml_d3d12_mmv_iq3_s_t128_dxil.h"
        #include "ggml_d3d12_mmv_iq3_s_t64_dxil.h"
        #include "ggml_d3d12_mmv_iq4_xs_t128_dxil.h"
        #include "ggml_d3d12_mmv_iq4_xs_t64_dxil.h"
        #include "ggml_d3d12_mmv_q4_0_t128_dxil.h"
        #include "ggml_d3d12_mmv_q4_0_t64_dxil.h"
        #include "ggml_d3d12_mmv_q4_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q4_k_t64_dxil.h"
        #include "ggml_d3d12_mmv_q5_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q5_k_t64_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t64_dxil.h"
        #include "ggml_d3d12_moe_combine_dxil.h"
        #include "ggml_d3d12_moe_emu.h"
        #include "ggml_d3d12_moe_tail_dxil.h"
        #include "ggml_d3d12_ssm_conv_dxil.h"
        #include "ggml_d3d12_ssm_copy_dxil.h"
        #include "ggml_d3d12_ssm_emu.h"
        #include "ggml_d3d12_ssm_gdn_dxil.h"
        #include "ggml_d3d12_ssm_get_rows_dxil.h"
        #include "ggml_d3d12_ssm_unary_dxil.h"
        #include "xllama/d3d12_dyn.h"
        #include "xllama/platform.h"

namespace xllama {
namespace {

using d3d12c::ComPtr;

enum Pso {
    kPsoQ40 = 0,
    kPsoQ4K = 1,
    kPsoQ6K = 2,
    kPsoIq3S = 3,
    kPsoIq4Xs = 4,
    kPsoQ5K = 5,
    kPsoF32 = 6,
    kPsoCount = 7
};
enum Elt { kEltBinary = 0, kEltRmsNorm = 1, kEltQuantQ8K = 2, kEltCount = 3 };
enum Moe { kMoeCombine = 0, kMoeTail = 1, kMoeCount = 2 };
// shaders/ggml_d3d12_ssm.hlsl, KERNEL 1..5 (ggml_d3d12_ssm_emu.h).
enum Ssm { kSsmUnary = 0, kSsmCopy = 1, kSsmGetRows = 2, kSsmConv = 3, kSsmGdn = 4, kSsmCount = 5 };
// Root constants: the 8 every kernel reads, then MUL_MAT_ID's 5
// (shaders/ggml_d3d12_mmid_{iq4_xs,iq3_s}.hlsl).
constexpr UINT kOpConstants = 13;
// MUL_MAT_ID kernels by expert type (d3d12_mmid_type_supported).
enum Mmid { kMmidIq4Xs = 0, kMmidIq3S = 1, kMmidCount = 2 };

int mmid_for(ggml_type t) {
    return t == GGML_TYPE_IQ4_XS ? kMmidIq4Xs : t == GGML_TYPE_IQ3_S ? kMmidIq3S : -1;
}

int pso_for(ggml_type t) {
    switch (t) {
    case GGML_TYPE_Q4_0:
        return kPsoQ40;
    case GGML_TYPE_Q4_K:
        return kPsoQ4K;
    case GGML_TYPE_Q6_K:
        return kPsoQ6K;
    case GGML_TYPE_IQ3_S:
        return kPsoIq3S;
    case GGML_TYPE_IQ4_XS:
        return kPsoIq4Xs;
    case GGML_TYPE_Q5_K:
        return kPsoQ5K;
    case GGML_TYPE_F32:
        return kPsoF32;
    default:
        return -1;
    }
}

// Process-wide D3D12 state behind the ggml device. One queue, serialised.
struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso[kPsoCount][2]; // [type][0 = 64 threads, 1 = 128]
    ComPtr<ID3D12PipelineState> pso_get_rows;      // Q6_K
    // MUL_MAT_ID, IQ4_XS or IQ3_S experts in an EXPERT_CACHE buffer this device
    // claims: [Mmid][0 = 64 threads, 1 = 128].
    ComPtr<ID3D12PipelineState> pso_mmid[kMmidCount][2];
    // shaders/ggml_d3d12_elt.hlsl; without them (elt_ok false) those ops and
    // the q8_K quantization run on the CPU inside graph_compute.
    ComPtr<ID3D12PipelineState> pso_elt[kEltCount];
    bool elt_ok = false;
    // shaders/ggml_d3d12_moe.hlsl: the routed experts' weighted sum and the
    // shared expert's gate + both residual adds, each one dispatch.
    ComPtr<ID3D12PipelineState> pso_moe[kMoeCount];
    bool moe_ok = false;
    // The linear-attention layers' ops (gated delta net, conv, the recurrent
    // state copies); without them those stay on the CPU.
    ComPtr<ID3D12PipelineState> pso_ssm[kSsmCount];
    bool ssm_ok = false;
    // KERNEL 5 reading and writing the recurrent cache rows itself, without
    // build_rs's GET_ROWS and the CPY back (d3d12_gdn_fuse_plan). Read when a
    // backend is created: XLLAMA_D3D12_SSM_FUSE=0 keeps both copies.
    bool gdn_fuse = true;
    // IQ3_S group geometry asked for by XLLAMA_D3D12_IQ3S_RSL / _RT (read when a
    // backend is created; d3d12_geom_for shapes each matmul from it).
    D3d12GeomReq iq3s_geom;
    // XLLAMA_D3D12_SHADER_DIR (read when a backend is created): <dir>\<name>.dxil
    // replaces the built-in blob of that name (shaders/generated/<name>_dxil.h)
    // for that backend; without the variable every slot is built-in again.
    // Every pipeline ever bound stays alive in pso_keep, since a kept command
    // list is matched by pipeline pointer and must never meet a reused address.
    std::vector<ComPtr<ID3D12PipelineState>> pso_builtin; // pso_slots() order
    std::vector<ComPtr<ID3D12PipelineState>> pso_keep;
    std::unordered_map<const ID3D12PipelineState*, std::string> pso_over; // -> blob path
    ComPtr<ID3D12QueryHeap> ts;
    ComPtr<ID3D12Resource> ts_rb;
    // XLLAMA_D3D12_PROFILE (read when a backend is created): a timestamp after
    // every dispatch, summed per kernel and grid in `kernels`.
    ComPtr<ID3D12QueryHeap> prof;
    ComPtr<ID3D12Resource> prof_rb;
    bool prof_on = false;
    ComPtr<ID3D12Resource> staging;  // weight upload ring (kStagingBytes)
    ComPtr<ID3D12Resource> readback; // get_tensor on weights (kStagingBytes)
    std::uint8_t* staging_ptr = nullptr;
    std::uint8_t* readback_ptr = nullptr;
    // q8 activations for the next submission (CPU-written, GPU-read; grows on
    // demand to the largest single input, ~6 MiB for a 512-token ubatch).
    ComPtr<ID3D12Resource> xq;
    std::uint8_t* xq_ptr = nullptr;
    std::size_t xq_bytes = 0;
    d3d12c::QueueFence fence;
    UINT64 ts_freq = 0;
    LUID luid = {};
    double last_gpu_ms = 0.0;
    // Per-backend-lifetime counters, logged when the backend is freed.
    std::uint64_t n_calls = 0;
    std::uint64_t n_dispatches = 0;
    std::uint64_t n_submits = 0;
    std::uint64_t n_replays = 0; // submits that executed a kept command list again
    std::uint64_t n_barriers = 0;
    std::uint64_t n_gdn_fused = 0; // delta-net layers run without their two state copies
    double wall_ms = 0.0;
    double gpu_ms = 0.0;
    // Of wall_ms: recording + submitting the graph submissions, then waiting for
    // them (GPU time + launch and completion latency). The rest is host work
    // between submissions: planning the ops, CPU fallbacks, q8 activations.
    double rec_ms = 0.0;
    double wait_ms = 0.0;
    // graph_compute calls by their first and last node, layer suffix dropped:
    // where the scheduler hands work to the CPU and back (seg_name).
    struct SegStat {
        std::uint64_t n = 0;
        double ms = 0.0;
    };
    std::map<std::string, SegStat> segs;
    // XLLAMA_D3D12_PROFILE: dispatches by pipeline and grid.
    std::map<std::tuple<const ID3D12PipelineState*, UINT, UINT, UINT>, SegStat> kernels;
    std::mutex mu;
    bool ok = false;
    std::string error;
};

// One replaceable pipeline: the base name of its built-in blob
// (shaders/generated/<name>_dxil.h) and the Gpu field that binds it.
struct PsoSlot {
    std::string name;
    ComPtr<ID3D12PipelineState>* pso;
};

// std::addressof, never &: WRL's ComPtr::operator& is a ComPtrRef whose
// conversion to ComPtr<T>* empties the ComPtr first (an out-parameter idiom),
// so `&g.pso[i][wd]` here released every pipeline at the end of init_gpu and
// 1.7.12.54 ran every op as a bare barrier (0 dispatches, 1/48 selftest cases).
std::vector<PsoSlot> pso_slots(Gpu& g) {
    static const char* const mmv[] = {"q4_0", "q4_k", "q6_k", "iq3_s", "iq4_xs", "q5_k", "f32"};
    static const char* const mmid[] = {"iq4_xs", "iq3_s"};
    static const char* const elt[] = {"binary", "rms_norm", "quant_q8_k"};
    static const char* const moe[] = {"combine", "tail"};
    static const char* const ssm[] = {"unary", "copy", "get_rows", "conv", "gdn"};
    static_assert(
        std::size(mmv) == std::size_t{kPsoCount} && std::size(mmid) == std::size_t{kMmidCount} &&
        std::size(elt) == std::size_t{kEltCount} && std::size(moe) == std::size_t{kMoeCount} &&
        std::size(ssm) == std::size_t{kSsmCount});
    static const char* const wide[2] = {"_t64", "_t128"};
    std::vector<PsoSlot> s;
    for (int i = 0; i < kPsoCount; ++i)
        for (int wd = 0; wd < 2; ++wd)
            s.push_back(
                {std::string("ggml_d3d12_mmv_") + mmv[i] + wide[wd], std::addressof(g.pso[i][wd])});
    s.push_back({"ggml_d3d12_get_rows_q6_k", std::addressof(g.pso_get_rows)});
    for (int i = 0; i < kMmidCount; ++i)
        for (int wd = 0; wd < 2; ++wd)
            s.push_back({std::string("ggml_d3d12_mmid_") + mmid[i] + wide[wd],
                         std::addressof(g.pso_mmid[i][wd])});
    for (int i = 0; i < kEltCount; ++i)
        s.push_back({std::string("ggml_d3d12_elt_") + elt[i], std::addressof(g.pso_elt[i])});
    for (int i = 0; i < kMoeCount; ++i)
        s.push_back({std::string("ggml_d3d12_moe_") + moe[i], std::addressof(g.pso_moe[i])});
    for (int i = 0; i < kSsmCount; ++i)
        s.push_back({std::string("ggml_d3d12_ssm_") + ssm[i], std::addressof(g.pso_ssm[i])});
    return s;
}

bool read_blob(const std::string& path, std::vector<char>* out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return false;
    out->clear();
    char buf[16384];
    std::size_t got = 0;
    while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0)
        out->insert(out->end(), buf, buf + got);
    const bool ok = std::ferror(f) == 0 && !out->empty();
    std::fclose(f);
    return ok;
}

// XLLAMA_D3D12_SHADER_DIR: bind <dir>\<name>.dxil where the file is there and
// builds, the built-in pipeline everywhere else (dir null or empty: all of
// them). Caller holds g.mu. Returns the number of slots replaced.
int apply_shader_dir(Gpu& g, const char* dir) {
    const std::vector<PsoSlot> slots = pso_slots(g);
    int replaced = 0;
    for (std::size_t i = 0; i < slots.size() && i < g.pso_builtin.size(); ++i) {
        *slots[i].pso = g.pso_builtin[i];
        if (!dir || !*dir || !g.pso_builtin[i])
            continue;
        const std::string path = std::string(dir) + "\\" + slots[i].name + ".dxil";
        std::vector<char> blob;
        if (!read_blob(path, &blob))
            continue;
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.root.Get();
        pd.CS.pShaderBytecode = blob.data();
        pd.CS.BytecodeLength = blob.size();
        ComPtr<ID3D12PipelineState> pso;
        const HRESULT hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso));
        if (FAILED(hr)) {
            log_output("[xllama] d3d12: " + slots[i].name +
                       ".dxil: " + d3d12c::hr_message("CreateComputePipelineState", hr) +
                       " — built-in kernel kept\n");
            continue;
        }
        g.pso_over[pso.Get()] = path;
        g.pso_keep.push_back(pso);
        *slots[i].pso = pso;
        ++replaced;
        log_output("[xllama] d3d12: " + slots[i].name + " <- " + path + " (" +
                   std::to_string(blob.size()) + " B)\n");
    }
    return replaced;
}

// Weight upload / readback ring. Mapped for the process lifetime and counted in
// its working set, so it stays small: 64 MiB each put the first GPU-layer
// smoke at 640 MB peak vs 311 MB on the CPU (D2b); 8 MiB costs a few more
// round trips at load only.
constexpr UINT64 kStagingBytes = 8ull << 20;

ComPtr<ID3D12RootSignature> make_root_sig(ID3D12Device* device, std::string* err) {
    // b0 constants, t0 weights, u0 output, u1 activations, u2 / u3 (elt: b or
    // norm weight, fused-MUL output), t1 (MUL_MAT_ID: expert ids), u4 (MoE
    // tail: the residual), u5 (fused delta net: the cache row of the new state).
    D3D12_ROOT_PARAMETER p[9] = {};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p[0].Constants.ShaderRegister = 0;
    p[0].Constants.Num32BitValues = kOpConstants;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    p[1].Descriptor.ShaderRegister = 0;
    for (UINT i = 0; i < 4; ++i) {
        p[2 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        p[2 + i].Descriptor.ShaderRegister = i;
    }
    p[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    p[6].Descriptor.ShaderRegister = 1;
    p[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    p[7].Descriptor.ShaderRegister = 4;
    p[8].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    p[8].Descriptor.ShaderRegister = 5;
    for (auto& q : p)
        q.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 9;
    desc.pParameters = p;
    auto serialize = d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        *err = "D3D12SerializeRootSignature not available";
        return {};
    }
    ComPtr<ID3DBlob> sig, blob_err;
    HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &blob_err);
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("D3D12SerializeRootSignature", hr);
        return {};
    }
    ComPtr<ID3D12RootSignature> root;
    hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                     IID_PPV_ARGS(&root));
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("CreateRootSignature", hr);
        return {};
    }
    return root;
}

bool init_gpu(Gpu& g) {
    std::string& err = g.error;
    g.device = d3d12c::create_device(&err);
    if (!g.device)
        return false;
    D3D12_FEATURE_DATA_ARCHITECTURE1 arch = {};
    if (FAILED(g.device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &arch, sizeof(arch))) ||
        !arch.CacheCoherentUMA) {
        // D3D12_Host relies on a CPU-cached GPU-visible heap (D1: cc_uma=1 on Series S).
        err = "adapter is not cache-coherent UMA";
        return false;
    }
    g.luid = g.device->GetAdapterLuid();
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = g.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g.queue));
    if (FAILED(hr)) {
        err = d3d12c::hr_message("CreateCommandQueue", hr);
        return false;
    }
    hr = g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g.alloc));
    if (SUCCEEDED(hr))
        hr = g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.alloc.Get(), nullptr,
                                         IID_PPV_ARGS(&g.cl));
    if (FAILED(hr)) {
        err = d3d12c::hr_message("CreateCommandList", hr);
        return false;
    }
    g.cl->Close();
    if (!g.fence.init(g.device.Get(), &err))
        return false;
    g.root = make_root_sig(g.device.Get(), &err);
    if (!g.root)
        return false;
    const void* blobs[kPsoCount][2] = {{kGgmlD3d12MmvQ40T64Dxil, kGgmlD3d12MmvQ40T128Dxil},
                                       {kGgmlD3d12MmvQ4KT64Dxil, kGgmlD3d12MmvQ4KT128Dxil},
                                       {kGgmlD3d12MmvQ6KT64Dxil, kGgmlD3d12MmvQ6KT128Dxil},
                                       {kGgmlD3d12MmvIq3ST64Dxil, kGgmlD3d12MmvIq3ST128Dxil},
                                       {kGgmlD3d12MmvIq4XsT64Dxil, kGgmlD3d12MmvIq4XsT128Dxil},
                                       {kGgmlD3d12MmvQ5KT64Dxil, kGgmlD3d12MmvQ5KT128Dxil},
                                       {kGgmlD3d12MmvF32T64Dxil, kGgmlD3d12MmvF32T128Dxil}};
    const size_t sizes[kPsoCount][2] = {
        {kGgmlD3d12MmvQ40T64DxilSize, kGgmlD3d12MmvQ40T128DxilSize},
        {kGgmlD3d12MmvQ4KT64DxilSize, kGgmlD3d12MmvQ4KT128DxilSize},
        {kGgmlD3d12MmvQ6KT64DxilSize, kGgmlD3d12MmvQ6KT128DxilSize},
        {kGgmlD3d12MmvIq3ST64DxilSize, kGgmlD3d12MmvIq3ST128DxilSize},
        {kGgmlD3d12MmvIq4XsT64DxilSize, kGgmlD3d12MmvIq4XsT128DxilSize},
        {kGgmlD3d12MmvQ5KT64DxilSize, kGgmlD3d12MmvQ5KT128DxilSize},
        {kGgmlD3d12MmvF32T64DxilSize, kGgmlD3d12MmvF32T128DxilSize}};
    for (int i = 0; i < kPsoCount; ++i) {
        for (int wd = 0; wd < 2; ++wd) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = g.root.Get();
            pd.CS.pShaderBytecode = blobs[i][wd];
            pd.CS.BytecodeLength = sizes[i][wd];
            hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso[i][wd]));
            if (FAILED(hr)) {
                err = d3d12c::hr_message("CreateComputePipelineState", hr);
                return false;
            }
        }
    }
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.root.Get();
        pd.CS.pShaderBytecode = kGgmlD3d12GetRowsQ6KDxil;
        pd.CS.BytecodeLength = kGgmlD3d12GetRowsQ6KDxilSize;
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_get_rows));
        if (FAILED(hr)) {
            err = d3d12c::hr_message("CreateComputePipelineState", hr);
            return false;
        }
    }
    {
        const void* mb[kMmidCount][2] = {{kGgmlD3d12MmidIq4XsT64Dxil, kGgmlD3d12MmidIq4XsT128Dxil},
                                         {kGgmlD3d12MmidIq3ST64Dxil, kGgmlD3d12MmidIq3ST128Dxil}};
        const size_t ms[kMmidCount][2] = {
            {kGgmlD3d12MmidIq4XsT64DxilSize, kGgmlD3d12MmidIq4XsT128DxilSize},
            {kGgmlD3d12MmidIq3ST64DxilSize, kGgmlD3d12MmidIq3ST128DxilSize}};
        for (int mk = 0; mk < kMmidCount; ++mk)
            for (int wd = 0; wd < 2; ++wd) {
                D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
                pd.pRootSignature = g.root.Get();
                pd.CS.pShaderBytecode = mb[mk][wd];
                pd.CS.BytecodeLength = ms[mk][wd];
                hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_mmid[mk][wd]));
                if (FAILED(hr)) {
                    err = d3d12c::hr_message("CreateComputePipelineState (mmid)", hr);
                    return false;
                }
            }
    }
    {
        const void* eb[kEltCount] = {kGgmlD3d12EltBinaryDxil, kGgmlD3d12EltRmsNormDxil,
                                     kGgmlD3d12EltQuantQ8KDxil};
        const size_t es[kEltCount] = {kGgmlD3d12EltBinaryDxilSize, kGgmlD3d12EltRmsNormDxilSize,
                                      kGgmlD3d12EltQuantQ8KDxilSize};
        g.elt_ok = true;
        for (int i = 0; i < kEltCount && g.elt_ok; ++i) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = g.root.Get();
            pd.CS.pShaderBytecode = eb[i];
            pd.CS.BytecodeLength = es[i];
            hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_elt[i]));
            if (FAILED(hr)) {
                // The matmuls still run; the ops between them go back to the CPU.
                log_output("[xllama] d3d12: " +
                           d3d12c::hr_message("CreateComputePipelineState (elt)", hr) +
                           " — elementwise ops stay on the CPU\n");
                g.elt_ok = false;
            }
        }
    }
    {
        const void* mb[kMoeCount] = {kGgmlD3d12MoeCombineDxil, kGgmlD3d12MoeTailDxil};
        const size_t ms[kMoeCount] = {kGgmlD3d12MoeCombineDxilSize, kGgmlD3d12MoeTailDxilSize};
        g.moe_ok = g.elt_ok; // their fallbacks are the elementwise kernels
        for (int i = 0; i < kMoeCount && g.moe_ok; ++i) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = g.root.Get();
            pd.CS.pShaderBytecode = mb[i];
            pd.CS.BytecodeLength = ms[i];
            hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_moe[i]));
            if (FAILED(hr)) {
                log_output("[xllama] d3d12: " +
                           d3d12c::hr_message("CreateComputePipelineState (moe)", hr) +
                           " — the MoE tail stays on the CPU\n");
                g.moe_ok = false;
            }
        }
    }
    {
        const void* sb[kSsmCount] = {kGgmlD3d12SsmUnaryDxil, kGgmlD3d12SsmCopyDxil,
                                     kGgmlD3d12SsmGetRowsDxil, kGgmlD3d12SsmConvDxil,
                                     kGgmlD3d12SsmGdnDxil};
        const size_t ss[kSsmCount] = {kGgmlD3d12SsmUnaryDxilSize, kGgmlD3d12SsmCopyDxilSize,
                                      kGgmlD3d12SsmGetRowsDxilSize, kGgmlD3d12SsmConvDxilSize,
                                      kGgmlD3d12SsmGdnDxilSize};
        g.ssm_ok = g.elt_ok; // the chain's ADD / MUL / RMS_NORM are elementwise kernels
        for (int i = 0; i < kSsmCount && g.ssm_ok; ++i) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = g.root.Get();
            pd.CS.pShaderBytecode = sb[i];
            pd.CS.BytecodeLength = ss[i];
            hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_ssm[i]));
            if (FAILED(hr)) {
                log_output("[xllama] d3d12: " +
                           d3d12c::hr_message("CreateComputePipelineState (ssm)", hr) +
                           " — the linear-attention ops stay on the CPU\n");
                g.ssm_ok = false;
            }
        }
    }
    if (SUCCEEDED(g.queue->GetTimestampFrequency(&g.ts_freq)) && g.ts_freq > 0) {
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = 2;
        g.device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&g.ts));
        g.ts_rb = d3d12c::create_buffer(g.device.Get(), 256, D3D12_HEAP_TYPE_READBACK,
                                        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                        "ts_rb", &err);
        if (!g.ts || !g.ts_rb) {
            g.ts.Reset();
            g.ts_rb.Reset();
            err.clear();
        }
    }
    g.staging = d3d12c::create_buffer(g.device.Get(), kStagingBytes, D3D12_HEAP_TYPE_UPLOAD,
                                      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                      "staging", &err);
    g.readback = d3d12c::create_buffer(g.device.Get(), kStagingBytes, D3D12_HEAP_TYPE_READBACK,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                       "readback", &err);
    if (!g.staging || !g.readback)
        return false;
    void* p = nullptr;
    if (FAILED(g.staging->Map(0, nullptr, &p)))
        return false;
    g.staging_ptr = static_cast<std::uint8_t*>(p);
    if (FAILED(g.readback->Map(0, nullptr, &p)))
        return false;
    g.readback_ptr = static_cast<std::uint8_t*>(p);
    for (const PsoSlot& s : pso_slots(g))
        g.pso_builtin.push_back(*s.pso);
    // The matmul pipelines are required above: one gone now means the slot
    // table broke them, and a backend whose ops record as bare barriers returns
    // garbage, where no backend leaves the model on the CPU.
    for (int i = 0; i < kPsoCount; ++i)
        for (int wd = 0; wd < 2; ++wd)
            if (!g.pso[i][wd]) {
                err = "a built-in pipeline was released after creation";
                return false;
            }
    return true;
}

Gpu& gpu() {
    static Gpu g;
    static std::once_flag once;
    std::call_once(once, [] { g.ok = init_gpu(g); });
    return g;
}

// Execute a closed list and wait for it. Caller holds g.mu. rec_ms / wait_ms,
// when given, gain the time from t0 (recording started) to the submit and the
// time then spent waiting for the fence.
bool execute_and_wait(Gpu& g, ID3D12GraphicsCommandList* cl,
                      std::chrono::steady_clock::time_point t0, double* rec_ms, double* wait_ms) {
    ID3D12CommandList* lists[] = {cl};
    g.queue->ExecuteCommandLists(1, lists);
    const auto t1 = std::chrono::steady_clock::now();
    const bool ok = g.fence.signal_and_wait(g.queue.Get(), /*spin=*/true);
    if (rec_ms)
        *rec_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (wait_ms)
        *wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1)
                        .count();
    return ok;
}

// Record into the shared list and run it to completion. Caller holds g.mu.
template <typename F>
bool run_now(Gpu& g, F&& record, double* rec_ms = nullptr, double* wait_ms = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    g.alloc->Reset();
    g.cl->Reset(g.alloc.Get(), nullptr);
    record(g.cl.Get());
    g.cl->Close();
    return execute_and_wait(g, g.cl.Get(), t0, rec_ms, wait_ms);
}

// --- Buffers ---

struct BufCtx {
    std::vector<ComPtr<ID3D12Heap>> heaps; // backing of a reserved weights buffer; outlives res
    ComPtr<ID3D12Resource> res;
    D3D12_GPU_VIRTUAL_ADDRESS va = 0;
    std::uint8_t* host = nullptr; // mapped pointer (D3D12_Host only)
    bool is_host = false;
    bool wc = false;      // D3D12_Host created write-combined (XLLAMA_D3D12_WC_MIB)
    bool settled = false; // memory type final: checked on the first base / clear
};

// Series X Dev Mode (2026-11-09): a committed DEFAULT buffer of 834,355,456 B
// (Qwen3.5-9B's Q6_K output.weight) fails with E_OUTOFMEMORY in a fresh
// process while DXGI reports 4.1 GB of budget free; Series S placed a single
// 1834 MiB buffer. A reserved buffer gives one contiguous GPU VA backed by
// small heaps, so a size limit on one allocation stops mattering.
constexpr UINT64 kTileBytes = 64ull << 10;
constexpr UINT64 kTiledHeapBytes = 64ull << 20;

D3D12_HEAP_PROPERTIES custom_l0_props() {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    return hp;
}

ComPtr<ID3D12Heap> create_heap(ID3D12Device* device, UINT64 bytes, bool custom_l0,
                               HRESULT* hr_out) {
    D3D12_HEAP_DESC hd = {};
    hd.SizeInBytes = bytes;
    if (custom_l0)
        hd.Properties = custom_l0_props();
    else
        hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    ComPtr<ID3D12Heap> heap;
    const HRESULT hr = device->CreateHeap(&hd, IID_PPV_ARGS(&heap));
    if (hr_out)
        *hr_out = hr;
    if (FAILED(hr))
        return {};
    return heap;
}

// Reserved buffer of `bytes` (rounded up to 64 KiB) whose tiles are mapped,
// in order, onto kTiledHeapBytes heaps (DEFAULT, else CUSTOM L0 per heap).
ComPtr<ID3D12Resource> create_tiled_buffer(Gpu& g, UINT64 bytes,
                                           std::vector<ComPtr<ID3D12Heap>>* heaps,
                                           std::string* err) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS o = {};
    if (FAILED(g.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof(o))) ||
        o.TiledResourcesTier == D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED) {
        *err = "tiled resources not supported";
        return {};
    }
    bytes = (bytes + kTileBytes - 1) / kTileBytes * kTileBytes;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> res;
    HRESULT hr = g.device->CreateReservedResource(&rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                  IID_PPV_ARGS(&res));
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("CreateReservedResource", hr);
        return {};
    }
    const UINT64 tiles = bytes / kTileBytes, per_heap = kTiledHeapBytes / kTileBytes;
    std::lock_guard<std::mutex> lock(g.mu);
    for (UINT64 t0 = 0; t0 < tiles; t0 += per_heap) {
        const UINT64 n = std::min(per_heap, tiles - t0);
        HRESULT hr_default = S_OK;
        ComPtr<ID3D12Heap> heap = create_heap(g.device.Get(), n * kTileBytes, false, &hr_default);
        if (!heap)
            heap = create_heap(g.device.Get(), n * kTileBytes, true, &hr);
        if (!heap) {
            *err = d3d12c::hr_message("CreateHeap (tiled backing)", hr_default) + " after " +
                   std::to_string(heaps->size()) + " heaps";
            heaps->clear();
            return {};
        }
        D3D12_TILED_RESOURCE_COORDINATE coord = {};
        coord.X = static_cast<UINT>(t0);
        D3D12_TILE_REGION_SIZE region = {};
        region.NumTiles = static_cast<UINT>(n);
        region.UseBox = FALSE;
        const D3D12_TILE_RANGE_FLAGS range_flags = D3D12_TILE_RANGE_FLAG_NONE;
        const UINT heap_start = 0, range_tiles = static_cast<UINT>(n);
        g.queue->UpdateTileMappings(res.Get(), 1, &coord, &region, heap.Get(), 1, &range_flags,
                                    &heap_start, &range_tiles, D3D12_TILE_MAPPING_FLAG_NONE);
        heaps->push_back(heap);
    }
    if (!g.fence.signal_and_wait(g.queue.Get(), /*spin=*/false)) {
        *err = "fence wait after UpdateTileMappings failed";
        heaps->clear();
        return {};
    }
    return res;
}

std::size_t g_weights_max_buffer = 0; // 0 = no limit (ggml_d3d12_set_weights_max_buffer)

void settle_host(ggml_backend_buffer_t b, BufCtx* c);

void* buf_base(ggml_backend_buffer_t b) {
    auto* c = static_cast<BufCtx*>(b->context);
    if (c->wc && !c->settled)
        settle_host(b, c);
    return c->is_host ? static_cast<void*>(c->host)
                      : reinterpret_cast<void*>(static_cast<std::uintptr_t>(c->va));
}

void buf_free(ggml_backend_buffer_t b) {
    delete static_cast<BufCtx*>(b->context);
}

std::size_t tensor_offset(const ggml_tensor* t) {
    return static_cast<std::size_t>(static_cast<const std::uint8_t*>(t->data) -
                                    static_cast<const std::uint8_t*>(buf_base(t->buffer)));
}

D3D12_GPU_VIRTUAL_ADDRESS tensor_va(const ggml_tensor* t) {
    return static_cast<BufCtx*>(t->buffer->context)->va + tensor_offset(t);
}

// D3D12_Host: plain memory.
void host_memset(ggml_backend_buffer_t, ggml_tensor* t, uint8_t v, size_t off, size_t size) {
    std::memset(static_cast<std::uint8_t*>(t->data) + off, v, size);
}
void host_set(ggml_backend_buffer_t, ggml_tensor* t, const void* data, size_t off, size_t size) {
    std::memcpy(static_cast<std::uint8_t*>(t->data) + off, data, size);
}
// A write-combined buffer is uncached for the CPU: reading the 62.8 MiB
// recurrent state of the 35B out of one ran at ~127 MB/s on Series X (KV
// checkpoint 494 ms against 16 ms write-back, 2026-11-11), while the GPU reads
// it at full speed. Reads of kWcGpuReadMin or more go through the readback
// ring, as for weights; buffers decay to COMMON after every submission, so the
// copy source is promoted implicitly. A failed submission falls back to memcpy.
constexpr std::size_t kWcGpuReadMin = 64u << 10;

void host_get(ggml_backend_buffer_t b, const ggml_tensor* t, void* data, size_t off, size_t size) {
    const auto* src = static_cast<const std::uint8_t*>(t->data) + off;
    auto* c = static_cast<BufCtx*>(b->context);
    if (!c->wc || size < kWcGpuReadMin) {
        std::memcpy(data, src, size);
        return;
    }
    const std::size_t src_off = tensor_offset(t) + off;
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    for (std::size_t done = 0; done < size;) {
        const std::size_t chunk = std::min<std::size_t>(size - done, kStagingBytes);
        const bool ok = run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(g.readback.Get(), 0, c->res.Get(), src_off + done, chunk);
        });
        std::memcpy(static_cast<std::uint8_t*>(data) + done, ok ? g.readback_ptr : src + done,
                    chunk);
        done += chunk;
    }
}
bool host_cpy(ggml_backend_buffer_t, const ggml_tensor* src, ggml_tensor* dst) {
    if (!ggml_backend_buffer_is_host(src->buffer))
        return false;
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;
}
void host_clear(ggml_backend_buffer_t b, uint8_t v) {
    std::memset(buf_base(b), v, b->size);
}

const ggml_backend_buffer_i kHostBufIface = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ host_memset,
    /* .set_tensor    = */ host_set,
    /* .get_tensor    = */ host_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ host_cpy,
    /* .clear         = */ host_clear,
    /* .reset         = */ nullptr,
};

// D3D12_Weights: DEFAULT heap, filled through the staging ring. The buffer
// stays in COMMON; buffers promote implicitly to COPY_DEST / shader read and
// decay back at the end of each submission.
void weights_write(ggml_backend_buffer_t b, std::size_t dst_off, const void* data, std::size_t size,
                   int fill) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    auto* c = static_cast<BufCtx*>(b->context);
    for (std::size_t done = 0; done < size;) {
        const std::size_t chunk = std::min<std::size_t>(size - done, kStagingBytes);
        if (data)
            std::memcpy(g.staging_ptr, static_cast<const std::uint8_t*>(data) + done, chunk);
        else
            std::memset(g.staging_ptr, fill, chunk);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(c->res.Get(), dst_off + done, g.staging.Get(), 0, chunk);
        });
        done += chunk;
    }
}
void weights_memset(ggml_backend_buffer_t b, ggml_tensor* t, uint8_t v, size_t off, size_t size) {
    weights_write(b, tensor_offset(t) + off, nullptr, size, v);
}
void weights_set(ggml_backend_buffer_t b, ggml_tensor* t, const void* data, size_t off,
                 size_t size) {
    weights_write(b, tensor_offset(t) + off, data, size, 0);
}
void weights_get(ggml_backend_buffer_t b, const ggml_tensor* t, void* data, size_t off,
                 size_t size) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    auto* c = static_cast<BufCtx*>(b->context);
    const std::size_t src_off = tensor_offset(t) + off;
    for (std::size_t done = 0; done < size;) {
        const std::size_t chunk = std::min<std::size_t>(size - done, kStagingBytes);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(g.readback.Get(), 0, c->res.Get(), src_off + done, chunk);
        });
        std::memcpy(static_cast<std::uint8_t*>(data) + done, g.readback_ptr, chunk);
        done += chunk;
    }
}
void weights_clear(ggml_backend_buffer_t b, uint8_t v) {
    weights_write(b, 0, nullptr, b->size, v);
}

const ggml_backend_buffer_i kWeightsBufIface = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ weights_memset,
    /* .set_tensor    = */ weights_set,
    /* .get_tensor    = */ weights_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ weights_clear,
    /* .reset         = */ nullptr,
};

// --- Buffer types ---

ggml_backend_dev_t device_ptr();

const char* host_buft_name(ggml_backend_buffer_type_t) {
    return "D3D12_Host";
}
const char* weights_buft_name(ggml_backend_buffer_type_t) {
    return "D3D12_Weights";
}
size_t buft_alignment(ggml_backend_buffer_type_t) {
    return 256;
}
bool buft_is_host_true(ggml_backend_buffer_type_t) {
    return true;
}
// The Q4_0/Q6_K/IQ3_S kernels read whole aligned dwords, up to 2 bytes past a
// tensor's last block; root descriptors have no bounds check, so pad.
size_t weights_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor* t) {
    return ggml_nbytes(t) + 16;
}

// XLLAMA_D3D12_WC_MIB=N: D3D12_Host buffers of at least N MiB are write-combined
// (CPU-uncached) instead of write-back. Profile of the 35B on Series X
// (2026-11-09): kernels reading D3D12_Weights (DEFAULT) reach 186-267 GB/s,
// the same IQ4_XS matvec over expert slots in write-back D3D12_Host ~65 GB/s,
// and the ssm kernels on the write-back recurrent state 50-80 GB/s. The CPU
// only writes the big ones (ReadFile into slots); it reads the small output
// and model buffers, which a threshold keeps write-back. Unset or 0 = off.
// A compute buffer is never write-combined, whatever its size: the CPU splits
// read their inputs from it (WC_MIB=16 put the 34 MiB one there: decode 2.2x
// slower); settle_host moves one back to write-back on its first use.
bool host_wc(size_t bytes) {
    const char* v = std::getenv("XLLAMA_D3D12_WC_MIB");
    if (!v || !*v)
        return false;
    char* end = nullptr;
    const unsigned long long mib = std::strtoull(v, &end, 10);
    return end != v && mib > 0 && bytes >= (static_cast<std::size_t>(mib) << 20);
}

// A mapped D3D12_Host resource (CUSTOM, L0), write-back or write-combined.
bool create_host_res(Gpu& g, size_t size, bool wc, BufCtx* c, std::string* err) {
    D3D12_HEAP_PROPERTIES hp = custom_l0_props();
    if (wc)
        hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
    c->res = d3d12c::create_buffer_props(
        g.device.Get(), size, hp, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, wc ? "D3D12_Host_WC" : "D3D12_Host", err);
    void* p = nullptr;
    c->host =
        c->res && SUCCEEDED(c->res->Map(0, nullptr, &p)) ? static_cast<std::uint8_t*>(p) : nullptr;
    if (c->res && !c->host)
        c->res.Reset();
    c->va = c->res ? c->res->GetGPUVirtualAddress() : 0;
    c->wc = wc && c->res;
    return c->res != nullptr;
}

// The memory type is chosen at allocation, by size; what the buffer is for is
// only known later: ggml-alloc marks a graph allocator's buffer COMPUTE right
// after allocating it and before its first get_base (ggml_vbuffer_alloc), the
// recurrent state and the expert slots stay ANY. On the first base or clear,
// before any tensor has an address in it, a write-combined compute buffer is
// replaced by a write-back one of the same size (released first, so the heap
// never holds both).
void settle_host(ggml_backend_buffer_t b, BufCtx* c) {
    c->settled = true;
    if (ggml_backend_buffer_get_usage(b) != GGML_BACKEND_BUFFER_USAGE_COMPUTE)
        return;
    Gpu& g = gpu();
    c->res.Reset();
    std::string err;
    const bool wb = create_host_res(g, b->size, false, c, &err);
    if (!wb && !create_host_res(g, b->size, true, c, &err))
        GGML_ABORT("d3d12: compute buffer of %zu bytes lost its memory: %s", b->size, err.c_str());
    char msg[160];
    std::snprintf(msg, sizeof(msg), "[xllama] d3d12: compute buffer %.1f MiB %s\n",
                  static_cast<double>(b->size) / 1048576.0,
                  wb ? "moved to write-back" : "stays write-combined (write-back failed)");
    log_output(wb ? std::string(msg) : std::string(msg) + "  " + err + "\n");
}

ggml_backend_buffer_t alloc_buf(ggml_backend_buffer_type_t buft, size_t size, bool host) {
    Gpu& g = gpu();
    if (!g.ok)
        return nullptr;
    size = std::max<size_t>(size, 256);
    auto* c = new BufCtx;
    c->is_host = host;
    std::string err;
    bool wc = false;
    if (host) {
        if (host_wc(size)) {
            std::string err_wc;
            wc = create_host_res(g, size, true, c, &err_wc);
            if (!wc)
                log_output(
                    "[xllama] d3d12: write-combined host buffer failed, write-back instead: " +
                    err_wc + "\n");
        }
        if (!c->res)
            create_host_res(g, size, false, c, &err);
    } else {
        c->res = d3d12c::create_buffer(g.device.Get(), size, D3D12_HEAP_TYPE_DEFAULT,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON,
                                       "D3D12_Weights", &err);
        const char* fallback = nullptr;
        if (!c->res) {
            std::string err_l0;
            c->res = d3d12c::create_buffer_props(
                g.device.Get(), size, custom_l0_props(), D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COMMON, "D3D12_Weights_L0", &err_l0);
            if (c->res)
                fallback = "CUSTOM L0 committed";
            else
                err += "; " + err_l0;
        }
        if (!c->res) {
            std::string err_tiled;
            c->res = create_tiled_buffer(g, size, &c->heaps, &err_tiled);
            if (c->res)
                fallback = "reserved + 64 MiB heaps";
            else
                err += "; " + err_tiled;
        }
        if (fallback) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "[xllama] d3d12: weights %.1f MiB: DEFAULT committed failed, %s\n",
                          static_cast<double>(size) / 1048576.0, fallback);
            log_output(msg);
        }
    }
    if (!c->res || (host && !c->host)) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "[xllama] d3d12: allocation of %zu bytes failed: %s\n",
                      size, err.c_str());
        log_output(msg);
        delete c;
        return nullptr;
    }
    c->va = c->res->GetGPUVirtualAddress();
    if (size >= (1u << 20)) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "[xllama] d3d12: %s buffer %.1f MiB%s\n",
                      host ? "D3D12_Host" : "D3D12_Weights", static_cast<double>(size) / 1048576.0,
                      wc ? " (write-combined)" : "");
        log_output(msg);
    }
    return ggml_backend_buffer_init(buft, host ? kHostBufIface : kWeightsBufIface, c, size);
}

ggml_backend_buffer_t host_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    return alloc_buf(buft, size, true);
}
ggml_backend_buffer_t weights_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    return alloc_buf(buft, size, false);
}
size_t weights_max_size(ggml_backend_buffer_type_t) {
    return g_weights_max_buffer ? g_weights_max_buffer : SIZE_MAX;
}

ggml_backend_buffer_type kHostBuft = {
    /* .iface = */
    {
        /* .get_name         = */ host_buft_name,
        /* .alloc_buffer     = */ host_alloc,
        /* .alloc_buffer_n   = */ nullptr,
        /* .get_alignment    = */ buft_alignment,
        /* .get_max_size     = */ nullptr,
        /* .get_alloc_size   = */ nullptr,
        /* .get_alloc_size_n = */ nullptr,
        /* .is_host          = */ buft_is_host_true,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};
ggml_backend_buffer_type kWeightsBuft = {
    /* .iface = */
    {
        /* .get_name         = */ weights_buft_name,
        /* .alloc_buffer     = */ weights_alloc,
        /* .alloc_buffer_n   = */ nullptr,
        /* .get_alignment    = */ buft_alignment,
        /* .get_max_size     = */ weights_max_size,
        /* .get_alloc_size   = */ weights_alloc_size,
        /* .get_alloc_size_n = */ nullptr,
        /* .is_host          = */ nullptr,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

bool is_ours(ggml_backend_buffer_type_t buft) {
    return buft == &kHostBuft || buft == &kWeightsBuft;
}

// The expert cache's buffer type (llama.cpp patch 0004) offered to this device
// by llama_expert_cache_use_host_of: claiming it puts the resident expert slots
// in D3D12_Host buffers, which the MUL_MAT_ID kernel reads in place while the
// cache's CPU ops keep reading the experts into them (unified memory).
bool is_expert_cache(ggml_backend_buffer_type_t buft) {
    return buft && buft->device && buft->device == kHostBuft.device && buft->iface.get_name &&
           std::strcmp(buft->iface.get_name(buft), "EXPERT_CACHE") == 0;
}

// An expert tensor the MUL_MAT_ID kernel can read: in a D3D12_Host buffer (by
// its iface — the buffer's buft is the cache's) of a cache this device claimed.
// When D3D12_Host cannot hold the slots, the cache falls back to CPU buffers
// and points its type at the CPU device, so neither test passes.
bool in_expert_cache(const ggml_tensor* t) {
    const ggml_backend_buffer_t b = t ? t->buffer : nullptr;
    return b && is_expert_cache(b->buft) && b->iface.get_base == kHostBufIface.get_base;
}

// A pipeline's kernel, for the profile.
std::string pso_name(const Gpu& g, const ID3D12PipelineState* p) {
    static const char* const kTypes[] = {"q4_0", "q4_K", "q6_K", "iq3_s", "iq4_xs", "q5_K", "f32"};
    static const char* const kElts[] = {"elt binary", "elt rms_norm", "elt quant q8_K"};
    static const char* const kMoes[] = {"moe combine", "moe tail"};
    static const char* const kSsms[] = {"ssm unary", "ssm copy", "ssm get_rows", "ssm conv",
                                        "ssm gated delta net"};
    static_assert(static_cast<int>(std::size(kTypes)) == kPsoCount &&
                      static_cast<int>(std::size(kElts)) == kEltCount &&
                      static_cast<int>(std::size(kMoes)) == kMoeCount &&
                      static_cast<int>(std::size(kSsms)) == kSsmCount,
                  "a name per kernel");
    if (const auto it = g.pso_over.find(p); it != g.pso_over.end())
        return it->second; // an XLLAMA_D3D12_SHADER_DIR blob, by its path
    for (int i = 0; i < kPsoCount; ++i)
        for (int wd = 0; wd < 2; ++wd)
            if (g.pso[i][wd].Get() == p)
                return std::string("mmv ") + kTypes[i] + (wd ? " t128" : " t64");
    if (g.pso_get_rows.Get() == p)
        return "get_rows q6_K";
    static const char* const kMmids[] = {"iq4_xs", "iq3_s"};
    static_assert(static_cast<int>(std::size(kMmids)) == kMmidCount, "a name per mmid kernel");
    for (int mk = 0; mk < kMmidCount; ++mk)
        for (int wd = 0; wd < 2; ++wd)
            if (g.pso_mmid[mk][wd].Get() == p)
                return std::string("mmid ") + kMmids[mk] + (wd ? " t128" : " t64");
    for (int i = 0; i < kEltCount; ++i)
        if (g.pso_elt[i].Get() == p)
            return kElts[i];
    for (int i = 0; i < kMoeCount; ++i)
        if (g.pso_moe[i].Get() == p)
            return kMoes[i];
    for (int i = 0; i < kSsmCount; ++i)
        if (g.pso_ssm[i].Get() == p)
            return kSsms[i];
    return "?";
}

// --- Backend ---

const char* backend_name(ggml_backend_t) {
    return "D3D12";
}
void backend_free(ggml_backend_t b) {
    Gpu& g = gpu();
    {
        std::lock_guard<std::mutex> lock(g.mu);
        char msg[384];
        std::snprintf(msg, sizeof(msg),
                      "[xllama] d3d12: %llu graph_compute calls, %llu submits, %llu dispatches, "
                      "%.1f ms wall (%.1f ms GPU; %.1f ms recording, %.1f ms waiting, %.1f ms "
                      "host); %llu submits replayed a kept list; %llu barriers; %llu delta-net "
                      "layers fused\n",
                      static_cast<unsigned long long>(g.n_calls),
                      static_cast<unsigned long long>(g.n_submits),
                      static_cast<unsigned long long>(g.n_dispatches), g.wall_ms, g.gpu_ms,
                      g.rec_ms, g.wait_ms, g.wall_ms - g.rec_ms - g.wait_ms,
                      static_cast<unsigned long long>(g.n_replays),
                      static_cast<unsigned long long>(g.n_barriers),
                      static_cast<unsigned long long>(g.n_gdn_fused));
        log_output(msg);
        std::vector<
            std::pair<std::tuple<const ID3D12PipelineState*, UINT, UINT, UINT>, Gpu::SegStat>>
            ks(g.kernels.begin(), g.kernels.end());
        std::sort(ks.begin(), ks.end(),
                  [](const auto& x, const auto& y) { return x.second.ms > y.second.ms; });
        if (ks.size() > 40)
            ks.resize(40);
        for (const auto& [key, s] : ks) {
            char line[256];
            std::snprintf(
                line, sizeof(line), "[xllama] d3d12 kernel: %9llu x %9.1f ms  %s grid %ux%ux%u\n",
                static_cast<unsigned long long>(s.n), s.ms, pso_name(g, std::get<0>(key)).c_str(),
                std::get<1>(key), std::get<2>(key), std::get<3>(key));
            log_output(line);
        }
        std::vector<std::pair<std::string, Gpu::SegStat>> top(g.segs.begin(), g.segs.end());
        std::sort(top.begin(), top.end(),
                  [](const auto& x, const auto& y) { return x.second.ms > y.second.ms; });
        if (top.size() > 24)
            top.resize(24);
        for (const auto& [key, s] : top) {
            char line[384];
            std::snprintf(line, sizeof(line), "[xllama] d3d12 split: %8llu x %9.1f ms  %s\n",
                          static_cast<unsigned long long>(s.n), s.ms, key.c_str());
            log_output(line);
        }
        g.n_calls = g.n_submits = g.n_dispatches = g.n_replays = g.n_barriers = g.n_gdn_fused = 0;
        g.wall_ms = g.gpu_ms = g.rec_ms = g.wait_ms = 0.0;
        g.segs.clear();
        g.kernels.clear();
    }
    delete b;
}

// A node's name without its layer suffix or view decoration ("ffn_moe_up-12
// (view)" -> "ffn_moe_up"), so the same split of every layer is one key.
std::string seg_name(const ggml_tensor* t) {
    std::string s = t->name[0] ? t->name : ggml_op_desc(t);
    const std::size_t paren = s.find(" (");
    if (paren != std::string::npos)
        s.resize(paren);
    const std::size_t dash = s.rfind('-');
    if (dash != std::string::npos && dash + 1 < s.size() &&
        s.find_first_not_of("0123456789", dash + 1) == std::string::npos)
        s.resize(dash);
    return s;
}

// Grow the q8 activation scratch. Caller holds g.mu and has nothing in flight.
bool ensure_xq(Gpu& g, std::size_t bytes) {
    if (bytes <= g.xq_bytes)
        return true;
    const std::size_t want = std::max<std::size_t>(bytes, std::size_t(1) << 20);
    g.xq.Reset();
    g.xq_ptr = nullptr;
    g.xq_bytes = 0;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    std::string err;
    g.xq = d3d12c::create_buffer_props(g.device.Get(), want, hp,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "D3D12_Xq", &err);
    void* p = nullptr;
    if (!g.xq || FAILED(g.xq->Map(0, nullptr, &p))) {
        log_output("[xllama] d3d12: q8 activation scratch allocation failed: " + err + "\n");
        g.xq.Reset();
        return false;
    }
    g.xq_ptr = static_cast<std::uint8_t*>(p);
    g.xq_bytes = want;
    return true;
}

// The tensor that owns a node's memory (follows view chains).
const ggml_tensor* base_of(const ggml_tensor* t) {
    while (t->view_src)
        t = t->view_src;
    return t;
}

// Host address range [lo, hi) a dispatch reads or writes.
struct Rng {
    std::uintptr_t lo = 0, hi = 0;
};

Rng rng_of(const ggml_tensor* t) {
    const auto lo = reinterpret_cast<std::uintptr_t>(t->data);
    return {lo, lo + ggml_nbytes(t)};
}

bool overlaps(const Rng& a, const Rng& b) {
    return a.lo < b.hi && b.lo < a.hi;
}

bool any_overlap(const std::vector<Rng>& v, const Rng& r) {
    for (const Rng& x : v)
        if (overlaps(x, r))
            return true;
    return false;
}

// An operand of the elementwise kernels: f32 rows in D3D12_Host, addressed
// with 32-bit byte offsets from the tensor start.
bool elt_operand(const ggml_tensor* t) {
    return t && t->buffer && t->buffer->buft == &kHostBuft && t->type == GGML_TYPE_F32 &&
           t->nb[0] == sizeof(float) && t->nb[1] % sizeof(float) == 0 &&
           ggml_nbytes(t) <= 0xffffffffull;
}

// Each element is read and written by one thread, so an output may be one of
// the inputs exactly (an in-place op) or lie apart from it — not partly over it.
bool elt_rows(const ggml_tensor* t) {
    return d3d12_elt_rows_supported(t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

// SwiGLU operands: rows from kD3d12GluMinRow wide (the expert FFNs of
// Qwen3.6-35B-A3B are 512) and [n_ff, n_expert_used, n_tokens] expert outputs,
// whose evenly spaced rows the binary kernel takes as ne1 * ne2 rows. The
// SwiGLU then joins the matmuls on either side of it in one submission.
constexpr std::int64_t kD3d12GluMinRow = 256;
bool glu_rows(const ggml_tensor* t) {
    if (t->ne[3] != 1 || (t->ne[2] > 1 && t->nb[2] != t->ne[1] * t->nb[1]))
        return false;
    const std::int64_t rows = t->ne[1] * t->ne[2];
    return t->ne[0] >= kD3d12GluMinRow && rows >= 1 && rows <= kD3d12MaxGroups &&
           (t->ne[0] + 255) / 256 <= kD3d12MaxGroups && t->ne[0] * rows <= (std::int64_t(1) << 28);
}

bool same_or_apart(const ggml_tensor* d, const ggml_tensor* s) {
    if (d->data == s->data && d->nb[1] == s->nb[1])
        return true;
    return !overlaps(rng_of(d), rng_of(s));
}

// --- The linear-attention (gated delta net) chain (ggml_d3d12_ssm_emu.h) ---

// Rows of t as the elementwise kernels walk them, nb1 apart: ne1 * ne2 * ne3
// when the outer dims are packed over the rows, else 0.
std::int64_t flat_rows(const ggml_tensor* t) {
    if ((t->ne[2] > 1 && t->nb[2] != t->ne[1] * t->nb[1]) ||
        (t->ne[3] > 1 && t->nb[3] != t->ne[2] * t->nb[2]))
        return 0;
    return t->ne[1] * t->ne[2] * t->ne[3];
}

// The elementwise kernels' limits over flat rows of any width: the delta net's
// [32, T] gates and [128, heads] q, k and outputs are narrower than
// kD3d12EltMinRow, which keeps the CPU's small ops off the GPU elsewhere.
bool flat_ok(const ggml_tensor* t) {
    const std::int64_t rows = flat_rows(t);
    return t->ne[0] >= 1 && rows >= 1 && rows <= kD3d12MaxGroups &&
           (t->ne[0] + 255) / 256 <= kD3d12MaxGroups && t->ne[0] * rows <= (std::int64_t(1) << 28);
}

// An operand of the delta-net kernels: f32 in D3D12_Host, 32-bit offsets.
bool ssm_operand(const ggml_tensor* t) {
    return t && t->buffer && t->buffer->buft == &kHostBuft && t->type == GGML_TYPE_F32 &&
           ggml_nbytes(t) <= 0xffffffffull;
}

// The output of a one-thread-per-element kernel may be its input exactly (an
// in-place op) or lie apart from it.
bool ssm_same_or_apart(const ggml_tensor* d, const ggml_tensor* s) {
    if (d->data == s->data && std::equal(std::begin(d->nb), std::end(d->nb), std::begin(s->nb)))
        return true;
    return !overlaps(rng_of(d), rng_of(s));
}

// "blk.N.ssm_*" but ssm_out, whose matmul ends the layer's chain.
bool ssm_weight_name(const char* name) {
    if (std::strncmp(name, "blk.", 4) != 0)
        return false;
    const char* p = std::strchr(name + 4, '.');
    return p && std::strncmp(p + 1, "ssm_", 4) == 0 && std::strncmp(p + 1, "ssm_out", 7) != 0;
}

// Where a delta-net chain starts (qwen35moe build_layer_attn_linear): the
// recurrent states (llama-memory-recurrent.cpp), a layer's ssm_* weights or
// the matmuls by them, and the z projection, whose SILU gates the output
// (a full-attention layer gates with a SIGMOID instead).
bool ssm_anchor(const ggml_tensor* t, const ggml_tensor* user) {
    if (t->op == GGML_OP_NONE)
        return std::strncmp(t->name, "cache_r_l", 9) == 0 ||
               std::strncmp(t->name, "cache_s_l", 9) == 0 || ssm_weight_name(t->name);
    if (t->op != GGML_OP_MUL_MAT || !t->src[0])
        return false;
    const char* w = t->src[0]->name;
    if (ssm_weight_name(w))
        return true;
    const char* gate = std::strstr(w, ".attn_gate.");
    return gate && user->op == GGML_OP_UNARY && ggml_get_unary_op(user) == GGML_UNARY_OP_SILU;
}

bool ssm_op(const ggml_tensor* t) {
    switch (t->op) {
    case GGML_OP_UNARY:
    case GGML_OP_SCALE:
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_CONCAT:
    case GGML_OP_GET_ROWS:
    case GGML_OP_SSM_CONV:
    case GGML_OP_GATED_DELTA_NET:
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_RMS_NORM:
        return true;
    default:
        return false;
    }
}

// Whether t belongs to a delta-net chain: an operand is an anchor, or a chain
// op whose own operands reach one within `depth` more steps. Only these small
// ops move to the GPU; the same op types elsewhere stay where they ran.
bool ssm_chain(const ggml_tensor* t, int depth = 3) {
    for (const ggml_tensor* s : t->src) {
        if (!s)
            continue;
        const ggml_tensor* b = base_of(s);
        if (ssm_anchor(b, t) || (depth > 0 && ssm_op(b) && ssm_chain(b, depth - 1)))
            return true;
    }
    return false;
}

// An f32 operand of the delta-net kernels as supports_op sees it.
bool ssm_f32(const ggml_tensor* t) {
    const ggml_tensor* b = base_of(t);
    return t->type == GGML_TYPE_F32 && (!b->buffer || b->buffer->buft == &kHostBuft);
}

// The bytes one non-empty block of a copy plan writes into `node`.
Rng copy_rng(const ggml_tensor* node, const D3d12SsmCopy& c) {
    std::uint64_t last = 0;
    for (int k = 0; k < 4; ++k)
        last += static_cast<std::uint64_t>(c.params.ne[k] - 1) * c.params.dst_stride[k];
    const auto lo = reinterpret_cast<std::uintptr_t>(node->data) + c.dst_offset;
    return {lo, lo + (last + 1) * sizeof(float)};
}

static_assert(sizeof(D3d12SsmUnaryParams) <= kOpConstants * sizeof(std::uint32_t) &&
                  sizeof(D3d12SsmCopyParams) <= kOpConstants * sizeof(std::uint32_t) &&
                  sizeof(D3d12SsmGetRowsParams) <= kOpConstants * sizeof(std::uint32_t) &&
                  sizeof(D3d12SsmConvParams) <= kOpConstants * sizeof(std::uint32_t) &&
                  sizeof(D3d12GdnParams) <= kOpConstants * sizeof(std::uint32_t),
              "the delta-net constants fit the root signature");

// CPU versions of the elementwise kernels, for operands the GPU path refuses.
float* row4_f32(const ggml_tensor* t, std::int64_t i1, std::int64_t i2, std::int64_t i3) {
    return reinterpret_cast<float*>(static_cast<char*>(t->data) + i3 * t->nb[3] + i2 * t->nb[2] +
                                    i1 * t->nb[1]);
}

void cpu_binary(const ggml_tensor* d, int kind) {
    const ggml_tensor* a = d->src[0];
    const ggml_tensor* b = d->src[1];
    for (int k = 1; k < GGML_MAX_DIMS; ++k)
        GGML_ASSERT(b->ne[k] == 1 || b->ne[k] == d->ne[k]);
    for (std::int64_t i3 = 0; i3 < d->ne[3]; ++i3)
        for (std::int64_t i2 = 0; i2 < d->ne[2]; ++i2)
            for (std::int64_t i1 = 0; i1 < d->ne[1]; ++i1) {
                const float* pa = row4_f32(a, i1, i2, i3);
                const float* pb = row4_f32(b, b->ne[1] == 1 ? 0 : i1, b->ne[2] == 1 ? 0 : i2,
                                           b->ne[3] == 1 ? 0 : i3);
                float* pd = row4_f32(d, i1, i2, i3);
                for (std::int64_t i = 0; i < d->ne[0]; ++i) {
                    const float x = pa[i], y = pb[i];
                    pd[i] = kind == 0 ? x + y : kind == 1 ? x * y : x / (1.0f + std::exp(-x)) * y;
                }
            }
}

void cpu_rms_norm(const ggml_tensor* d, float eps) {
    const ggml_tensor* a = d->src[0];
    GGML_ASSERT(ggml_are_same_shape(d, a));
    for (std::int64_t i3 = 0; i3 < d->ne[3]; ++i3)
        for (std::int64_t i2 = 0; i2 < d->ne[2]; ++i2)
            for (std::int64_t i1 = 0; i1 < d->ne[1]; ++i1) {
                const float* pa = row4_f32(a, i1, i2, i3);
                double sum = 0.0; // as ggml-cpu
                for (std::int64_t i = 0; i < d->ne[0]; ++i)
                    sum += static_cast<double>(pa[i] * pa[i]);
                const float mean = static_cast<float>(sum / static_cast<double>(d->ne[0]));
                const float scale = 1.0f / std::sqrt(mean + eps);
                float* pd = row4_f32(d, i1, i2, i3);
                for (std::int64_t i = 0; i < d->ne[0]; ++i)
                    pd[i] = pa[i] * scale;
            }
}

// One recorded command: a dispatch, or a UAV barrier (pso null). Root
// parameters left 0 are bound to u[0]; the kernel does not read them.
struct Op {
    ID3D12PipelineState* pso = nullptr;
    std::uint32_t c[kOpConstants] = {};
    D3D12_GPU_VIRTUAL_ADDRESS t0 = 0;
    D3D12_GPU_VIRTUAL_ADDRESS t1 = 0;
    D3D12_GPU_VIRTUAL_ADDRESS u[6] = {}; // u[5]: only the kernels that read u5 set it
    UINT gx = 0, gy = 0, gz = 1;
};

void set_c(Op& op, std::initializer_list<std::uint64_t> c) {
    GGML_ASSERT(c.size() <= kOpConstants);
    std::size_t i = 0;
    for (const std::uint64_t v : c)
        op.c[i++] = static_cast<std::uint32_t>(v);
}

// XLLAMA_D3D12_PROFILE's timestamps per submission: one before the first
// dispatch, then one after each; dispatches past the last go uncounted.
constexpr UINT kProfStamps = 1024;

// The commands of one submission; returns its dispatch count. A pure function
// of ops, ts and prof (with the process-lifetime root signature, PSOs and
// query heaps), which is what lets a kept list stand in for a fresh recording.
std::uint64_t record_ops(Gpu& g, ID3D12GraphicsCommandList* cl, const std::vector<Op>& ops, bool ts,
                         bool prof) {
    std::uint64_t n = 0;
    UINT stamps = 0;
    cl->SetComputeRootSignature(g.root.Get());
    if (ts)
        cl->EndQuery(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    if (prof)
        cl->EndQuery(g.prof.Get(), D3D12_QUERY_TYPE_TIMESTAMP, stamps++);
    ID3D12PipelineState* bound = nullptr;
    for (const Op& op : ops) {
        if (!op.pso) {
            D3D12_RESOURCE_BARRIER b = {};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            b.UAV.pResource = nullptr; // every UAV access
            cl->ResourceBarrier(1, &b);
            continue;
        }
        if (op.pso != bound) {
            cl->SetPipelineState(op.pso);
            bound = op.pso;
        }
        cl->SetComputeRoot32BitConstants(0, kOpConstants, op.c, 0);
        cl->SetComputeRootShaderResourceView(1, op.t0 ? op.t0 : op.u[0]);
        for (UINT i = 0; i < 4; ++i)
            cl->SetComputeRootUnorderedAccessView(2 + i, op.u[i] ? op.u[i] : op.u[0]);
        cl->SetComputeRootShaderResourceView(6, op.t1 ? op.t1 : op.u[0]);
        cl->SetComputeRootUnorderedAccessView(7, op.u[4] ? op.u[4] : op.u[0]);
        if (op.u[5]) // no other shader declares u5: an unset root UAV it does not read is valid
            cl->SetComputeRootUnorderedAccessView(8, op.u[5]);
        cl->Dispatch(op.gx, op.gy, op.gz);
        ++n;
        if (prof && stamps < kProfStamps)
            cl->EndQuery(g.prof.Get(), D3D12_QUERY_TYPE_TIMESTAMP, stamps++);
    }
    if (ts) {
        cl->EndQuery(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        cl->ResolveQueryData(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, g.ts_rb.Get(), 0);
    }
    if (prof)
        cl->ResolveQueryData(g.prof.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, stamps, g.prof_rb.Get(),
                             0);
    return n;
}

// Each profiled dispatch's time, from the stamps record_ops resolved: the gap
// from the previous stamp, so a barrier counts towards the dispatch after it.
void add_profile(Gpu& g, const std::vector<Op>& ops) {
    void* p = nullptr;
    if (FAILED(g.prof_rb->Map(0, nullptr, &p)))
        return;
    const auto* t = static_cast<const std::uint64_t*>(p);
    UINT k = 1;
    for (const Op& op : ops) {
        if (!op.pso)
            continue;
        if (k >= kProfStamps)
            break;
        Gpu::SegStat& s = g.kernels[std::make_tuple(op.pso, op.gx, op.gy, op.gz)];
        ++s.n;
        if (t[k] > t[k - 1])
            s.ms += 1000.0 * static_cast<double>(t[k] - t[k - 1]) / static_cast<double>(g.ts_freq);
        ++k;
    }
    g.prof_rb->Unmap(0, nullptr);
}

bool same_op(const Op& a, const Op& b) {
    return a.pso == b.pso && std::equal(std::begin(a.c), std::end(a.c), std::begin(b.c)) &&
           a.t0 == b.t0 && a.t1 == b.t1 &&
           std::equal(std::begin(a.u), std::end(a.u), std::begin(b.u)) && a.gx == b.gx &&
           a.gy == b.gy && a.gz == b.gz;
}

std::uint64_t ops_hash(const std::vector<Op>& ops, bool ts, bool prof) {
    std::uint64_t h = 1469598103934665603ull ^ (ts ? 1u : 0u) ^ (prof ? 2u : 0u);
    auto mix = [&h](std::uint64_t v) {
        h = (h ^ v) * 1099511628211ull;
        h ^= h >> 29;
    };
    for (const Op& op : ops) {
        mix(reinterpret_cast<std::uintptr_t>(op.pso));
        for (const std::uint32_t c : op.c)
            mix(c);
        mix(op.t0);
        mix(op.t1);
        for (const D3D12_GPU_VIRTUAL_ADDRESS u : op.u)
            mix(u);
        mix((static_cast<std::uint64_t>(op.gx) << 32) | op.gy);
        mix(op.gz);
    }
    return h;
}

// Once llama.cpp reuses the decode graph, every token submits the same ops on
// the same addresses (row ids and expert ids change in the scratch, not in the
// commands). A submission seen a second time is recorded into a list of its
// own and kept; from then on that list is executed again instead of recorded:
// ~57 us of D3D12 calls per submission on Series X (1.7.12.21, 35B decode).
// Every submission is waited for, so a kept list is never in flight when it is
// executed again or dropped.
struct KeptList {
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    std::vector<Op> ops;
    std::uint64_t n = 0;
};
struct ListCache {
    std::unordered_map<std::uint64_t, KeptList> kept; // by ops_hash
    std::unordered_set<std::uint64_t> seen;
};
// The 35B decode graph has ~84 GPU submissions; prefill adds its own.
constexpr std::size_t kMaxKeptLists = 256;

ListCache& list_cache() { // guarded by gpu().mu
    static ListCache c;
    return c;
}

// Record and run one submission. Caller holds g.mu.
bool submit_ops(Gpu& g, const std::vector<Op>& ops) {
    const bool ts = g.ts && g.ts_rb;
    const bool prof = g.prof_on && g.prof && g.prof_rb && g.ts_freq > 0;
    ListCache& lc = list_cache();
    const std::uint64_t h = ops_hash(ops, ts, prof);
    const auto it = lc.kept.find(h);
    std::uint64_t n = 0;
    bool ran = false;
    bool done = false;
    if (it != lc.kept.end() && it->second.ops.size() == ops.size() &&
        std::equal(ops.begin(), ops.end(), it->second.ops.begin(), same_op)) {
        ran = execute_and_wait(g, it->second.cl.Get(), std::chrono::steady_clock::now(), &g.rec_ms,
                               &g.wait_ms);
        n = it->second.n;
        ++g.n_replays;
        done = true;
    } else if (it == lc.kept.end() && !lc.seen.insert(h).second) {
        if (lc.kept.size() >= kMaxKeptLists)
            lc.kept.clear();
        if (lc.seen.size() > 4 * kMaxKeptLists)
            lc.seen.clear();
        const auto t0 = std::chrono::steady_clock::now();
        KeptList k;
        HRESULT hr = g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                      IID_PPV_ARGS(&k.alloc));
        if (SUCCEEDED(hr))
            hr = g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, k.alloc.Get(),
                                             nullptr, IID_PPV_ARGS(&k.cl));
        if (SUCCEEDED(hr)) {
            k.n = record_ops(g, k.cl.Get(), ops, ts, prof);
            hr = k.cl->Close();
        }
        if (SUCCEEDED(hr)) {
            ran = execute_and_wait(g, k.cl.Get(), t0, &g.rec_ms, &g.wait_ms);
            n = k.n;
            done = true;
            if (ran) {
                k.ops = ops;
                lc.kept.emplace(h, std::move(k));
            }
        }
    }
    if (!done)
        ran = run_now(
            g, [&](ID3D12GraphicsCommandList* cl) { n = record_ops(g, cl, ops, ts, prof); },
            &g.rec_ms, &g.wait_ms);
    if (!ran)
        return false;
    if (prof)
        add_profile(g, ops);
    g.n_barriers += static_cast<std::uint64_t>(
        std::count_if(ops.begin(), ops.end(), [](const Op& op) { return !op.pso; }));
    if (ts) {
        void* p = nullptr;
        if (SUCCEEDED(g.ts_rb->Map(0, nullptr, &p))) {
            const auto* t = static_cast<const std::uint64_t*>(p);
            g.last_gpu_ms += t[1] > t[0] ? 1000.0 * static_cast<double>(t[1] - t[0]) /
                                               static_cast<double>(g.ts_freq)
                                         : 0.0;
            g.ts_rb->Unmap(0, nullptr);
        }
    }
    g.n_dispatches += n;
    ++g.n_submits;
    return true;
}

// Whether the nodes of a MoE plan can run as one dispatch that writes only
// `out`: all are nodes of this graph from index `from` on, and nothing outside
// them reads the others (ggml's fusion check, with the graph's use counts).
bool moe_fusable(ggml_cgraph* cgraph, int from, const std::vector<const ggml_tensor*>& sub,
                 const ggml_tensor* out) {
    std::vector<int> idx;
    std::vector<ggml_op> ops;
    int out_idx = -1;
    const int n = ggml_graph_n_nodes(cgraph);
    for (const ggml_tensor* t : sub) {
        int j = from;
        while (j < n && ggml_graph_node(cgraph, j) != t)
            ++j;
        if (j == n)
            return false;
        idx.push_back(j);
        ops.push_back(t->op);
        if (t == out)
            out_idx = j;
    }
    return out_idx >= 0 &&
           ggml_can_fuse_subgraph_ext(cgraph, idx.data(), static_cast<int>(idx.size()), ops.data(),
                                      &out_idx, 1);
}

// A delta-net layer KERNEL 5 runs fused (d3d12_gdn_fuse_plan): the cache row
// it reads, at the row id the graph input holds now, and the row it writes.
struct GdnFused {
    D3D12_GPU_VIRTUAL_ADDRESS src_va = 0, dst_va = 0;
    Rng src, dst;
};

bool gdn_fuse_ok(const ggml_tensor* node, const D3d12GdnFusePlan& f, GdnFused* out) {
    D3d12GdnPlan p;
    if (!d3d12_gdn_plan(node, &p) || !ssm_operand(node) || !ssm_operand(f.states) ||
        !ssm_operand(f.dst) || !f.ids->buffer || !ggml_backend_buffer_is_host(f.ids->buffer))
        return false;
    for (const ggml_tensor* t : {p.q, p.k, p.v, p.g, p.beta})
        if (!ssm_operand(t) || overlaps(rng_of(t), rng_of(node)))
            return false;
    const std::int32_t row = *static_cast<const std::int32_t*>(f.ids->data);
    if (row < 0 || row >= f.states->ne[1])
        return false;
    const std::size_t at = static_cast<std::size_t>(row) * f.states->nb[1];
    const auto lo = reinterpret_cast<std::uintptr_t>(f.states->data) + at;
    out->src = {lo, lo + ggml_nbytes(f.dst)};
    out->dst = rng_of(f.dst);
    out->src_va = tensor_va(f.states) + at;
    out->dst_va = tensor_va(f.dst);
    // Load4 / Store4 from the bases: 16-byte aligned rows.
    return !overlaps(out->src, rng_of(node)) && !overlaps(out->dst, rng_of(node)) &&
           out->src_va % 16 == 0 && out->dst_va % 16 == 0;
}

ggml_status backend_graph_compute(ggml_backend_t, ggml_cgraph* cgraph) {
    Gpu& g = gpu();
    std::vector<ggml_tensor*> nodes;
    std::vector<int> node_idx; // each node's index in cgraph, for moe_fusable
    for (int i = 0; i < ggml_graph_n_nodes(cgraph); ++i) {
        ggml_tensor* node = ggml_graph_node(cgraph, i);
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)
            continue;
        switch (node->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_GET_ROWS:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_GLU:
        case GGML_OP_RMS_NORM:
        case GGML_OP_UNARY:
        case GGML_OP_SCALE:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
        case GGML_OP_CONCAT:
        case GGML_OP_SSM_CONV:
        case GGML_OP_GATED_DELTA_NET:
            nodes.push_back(node);
            node_idx.push_back(i);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        default:
            GGML_ABORT("d3d12: unsupported op %s", ggml_op_desc(node));
        }
    }
    if (nodes.empty())
        return GGML_STATUS_SUCCESS;

    std::lock_guard<std::mutex> lock(g.mu);
    const auto t0 = std::chrono::steady_clock::now();
    g.last_gpu_ms = 0.0;
    // Delta-net layers that read and write their recurrent cache rows
    // themselves: their state GET_ROWS and the CPY of the new state do not run.
    std::unordered_map<const ggml_tensor*, GdnFused> gdn_fused;
    std::unordered_set<const ggml_tensor*> fused_away;
    if (g.ssm_ok && g.gdn_fuse) {
        auto uses = [cgraph](const ggml_tensor* t) -> int {
            const std::size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
            return pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, pos)
                       ? cgraph->use_counts[pos]
                       : 0;
        };
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i]->op != GGML_OP_GATED_DELTA_NET)
                continue;
            D3d12GdnFusePlan f;
            GdnFused fz;
            if (!d3d12_gdn_fuse_plan(cgraph, node_idx[i], uses, &f) ||
                !gdn_fuse_ok(nodes[i], f, &fz))
                continue;
            gdn_fused.emplace(nodes[i], fz);
            fused_away.insert(f.get_rows);
            fused_away.insert(f.cpy);
        }
        g.n_gdn_fused += gdn_fused.size();
    }
    // The split's ops go into one submission. Dispatches run back to back, so
    // a UAV barrier goes in front of one that reads what another wrote since
    // the last barrier, or writes what one read or wrote (the graph allocator
    // reuses memory). CPU work — quantizing a Q4_0 input, staging row ids, an
    // op whose operands the GPU path refuses — happens now, before the GPU
    // runs, so it first runs the submission when that writes what it reads.
    // Matmuls sharing an input share one quantized copy.
    std::vector<Op> ops;
    std::vector<Rng> bar_r, bar_w, batch_w;
    struct Quantized {
        const void* data;
        std::size_t nb1;
        ggml_type type;
        std::size_t off;
        std::uint32_t row;
        int ncols;
        Rng src;
    };
    std::vector<Quantized> done;
    std::size_t used = 0;
    auto flush = [&]() -> bool {
        const bool ok = ops.empty() || submit_ops(g, ops);
        ops.clear();
        bar_r.clear();
        bar_w.clear();
        batch_w.clear();
        done.clear();
        used = 0;
        return ok;
    };
    auto record = [&](const Op& op, std::initializer_list<Rng> reads,
                      std::initializer_list<Rng> writes) {
        bool hazard = false;
        for (const Rng& r : reads)
            hazard = hazard || any_overlap(bar_w, r);
        for (const Rng& w : writes)
            hazard = hazard || any_overlap(bar_w, w) || any_overlap(bar_r, w);
        if (hazard) {
            ops.push_back(Op{});
            bar_r.clear();
            bar_w.clear();
        }
        ops.push_back(op);
        bar_r.insert(bar_r.end(), reads.begin(), reads.end());
        bar_w.insert(bar_w.end(), writes.begin(), writes.end());
        batch_w.insert(batch_w.end(), writes.begin(), writes.end());
        // A quantized copy of memory this op overwrites is stale.
        done.erase(std::remove_if(done.begin(), done.end(),
                                  [&](const Quantized& q) {
                                      for (const Rng& w : writes)
                                          if (overlaps(q.src, w))
                                              return true;
                                      return false;
                                  }),
                   done.end());
    };
    // Scratch in g.xq; offsets stay 256-byte aligned for root UAV addresses.
    auto reserve = [&](std::size_t bytes, std::size_t* off) -> bool {
        const std::size_t aligned = (bytes + 255) & ~std::size_t(255);
        if (used + aligned > g.xq_bytes && !flush())
            return false;
        if (!ensure_xq(g, used + aligned))
            return false;
        *off = used;
        used += aligned;
        return true;
    };
    auto xq_va = [&](std::size_t off) { return g.xq->GetGPUVirtualAddress() + off; };
    auto xq_rng = [&](std::size_t off, std::size_t bytes) {
        const auto lo = reinterpret_cast<std::uintptr_t>(g.xq_ptr + off);
        return Rng{lo, lo + bytes};
    };
    // ncols f32 columns of x, nb1 apart, as the vec_dot type of weights `wt`
    // in g.xq at *off: a copy an earlier matmul of this submission made, else
    // q8_K on the GPU or the CPU quantizer (q8_0 for Q4_0 weights).
    auto quantize = [&](const ggml_tensor* x, ggml_type wt, int ncols, std::size_t* off) -> bool {
        const ggml_type at = d3d12_activation_type(wt);
        const int k = static_cast<int>(x->ne[0]);
        const std::uint32_t row = static_cast<std::uint32_t>(ggml_row_size(at, k));
        const std::size_t bytes = static_cast<std::size_t>(row) * ncols;
        *off = SIZE_MAX;
        for (const Quantized& q : done)
            if (q.data == x->data && q.nb1 == x->nb[1] && q.type == at && q.row == row &&
                q.ncols >= ncols)
                *off = q.off;
        if (*off != SIZE_MAX)
            return true;
        const Rng xr = rng_of(x);
        const bool gpu_q = g.elt_ok && at == GGML_TYPE_Q8_K && elt_operand(x) &&
                           ncols <= static_cast<int>(kD3d12MaxGroups);
        if (!gpu_q && any_overlap(batch_w, xr) && !flush())
            return false;
        if (!reserve(bytes, off))
            return false;
        if (gpu_q) {
            Op q;
            q.pso = g.pso_elt[kEltQuantQ8K].Get();
            set_c(q, {static_cast<std::uint64_t>(k / kD3d12Chunk), x->nb[1] / sizeof(float), row});
            q.u[0] = xq_va(*off);
            q.u[1] = tensor_va(x);
            q.gx = static_cast<UINT>(k / kD3d12Chunk);
            q.gy = static_cast<UINT>(ncols);
            record(q, {xr}, {xq_rng(*off, bytes)});
        } else {
            d3d12_quantize_activations(wt, static_cast<const float*>(x->data),
                                       x->nb[1] / sizeof(float), k, ncols, g.xq_ptr + *off);
        }
        done.push_back({x->data, x->nb[1], at, *off, row, ncols, xr});
        return true;
    };
    // A MoE layer's tail at nodes[i], a MUL (ggml_d3d12_moe_emu.h): the routed
    // experts' MUL by their weights with the n - 1 ADDs of its views as one
    // combine; else the shared expert's gate with the ADD to the routed experts
    // and the residual ADD after it as one tail; else that MUL alone. Fused
    // nodes must follow it back to back, so nothing runs between them. Returns
    // how many nodes after nodes[i] it consumed, -1 when nodes[i] is none of
    // these, -2 on a failed flush.
    auto moe = [&](std::size_t i) -> int {
        const ggml_tensor* first = nodes[i];
        const std::int64_t n_add = first->ne[1] - 1;
        D3d12MoeCombinePlan cp;
        if (n_add >= 1 && i + static_cast<std::size_t>(n_add) < nodes.size()) {
            const ggml_tensor* last = nodes[i + static_cast<std::size_t>(n_add)];
            bool ok = d3d12_moe_combine_plan(last, &cp) && cp.nodes.front() == first &&
                      cp.params.n == static_cast<std::uint32_t>(n_add) + 1 && elt_operand(last) &&
                      elt_operand(cp.e) && elt_operand(cp.w) &&
                      !overlaps(rng_of(last), rng_of(cp.e)) &&
                      !overlaps(rng_of(last), rng_of(cp.w));
            const std::size_t n = cp.params.n;
            for (std::size_t k = 1; ok && k < n; ++k)
                ok = cp.nodes[n + k] == nodes[i + k]; // A_1 .. A_(n-1)
            if (ok && moe_fusable(cgraph, node_idx[i], cp.nodes, last)) {
                Op c;
                c.pso = g.pso_moe[kMoeCombine].Get();
                static_assert(sizeof(cp.params) <= sizeof(c.c), "8 root constants");
                std::memcpy(c.c, &cp.params, sizeof(cp.params));
                c.u[0] = tensor_va(last);
                c.u[1] = tensor_va(cp.e);
                c.u[2] = tensor_va(cp.w);
                c.gx = cp.groups_x;
                c.gy = cp.groups_y;
                record(c, {rng_of(cp.e), rng_of(cp.w)}, {rng_of(last)});
                return static_cast<int>(n_add);
            }
        }
        for (int extra = 2; extra >= 0; --extra) {
            D3d12MoeTailPlan tp;
            const std::size_t j = i + static_cast<std::size_t>(extra);
            if (j >= nodes.size() || !d3d12_moe_tail_plan(nodes[j], &tp) ||
                tp.nodes.size() != static_cast<std::size_t>(extra) + 1 ||
                tp.nodes.front() != first || (extra == 2 && tp.nodes[1] != nodes[i + 1]))
                continue;
            ggml_tensor* dst = nodes[j];
            const bool ok = elt_operand(dst) && elt_operand(tp.s) && elt_operand(tp.g) &&
                            (!tp.m || elt_operand(tp.m)) && (!tp.q || elt_operand(tp.q)) &&
                            same_or_apart(dst, tp.s) && (!tp.m || same_or_apart(dst, tp.m)) &&
                            (!tp.q || same_or_apart(dst, tp.q)) &&
                            !overlaps(rng_of(dst), rng_of(tp.g));
            if (extra > 0 && !(ok && moe_fusable(cgraph, node_idx[i], tp.nodes, dst)))
                continue;
            if (!ok) {
                // The gate MUL alone on the CPU: the kernel's arithmetic, in place.
                if (!flush())
                    return -2;
                d3d12_moe_tail_emulate(tp.params, tp.groups_x, tp.groups_y,
                                       static_cast<const float*>(tp.s->data),
                                       static_cast<const float*>(tp.g->data), nullptr, nullptr,
                                       static_cast<float*>(dst->data));
                return 0;
            }
            Op t;
            t.pso = g.pso_moe[kMoeTail].Get();
            static_assert(sizeof(tp.params) <= sizeof(t.c), "8 root constants");
            std::memcpy(t.c, &tp.params, sizeof(tp.params));
            t.u[0] = tensor_va(dst);
            t.u[1] = tensor_va(tp.s);
            t.u[2] = tensor_va(tp.g);
            if (tp.m)
                t.u[3] = tensor_va(tp.m);
            if (tp.q)
                t.u[4] = tensor_va(tp.q);
            t.gx = tp.groups_x;
            t.gy = tp.groups_y;
            record(t,
                   {rng_of(tp.s), rng_of(tp.g), rng_of(tp.m ? tp.m : tp.s),
                    rng_of(tp.q ? tp.q : tp.s)},
                   {rng_of(dst)});
            return extra;
        }
        return -1;
    };

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        ggml_tensor* node = nodes[i];
        GGML_ASSERT(node->buffer && is_ours(node->buffer->buft));
        if (ggml_nelements(node) == 0)
            continue; // a ubatch without output rows: nothing to compute
        if (!fused_away.empty() && fused_away.count(node))
            continue; // a delta-net state copy the fused KERNEL 5 does itself
        const ggml_tensor* a = node->src[0];
        const ggml_tensor* b = node->src[1];
        switch (node->op) {
        case GGML_OP_GET_ROWS: {
            GGML_ASSERT(b->buffer && ggml_backend_buffer_is_host(b->buffer));
            // Row ids are tiny: stage them next to the quantized activations.
            const std::size_t bytes = static_cast<std::size_t>(b->ne[0]) * sizeof(std::int32_t);
            std::size_t off = 0;
            if (any_overlap(batch_w, rng_of(b)) && !flush())
                return GGML_STATUS_FAILED;
            if (!a->buffer || a->buffer->buft != &kWeightsBuft) {
                // A recurrent state's rows (build_rs): KERNEL 3, f32.
                D3d12SsmGetRowsPlan p;
                GGML_ASSERT(d3d12_ssm_get_rows_plan(node, &p));
                const auto* in = static_cast<const std::int32_t*>(b->data);
                for (std::int64_t j = 0; j < b->ne[0]; ++j)
                    GGML_ASSERT(in[j] >= 0 && in[j] < a->ne[1]);
                if (!g.ssm_ok || !ssm_operand(a) || !ssm_operand(node) ||
                    overlaps(rng_of(a), rng_of(node))) {
                    if (!flush())
                        return GGML_STATUS_FAILED;
                    d3d12_ssm_get_rows_emulate(p.params, p.groups_x, p.groups_y,
                                               static_cast<const float*>(a->data), in,
                                               static_cast<float*>(node->data));
                    break;
                }
                if (!reserve(bytes, &off))
                    return GGML_STATUS_FAILED;
                std::memcpy(g.xq_ptr + off, b->data, bytes);
                Op op;
                op.pso = g.pso_ssm[kSsmGetRows].Get();
                std::memcpy(op.c, &p.params, sizeof(p.params));
                op.u[0] = tensor_va(node);
                op.u[1] = tensor_va(a);
                op.u[2] = xq_va(off);
                op.gx = p.groups_x;
                op.gy = p.groups_y;
                record(op, {rng_of(a), xq_rng(off, bytes)}, {rng_of(node)});
                break;
            }
            if (!reserve(bytes, &off))
                return GGML_STATUS_FAILED;
            std::memcpy(g.xq_ptr + off, b->data, bytes);
            // The kernel does not bound-check rows; ggml-cpu asserts the same.
            const auto* ids = reinterpret_cast<const std::int32_t*>(g.xq_ptr + off);
            for (std::int64_t j = 0; j < b->ne[0]; ++j)
                GGML_ASSERT(ids[j] >= 0 && ids[j] < a->ne[1]);
            Op op;
            op.pso = g.pso_get_rows.Get();
            set_c(op,
                  {static_cast<std::uint64_t>(node->ne[1]), static_cast<std::uint64_t>(a->ne[0]),
                   static_cast<std::uint64_t>(a->ne[0] / kD3d12Chunk), a->nb[1], 0,
                   node->nb[1] / sizeof(float)});
            op.t0 = tensor_va(a);
            op.u[0] = tensor_va(node);
            op.u[1] = xq_va(off);
            op.gx = static_cast<UINT>(a->ne[0] / kD3d12Chunk);
            op.gy = static_cast<UINT>(node->ne[1]);
            record(op, {xq_rng(off, bytes)}, {rng_of(node)});
            break;
        }
        case GGML_OP_MUL_MAT: {
            const ggml_tensor* w = a;
            const ggml_tensor* x = b;
            GGML_ASSERT(w->buffer && w->buffer->buft == &kWeightsBuft);
            GGML_ASSERT(x->buffer && ggml_backend_buffer_is_host(x->buffer));
            const int ncols = static_cast<int>(x->ne[1]);
            D3D12_GPU_VIRTUAL_ADDRESS x_va = 0;
            std::uint64_t x_stride = 0;
            Rng xr;
            if (w->type == GGML_TYPE_F32 && elt_operand(x)) {
                // The f32 kernel reads the activations where they are: no
                // copy, so no flush when this split's GPU work wrote them.
                x_va = tensor_va(x);
                x_stride = x->nb[1];
                xr = rng_of(x);
            } else {
                const ggml_type at = d3d12_activation_type(w->type);
                const std::uint32_t row = static_cast<std::uint32_t>(ggml_row_size(at, x->ne[0]));
                std::size_t off = 0;
                if (!quantize(x, w->type, ncols, &off))
                    return GGML_STATUS_FAILED;
                x_va = xq_va(off);
                x_stride = row;
                xr = xq_rng(off, static_cast<std::size_t>(row) * ncols);
            }
            const D3d12Dispatch d = d3d12_mm_dispatch(node->ne[0], node->ne[1]);
            const int wide = d3d12_mm_threads(w->ne[0]) == kD3d12MmvThreadsLong ? 1 : 0;
            Op m;
            m.pso = g.pso[pso_for(w->type)][wide].Get();
            set_c(m, {static_cast<std::uint64_t>(w->ne[1]), static_cast<std::uint64_t>(w->ne[0]),
                      static_cast<std::uint64_t>(w->ne[0] / kD3d12Chunk), w->nb[1], x_stride,
                      node->nb[1] / sizeof(float)});
            m.t0 = tensor_va(w);
            m.u[0] = tensor_va(node);
            m.u[1] = x_va;
            m.gx = d.groups_x;
            m.gy = d.groups_y;
            if (w->type == GGML_TYPE_IQ3_S) { // pad0 = the group geometry (fewer, wider groups)
                const D3d12Geom geo =
                    d3d12_geom_for(g.iq3s_geom, static_cast<std::uint32_t>(w->ne[0] / kD3d12Chunk),
                                   static_cast<std::uint32_t>(d3d12_mm_threads(w->ne[0])));
                const std::uint64_t rows = d3d12_geom_rows(geo);
                m.gx =
                    static_cast<UINT>((static_cast<std::uint64_t>(node->ne[0]) + rows - 1) / rows);
                m.c[6] = d3d12_geom_pack(geo);
            }
            record(m, {xr}, {rng_of(node)});
            break;
        }
        case GGML_OP_MUL_MAT_ID: {
            const ggml_tensor* x = b;
            const ggml_tensor* ids = node->src[2];
            D3d12MmidPlan plan;
            GGML_ASSERT(in_expert_cache(a) && d3d12_mmid_plan(node, &plan, g.iq3s_geom));
            GGML_ASSERT(x->buffer && ggml_backend_buffer_is_host(x->buffer));
            GGML_ASSERT(ids->buffer && ggml_backend_buffer_is_host(ids->buffer));
            // Columns nb1 apart across tokens too: the one-dispatch q8_K path.
            const int ncols = static_cast<int>(x->ne[1] * x->ne[2]);
            const bool flat = x->ne[2] == 1 || x->nb[2] == x->ne[1] * x->nb[1];
            const bool gpu_q =
                flat && g.elt_ok && elt_operand(x) && ncols <= static_cast<int>(kD3d12MaxGroups);
            // Ids and X share the scratch: flush first if either must, so the
            // second reservation cannot recycle the first.
            if ((any_overlap(batch_w, rng_of(ids)) ||
                 (!gpu_q && any_overlap(batch_w, rng_of(x)))) &&
                !flush())
                return GGML_STATUS_FAILED;
            const std::size_t need = ((plan.ids_bytes + 255) & ~std::size_t(255)) +
                                     ((plan.x_bytes + 255) & ~std::size_t(255));
            if (used + need > g.xq_bytes && (!flush() || !ensure_xq(g, need)))
                return GGML_STATUS_FAILED;
            // The kernel does not bound-check slot ids; ggml-cpu asserts the same.
            const auto* idp = static_cast<const std::uint8_t*>(ids->data);
            for (std::int64_t t = 0; t < ids->ne[1]; ++t)
                for (std::int64_t j = 0; j < ids->ne[0]; ++j) {
                    std::int32_t id = 0;
                    std::memcpy(&id, idp + t * ids->nb[1] + j * sizeof(std::int32_t), sizeof(id));
                    GGML_ASSERT(id >= 0 && id < a->ne[2]);
                }
            std::size_t ioff = 0, xoff = 0;
            if (!reserve(plan.ids_bytes, &ioff))
                return GGML_STATUS_FAILED;
            std::memcpy(g.xq_ptr + ioff, ids->data, plan.ids_bytes);
            if (flat) {
                if (!quantize(x, a->type, ncols, &xoff))
                    return GGML_STATUS_FAILED;
            } else {
                if (!reserve(plan.x_bytes, &xoff))
                    return GGML_STATUS_FAILED;
                d3d12_mmid_quantize_activations(node, x->data, g.xq_ptr + xoff);
            }
            Op m;
            m.pso =
                g.pso_mmid[mmid_for(a->type)][plan.threads == kD3d12MmvThreadsLong ? 1 : 0].Get();
            static_assert(sizeof(plan.params) == sizeof(m.c), "13 root constants");
            std::memcpy(m.c, &plan.params, sizeof(m.c));
            m.t0 = tensor_va(a);
            m.t1 = xq_va(ioff);
            m.u[0] = tensor_va(node);
            m.u[1] = xq_va(xoff);
            m.gx = plan.groups_x;
            m.gy = plan.groups_y;
            record(m, {xq_rng(xoff, plan.x_bytes), xq_rng(ioff, plan.ids_bytes)}, {rng_of(node)});
            break;
        }
        // The delta net's kernels (ggml_d3d12_ssm_emu.h). supports_op admits
        // only nodes their plans take; an operand the GPU path refuses runs the
        // kernel's arithmetic on the CPU, after the submission so far.
        case GGML_OP_UNARY:
        case GGML_OP_SCALE: {
            D3d12SsmUnaryPlan p;
            GGML_ASSERT(d3d12_ssm_unary_plan(node, &p));
            const ggml_tensor* s = p.src;
            if (!g.ssm_ok || !ssm_operand(node) || !ssm_operand(s) || !ssm_same_or_apart(node, s)) {
                if (!flush())
                    return GGML_STATUS_FAILED;
                d3d12_ssm_unary_emulate(p.params, p.groups_x, p.groups_y,
                                        static_cast<const float*>(s->data),
                                        static_cast<float*>(node->data));
                break;
            }
            Op op;
            op.pso = g.pso_ssm[kSsmUnary].Get();
            std::memcpy(op.c, &p.params, sizeof(p.params));
            op.u[0] = tensor_va(node);
            op.u[1] = tensor_va(s);
            op.gx = p.groups_x;
            op.gy = p.groups_y;
            record(op, {rng_of(s)}, {rng_of(node)});
            break;
        }
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
        case GGML_OP_CONCAT: {
            D3d12SsmCopyPlan p;
            GGML_ASSERT(d3d12_ssm_copy_plan(node, &p));
            bool gpu_ok = g.ssm_ok && ssm_operand(node);
            for (const D3d12SsmCopy& c : p.copies)
                gpu_ok = gpu_ok && (c.groups_x == 0 ||
                                    (ssm_operand(c.src) && !overlaps(rng_of(c.src), rng_of(node))));
            for (const D3d12SsmCopy& c : p.copies) {
                if (c.groups_x == 0)
                    continue;
                if (!gpu_ok) {
                    if (!flush())
                        return GGML_STATUS_FAILED;
                    d3d12_ssm_copy_emulate(
                        c.params, c.groups_x, static_cast<const float*>(c.src->data),
                        reinterpret_cast<float*>(static_cast<char*>(node->data) + c.dst_offset));
                    continue;
                }
                Op op;
                op.pso = g.pso_ssm[kSsmCopy].Get();
                std::memcpy(op.c, &c.params, sizeof(c.params));
                op.u[0] = tensor_va(node) + c.dst_offset;
                op.u[1] = tensor_va(c.src);
                op.gx = c.groups_x;
                op.gy = 1;
                record(op, {rng_of(c.src)}, {copy_rng(node, c)});
            }
            break;
        }
        case GGML_OP_SSM_CONV: {
            // With the SILU after it as one dispatch when nothing else reads
            // the conv and the SILU's memory is not the conv input's.
            ggml_tensor* s = i + 1 < nodes.size() ? nodes[i + 1] : nullptr;
            D3d12SsmConvPlan p;
            const bool fuse =
                s && s->op == GGML_OP_UNARY && s->src[0] == node && d3d12_ssm_conv_plan(s, &p) &&
                ssm_operand(s) && !overlaps(rng_of(s), rng_of(p.x)) &&
                !overlaps(rng_of(s), rng_of(p.w)) && moe_fusable(cgraph, node_idx[i], p.nodes, s);
            if (!fuse)
                GGML_ASSERT(d3d12_ssm_conv_plan(node, &p));
            ggml_tensor* out = fuse ? s : node;
            if (fuse)
                ++i; // the SILU is done
            if (!g.ssm_ok || !ssm_operand(out) || !ssm_operand(p.x) || !ssm_operand(p.w) ||
                overlaps(rng_of(out), rng_of(p.x)) || overlaps(rng_of(out), rng_of(p.w))) {
                if (!flush())
                    return GGML_STATUS_FAILED;
                d3d12_ssm_conv_emulate(p.params, p.groups_x, p.groups_y, p.groups_z,
                                       static_cast<const float*>(p.x->data),
                                       static_cast<const float*>(p.w->data),
                                       static_cast<float*>(out->data));
                break;
            }
            Op op;
            op.pso = g.pso_ssm[kSsmConv].Get();
            std::memcpy(op.c, &p.params, sizeof(p.params));
            op.t0 = tensor_va(p.w);
            op.u[0] = tensor_va(out);
            op.u[1] = tensor_va(p.x);
            op.gx = p.groups_x;
            op.gy = p.groups_y;
            op.gz = p.groups_z;
            record(op, {rng_of(p.x), rng_of(p.w)}, {rng_of(out)});
            break;
        }
        case GGML_OP_GATED_DELTA_NET: {
            D3d12GdnPlan p;
            GGML_ASSERT(d3d12_gdn_plan(node, &p));
            const auto fz = gdn_fused.find(node);
            if (fz != gdn_fused.end()) {
                // gdn_fuse_ok took the operands; p.state was never gathered.
                p.params.k_kda |= kD3d12GdnFused;
                Op op;
                op.pso = g.pso_ssm[kSsmGdn].Get();
                std::memcpy(op.c, &p.params, sizeof(p.params));
                op.u[0] = tensor_va(node);
                op.u[1] = tensor_va(p.q);
                op.u[2] = tensor_va(p.k);
                op.u[3] = tensor_va(p.v);
                op.u[4] = tensor_va(p.g);
                op.u[5] = fz->second.dst_va;
                op.t0 = tensor_va(p.beta);
                op.t1 = fz->second.src_va;
                op.gx = p.groups_x;
                op.gy = p.groups_y;
                op.gz = p.groups_z;
                const auto lo = reinterpret_cast<std::uintptr_t>(node->data);
                const Rng attn{lo, lo + static_cast<std::size_t>(kD3d12GdnSv) * p.params.h *
                                            p.params.t * p.params.b * sizeof(float)};
                record(op,
                       {rng_of(p.q), rng_of(p.k), rng_of(p.v), rng_of(p.g), rng_of(p.beta),
                        fz->second.src},
                       {attn, fz->second.dst});
                break;
            }
            const ggml_tensor* in[6] = {p.q, p.k, p.v, p.g, p.beta, p.state};
            bool gpu_ok = g.ssm_ok && ssm_operand(node);
            for (const ggml_tensor* t : in)
                gpu_ok = gpu_ok && ssm_operand(t) && !overlaps(rng_of(t), rng_of(node));
            if (!gpu_ok) {
                if (!flush())
                    return GGML_STATUS_FAILED;
                auto f = [](const ggml_tensor* t) { return static_cast<const float*>(t->data); };
                d3d12_gdn_emulate(p.params, p.groups_x, p.groups_y, p.groups_z, f(p.q), f(p.k),
                                  f(p.v), f(p.g), f(p.beta), f(p.state),
                                  static_cast<float*>(node->data));
                break;
            }
            Op op;
            op.pso = g.pso_ssm[kSsmGdn].Get();
            std::memcpy(op.c, &p.params, sizeof(p.params));
            op.u[0] = tensor_va(node);
            op.u[1] = tensor_va(p.q);
            op.u[2] = tensor_va(p.k);
            op.u[3] = tensor_va(p.v);
            op.u[4] = tensor_va(p.g);
            op.t0 = tensor_va(p.beta);
            op.t1 = tensor_va(p.state);
            op.gx = p.groups_x;
            op.gy = p.groups_y;
            op.gz = p.groups_z;
            record(op,
                   {rng_of(p.q), rng_of(p.k), rng_of(p.v), rng_of(p.g), rng_of(p.beta),
                    rng_of(p.state)},
                   {rng_of(node)});
            break;
        }
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_GLU: {
            if (node->op == GGML_OP_MUL && g.moe_ok) {
                const int took = moe(i);
                if (took == -2)
                    return GGML_STATUS_FAILED;
                if (took >= 0) {
                    i += static_cast<std::size_t>(took);
                    break;
                }
            }
            const int kind = node->op == GGML_OP_ADD ? 0 : node->op == GGML_OP_MUL ? 1 : 2;
            // supports_op admits these shapes only: b is a's shape or one row.
            GGML_ASSERT(ggml_are_same_shape(node, a) && b->ne[0] == node->ne[0] &&
                        (b->ne[1] == 1 || b->ne[1] == node->ne[1]));
            // SwiGLU rows may be [n_ff, n_expert_used, n_tokens], flattened;
            // the delta net's rows are flat over ne1..ne3, b a's shape or one row.
            const bool glu = kind == 2;
            const bool ssm = g.ssm_ok && !glu && flat_ok(node) && flat_ok(a) &&
                             (ggml_nrows(b) == 1 || (flat_ok(b) && ggml_are_same_shape(node, b)));
            const bool rows_ok = (elt_rows(node) && (b->ne[1] == 1 || b->ne[1] == node->ne[1])) ||
                                 ssm ||
                                 (glu && glu_rows(node) && glu_rows(a) && glu_rows(b) &&
                                  ggml_are_same_shape(node, b));
            const std::int64_t rows = glu ? node->ne[1] * node->ne[2] : flat_rows(node);
            if (g.elt_ok && rows_ok && elt_operand(node) && elt_operand(a) && elt_operand(b) &&
                same_or_apart(node, a) && same_or_apart(node, b)) {
                Op e;
                e.pso = g.pso_elt[kEltBinary].Get();
                set_c(e, {static_cast<std::uint64_t>(node->ne[0]), static_cast<std::uint64_t>(rows),
                          a->nb[1] / sizeof(float),
                          ggml_nrows(b) == 1 && !glu ? 0 : b->nb[1] / sizeof(float),
                          node->nb[1] / sizeof(float), static_cast<std::uint64_t>(kind)});
                e.u[0] = tensor_va(node);
                e.u[1] = tensor_va(a);
                e.u[2] = tensor_va(b);
                e.gx = static_cast<UINT>((node->ne[0] + 255) / 256);
                e.gy = static_cast<UINT>(rows);
                record(e, {rng_of(a), rng_of(b)}, {rng_of(node)});
            } else {
                GGML_ASSERT(ggml_backend_buffer_is_host(a->buffer) &&
                            ggml_backend_buffer_is_host(b->buffer));
                if (!flush())
                    return GGML_STATUS_FAILED;
                cpu_binary(node, kind);
            }
            break;
        }
        case GGML_OP_RMS_NORM: {
            float eps = 0.0f;
            std::memcpy(&eps, node->op_params, sizeof(float));
            std::uint32_t eps_bits = 0;
            std::memcpy(&eps_bits, &eps, sizeof(eps_bits));
            // build_norm's MUL by the norm weight right after it is fused: the
            // kernel writes the norm, then the product.
            ggml_tensor* m = i + 1 < nodes.size() ? nodes[i + 1] : nullptr;
            const ggml_tensor* w = m ? m->src[1] : nullptr;
            const bool fuse = m && m->op == GGML_OP_MUL && m->src[0] == node && w &&
                              elt_operand(m) && elt_operand(w) && ggml_are_same_shape(m, node) &&
                              flat_ok(m) && ggml_nrows(w) == 1 && w->ne[0] == node->ne[0] &&
                              same_or_apart(m, node) && same_or_apart(m, a) &&
                              !overlaps(rng_of(m), rng_of(w));
            // The delta net's: q and k by head ([128, 16] views of the conv
            // output), its output by head before the gated norm.
            const bool ssm = g.ssm_ok && flat_ok(node) && flat_ok(a);
            if (g.elt_ok && (elt_rows(node) || ssm) && elt_operand(node) && elt_operand(a) &&
                same_or_apart(node, a)) {
                Op e;
                e.pso = g.pso_elt[kEltRmsNorm].Get();
                set_c(e, {static_cast<std::uint64_t>(node->ne[0]), a->nb[1] / sizeof(float),
                          node->nb[1] / sizeof(float), eps_bits, fuse ? 1u : 0u,
                          fuse ? m->nb[1] / sizeof(float) : 0u});
                e.u[0] = tensor_va(node);
                e.u[1] = tensor_va(a);
                e.gx = static_cast<UINT>(flat_rows(node));
                e.gy = 1;
                if (fuse) {
                    e.u[2] = tensor_va(w);
                    e.u[3] = tensor_va(m);
                    record(e, {rng_of(a), rng_of(w)}, {rng_of(node), rng_of(m)});
                    ++i; // the MUL is done
                } else {
                    record(e, {rng_of(a)}, {rng_of(node)});
                }
            } else {
                GGML_ASSERT(ggml_backend_buffer_is_host(a->buffer));
                if (!flush())
                    return GGML_STATUS_FAILED;
                cpu_rms_norm(node, eps);
            }
            break;
        }
        default:
            GGML_ABORT("d3d12: unsupported op %s", ggml_op_desc(node));
        }
    }
    if (!flush())
        return GGML_STATUS_FAILED;
    ++g.n_calls;
    g.gpu_ms += g.last_gpu_ms;
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    g.wall_ms += ms;
    Gpu::SegStat& seg = g.segs[seg_name(nodes.front()) + " .. " + seg_name(nodes.back()) + " /" +
                               std::to_string(nodes.size())];
    ++seg.n;
    seg.ms += ms;
    return GGML_STATUS_SUCCESS;
}

ggml_guid_t backend_guid() {
    static ggml_guid guid = {0x78, 0x6c, 0x6c, 0x61, 0x6d, 0x61, 0x2d, 0x64,
                             0x33, 0x64, 0x31, 0x32, 0x2d, 0x76, 0x30, 0x31};
    return &guid;
}

const ggml_backend_i kBackendIface = {
    /* .get_name            = */ backend_name,
    /* .free                = */ backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr, // graph_compute completes before returning
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

// --- Device ---

const char* dev_name(ggml_backend_dev_t) {
    return "D3D12";
}
const char* dev_description(ggml_backend_dev_t) {
    return "xllama D3D12 compute (Q4_0/Q4_K/Q5_K/Q6_K/IQ3_S/IQ4_XS matmul, Q6_K get_rows, f32 "
           "add/mul/rms_norm/swiglu, q8_K quant)";
}
void dev_memory(ggml_backend_dev_t, size_t* free, size_t* total) {
    *free = *total = 0;
    Gpu& g = gpu();
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter3> adapter;
    if (!g.ok || FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumAdapterByLuid(g.luid, IID_PPV_ARGS(&adapter))))
        return;
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        *total = static_cast<size_t>(info.Budget);
        *free = info.Budget > info.CurrentUsage
                    ? static_cast<size_t>(info.Budget - info.CurrentUsage)
                    : 0;
    }
}
enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}
void dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props* props) {
    props->name = dev_name(dev);
    props->description = dev_description(dev);
    props->type = dev_type(dev);
    props->device_id = nullptr;
    dev_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {/* async */ false, /* host_buffer */ false, /* buffer_from_host_ptr */ false,
                   /* events */ false, /* mmap_support */ false};
}
ggml_backend_t dev_init_backend(ggml_backend_dev_t dev, const char*) {
    Gpu& g = gpu();
    if (!g.ok)
        return nullptr;
    {
        // Read per backend: a catalogue entry's "env" is applied before its load.
        std::lock_guard<std::mutex> lock(g.mu);
        const char* s = std::getenv("XLLAMA_D3D12_PROFILE");
        g.prof_on = s && *s && *s != '0' && g.ts_freq > 0;
        if (g.prof_on && !g.prof) {
            D3D12_QUERY_HEAP_DESC qhd = {};
            qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
            qhd.Count = kProfStamps;
            std::string err;
            if (SUCCEEDED(g.device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&g.prof))))
                g.prof_rb = d3d12c::create_buffer(
                    g.device.Get(), kProfStamps * sizeof(std::uint64_t), D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, "prof_rb", &err);
            if (!g.prof || !g.prof_rb) {
                g.prof.Reset();
                g.prof_rb.Reset();
                g.prof_on = false;
            }
        }
        if (g.prof_on)
            log_output("[xllama] d3d12: profiling every dispatch (XLLAMA_D3D12_PROFILE)\n");
        const char* f = std::getenv("XLLAMA_D3D12_SSM_FUSE");
        g.gdn_fuse = !(f && *f == '0');
        if (!g.gdn_fuse)
            log_output("[xllama] d3d12: delta-net state copies kept (XLLAMA_D3D12_SSM_FUSE=0)\n");
        g.iq3s_geom.rsl = d3d12_geom_parse(std::getenv("XLLAMA_D3D12_IQ3S_RSL"), 1);
        g.iq3s_geom.rt = d3d12_geom_parse(std::getenv("XLLAMA_D3D12_IQ3S_RT"), 1);
        if (g.iq3s_geom.rsl != 1 || g.iq3s_geom.rt != 1)
            log_output("[xllama] d3d12: IQ3_S geometry rsl " +
                       (g.iq3s_geom.rsl ? std::to_string(g.iq3s_geom.rsl) : std::string("auto")) +
                       " rt " + std::to_string(g.iq3s_geom.rt) + " (XLLAMA_D3D12_IQ3S_RSL/_RT)\n");
        const int replaced = apply_shader_dir(g, std::getenv("XLLAMA_D3D12_SHADER_DIR"));
        if (replaced > 0)
            log_output("[xllama] d3d12: " + std::to_string(replaced) +
                       " kernel(s) from XLLAMA_D3D12_SHADER_DIR\n");
    }
    return new ggml_backend{backend_guid(), kBackendIface, dev, nullptr};
}
ggml_backend_buffer_type_t dev_buffer_type(ggml_backend_dev_t) {
    return &kHostBuft;
}

// An f32 operand of the elementwise kernels as supports_op sees it: activations
// have no buffer yet; anything already placed must be in D3D12_Host.
bool elt_f32(const ggml_tensor* t) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->nb[1] % sizeof(float) == 0 &&
           (!t->buffer || t->buffer->buft == &kHostBuft);
}

// The output of a D3D12 matmul: a weight in D3D12_Weights, or the routed
// experts' MUL_MAT_ID over the expert cache.
bool d3d12_matmul_out(const ggml_tensor* t) {
    t = base_of(t);
    const ggml_tensor* w = t->src[0];
    return (t->op == GGML_OP_MUL_MAT && w && w->buffer && w->buffer->buft == &kWeightsBuft) ||
           (t->op == GGML_OP_MUL_MAT_ID && w && in_expert_cache(w));
}

// G, O or P of a MoE layer's tail (ggml_d3d12_moe_emu.h) — the routed experts'
// MUL by their weights is a G too — whose gated operand comes straight from a
// D3D12 matmul. Gates fed by CPU ops (the delta net's) stay on the CPU.
bool moe_tail_node(const ggml_tensor* op) {
    D3d12MoeTailPlan p;
    return gpu().moe_ok && d3d12_moe_tail_plan(op, &p) && d3d12_matmul_out(p.s) && elt_f32(op) &&
           elt_f32(p.s) && elt_f32(p.g) && (!p.m || elt_f32(p.m)) && (!p.q || elt_f32(p.q));
}

// Whether t is computed from a D3D12 matmul or get_rows within `depth`
// elementwise ops. Only those elementwise ops move to the GPU: the ones fed by
// CPU ops (the embedding lookup, the MoE router, the last layer's row
// selection) stay with them instead of adding a CPU -> GPU -> CPU round trip.
// A MoE tail counts as a matmul: the next layer's norm joins it.
bool fed_by_d3d12(const ggml_tensor* t, int depth = 2) {
    while (t) {
        t = base_of(t);
        if ((t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_GET_ROWS) && t->src[0] &&
            t->src[0]->buffer && t->src[0]->buffer->buft == &kWeightsBuft)
            return true;
        if ((t->op == GGML_OP_ADD || t->op == GGML_OP_MUL) && moe_tail_node(t))
            return true;
        if (depth-- <= 0)
            return false;
        if (t->op != GGML_OP_ADD && t->op != GGML_OP_MUL && t->op != GGML_OP_RMS_NORM &&
            t->op != GGML_OP_GLU)
            return false;
        t = t->src[0];
    }
    return false;
}

bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor* op) {
    switch (op->op) {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        return true;
    case GGML_OP_MUL_MAT: {
        const ggml_tensor* w = op->src[0];
        const ggml_tensor* x = op->src[1];
        D3d12MatmulDesc d;
        d.src0_type = w->type;
        d.ne00 = w->ne[0];
        d.ne01 = w->ne[1];
        d.ne02 = w->ne[2];
        d.ne03 = w->ne[3];
        d.ne10 = x->ne[0];
        d.ne11 = x->ne[1];
        d.ne12 = x->ne[2];
        d.ne13 = x->ne[3];
        d.src1_type = x->type;
        d.dst_type = op->type;
        d.src0_contiguous = ggml_is_contiguous(w);
        d.src1_contiguous = ggml_is_contiguous(x);
        d.src0_in_weight_buffer = w->buffer && w->buffer->buft == &kWeightsBuft;
        const bool ok = d3d12_mm_supported(d);
        // A refused matmul whose weight already sits in D3D12_Weights means a
        // placement the backend then cannot run — log the first few (D2b #309).
        static int logged = 0;
        if (!ok && w->buffer && w->buffer->buft == &kWeightsBuft && logged < 8) {
            ++logged;
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "[xllama] d3d12: MUL_MAT refused: %s %lldx%lld x [%lld,%lld,%lld,%lld] "
                          "src1=%s dst=%s\n",
                          ggml_type_name(w->type), static_cast<long long>(w->ne[0]),
                          static_cast<long long>(w->ne[1]), static_cast<long long>(x->ne[0]),
                          static_cast<long long>(x->ne[1]), static_cast<long long>(x->ne[2]),
                          static_cast<long long>(x->ne[3]), ggml_type_name(x->type),
                          ggml_type_name(op->type));
            log_output(msg);
        }
        return ok;
    }
    case GGML_OP_GET_ROWS: {
        const ggml_tensor* w = op->src[0];
        const ggml_tensor* ids = op->src[1];
        D3d12GetRowsDesc d;
        d.src0_type = w->type;
        d.ne00 = w->ne[0];
        d.ne01 = w->ne[1];
        d.ne02 = w->ne[2];
        d.ne03 = w->ne[3];
        d.src1_type = ids->type;
        d.ne10 = ids->ne[0];
        d.ne11 = ids->ne[1];
        d.ne12 = ids->ne[2];
        d.dst_type = op->type;
        d.src0_contiguous = ggml_is_contiguous(w);
        d.src1_contiguous = ggml_is_contiguous(ids);
        d.src0_in_weight_buffer = w->buffer && w->buffer->buft == &kWeightsBuft;
        if (d3d12_get_rows_supported(d))
            return true;
        // A recurrent state's rows (build_rs), f32 in D3D12_Host.
        D3d12SsmGetRowsPlan p;
        return gpu().ssm_ok && !d.src0_in_weight_buffer && ssm_f32(op) && ssm_f32(w) &&
               d3d12_ssm_get_rows_plan(op, &p) && ssm_chain(op);
    }
    case GGML_OP_UNARY:
    case GGML_OP_SCALE: {
        D3d12SsmUnaryPlan p;
        return gpu().ssm_ok && ssm_f32(op) && ssm_f32(op->src[0]) && d3d12_ssm_unary_plan(op, &p) &&
               ssm_chain(op);
    }
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_CONCAT: {
        D3d12SsmCopyPlan p;
        if (!gpu().ssm_ok || !ssm_f32(op) || !d3d12_ssm_copy_plan(op, &p) || !ssm_chain(op))
            return false;
        for (const D3d12SsmCopy& c : p.copies)
            if (!ssm_f32(c.src))
                return false;
        return true;
    }
    case GGML_OP_SSM_CONV: {
        D3d12SsmConvPlan p;
        return gpu().ssm_ok && ssm_f32(op) && d3d12_ssm_conv_plan(op, &p) && ssm_f32(p.x) &&
               ssm_f32(p.w) && ssm_chain(op);
    }
    case GGML_OP_GATED_DELTA_NET: {
        D3d12GdnPlan p;
        if (!gpu().ssm_ok || !ssm_f32(op) || !d3d12_gdn_plan(op, &p) || !ssm_chain(op))
            return false;
        for (const ggml_tensor* t : {p.q, p.k, p.v, p.g, p.beta, p.state})
            if (!ssm_f32(t))
                return false;
        return true;
    }
    case GGML_OP_MUL_MAT_ID: {
        // The routed experts of a MoE layer, read in place from the expert
        // cache's slots; activations and ids reach D3D12_Host as split inputs.
        D3d12MmidPlan plan;
        const ggml_tensor* x = op->src[1];
        const ggml_tensor* ids = op->src[2];
        const int mk = mmid_for(op->src[0]->type);
        return mk >= 0 && gpu().pso_mmid[mk][0] && in_expert_cache(op->src[0]) &&
               d3d12_mmid_plan(op, &plan) &&
               static_cast<std::uint64_t>(x->ne[1]) * x->ne[2] <= kD3d12MaxGroups &&
               (!x->buffer || ggml_backend_buffer_is_host(x->buffer)) &&
               (!ids->buffer || ggml_backend_buffer_is_host(ids->buffer));
    }
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_GLU:
    case GGML_OP_RMS_NORM: {
        const ggml_tensor* a = op->src[0];
        const ggml_tensor* b = op->src[1];
        if ((op->op == GGML_OP_MUL || op->op == GGML_OP_ADD) && moe_tail_node(op))
            return true;
        if (op->op == GGML_OP_ADD && a && b && b->op == GGML_OP_VIEW && b->view_src &&
            b->view_src->op == GGML_OP_MUL && moe_tail_node(b->view_src))
            // A_1 .. A_(n-1), the sum over the routed experts' weighted outputs:
            // one combine, or binary kernels over the views when it cannot fuse.
            return gpu().elt_ok && elt_rows(op) && elt_f32(op) && elt_f32(a) && elt_f32(b) &&
                   ggml_are_same_shape(op, a) && ggml_are_same_shape(op, b);
        // The delta net's: rows of any width, flat over ne1..ne3; b a's shape
        // or one row (ssm_dt, ssm_a, the gated norm's weight).
        if (op->op != GGML_OP_GLU && a && gpu().ssm_ok && flat_ok(op) && flat_ok(a) &&
            ggml_are_same_shape(op, a) && elt_f32(op) && ssm_f32(op) && elt_f32(a) && ssm_f32(a) &&
            (op->op == GGML_OP_RMS_NORM ||
             (b && elt_f32(b) && ssm_f32(b) && b->ne[0] == op->ne[0] &&
              (ggml_nrows(b) == 1 || (flat_ok(b) && ggml_are_same_shape(op, b))))) &&
            ssm_chain(op))
            return true;
        if (op->op == GGML_OP_GLU && a && gpu().elt_ok && glu_rows(op) && !elt_rows(op)) {
            // A SwiGLU narrower than kD3d12EltMinRow: the routed experts' between two
            // MUL_MAT_IDs, or a shared expert's (n_ff_shexp 512 on Qwen3.6-35B-A3B)
            // between its D3D12 matmuls, which kept the shared FFN in two submissions.
            const ggml_tensor* src = base_of(a);
            return ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && b && elt_f32(op) && elt_f32(a) &&
                   elt_f32(b) && op->op_params[1] == 0 && ggml_are_same_shape(op, a) &&
                   ggml_are_same_shape(op, b) &&
                   ((src->op == GGML_OP_MUL_MAT_ID && in_expert_cache(src->src[0])) ||
                    fed_by_d3d12(a, 0));
        }
        if (!gpu().elt_ok || !elt_f32(op) || !elt_f32(a) || !elt_rows(op) ||
            !ggml_are_same_shape(op, a))
            return false;
        switch (op->op) {
        case GGML_OP_ADD: // the residual adds after o_proj and ffn_down
            return b && elt_f32(b) && ggml_are_same_shape(op, b) && fed_by_d3d12(a);
        case GGML_OP_MUL:
            // A norm by its weight. The weight is checked by name and buffer,
            // not by what feeds src0: the loader probes it with a dummy src0,
            // and a norm weight in D3D12_Host lets the norm join the matmuls.
            return b && b->buffer && elt_f32(b) && d3d12_norm_weight_name(b->name) &&
                   ggml_nrows(b) == 1 && b->ne[0] == op->ne[0];
        case GGML_OP_GLU:
            return ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && b && elt_f32(b) &&
                   op->op_params[1] == 0 /* not swapped */ && ggml_are_same_shape(op, b) &&
                   fed_by_d3d12(a);
        default: // RMS_NORM
            return fed_by_d3d12(a);
        }
    }
    default:
        return false;
    }
}
bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return is_ours(buft) || (is_expert_cache(buft) && gpu().pso_mmid[kMmidIq4Xs][0]);
}

const ggml_backend_device_i kDeviceIface = {
    /* .get_name             = */ dev_name,
    /* .get_description      = */ dev_description,
    /* .get_memory           = */ dev_memory,
    /* .get_type             = */ dev_type,
    /* .get_props            = */ dev_props,
    /* .init_backend         = */ dev_init_backend,
    /* .get_buffer_type      = */ dev_buffer_type,
    // The default buft doubles as the host buft: llama.cpp then allocates the
    // CPU backend's compute buffer in D3D12_Host, which supports_buft accepts,
    // so the scheduler stops copying every matmul input into a second buffer.
    /* .get_host_buffer_type = */ dev_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// --- Registry ---

const char* reg_name(ggml_backend_reg_t) {
    return "D3D12";
}
size_t reg_device_count(ggml_backend_reg_t) {
    return 1;
}
ggml_backend_dev_t reg_device(ggml_backend_reg_t, size_t index) {
    GGML_ASSERT(index == 0);
    return device_ptr();
}

// Weights live in the extra buft; llama.cpp lists it after the default one.
ggml_backend_buffer_type_t* dev_extra_bufts(ggml_backend_dev_t) {
    static ggml_backend_buffer_type_t list[] = {&kWeightsBuft, nullptr};
    return list;
}

void* reg_proc_address(ggml_backend_reg_t, const char* name) {
    if (std::strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0)
        return reinterpret_cast<void*>(dev_extra_bufts);
    return nullptr;
}

const ggml_backend_reg_i kRegIface = {
    /* .get_name         = */ reg_name,
    /* .get_device_count = */ reg_device_count,
    /* .get_device       = */ reg_device,
    /* .get_proc_address = */ reg_proc_address,
};

ggml_backend_reg kReg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ kRegIface,
    /* .context     = */ nullptr,
};

ggml_backend_device kDevice = {
    /* .iface   = */ kDeviceIface,
    /* .reg     = */ &kReg,
    /* .context = */ nullptr,
};

ggml_backend_dev_t device_ptr() {
    return &kDevice;
}

} // namespace

bool ggml_d3d12_register() {
    static std::once_flag once;
    static bool registered = false;
    std::call_once(once, [] {
        Gpu& g = gpu();
        if (!g.ok) {
            log_output(("[xllama] d3d12: backend unavailable: " + g.error + "\n").c_str());
            return;
        }
        kHostBuft.device = &kDevice;
        kWeightsBuft.device = &kDevice;
        ggml_backend_register(&kReg);
        registered = true;
    });
    return registered;
}

ggml_backend_buffer_type_t ggml_d3d12_weights_buft() {
    return ggml_d3d12_register() ? &kWeightsBuft : nullptr;
}

void ggml_d3d12_set_weights_max_buffer(std::size_t bytes) {
    g_weights_max_buffer = bytes;
}

// --- Memory probe (gpumem.flag) ---

namespace {

constexpr std::size_t kProbeStep = 64ull << 20;
constexpr std::size_t kProbeHeapFloorMb =
    256; // same margin as ramceil: stop before the PLM kills us

std::string probe_state(Gpu& g) {
    char b[256];
    std::size_t budget = 0, usage = 0, nl_budget = 0, nl_usage = 0;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter3> adapter;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) &&
        SUCCEEDED(factory->EnumAdapterByLuid(g.luid, IID_PPV_ARGS(&adapter)))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO i = {};
        if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &i))) {
            budget = static_cast<std::size_t>(i.Budget >> 20);
            usage = static_cast<std::size_t>(i.CurrentUsage >> 20);
        }
        if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &i))) {
            nl_budget = static_cast<std::size_t>(i.Budget >> 20);
            nl_usage = static_cast<std::size_t>(i.CurrentUsage >> 20);
        }
    }
    std::snprintf(b, sizeof(b),
                  "dxgi_local %zu/%zu MB, dxgi_nonlocal %zu/%zu MB, avail_phys %zu MB, ws %zu MB",
                  usage, budget, nl_usage, nl_budget, avail_phys_mb(), working_set_mb());
    return b;
}

struct ProbeHold {
    std::vector<ComPtr<ID3D12Resource>> res;
    std::vector<ComPtr<ID3D12Heap>> heaps;
    std::vector<void*> heap_blocks;
    std::vector<void*> va_blocks; // under heaps opened from their address: freed after them
    ~ProbeHold() {
        res.clear();
        heaps.clear();
        for (void* p : heap_blocks)
            std::free(p);
        for (void* p : va_blocks)
            VirtualFree(p, 0, MEM_RELEASE);
    }
};

volatile std::uint8_t g_probe_sink = 0;

// A GPU buffer over CPU memory: committed, touched pages (VirtualAllocFromApp),
// a heap opened on them (ID3D12Device3::OpenExistingHeapFromAddress) and a
// cross-adapter buffer placed at its start. Series X Dev Mode (2026-11-10):
// committed resources of every heap kind stop at the 4147 MB DXGI budget while
// plain heap memory still grows by ~700 MB on top; whether memory reached this
// way counts against that budget is what the probe measures.
ComPtr<ID3D12Resource> existing_buffer(Gpu& g, std::size_t bytes, void** mem,
                                       ComPtr<ID3D12Heap>* heap_out, std::string* err) {
    ComPtr<ID3D12Device3> d3;
    if (FAILED(g.device.As(&d3))) {
        *err = "no ID3D12Device3";
        return {};
    }
    void* p = VirtualAllocFromApp(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) {
        *err = "VirtualAllocFromApp error " + std::to_string(GetLastError());
        return {};
    }
    auto* page = static_cast<std::uint8_t*>(p);
    for (std::size_t off = 0; off < bytes; off += 4096)
        page[off] = static_cast<std::uint8_t>(off >> 12);
    ComPtr<ID3D12Heap> heap;
    HRESULT hr = d3->OpenExistingHeapFromAddress(p, IID_PPV_ARGS(&heap));
    ComPtr<ID3D12Resource> res;
    if (SUCCEEDED(hr)) {
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags =
            D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        hr = g.device->CreatePlacedResource(heap.Get(), 0, &rd, D3D12_RESOURCE_STATE_COMMON,
                                            nullptr, IID_PPV_ARGS(&res));
        if (FAILED(hr))
            *err = d3d12c::hr_message("CreatePlacedResource (existing heap)", hr);
    } else {
        *err = d3d12c::hr_message("OpenExistingHeapFromAddress", hr);
    }
    if (!res) {
        heap.Reset();
        VirtualFree(p, 0, MEM_RELEASE);
        return {};
    }
    *mem = p;
    *heap_out = heap;
    return res;
}

// One 64 MiB existing-heap buffer: a CPU pattern read by a GPU copy, a GPU copy
// read by the CPU, then the GPU time of one whole-buffer copy into a DEFAULT
// buffer from it, from a write-back and from a write-combined CUSTOM L0 buffer.
void probe_existing_rw(Gpu& g, const std::function<void(const std::string&)>& emit) {
    constexpr std::size_t kMiB = 1u << 20;
    const std::size_t bytes = kProbeStep;
    void* mem = nullptr;
    ComPtr<ID3D12Heap> heap;
    std::string err;
    auto res = existing_buffer(g, bytes, &mem, &heap, &err);
    if (!res) {
        emit("existing rw: FAIL " + err + "; " + probe_state(g));
        return;
    }
    auto* m = static_cast<std::uint8_t*>(mem);
    for (std::size_t i = 0; i < bytes; ++i)
        m[i] = static_cast<std::uint8_t>(i * 13 + 5);
    auto dst =
        d3d12c::create_buffer(g.device.Get(), bytes, D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, "probe", &err);
    D3D12_HEAP_PROPERTIES wc = custom_l0_props();
    wc.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
    auto src_wb = d3d12c::create_buffer_props(g.device.Get(), bytes, custom_l0_props(),
                                              D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON,
                                              "probe", &err);
    auto src_wc = d3d12c::create_buffer_props(g.device.Get(), bytes, wc, D3D12_RESOURCE_FLAG_NONE,
                                              D3D12_RESOURCE_STATE_COMMON, "probe", &err);
    bool rd_ok = false, wr_ok = false;
    double ms_ex = -1, ms_wb = -1, ms_wc = -1;
    {
        std::lock_guard<std::mutex> lock(g.mu);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(g.readback.Get(), 0, res.Get(), bytes - kMiB, kMiB);
        });
        rd_ok = std::memcmp(g.readback_ptr, m + bytes - kMiB, kMiB) == 0;
        for (std::size_t i = 0; i < kMiB; ++i)
            g.staging_ptr[i] = static_cast<std::uint8_t>(i * 7 + 3);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(res.Get(), 0, g.staging.Get(), 0, kMiB);
        });
        wr_ok = std::memcmp(m, g.staging_ptr, kMiB) == 0;
        auto time_copy = [&](ID3D12Resource* src, double* ms) {
            if (!dst || !src)
                return;
            for (int rep = 0; rep < 2; ++rep) { // the first run pays residency
                double w = 0;
                run_now(
                    g,
                    [&](ID3D12GraphicsCommandList* cl) {
                        cl->CopyBufferRegion(dst.Get(), 0, src, 0, bytes);
                    },
                    nullptr, &w);
                *ms = w;
            }
        };
        time_copy(res.Get(), &ms_ex);
        time_copy(src_wb.Get(), &ms_wb);
        time_copy(src_wc.Get(), &ms_wc);
    }
    char b[256];
    std::snprintf(b, sizeof(b),
                  "existing rw: cpu->gpu %s, gpu->cpu %s; 64 MiB copy ms existing %.2f "
                  "custom_wb %.2f custom_wc %.2f; ",
                  rd_ok ? "MATCH" : "MISMATCH", wr_ok ? "MATCH" : "MISMATCH", ms_ex, ms_wb, ms_wc);
    emit(std::string(b) + probe_state(g));
    res.Reset();
    heap.Reset();
    VirtualFree(mem, 0, MEM_RELEASE);
}

// Adds kProbeStep units of one kind until an allocation fails, the cap is hit
// or (heap) avail_phys drops under the floor. Returns MiB added.
std::size_t probe_fill(Gpu& g, const char* kind, std::size_t cap_mb, ProbeHold* hold,
                       const std::function<void(const std::string&)>& emit) {
    std::size_t mb = 0;
    std::string stop = "cap";
    while (mb + (kProbeStep >> 20) <= cap_mb) {
        std::string err;
        bool ok = false;
        if (std::strcmp(kind, "heap") == 0) {
            if (avail_phys_mb() < kProbeHeapFloorMb + (kProbeStep >> 20)) {
                stop = "avail_floor";
                break;
            }
            void* p = std::malloc(kProbeStep);
            if (p) {
                auto* bytes = static_cast<std::uint8_t*>(p);
                for (std::size_t off = 0; off < kProbeStep; off += 4096)
                    bytes[off] = static_cast<std::uint8_t>(off >> 12);
                g_probe_sink = g_probe_sink + bytes[kProbeStep - 4096];
                hold->heap_blocks.push_back(p);
                ok = true;
            } else {
                err = "malloc null";
            }
        } else if (std::strcmp(kind, "existing") == 0) {
            if (avail_phys_mb() < kProbeHeapFloorMb + (kProbeStep >> 20)) {
                stop = "avail_floor";
                break;
            }
            void* mem = nullptr;
            ComPtr<ID3D12Heap> heap;
            auto r = existing_buffer(g, kProbeStep, &mem, &heap, &err);
            if (r) {
                hold->res.push_back(r);
                hold->heaps.push_back(heap);
                hold->va_blocks.push_back(mem);
                ok = true;
            }
        } else if (std::strcmp(kind, "default") == 0) {
            auto r = d3d12c::create_buffer(g.device.Get(), kProbeStep, D3D12_HEAP_TYPE_DEFAULT,
                                           D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON,
                                           "probe", &err);
            if (r) {
                hold->res.push_back(r);
                ok = true;
            }
        } else if (std::strcmp(kind, "custom_l0") == 0) {
            auto r = d3d12c::create_buffer_props(g.device.Get(), kProbeStep, custom_l0_props(),
                                                 D3D12_RESOURCE_FLAG_NONE,
                                                 D3D12_RESOURCE_STATE_COMMON, "probe", &err);
            if (r) {
                hold->res.push_back(r);
                ok = true;
            }
        } else { // "heapobj": ID3D12Heap DEFAULT, buffers only
            HRESULT hr = S_OK;
            auto h = create_heap(g.device.Get(), kProbeStep, false, &hr);
            if (h) {
                hold->heaps.push_back(h);
                ok = true;
            } else {
                err = d3d12c::hr_message("CreateHeap", hr);
            }
        }
        if (!ok) {
            stop = err;
            break;
        }
        mb += kProbeStep >> 20;
        if (mb % 1024 == 0)
            emit(std::string("  fill ") + kind + " " + std::to_string(mb) +
                 " MB: " + probe_state(g));
    }
    emit(std::string("fill ") + kind + " total " + std::to_string(mb) + " MB (stop: " + stop +
         "); " + probe_state(g));
    return mb;
}

// One reserved buffer of `mb`, every 64 MiB tile range on its own heap; then a
// 1 MiB pattern written through the staging ring at its last MiB and read back.
void probe_tiled(Gpu& g, std::size_t mb, const std::function<void(const std::string&)>& emit) {
    std::vector<ComPtr<ID3D12Heap>> heaps;
    std::string err;
    const UINT64 bytes = static_cast<UINT64>(mb) << 20;
    auto res = create_tiled_buffer(g, bytes, &heaps, &err);
    if (!res) {
        emit("tiled " + std::to_string(mb) + " MB: FAIL " + err + "; " + probe_state(g));
        return;
    }
    const UINT64 off = bytes - (1ull << 20);
    bool same = false;
    {
        std::lock_guard<std::mutex> lock(g.mu);
        for (std::size_t i = 0; i < (1u << 20); ++i)
            g.staging_ptr[i] = static_cast<std::uint8_t>(i * 7 + 3);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(res.Get(), off, g.staging.Get(), 0, 1ull << 20);
        });
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(g.readback.Get(), 0, res.Get(), off, 1ull << 20);
        });
        same = std::memcmp(g.staging_ptr, g.readback_ptr, 1u << 20) == 0;
    }
    emit("tiled " + std::to_string(mb) + " MB: ok, " + std::to_string(heaps.size()) +
         " heaps, readback " + (same ? "MATCH" : "MISMATCH") + "; " + probe_state(g));
}

} // namespace

void run_d3d12_memprobe(const std::function<void(const std::string&)>& emit) {
    if (!emit)
        return;
    Gpu& g = gpu();
    if (!g.ok) {
        emit("d3d12 init failed: " + g.error);
        return;
    }
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS o = {};
        D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT va = {};
        D3D12_FEATURE_DATA_ARCHITECTURE1 arch = {};
        g.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof(o));
        g.device->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &va, sizeof(va));
        g.device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &arch, sizeof(arch));
        char b[256];
        std::snprintf(
            b, sizeof(b),
            "caps: uma=%d cc_uma=%d tiled_tier=%d heap_tier=%d va_bits_res=%u va_bits_proc=%u",
            arch.UMA ? 1 : 0, arch.CacheCoherentUMA ? 1 : 0, static_cast<int>(o.TiledResourcesTier),
            static_cast<int>(o.ResourceHeapTier), va.MaxGPUVirtualAddressBitsPerResource,
            va.MaxGPUVirtualAddressBitsPerProcess);
        emit(b);
        emit("start: " + probe_state(g));
    }
    // A: one allocation of each size, released at once.
    static const std::size_t sizes_mb[] = {64,  128, 256, 384, 512,  640,  704,
                                           768, 800, 832, 896, 1024, 1536, 2048};
    for (const char* kind : {"default", "custom_l0", "heapobj"}) {
        std::string line = std::string("single ") + kind + ":";
        for (std::size_t mb : sizes_mb) {
            std::string err;
            bool ok = false;
            const UINT64 bytes = static_cast<UINT64>(mb) << 20;
            if (std::strcmp(kind, "default") == 0) {
                ok = d3d12c::create_buffer(g.device.Get(), bytes, D3D12_HEAP_TYPE_DEFAULT,
                                           D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON,
                                           "probe", &err) != nullptr;
            } else if (std::strcmp(kind, "custom_l0") == 0) {
                ok = d3d12c::create_buffer_props(
                         g.device.Get(), bytes, custom_l0_props(), D3D12_RESOURCE_FLAG_NONE,
                         D3D12_RESOURCE_STATE_COMMON, "probe", &err) != nullptr;
            } else {
                ok = create_heap(g.device.Get(), bytes, false, nullptr) != nullptr;
            }
            line += " " + std::to_string(mb) + (ok ? "=ok" : "=FAIL");
        }
        emit(line);
    }
    // B: GPU first, then heap on top.
    {
        ProbeHold hold;
        probe_fill(g, "default", 12288, &hold, emit);
        probe_fill(g, "custom_l0", 12288, &hold, emit);
        probe_fill(g, "heapobj", 12288, &hold, emit);
        probe_fill(g, "heap", 8192, &hold, emit);
    }
    emit("released: " + probe_state(g));
    // C: heap first, then GPU on top: equal totals in B and C = one shared pool.
    {
        ProbeHold hold;
        probe_fill(g, "heap", 8192, &hold, emit);
        probe_fill(g, "default", 12288, &hold, emit);
        probe_fill(g, "custom_l0", 12288, &hold, emit);
    }
    emit("released: " + probe_state(g));
    // E: GPU buffers over CPU memory, alone and on top of a full DXGI budget.
    // Before D: on the Series X (1.7.12.49, 2026-11-11) the process ended inside
    // D's first reserved buffer without a line.
    emit("existing: start");
    probe_existing_rw(g, emit);
    {
        ProbeHold hold;
        probe_fill(g, "existing", 8192, &hold, emit);
    }
    emit("released: " + probe_state(g));
    {
        ProbeHold hold;
        probe_fill(g, "default", 12288, &hold, emit);
        probe_fill(g, "existing", 8192, &hold, emit);
        probe_fill(g, "heap", 8192, &hold, emit);
    }
    emit("released: " + probe_state(g));
    // D: reserved buffers backed by 64 MiB heaps, last, a line before each.
    for (std::size_t mb : {1024, 2048, 3072}) {
        emit("tiled " + std::to_string(mb) + " MB: start");
        probe_tiled(g, mb, emit);
    }
    emit("done: " + probe_state(g));
}

// --- Selftest ---

namespace {

constexpr int kSelftestTimedRuns = 5;

struct SelftestCase {
    ggml_type type;
    const char* name;
    int n, k, ncols;
};

// Max |gpu - cpu| / max |cpu| against the CPU backend's own path: the
// vec_dot type's from_float, then ggml's vec_dot (#312, gate A).
double reference_rel_err(ggml_type t, const std::vector<std::uint8_t>& q, std::size_t row_bytes,
                         const std::vector<float>& x, const std::vector<float>& y, int n, int k,
                         int ncols) {
    ggml_cpu_init();
    const auto* wt = ggml_get_type_traits_cpu(t);
    std::vector<std::uint8_t> xq(ggml_row_size(wt->vec_dot_type, k));
    double max_ref = 0.0, max_diff = 0.0;
    for (int c = 0; c < ncols; ++c) {
        ggml_get_type_traits_cpu(wt->vec_dot_type)
            ->from_float(x.data() + static_cast<std::size_t>(c) * k, xq.data(), k);
        for (int r = 0; r < n; ++r) {
            float ref = 0.f;
            wt->vec_dot(k, &ref, 0, q.data() + static_cast<std::size_t>(r) * row_bytes, 0,
                        xq.data(), 0, 1);
            const double got = y[static_cast<std::size_t>(c) * n + r];
            // std::max ignores a NaN operand, which would hide corrupt output.
            if (!std::isfinite(ref) || !std::isfinite(got))
                return std::numeric_limits<double>::infinity();
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref)));
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(ref) - got));
        }
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

D3d12SelftestRow run_case(ggml_backend_t backend, const SelftestCase& sc) {
    D3d12SelftestRow row;
    row.type = sc.name;
    row.n = sc.n;
    row.k = sc.k;
    row.ncols = sc.ncols;

    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_w = ggml_init(ip);
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_2d(ctx_w, sc.type, sc.k, sc.n);
    ggml_tensor* x = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, sc.k, sc.ncols);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &kWeightsBuft);
    ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &kHostBuft);
    ggml_gallocr_t galloc = nullptr;

    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_x);
        ggml_free(ctx_w);
    };
    if (!wbuf || !xbuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }

    std::mt19937 rng(1234u + static_cast<unsigned>(sc.n + sc.k + sc.ncols));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> wf(static_cast<std::size_t>(sc.n) * sc.k);
    for (float& v : wf)
        v = uni(rng);
    std::vector<std::uint8_t> q(ggml_nbytes(w));
    ggml_quantize_chunk(sc.type, wf.data(), q.data(), 0, sc.n, sc.k, nullptr);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    std::vector<float> xf(static_cast<std::size_t>(sc.k) * sc.ncols);
    for (float& v : xf)
        v = uni(rng);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));

    ggml_tensor* y = ggml_mul_mat(ctx_g, w, x);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    // Warm-up, then the median of kSelftestTimedRuns: one sample of a ~9 MB
    // matmul swung from 78 to 109 GB/s across back-to-back selftests (CI
    // 1.6.0.1154, same blob).
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    std::vector<double> ms;
    for (int i = 0; i < kSelftestTimedRuns && st == GGML_STATUS_SUCCESS; ++i) {
        st = ggml_backend_graph_compute(backend, gf);
        ms.push_back(gpu().last_gpu_ms);
    }
    row.d3d12_ran = st == GGML_STATUS_SUCCESS;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
    row.gpu_ms = ms[ms.size() / 2];
    row.packed_gbs = row.gpu_ms > 0.0 ? static_cast<double>(q.size()) / 1e6 / row.gpu_ms : 0.0;
    std::vector<float> yf(static_cast<std::size_t>(sc.n) * sc.ncols);
    ggml_backend_tensor_get(y, yf.data(), 0, yf.size() * sizeof(float));
    row.rel_err = reference_rel_err(sc.type, q, w->nb[1], xf, yf, sc.n, sc.k, sc.ncols);
    row.ok = row.rel_err <= kD3d12SelftestRelTol;
    if (!row.ok)
        row.error = "mismatch vs the CPU backend (q8 vec_dot)";
    cleanup();
    return row;
}

// GET_ROWS from a Q6_K weight (the tied token_embd path, #309): the GPU must
// equal ggml's dequantizer bit for bit, as the host emulation does.
D3d12SelftestRow run_get_rows_case(ggml_backend_t backend, int n, int k) {
    D3d12SelftestRow row;
    row.type = "get_rows_q6_k";
    row.n = n;
    row.k = k;
    const std::int32_t ids_v[] = {0, 1, n / 2 + 1, n - 1, 7, n / 3};
    const int n_ids = static_cast<int>(sizeof(ids_v) / sizeof(ids_v[0]));
    row.ncols = n_ids;

    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_w = ggml_init(ip);
    ggml_context* ctx_i = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q6_K, k, n);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx_i, GGML_TYPE_I32, n_ids);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &kWeightsBuft);
    ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_i, &kHostBuft);
    ggml_gallocr_t galloc = nullptr;
    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (ibuf)
            ggml_backend_buffer_free(ibuf);
        ggml_free(ctx_g);
        ggml_free(ctx_i);
        ggml_free(ctx_w);
    };
    if (!wbuf || !ibuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }
    std::mt19937 rng(99u);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> wf(static_cast<std::size_t>(n) * k);
    for (float& v : wf)
        v = uni(rng);
    std::vector<std::uint8_t> q(ggml_nbytes(w));
    ggml_quantize_chunk(GGML_TYPE_Q6_K, wf.data(), q.data(), 0, n, k, nullptr);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    ggml_backend_tensor_set(ids, ids_v, 0, sizeof(ids_v));

    ggml_tensor* y = ggml_get_rows(ctx_g, w, ids);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    row.d3d12_ran = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    row.gpu_ms = gpu().last_gpu_ms;
    std::vector<float> got(static_cast<std::size_t>(k) * n_ids), ref(static_cast<std::size_t>(k));
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
    double max_ref = 0.0, max_diff = 0.0;
    for (int i = 0; i < n_ids; ++i) {
        ggml_get_type_traits(GGML_TYPE_Q6_K)
            ->to_float(q.data() + static_cast<std::size_t>(ids_v[i]) * w->nb[1], ref.data(), k);
        for (int j = 0; j < k; ++j) {
            const double actual = got[static_cast<std::size_t>(i) * k + j];
            if (!std::isfinite(ref[j]) || !std::isfinite(actual)) {
                row.rel_err = std::numeric_limits<double>::infinity();
                row.error = "non-finite get_rows reference or output";
                cleanup();
                return row;
            }
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[j])));
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(ref[j]) - actual));
        }
    }
    row.rel_err = max_ref > 0.0 ? max_diff / max_ref : max_diff;
    row.ok = max_diff == 0.0; // bit-exact, like the host emulation
    if (!row.ok)
        row.error = "mismatch vs ggml dequantize_row_q6_K";
    cleanup();
    return row;
}

// The buffer type llama_expert_cache_use_host_of hands this device: its name
// and device are what is_expert_cache checks, its buffers are D3D12_Host ones.
const char* selftest_expert_cache_name(ggml_backend_buffer_type_t) {
    return "EXPERT_CACHE";
}
ggml_backend_buffer_t selftest_expert_cache_alloc(ggml_backend_buffer_type_t buft, size_t size);
ggml_backend_buffer_type kSelftestExpertCacheBuft = {
    /* .iface = */
    {
        /* .get_name         = */ selftest_expert_cache_name,
        /* .alloc_buffer     = */ selftest_expert_cache_alloc,
        /* .alloc_buffer_n   = */ nullptr,
        /* .get_alignment    = */ buft_alignment,
        /* .get_max_size     = */ nullptr,
        /* .get_alloc_size   = */ nullptr,
        /* .get_alloc_size_n = */ nullptr,
        /* .is_host          = */ buft_is_host_true,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};
ggml_backend_buffer_t selftest_expert_cache_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    return alloc_buf(buft, size, true);
}

// MUL_MAT_ID over IQ4_XS or IQ3_S expert slots (Qwen3.6-35B-A3B's routed
// experts): ne11 = 1 is gate/up (one activation per token), ne11 = n_ids is
// down (one per expert). Reference: the CPU backend's q8_K from_float and vec_dot.
D3d12SelftestRow run_mmid_case(ggml_backend_t backend, ggml_type wtype, int n, int k, int ne11,
                               int n_tok) {
    constexpr int kSlots = 12;
    constexpr int kIds = 8;
    D3d12SelftestRow row;
    row.type = std::string("mmid_") + ggml_type_name(wtype);
    row.n = n;
    row.k = k;
    row.ncols = kIds * n_tok;

    kSelftestExpertCacheBuft.device = kHostBuft.device;
    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_w = ggml_init(ip);
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_tensor* as = ggml_new_tensor_3d(ctx_w, wtype, k, n, kSlots);
    ggml_tensor* x = ggml_new_tensor_3d(ctx_x, GGML_TYPE_F32, k, ne11, n_tok);
    ggml_tensor* ids = ggml_new_tensor_2d(ctx_x, GGML_TYPE_I32, kIds, n_tok);
    ggml_backend_buffer_t wbuf =
        ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &kSelftestExpertCacheBuft);
    ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &kHostBuft);
    ggml_gallocr_t galloc = nullptr;
    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_x);
        ggml_free(ctx_w);
    };
    if (!wbuf || !xbuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }
    if (!in_expert_cache(as)) {
        row.error = "expert slots not recognised as EXPERT_CACHE";
        cleanup();
        return row;
    }

    std::mt19937 rng(4242u + static_cast<unsigned>(n + k + ne11 + n_tok));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> wf(static_cast<std::size_t>(n) * k * kSlots);
    for (float& v : wf)
        v = uni(rng);
    std::vector<std::uint8_t> q(ggml_nbytes(as));
    ggml_quantize_chunk(wtype, wf.data(), q.data(), 0, static_cast<std::int64_t>(n) * kSlots, k,
                        nullptr);
    ggml_backend_tensor_set(as, q.data(), 0, q.size());
    std::vector<float> xf(static_cast<std::size_t>(k) * ne11 * n_tok);
    for (float& v : xf)
        v = uni(rng);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
    // Distinct slots per token, as top-k routing gives, including the last.
    std::vector<std::int32_t> idv(static_cast<std::size_t>(kIds) * n_tok);
    for (int t = 0; t < n_tok; ++t)
        for (int i = 0; i < kIds; ++i)
            idv[static_cast<std::size_t>(t) * kIds + i] = (kSlots - 1 + 7 * i + 5 * t) % kSlots;
    ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * sizeof(std::int32_t));

    ggml_tensor* y = ggml_mul_mat_id(ctx_g, as, x, ids);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    std::vector<double> ms;
    for (int i = 0; i < kSelftestTimedRuns && st == GGML_STATUS_SUCCESS; ++i) {
        st = ggml_backend_graph_compute(backend, gf);
        ms.push_back(gpu().last_gpu_ms);
    }
    row.d3d12_ran = st == GGML_STATUS_SUCCESS;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
    row.gpu_ms = ms[ms.size() / 2];
    // Bytes of the experts the dispatch reads: kIds of them per token.
    row.packed_gbs =
        row.gpu_ms > 0.0 ? static_cast<double>(as->nb[2]) * kIds * n_tok / 1e6 / row.gpu_ms : 0.0;

    std::vector<float> yf(static_cast<std::size_t>(n) * kIds * n_tok);
    ggml_backend_tensor_get(y, yf.data(), 0, yf.size() * sizeof(float));
    ggml_cpu_init();
    const auto* wt = ggml_get_type_traits_cpu(wtype);
    std::vector<std::uint8_t> xq(ggml_row_size(wt->vec_dot_type, k));
    double max_ref = 0.0, max_diff = 0.0;
    for (int t = 0; t < n_tok; ++t)
        for (int i = 0; i < kIds; ++i) {
            const int slot = idv[static_cast<std::size_t>(t) * kIds + i];
            const std::size_t col = static_cast<std::size_t>(t) * ne11 + (ne11 == 1 ? 0 : i);
            ggml_get_type_traits_cpu(wt->vec_dot_type)
                ->from_float(xf.data() + col * k, xq.data(), k);
            for (int r = 0; r < n; ++r) {
                float ref = 0.f;
                wt->vec_dot(k, &ref, 0, q.data() + slot * as->nb[2] + r * as->nb[1], 0, xq.data(),
                            0, 1);
                const double got = yf[(static_cast<std::size_t>(t) * kIds + i) * n + r];
                if (!std::isfinite(ref) || !std::isfinite(got)) {
                    row.rel_err = std::numeric_limits<double>::infinity();
                    row.error = "non-finite mul_mat_id reference or output";
                    cleanup();
                    return row;
                }
                max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref)));
                max_diff = std::max(max_diff, std::fabs(static_cast<double>(ref) - got));
            }
        }
    row.rel_err = max_ref > 0.0 ? max_diff / max_ref : max_diff;
    row.ok = row.rel_err <= kD3d12SelftestRelTol;
    if (!row.ok)
        row.error = "mismatch vs the CPU backend (q8_K vec_dot, mul_mat_id)";
    cleanup();
    return row;
}

// A MoE layer's tail as llama builds it (qwen35moe, patches 0007/0008): mode -1
// is the routed experts' combine, E [ne0, n, T] by w [1, n, T] then the n - 1
// ADDs of its views; modes 0..2 the tail, s [ne0, T] (mode 0: [ne0, n, T], the
// combine's MUL alone) by g, + m, + q. One dispatch, bit for bit the host
// emulation, which tests/test_ggml_d3d12_moe.cpp holds to ggml's CPU ops.
D3d12SelftestRow run_moe_case(ggml_backend_t backend, int mode, int ne0, int n, int n_tok) {
    D3d12SelftestRow row;
    row.type = mode < 0 ? "moe_combine" : "moe_tail" + std::to_string(mode);
    row.n = ne0;
    row.k = n;
    row.ncols = n_tok;

    ggml_init_params ip = {};
    ip.mem_size =
        static_cast<std::size_t>(4 * n + 16) * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    const bool three_d = mode <= 0;
    const int rows = three_d ? n : 1;
    ggml_tensor* s = ggml_new_tensor_3d(ctx_x, GGML_TYPE_F32, ne0, rows, n_tok);
    ggml_tensor* gt = ggml_new_tensor_3d(ctx_x, GGML_TYPE_F32, 1, rows, n_tok);
    ggml_tensor* m = ggml_new_tensor_3d(ctx_x, GGML_TYPE_F32, ne0, 1, n_tok);
    ggml_tensor* q = ggml_new_tensor_3d(ctx_x, GGML_TYPE_F32, ne0, 1, n_tok);
    ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &kHostBuft);
    ggml_gallocr_t galloc = nullptr;
    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_x);
    };
    if (!xbuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }
    std::mt19937 rng(777u + static_cast<unsigned>(mode + 2 + ne0 + n + n_tok));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    auto fill = [&](ggml_tensor* t) {
        std::vector<float> v(static_cast<std::size_t>(ggml_nelements(t)));
        for (float& x : v)
            x = uni(rng);
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
        return v;
    };
    const std::vector<float> sv = fill(s), gv = fill(gt), mv = fill(m), qv = fill(q);

    // 2-d views of m and q: [ne0, T], as llama's moe_out and residual are.
    ggml_tensor* m2 = ggml_reshape_2d(ctx_g, m, ne0, n_tok);
    ggml_tensor* q2 = ggml_reshape_2d(ctx_g, q, ne0, n_tok);
    ggml_tensor* out = nullptr;
    if (mode < 0) {
        ggml_tensor* w = ggml_mul(ctx_g, s, gt);
        for (int i = 0; i < n; ++i) {
            ggml_tensor* v = ggml_view_2d(ctx_g, w, ne0, n_tok, w->nb[2],
                                          static_cast<std::size_t>(i) * w->nb[1]);
            out = out ? ggml_add(ctx_g, out, v) : v;
        }
    } else if (mode == 0) {
        out = ggml_mul(ctx_g, s, gt);
    } else {
        ggml_tensor* s2 = ggml_reshape_2d(ctx_g, s, ne0, n_tok);
        ggml_tensor* g2 = ggml_reshape_2d(ctx_g, gt, 1, n_tok);
        out = ggml_add(ctx_g, m2, ggml_mul(ctx_g, s2, g2));
        if (mode == 2)
            out = ggml_add(ctx_g, out, q2);
    }
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, out);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    const std::uint64_t d0 = gpu().n_dispatches;
    row.d3d12_ran = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    const std::uint64_t dispatches = gpu().n_dispatches - d0;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    row.gpu_ms = gpu().last_gpu_ms;

    std::vector<float> got(static_cast<std::size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    std::vector<float> ref(got.size());
    bool planned = false;
    if (mode < 0) {
        D3d12MoeCombinePlan p;
        planned = d3d12_moe_combine_plan(out, &p) &&
                  p.params.dst_token_stride == static_cast<std::uint32_t>(ne0);
        if (planned)
            d3d12_moe_combine_emulate(p.params, p.groups_x, p.groups_y, sv.data(), gv.data(),
                                      ref.data());
    } else {
        D3d12MoeTailPlan p;
        // One token has no second row, so the plan's row stride is 0 there.
        planned = d3d12_moe_tail_plan(out, &p) &&
                  (p.groups_y == 1 || p.params.dst_row_stride == static_cast<std::uint32_t>(ne0));
        if (planned)
            d3d12_moe_tail_emulate(p.params, p.groups_x, p.groups_y, sv.data(), gv.data(),
                                   mv.data(), qv.data(), ref.data());
    }
    if (!planned) {
        row.error = "no MoE plan for the graph";
        cleanup();
        return row;
    }
    double max_ref = 0.0, max_diff = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[i])));
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(ref[i]) - got[i]));
    }
    row.rel_err = max_ref > 0.0 ? max_diff / max_ref : max_diff;
    row.ok =
        std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0 && dispatches == 1;
    if (dispatches != 1)
        row.error = "not fused: " + std::to_string(dispatches) + " dispatches";
    else if (!row.ok)
        row.error = "mismatch vs the host emulation (bit for bit)";
    cleanup();
    return row;
}

// The delta net's kernels at Qwen3.6-35B-A3B's decode shapes (one token),
// on the GPU against their host emulations: the copies and get_rows bit for
// bit, the rest within the emulation tests' tolerances. kind 0 SIGMOID [32]
// (beta), 1 SOFTPLUS [32] (the gate), 2 SILU [128, 32] (z), 3 SCALE [128, 16]
// (q), 4 CONCAT of the conv states and the transposed qkv, 5 GET_ROWS of a
// conv state, 6 SILU(SSM_CONV), 7 GATED_DELTA_NET.
D3d12SelftestRow run_ssm_case(ggml_backend_t backend, int kind) {
    static const char* const kNames[] = {"ssm_sigmoid", "ssm_softplus", "ssm_silu", "ssm_scale",
                                         "ssm_concat",  "ssm_get_rows", "ssm_conv", "ssm_gdn"};
    constexpr std::int64_t kCh = 8192, kConv = 4, kSv = kD3d12GdnSv, kHk = 16, kHv = 32;
    D3d12SelftestRow row;
    row.type = kNames[kind];
    row.ncols = 1;

    ggml_init_params ip = {};
    ip.mem_size = 32 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    std::vector<ggml_tensor*> in; // the operands, filled with uniform(-1, 1)
    auto add = [&](std::int64_t n0, std::int64_t n1, std::int64_t n2 = 1, std::int64_t n3 = 1) {
        in.push_back(ggml_new_tensor_4d(ctx_x, GGML_TYPE_F32, n0, n1, n2, n3));
        return in.back();
    };
    ggml_tensor* ids = nullptr;
    switch (kind) {
    case 0:
    case 1:
        add(kHv, 1);
        break;
    case 2:
        add(kSv, kHv);
        break;
    case 3:
        add(kSv, kHk);
        break;
    case 4:
        add(kConv - 1, kCh); // conv states
        add(kCh, 1);         // qkv
        break;
    case 5:
        add((kConv - 1) * kCh, 2); // two cells' conv states
        ids = ggml_new_tensor_1d(ctx_x, GGML_TYPE_I32, 1);
        break;
    case 6:
        add(kConv, kCh, 1); // conv input [d_conv - 1 + T, channels, S]
        add(kConv, kCh);    // conv1d
        break;
    default:
        add(kSv, kHk, 1, 1);   // q
        add(kSv, kHk, 1, 1);   // k
        add(kSv, kHv, 1, 1);   // v
        add(1, kHv, 1, 1);     // g
        add(1, kHv, 1, 1);     // beta
        add(kSv, kSv, kHv, 1); // state
        break;
    }
    ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &kHostBuft);
    ggml_gallocr_t galloc = nullptr;
    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_x);
    };
    if (!xbuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }
    std::mt19937 rng(4242u + static_cast<unsigned>(kind));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<std::vector<float>> vals;
    for (ggml_tensor* t : in) {
        std::vector<float> v(static_cast<std::size_t>(ggml_nelements(t)));
        for (float& x : v)
            x = uni(rng);
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
        vals.push_back(std::move(v));
    }
    const std::int32_t id = 1;
    if (ids)
        ggml_backend_tensor_set(ids, &id, 0, sizeof(id));

    ggml_tensor* out = nullptr;
    switch (kind) {
    case 0:
        out = ggml_sigmoid(ctx_g, in[0]);
        break;
    case 1:
        out = ggml_softplus(ctx_g, in[0]);
        break;
    case 2:
        out = ggml_silu(ctx_g, in[0]);
        break;
    case 3:
        out = ggml_scale(ctx_g, in[0], 1.0f / std::sqrt(static_cast<float>(kSv)));
        break;
    case 4:
        out = ggml_concat(ctx_g, in[0], ggml_transpose(ctx_g, in[1]), 0);
        break;
    case 5:
        out = ggml_get_rows(ctx_g, in[0], ids);
        break;
    case 6:
        out = ggml_silu(ctx_g, ggml_ssm_conv(ctx_g, in[0], in[1]));
        break;
    default:
        out = ggml_gated_delta_net(ctx_g, in[0], in[1], in[2], in[3], in[4], in[5], 1);
        break;
    }
    row.n = static_cast<int>(out->ne[0]);
    row.k = static_cast<int>(out->ne[1]);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, out);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    const std::uint64_t d0 = gpu().n_dispatches;
    row.d3d12_ran = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    const std::uint64_t dispatches = gpu().n_dispatches - d0;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    row.gpu_ms = gpu().last_gpu_ms;

    std::vector<float> got(static_cast<std::size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    std::vector<float> ref(got.size(), 0.0f);
    bool planned = false;
    std::uint64_t want = 1;
    double tol = 0.0; // bit for bit
    if (kind <= 3) {
        D3d12SsmUnaryPlan p;
        planned = d3d12_ssm_unary_plan(out, &p);
        if (planned)
            d3d12_ssm_unary_emulate(p.params, p.groups_x, p.groups_y, vals[0].data(), ref.data());
        tol = kind == 3 ? 0.0 : 1e-5;
    } else if (kind == 4) {
        D3d12SsmCopyPlan p;
        planned = d3d12_ssm_copy_plan(out, &p) && p.copies.size() == 2;
        if (planned)
            for (const D3d12SsmCopy& c : p.copies)
                d3d12_ssm_copy_emulate(c.params, c.groups_x,
                                       (base_of(c.src) == in[0] ? vals[0] : vals[1]).data(),
                                       ref.data() + c.dst_offset / sizeof(float));
        want = 2;
    } else if (kind == 5) {
        D3d12SsmGetRowsPlan p;
        planned = d3d12_ssm_get_rows_plan(out, &p);
        if (planned)
            d3d12_ssm_get_rows_emulate(p.params, p.groups_x, p.groups_y, vals[0].data(), &id,
                                       ref.data());
    } else if (kind == 6) {
        D3d12SsmConvPlan p;
        planned = d3d12_ssm_conv_plan(out, &p) && p.params.silu == 1;
        if (planned)
            d3d12_ssm_conv_emulate(p.params, p.groups_x, p.groups_y, p.groups_z, vals[0].data(),
                                   vals[1].data(), ref.data());
        tol = 1e-5;
    } else {
        D3d12GdnPlan p;
        planned = d3d12_gdn_plan(out, &p);
        if (planned)
            d3d12_gdn_emulate(p.params, p.groups_x, p.groups_y, p.groups_z, vals[0].data(),
                              vals[1].data(), vals[2].data(), vals[3].data(), vals[4].data(),
                              vals[5].data(), ref.data());
        tol = 1e-4;
    }
    if (!planned) {
        row.error = "no delta-net plan for the graph";
        cleanup();
        return row;
    }
    row.rel_err = d3d12_ssm_max_err(got.data(), ref.data(), got.size());
    const bool match = tol == 0.0
                           ? std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0
                           : row.rel_err <= tol;
    row.ok = match && dispatches == want;
    if (dispatches != want)
        row.error = std::to_string(dispatches) + " dispatches, want " + std::to_string(want);
    else if (!match)
        row.error = tol == 0.0 ? "mismatch vs the host emulation (bit for bit)"
                               : "mismatch vs the host emulation";
    cleanup();
    return row;
}

// The fixed cost of one graph_compute: an empty command list submitted and
// waited for, median wall of 256 (in gpu_ms). Every CPU<->GPU split of a
// model pays it once per ubatch.
D3d12SelftestRow run_submit_rtt() {
    D3d12SelftestRow row;
    row.type = "submit_rtt";
    row.n = 256;
    Gpu& g = gpu();
    std::vector<double> ms;
    std::lock_guard<std::mutex> lock(g.mu);
    for (int i = 0; i < row.n; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!run_now(g, [](ID3D12GraphicsCommandList*) {})) {
            row.error = "submit failed";
            return row;
        }
        ms.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count());
    }
    std::sort(ms.begin(), ms.end());
    row.gpu_ms = ms[ms.size() / 2];
    row.d3d12_ran = row.ok = true;
    return row;
}

} // namespace

void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    if (!ggml_d3d12_register()) {
        D3d12SelftestRow r;
        r.type = "-";
        r.error = gpu().error.empty() ? "d3d12 backend unavailable" : gpu().error;
        out->push_back(std::move(r));
        return;
    }
    ggml_backend_t backend = dev_init_backend(&kDevice, nullptr);
    // Shapes of the D2 target models; q6_k 2048x11008 has 2-byte-aligned rows.
    const SelftestCase cases[] = {
        {GGML_TYPE_Q4_0, "q4_0", 8192, 2048, 1},
        {GGML_TYPE_Q4_0, "q4_0", 2048, 8192, 1},
        {GGML_TYPE_Q4_K, "q4_k", 11008, 2048, 1},
        {GGML_TYPE_Q4_K, "q4_k", 2048, 11008, 1},
        {GGML_TYPE_Q6_K, "q6_k", 2048, 11008, 1},
        {GGML_TYPE_Q6_K, "q6_k", 65536, 1024, 1},
        {GGML_TYPE_Q4_0, "q4_0", 1024, 1024, 7},
        {GGML_TYPE_Q4_K, "q4_k", 1024, 1024, 7},
        {GGML_TYPE_Q6_K, "q6_k", 1024, 1024, 7},
        {GGML_TYPE_Q4_0, "q4_0", 1024, 1024, 512},
        {GGML_TYPE_Q4_K, "q4_k", 1024, 1024, 512},
        {GGML_TYPE_Q6_K, "q6_k", 1024, 1024, 512},
        // Qwen3.5-9B IQ3_M: ffn_gate/up 4096->12288 and ffn_down 12288->4096 are IQ3_S.
        {GGML_TYPE_IQ3_S, "iq3_s", 12288, 4096, 1},
        {GGML_TYPE_IQ3_S, "iq3_s", 4096, 12288, 1},
        {GGML_TYPE_IQ3_S, "iq3_s", 1024, 1024, 7},
        // Qwen3.6-35B-A3B IQ3_S (the 1.7.12.53 kernel profile's mmv iq3_s grids):
        // 4096 -> 2048 (t128), 2048 -> 512 and 512 -> 2048 (shared expert), plus
        // wide rows and a tail no geometry divides (XLLAMA_D3D12_IQ3S_RSL / _RT).
        {GGML_TYPE_IQ3_S, "iq3_s", 2048, 4096, 1},
        {GGML_TYPE_IQ3_S, "iq3_s", 512, 2048, 1},
        {GGML_TYPE_IQ3_S, "iq3_s", 2048, 512, 1},
        {GGML_TYPE_IQ3_S, "iq3_s", 8192, 2048, 1},
        {GGML_TYPE_IQ3_S, "iq3_s", 1030, 1024, 3},
        // Qwen3.6-35B-A3B IQ4_XS: attn_gate 2048->4096, ssm_out 4096->2048,
        // ffn_down_shexp 512->2048, ssm_alpha 2048->32; Q5_K attn_qkv and attn_v.
        {GGML_TYPE_IQ4_XS, "iq4_xs", 4096, 2048, 1},
        {GGML_TYPE_IQ4_XS, "iq4_xs", 2048, 4096, 1},
        {GGML_TYPE_IQ4_XS, "iq4_xs", 2048, 512, 1},
        {GGML_TYPE_IQ4_XS, "iq4_xs", 32, 2048, 1},
        {GGML_TYPE_IQ4_XS, "iq4_xs", 1024, 1024, 7},
        {GGML_TYPE_Q5_K, "q5_k", 8192, 2048, 1},
        {GGML_TYPE_Q5_K, "q5_k", 512, 2048, 1},
        {GGML_TYPE_Q5_K, "q5_k", 1024, 1024, 7},
        // Its router (ffn_gate_inp 2048 -> 256) and shared expert gate (2048 -> 1), kept in f32.
        {GGML_TYPE_F32, "f32", 256, 2048, 1},
        {GGML_TYPE_F32, "f32", 1, 2048, 1},
        {GGML_TYPE_F32, "f32", 256, 2048, 7},
    };
    for (const auto& sc : cases)
        out->push_back(run_case(backend, sc));
    out->push_back(run_get_rows_case(backend, 151936, 2048)); // Coder-3B tied token_embd
    out->push_back(run_get_rows_case(backend, 65536, 1024));  // LFM2.5-350M
    // Qwen3.6-35B-A3B experts (i1-IQ4_XS and i1-IQ3_S): gate/up 2048 -> 512,
    // down 512 -> 2048; one token (decode) and five (a prefill ubatch, both blobs).
    for (const ggml_type wtype : {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S})
        if (gpu().pso_mmid[mmid_for(wtype)][0]) {
            out->push_back(run_mmid_case(backend, wtype, 512, 2048, 1, 1));
            out->push_back(run_mmid_case(backend, wtype, 2048, 512, 8, 1));
            out->push_back(run_mmid_case(backend, wtype, 512, 2048, 1, 5));
            out->push_back(run_mmid_case(backend, wtype, 2048, 512, 8, 5));
        }
    if (gpu().moe_ok) {
        // Qwen3.6-35B-A3B: n_embd 2048, 8 routed experts; decode and a prefill ubatch.
        out->push_back(run_moe_case(backend, -1, 2048, 8, 1));
        out->push_back(run_moe_case(backend, -1, 2048, 8, 4));
        out->push_back(run_moe_case(backend, 0, 2048, 8, 1));
        out->push_back(run_moe_case(backend, 1, 2048, 1, 1));
        out->push_back(run_moe_case(backend, 2, 2048, 1, 1));
        out->push_back(run_moe_case(backend, 2, 2048, 1, 5));
    }
    if (gpu().ssm_ok)
        for (int kind = 0; kind < 8; ++kind)
            out->push_back(run_ssm_case(backend, kind));
    out->push_back(run_submit_rtt());
    ggml_backend_free(backend);
}

} // namespace xllama

    #endif // _WIN32

#endif // XLLAMA_USE_LLAMA
