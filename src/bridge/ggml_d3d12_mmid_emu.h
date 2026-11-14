// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MUL_MAT_ID on the d3d12 backend (shaders/ggml_d3d12_mmid_{iq4_xs,iq3_s,iq2_s,iq2_xxs}.hlsl):
// the kernels' constants and grid from a GGML_OP_MUL_MAT_ID node, their
// activation layout, and a host emulation of each (tests compare it with ggml's
// CPU mul_mat_id; a selftest can compare it with the GPU). The kernels share
// the constants, grid and thread layout; only the expert block differs.
//
// The node: dst[:, i, t] = as[:, :, ids[i, t]] x src1[:, i % ne11, t], with
//   as   = the experts, IQ4_XS, IQ3_S, IQ2_S or IQ2_XXS [K, N, n_as] (with the expert cache: the
//          resident slots, and ids the slot ids)
//   src1 = the activations, F32 [K, ne11, T]; ne11 = 1 for gate/up (one
//          column per token), n_ids for down (one per expert)
//   ids  = I32 [n_ids, T], often a view (nb1 > 4 * n_ids)
//   dst  = F32 [N, n_ids, T]
// X (u1) is src1's rows quantized to q8_K, row t * ne11 + c
// (d3d12_mmid_quantize_activations). One thread group per (R rows, token t,
// id i): grid (ceil(N / R), n_ids * T), the t64 or t128 blob by
// d3d12_mm_threads(K) as for MUL_MAT, R = d3d12_geom_rows(geometry): 4 for
// IQ4_XS, whose kernel has no geometry, and for IQ3_S / IQ2_S by the plan's request
// (pad0, shaders/ggml_d3d12_geom.hlsli). Offsets are 32-bit, as in the shader.

#pragma once

#include "xllama/ggml_d3d12.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace xllama {

// b0: the 8 constants of every ggml_d3d12 kernel (ggml_d3d12_common.hlsli),
// then the 5 of MUL_MAT_ID; 13 root constants in this order.
struct D3d12MmidParams {
    std::uint32_t n = 0;           // rows per expert (ne01)
    std::uint32_t k_dim = 0;       // ne00, multiple of 256
    std::uint32_t nchunk = 0;      // k_dim / 256
    std::uint32_t w_row_bytes = 0; // as nb1
    std::uint32_t x_stride = 0;    // bytes between tokens in X
    std::uint32_t y_stride = 0;    // floats between tokens in Y (dst nb2 / 4)
    std::uint32_t pad0 = 0;        // IQ3_S / IQ2_S: d3d12_geom_pack(geometry)
    std::uint32_t pad1 = 0;
    std::uint32_t expert_stride = 0;   // bytes between experts in W (as nb2)
    std::uint32_t n_ids = 0;           // experts per token (ids ne0)
    std::uint32_t ids_stride = 0;      // bytes between tokens in Ids (ids nb1)
    std::uint32_t x_col_stride = 0;    // bytes between a token's columns, 0 = one
    std::uint32_t y_expert_stride = 0; // floats between a token's experts (dst nb1 / 4)
};
static_assert(sizeof(D3d12MmidParams) == 13 * sizeof(std::uint32_t), "13 root constants");

struct D3d12MmidPlan {
    D3d12MmidParams params;
    std::uint32_t groups_x = 0; // ceil(N / d3d12_geom_rows(geom))
    std::uint32_t groups_y = 0; // n_ids * T
    std::uint32_t threads = 0;  // d3d12_mm_threads(K): which blob
    std::size_t x_bytes = 0;    // X (u1), d3d12_mmid_quantize_activations
    std::size_t ids_bytes = 0;  // Ids (t1) from ids->data
    D3d12Geom geom;             // IQ3_S / IQ2_S only; {1, 1} otherwise
};

// The expert types with a MUL_MAT_ID kernel.
inline bool d3d12_mmid_type_supported(ggml_type t) {
    return t == GGML_TYPE_IQ4_XS || t == GGML_TYPE_IQ3_S || t == GGML_TYPE_IQ2_S ||
           t == GGML_TYPE_IQ2_XXS;
}

// false when no kernel runs dst: weights not IQ4_XS, IQ3_S, IQ2_S or IQ2_XXS, shapes outside
// ggml_mul_mat_id's 3-d case, ne11 neither 1 nor n_ids, a grid past
// kD3d12MaxGroups or an offset past 32 bits. req shapes IQ3_S / IQ2_S groups
// (d3d12_geom_for).
inline bool d3d12_mmid_plan(const ggml_tensor* dst, D3d12MmidPlan* plan,
                            const D3d12GeomReq& req = {}) {
    if (!dst || !plan || dst->op != GGML_OP_MUL_MAT_ID)
        return false;
    const ggml_tensor* as = dst->src[0];
    const ggml_tensor* b = dst->src[1];
    const ggml_tensor* ids = dst->src[2];
    if (!as || !b || !ids || !d3d12_mmid_type_supported(as->type) || b->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32)
        return false;
    const std::int64_t k = as->ne[0], n = as->ne[1], n_as = as->ne[2];
    const std::int64_t n_ids = ids->ne[0], n_tok = ids->ne[1], ne11 = b->ne[1];
    if (k <= 0 || k % kD3d12Chunk != 0 || n <= 0 || n_as <= 0 || n_ids <= 0 || n_tok <= 0)
        return false;
    if (as->ne[3] != 1 || b->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 || dst->ne[3] != 1)
        return false;
    if (b->ne[0] != k || b->ne[2] != n_tok || (ne11 != 1 && ne11 != n_ids))
        return false;
    if (dst->ne[0] != n || dst->ne[1] != n_ids || dst->ne[2] != n_tok)
        return false;
    // IQ3_S / IQ2_S / IQ2_XXS blocks (110 / 82 / 66 B) already sit on 2-byte boundaries inside a
    // row, and their kernels read through ld16 / ld32 or a slab from the dword below: any even row
    // stride runs (Qwen3-Coder-30B-A3B's ffn_down, K = 768: 330 B rows). IQ4_XS: aligned Load2s.
    const std::size_t row_align = d3d12_geom_type(as->type) ? 2u : 4u;
    if (as->nb[0] != ggml_type_size(as->type) || as->nb[1] % row_align != 0 || as->nb[2] % 4 != 0 ||
        b->nb[0] != sizeof(float) || ids->nb[0] != sizeof(std::int32_t) || ids->nb[1] % 4 != 0 ||
        dst->nb[0] != sizeof(float) || dst->nb[1] % 4 != 0 || dst->nb[2] % 4 != 0)
        return false;
    if (as->nb[1] < ggml_row_size(as->type, k))
        return false;

    const std::uint64_t u32 = std::numeric_limits<std::uint32_t>::max();
    const std::uint64_t xq_row = ggml_row_size(d3d12_activation_type(as->type), k);
    const std::uint64_t x_stride = static_cast<std::uint64_t>(ne11) * xq_row;
    const std::uint32_t threads = static_cast<std::uint32_t>(d3d12_mm_threads(k));
    const D3d12Geom geom =
        d3d12_geom_type(as->type)
            ? d3d12_geom_for(req, static_cast<std::uint32_t>(k / kD3d12Chunk), threads)
            : D3d12Geom{};
    const std::uint64_t rows = d3d12_geom_rows(geom);
    const std::uint64_t groups_x = (static_cast<std::uint64_t>(n) + rows - 1) / rows;
    const std::uint64_t groups_y = static_cast<std::uint64_t>(n_ids) * n_tok;
    const std::uint64_t w_end = static_cast<std::uint64_t>(n_as) * as->nb[2];
    const std::uint64_t ids_end = (n_tok - 1) * static_cast<std::uint64_t>(ids->nb[1]) +
                                  4u * static_cast<std::uint64_t>(n_ids);
    const std::uint64_t x_end = static_cast<std::uint64_t>(n_tok) * x_stride;
    const std::uint64_t y_end = (n_tok - 1) * static_cast<std::uint64_t>(dst->nb[2] / 4) +
                                (n_ids - 1) * static_cast<std::uint64_t>(dst->nb[1] / 4) + n;
    if (groups_x > kD3d12MaxGroups || groups_y > kD3d12MaxGroups || w_end > u32 || ids_end > u32 ||
        x_end > u32 || y_end > u32 || as->nb[1] > u32)
        return false;

    D3d12MmidParams& p = plan->params;
    p = D3d12MmidParams{};
    p.n = static_cast<std::uint32_t>(n);
    p.k_dim = static_cast<std::uint32_t>(k);
    p.nchunk = static_cast<std::uint32_t>(k / kD3d12Chunk);
    p.w_row_bytes = static_cast<std::uint32_t>(as->nb[1]);
    p.x_stride = static_cast<std::uint32_t>(x_stride);
    p.y_stride = static_cast<std::uint32_t>(dst->nb[2] / 4);
    p.pad0 = d3d12_geom_type(as->type) ? d3d12_geom_pack(geom) : 0u;
    p.expert_stride = static_cast<std::uint32_t>(as->nb[2]);
    p.n_ids = static_cast<std::uint32_t>(n_ids);
    p.ids_stride = static_cast<std::uint32_t>(ids->nb[1]);
    p.x_col_stride = ne11 == 1 ? 0u : static_cast<std::uint32_t>(xq_row);
    p.y_expert_stride = static_cast<std::uint32_t>(dst->nb[1] / 4);
    plan->groups_x = static_cast<std::uint32_t>(groups_x);
    plan->groups_y = static_cast<std::uint32_t>(groups_y);
    plan->threads = threads;
    plan->geom = geom;
    plan->x_bytes = static_cast<std::size_t>(x_end);
    plan->ids_bytes = static_cast<std::size_t>(ids_end);
    return true;
}

// X for a planned node: src1's rows (src1_data laid out by src1's nb) through
// the vec_dot type's from_float, the CPU backend's own quantization. `out`
// holds plan.x_bytes.
inline void d3d12_mmid_quantize_activations(const ggml_tensor* dst, const void* src1_data,
                                            std::uint8_t* out) {
    const ggml_tensor* b = dst->src[1];
    const ggml_type at = d3d12_activation_type(dst->src[0]->type);
    const auto from_float = ggml_get_type_traits_cpu(at)->from_float;
    const std::size_t row = ggml_row_size(at, b->ne[0]);
    const auto* base = static_cast<const std::uint8_t*>(src1_data);
    for (std::int64_t t = 0; t < b->ne[2]; ++t)
        for (std::int64_t c = 0; c < b->ne[1]; ++c) {
            const auto* src = reinterpret_cast<const float*>(base + t * b->nb[2] + c * b->nb[1]);
            from_float(src, out + static_cast<std::size_t>(t * b->ne[1] + c) * row, b->ne[0]);
        }
}

namespace d3d12_mmid_detail {

inline constexpr std::uint32_t kIq3sGrid[512] = {
#include "../../shaders/ggml_d3d12_iq3s_grid.inc"
};

// 1024 entries x 8 two-bit codes (8 -> 0, 25 -> 1, 43 -> 2), two entries a dword.
inline constexpr std::uint32_t kIq2sGrid[512] = {
#include "../../shaders/ggml_d3d12_iq2s_grid.inc"
};

// IQ2_XXS: 256 entries in the same codes, two entries a dword.
inline constexpr std::uint32_t kIq2xxsGrid[128] = {
#include "../../shaders/ggml_d3d12_iq2xxs_grid.inc"
};

inline std::uint32_t ld16(const std::uint8_t* at) {
    std::uint16_t v;
    std::memcpy(&v, at, 2);
    return v;
}

inline std::uint32_t ld32(const std::uint8_t* at) {
    std::uint32_t v;
    std::memcpy(&v, at, 4);
    return v;
}

// One thread's integer sum for one row and one 256-element chunk, already
// times the sub-block scale, and the block's d: the kernels' acc +=
// (d * dy) * (float)(ls * sumi). bb is the weight block, qx the q8_K quants.
inline std::int32_t block_lsumi(ggml_type t, const std::uint8_t* bb, std::uint32_t itid,
                                const std::int8_t* qx, float* d) {
    *d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(ld16(bb)));
    if (t == GGML_TYPE_IQ3_S) { // ggml_d3d12_mmid_iq3_s.hlsl
        const std::uint32_t ib32 = itid >> 1, h = itid & 1u, e = 32u * ib32 + 16u * h;
        const std::uint32_t qs = ld32(bb + 2u + 8u * ib32 + 4u * h);
        const std::uint32_t hb = (bb[66u + ib32] >> (4u * h)) & 0xFu;
        const std::uint32_t sg = ld16(bb + 74u + 4u * ib32 + 2u * h);
        const std::int32_t ls =
            2 * static_cast<std::int32_t>((bb[106u + (ib32 >> 1)] >> (4u * (ib32 & 1u))) & 0xFu) +
            1;
        std::int32_t sumi = 0;
        for (std::uint32_t k = 0; k < 4; ++k) {
            const std::uint32_t g = kIq3sGrid[((qs >> (8u * k)) & 0xFFu) | (((hb >> k) & 1u) << 8)];
            const std::uint32_t s4 = (sg >> (4u * k)) & 0xFu;
            for (std::uint32_t j = 0; j < 4; ++j) {
                const std::int32_t mag = static_cast<std::int32_t>((g >> (8u * j)) & 0xFFu);
                sumi += (((s4 >> j) & 1u) ? -mag : mag) * qx[e + 4u * k + j];
            }
        }
        return ls * sumi;
    }
    if (t == GGML_TYPE_IQ2_S) { // ggml_d3d12_mmid_iq2_s.hlsl: ggml's final 0.125 folded into d
        *d *= 0.125f;
        const std::uint32_t ib32 = itid >> 1, h = itid & 1u, e = 32u * ib32 + 16u * h;
        const std::uint32_t qs = ld16(bb + 2u + 4u * ib32 + 2u * h);
        const std::uint32_t sg = ld16(bb + 34u + 4u * ib32 + 2u * h);
        const std::uint32_t qh = static_cast<std::uint32_t>(bb[66u + ib32]) >> (4u * h);
        const std::int32_t ls =
            2 * static_cast<std::int32_t>((bb[74u + ib32] >> (4u * h)) & 0xFu) + 1;
        std::int32_t sumi = 0;
        for (std::uint32_t l = 0; l < 2; ++l) {
            const std::uint32_t idx = ((qs >> (8u * l)) & 0xFFu) | (((qh >> (2u * l)) & 3u) << 8);
            const std::uint32_t c = (kIq2sGrid[idx >> 1] >> (16u * (idx & 1u))) & 0xFFFFu;
            const std::uint32_t s8 = (sg >> (8u * l)) & 0xFFu;
            for (std::uint32_t j = 0; j < 8; ++j) {
                const std::uint32_t b = (c >> (2u * j)) & 3u;
                const std::int32_t mag = static_cast<std::int32_t>(8u + 17u * b + (b >> 1));
                sumi += (((s8 >> j) & 1u) ? -mag : mag) * qx[e + 8u * l + j];
            }
        }
        return ls * sumi;
    }
    if (t == GGML_TYPE_IQ2_XXS) { // ggml_d3d12_mmid_iq2_xxs.hlsl: ggml's final 0.125 folded into d
        *d *= 0.125f;
        const std::uint32_t ib32 = itid >> 1, h = itid & 1u, e = 32u * ib32 + 16u * h;
        const std::uint8_t* g8 = bb + 2u + 8u * ib32;
        const std::uint32_t qs = ld16(g8 + 2u * h);
        const std::uint32_t a1 = ld16(g8 + 4u) | (ld16(g8 + 6u) << 16);
        const std::int32_t ls = 2 * static_cast<std::int32_t>(a1 >> 28) + 1;
        const std::uint32_t sv = a1 >> (14u * h);
        std::int32_t sumi = 0;
        for (std::uint32_t l = 0; l < 2; ++l) {
            const std::uint32_t idx = (qs >> (8u * l)) & 0xFFu;
            const std::uint32_t c = (kIq2xxsGrid[idx >> 1] >> (16u * (idx & 1u))) & 0xFFFFu;
            std::uint32_t s7 = (sv >> (7u * l)) & 127u, p = s7 ^ (s7 >> 4);
            p ^= p >> 2;
            p ^= p >> 1;
            const std::uint32_t s8 = s7 | ((p & 1u) << 7);
            for (std::uint32_t j = 0; j < 8; ++j) {
                const std::uint32_t b = (c >> (2u * j)) & 3u;
                const std::int32_t mag = static_cast<std::int32_t>(8u + 17u * b + (b >> 1));
                sumi += (((s8 >> j) & 1u) ? -mag : mag) * qx[e + 8u * l + j];
            }
        }
        return ls * sumi;
    }
    // IQ4_XS, ggml_d3d12_mmid_iq4_xs.hlsl
    static constexpr std::int32_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                            1,    13,   25,  38,  53,  69,  89,  113};
    const std::uint32_t ib = itid >> 1, h = itid & 1u, e_lo = 32u * ib + 8u * h;
    const std::uint32_t hdr0 = ld32(bb), hdr1 = ld32(bb + 4);
    const std::uint32_t qa = ld32(bb + 8u + 16u * ib + 8u * h);
    const std::uint32_t qb = ld32(bb + 12u + 16u * ib + 8u * h);
    const std::int32_t ls = static_cast<std::int32_t>(((hdr1 >> (4u * ib)) & 0xFu) |
                                                      (((hdr0 >> (16u + 2u * ib)) & 3u) << 4)) -
                            32;
    std::int32_t sumi = 0;
    for (std::uint32_t i = 0; i < 4; ++i) {
        const std::uint32_t s = 8u * i;
        sumi += kv[(qa >> s) & 0xFu] * qx[e_lo + i] + kv[(qb >> s) & 0xFu] * qx[e_lo + 4 + i] +
                kv[(qa >> (s + 4u)) & 0xFu] * qx[e_lo + 16 + i] +
                kv[(qb >> (s + 4u)) & 0xFu] * qx[e_lo + 20 + i];
    }
    return ls * sumi;
}

} // namespace d3d12_mmid_detail

// The kernel for expert type t on the host: per group and thread the same
// blocks in the same order, the same float operations and the same tree
// reduction as the HLSL, IQ3_S / IQ2_S with the geometry in p.pad0
// (ggml_d3d12_geom.hlsli). w, ids and xq are the t0, t1 and u1 buffers, y the
// u0 buffer (rows past n are never written). threads is the blob's NUM_THREADS
// (64 or 128).
inline void d3d12_mmid_emulate(ggml_type t, const D3d12MmidParams& p, std::uint32_t groups_x,
                               std::uint32_t groups_y, std::uint32_t threads, const std::uint8_t* w,
                               const std::uint8_t* ids, const std::uint8_t* xq, float* y) {
    constexpr std::uint32_t rows = kD3d12MmvRows;
    if (!d3d12_mmid_type_supported(t))
        return;
    if (threads != static_cast<std::uint32_t>(kD3d12MmvThreadsShort) &&
        threads != static_cast<std::uint32_t>(kD3d12MmvThreadsLong))
        return;
    if (p.n == 0 || p.n_ids == 0)
        return;
    const std::uint32_t block_bytes = static_cast<std::uint32_t>(ggml_type_size(t));
    const std::uint32_t in_flight = threads / 16u;
    const bool geo = d3d12_geom_type(t);
    const std::uint32_t rsl = geo ? std::clamp(p.pad0 & 0xffu, 1u, in_flight) : 1u;
    const std::uint32_t rt = geo ? std::max((p.pad0 >> 8) & 0xffu, 1u) : 1u;
    const std::uint32_t cl = in_flight / rsl; // chunk lanes per row set
    const std::uint32_t seg = cl * 16u;       // threads per row set
    float red[rows][kD3d12MmvThreadsLong];
    for (std::uint32_t gy = 0; gy < groups_y; ++gy) {
        const std::uint32_t token = gy / p.n_ids;
        const std::uint32_t slot_i = gy - token * p.n_ids;
        const std::uint32_t wb =
            d3d12_mmid_detail::ld32(ids + token * p.ids_stride + 4u * slot_i) * p.expert_stride;
        const std::uint32_t xc = token * p.x_stride + slot_i * p.x_col_stride;
        const std::uint32_t ybase = token * p.y_stride + slot_i * p.y_expert_stride;
        for (std::uint32_t gx = 0; gx < groups_x; ++gx)
            for (std::uint32_t tile = 0; tile < rt; ++tile) {
                for (std::uint32_t tid = 0; tid < threads; ++tid) {
                    const std::uint32_t itid = tid & 15u;
                    const std::uint32_t rs = (tid >> 4) / cl, ix = (tid >> 4) - rs * cl;
                    const std::uint32_t row0 = ((gx * rt + tile) * rsl + rs) * rows;
                    float acc[rows] = {};
                    for (std::uint32_t blk = ix; blk < p.nchunk; blk += cl) {
                        const std::uint8_t* xb = xq + xc + blk * 292u;
                        float dy;
                        std::memcpy(&dy, xb, 4);
                        const auto* qx = reinterpret_cast<const std::int8_t*>(xb + 4);
                        for (std::uint32_t r = 0; r < rows; ++r) {
                            const std::uint32_t row = std::min(row0 + r, p.n - 1u);
                            float d;
                            const std::int32_t lsumi = d3d12_mmid_detail::block_lsumi(
                                t, w + wb + row * p.w_row_bytes + blk * block_bytes, itid, qx, &d);
                            acc[r] += (d * dy) * static_cast<float>(lsumi);
                        }
                    }
                    for (std::uint32_t r = 0; r < rows; ++r)
                        red[r][tid] = acc[r];
                }
                for (std::uint32_t base = 0; base < threads; base += seg) {
                    for (std::uint32_t stride = seg / 2u; stride > 0u; stride >>= 1u)
                        for (std::uint32_t t2 = base; t2 < base + stride; ++t2)
                            for (std::uint32_t r = 0; r < rows; ++r)
                                red[r][t2] += red[r][t2 + stride];
                    const std::uint32_t row0 = ((gx * rt + tile) * rsl + base / seg) * rows;
                    for (std::uint32_t r = 0; r < rows; ++r)
                        if (row0 + r < p.n)
                            y[ybase + row0 + r] = red[r][base];
                }
            }
    }
}

} // namespace xllama
