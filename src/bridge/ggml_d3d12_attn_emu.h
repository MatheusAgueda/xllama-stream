// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The full-attention layers of qwen35moe on the d3d12 backend
// (shaders/ggml_d3d12_attn.hlsl): the nodes each kernel takes, their
// constants and grids, and a host emulation of each kernel (tests compare it
// with ggml's CPU ops; a selftest can compare it with the GPU).
//
// One layer with flash attention (qwen35moe.cpp build_layer_attn,
// llama-graph.cpp build_attn / build_attn_mha, llama-kv-cache.cpp cpy_k /
// cpy_v / get_k / get_v), after the q / k / v matmuls and the q / k norms:
//   ROPE(Qcur, pos), ROPE(Kcur, pos), IMROPE              KERNEL 1, one each
//   SET_ROWS(k cache, Kcur, k_idxs), SET_ROWS(v cache, Vcur, v_idxs)
//                                                         KERNEL 2, one each
//   FLASH_ATTN_EXT(permute(Q), k view, v view, kq_mask)   KERNEL 3, and
//                                                         KERNEL 4 when split
// The plans take more than qwen35moe needs: NORMAL, NEOX and MROPE ropes
// with n_offs, freq factors and YaRN; set_rows into f32; head sizes 8..256
// and 1, 2, 4, 8 or 16 q heads per kv head.
//
// Each d3d12_attn_*_plan(node, &plan) is false unless the kernel takes the
// node as it is; a plan of an empty node has no groups (nothing to dispatch).
// The params structs are the root constants in order (copy them to Op::c).
// Offsets are 32-bit, as in the shader; the plans check that they fit.

#pragma once

#include "ggml_d3d12_moe_emu.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace xllama {

inline constexpr std::uint32_t kD3d12AttnThreads = 256;     // numthreads of every kernel
inline constexpr std::uint32_t kD3d12AttnMaxHead = 256;     // KERNEL 3 head size, at most
inline constexpr std::uint32_t kD3d12AttnMaxGroup = 16;     // KERNEL 3 q heads per kv head, at most
inline constexpr std::uint32_t kD3d12AttnPartialHeader = 4; // M, S, 0, 0 before a partial's output

// KERNEL 1 rotation modes (c1 bits 16..23).
inline constexpr std::uint32_t kD3d12RopeNormal = 0;
inline constexpr std::uint32_t kD3d12RopeNeox = 1;
inline constexpr std::uint32_t kD3d12RopeMrope = 2;
inline constexpr std::uint32_t kD3d12RopeImrope = 3;
inline constexpr std::int32_t kD3d12RopeHighNegInf = -32768; // corr dims high = -inf

// KERNEL 2 flags.
inline constexpr std::uint32_t kD3d12SetRowsF16 = 1;
inline constexpr std::uint32_t kD3d12SetRowsI64 = 2;

// KERNEL 3 flags (c12, above n_head).
inline constexpr std::uint32_t kD3d12FaMask = 1u << 16;
inline constexpr std::uint32_t kD3d12FaDirect = 1u << 17;

static_assert(kD3d12AttnThreads == kD3d12MoeThreads, "grids are sized like the moe kernels'");

// KERNEL 1, rope. u1 = src (may be dst), u2 = positions, t0 = freq factors
// (unread without them), u0 = dst. Grid (groups_x).
struct D3d12AttnRopeParams {
    std::uint32_t ne0_dims = 0;  // ne0 | n_dims << 16
    std::uint32_t offs_mode = 0; // n_offs | mode << 16 | (freq factors) << 24
    std::uint32_t ne1_ne2 = 0;   // ne1 | ne2 << 16
    std::uint32_t sections = 0;  // the four mrope sections, a byte each
    std::uint32_t src_row_stride = 0;
    std::uint32_t src_plane_stride = 0;
    std::uint32_t dst_row_stride = 0;
    std::uint32_t dst_plane_stride = 0;
    std::uint32_t theta_scale_bits = 0; // powf(freq_base, -2 / n_dims)
    std::uint32_t freq_scale_bits = 0;
    std::uint32_t mscale_bits = 0; // attn_factor, with YaRN's magnitude scale when ext != 0
    std::uint32_t ext_factor_bits = 0;
    std::uint32_t corr_dims = 0; // low | high << 16 (int16, kD3d12RopeHighNegInf); 0 if ext == 0
};
static_assert(sizeof(D3d12AttnRopeParams) == 13 * sizeof(std::uint32_t), "13 root constants");

struct D3d12AttnRopePlan {
    D3d12AttnRopeParams params;
    std::uint32_t groups_x = 0;
    const ggml_tensor* src = nullptr;
    const ggml_tensor* pos = nullptr;
    const ggml_tensor* ff = nullptr; // null without freq factors
};

// KERNEL 2, set_rows. u1 = src, u2 = idx, u0 = the node's data (the cache).
// Grid (groups_x, rows).
struct D3d12AttnSetRowsParams {
    std::uint32_t ne0 = 0;
    std::uint32_t src_row_stride = 0; // floats
    std::uint32_t dst_row_stride = 0; // dst elements
    std::uint32_t idx_stride = 0;     // 32-bit words
    std::uint32_t flags = 0;          // kD3d12SetRowsF16 | kD3d12SetRowsI64
    std::uint32_t dst_rows = 0;       // an index at or past it is skipped
    std::uint32_t rows = 0;
};

struct D3d12AttnSetRowsPlan {
    D3d12AttnSetRowsParams params;
    std::uint32_t groups_x = 0;
    std::uint32_t groups_y = 0;
    const ggml_tensor* src = nullptr;
    const ggml_tensor* idx = nullptr;
};

// KERNEL 3, flash attention. u1 = q, t0 = k, t1 = v, u2 = mask (unread
// without one), u3 = partials (unread when direct), u0 = dst. Grid (chunks,
// kv heads, tokens).
struct D3d12AttnFaParams {
    std::uint32_t head_group = 0; // head size | q heads per kv head << 16
    std::uint32_t n_kv = 0;
    std::uint32_t tokens_chunks = 0;  // N | n_chunks << 16
    std::uint32_t chunk_len = 0;      // kv positions per chunk, a multiple of 256 / G
    std::uint32_t q_token_stride = 0; // floats
    std::uint32_t q_head_stride = 0;
    std::uint32_t k_cell_stride = 0; // 32-bit words
    std::uint32_t k_head_stride = 0;
    std::uint32_t v_cell_stride = 0;
    std::uint32_t v_head_stride = 0;
    std::uint32_t mask_row_stride = 0; // halves
    std::uint32_t scale_bits = 0;
    std::uint32_t heads_flags = 0; // n_head | kD3d12FaMask | kD3d12FaDirect
};
static_assert(sizeof(D3d12AttnFaParams) == 13 * sizeof(std::uint32_t), "13 root constants");

// KERNEL 4, the merge of the chunks. u1 = partials, u0 = dst. Grid (n_head, N).
struct D3d12AttnFaReduceParams {
    std::uint32_t head = 0; // head size
    std::uint32_t n_chunks = 0;
    std::uint32_t n_head = 0;
};

struct D3d12AttnFaPlan {
    D3d12AttnFaParams params;
    std::uint32_t groups_x = 0; // chunks
    std::uint32_t groups_y = 0; // kv heads
    std::uint32_t groups_z = 0; // tokens
    // Floats of u3 when the kv range is split (KERNEL 4 then runs after
    // KERNEL 3, a UAV barrier between), 0 when KERNEL 3 writes dst itself.
    std::uint64_t scratch_floats = 0;
    D3d12AttnFaReduceParams reduce;
    std::uint32_t reduce_groups_x = 0; // n_head
    std::uint32_t reduce_groups_y = 0; // N
    const ggml_tensor* q = nullptr;
    const ggml_tensor* k = nullptr;
    const ggml_tensor* v = nullptr;
    const ggml_tensor* mask = nullptr; // null without one
};

namespace attn_detail {

using moe_detail::f32;
using moe_detail::fits32;

inline std::uint32_t bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline float from_bits(std::uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// One rounding per operation: the volatile store keeps the compiler from
// contracting into an fma, as `precise` does for dxc.
inline float add_rn(float a, float b) {
    volatile float r = a + b;
    return r;
}
inline float sub_rn(float a, float b) {
    volatile float r = a - b;
    return r;
}
inline float mul_rn(float a, float b) {
    volatile float r = a * b;
    return r;
}
inline float div_rn(float a, float b) {
    volatile float r = a / b;
    return r;
}

inline bool neg_inf(float x) {
    return bits(x) == 0xff800000u;
}

inline float fmax_gpu(float a, float b) { // HLSL max without NaNs
    return a > b ? a : b;
}

inline bool f16(const ggml_tensor* t) {
    return t && t->type == GGML_TYPE_F16 && t->nb[0] == sizeof(std::uint16_t);
}

// nb bytes as a whole number of `unit`-byte elements below 2^32 bytes.
inline bool units(std::size_t nb, std::size_t unit, std::uint64_t* out) {
    if (nb % unit != 0 || nb > std::numeric_limits<std::uint32_t>::max())
        return false;
    *out = nb / unit;
    return true;
}

// The 32-bit byte address of the element past the last one touched.
inline bool bytes_fit(std::uint64_t elems, std::uint64_t elem_size) {
    return elems <= std::numeric_limits<std::uint32_t>::max() / elem_size;
}

// The position set (0 t, 1 h, 2 w, 3 e) of rotated pair j in an mrope.
inline std::uint32_t rope_sector_set(std::uint32_t mode, std::uint32_t sections, std::uint32_t j) {
    const std::uint32_t s0 = sections & 0xffu, s1 = (sections >> 8) & 0xffu;
    const std::uint32_t s2 = (sections >> 16) & 0xffu, s3 = sections >> 24;
    const std::uint32_t sector = j % (s0 + s1 + s2 + s3);
    if (mode == kD3d12RopeImrope) {
        if (sector % 3 == 1 && sector < 3 * s1)
            return 1;
        if (sector % 3 == 2 && sector < 3 * s2)
            return 2;
        if (sector % 3 == 0 && sector < 3 * s0)
            return 0;
        return 3;
    }
    const std::uint32_t sec_w = s0 + s1;
    if (sector >= s0 && sector < sec_w)
        return 1;
    if (sector >= sec_w && sector < sec_w + s2)
        return 2;
    if (sector >= sec_w + s2)
        return 3;
    return 0;
}

// D3D's float to uint: NaN 0, saturated.
inline std::uint32_t ftou(float f) {
    if (!(f >= 0.0f))
        return 0;
    if (f >= 4294967296.0f)
        return std::numeric_limits<std::uint32_t>::max();
    return static_cast<std::uint32_t>(f);
}

} // namespace attn_detail

// The shader's sin and cos, op for op (see attn_sincos in the .hlsl).
inline void d3d12_attn_sincos(float th, float* sn, float* cs) {
    using namespace attn_detail;
    const float x = std::fabs(th);
    std::uint32_t j = ftou(mul_rn(x, from_bits(0x3fa2f983u)));
    float y = static_cast<float>(j);
    if ((j & 1u) != 0) {
        j += 1;
        y = add_rn(y, 1.0f);
    }
    j &= 7u;
    float r = x;
    for (std::uint32_t c : {0x3f480000u, 0x3b840000u, 0x38f80000u, 0x35d40000u, 0x320885a3u})
        r = sub_rn(r, mul_rn(y, from_bits(c)));
    const float z = mul_rn(r, r);
    float t = mul_rn(from_bits(0xb94ca1f9u), z);
    t = add_rn(t, from_bits(0x3c08839eu));
    t = mul_rn(t, z);
    t = add_rn(t, from_bits(0xbe2aaaa3u));
    t = mul_rn(t, z);
    t = mul_rn(t, r);
    const float ps = add_rn(t, r);
    float u = mul_rn(from_bits(0x37ccf5ceu), z);
    u = add_rn(u, from_bits(0xbab6061au));
    u = mul_rn(u, z);
    u = add_rn(u, from_bits(0x3d2aaaa5u));
    u = mul_rn(u, z);
    u = mul_rn(u, z);
    const float pc = add_rn(sub_rn(u, mul_rn(0.5f, z)), 1.0f);
    float s = 0.0f, c = 0.0f;
    if (j == 0) {
        s = ps;
        c = pc;
    } else if (j == 2) {
        s = pc;
        c = -ps;
    } else if (j == 4) {
        s = -ps;
        c = -pc;
    } else {
        s = -pc;
        c = ps;
    }
    *sn = th < 0.0f ? -s : s;
    *cs = c;
}

// The shader's f32 -> f16, round to nearest even (as F16C).
inline std::uint16_t d3d12_attn_f32_to_f16(float v) {
    using namespace attn_detail;
    std::uint32_t x = bits(v);
    const std::uint32_t sign = x & 0x80000000u;
    x ^= sign;
    std::uint32_t o = 0;
    if (x >= 0x47800000u) {
        o = x > 0x7f800000u ? 0x7e00u : 0x7c00u;
    } else if (x < 0x38800000u) {
        o = bits(add_rn(from_bits(x), 0.5f)) - 0x3f000000u;
    } else {
        const std::uint32_t odd = (x >> 13) & 1u;
        x += 0xc8000fffu;
        x += odd;
        o = x >> 13;
    }
    return static_cast<std::uint16_t>(o | (sign >> 16));
}

// f16tof32, exact.
inline float d3d12_attn_f16_to_f32(std::uint16_t h) {
    using namespace attn_detail;
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t ex = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    if (ex == 0) {
        const float f = static_cast<float>(man) * 5.9604644775390625e-8f; // man * 2^-24
        return from_bits(bits(f) | sign);
    }
    if (ex == 31)
        return from_bits(sign | 0x7f800000u | (man << 13));
    return from_bits(sign | ((ex + 112u) << 23) | (man << 13));
}

// ROPE of f32 by NORMAL, NEOX, MROPE or IMROPE (not VISION), with or
// without freq factors, any n_offs, in place or not.
inline bool d3d12_attn_rope_plan(const ggml_tensor* node, D3d12AttnRopePlan* plan) {
    using namespace attn_detail;
    if (!node || !plan || node->op != GGML_OP_ROPE || !f32(node))
        return false;
    D3d12AttnRopePlan p;
    p.src = node->src[0];
    p.pos = node->src[1];
    p.ff = node->src[2];
    if (!f32(p.src) || !ggml_are_same_shape(node, p.src) || node->ne[3] != 1 || !p.pos ||
        p.pos->type != GGML_TYPE_I32 || p.pos->nb[0] != sizeof(std::int32_t))
        return false;
    const std::int32_t* op = node->op_params;
    const std::int32_t n_dims = op[1], mode = op[2], n_ctx_orig = op[4], n_offs = op[15];
    float freq_base = 0, freq_scale = 0, ext_factor = 0, attn_factor = 0, beta_fast = 0,
          beta_slow = 0;
    std::memcpy(&freq_base, op + 5, sizeof(float));
    std::memcpy(&freq_scale, op + 6, sizeof(float));
    std::memcpy(&ext_factor, op + 7, sizeof(float));
    std::memcpy(&attn_factor, op + 8, sizeof(float));
    std::memcpy(&beta_fast, op + 9, sizeof(float));
    std::memcpy(&beta_slow, op + 10, sizeof(float));
    std::uint32_t m = 0;
    switch (mode) {
    case GGML_ROPE_TYPE_NORMAL:
        m = kD3d12RopeNormal;
        break;
    case GGML_ROPE_TYPE_NEOX:
        m = kD3d12RopeNeox;
        break;
    case GGML_ROPE_TYPE_MROPE:
        m = kD3d12RopeMrope;
        break;
    case GGML_ROPE_TYPE_IMROPE:
        m = kD3d12RopeImrope;
        break;
    default:
        return false;
    }
    const std::int64_t ne0 = node->ne[0], ne1 = node->ne[1], ne2 = node->ne[2];
    if (ne0 % 2 != 0 || ne0 >= (1 << 16) || ne1 >= (1 << 16) || ne2 >= (1 << 16) || n_dims <= 0 ||
        n_dims % 2 != 0 || n_offs < 0 || n_offs % 2 != 0 || n_offs + n_dims > ne0)
        return false;
    std::uint32_t sections = 0;
    if (m >= kD3d12RopeMrope) {
        std::int32_t sum = 0;
        for (int k = 0; k < GGML_MROPE_SECTIONS; ++k) {
            if (op[11 + k] < 0 || op[11 + k] > 255)
                return false;
            sum += op[11 + k];
            sections |= static_cast<std::uint32_t>(op[11 + k]) << (8 * k);
        }
        if (sum <= 0 || sum > ne0 || p.pos->ne[0] != 4 * ne2)
            return false;
    } else if (p.pos->ne[0] != ne2) {
        return false;
    }
    if (p.ff && (!f32(p.ff) || p.ff->ne[0] < n_dims / 2))
        return false;
    std::uint32_t corr = 0;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float dims[2] = {};
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, dims);
        const float low = dims[0], high = dims[1];
        if (!(low >= 0.0f && low <= 32767.0f) || low != std::floor(low))
            return false;
        std::int32_t hi = 0;
        if (std::isinf(high) && high < 0.0f)
            hi = kD3d12RopeHighNegInf;
        else if (high >= -32767.0f && high <= 32767.0f && high == std::floor(high))
            hi = static_cast<std::int32_t>(high);
        else
            return false;
        corr = static_cast<std::uint32_t>(low) | (static_cast<std::uint32_t>(hi) << 16);
        // ggml-cpu rope_yarn: mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale)
        mscale = mul_rn(mscale, add_rn(1.0f, mul_rn(0.1f, std::log(div_rn(1.0f, freq_scale)))));
    }
    if (ggml_is_empty(node)) {
        *plan = p;
        return true;
    }
    std::uint64_t ss1 = 0, ss2 = 0, ds1 = 0, ds2 = 0;
    if (!units(p.src->nb[1], sizeof(float), &ss1) || !units(p.src->nb[2], sizeof(float), &ss2) ||
        !units(node->nb[1], sizeof(float), &ds1) || !units(node->nb[2], sizeof(float), &ds2))
        return false;
    const std::uint64_t last1 = static_cast<std::uint64_t>(ne1 - 1),
                        last2 = static_cast<std::uint64_t>(ne2 - 1);
    const std::uint64_t pairs =
        static_cast<std::uint64_t>(ne0 / 2) * static_cast<std::uint64_t>(ne1) * ne2;
    if (!fits32(last2 * ss2 + last1 * ss1 + ne0) || !fits32(last2 * ds2 + last1 * ds1 + ne0) ||
        pairs > static_cast<std::uint64_t>(kD3d12MaxGroups) * kD3d12AttnThreads)
        return false;
    p.groups_x = static_cast<std::uint32_t>((pairs + kD3d12AttnThreads - 1) / kD3d12AttnThreads);
    p.params.ne0_dims =
        static_cast<std::uint32_t>(ne0) | (static_cast<std::uint32_t>(n_dims) << 16);
    p.params.offs_mode = static_cast<std::uint32_t>(n_offs) | (m << 16) | ((p.ff ? 1u : 0u) << 24);
    p.params.ne1_ne2 = static_cast<std::uint32_t>(ne1) | (static_cast<std::uint32_t>(ne2) << 16);
    p.params.sections = sections;
    p.params.src_row_stride = static_cast<std::uint32_t>(ss1);
    p.params.src_plane_stride = static_cast<std::uint32_t>(ss2);
    p.params.dst_row_stride = static_cast<std::uint32_t>(ds1);
    p.params.dst_plane_stride = static_cast<std::uint32_t>(ds2);
    p.params.theta_scale_bits = bits(powf(freq_base, -2.0f / n_dims));
    p.params.freq_scale_bits = bits(freq_scale);
    p.params.mscale_bits = bits(mscale);
    p.params.ext_factor_bits = bits(ext_factor);
    p.params.corr_dims = corr;
    *plan = p;
    return true;
}

// SET_ROWS of 2-d f32 rows into a 2-d f16 or f32 tensor by a 1-d i32 or i64
// index (one stream: ne2 = ne3 = 1).
inline bool d3d12_attn_set_rows_plan(const ggml_tensor* node, D3d12AttnSetRowsPlan* plan) {
    using namespace attn_detail;
    if (!node || !plan || node->op != GGML_OP_SET_ROWS)
        return false;
    D3d12AttnSetRowsPlan p;
    p.src = node->src[0];
    p.idx = node->src[1];
    const bool to_f16 = f16(node);
    if ((!to_f16 && !f32(node)) || !f32(p.src) || !p.idx)
        return false;
    const bool i64 = p.idx->type == GGML_TYPE_I64;
    if ((!i64 && p.idx->type != GGML_TYPE_I32) ||
        p.idx->nb[0] != (i64 ? sizeof(std::int64_t) : sizeof(std::int32_t)) || p.idx->ne[1] != 1 ||
        p.idx->ne[2] != 1 || p.idx->ne[3] != 1 || p.idx->ne[0] != p.src->ne[1] ||
        p.src->ne[2] != 1 || p.src->ne[3] != 1 || node->ne[0] != p.src->ne[0] || node->ne[2] != 1 ||
        node->ne[3] != 1 || p.src->ne[1] > kD3d12MaxGroups)
        return false;
    const std::int64_t ne0 = node->ne[0], rows = p.src->ne[1];
    if (to_f16 && ne0 % 2 != 0)
        return false;
    if (ne0 == 0 || rows == 0) {
        *plan = p;
        return true;
    }
    const std::size_t esize = to_f16 ? sizeof(std::uint16_t) : sizeof(float);
    std::uint64_t ss = 0, ds = 0;
    if (!units(p.src->nb[1], sizeof(float), &ss) || !units(node->nb[1], esize, &ds) ||
        (to_f16 && ds % 2 != 0) || node->ne[1] > std::numeric_limits<std::uint32_t>::max())
        return false;
    const std::uint64_t w = static_cast<std::uint64_t>(ne0);
    const std::uint64_t r = static_cast<std::uint64_t>(rows);
    const std::uint64_t idx_words = i64 ? 2 : 1;
    if (!fits32((r - 1) * ss + w) ||
        !bytes_fit((static_cast<std::uint64_t>(node->ne[1]) - 1) * ds + w, esize) ||
        !fits32(r * idx_words))
        return false;
    const std::uint64_t per = to_f16 ? 2 : 1;
    const std::uint64_t gx = (w + kD3d12AttnThreads * per - 1) / (kD3d12AttnThreads * per);
    if (gx > kD3d12MaxGroups)
        return false;
    p.groups_x = static_cast<std::uint32_t>(gx);
    p.groups_y = static_cast<std::uint32_t>(rows);
    p.params.ne0 = static_cast<std::uint32_t>(ne0);
    p.params.src_row_stride = static_cast<std::uint32_t>(ss);
    p.params.dst_row_stride = static_cast<std::uint32_t>(ds);
    p.params.idx_stride = static_cast<std::uint32_t>(idx_words);
    p.params.flags = (to_f16 ? kD3d12SetRowsF16 : 0u) | (i64 ? kD3d12SetRowsI64 : 0u);
    p.params.dst_rows = static_cast<std::uint32_t>(node->ne[1]);
    p.params.rows = static_cast<std::uint32_t>(rows);
    *plan = p;
    return true;
}

// FLASH_ATTN_EXT with f32 q, f16 k and v of the same head size (8..256, a
// multiple of 8), an f16 mask or none, no ALiBi, softcap or sinks, any
// precision (accumulation is f32). chunk_tiles = tiles of 256 / G positions
// per chunk; 0 picks: one chunk when there are few tiles or groups enough,
// otherwise about 128 groups over the kv range.
inline bool d3d12_attn_fa_plan(const ggml_tensor* node, D3d12AttnFaPlan* plan,
                               std::uint32_t chunk_tiles = 0) {
    using namespace attn_detail;
    if (!node || !plan || node->op != GGML_OP_FLASH_ATTN_EXT || !f32(node) ||
        !ggml_is_contiguous(node))
        return false;
    D3d12AttnFaPlan p;
    p.q = node->src[0];
    p.k = node->src[1];
    p.v = node->src[2];
    p.mask = node->src[3];
    if (!f32(p.q) || !f16(p.k) || !f16(p.v) || node->src[4])
        return false;
    float scale = 0.0f, max_bias = 0.0f, softcap = 0.0f;
    std::memcpy(&scale, &node->op_params[0], sizeof(float));
    std::memcpy(&max_bias, &node->op_params[1], sizeof(float));
    std::memcpy(&softcap, &node->op_params[2], sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f || !std::isfinite(scale))
        return false;
    const std::int64_t dk = p.q->ne[0], n_tok = p.q->ne[1], n_head = p.q->ne[2];
    const std::int64_t n_kv = p.k->ne[1], n_head_kv = p.k->ne[2];
    if (dk < 8 || dk > kD3d12AttnMaxHead || dk % 8 != 0 || p.k->ne[0] != dk || p.v->ne[0] != dk ||
        p.v->ne[1] != n_kv || p.v->ne[2] != n_head_kv || p.q->ne[3] != 1 || p.k->ne[3] != 1 ||
        p.v->ne[3] != 1 || node->ne[0] != dk || node->ne[1] != n_head || node->ne[2] != n_tok ||
        node->ne[3] != 1 || n_tok < 1 || n_tok > kD3d12MaxGroups || n_head < 1 ||
        n_head >= (1 << 16) || n_head_kv < 1 || n_head_kv > kD3d12MaxGroups ||
        n_head % n_head_kv != 0 || n_kv < 0 || n_kv > (1 << 24))
        return false;
    const std::int64_t grp = n_head / n_head_kv;
    if (grp > kD3d12AttnMaxGroup || (grp & (grp - 1)) != 0)
        return false;
    std::uint64_t qt = 0, qh = 0, kc = 0, kh = 0, vc = 0, vh = 0, mr = 0;
    if (!units(p.q->nb[1], sizeof(float), &qt) || !units(p.q->nb[2], sizeof(float), &qh) ||
        !units(p.k->nb[1], 4, &kc) || !units(p.k->nb[2], 4, &kh) || !units(p.v->nb[1], 4, &vc) ||
        !units(p.v->nb[2], 4, &vh))
        return false;
    const std::uint64_t ud = static_cast<std::uint64_t>(dk);
    const std::uint64_t ut = static_cast<std::uint64_t>(n_tok);
    const std::uint64_t uh = static_cast<std::uint64_t>(n_head);
    const std::uint64_t ukv = static_cast<std::uint64_t>(n_kv);
    const std::uint64_t uhk = static_cast<std::uint64_t>(n_head_kv);
    if (!fits32((ut - 1) * qt + (uh - 1) * qh + ud) || !fits32(ut * uh * ud))
        return false;
    if (n_kv > 0 && (!fits32((ukv - 1) * kc + (uhk - 1) * kh + ud / 2) ||
                     !fits32((ukv - 1) * vc + (uhk - 1) * vh + ud / 2)))
        return false;
    if (p.mask) {
        if (!f16(p.mask) || p.mask->ne[0] < n_kv || p.mask->ne[1] < n_tok || p.mask->ne[2] != 1 ||
            p.mask->ne[3] != 1 || !units(p.mask->nb[1], sizeof(std::uint16_t), &mr) ||
            !fits32(((ut - 1) * mr + ukv) / 2 + 1))
            return false;
    }
    const std::uint64_t pos_per_tile = kD3d12AttnThreads / static_cast<std::uint64_t>(grp);
    const std::uint64_t tiles = (ukv + pos_per_tile - 1) / pos_per_tile;
    std::uint64_t ct = tiles; // tiles per chunk
    if (chunk_tiles > 0) {
        ct = chunk_tiles;
    } else if (tiles > 4 && ut * uhk < 64) {
        const std::uint64_t want = (128 + ut * uhk - 1) / (ut * uhk);
        const std::uint64_t n0 = tiles < want ? tiles : want;
        ct = (tiles + n0 - 1) / n0;
    }
    if (ct == 0)
        ct = 1;
    const std::uint64_t n_chunks = tiles == 0 ? 1 : (tiles + ct - 1) / ct;
    const bool direct = n_chunks == 1;
    if (n_chunks > kD3d12MaxGroups || !fits32(ct * pos_per_tile + ukv))
        return false;
    const std::uint64_t scratch = direct ? 0 : ut * uh * n_chunks * (ud + kD3d12AttnPartialHeader);
    if (!fits32(scratch))
        return false;
    p.groups_x = static_cast<std::uint32_t>(n_chunks);
    p.groups_y = static_cast<std::uint32_t>(n_head_kv);
    p.groups_z = static_cast<std::uint32_t>(n_tok);
    p.scratch_floats = scratch;
    p.params.head_group = static_cast<std::uint32_t>(dk) | (static_cast<std::uint32_t>(grp) << 16);
    p.params.n_kv = static_cast<std::uint32_t>(n_kv);
    p.params.tokens_chunks =
        static_cast<std::uint32_t>(n_tok) | (static_cast<std::uint32_t>(n_chunks) << 16);
    p.params.chunk_len = static_cast<std::uint32_t>(ct * pos_per_tile);
    p.params.q_token_stride = static_cast<std::uint32_t>(qt);
    p.params.q_head_stride = static_cast<std::uint32_t>(qh);
    p.params.k_cell_stride = static_cast<std::uint32_t>(kc);
    p.params.k_head_stride = static_cast<std::uint32_t>(kh);
    p.params.v_cell_stride = static_cast<std::uint32_t>(vc);
    p.params.v_head_stride = static_cast<std::uint32_t>(vh);
    p.params.mask_row_stride = static_cast<std::uint32_t>(mr);
    p.params.scale_bits = bits(scale);
    p.params.heads_flags = static_cast<std::uint32_t>(n_head) | (p.mask ? kD3d12FaMask : 0u) |
                           (direct ? kD3d12FaDirect : 0u);
    if (!direct) {
        p.reduce.head = static_cast<std::uint32_t>(dk);
        p.reduce.n_chunks = static_cast<std::uint32_t>(n_chunks);
        p.reduce.n_head = static_cast<std::uint32_t>(n_head);
        p.reduce_groups_x = static_cast<std::uint32_t>(n_head);
        p.reduce_groups_y = static_cast<std::uint32_t>(n_tok);
    }
    *plan = p;
    return true;
}

// The kernels on the host, in the shader's order. Rope and set_rows round
// once per operation as the shader's `precise` code does; sin and cos are
// the shader's own, so the GPU matches these bit for bit except where a
// division is involved (freq factors, the YaRN ramp: dxc's is 2.5 ulp).
// Flash attention sums in the shader's order but dxc may form mads there and
// exp is the GPU's: compare it within a tolerance. Pointers are the
// bindings' bases.

inline void d3d12_attn_rope_emulate(const D3d12AttnRopeParams& p, std::uint32_t groups_x,
                                    const float* src, const std::int32_t* pos, const float* ff,
                                    float* dst) {
    using namespace attn_detail;
    const std::uint32_t ne0 = p.ne0_dims & 0xffffu, n_dims = p.ne0_dims >> 16;
    const std::uint32_t n_offs = p.offs_mode & 0xffffu, mode = (p.offs_mode >> 16) & 0xffu;
    const bool has_ff = (p.offs_mode >> 24) != 0;
    const std::uint32_t ne1 = p.ne1_ne2 & 0xffffu, ne2 = p.ne1_ne2 >> 16;
    const std::uint32_t half_ne0 = ne0 / 2, half_dims = n_dims / 2;
    const float ts = from_bits(p.theta_scale_bits), fs = from_bits(p.freq_scale_bits);
    const float ms = from_bits(p.mscale_bits), ext = from_bits(p.ext_factor_bits);
    const std::uint64_t total = static_cast<std::uint64_t>(half_ne0) * ne1 * ne2;
    for (std::uint64_t f = 0;
         f < total && f < static_cast<std::uint64_t>(groups_x) * kD3d12AttnThreads; ++f) {
        std::uint32_t i = static_cast<std::uint32_t>(f);
        const std::uint32_t r = i % half_ne0;
        i /= half_ne0;
        const std::uint32_t i1 = i % ne1, i2 = i / ne1;
        const std::uint32_t sb = i2 * p.src_plane_stride + i1 * p.src_row_stride;
        const std::uint32_t db = i2 * p.dst_plane_stride + i1 * p.dst_row_stride;
        if (r >= half_dims) {
            const std::uint32_t e = 2 * (r - half_dims);
            const std::uint32_t i0 = e < n_offs ? e : e + n_dims;
            const float a = src[sb + i0], b = src[sb + i0 + 1];
            dst[db + i0] = a;
            dst[db + i0 + 1] = b;
            continue;
        }
        std::uint32_t pi = i2;
        if (mode >= kD3d12RopeMrope)
            pi = i2 + rope_sector_set(mode, p.sections, r) * ne2;
        float theta = static_cast<float>(pos[pi]);
        for (std::uint32_t k = 0; k < r; ++k)
            theta = mul_rn(theta, ts);
        if (has_ff)
            theta = div_rn(theta, ff[r]);
        const float interp = mul_rn(fs, theta);
        float th = interp;
        if (ext != 0.0f) {
            const float low = static_cast<float>(p.corr_dims & 0xffffu);
            const std::int32_t hi = static_cast<std::int16_t>(p.corr_dims >> 16);
            const float high = hi == kD3d12RopeHighNegInf ? -std::numeric_limits<float>::infinity()
                                                          : static_cast<float>(hi);
            const float den = fmax_gpu(0.001f, sub_rn(high, low));
            const float y = div_rn(sub_rn(static_cast<float>(r), low), den);
            const float ramp = sub_rn(1.0f, std::fmin(1.0f, fmax_gpu(0.0f, y)));
            const float mix = mul_rn(ramp, ext);
            th = add_rn(mul_rn(interp, sub_rn(1.0f, mix)), mul_rn(theta, mix));
        }
        float sn = 0.0f, cs = 0.0f;
        d3d12_attn_sincos(th, &sn, &cs);
        const float c = mul_rn(cs, ms), s = mul_rn(sn, ms);
        std::uint32_t x0i = 0, x1i = 0;
        if (mode == kD3d12RopeNormal) {
            x0i = n_offs + 2 * r;
            x1i = x0i + 1;
        } else {
            x0i = n_offs + r;
            x1i = x0i + half_dims;
        }
        const float x0 = src[sb + x0i], x1 = src[sb + x1i];
        dst[db + x0i] = sub_rn(mul_rn(x0, c), mul_rn(x1, s));
        dst[db + x1i] = add_rn(mul_rn(x0, s), mul_rn(x1, c));
    }
}

// idx: the index tensor's 32-bit words (an i64 is low, high). dst: the
// cache's base, f16 or f32 by the flags.
inline void d3d12_attn_set_rows_emulate(const D3d12AttnSetRowsParams& p, std::uint32_t groups_x,
                                        std::uint32_t groups_y, const float* src,
                                        const std::uint32_t* idx, void* dst) {
    const bool to_f16 = (p.flags & kD3d12SetRowsF16) != 0;
    const bool i64 = (p.flags & kD3d12SetRowsI64) != 0;
    const std::uint32_t per = to_f16 ? 2 : 1;
    auto* out = static_cast<unsigned char*>(dst);
    for (std::uint32_t r = 0; r < groups_y; ++r) {
        const std::uint32_t ib = r * p.idx_stride;
        const std::uint32_t di = idx[ib];
        if (di >= p.dst_rows || (i64 && idx[ib + 1] != 0))
            continue;
        for (std::uint32_t t = 0; t < groups_x * kD3d12AttnThreads; ++t) {
            const std::uint32_t c = t * per;
            if (c >= p.ne0)
                break;
            const std::uint64_t de = static_cast<std::uint64_t>(di) * p.dst_row_stride + c;
            const float* s = src + static_cast<std::uint64_t>(r) * p.src_row_stride + c;
            if (to_f16) {
                const std::uint16_t h[2] = {d3d12_attn_f32_to_f16(s[0]),
                                            d3d12_attn_f32_to_f16(s[1])};
                std::memcpy(out + de * sizeof(std::uint16_t), h, sizeof(h));
            } else {
                std::memcpy(out + de * sizeof(float), s, sizeof(float));
            }
        }
    }
}

// k, v, mask: halves from the bindings' bases.
inline void d3d12_attn_fa_emulate(const D3d12AttnFaParams& p, std::uint32_t groups_x,
                                  std::uint32_t groups_y, std::uint32_t groups_z, const float* q,
                                  const std::uint16_t* k, const std::uint16_t* v,
                                  const std::uint16_t* mask, float* partials, float* dst) {
    using namespace attn_detail;
    const std::uint32_t dk = p.head_group & 0xffffu, grp = p.head_group >> 16;
    const std::uint32_t tile_len = kD3d12AttnThreads / grp;
    const std::uint32_t n_chunks = p.tokens_chunks >> 16;
    const std::uint32_t n_head = p.heads_flags & 0xffffu;
    const bool has_mask = (p.heads_flags & kD3d12FaMask) != 0;
    const bool direct = (p.heads_flags & kD3d12FaDirect) != 0;
    const float scale = from_bits(p.scale_bits);
    const float ninf = -std::numeric_limits<float>::infinity();
    std::vector<float> qs(static_cast<std::size_t>(grp) * dk);
    std::vector<float> acc(static_cast<std::size_t>(grp) * dk);
    std::vector<float> sc(kD3d12AttnThreads), wt(kD3d12AttnThreads);
    std::vector<float> m_run(grp), s_run(grp), ms(grp);
    for (std::uint32_t iq1 = 0; iq1 < groups_z; ++iq1)
        for (std::uint32_t hkv = 0; hkv < groups_y; ++hkv)
            for (std::uint32_t chunk = 0; chunk < groups_x; ++chunk) {
                const std::uint32_t begin = chunk * p.chunk_len;
                const std::uint32_t end =
                    begin + p.chunk_len < p.n_kv ? begin + p.chunk_len : p.n_kv;
                for (std::uint32_t g = 0; g < grp; ++g) {
                    for (std::uint32_t e = 0; e < dk; ++e)
                        qs[g * dk + e] =
                            q[iq1 * p.q_token_stride + (hkv * grp + g) * p.q_head_stride + e];
                    m_run[g] = ninf;
                    s_run[g] = 0.0f;
                }
                std::fill(acc.begin(), acc.end(), 0.0f);
                for (std::uint32_t tile = begin; tile < end; tile += tile_len) {
                    const std::uint32_t np = end - tile < tile_len ? end - tile : tile_len;
                    for (std::uint32_t g = 0; g < grp; ++g)
                        for (std::uint32_t pp = 0; pp < tile_len; ++pp) {
                            float s = ninf;
                            if (pp < np) {
                                const std::uint32_t pos = tile + pp;
                                float m = 0.0f;
                                if (has_mask)
                                    m = d3d12_attn_f16_to_f32(mask[iq1 * p.mask_row_stride + pos]);
                                if (!neg_inf(m)) {
                                    float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                                    const std::uint16_t* kr =
                                        k + 2 * (pos * p.k_cell_stride + hkv * p.k_head_stride);
                                    for (std::uint32_t e = 0; e < dk; ++e)
                                        a[e % 4] =
                                            add_rn(a[e % 4], mul_rn(d3d12_attn_f16_to_f32(kr[e]),
                                                                    qs[g * dk + e]));
                                    s = mul_rn(add_rn(add_rn(a[0], a[1]), add_rn(a[2], a[3])),
                                               scale);
                                    s = add_rn(s, m);
                                }
                            }
                            sc[g * tile_len + pp] = s;
                        }
                    for (std::uint32_t g = 0; g < grp; ++g) {
                        float mx = ninf;
                        for (std::uint32_t i = 0; i < np; ++i)
                            mx = fmax_gpu(mx, sc[g * tile_len + i]);
                        const float m_new = fmax_gpu(m_run[g], mx);
                        ms[g] = 1.0f;
                        if (bits(m_new) != bits(m_run[g]))
                            ms[g] = neg_inf(m_run[g]) ? 0.0f : std::exp(m_run[g] - m_new);
                        for (std::uint32_t pp = 0; pp < tile_len; ++pp) {
                            const float s = sc[g * tile_len + pp];
                            wt[g * tile_len + pp] = neg_inf(s) ? 0.0f : std::exp(s - m_new);
                        }
                        m_run[g] = m_new;
                    }
                    for (std::uint32_t g = 0; g < grp; ++g)
                        for (std::uint32_t d = 0; d < dk; ++d)
                            acc[g * dk + d] = mul_rn(acc[g * dk + d], ms[g]);
                    for (std::uint32_t i = 0; i < np; ++i) {
                        const std::uint16_t* vr =
                            v + 2 * ((tile + i) * p.v_cell_stride + hkv * p.v_head_stride);
                        for (std::uint32_t d = 0; d < dk; ++d) {
                            const float vv = d3d12_attn_f16_to_f32(vr[d]);
                            for (std::uint32_t g = 0; g < grp; ++g) {
                                const float w = wt[g * tile_len + i];
                                if (w != 0.0f)
                                    acc[g * dk + d] = add_rn(acc[g * dk + d], mul_rn(w, vv));
                            }
                        }
                    }
                    for (std::uint32_t g = 0; g < grp; ++g) {
                        s_run[g] = mul_rn(s_run[g], ms[g]);
                        for (std::uint32_t i = 0; i < np; ++i) {
                            const float w = wt[g * tile_len + i];
                            if (w != 0.0f)
                                s_run[g] = add_rn(s_run[g], w);
                        }
                    }
                }
                for (std::uint32_t g = 0; g < grp; ++g) {
                    const std::uint32_t h = hkv * grp + g;
                    if (direct) {
                        const float inv = s_run[g] == 0.0f ? 0.0f : 1.0f / s_run[g];
                        for (std::uint32_t d = 0; d < dk; ++d)
                            dst[(iq1 * n_head + h) * dk + d] =
                                s_run[g] == 0.0f ? 0.0f : mul_rn(acc[g * dk + d], inv);
                    } else {
                        float* pb = partials + static_cast<std::uint64_t>(
                                                   (iq1 * n_head + h) * n_chunks + chunk) *
                                                   (dk + kD3d12AttnPartialHeader);
                        pb[0] = m_run[g];
                        pb[1] = s_run[g];
                        pb[2] = 0.0f;
                        pb[3] = 0.0f;
                        for (std::uint32_t d = 0; d < dk; ++d)
                            pb[kD3d12AttnPartialHeader + d] = acc[g * dk + d];
                    }
                }
            }
}

inline void d3d12_attn_fa_reduce_emulate(const D3d12AttnFaReduceParams& p, std::uint32_t groups_x,
                                         std::uint32_t groups_y, const float* partials,
                                         float* dst) {
    using namespace attn_detail;
    const std::uint32_t dk = p.head;
    for (std::uint32_t iq1 = 0; iq1 < groups_y; ++iq1)
        for (std::uint32_t h = 0; h < groups_x; ++h)
            for (std::uint32_t d = 0; d < dk; ++d) {
                const std::uint32_t row = iq1 * p.n_head + h;
                float mf = -std::numeric_limits<float>::infinity(), s = 0.0f, acc = 0.0f;
                for (std::uint32_t c = 0; c < p.n_chunks; ++c) {
                    const float* pb = partials + static_cast<std::uint64_t>(row * p.n_chunks + c) *
                                                     (dk + kD3d12AttnPartialHeader);
                    if (pb[1] == 0.0f)
                        continue;
                    const float mn = fmax_gpu(mf, pb[0]);
                    const float so = neg_inf(mf) ? 0.0f : std::exp(mf - mn);
                    const float sn = std::exp(pb[0] - mn);
                    acc = add_rn(mul_rn(acc, so), mul_rn(pb[kD3d12AttnPartialHeader + d], sn));
                    s = add_rn(mul_rn(s, so), mul_rn(pb[1], sn));
                    mf = mn;
                }
                dst[row * dk + d] = s == 0.0f ? 0.0f : mul_rn(acc, 1.0f / s);
            }
}

// max |got - ref| / (1 + |ref|) over n floats; infinity on a NaN in one only.
inline double d3d12_attn_max_err(const float* got, const float* ref, std::size_t n) {
    double e = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(got[i]) || std::isnan(ref[i])) {
            if (std::isnan(got[i]) != std::isnan(ref[i]))
                return std::numeric_limits<double>::infinity();
            continue;
        }
        const double d = std::fabs(static_cast<double>(got[i]) - ref[i]) /
                         (1.0 + std::fabs(static_cast<double>(ref[i])));
        if (d > e)
            e = d;
    }
    return e;
}

} // namespace xllama
