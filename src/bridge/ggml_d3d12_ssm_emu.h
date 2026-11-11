// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The linear-attention (gated delta net) layers of qwen35moe on the d3d12
// backend (shaders/ggml_d3d12_ssm.hlsl): the nodes each kernel takes, their
// constants and grids, and a host emulation of each kernel (tests compare it
// with ggml's CPU ops; a selftest can compare it with the GPU).
//
// One layer at decode (qwen35moe.cpp build_layer_attn_linear,
// delta-net-base.cpp build_conv_state / build_delta_net_fused,
// llama-graph.cpp build_rs), after the four input matmuls:
//   beta = SIGMOID(mm)                                    KERNEL 1
//   gate = MUL(SOFTPLUS(ADD(mm, ssm_dt)), ssm_a)          KERNEL 1 (ADD, MUL: elt)
//   build_rs, twice: SCALE(view, 0) (empty unless a state is cleared) KERNEL 1,
//     GET_ROWS(states, s_copy) KERNEL 3, CPY of the extra states (empty at
//     one sequence) KERNEL 2
//   conv_input = CONCAT(conv_states, TRANSPOSE(qkv), 0)   KERNEL 2, twice
//   CPY(conv_state_last, conv cache view)                 KERNEL 2
//   SILU(SSM_CONV(conv_input, conv1d))                    KERNEL 4, one dispatch
//   q, k = SCALE(RMS_NORM(view), 1 / sqrt(128))           elt RMS_NORM, KERNEL 1
//   GATED_DELTA_NET(q, k, v, gate, beta, state)           KERNEL 5
//   CPY(new_state, ssm cache view)                        KERNEL 2
//   MUL(MUL(RMS_NORM(out), ssm_norm), SILU(z))            elt, KERNEL 1
//
// Each d3d12_ssm_*_plan(node, &plan) is false unless the kernel takes the node
// as it is; a plan of an empty node has no groups (nothing to dispatch). The
// params structs are the root constants in order (copy them to Op::c). Offsets
// are 32-bit floats, as in the shader; the plans check that they fit.

#pragma once

#include "ggml_d3d12_moe_emu.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace xllama {

inline constexpr std::uint32_t kD3d12SsmThreads = 256; // numthreads of KERNEL 1..4
inline constexpr std::uint32_t kD3d12GdnThreads = 128; // numthreads of KERNEL 5
inline constexpr std::uint32_t kD3d12GdnSv = 128;      // the S_v KERNEL 5 is built for
inline constexpr std::uint32_t kD3d12GdnRows = 32;     // state rows per KERNEL 5 group
// c3 bit of a KERNEL 5 dispatch that writes the new state to u5 (d3d12_gdn_fuse_plan).
inline constexpr std::uint32_t kD3d12GdnFused = 1u << 17;

// KERNEL 1 ops.
inline constexpr std::uint32_t kD3d12SsmScale = 0;
inline constexpr std::uint32_t kD3d12SsmSigmoid = 1;
inline constexpr std::uint32_t kD3d12SsmSoftplus = 2;
inline constexpr std::uint32_t kD3d12SsmSilu = 3;

static_assert(kD3d12SsmThreads == kD3d12MoeThreads, "moe_detail::groups sizes the grids");

// KERNEL 1, unary. u1 = src (may be dst), u0 = dst. Grid (ceil(ne0 / 256), rows).
struct D3d12SsmUnaryParams {
    std::uint32_t ne0 = 0;
    std::uint32_t src_row_stride = 0;
    std::uint32_t dst_row_stride = 0;
    std::uint32_t op = 0;
    std::uint32_t scale_bits = 0; // kD3d12SsmScale: the float's bits
};

struct D3d12SsmUnaryPlan {
    D3d12SsmUnaryParams params;
    std::uint32_t groups_x = 0;
    std::uint32_t groups_y = 0;
    const ggml_tensor* src = nullptr;
};

// KERNEL 2, copy: a 4-d block, strides in floats. u1 = src, u0 = the node's
// data + dst_offset bytes. Grid (groups_x).
struct D3d12SsmCopyParams {
    std::uint32_t ne[4] = {};
    std::uint32_t src_stride[4] = {};
    std::uint32_t dst_stride[4] = {};
    std::uint32_t pad0 = 0;
};
static_assert(sizeof(D3d12SsmCopyParams) == 13 * sizeof(std::uint32_t), "13 root constants");

struct D3d12SsmCopy {
    D3d12SsmCopyParams params;
    std::uint32_t groups_x = 0;
    const ggml_tensor* src = nullptr;
    std::uint64_t dst_offset = 0;
};

// One block for CPY, CONT and DUP, two for CONCAT (src[0], then src[1]).
struct D3d12SsmCopyPlan {
    std::vector<D3d12SsmCopy> copies;
};

// KERNEL 3, get_rows. u1 = src, u2 = ids, u0 = dst. Grid (ceil(ne0 / 256), rows).
struct D3d12SsmGetRowsParams {
    std::uint32_t ne0 = 0;
    std::uint32_t src_row_stride = 0;
    std::uint32_t dst_row_stride = 0;
    std::uint32_t ids_stride = 0; // in int32
};

struct D3d12SsmGetRowsPlan {
    D3d12SsmGetRowsParams params;
    std::uint32_t groups_x = 0;
    std::uint32_t groups_y = 0;
    const ggml_tensor* src = nullptr;
    const ggml_tensor* ids = nullptr;
};

// KERNEL 4, conv. t0 = w, u1 = x, u0 = dst. Grid (ceil(channels / 256), T, seqs).
struct D3d12SsmConvParams {
    std::uint32_t channels = 0;
    std::uint32_t d_conv = 0;
    std::uint32_t x_channel_stride = 0;
    std::uint32_t x_seq_stride = 0;
    std::uint32_t w_channel_stride = 0;
    std::uint32_t dst_token_stride = 0;
    std::uint32_t dst_seq_stride = 0;
    std::uint32_t silu = 0;
};

struct D3d12SsmConvPlan {
    D3d12SsmConvParams params;
    std::uint32_t groups_x = 0;
    std::uint32_t groups_y = 0;
    std::uint32_t groups_z = 0;
    const ggml_tensor* x = nullptr;
    const ggml_tensor* w = nullptr;
    std::vector<const ggml_tensor*> nodes; // SSM_CONV[, SILU], the output last
};

// KERNEL 5, gated delta net. u1 = q, u2 = k, u3 = v, u4 = g, t0 = beta,
// t1 = state, u0 = dst. Grid (S_v / 32, H, B).
struct D3d12GdnParams {
    std::uint32_t h = 0;          // v heads
    std::uint32_t t = 0;          // tokens per sequence
    std::uint32_t b = 0;          // sequences
    std::uint32_t k_kda = 0;      // K | (g per channel) << 16
    std::uint32_t heads_qk = 0;   // q heads | k heads << 16
    std::uint32_t seq_div_qk = 0; // B / q seqs | B / k seqs << 16
    std::uint32_t q_token_stride = 0;
    std::uint32_t q_seq_stride = 0;
    std::uint32_t k_token_stride = 0;
    std::uint32_t k_seq_stride = 0;
    std::uint32_t v_token_stride = 0;
    std::uint32_t v_seq_stride = 0;
    std::uint32_t scale_bits = 0; // 1 / sqrt(S_v)
};
static_assert(sizeof(D3d12GdnParams) == 13 * sizeof(std::uint32_t), "13 root constants");

struct D3d12GdnPlan {
    D3d12GdnParams params;
    std::uint32_t groups_x = 0; // S_v / 32
    std::uint32_t groups_y = 0; // H
    std::uint32_t groups_z = 0; // B
    const ggml_tensor* q = nullptr;
    const ggml_tensor* k = nullptr;
    const ggml_tensor* v = nullptr;
    const ggml_tensor* g = nullptr;
    const ggml_tensor* beta = nullptr;
    const ggml_tensor* state = nullptr;
};

namespace ssm_detail {

using moe_detail::f32;
using moe_detail::fits32;

inline std::uint32_t bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline bool floats(std::size_t nb, std::uint64_t* out) {
    if (nb % sizeof(float) != 0 || !fits32(nb / sizeof(float)))
        return false;
    *out = nb / sizeof(float);
    return true;
}

inline bool below_limit(std::int64_t v, std::int64_t below) {
    return v >= 1 && v < below;
}

// F32 with nb[0] a float and no negative extent (an empty tensor passes).
inline bool f32_any(const ggml_tensor* t) {
    if (!f32(t))
        return false;
    for (int k = 0; k < GGML_MAX_DIMS; ++k)
        if (t->ne[k] < 0)
            return false;
    return true;
}

// F32 with every stride a whole number of floats, nb[0] included (a copy's
// source, such as the TRANSPOSE that feeds the conv CONCAT).
inline bool f32_strided(const ggml_tensor* t) {
    if (!t || t->type != GGML_TYPE_F32)
        return false;
    for (int k = 0; k < GGML_MAX_DIMS; ++k)
        if (t->ne[k] < 0 || t->nb[k] % sizeof(float) != 0)
            return false;
    return true;
}

// The last float touched, +1, over a block of ne with strides st, fits 32-bit.
inline bool span_fits(const std::uint32_t* ne, const std::uint32_t* st) {
    std::uint64_t last = 0;
    for (int k = 0; k < 4; ++k)
        last += static_cast<std::uint64_t>(ne[k] - 1) * st[k];
    return fits32(last + 1);
}

// A copy of `ne` elements laid out by src_nb (bytes) into one laid out by
// dst_ne / dst_nb: same shape, or one side contiguous and the elements matched
// in row-major order, as ggml_compute_forward_dup does.
inline bool block(const ggml_tensor* src, const std::int64_t* dst_ne, const std::size_t* dst_nb,
                  std::uint64_t dst_offset, D3d12SsmCopy* out) {
    if (!f32_strided(src))
        return false;
    bool same = true;
    std::uint64_t n = 1, n_dst = 1;
    for (int k = 0; k < 4; ++k) {
        same = same && src->ne[k] == dst_ne[k];
        n *= static_cast<std::uint64_t>(src->ne[k]);
        n_dst *= static_cast<std::uint64_t>(dst_ne[k]);
    }
    if (n != n_dst || dst_nb[0] != sizeof(float))
        return false;
    D3d12SsmCopy c;
    c.src = src;
    c.dst_offset = dst_offset;
    if (n == 0) {
        *out = c;
        return true;
    }
    if (n > static_cast<std::uint64_t>(kD3d12MaxGroups) * kD3d12SsmThreads)
        return false;
    auto packed = [](const std::int64_t* ne, int k) {
        std::uint64_t s = 1;
        for (int i = 0; i < k; ++i)
            s *= static_cast<std::uint64_t>(ne[i]);
        return s;
    };
    auto contiguous = [&](const std::int64_t* ne, const std::size_t* nb) {
        for (int k = 0; k < 4; ++k)
            if (ne[k] > 1 && nb[k] != packed(ne, k) * sizeof(float))
                return false;
        return true;
    };
    const bool dst_packed = contiguous(dst_ne, dst_nb);
    if (!same && !dst_packed && !contiguous(src->ne, src->nb))
        return false;
    // Walk the source's shape unless only the source is contiguous; a dim of
    // one element gets stride 0, whatever its nb.
    const std::int64_t* ne = same || dst_packed ? src->ne : dst_ne;
    for (int k = 0; k < 4; ++k) {
        std::uint64_t ss = 0, ds = 0;
        if (ne[k] > 1) {
            if (same) {
                if (!floats(src->nb[k], &ss) || !floats(dst_nb[k], &ds))
                    return false;
            } else if (dst_packed) {
                if (!floats(src->nb[k], &ss))
                    return false;
                ds = packed(src->ne, k);
            } else {
                ss = packed(dst_ne, k);
                if (!floats(dst_nb[k], &ds))
                    return false;
            }
        }
        if (!fits32(ss) || !fits32(ds))
            return false;
        c.params.src_stride[k] = static_cast<std::uint32_t>(ss);
        c.params.dst_stride[k] = static_cast<std::uint32_t>(ds);
    }
    for (int k = 0; k < 4; ++k)
        c.params.ne[k] = static_cast<std::uint32_t>(ne[k]);
    if (!span_fits(c.params.ne, c.params.src_stride) ||
        !span_fits(c.params.ne, c.params.dst_stride))
        return false;
    c.groups_x = static_cast<std::uint32_t>((n + kD3d12SsmThreads - 1) / kD3d12SsmThreads);
    *out = c;
    return true;
}

} // namespace ssm_detail

// SIGMOID, SOFTPLUS, SILU (UNARY) and SCALE without a bias, f32, rows of
// src and dst evenly spaced.
inline bool d3d12_ssm_unary_plan(const ggml_tensor* node, D3d12SsmUnaryPlan* plan) {
    using namespace ssm_detail;
    if (!node || !plan || !f32_any(node))
        return false;
    D3d12SsmUnaryPlan p;
    if (node->op == GGML_OP_UNARY) {
        switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_SIGMOID:
            p.params.op = kD3d12SsmSigmoid;
            break;
        case GGML_UNARY_OP_SOFTPLUS:
            p.params.op = kD3d12SsmSoftplus;
            break;
        case GGML_UNARY_OP_SILU:
            p.params.op = kD3d12SsmSilu;
            break;
        default:
            return false;
        }
    } else if (node->op == GGML_OP_SCALE) {
        float s = 0.0f, b = 0.0f;
        std::memcpy(&s, &node->op_params[0], sizeof(float));
        std::memcpy(&b, &node->op_params[1], sizeof(float));
        if (b != 0.0f)
            return false;
        p.params.op = kD3d12SsmScale;
        p.params.scale_bits = bits(s);
    } else {
        return false;
    }
    p.src = node->src[0];
    if (!f32_any(p.src) || !ggml_are_same_shape(node, p.src))
        return false;
    if (ggml_is_empty(node)) {
        *plan = p;
        return true;
    }
    // Element-wise: both contiguous is one flat row ([1, 32, T, S] beta is
    // then one group, not 32 with a thread each).
    const std::uint64_t n = static_cast<std::uint64_t>(ggml_nelements(node));
    if (ggml_is_contiguous(p.src) && ggml_is_contiguous(node) &&
        n <= static_cast<std::uint64_t>(kD3d12MaxGroups) * kD3d12SsmThreads) {
        p.params.ne0 = static_cast<std::uint32_t>(n);
        p.groups_x = static_cast<std::uint32_t>((n + kD3d12SsmThreads - 1) / kD3d12SsmThreads);
        p.groups_y = 1;
        *plan = p;
        return true;
    }
    std::uint32_t ss = 0, ds = 0;
    const std::uint64_t rows = static_cast<std::uint64_t>(ggml_nrows(node));
    const std::uint64_t w = static_cast<std::uint64_t>(node->ne[0]);
    if (!moe_detail::row_stride(p.src, &ss) || !moe_detail::row_stride(node, &ds) ||
        !moe_detail::rows_fit(rows, ss, w) || !moe_detail::rows_fit(rows, ds, w) ||
        !moe_detail::groups(node->ne[0], static_cast<std::int64_t>(rows), &p.groups_x, &p.groups_y))
        return false;
    p.params.ne0 = static_cast<std::uint32_t>(w);
    p.params.src_row_stride = ss;
    p.params.dst_row_stride = ds;
    *plan = p;
    return true;
}

// CPY / CONT / DUP f32 -> f32, CONCAT of two f32 tensors along any dim.
inline bool d3d12_ssm_copy_plan(const ggml_tensor* node, D3d12SsmCopyPlan* plan) {
    using namespace ssm_detail;
    if (!node || !plan || !f32_any(node))
        return false;
    D3d12SsmCopyPlan p;
    switch (node->op) {
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP: {
        D3d12SsmCopy c;
        if (!block(node->src[0], node->ne, node->nb, 0, &c))
            return false;
        p.copies.push_back(c);
        break;
    }
    case GGML_OP_CONCAT: {
        const std::int32_t dim = node->op_params[0]; // ggml_concat's dim
        const ggml_tensor* a = node->src[0];
        if (dim < 0 || dim >= GGML_MAX_DIMS || !a)
            return false;
        D3d12SsmCopy ca, cb;
        const std::uint64_t off = static_cast<std::uint64_t>(a->ne[dim]) * node->nb[dim];
        if (!block(a, a->ne, node->nb, 0, &ca) ||
            !block(node->src[1], node->src[1]->ne, node->nb, off, &cb))
            return false;
        p.copies.push_back(ca);
        p.copies.push_back(cb);
        break;
    }
    default:
        return false;
    }
    *plan = std::move(p);
    return true;
}

// GET_ROWS of a 2-d f32 tensor by a 1-d int32 index.
inline bool d3d12_ssm_get_rows_plan(const ggml_tensor* node, D3d12SsmGetRowsPlan* plan) {
    using namespace ssm_detail;
    if (!node || !plan || node->op != GGML_OP_GET_ROWS || !f32_any(node))
        return false;
    D3d12SsmGetRowsPlan p;
    p.src = node->src[0];
    p.ids = node->src[1];
    if (!f32_any(p.src) || !p.ids || p.ids->type != GGML_TYPE_I32 ||
        p.ids->nb[0] != sizeof(std::int32_t) || p.src->ne[2] != 1 || p.src->ne[3] != 1 ||
        p.ids->ne[1] != 1 || p.ids->ne[2] != 1 || p.ids->ne[3] != 1 ||
        node->ne[0] != p.src->ne[0] || node->ne[1] != p.ids->ne[0] || node->ne[2] != 1 ||
        node->ne[3] != 1)
        return false;
    if (ggml_is_empty(node)) {
        *plan = p;
        return true;
    }
    std::uint64_t ss = 0, ds = 0;
    if (!floats(p.src->nb[1], &ss) || !floats(node->nb[1], &ds))
        return false;
    const std::uint64_t w = static_cast<std::uint64_t>(node->ne[0]);
    if (!fits32((static_cast<std::uint64_t>(p.src->ne[1]) - 1) * ss + w) ||
        !moe_detail::rows_fit(static_cast<std::uint64_t>(node->ne[1]),
                              static_cast<std::uint32_t>(ds), w) ||
        !moe_detail::groups(node->ne[0], node->ne[1], &p.groups_x, &p.groups_y))
        return false;
    p.params.ne0 = static_cast<std::uint32_t>(w);
    p.params.src_row_stride = static_cast<std::uint32_t>(ss);
    p.params.dst_row_stride = static_cast<std::uint32_t>(ds);
    p.params.ids_stride = 1;
    *plan = p;
    return true;
}

// SSM_CONV, or SILU(SSM_CONV) as one dispatch (silu = 1): the backend checks
// that nothing else reads the SSM_CONV (ggml_can_fuse_subgraph).
inline bool d3d12_ssm_conv_plan(const ggml_tensor* node, D3d12SsmConvPlan* plan) {
    using namespace ssm_detail;
    if (!node || !plan || !f32_any(node))
        return false;
    D3d12SsmConvPlan p;
    const ggml_tensor* conv = node;
    if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SILU) {
        conv = node->src[0];
        if (!conv || conv->op != GGML_OP_SSM_CONV || !f32_any(conv) ||
            !ggml_are_same_shape(conv, node))
            return false;
        p.params.silu = 1;
        p.nodes = {conv, node};
    } else if (node->op == GGML_OP_SSM_CONV) {
        p.nodes = {node};
    } else {
        return false;
    }
    p.x = conv->src[0];
    p.w = conv->src[1];
    if (!f32_any(p.x) || !f32_any(p.w) || p.x->ne[3] != 1 || p.w->ne[2] != 1 || p.w->ne[3] != 1)
        return false;
    const std::int64_t d_conv = p.w->ne[0], ch = p.w->ne[1];
    const std::int64_t n_t = node->ne[1], n_s = node->ne[2];
    if (node->ne[0] != ch || node->ne[3] != 1 || p.x->ne[1] != ch || p.x->ne[2] != n_s ||
        p.x->ne[0] != d_conv - 1 + n_t || p.x->nb[1] != p.x->ne[0] * sizeof(float))
        return false;
    if (ggml_is_empty(node)) {
        *plan = std::move(p);
        return true;
    }
    std::uint64_t xc = 0, xs = 0, wc = 0, dt = 0, ds = 0;
    if (!floats(p.x->nb[1], &xc) || !floats(p.x->nb[2], &xs) || !floats(p.w->nb[1], &wc) ||
        !floats(node->nb[1], &dt) || !floats(node->nb[2], &ds) || !below_limit(d_conv, 1 << 16) ||
        n_t > kD3d12MaxGroups || n_s > kD3d12MaxGroups)
        return false;
    const std::uint64_t c1 = static_cast<std::uint64_t>(ch) - 1;
    if (!fits32(static_cast<std::uint64_t>(n_s - 1) * xs + c1 * xc +
                static_cast<std::uint64_t>(p.x->ne[0])) ||
        !fits32(c1 * wc + static_cast<std::uint64_t>(d_conv)) ||
        !fits32(static_cast<std::uint64_t>(n_s - 1) * ds +
                static_cast<std::uint64_t>(n_t - 1) * dt + c1 + 1))
        return false;
    std::uint32_t gy = 0;
    if (!moe_detail::groups(ch, 1, &p.groups_x, &gy))
        return false;
    p.groups_y = static_cast<std::uint32_t>(n_t);
    p.groups_z = static_cast<std::uint32_t>(n_s);
    p.params.channels = static_cast<std::uint32_t>(ch);
    p.params.d_conv = static_cast<std::uint32_t>(d_conv);
    p.params.x_channel_stride = static_cast<std::uint32_t>(xc);
    p.params.x_seq_stride = static_cast<std::uint32_t>(xs);
    p.params.w_channel_stride = static_cast<std::uint32_t>(wc);
    p.params.dst_token_stride = static_cast<std::uint32_t>(dt);
    p.params.dst_seq_stride = static_cast<std::uint32_t>(ds);
    *plan = std::move(p);
    return true;
}

// GATED_DELTA_NET with S_v = 128: q, k, v with their heads S_v apart, g, beta
// and state contiguous (as ggml requires), K < 65536.
inline bool d3d12_gdn_plan(const ggml_tensor* node, D3d12GdnPlan* plan) {
    using namespace ssm_detail;
    if (!node || !plan || node->op != GGML_OP_GATED_DELTA_NET || !f32_any(node) ||
        !ggml_is_contiguous(node))
        return false;
    D3d12GdnPlan p;
    p.q = node->src[0];
    p.k = node->src[1];
    p.v = node->src[2];
    p.g = node->src[3];
    p.beta = node->src[4];
    p.state = node->src[5];
    for (const ggml_tensor* t : {p.q, p.k, p.v, p.g, p.beta, p.state})
        if (!f32_any(t))
            return false;
    const std::int64_t sv = kD3d12GdnSv;
    const std::int64_t H = p.v->ne[1], T = p.v->ne[2], B = p.v->ne[3];
    const std::int64_t K = node->op_params[0]; // snapshot slots
    const bool kda = p.g->ne[0] == sv;
    if (p.v->ne[0] != sv || p.q->ne[0] != sv || p.k->ne[0] != sv ||
        !below_limit(H, kD3d12MaxGroups + 1) || !below_limit(B, kD3d12MaxGroups + 1) || T < 1 ||
        !below_limit(K, 1 << 16) || !below_limit(p.q->ne[1], 1 << 16) ||
        !below_limit(p.k->ne[1], 1 << 16) || p.q->ne[2] != T || p.k->ne[2] != T ||
        !below_limit(p.q->ne[3], B + 1) || !below_limit(p.k->ne[3], B + 1) || B % p.q->ne[3] != 0 ||
        B % p.k->ne[3] != 0 || B / p.q->ne[3] >= (1 << 16) || B / p.k->ne[3] >= (1 << 16))
        return false;
    for (const ggml_tensor* t : {p.q, p.k, p.v})
        if (t->nb[1] != sv * sizeof(float))
            return false;
    if ((p.g->ne[0] != 1 && !kda) || p.g->ne[1] != H || p.g->ne[2] != T || p.g->ne[3] != B ||
        p.beta->ne[0] != 1 || p.beta->ne[1] != H || p.beta->ne[2] != T || p.beta->ne[3] != B ||
        p.state->ne[0] != sv || p.state->ne[1] != sv || p.state->ne[2] != H ||
        p.state->ne[3] != B || !ggml_is_contiguous(p.g) || !ggml_is_contiguous(p.beta) ||
        !ggml_is_contiguous(p.state))
        return false;
    const std::uint64_t attn = static_cast<std::uint64_t>(sv * H) * T * B;
    const std::uint64_t per_snap = static_cast<std::uint64_t>(sv * sv * H) * B;
    if (static_cast<std::uint64_t>(ggml_nelements(node)) != attn + K * per_snap ||
        !fits32(attn + K * per_snap))
        return false;
    std::uint64_t st[6] = {};
    const ggml_tensor* qkv[3] = {p.q, p.k, p.v};
    for (int i = 0; i < 3; ++i) {
        const ggml_tensor* t = qkv[i];
        if (!floats(t->nb[2], &st[2 * i]) || !floats(t->nb[3], &st[2 * i + 1]) ||
            !fits32(static_cast<std::uint64_t>(t->ne[3] - 1) * st[2 * i + 1] +
                    static_cast<std::uint64_t>(T - 1) * st[2 * i] +
                    static_cast<std::uint64_t>(t->ne[1]) * sv))
            return false;
    }
    p.groups_x = kD3d12GdnSv / kD3d12GdnRows;
    p.groups_y = static_cast<std::uint32_t>(H);
    p.groups_z = static_cast<std::uint32_t>(B);
    p.params.h = static_cast<std::uint32_t>(H);
    p.params.t = static_cast<std::uint32_t>(T);
    p.params.b = static_cast<std::uint32_t>(B);
    p.params.k_kda = static_cast<std::uint32_t>(K) | (kda ? 1u << 16 : 0u);
    p.params.heads_qk = static_cast<std::uint32_t>(p.q->ne[1] | (p.k->ne[1] << 16));
    p.params.seq_div_qk = static_cast<std::uint32_t>((B / p.q->ne[3]) | ((B / p.k->ne[3]) << 16));
    p.params.q_token_stride = static_cast<std::uint32_t>(st[0]);
    p.params.q_seq_stride = static_cast<std::uint32_t>(st[1]);
    p.params.k_token_stride = static_cast<std::uint32_t>(st[2]);
    p.params.k_seq_stride = static_cast<std::uint32_t>(st[3]);
    p.params.v_token_stride = static_cast<std::uint32_t>(st[4]);
    p.params.v_seq_stride = static_cast<std::uint32_t>(st[5]);
    p.params.scale_bits = bits(1.0f / std::sqrt(static_cast<float>(kD3d12GdnSv)));
    *plan = p;
    return true;
}

// The state's round trip through the recurrent cache at one sequence, as
// build_rs and build_delta_net_fused make it: GET_ROWS(states, ids) gathers
// the cache row the GATED_DELTA_NET reads (through RESHAPE / VIEW nodes), and
// a CPY puts the new state, a VIEW of the op's output after its attention
// rows, back into a cache row. KERNEL 5 fused (kD3d12GdnFused) reads the
// gathered row in place and stores the new state into the destination row,
// so neither copy runs: 2 x 2 MiB a layer per token at Qwen3.6-35B-A3B.
// True only when that is the same computation: one sequence, K = 1, one row
// id from a graph input (its value is known before the graph runs), the
// GET_ROWS, the views and the new-state VIEW each read once and no graph
// output, the CPY's destination contiguous in the same cache tensor, and no
// node between the GET_ROWS and the op writing the cache or between the op
// and the CPY reading it. `uses(t)` is how many nodes of the whole graph read
// t (ggml's use counts: a split of it holds only some of them).
struct D3d12GdnFusePlan {
    const ggml_tensor* get_rows = nullptr; // does not run
    const ggml_tensor* cpy = nullptr;      // does not run
    const ggml_tensor* states = nullptr;   // the GET_ROWS source: cache rows of S_v * S_v * H
    const ggml_tensor* ids = nullptr;      // its one row id
    const ggml_tensor* dst = nullptr;      // the CPY destination: the cache row written
};

namespace ssm_detail {

inline const ggml_tensor* view_root(const ggml_tensor* t) {
    while (t->view_src)
        t = t->view_src;
    return t;
}

inline bool view_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

inline bool reads(const ggml_tensor* n, const ggml_tensor* root) {
    for (const ggml_tensor* s : n->src)
        if (s && view_root(s) == root)
            return true;
    return false;
}

inline int src_count(const ggml_tensor* n, const ggml_tensor* t) {
    int r = 0;
    for (const ggml_tensor* s : n->src)
        r += s == t;
    return r;
}

// Index of t among the nodes before `before`, or -1.
inline int node_before(ggml_cgraph* cgraph, const ggml_tensor* t, int before) {
    for (int i = before - 1; i >= 0; --i)
        if (ggml_graph_node(cgraph, i) == t)
            return i;
    return -1;
}

// True when, of the nodes after t (at t_idx), `child` reads t once and every
// other reader is a VIEW of no elements (no data of t; a backend skips empty
// nodes), and t is no graph output. build_rs leaves such views: the extra
// states of a one-sequence ubatch are 0 rows.
template <class Uses>
bool sole_reader(ggml_cgraph* cgraph, int t_idx, const ggml_tensor* t, const ggml_tensor* child,
                 const Uses& uses) {
    if (t->flags & GGML_TENSOR_FLAG_OUTPUT)
        return false;
    const int n_nodes = ggml_graph_n_nodes(cgraph);
    const int want = uses(t);
    int seen = 0, hit = 0;
    for (int j = t_idx + 1; j < n_nodes && seen < want; ++j) {
        const ggml_tensor* r = ggml_graph_node(cgraph, j);
        const int c = src_count(r, t);
        if (c == 0)
            continue;
        seen += c;
        if (r == child)
            hit += c;
        else if (r->op != GGML_OP_VIEW || ggml_nelements(r) != 0)
            return false;
    }
    return seen == want && hit == 1;
}

} // namespace ssm_detail

template <class Uses>
bool d3d12_gdn_fuse_plan(ggml_cgraph* cgraph, int gdn_idx, const Uses& uses,
                         D3d12GdnFusePlan* plan) {
    using namespace ssm_detail;
    const int n_nodes = ggml_graph_n_nodes(cgraph);
    if (!plan || gdn_idx < 0 || gdn_idx >= n_nodes)
        return false;
    const ggml_tensor* node = ggml_graph_node(cgraph, gdn_idx);
    D3d12GdnPlan p;
    if (!d3d12_gdn_plan(node, &p) || (p.params.k_kda & 0xffffu) != 1u || p.params.b != 1u)
        return false;
    auto once = [&](const ggml_tensor* t) {
        return uses(t) == 1 && (t->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
    };
    // From the op's state up to the GET_ROWS, each tensor read by the one below.
    const ggml_tensor* child = node;
    const ggml_tensor* g = p.state;
    int child_idx = gdn_idx, g_idx = -1;
    for (;;) {
        g_idx = node_before(cgraph, g, child_idx);
        if (g_idx < 0 || !ggml_is_contiguous(g) || !sole_reader(cgraph, g_idx, g, child, uses))
            return false;
        if (g->op != GGML_OP_RESHAPE && !(g->op == GGML_OP_VIEW && g->view_offs == 0))
            break;
        child = g;
        child_idx = g_idx;
        g = g->src[0];
    }
    const std::int64_t n = static_cast<std::int64_t>(kD3d12GdnSv) * kD3d12GdnSv * p.params.h;
    if (g->op != GGML_OP_GET_ROWS || g->type != GGML_TYPE_F32 || g->ne[0] != n ||
        ggml_nrows(g) != 1 || (g->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)
        return false;
    const ggml_tensor* states = g->src[0];
    const ggml_tensor* ids = g->src[1];
    if (!states || !ids || states->type != GGML_TYPE_F32 || states->ne[0] != n ||
        states->nb[0] != sizeof(float) || states->ne[2] != 1 || states->ne[3] != 1 ||
        ids->type != GGML_TYPE_I32 || ggml_nelements(ids) != 1 || view_root(ids)->op != GGML_OP_NONE)
        return false;
    const ggml_tensor* cache = view_root(states);
    for (int i = g_idx + 1; i < gdn_idx; ++i) {
        const ggml_tensor* t = ggml_graph_node(cgraph, i);
        if (!view_op(t->op) && ggml_nelements(t) != 0 && view_root(t) == cache)
            return false; // writes the cache after the GET_ROWS read it
    }
    // Every reader of the op is a VIEW: of the attention rows, or the new state.
    const std::size_t attn_bytes = static_cast<std::size_t>(kD3d12GdnSv) * p.params.h *
                                   p.params.t * sizeof(float);
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT)
        return false;
    const ggml_tensor* v = nullptr;
    int v_idx = -1, seen = 0;
    const int want = uses(node);
    for (int j = gdn_idx + 1; j < n_nodes && seen < want; ++j) {
        const ggml_tensor* t = ggml_graph_node(cgraph, j);
        const int r = src_count(t, node);
        if (r == 0)
            continue;
        seen += r;
        if (t->op != GGML_OP_VIEW || t->view_src != node)
            return false;
        if (t->view_offs == attn_bytes && ggml_nelements(t) == n && ggml_is_contiguous(t)) {
            if (v || !once(t))
                return false;
            v = t;
            v_idx = j;
        } else if (t->view_offs + ggml_nbytes(t) > attn_bytes) {
            return false;
        }
    }
    if (!v || seen != want)
        return false;
    const ggml_tensor* cpy = nullptr;
    int c_idx = v_idx + 1;
    while (c_idx < n_nodes && !(ggml_graph_node(cgraph, c_idx)->op == GGML_OP_CPY &&
                                ggml_graph_node(cgraph, c_idx)->src[0] == v))
        ++c_idx;
    if (c_idx == n_nodes)
        return false;
    cpy = ggml_graph_node(cgraph, c_idx);
    if ((cpy->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)
        return false;
    const ggml_tensor* dst = cpy->src[1];
    if (!dst || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
        ggml_nelements(dst) != n || view_root(dst) != cache)
        return false;
    for (int i = gdn_idx + 1; i < c_idx; ++i) {
        const ggml_tensor* t = ggml_graph_node(cgraph, i);
        if (!view_op(t->op) && ggml_nelements(t) != 0 && (reads(t, cache) || view_root(t) == cache))
            return false; // would see the new state before the CPY put it there
    }
    plan->get_rows = g;
    plan->cpy = cpy;
    plan->states = states;
    plan->ids = ids;
    plan->dst = dst;
    return true;
}

// The kernels on the host, one rounding per operation in the shader's order
// (moe_detail::mul_rn / add_rn keep the compiler from forming an fma, as
// `precise` does for dxc). exp and log are the C library's, so a GPU result
// differs from these within d3d12_ssm_max_err's tolerance, not bit for bit;
// the copies and get_rows are exact. Pointers are the bindings' bases.

inline void d3d12_ssm_unary_emulate(const D3d12SsmUnaryParams& p, std::uint32_t groups_x,
                                    std::uint32_t groups_y, const float* src, float* dst) {
    using namespace moe_detail;
    float s = 0.0f;
    std::memcpy(&s, &p.scale_bits, sizeof(s));
    for (std::uint32_t r = 0; r < groups_y; ++r)
        for (std::uint32_t c = 0; c < groups_x * kD3d12SsmThreads && c < p.ne0; ++c) {
            const float x = src[r * p.src_row_stride + c];
            float y;
            if (p.op == kD3d12SsmScale)
                y = mul_rn(x, s);
            else if (p.op == kD3d12SsmSigmoid)
                y = 1.0f / add_rn(1.0f, std::exp(-x));
            else if (p.op == kD3d12SsmSoftplus)
                y = x > 20.0f ? x : std::log(add_rn(1.0f, std::exp(x)));
            else
                y = x / add_rn(1.0f, std::exp(-x));
            dst[r * p.dst_row_stride + c] = y;
        }
}

inline void d3d12_ssm_copy_emulate(const D3d12SsmCopyParams& p, std::uint32_t groups_x,
                                   const float* src, float* dst) {
    const std::uint64_t n = static_cast<std::uint64_t>(p.ne[0]) * p.ne[1] * p.ne[2] * p.ne[3];
    for (std::uint64_t f = 0; f < n && f < static_cast<std::uint64_t>(groups_x) * kD3d12SsmThreads;
         ++f) {
        std::uint32_t i = static_cast<std::uint32_t>(f);
        const std::uint32_t i0 = i % p.ne[0];
        i /= p.ne[0];
        const std::uint32_t i1 = i % p.ne[1];
        i /= p.ne[1];
        const std::uint32_t i2 = i % p.ne[2];
        const std::uint32_t i3 = i / p.ne[2];
        const std::uint32_t s = i0 * p.src_stride[0] + i1 * p.src_stride[1] + i2 * p.src_stride[2] +
                                i3 * p.src_stride[3];
        const std::uint32_t d = i0 * p.dst_stride[0] + i1 * p.dst_stride[1] + i2 * p.dst_stride[2] +
                                i3 * p.dst_stride[3];
        std::memcpy(dst + d, src + s, sizeof(float));
    }
}

inline void d3d12_ssm_get_rows_emulate(const D3d12SsmGetRowsParams& p, std::uint32_t groups_x,
                                       std::uint32_t groups_y, const float* src,
                                       const std::int32_t* ids, float* dst) {
    for (std::uint32_t r = 0; r < groups_y; ++r) {
        const std::uint32_t row = static_cast<std::uint32_t>(ids[r * p.ids_stride]);
        for (std::uint32_t c = 0; c < groups_x * kD3d12SsmThreads && c < p.ne0; ++c)
            std::memcpy(dst + r * p.dst_row_stride + c, src + row * p.src_row_stride + c,
                        sizeof(float));
    }
}

inline void d3d12_ssm_conv_emulate(const D3d12SsmConvParams& p, std::uint32_t groups_x,
                                   std::uint32_t groups_y, std::uint32_t groups_z, const float* x,
                                   const float* w, float* dst) {
    using namespace moe_detail;
    for (std::uint32_t s = 0; s < groups_z; ++s)
        for (std::uint32_t t = 0; t < groups_y; ++t)
            for (std::uint32_t ch = 0; ch < groups_x * kD3d12SsmThreads && ch < p.channels; ++ch) {
                const std::uint32_t xb = s * p.x_seq_stride + ch * p.x_channel_stride + t;
                const std::uint32_t wb = ch * p.w_channel_stride;
                float acc = 0.0f;
                for (std::uint32_t i = 0; i < p.d_conv; ++i)
                    acc = add_rn(acc, mul_rn(x[xb + i], w[wb + i]));
                if (p.silu)
                    acc = acc / add_rn(1.0f, std::exp(-acc));
                dst[s * p.dst_seq_stride + t * p.dst_token_stride + ch] = acc;
            }
}

// Lane l of a row holds elements 4 * (l + 4 * e) + c, e = 0..7, c = 0..3, and
// sums its products in that order; the row combines the four lanes as
// (l0 + l1) + (l2 + l3).
// state_out: the fused kernel's u5 (kD3d12GdnFused, K = 1), where the new
// state goes instead of after the output rows; null = not fused.
inline void d3d12_gdn_emulate(const D3d12GdnParams& p, std::uint32_t groups_x,
                              std::uint32_t groups_y, std::uint32_t groups_z, const float* q,
                              const float* k, const float* v, const float* g, const float* beta,
                              const float* state, float* dst, float* state_out = nullptr) {
    using namespace moe_detail;
    constexpr std::uint32_t sv = kD3d12GdnSv;
    const std::uint32_t K = p.k_kda & 0xffffu;
    const bool kda = ((p.k_kda >> 16) & 1u) != 0;
    const std::uint32_t T = p.t, H = p.h, B = p.b;
    float scale = 0.0f;
    std::memcpy(&scale, &p.scale_bits, sizeof(scale));
    const std::uint32_t snap = sv * sv, attn = sv * H * T * B, per_snap = snap * H * B;
    std::vector<float> m(sv);
    auto lane_index = [](std::uint32_t l, std::uint32_t n) { // n-th element of lane l
        return 4 * (l + 4 * (n / 4)) + n % 4;
    };
    for (std::uint32_t s = 0; s < groups_z; ++s)
        for (std::uint32_t h = 0; h < groups_y; ++h)
            for (std::uint32_t j = 0; j < groups_x * kD3d12GdnRows; ++j) {
                const std::uint32_t iq1 = h % (p.heads_qk & 0xffffu), ik1 = h % (p.heads_qk >> 16);
                const std::uint32_t iq3 = s / (p.seq_div_qk & 0xffffu);
                const std::uint32_t ik3 = s / (p.seq_div_qk >> 16);
                const std::uint32_t own = (s * H + h) * snap + j * sv;
                for (std::uint32_t i = 0; i < sv; ++i)
                    m[i] = state[own + i];
                for (std::uint32_t t = 0; t < T; ++t) {
                    const float* qt = q + iq3 * p.q_seq_stride + t * p.q_token_stride + iq1 * sv;
                    const float* kt = k + ik3 * p.k_seq_stride + t * p.k_token_stride + ik1 * sv;
                    const std::uint32_t gb = (s * T + t) * H + h;
                    const float ge = kda ? 0.0f : std::exp(g[gb]);
                    float part[4];
                    for (std::uint32_t l = 0; l < 4; ++l) {
                        float acc = 0.0f;
                        for (std::uint32_t n = 0; n < sv / 4; ++n) {
                            const std::uint32_t i = lane_index(l, n);
                            m[i] = mul_rn(m[i], kda ? std::exp(g[gb * sv + i]) : ge);
                            acc = add_rn(acc, mul_rn(m[i], kt[i]));
                        }
                        part[l] = acc;
                    }
                    const float sk = add_rn(add_rn(part[0], part[1]), add_rn(part[2], part[3]));
                    const float vj = v[s * p.v_seq_stride + t * p.v_token_stride + h * sv + j];
                    const float d = mul_rn(vj - sk, beta[gb]);
                    for (std::uint32_t l = 0; l < 4; ++l) {
                        float acc = 0.0f;
                        for (std::uint32_t n = 0; n < sv / 4; ++n) {
                            const std::uint32_t i = lane_index(l, n);
                            m[i] = add_rn(m[i], mul_rn(kt[i], d));
                            acc = add_rn(acc, mul_rn(m[i], qt[i]));
                        }
                        part[l] = acc;
                    }
                    const float y = add_rn(add_rn(part[0], part[1]), add_rn(part[2], part[3]));
                    dst[gb * sv + j] = mul_rn(y, scale);
                    const std::uint32_t slot = T - 1 - t;
                    if (K > 1 && slot < K)
                        std::memcpy(dst + attn + slot * per_snap + own, m.data(),
                                    sv * sizeof(float));
                }
                if (state_out)
                    std::memcpy(state_out + own, m.data(), sv * sizeof(float));
                else if (K == 1)
                    std::memcpy(dst + attn + own, m.data(), sv * sizeof(float));
            }
}

// max |got - ref| / (1 + |ref|) over n floats; infinity on a NaN in one only.
inline double d3d12_ssm_max_err(const float* got, const float* ref, std::size_t n) {
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
