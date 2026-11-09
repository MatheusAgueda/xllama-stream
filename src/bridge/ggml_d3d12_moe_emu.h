// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The tail of a MoE layer on the d3d12 backend (shaders/ggml_d3d12_moe.hlsl):
// the node patterns each kernel replaces, their constants and grids, and a
// host emulation of both kernels (tests compare it with ggml's CPU ops; a
// selftest can compare it with the GPU).
//
// KERNEL 1, combine. llama-graph.cpp build_moe_ffn ends with
//   W   = MUL(E, w)        E = F32 [ne0, n, T] (the MUL_MAT_ID output),
//                          w = F32 [1, n, T] (the expert weights)
//   V_i = VIEW(W, [ne0, T], nb1 = W nb2, offs = i * W nb1), i = 0 .. n-1
//   A_1 = ADD(V_0, V_1), A_i = ADD(A_(i-1), V_i), A_(n-1) = ffn_moe_out
// d3d12_moe_combine_plan(A_(n-1)) walks that chain back. The kernel writes
// A_(n-1) from E and w; W, the views and A_1 .. A_(n-2) are not computed.
//
// KERNEL 2, tail. qwen35moe.cpp build_layer_ffn then has
//   G = MUL(s, g)   s = F32 [ne0, rows] (the shared expert), g = F32 with
//                   ne0 = 1, one value per row (its sigmoid gate)
//   O = ADD(m, G)   m = ffn_moe_out                 (either operand order)
//   P = ADD(O, q)   q = the attention residual      (either operand order)
// d3d12_moe_tail_plan(node) takes P (mode 2), O (mode 1) or G (mode 0),
// whichever `node` is, with all of its pattern.
//
// plan.nodes lists the nodes a dispatch replaces in graph order, the output
// last. The backend dispatches at the output and skips the others, after
// ggml_can_fuse_subgraph(cgraph, ...) over them with the output as the only
// one (no node outside reads them). Offsets are 32-bit, as in the shader.

#pragma once

#include "xllama/ggml_d3d12.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace xllama {

inline constexpr std::uint32_t kD3d12MoeThreads = 256; // numthreads of both kernels

// b0 of KERNEL 1, 8 root constants in this order (strides in floats).
struct D3d12MoeCombineParams {
    std::uint32_t ne0 = 0;
    std::uint32_t n = 0;                // experts per token
    std::uint32_t e_expert_stride = 0;  // E nb1 / 4
    std::uint32_t e_token_stride = 0;   // E nb2 / 4
    std::uint32_t w_expert_stride = 0;  // w nb1 / 4, 0 when w has one expert column
    std::uint32_t w_token_stride = 0;   // w nb2 / 4, 0 when w has one token
    std::uint32_t dst_token_stride = 0; // A_(n-1) nb1 / 4
    std::uint32_t pad0 = 0;
};
static_assert(sizeof(D3d12MoeCombineParams) == 8 * sizeof(std::uint32_t), "8 root constants");

struct D3d12MoeCombinePlan {
    D3d12MoeCombineParams params;
    std::uint32_t groups_x = 0;            // ceil(ne0 / 256)
    std::uint32_t groups_y = 0;            // T
    const ggml_tensor* e = nullptr;        // u1
    const ggml_tensor* w = nullptr;        // u2
    std::vector<const ggml_tensor*> nodes; // W, V_0 .. V_(n-1), A_1 .. A_(n-1)
};

// b0 of KERNEL 2, 8 root constants in this order (strides in floats).
struct D3d12MoeTailParams {
    std::uint32_t ne0 = 0;
    std::uint32_t s_row_stride = 0;
    std::uint32_t g_row_stride = 0;
    std::uint32_t m_row_stride = 0; // mode >= 1
    std::uint32_t q_row_stride = 0; // mode 2
    std::uint32_t dst_row_stride = 0;
    std::uint32_t mode = 0; // 0: s * g, 1: m + s * g, 2: (m + s * g) + q
    std::uint32_t pad0 = 0;
};
static_assert(sizeof(D3d12MoeTailParams) == 8 * sizeof(std::uint32_t), "8 root constants");

struct D3d12MoeTailPlan {
    D3d12MoeTailParams params;
    std::uint32_t groups_x = 0; // ceil(ne0 / 256)
    std::uint32_t groups_y = 0; // rows
    const ggml_tensor* s = nullptr;
    const ggml_tensor* g = nullptr;
    const ggml_tensor* m = nullptr;        // null below mode 1
    const ggml_tensor* q = nullptr;        // null below mode 2
    std::vector<const ggml_tensor*> nodes; // G[, O[, P]]
};

namespace moe_detail {

inline bool f32(const ggml_tensor* t) {
    return t && t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float);
}

inline bool fits32(std::uint64_t floats) {
    return floats <= std::numeric_limits<std::uint32_t>::max() / sizeof(float);
}

// The stride in floats between the rows of t (dims 1..3 flattened) when they
// are evenly spaced; 0 for a single row.
inline bool row_stride(const ggml_tensor* t, std::uint32_t* out) {
    if (!f32(t))
        return false;
    std::uint64_t s = 0, below = 1;
    for (int k = 1; k < GGML_MAX_DIMS; ++k) {
        if (t->ne[k] == 1)
            continue;
        if (t->ne[k] < 1 || t->nb[k] % sizeof(float) != 0 || t->nb[k] == 0)
            return false;
        const std::uint64_t nbk = t->nb[k] / sizeof(float);
        if (s == 0)
            s = nbk;
        else if (nbk != s * below)
            return false;
        below *= static_cast<std::uint64_t>(t->ne[k]);
    }
    if (!fits32(s))
        return false;
    *out = static_cast<std::uint32_t>(s);
    return true;
}

// The last float a kernel touches through a binding, +1, fits 32-bit offsets.
inline bool rows_fit(std::uint64_t rows, std::uint32_t stride, std::uint64_t width) {
    return rows == 0 || fits32((rows - 1) * stride + width);
}

inline bool groups(std::int64_t ne0, std::int64_t rows, std::uint32_t* gx, std::uint32_t* gy) {
    const std::uint64_t x =
        (static_cast<std::uint64_t>(ne0) + kD3d12MoeThreads - 1) / kD3d12MoeThreads;
    if (ne0 <= 0 || rows <= 0 || x > kD3d12MaxGroups ||
        static_cast<std::uint64_t>(rows) > kD3d12MaxGroups)
        return false;
    *gx = static_cast<std::uint32_t>(x);
    *gy = static_cast<std::uint32_t>(rows);
    return true;
}

// MUL(s, g) with g one value per row of s.
inline bool is_gate(const ggml_tensor* t) {
    if (!t || t->op != GGML_OP_MUL)
        return false;
    const ggml_tensor* s = t->src[0];
    const ggml_tensor* g = t->src[1];
    if (!f32(t) || !f32(s) || !g || g->type != GGML_TYPE_F32 || g->ne[0] != 1 || s->ne[0] < 1 ||
        !ggml_are_same_shape(t, s))
        return false;
    for (int k = 1; k < GGML_MAX_DIMS; ++k)
        if (g->ne[k] != s->ne[k])
            return false;
    return true;
}

// ADD(x, y) with one operand a gate: that gate, and the other operand in *other.
inline const ggml_tensor* add_of_gate(const ggml_tensor* t, const ggml_tensor** other) {
    if (!t || t->op != GGML_OP_ADD || !t->src[0] || !t->src[1])
        return nullptr;
    for (int side = 1; side >= 0; --side) // llama's order is ADD(moe_out, gated) first
        if (is_gate(t->src[side])) {
            *other = t->src[1 - side];
            return t->src[side];
        }
    return nullptr;
}

} // namespace moe_detail

// false unless `last` is A_(n-1) of the whole chain over all n experts of W,
// with the shapes above (w may broadcast over experts or tokens) and a grid
// and offsets the kernel takes.
inline bool d3d12_moe_combine_plan(const ggml_tensor* last, D3d12MoeCombinePlan* plan) {
    using namespace moe_detail;
    if (!last || !plan || last->op != GGML_OP_ADD || !f32(last))
        return false;
    std::vector<const ggml_tensor*> adds, views; // back to front
    const ggml_tensor* a = last;
    while (a && a->op == GGML_OP_ADD) {
        adds.push_back(a);
        views.push_back(a->src[1]);
        a = a->src[0];
    }
    if (!a)
        return false;
    views.push_back(a);
    const std::size_t n = views.size();
    const ggml_tensor* mul = a->view_src;
    if (!mul || mul->op != GGML_OP_MUL || !f32(mul) || static_cast<std::size_t>(mul->ne[1]) != n ||
        mul->ne[3] != 1)
        return false;
    const ggml_tensor* e = mul->src[0];
    const ggml_tensor* w = mul->src[1];
    const std::int64_t ne0 = mul->ne[0], n_tok = mul->ne[2];
    if (!f32(e) || !ggml_are_same_shape(mul, e) || !w || w->type != GGML_TYPE_F32 ||
        w->ne[0] != 1 || (w->ne[1] != 1 && w->ne[1] != mul->ne[1]) ||
        (w->ne[2] != 1 && w->ne[2] != n_tok) || w->ne[3] != 1)
        return false;
    for (std::size_t i = 0; i < n; ++i) {
        const ggml_tensor* v = views[n - 1 - i];
        if (!v || v->op != GGML_OP_VIEW || v->view_src != mul || !f32(v) ||
            v->view_offs != i * mul->nb[1] || v->ne[0] != ne0 || v->ne[1] != n_tok ||
            v->ne[2] != 1 || v->ne[3] != 1 || (n_tok > 1 && v->nb[1] != mul->nb[2]))
            return false;
    }
    for (const ggml_tensor* add : adds)
        if (!f32(add) || add->ne[0] != ne0 || add->ne[1] != n_tok || add->ne[2] != 1 ||
            add->ne[3] != 1)
            return false;
    if (e->nb[1] % sizeof(float) != 0 || e->nb[2] % sizeof(float) != 0 ||
        w->nb[1] % sizeof(float) != 0 || w->nb[2] % sizeof(float) != 0 ||
        last->nb[1] % sizeof(float) != 0)
        return false;

    D3d12MoeCombinePlan p;
    p.params.ne0 = static_cast<std::uint32_t>(ne0);
    p.params.n = static_cast<std::uint32_t>(n);
    const std::uint64_t es1 = e->nb[1] / sizeof(float), es2 = e->nb[2] / sizeof(float);
    const std::uint64_t ws1 = w->ne[1] == 1 ? 0 : w->nb[1] / sizeof(float);
    const std::uint64_t ws2 = w->ne[2] == 1 ? 0 : w->nb[2] / sizeof(float);
    const std::uint64_t ds1 = last->nb[1] / sizeof(float);
    if (!fits32(es1) || !fits32(es2) || !fits32(ws1) || !fits32(ws2) || !fits32(ds1))
        return false;
    const std::uint64_t t1 = static_cast<std::uint64_t>(n_tok) - 1, n1 = n - 1;
    if (!fits32(t1 * es2 + n1 * es1 + static_cast<std::uint64_t>(ne0)) ||
        !fits32(t1 * ws2 + n1 * ws1 + 1) || !fits32(t1 * ds1 + static_cast<std::uint64_t>(ne0)))
        return false;
    if (!groups(ne0, n_tok, &p.groups_x, &p.groups_y))
        return false;
    p.params.e_expert_stride = static_cast<std::uint32_t>(es1);
    p.params.e_token_stride = static_cast<std::uint32_t>(es2);
    p.params.w_expert_stride = static_cast<std::uint32_t>(ws1);
    p.params.w_token_stride = static_cast<std::uint32_t>(ws2);
    p.params.dst_token_stride = static_cast<std::uint32_t>(ds1);
    p.e = e;
    p.w = w;
    p.nodes.reserve(2 * n);
    p.nodes.push_back(mul);
    for (std::size_t i = 0; i < n; ++i)
        p.nodes.push_back(views[n - 1 - i]);
    for (std::size_t i = adds.size(); i-- > 0;)
        p.nodes.push_back(adds[i]);
    *plan = std::move(p);
    return true;
}

// false unless `node` is P, O or G above with every operand F32, s, m, q and
// the output one shape, rows evenly spaced and a grid the kernel takes.
inline bool d3d12_moe_tail_plan(const ggml_tensor* node, D3d12MoeTailPlan* plan) {
    using namespace moe_detail;
    if (!node || !plan || !f32(node))
        return false;
    D3d12MoeTailPlan p;
    const ggml_tensor* gate = nullptr;
    if (node->op == GGML_OP_ADD && node->src[0] && node->src[1]) {
        for (int side = 0; side < 2 && !gate; ++side) { // P = ADD(O, q): llama's order first
            const ggml_tensor* m = nullptr;
            const ggml_tensor* gt = add_of_gate(node->src[side], &m);
            if (gt) {
                gate = gt;
                p.params.mode = 2;
                p.m = m;
                p.q = node->src[1 - side];
                p.nodes = {gt, node->src[side], node};
            }
        }
        if (!gate) {
            const ggml_tensor* m = nullptr;
            gate = add_of_gate(node, &m);
            if (!gate)
                return false;
            p.params.mode = 1;
            p.m = m;
            p.nodes = {gate, node};
        }
    } else if (is_gate(node)) {
        gate = node;
        p.params.mode = 0;
        p.nodes = {node};
    } else {
        return false;
    }
    p.s = gate->src[0];
    p.g = gate->src[1];
    const std::int64_t ne0 = node->ne[0];
    const std::uint64_t rows = static_cast<std::uint64_t>(ggml_nrows(node));
    for (const ggml_tensor* t : {p.m, p.q})
        if (t && (!f32(t) || !ggml_are_same_shape(t, node)))
            return false;
    for (const ggml_tensor* t : p.nodes)
        if (!ggml_are_same_shape(t, node))
            return false;
    std::uint32_t ss = 0, gs = 0, ms = 0, qs = 0, ds = 0;
    if (!row_stride(p.s, &ss) || !row_stride(p.g, &gs) || (p.m && !row_stride(p.m, &ms)) ||
        (p.q && !row_stride(p.q, &qs)) || !row_stride(node, &ds))
        return false;
    const std::uint64_t w = static_cast<std::uint64_t>(ne0);
    if (!rows_fit(rows, ss, w) || !rows_fit(rows, gs, 1) || !rows_fit(rows, ms, w) ||
        !rows_fit(rows, qs, w) || !rows_fit(rows, ds, w))
        return false;
    if (!groups(ne0, static_cast<std::int64_t>(rows), &p.groups_x, &p.groups_y))
        return false;
    p.params.ne0 = static_cast<std::uint32_t>(ne0);
    p.params.s_row_stride = ss;
    p.params.g_row_stride = gs;
    p.params.m_row_stride = ms;
    p.params.q_row_stride = qs;
    p.params.dst_row_stride = ds;
    *plan = std::move(p);
    return true;
}

// The kernels on the host, one rounding per operation in the shader's order:
// the volatile stores keep the compiler from contracting a multiply and an add
// into an fma, as `precise` does for dxc. Pointers are the bindings' bases.
namespace moe_detail {
inline float mul_rn(float a, float b) {
    volatile float r = a * b;
    return r;
}
inline float add_rn(float a, float b) {
    volatile float r = a + b;
    return r;
}
} // namespace moe_detail

inline void d3d12_moe_combine_emulate(const D3d12MoeCombineParams& p, std::uint32_t groups_x,
                                      std::uint32_t groups_y, const float* e, const float* w,
                                      float* dst) {
    using namespace moe_detail;
    if (p.n == 0)
        return;
    for (std::uint32_t t = 0; t < groups_y; ++t)
        for (std::uint32_t gx = 0; gx < groups_x; ++gx)
            for (std::uint32_t tid = 0; tid < kD3d12MoeThreads; ++tid) {
                const std::uint32_t c = gx * kD3d12MoeThreads + tid;
                if (c >= p.ne0)
                    continue;
                const std::uint32_t eb = t * p.e_token_stride + c, wb = t * p.w_token_stride;
                float acc = mul_rn(e[eb], w[wb]);
                for (std::uint32_t i = 1; i < p.n; ++i)
                    acc = add_rn(
                        acc, mul_rn(e[eb + i * p.e_expert_stride], w[wb + i * p.w_expert_stride]));
                dst[t * p.dst_token_stride + c] = acc;
            }
}

inline void d3d12_moe_tail_emulate(const D3d12MoeTailParams& p, std::uint32_t groups_x,
                                   std::uint32_t groups_y, const float* s, const float* g,
                                   const float* m, const float* q, float* dst) {
    using namespace moe_detail;
    for (std::uint32_t r = 0; r < groups_y; ++r)
        for (std::uint32_t gx = 0; gx < groups_x; ++gx)
            for (std::uint32_t tid = 0; tid < kD3d12MoeThreads; ++tid) {
                const std::uint32_t c = gx * kD3d12MoeThreads + tid;
                if (c >= p.ne0)
                    continue;
                float y = mul_rn(s[r * p.s_row_stride + c], g[r * p.g_row_stride]);
                if (p.mode >= 1)
                    y = add_rn(m[r * p.m_row_stride + c], y);
                if (p.mode >= 2)
                    y = add_rn(y, q[r * p.q_row_stride + c]);
                dst[r * p.dst_row_stride + c] = y;
            }
}

} // namespace xllama
