// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// FLASH_ATTN_EXT on the d3d12 backend (shaders/ggml_d3d12_fap.hlsl), prefill
// ubatches and decode tokens alike: the nodes the kernel takes, its constants
// and grid, and a host emulation of the kernel (tests compare it with a double
// reference and ggml's CPU op).
//
// The node of llama-graph.cpp build_attn_mha with flash attention on: q is
// the roped Q permuted to [D, N, n_head], k and v are views of the f16 KV
// cache permuted to [D, n_kv, n_head_kv], the mask is the f16 kq_mask
// [n_kv, N padded], dst is [D, n_head, N]. With the KV cache in a CPU buffer
// the scheduler copies k, v, q and the mask into D3D12_Host as split inputs;
// with it in D3D12_Host (llama.cpp patch 0026, LLAMA_KV_ON_DEVICE) the kernel
// reads the cache in place.
//
// A few tokens give a few groups (N = 1 of Qwen3.6: 2, one per kv head), each
// walking all n_kv positions alone. d3d12_fap_chunk splits the positions into
// chunks (split-K, grid z): every (group, chunk) writes its rows' running max,
// sum and unnormalised output to a scratch buffer (u3), and a second dispatch
// of the same kernel (kD3d12FapReduce) combines the chunks into dst.
//
// d3d12_fap_plan(node, &plan) is false unless the kernel takes the node as it
// is. The params struct is the root constants in order (copy it to Op::c).

#pragma once

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace xllama {

inline constexpr std::uint32_t kD3d12FapThreads = 256; // numthreads
inline constexpr std::uint32_t kD3d12FapRows = 16;     // q rows (token, head) per group
inline constexpr std::uint32_t kD3d12FapTile = 16;     // kv positions per tile
inline constexpr std::uint32_t kD3d12FapMaxD = 256;    // head size the groupshared rows hold
inline constexpr std::uint32_t kD3d12FapMask = 1;      // flags: a mask is bound
inline constexpr std::uint32_t kD3d12FapSplit =
    2; // flags: chunk z of the positions, partials to u3
inline constexpr std::uint32_t kD3d12FapReduce =
    4; // flags: u3's partials of (head x, token y) to dst
// Split-K policy: plans below kD3d12FapSplitBelow groups are split into at
// most kD3d12FapMaxChunks chunks of at least kD3d12FapMinChunk positions,
// aiming at kD3d12FapTargetGroups groups (Series X: 52 CUs, 2 groups each).
inline constexpr std::uint32_t kD3d12FapSplitBelow = 64;
inline constexpr std::uint32_t kD3d12FapMaxChunks = 64;
inline constexpr std::uint32_t kD3d12FapMinChunk = 64;
inline constexpr std::uint32_t kD3d12FapTargetGroups = 128;

// t0 = k, t1 = v (f16), u1 = q (f32), u2 = mask (f16), u0 = dst (f32),
// u3 = partials (f32, split and reduce only).
// Grid (groups_x, groups_y): 16 / G tokens of one kv head per group; split:
// (groups_x, groups_y, chunks); reduce: (n_head, N).
struct D3d12FapParams {
    std::uint32_t d_g = 0; // D | G << 16
    std::uint32_t n_kv = 0;
    std::uint32_t n_head = 0;          // N | n_head << 16
    std::uint32_t q_token_stride = 0;  // floats
    std::uint32_t q_head_stride = 0;   // floats
    std::uint32_t k_cell_stride = 0;   // 32-bit words
    std::uint32_t k_head_stride = 0;   // 32-bit words
    std::uint32_t v_cell_stride = 0;   // 32-bit words
    std::uint32_t v_head_stride = 0;   // 32-bit words
    std::uint32_t mask_row_stride = 0; // halves
    std::uint32_t scale_bits = 0;
    std::uint32_t flags = 0; // kD3d12FapMask | kD3d12FapSplit | kD3d12FapReduce
    std::uint32_t chunk = 0; // split / reduce: positions per chunk (a multiple of the tile)
};
static_assert(sizeof(D3d12FapParams) == 13 * sizeof(std::uint32_t), "13 root constants");

struct D3d12FapPlan {
    D3d12FapParams params;
    std::uint32_t groups_x = 0;
    std::uint32_t groups_y = 0;
    const ggml_tensor* q = nullptr;
    const ggml_tensor* k = nullptr;
    const ggml_tensor* v = nullptr;
    const ggml_tensor* mask = nullptr; // null: every position is seen
};

namespace fap_detail {

inline bool fits32(std::uint64_t v) {
    return v <= 0xffffffffull;
}

inline std::uint32_t bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline float from_bits(std::uint32_t u) {
    float f = 0.f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline float op_float(const ggml_tensor* t, int i) {
    float f = 0.f;
    std::memcpy(&f, reinterpret_cast<const char*>(t->op_params) + i * sizeof(float), sizeof(f));
    return f;
}

// One f16 tensor of [D, n_kv, heads, 1] with 32-bit aligned rows and heads.
inline bool kv_operand(const ggml_tensor* t, std::int64_t d, std::int64_t n_kv,
                       std::int64_t heads) {
    return t && t->type == GGML_TYPE_F16 && t->ne[0] == d && t->ne[1] == n_kv &&
           t->ne[2] == heads && t->ne[3] == 1 && t->nb[0] == sizeof(ggml_fp16_t) &&
           t->nb[1] % 4 == 0 && t->nb[2] % 4 == 0;
}

} // namespace fap_detail

inline bool d3d12_fap_plan(const ggml_tensor* node, D3d12FapPlan* plan) {
    using namespace fap_detail;
    if (!node || node->op != GGML_OP_FLASH_ATTN_EXT || node->type != GGML_TYPE_F32)
        return false;
    const ggml_tensor* q = node->src[0];
    const ggml_tensor* k = node->src[1];
    const ggml_tensor* v = node->src[2];
    const ggml_tensor* mask = node->src[3];
    if (!q || !k || !v || node->src[4]) // sinks: not in the kernel
        return false;
    const float scale = op_float(node, 0);
    if (!std::isfinite(scale) || op_float(node, 1) != 0.f || op_float(node, 2) != 0.f)
        return false; // ALiBi (max_bias) and softcap are not in the kernel
    const std::int64_t d = q->ne[0], n = q->ne[1], heads = q->ne[2];
    const std::int64_t n_kv = k->ne[1], kv_heads = k->ne[2];
    if (q->type != GGML_TYPE_F32 || q->ne[3] != 1 || q->nb[0] != sizeof(float) ||
        q->nb[1] % sizeof(float) != 0 || q->nb[2] % sizeof(float) != 0)
        return false;
    if (d < 4 || d % 4 != 0 || d > kD3d12FapMaxD || n < 1 || n > 0xffff || heads < 1 ||
        heads > 0xffff || n_kv < 1 || kv_heads < 1 || heads % kv_heads != 0)
        return false;
    const std::int64_t g = heads / kv_heads;
    if (g > kD3d12FapRows || (g & (g - 1)) != 0)
        return false;
    if (!kv_operand(k, d, n_kv, kv_heads) || !kv_operand(v, d, n_kv, kv_heads))
        return false;
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < n_kv || mask->ne[1] < n ||
                 mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[0] != sizeof(ggml_fp16_t) ||
                 mask->nb[1] % sizeof(ggml_fp16_t) != 0))
        return false;
    if (node->ne[0] != d || node->ne[1] != heads || node->ne[2] != n || node->ne[3] != 1 ||
        !ggml_is_contiguous(node))
        return false;

    D3d12FapParams p;
    p.d_g = static_cast<std::uint32_t>(d | (g << 16));
    p.n_kv = static_cast<std::uint32_t>(n_kv);
    p.n_head = static_cast<std::uint32_t>(n | (heads << 16));
    const std::uint64_t qt = q->nb[1] / 4, qh = q->nb[2] / 4;
    const std::uint64_t kc = k->nb[1] / 4, kh = k->nb[2] / 4;
    const std::uint64_t vc = v->nb[1] / 4, vh = v->nb[2] / 4;
    const std::uint64_t mr = mask ? mask->nb[1] / 2 : 0;
    // The shader's byte addresses are 32-bit: the last byte each operand reads.
    const std::uint64_t hd = static_cast<std::uint64_t>(d / 2);
    if (!fits32(4 * ((n - 1) * qt + (heads - 1) * qh + d)) ||
        !fits32(4 * ((n_kv - 1) * kc + (kv_heads - 1) * kh + hd)) ||
        !fits32(4 * ((n_kv - 1) * vc + (kv_heads - 1) * vh + hd)) ||
        !fits32(2 * ((n - 1) * mr + n_kv) + 2) ||
        !fits32(4 * static_cast<std::uint64_t>(n * heads * d)))
        return false;
    p.q_token_stride = static_cast<std::uint32_t>(qt);
    p.q_head_stride = static_cast<std::uint32_t>(qh);
    p.k_cell_stride = static_cast<std::uint32_t>(kc);
    p.k_head_stride = static_cast<std::uint32_t>(kh);
    p.v_cell_stride = static_cast<std::uint32_t>(vc);
    p.v_head_stride = static_cast<std::uint32_t>(vh);
    p.mask_row_stride = static_cast<std::uint32_t>(mr);
    p.scale_bits = bits(scale);
    p.flags = mask ? kD3d12FapMask : 0;

    const std::int64_t tokens_per_group = kD3d12FapRows / g;
    const std::int64_t gx = (n + tokens_per_group - 1) / tokens_per_group;
    if (gx > 65535 || kv_heads > 65535)
        return false;
    if (plan) {
        plan->params = p;
        plan->groups_x = static_cast<std::uint32_t>(gx);
        plan->groups_y = static_cast<std::uint32_t>(kv_heads);
        plan->q = q;
        plan->k = k;
        plan->v = v;
        plan->mask = mask;
    }
    return true;
}

// Floats per partial record of head size d: M, S, two pad, O[d] (16-byte aligned).
inline std::uint32_t d3d12_fap_rec_floats(std::uint32_t d) {
    return d + 4;
}

// The chunk length for a planned node, or 0 to run it in one pass: only plans
// below kD3d12FapSplitBelow groups, with the partials of every (token, head,
// chunk) within scratch_bytes and at least two chunks.
inline std::uint32_t d3d12_fap_chunk(const D3d12FapPlan& plan, std::size_t scratch_bytes) {
    const std::uint32_t d = plan.params.d_g & 0xffffu;
    const std::uint32_t n = plan.params.n_head & 0xffffu, heads = plan.params.n_head >> 16;
    const std::uint32_t n_kv = plan.params.n_kv;
    const std::uint64_t groups = static_cast<std::uint64_t>(plan.groups_x) * plan.groups_y;
    if (groups == 0 || groups >= kD3d12FapSplitBelow)
        return 0;
    std::uint64_t want = (kD3d12FapTargetGroups + groups - 1) / groups;
    want = std::min<std::uint64_t>(want, kD3d12FapMaxChunks);
    want = std::min<std::uint64_t>(want, n_kv / kD3d12FapMinChunk);
    const std::uint64_t per_chunk =
        static_cast<std::uint64_t>(n) * heads * d3d12_fap_rec_floats(d) * 4;
    want = std::min<std::uint64_t>(want, per_chunk ? scratch_bytes / per_chunk : 0);
    if (want < 2)
        return 0;
    std::uint64_t chunk = (n_kv + want - 1) / want;
    chunk = (chunk + kD3d12FapTile - 1) / kD3d12FapTile * kD3d12FapTile;
    return (n_kv + chunk - 1) / chunk >= 2 ? static_cast<std::uint32_t>(chunk) : 0;
}

inline std::uint32_t d3d12_fap_chunks(std::uint32_t n_kv, std::uint32_t chunk) {
    return chunk ? (n_kv + chunk - 1) / chunk : 1;
}

namespace fap_detail {

// One q row of the kernel over positions [pb, pe) in its order of operations:
// tiles of 16 from pb, a running max and sum, f32 throughout. A tile the row
// does not see at all changes nothing in it, so skipping it per row (the
// kernel skips per group) gives the same floats.
inline void row_pass(const D3d12FapParams& p, std::uint32_t tok, std::uint32_t gy, const float* qr,
                     const ggml_fp16_t* k, const ggml_fp16_t* v, const ggml_fp16_t* mask,
                     std::uint32_t pb, std::uint32_t pe, std::vector<float>& o, float* m_out,
                     float* s_out) {
    const std::uint32_t d = p.d_g & 0xffffu;
    const float scale = from_bits(p.scale_bits);
    const bool has_mask = (p.flags & kD3d12FapMask) != 0;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    float s[kD3d12FapTile];
    float m_run = neg_inf, sum = 0.f;
    std::fill(o.begin(), o.end(), 0.f);
    for (std::uint32_t pos0 = pb; pos0 < pe; pos0 += kD3d12FapTile) {
        const std::uint32_t np = std::min(kD3d12FapTile, pe - pos0);
        bool seen = false;
        float mx = neg_inf;
        for (std::uint32_t l = 0; l < kD3d12FapTile; ++l) {
            s[l] = neg_inf;
            if (l >= np)
                continue;
            const float m =
                has_mask ? ggml_fp16_to_fp32(
                               mask[static_cast<std::size_t>(tok) * p.mask_row_stride + pos0 + l])
                         : 0.f;
            if (m == neg_inf)
                continue;
            seen = true;
            const ggml_fp16_t* kr = k + 2 * (static_cast<std::size_t>(pos0 + l) * p.k_cell_stride +
                                             static_cast<std::size_t>(gy) * p.k_head_stride);
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            for (std::uint32_t e = 0; e < d; e += 4) {
                a0 += qr[e] * ggml_fp16_to_fp32(kr[e]);
                a1 += qr[e + 1] * ggml_fp16_to_fp32(kr[e + 1]);
                a2 += qr[e + 2] * ggml_fp16_to_fp32(kr[e + 2]);
                a3 += qr[e + 3] * ggml_fp16_to_fp32(kr[e + 3]);
            }
            s[l] = ((a0 + a1) + (a2 + a3)) * scale + m;
            mx = std::max(mx, s[l]);
        }
        if (!seen)
            continue;
        const float m_new = std::max(m_run, mx);
        float alpha = 1.f;
        if (m_new != m_run)
            alpha = m_run == neg_inf ? 0.f : std::exp(m_run - m_new);
        float pw[kD3d12FapTile];
        float ps = 0.f;
        for (std::uint32_t l = 0; l < kD3d12FapTile; ++l) {
            pw[l] = s[l] == neg_inf ? 0.f : std::exp(s[l] - m_new);
            ps += pw[l];
        }
        m_run = m_new;
        sum = sum * alpha + ps;
        for (std::uint32_t e = 0; e < d; ++e)
            o[e] *= alpha;
        for (std::uint32_t i = 0; i < np; ++i) {
            if (pw[i] == 0.f)
                continue;
            const ggml_fp16_t* vr = v + 2 * (static_cast<std::size_t>(pos0 + i) * p.v_cell_stride +
                                             static_cast<std::size_t>(gy) * p.v_head_stride);
            for (std::uint32_t e = 0; e < d; ++e)
                o[e] += pw[i] * ggml_fp16_to_fp32(vr[e]);
        }
    }
    *m_out = m_run;
    *s_out = sum;
}

} // namespace fap_detail

// The one-pass kernel on the host. q, dst: the tensors' data; k, v, mask:
// their data as halves (strides in the params are words / halves).
inline void d3d12_fap_emulate(const D3d12FapParams& p, std::uint32_t groups_x,
                              std::uint32_t groups_y, const float* q, const ggml_fp16_t* k,
                              const ggml_fp16_t* v, const ggml_fp16_t* mask, float* dst) {
    const std::uint32_t d = p.d_g & 0xffffu, g = p.d_g >> 16;
    const std::uint32_t n = p.n_head & 0xffffu, heads = p.n_head >> 16;
    const std::uint32_t tokens_per_group = kD3d12FapRows / g;
    std::vector<float> o(d);
    for (std::uint32_t gy = 0; gy < groups_y; ++gy)
        for (std::uint32_t gx = 0; gx < groups_x; ++gx)
            for (std::uint32_t r = 0; r < kD3d12FapRows; ++r) {
                const std::uint32_t tok = gx * tokens_per_group + r / g;
                const std::uint32_t head = gy * g + r % g;
                if (tok >= n)
                    continue;
                const float* qr = q + static_cast<std::size_t>(tok) * p.q_token_stride +
                                  static_cast<std::size_t>(head) * p.q_head_stride;
                float m_run = 0.f, sum = 0.f;
                fap_detail::row_pass(p, tok, gy, qr, k, v, mask, 0, p.n_kv, o, &m_run, &sum);
                const float inv = sum == 0.f ? 0.f : 1.f / sum;
                float* out = dst + (static_cast<std::size_t>(tok) * heads + head) * d;
                for (std::uint32_t e = 0; e < d; ++e)
                    out[e] = o[e] * inv;
            }
}

// The split dispatch on the host: the partial record of (tok, head, chunk z)
// at ((tok * n_head + head) * chunks + z) * rec_floats.
inline void d3d12_fap_emulate_split(const D3d12FapParams& p, std::uint32_t chunk,
                                    std::uint32_t groups_x, std::uint32_t groups_y, const float* q,
                                    const ggml_fp16_t* k, const ggml_fp16_t* v,
                                    const ggml_fp16_t* mask, float* part) {
    const std::uint32_t d = p.d_g & 0xffffu, g = p.d_g >> 16;
    const std::uint32_t n = p.n_head & 0xffffu, heads = p.n_head >> 16;
    const std::uint32_t tokens_per_group = kD3d12FapRows / g;
    const std::uint32_t chunks = d3d12_fap_chunks(p.n_kv, chunk), rec = d3d12_fap_rec_floats(d);
    std::vector<float> o(d);
    for (std::uint32_t z = 0; z < chunks; ++z)
        for (std::uint32_t gy = 0; gy < groups_y; ++gy)
            for (std::uint32_t gx = 0; gx < groups_x; ++gx)
                for (std::uint32_t r = 0; r < kD3d12FapRows; ++r) {
                    const std::uint32_t tok = gx * tokens_per_group + r / g;
                    const std::uint32_t head = gy * g + r % g;
                    if (tok >= n)
                        continue;
                    const float* qr = q + static_cast<std::size_t>(tok) * p.q_token_stride +
                                      static_cast<std::size_t>(head) * p.q_head_stride;
                    const std::uint32_t pb = z * chunk, pe = std::min(p.n_kv, pb + chunk);
                    float m_run = 0.f, sum = 0.f;
                    fap_detail::row_pass(p, tok, gy, qr, k, v, mask, pb, pe, o, &m_run, &sum);
                    float* w =
                        part + ((static_cast<std::size_t>(tok) * heads + head) * chunks + z) * rec;
                    w[0] = m_run;
                    w[1] = sum;
                    w[2] = w[3] = 0.f;
                    std::copy(o.begin(), o.end(), w + 4);
                }
}

// The reduce dispatch on the host, rows of tokens below n_tok (the split's
// rows): per row the chunks' max M, then in chunk order the weight
// exp(M_z - M) (0 for a chunk the row did not see), S and O summed in f32,
// dst = O * (1 / S), 0 where S is 0.
inline void d3d12_fap_emulate_reduce(const D3d12FapParams& p, std::uint32_t chunk,
                                     std::uint32_t n_tok, const float* part, float* dst) {
    const std::uint32_t d = p.d_g & 0xffffu;
    const std::uint32_t heads = p.n_head >> 16;
    const std::uint32_t chunks = d3d12_fap_chunks(p.n_kv, chunk), rec = d3d12_fap_rec_floats(d);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<float> o(d);
    for (std::uint32_t tok = 0; tok < n_tok; ++tok)
        for (std::uint32_t head = 0; head < heads; ++head) {
            const float* r0 = part + (static_cast<std::size_t>(tok) * heads + head) * chunks * rec;
            float mx = neg_inf;
            for (std::uint32_t z = 0; z < chunks; ++z)
                mx = std::max(mx, r0[z * rec]);
            float sum = 0.f;
            std::fill(o.begin(), o.end(), 0.f);
            for (std::uint32_t z = 0; z < chunks; ++z) {
                const float* r = r0 + static_cast<std::size_t>(z) * rec;
                if (r[0] == neg_inf)
                    continue;
                const float w = std::exp(r[0] - mx);
                sum += w * r[1];
                for (std::uint32_t e = 0; e < d; ++e)
                    o[e] += w * r[4 + e];
            }
            const float inv = sum == 0.f ? 0.f : 1.f / sum;
            float* out = dst + (static_cast<std::size_t>(tok) * heads + head) * d;
            for (std::uint32_t e = 0; e < d; ++e)
                out[e] = o[e] * inv;
        }
}

// Both dispatches of a split plan on the host (tokens the groups cover).
inline void d3d12_fap_emulate_chunked(const D3d12FapParams& p, std::uint32_t chunk,
                                      std::uint32_t groups_x, std::uint32_t groups_y,
                                      const float* q, const ggml_fp16_t* k, const ggml_fp16_t* v,
                                      const ggml_fp16_t* mask, float* dst) {
    const std::uint32_t d = p.d_g & 0xffffu, g = p.d_g >> 16;
    const std::uint32_t n = p.n_head & 0xffffu, heads = p.n_head >> 16;
    const std::uint32_t n_tok = std::min(n, groups_x * (kD3d12FapRows / g));
    std::vector<float> part(static_cast<std::size_t>(n_tok) * heads *
                                d3d12_fap_chunks(p.n_kv, chunk) * d3d12_fap_rec_floats(d),
                            0.f);
    d3d12_fap_emulate_split(p, chunk, groups_x, groups_y, q, k, v, mask, part.data());
    d3d12_fap_emulate_reduce(p, chunk, n_tok, part.data(), dst);
}

} // namespace xllama
