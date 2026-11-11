// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "../src/bridge/ggml_d3d12_ssm_emu.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

using namespace xllama;

namespace {

constexpr float kPoison = -7.f;

struct Ctx {
    ggml_context* ctx = nullptr;
    Ctx(std::size_t data_bytes, bool no_alloc) {
        ggml_init_params ip{};
        ip.mem_size = (no_alloc ? 0 : data_bytes) + 512 * ggml_tensor_overhead() +
                      ggml_graph_overhead() + (8u << 20);
        ip.no_alloc = no_alloc;
        ctx = ggml_init(ip);
        REQUIRE(ctx != nullptr);
    }
    Ctx(const Ctx&) = delete;
    Ctx& operator=(const Ctx&) = delete;
    ~Ctx() {
        if (ctx)
            ggml_free(ctx);
    }
};

void fill(ggml_tensor* t, std::mt19937* rng, float lo, float hi) {
    if (!rng || !t->data)
        return;
    std::uniform_real_distribution<float> d(lo, hi);
    auto* f = static_cast<float*>(t->data);
    for (std::int64_t i = 0; i < ggml_nelements(t); ++i)
        f[i] = d(*rng);
}

// Random rows of unit length, as the l2-normalized q and k of a layer.
void fill_unit_rows(ggml_tensor* t, std::mt19937* rng) {
    if (!rng || !t->data)
        return;
    fill(t, rng, -1.f, 1.f);
    auto* f = static_cast<float*>(t->data);
    for (std::int64_t r = 0; r < ggml_nrows(t); ++r) {
        float* row = f + r * t->ne[0];
        double ss = 0.0;
        for (std::int64_t i = 0; i < t->ne[0]; ++i)
            ss += static_cast<double>(row[i]) * row[i];
        const float inv = static_cast<float>(1.0 / std::sqrt(ss));
        for (std::int64_t i = 0; i < t->ne[0]; ++i)
            row[i] *= inv;
    }
}

const float* fdata(const ggml_tensor* t) {
    return static_cast<const float*>(t->data);
}

void compute(ggml_context* ctx, ggml_cgraph* gf) {
    ggml_cpu_init();
    REQUIRE(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS);
}

// Float offsets of t's elements from t->data, in row-major order.
std::vector<std::size_t> offsets(const ggml_tensor* t) {
    std::vector<std::size_t> o;
    o.reserve(static_cast<std::size_t>(ggml_nelements(t)));
    for (std::int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (std::int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (std::int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                for (std::int64_t i0 = 0; i0 < t->ne[0]; ++i0)
                    o.push_back((i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]) /
                                sizeof(float));
    return o;
}

// A poisoned buffer over node's span, for an emulation to write as the GPU
// would write the node's memory.
std::vector<float> poisoned(const ggml_tensor* node) {
    return std::vector<float>(ggml_nbytes(node) / sizeof(float) + 16, kPoison);
}

// The emulation's `got` (laid out like node from node->data) against the CPU's
// node: bit for bit when tol is 0, else d3d12_ssm_max_err <= tol; the floats
// outside the node keep the poison.
void check_out(const ggml_tensor* node, const std::vector<float>& got, double tol) {
    const std::vector<std::size_t> o = offsets(node);
    std::vector<float> g, r;
    std::vector<bool> mine(got.size(), false);
    for (std::size_t off : o) {
        REQUIRE(off < got.size());
        g.push_back(got[off]);
        r.push_back(fdata(node)[off]);
        mine[off] = true;
    }
    if (tol == 0.0) {
        CHECK(std::memcmp(g.data(), r.data(), g.size() * sizeof(float)) == 0);
    } else {
        const double e = d3d12_ssm_max_err(g.data(), r.data(), g.size());
        CAPTURE(e);
        CHECK(e <= tol);
    }
    std::size_t touched = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (!mine[i] && std::memcmp(&got[i], &kPoison, sizeof(float)) != 0)
            ++touched;
    CHECK(touched == 0);
}

bool before(const ggml_cgraph* gf, const ggml_tensor* a, const ggml_tensor* b) {
    int ia = -1, ib = -1;
    for (int i = 0; i < ggml_graph_n_nodes(const_cast<ggml_cgraph*>(gf)); ++i) {
        const ggml_tensor* n = ggml_graph_node(const_cast<ggml_cgraph*>(gf), i);
        if (n == a)
            ia = i;
        if (n == b)
            ib = i;
    }
    return ia >= 0 && ib > ia;
}

void check_unary(const ggml_tensor* node, double tol) {
    D3d12SsmUnaryPlan p;
    REQUIRE(d3d12_ssm_unary_plan(node, &p));
    std::vector<float> out = poisoned(node);
    d3d12_ssm_unary_emulate(p.params, p.groups_x, p.groups_y, fdata(p.src), out.data());
    check_out(node, out, tol);
}

void check_copy(const ggml_tensor* node) {
    D3d12SsmCopyPlan p;
    REQUIRE(d3d12_ssm_copy_plan(node, &p));
    std::vector<float> out = poisoned(node);
    for (const D3d12SsmCopy& c : p.copies) {
        REQUIRE(c.dst_offset % sizeof(float) == 0);
        d3d12_ssm_copy_emulate(c.params, c.groups_x, fdata(c.src),
                               out.data() + c.dst_offset / sizeof(float));
    }
    check_out(node, out, 0.0);
}

struct LayerDims {
    std::int64_t T, S; // tokens per sequence, sequences
    std::int64_t sv = 128, hv = 32, hk = 16, d_conv = 4;
    std::int64_t channels() const {
        return 2 * hk * sv + hv * sv;
    }
};

// One qwen35moe linear-attention layer from its input projections to the
// gated norm, node for node as qwen35moe.cpp build_layer_attn_linear,
// delta-net-base.cpp build_conv_state / build_recurrent_attn /
// build_delta_net_fused and llama-graph.cpp build_rs make it at
// n_rs_seq = 0, kv_head 0, every sequence's state at its own slot.
struct Layer {
    ggml_cgraph* gf = nullptr;
    ggml_tensor* conv_all = nullptr; // the conv and ssm caches
    ggml_tensor* ssm_all = nullptr;
    ggml_tensor* s_copy = nullptr;
    ggml_tensor* conv1d = nullptr;
    ggml_tensor* beta = nullptr;     // SIGMOID
    ggml_tensor* softplus = nullptr; // SOFTPLUS(ADD(alpha, ssm_dt))
    ggml_tensor* gate = nullptr;     // [1, H, T, S] reshape of MUL(softplus, ssm_a)
    ggml_tensor* rs_zero[2] = {};    // build_rs SCALE(view, 0): conv, ssm
    ggml_tensor* rs_rows[2] = {};    // build_rs GET_ROWS
    ggml_tensor* rs_extra[2] = {};   // build_rs CPY of the extra states
    ggml_tensor* concat = nullptr;   // conv_input
    ggml_tensor* conv_cpy = nullptr; // conv_state_last -> conv cache
    ggml_tensor* conv = nullptr;     // SSM_CONV
    ggml_tensor* silu = nullptr;     // SILU(SSM_CONV)
    ggml_tensor* q = nullptr;        // SCALE(RMS_NORM(view)), the l2 norm
    ggml_tensor* k = nullptr;
    ggml_tensor* v = nullptr; // view of silu
    ggml_tensor* gdn = nullptr;
    ggml_tensor* state_cpy = nullptr; // new_state -> ssm cache
    ggml_tensor* z_silu = nullptr;
    ggml_tensor* out = nullptr; // MUL(MUL(RMS_NORM(output), ssm_norm), SILU(z))
};

ggml_tensor* build_rs(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* s, ggml_tensor* s_copy,
                      std::int64_t state_size, std::int64_t n_seqs, ggml_tensor** zero,
                      ggml_tensor** rows, ggml_tensor** extra) {
    const std::int64_t n_rs = n_seqs; // rs_zero = -1: no state to clear
    ggml_tensor* states = ggml_reshape_2d(ctx, s, state_size, s->ne[1]);
    *zero = ggml_scale_inplace(ctx, ggml_view_1d(ctx, states, 0, 0), 0);
    ggml_build_forward_expand(gf, *zero);
    *rows = ggml_get_rows(ctx, states, s_copy);
    ggml_tensor* out = ggml_view_2d(ctx, *rows, state_size, n_seqs, (*rows)->nb[1], 0);
    ggml_build_forward_expand(gf, out);
    ggml_tensor* states_extra = ggml_view_2d(ctx, *rows, state_size, n_rs - n_seqs, (*rows)->nb[1],
                                             n_seqs * (*rows)->nb[1]);
    *extra = ggml_cpy(ctx, states_extra,
                      ggml_view_2d(ctx, s, state_size, n_rs - n_seqs, s->nb[1], n_seqs * s->nb[1]));
    ggml_build_forward_expand(gf, *extra);
    return out;
}

void build_layer(ggml_context* ctx, const LayerDims& d, Layer& L, std::mt19937* rng) {
    const std::int64_t ch = d.channels(), T = d.T, S = d.S, sv = d.sv, hv = d.hv, hk = d.hk;
    const std::int64_t n_embd_r = (d.d_conv - 1) * ch, n_embd_s = sv * sv * hv;
    const ggml_type f = GGML_TYPE_F32;
    L.gf = ggml_new_graph(ctx);

    ggml_tensor* qkv_mm = ggml_new_tensor_2d(ctx, f, ch, T * S);
    ggml_tensor* z_mm = ggml_new_tensor_2d(ctx, f, sv * hv, T * S);
    ggml_tensor* beta_mm = ggml_new_tensor_2d(ctx, f, hv, T * S);
    ggml_tensor* alpha_mm = ggml_new_tensor_2d(ctx, f, hv, T * S);
    ggml_tensor* ssm_dt = ggml_new_tensor_1d(ctx, f, hv);
    ggml_tensor* ssm_a = ggml_new_tensor_1d(ctx, f, hv);
    L.conv1d = ggml_new_tensor_2d(ctx, f, d.d_conv, ch);
    ggml_tensor* ssm_norm = ggml_new_tensor_1d(ctx, f, sv);
    L.conv_all = ggml_new_tensor_2d(ctx, f, n_embd_r, S);
    L.ssm_all = ggml_new_tensor_2d(ctx, f, n_embd_s, S);
    L.s_copy = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    fill(qkv_mm, rng, -1.f, 1.f);
    fill(z_mm, rng, -2.f, 2.f);
    fill(beta_mm, rng, -3.f, 3.f);
    fill(alpha_mm, rng, -3.f, 3.f);
    fill(ssm_dt, rng, -1.f, 1.f);
    fill(ssm_a, rng, -2.f, -0.1f);
    fill(L.conv1d, rng, -0.5f, 0.5f);
    fill(ssm_norm, rng, 0.5f, 1.5f);
    fill(L.conv_all, rng, -1.f, 1.f);
    fill(L.ssm_all, rng, -0.2f, 0.2f);
    if (L.s_copy->data)
        for (std::int64_t s = 0; s < S; ++s) // reversed: the gather moves rows
            static_cast<std::int32_t*>(L.s_copy->data)[s] = static_cast<std::int32_t>(S - 1 - s);

    ggml_tensor* qkv = ggml_reshape_3d(ctx, qkv_mm, ch, T, S);
    L.beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, beta_mm, 1, hv, T, S));
    ggml_tensor* alpha = ggml_reshape_3d(ctx, alpha_mm, hv, T, S);
    L.softplus = ggml_softplus(ctx, ggml_add(ctx, alpha, ssm_dt));
    L.gate = ggml_reshape_4d(ctx, ggml_mul(ctx, L.softplus, ssm_a), 1, hv, T, S);

    ggml_tensor* conv_states = build_rs(ctx, L.gf, L.conv_all, L.s_copy, n_embd_r, S, &L.rs_zero[0],
                                        &L.rs_rows[0], &L.rs_extra[0]);
    conv_states = ggml_reshape_3d(ctx, conv_states, d.d_conv - 1, ch, S);
    L.concat = ggml_concat(ctx, conv_states, ggml_transpose(ctx, qkv), 0);
    ggml_tensor* last = ggml_view_3d(ctx, L.concat, d.d_conv - 1, ch, S, L.concat->nb[1],
                                     L.concat->nb[2], ggml_row_size(f, T));
    L.conv_cpy =
        ggml_cpy(ctx, last, ggml_view_2d(ctx, L.conv_all, n_embd_r, S, L.conv_all->nb[1], 0));
    ggml_build_forward_expand(L.gf, L.conv_cpy);

    ggml_tensor* state = build_rs(ctx, L.gf, L.ssm_all, L.s_copy, n_embd_s, S, &L.rs_zero[1],
                                  &L.rs_rows[1], &L.rs_extra[1]);
    state = ggml_reshape_4d(ctx, state, sv, sv, hv, S);
    L.conv = ggml_ssm_conv(ctx, L.concat, L.conv1d);
    L.silu = ggml_silu(ctx, L.conv);
    const std::size_t nb1 = ggml_row_size(f, ch);
    ggml_tensor* q = ggml_view_4d(ctx, L.silu, sv, hk, T, S, ggml_row_size(f, sv), nb1, nb1 * T, 0);
    ggml_tensor* k = ggml_view_4d(ctx, L.silu, sv, hk, T, S, ggml_row_size(f, sv), nb1, nb1 * T,
                                  ggml_row_size(f, hk * sv));
    L.v = ggml_view_4d(ctx, L.silu, sv, hv, T, S, ggml_row_size(f, sv), nb1, nb1 * T,
                       ggml_row_size(f, 2 * hk * sv));
    const float eps = 1e-6f;
    const float n = static_cast<float>(sv);
    L.q = ggml_scale(ctx, ggml_rms_norm(ctx, q, eps / n), 1.0f / sqrtf(n));
    L.k = ggml_scale(ctx, ggml_rms_norm(ctx, k, eps / n), 1.0f / sqrtf(n));
    L.gdn = ggml_gated_delta_net(ctx, L.q, L.k, L.v, L.gate, L.beta, state, 1);
    ggml_tensor* output = ggml_view_4d(ctx, L.gdn, sv, hv, T, S, ggml_row_size(f, sv),
                                       ggml_row_size(f, sv * hv), ggml_row_size(f, sv * hv * T), 0);
    ggml_tensor* new_state =
        ggml_view_4d(ctx, L.gdn, sv, sv, hv, S, ggml_row_size(f, sv), ggml_row_size(f, sv * sv),
                     ggml_row_size(f, sv * sv * hv), ggml_row_size(f, sv * hv * T * S));
    L.state_cpy =
        ggml_cpy(ctx, new_state, ggml_view_2d(ctx, L.ssm_all, n_embd_s, S, L.ssm_all->nb[1], 0));
    ggml_build_forward_expand(L.gf, L.state_cpy);

    L.z_silu = ggml_silu(ctx, ggml_reshape_4d(ctx, z_mm, sv, hv, T, S));
    L.out = ggml_mul(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, output, eps), ssm_norm), L.z_silu);
    ggml_build_forward_expand(L.gf, L.out);
}

std::uint32_t bits(float f) {
    return ssm_detail::bits(f);
}

} // namespace

TEST_CASE("ggml_d3d12 ssm: constants from a qwen35moe linear-attention layer") {
    // Qwen3.6-35B-A3B at decode: S_v 128, 32 v heads, 16 k heads, 8192 conv
    // channels, d_conv 4, one token of one sequence.
    {
        Ctx c(0, true);
        Layer L;
        build_layer(c.ctx, {1, 1}, L, nullptr);

        D3d12SsmUnaryPlan u;
        REQUIRE(d3d12_ssm_unary_plan(L.beta, &u)); // [1, 32, 1, 1]: one flat row
        CHECK(u.params.op == kD3d12SsmSigmoid);
        CHECK(u.params.ne0 == 32);
        CHECK(u.params.src_row_stride == 0);
        CHECK(u.params.dst_row_stride == 0);
        CHECK(u.groups_x == 1);
        CHECK(u.groups_y == 1);
        REQUIRE(d3d12_ssm_unary_plan(L.softplus, &u));
        CHECK(u.params.op == kD3d12SsmSoftplus);
        CHECK(u.params.ne0 == 32);
        REQUIRE(d3d12_ssm_unary_plan(L.q, &u));
        CHECK(u.params.op == kD3d12SsmScale);
        CHECK(u.params.scale_bits == bits(1.0f / sqrtf(128.0f)));
        CHECK(u.params.ne0 == 128 * 16);
        CHECK(u.groups_x == 8);
        REQUIRE(d3d12_ssm_unary_plan(L.z_silu, &u));
        CHECK(u.params.op == kD3d12SsmSilu);
        CHECK(u.params.ne0 == 128 * 32);
        CHECK(u.groups_x == 16);
        for (const ggml_tensor* zero : L.rs_zero) { // empty: nothing to dispatch
            REQUIRE(d3d12_ssm_unary_plan(zero, &u));
            CHECK(u.params.op == kD3d12SsmScale);
            CHECK(u.groups_x == 0);
            CHECK(u.groups_y == 0);
        }

        D3d12SsmGetRowsPlan g;
        REQUIRE(d3d12_ssm_get_rows_plan(L.rs_rows[0], &g));
        CHECK(g.params.ne0 == 3 * 8192);
        CHECK(g.params.src_row_stride == 3 * 8192);
        CHECK(g.params.dst_row_stride == 3 * 8192);
        CHECK(g.params.ids_stride == 1);
        CHECK(g.groups_x == 96);
        CHECK(g.groups_y == 1);
        CHECK(g.ids == L.s_copy);
        CHECK(g.src->data == L.conv_all->data);
        REQUIRE(d3d12_ssm_get_rows_plan(L.rs_rows[1], &g));
        CHECK(g.params.ne0 == 128 * 128 * 32);
        CHECK(g.groups_x == 2048);

        D3d12SsmCopyPlan cp;
        for (const ggml_tensor* extra : L.rs_extra) {
            REQUIRE(d3d12_ssm_copy_plan(extra, &cp));
            REQUIRE(cp.copies.size() == 1);
            CHECK(cp.copies[0].groups_x == 0);
        }
        REQUIRE(d3d12_ssm_copy_plan(L.concat, &cp));
        REQUIRE(cp.copies.size() == 2);
        const D3d12SsmCopy& a = cp.copies[0]; // conv states [3, 8192, 1]
        CHECK(a.params.ne[0] == 3);
        CHECK(a.params.ne[1] == 8192);
        CHECK(a.params.ne[2] == 1);
        CHECK(a.params.src_stride[0] == 1);
        CHECK(a.params.src_stride[1] == 3);
        CHECK(a.params.src_stride[2] == 0);
        CHECK(a.params.dst_stride[1] == 4);
        CHECK(a.dst_offset == 0);
        CHECK(a.groups_x == 96);
        const D3d12SsmCopy& b = cp.copies[1]; // TRANSPOSE(qkv) [1, 8192, 1]
        CHECK(b.params.ne[0] == 1);
        CHECK(b.params.ne[1] == 8192);
        CHECK(b.params.src_stride[0] == 0);
        CHECK(b.params.src_stride[1] == 1);
        CHECK(b.params.dst_stride[1] == 4);
        CHECK(b.dst_offset == 3 * sizeof(float));
        CHECK(b.groups_x == 32);
        CHECK(b.params.pad0 == 0);
        REQUIRE(d3d12_ssm_copy_plan(L.conv_cpy, &cp)); // [3, 8192] at 1 -> packed cache row
        REQUIRE(cp.copies.size() == 1);
        CHECK(cp.copies[0].params.ne[0] == 3);
        CHECK(cp.copies[0].params.ne[1] == 8192);
        CHECK(cp.copies[0].params.src_stride[1] == 4);
        CHECK(cp.copies[0].params.dst_stride[1] == 3);
        CHECK(cp.copies[0].groups_x == 96);
        REQUIRE(d3d12_ssm_copy_plan(L.state_cpy, &cp));
        REQUIRE(cp.copies.size() == 1);
        CHECK(cp.copies[0].params.ne[2] == 32);
        CHECK(cp.copies[0].params.src_stride[2] == 128 * 128);
        CHECK(cp.copies[0].params.dst_stride[2] == 128 * 128);
        CHECK(cp.copies[0].groups_x == 2048);

        D3d12SsmConvPlan cv;
        REQUIRE(d3d12_ssm_conv_plan(L.silu, &cv));
        CHECK(cv.params.silu == 1);
        CHECK(cv.params.channels == 8192);
        CHECK(cv.params.d_conv == 4);
        CHECK(cv.params.x_channel_stride == 4);
        CHECK(cv.params.x_seq_stride == 4 * 8192);
        CHECK(cv.params.w_channel_stride == 4);
        CHECK(cv.params.dst_token_stride == 8192);
        CHECK(cv.params.dst_seq_stride == 8192);
        CHECK(cv.groups_x == 32);
        CHECK(cv.groups_y == 1);
        CHECK(cv.groups_z == 1);
        CHECK(cv.x == L.concat);
        CHECK(cv.w == L.conv1d);
        REQUIRE(cv.nodes.size() == 2);
        CHECK(cv.nodes[0] == L.conv);
        CHECK(cv.nodes[1] == L.silu);
        CHECK(before(L.gf, L.conv, L.silu));
        REQUIRE(d3d12_ssm_conv_plan(L.conv, &cv));
        CHECK(cv.params.silu == 0);
        CHECK(cv.nodes.size() == 1);

        D3d12GdnPlan gd;
        REQUIRE(d3d12_gdn_plan(L.gdn, &gd));
        CHECK(gd.params.h == 32);
        CHECK(gd.params.t == 1);
        CHECK(gd.params.b == 1);
        CHECK(gd.params.k_kda == 1);
        CHECK(gd.params.heads_qk == (16u | 16u << 16));
        CHECK(gd.params.seq_div_qk == (1u | 1u << 16));
        CHECK(gd.params.q_token_stride == 128 * 16);
        CHECK(gd.params.q_seq_stride == 128 * 16);
        CHECK(gd.params.k_token_stride == 128 * 16);
        CHECK(gd.params.v_token_stride == 8192);
        CHECK(gd.params.v_seq_stride == 8192);
        CHECK(gd.params.scale_bits == bits(1.0f / sqrtf(128.0f)));
        CHECK(gd.groups_x == 4);
        CHECK(gd.groups_y == 32);
        CHECK(gd.groups_z == 1);
        CHECK(gd.q == L.q);
        CHECK(gd.v == L.v);
        CHECK(gd.g == L.gate);
        CHECK(gd.beta == L.beta);
    }
    // Two sequences of three tokens.
    {
        Ctx c(0, true);
        Layer L;
        build_layer(c.ctx, {3, 2}, L, nullptr);
        D3d12SsmCopyPlan cp;
        REQUIRE(d3d12_ssm_copy_plan(L.concat, &cp));
        REQUIRE(cp.copies.size() == 2);
        const D3d12SsmCopy& b = cp.copies[1]; // TRANSPOSE(qkv) [3, 8192, 2]
        CHECK(b.params.ne[0] == 3);
        CHECK(b.params.ne[2] == 2);
        CHECK(b.params.src_stride[0] == 8192);
        CHECK(b.params.src_stride[1] == 1);
        CHECK(b.params.src_stride[2] == 3 * 8192);
        CHECK(b.params.dst_stride[1] == 6);
        CHECK(b.params.dst_stride[2] == 6 * 8192);
        CHECK(b.dst_offset == 3 * sizeof(float));
        D3d12SsmConvPlan cv;
        REQUIRE(d3d12_ssm_conv_plan(L.silu, &cv));
        CHECK(cv.params.x_channel_stride == 6);
        CHECK(cv.params.x_seq_stride == 6 * 8192);
        CHECK(cv.params.dst_token_stride == 8192);
        CHECK(cv.params.dst_seq_stride == 3 * 8192);
        CHECK(cv.groups_y == 3);
        CHECK(cv.groups_z == 2);
        D3d12GdnPlan gd;
        REQUIRE(d3d12_gdn_plan(L.gdn, &gd));
        CHECK(gd.params.t == 3);
        CHECK(gd.params.b == 2);
        CHECK(gd.params.q_token_stride == 128 * 16);
        CHECK(gd.params.q_seq_stride == 128 * 16 * 3);
        CHECK(gd.params.v_token_stride == 8192);
        CHECK(gd.params.v_seq_stride == 8192 * 3);
        CHECK(gd.groups_z == 2);
        D3d12SsmUnaryPlan u;
        REQUIRE(d3d12_ssm_unary_plan(L.beta, &u));
        CHECK(u.params.ne0 == 32 * 3 * 2);
        CHECK(u.groups_y == 1);
    }
}

TEST_CASE("ggml_d3d12 ssm: nodes the kernels do not take") {
    D3d12SsmUnaryPlan u;
    D3d12SsmCopyPlan cp;
    D3d12SsmGetRowsPlan g;
    D3d12SsmConvPlan cv;
    D3d12GdnPlan gd;
    CHECK_FALSE(d3d12_ssm_unary_plan(nullptr, &u));
    CHECK_FALSE(d3d12_ssm_copy_plan(nullptr, &cp));
    CHECK_FALSE(d3d12_ssm_get_rows_plan(nullptr, &g));
    CHECK_FALSE(d3d12_ssm_conv_plan(nullptr, &cv));
    CHECK_FALSE(d3d12_gdn_plan(nullptr, &gd));

    Ctx c(0, true);
    ggml_context* ctx = c.ctx;
    const ggml_type f = GGML_TYPE_F32;
    ggml_tensor* a = ggml_new_tensor_2d(ctx, f, 64, 3);
    CHECK_FALSE(d3d12_ssm_unary_plan(ggml_gelu(ctx, a), &u));
    CHECK_FALSE(d3d12_ssm_unary_plan(ggml_scale_bias(ctx, a, 2.f, 1.f), &u)); // a bias: mad
    CHECK_FALSE(
        d3d12_ssm_unary_plan(ggml_sigmoid(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 3)), &u));
    CHECK_FALSE(d3d12_ssm_unary_plan(ggml_add(ctx, a, a), &u));

    CHECK_FALSE(
        d3d12_ssm_copy_plan(ggml_cpy(ctx, a, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 3)), &cp));
    {
        // Both sides strided, shapes differ: no order to match the elements in.
        ggml_tensor* src = ggml_transpose(ctx, ggml_new_tensor_2d(ctx, f, 4, 6)); // [6, 4]
        ggml_tensor* big = ggml_new_tensor_2d(ctx, f, 8, 8);
        ggml_tensor* dst = ggml_view_2d(ctx, big, 3, 8, big->nb[1], 0); // [3, 8], rows 8 apart
        CHECK_FALSE(d3d12_ssm_copy_plan(ggml_cpy(ctx, src, dst), &cp));
    }
    CHECK_FALSE(d3d12_ssm_copy_plan(a, &cp)); // a leaf

    {
        ggml_tensor* src3 = ggml_new_tensor_3d(ctx, f, 8, 4, 2);
        ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 2);
        CHECK_FALSE(d3d12_ssm_get_rows_plan(ggml_get_rows(ctx, src3, ids), &g)); // 3-d
        ggml_tensor* h = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 4);
        CHECK_FALSE(d3d12_ssm_get_rows_plan(
            ggml_get_rows(ctx, h, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2)), &g)); // f16 rows
    }

    {
        ggml_tensor* x = ggml_new_tensor_3d(ctx, f, 3 + 2, 16, 1);
        ggml_tensor* w = ggml_new_tensor_2d(ctx, f, 4, 16);
        ggml_tensor* conv = ggml_ssm_conv(ctx, x, w);
        CHECK_FALSE(d3d12_ssm_conv_plan(ggml_sigmoid(ctx, conv), &cv)); // not its SILU
        CHECK_FALSE(d3d12_ssm_conv_plan(ggml_silu(ctx, a), &cv));       // SILU of no conv
    }

    {
        // S_v = 64: the kernel is built for 128.
        ggml_tensor* q = ggml_new_tensor_4d(ctx, f, 64, 2, 1, 1);
        ggml_tensor* v = ggml_new_tensor_4d(ctx, f, 64, 2, 1, 1);
        ggml_tensor* gg = ggml_new_tensor_4d(ctx, f, 1, 2, 1, 1);
        ggml_tensor* s = ggml_new_tensor_4d(ctx, f, 64, 64, 2, 1);
        CHECK_FALSE(d3d12_gdn_plan(ggml_gated_delta_net(ctx, q, q, v, gg, gg, s, 1), &gd));
    }
    {
        // q heads 256 floats apart (a view of every other head).
        ggml_tensor* wide = ggml_new_tensor_4d(ctx, f, 128, 4, 1, 1);
        ggml_tensor* q =
            ggml_view_4d(ctx, wide, 128, 2, 1, 1, 2 * wide->nb[1], wide->nb[2], wide->nb[3], 0);
        ggml_tensor* v = ggml_new_tensor_4d(ctx, f, 128, 2, 1, 1);
        ggml_tensor* gg = ggml_new_tensor_4d(ctx, f, 1, 2, 1, 1);
        ggml_tensor* s = ggml_new_tensor_4d(ctx, f, 128, 128, 2, 1);
        CHECK_FALSE(d3d12_gdn_plan(ggml_gated_delta_net(ctx, q, v, v, gg, gg, s, 1), &gd));
        CHECK(d3d12_gdn_plan(ggml_gated_delta_net(ctx, v, v, v, gg, gg, s, 1), &gd));
    }
}

TEST_CASE("ggml_d3d12 ssm: kernel emulation matches ggml's CPU ops on a qwen35moe layer") {
    const LayerDims dims[] = {{1, 1}, {3, 2}};
    for (const LayerDims& d : dims) {
        CAPTURE(d.T);
        CAPTURE(d.S);
        Ctx c(160u << 20, false);
        Layer L;
        std::mt19937 rng(1234u + static_cast<unsigned>(d.T * 7 + d.S));
        build_layer(c.ctx, d, L, &rng);
        // The CPYs write the caches back: keep what GET_ROWS read.
        const std::vector<float> conv_before(fdata(L.conv_all),
                                             fdata(L.conv_all) + ggml_nelements(L.conv_all));
        const std::vector<float> ssm_before(fdata(L.ssm_all),
                                            fdata(L.ssm_all) + ggml_nelements(L.ssm_all));
        compute(c.ctx, L.gf);

        check_unary(L.beta, 1e-5);
        check_unary(L.softplus, 1e-5);
        check_unary(L.z_silu, 1e-5);
        check_unary(L.q, 0.0); // SCALE without a bias: one multiply, exact
        check_unary(L.k, 0.0);

        const std::vector<float>* before_rows[2] = {&conv_before, &ssm_before};
        for (int i = 0; i < 2; ++i) {
            D3d12SsmGetRowsPlan g;
            REQUIRE(d3d12_ssm_get_rows_plan(L.rs_rows[i], &g));
            std::vector<float> out = poisoned(L.rs_rows[i]);
            d3d12_ssm_get_rows_emulate(g.params, g.groups_x, g.groups_y, before_rows[i]->data(),
                                       static_cast<const std::int32_t*>(g.ids->data), out.data());
            check_out(L.rs_rows[i], out, 0.0);
        }

        check_copy(L.concat);
        check_copy(L.conv_cpy);
        check_copy(L.state_cpy);

        for (const ggml_tensor* node : {L.silu, L.conv}) {
            D3d12SsmConvPlan p;
            REQUIRE(d3d12_ssm_conv_plan(node, &p));
            std::vector<float> out = poisoned(node);
            d3d12_ssm_conv_emulate(p.params, p.groups_x, p.groups_y, p.groups_z, fdata(p.x),
                                   fdata(p.w), out.data());
            CAPTURE(p.params.silu);
            check_out(node, out, 1e-5);
        }

        D3d12GdnPlan p;
        REQUIRE(d3d12_gdn_plan(L.gdn, &p));
        std::vector<float> out = poisoned(L.gdn);
        d3d12_gdn_emulate(p.params, p.groups_x, p.groups_y, p.groups_z, fdata(p.q), fdata(p.k),
                          fdata(p.v), fdata(p.g), fdata(p.beta), fdata(p.state), out.data());
        check_out(L.gdn, out, 1e-4);
    }
}

namespace {

struct GdnCase {
    int hv, hk, T, B, qk_seqs, K;
    bool kda;
    int v_pad; // v is a view of a tensor v_pad heads wider, starting one head in
};

} // namespace

TEST_CASE("ggml_d3d12 ssm: gated delta net emulation matches ggml's CPU op") {
    const GdnCase cases[] = {
        {4, 2, 1, 1, 1, 1, false, 0}, // decode
        {4, 2, 4, 2, 2, 1, false, 1}, // 4 tokens of 2 sequences, v strided
        {4, 4, 3, 2, 1, 1, true, 0},  // per-channel gate, q and k shared by both sequences
        {2, 1, 4, 1, 1, 2, false, 0}, // 2 snapshots of 4 tokens
        {2, 2, 1, 2, 2, 3, true, 2},  // 3 slots, 1 token: slot 0 only
    };
    for (const GdnCase& gc : cases) {
        CAPTURE(gc.hv);
        CAPTURE(gc.T);
        CAPTURE(gc.B);
        CAPTURE(gc.K);
        CAPTURE(gc.kda);
        const std::int64_t sv = 128;
        Ctx c(64u << 20, false);
        ggml_context* ctx = c.ctx;
        const ggml_type f = GGML_TYPE_F32;
        std::mt19937 rng(99u + static_cast<unsigned>(gc.hv * 3 + gc.T * 5 + gc.B * 7 + gc.K));
        ggml_tensor* q = ggml_new_tensor_4d(ctx, f, sv, gc.hk, gc.T, gc.qk_seqs);
        ggml_tensor* k = ggml_new_tensor_4d(ctx, f, sv, gc.hk, gc.T, gc.qk_seqs);
        ggml_tensor* vb = ggml_new_tensor_4d(ctx, f, sv * (gc.hv + gc.v_pad), gc.T, gc.B, 1);
        ggml_tensor* v = ggml_view_4d(ctx, vb, sv, gc.hv, gc.T, gc.B, ggml_row_size(f, sv),
                                      vb->nb[1], vb->nb[2], gc.v_pad ? ggml_row_size(f, sv) : 0);
        ggml_tensor* g = ggml_new_tensor_4d(ctx, f, gc.kda ? sv : 1, gc.hv, gc.T, gc.B);
        ggml_tensor* beta = ggml_new_tensor_4d(ctx, f, 1, gc.hv, gc.T, gc.B);
        ggml_tensor* state = ggml_new_tensor_4d(ctx, f, sv, sv, gc.hv, gc.B);
        fill_unit_rows(q, &rng);
        fill_unit_rows(k, &rng);
        fill(vb, &rng, -1.f, 1.f);
        fill(g, &rng, -1.f, 0.f);
        fill(beta, &rng, 0.f, 1.f);
        fill(state, &rng, -0.2f, 0.2f);
        ggml_tensor* gdn = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, gc.K);
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, gdn);
        compute(ctx, gf);

        D3d12GdnPlan p;
        REQUIRE(d3d12_gdn_plan(gdn, &p));
        CHECK(p.params.k_kda == (static_cast<std::uint32_t>(gc.K) | (gc.kda ? 1u << 16 : 0u)));
        std::vector<float> out = poisoned(gdn);
        d3d12_gdn_emulate(p.params, p.groups_x, p.groups_y, p.groups_z, fdata(p.q), fdata(p.k),
                          fdata(p.v), fdata(p.g), fdata(p.beta), fdata(p.state), out.data());
        // The attention rows, then the slots the op writes (min(T, K)); the
        // rest is the caller's and stays as it was (the poison here).
        const std::size_t attn = static_cast<std::size_t>(sv * gc.hv * gc.T * gc.B);
        const std::size_t per_snap = static_cast<std::size_t>(sv * sv * gc.hv * gc.B);
        const std::size_t n = attn + static_cast<std::size_t>(std::min(gc.T, gc.K)) * per_snap;
        const double e = d3d12_ssm_max_err(out.data(), fdata(gdn), n);
        CAPTURE(e);
        CHECK(e <= 1e-4);
        std::size_t touched = 0;
        for (std::size_t i = n; i < out.size(); ++i)
            if (std::memcmp(&out[i], &kPoison, sizeof(float)) != 0)
                ++touched;
        CHECK(touched == 0);
    }
}

namespace {

// A linear-attention layer's state round trip as build_rs and the K = 1 path of
// build_recurrent_attn make it at one sequence: the cache rows reshaped, an
// empty clear, GET_ROWS of the ubatch's state, the empty copy of the extra
// states, the op, the attention rows read by a SCALE, and the CPY of the new
// state into cache row kv.
struct GdnRoundTrip {
    ggml_tensor *s = nullptr, *s_copy = nullptr, *states = nullptr, *states_all = nullptr;
    ggml_tensor *gdn = nullptr, *new_state = nullptr, *cpy_dst = nullptr, *cpy = nullptr;
    ggml_tensor *output = nullptr;
    ggml_cgraph* gf = nullptr;
    int gdn_idx = -1;
};

enum class GdnBreak { none, gather_reader, cache_reader, whole_reader, k2 };

GdnRoundTrip gdn_round_trip(ggml_context* ctx, std::mt19937* rng, int hv, int hk, int T, bool kda,
                            int cells, int src_row, int kv, GdnBreak brk) {
    const std::int64_t sv = 128;
    const std::int64_t n = sv * sv * hv;
    const ggml_type f = GGML_TYPE_F32;
    GdnRoundTrip r;
    r.gf = ggml_new_graph(ctx);
    r.s = ggml_new_tensor_2d(ctx, f, n, cells);
    r.s_copy = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(r.s_copy);
    fill(r.s, rng, -0.2f, 0.2f);
    if (r.s_copy->data)
        *static_cast<std::int32_t*>(r.s_copy->data) = src_row;
    ggml_tensor* q = ggml_new_tensor_4d(ctx, f, sv, hk, T, 1);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, f, sv, hk, T, 1);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, f, sv, hv, T, 1);
    ggml_tensor* g = ggml_new_tensor_4d(ctx, f, kda ? sv : 1, hv, T, 1);
    ggml_tensor* beta = ggml_new_tensor_4d(ctx, f, 1, hv, T, 1);
    fill_unit_rows(q, rng);
    fill_unit_rows(k, rng);
    fill(v, rng, -1.f, 1.f);
    fill(g, rng, -1.f, 0.f);
    fill(beta, rng, 0.f, 1.f);

    r.states = ggml_reshape_2d(ctx, r.s, n, r.s->ne[1]);
    ggml_build_forward_expand(r.gf, ggml_scale_inplace(ctx, ggml_view_1d(ctx, r.states, 0, 0), 0.f));
    r.states_all = ggml_get_rows(ctx, r.states, r.s_copy);
    ggml_tensor* out_states =
        ggml_view_2d(ctx, r.states_all, n, 1, r.states_all->nb[1], 0);
    ggml_build_forward_expand(r.gf, out_states);
    ggml_tensor* extra = ggml_view_2d(ctx, r.states_all, n, 0, r.states_all->nb[1], r.states_all->nb[1]);
    ggml_build_forward_expand(
        r.gf, ggml_cpy(ctx, extra, ggml_view_2d(ctx, r.s, n, 0, r.s->nb[1], (kv + 1) * r.s->nb[1])));
    if (brk == GdnBreak::gather_reader)
        ggml_build_forward_expand(r.gf, ggml_scale(ctx, r.states_all, 2.f));
    ggml_tensor* state = ggml_reshape_4d(ctx, out_states, sv, sv, hv, 1);
    r.gdn = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, brk == GdnBreak::k2 ? 2 : 1);
    const std::size_t attn_bytes = ggml_row_size(f, sv * hv * T);
    r.output = ggml_view_4d(ctx, r.gdn, sv, hv, T, 1, ggml_row_size(f, sv), ggml_row_size(f, sv * hv),
                            attn_bytes, 0);
    r.new_state = ggml_view_4d(ctx, r.gdn, sv, sv, hv, 1, ggml_row_size(f, sv),
                               ggml_row_size(f, sv * sv), ggml_row_size(f, n), attn_bytes);
    if (brk == GdnBreak::cache_reader) { // after the op, before the CPY
        ggml_build_forward_expand(r.gf, r.gdn);
        ggml_build_forward_expand(r.gf, ggml_scale(ctx, ggml_view_1d(ctx, r.s, 8, 0), 3.f));
    }
    if (brk == GdnBreak::whole_reader)
        ggml_build_forward_expand(r.gf, ggml_scale(ctx, r.gdn, 1.f));
    r.cpy_dst = ggml_view_2d(ctx, r.s, n, 1, r.s->nb[1], static_cast<std::size_t>(kv) * r.s->nb[1]);
    r.cpy = ggml_cpy(ctx, r.new_state, r.cpy_dst);
    ggml_build_forward_expand(r.gf, r.cpy);
    ggml_build_forward_expand(r.gf, ggml_scale(ctx, r.output, 1.f));
    for (int i = 0; i < ggml_graph_n_nodes(r.gf); ++i)
        if (ggml_graph_node(r.gf, i) == r.gdn)
            r.gdn_idx = i;
    return r;
}

// ggml's use counts: how many sources of the graph's nodes are t.
auto graph_uses(ggml_cgraph* gf) {
    return [gf](const ggml_tensor* t) {
        int c = 0;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i)
            for (const ggml_tensor* s : ggml_graph_node(gf, i)->src)
                c += s == t;
        return c;
    };
}

} // namespace

TEST_CASE("ggml_d3d12 ssm: the delta net reads and writes the recurrent cache rows") {
    struct Case {
        int hv, hk, T;
        bool kda;
        int src_row, kv;
    };
    const Case cases[] = {
        {4, 2, 1, false, 1, 1}, // decode, the state read and written in place
        {4, 2, 1, true, 0, 2},  // a sequence's first state, copied from another cell
        {2, 2, 3, false, 2, 2}, // 3 tokens
        {2, 1, 3, true, 2, 0},
    };
    const int cells = 3;
    for (const Case& cs : cases) {
        CAPTURE(cs.hv);
        CAPTURE(cs.T);
        CAPTURE(cs.kda);
        CAPTURE(cs.src_row);
        CAPTURE(cs.kv);
        Ctx c(32u << 20, false);
        std::mt19937 rng(7u + static_cast<unsigned>(cs.hv * 11 + cs.T * 13 + cs.src_row * 3 + cs.kv));
        GdnRoundTrip r =
            gdn_round_trip(c.ctx, &rng, cs.hv, cs.hk, cs.T, cs.kda, cells, cs.src_row, cs.kv, GdnBreak::none);
        REQUIRE(r.gdn_idx >= 0);
        const std::size_t n = static_cast<std::size_t>(r.s->ne[0]);
        const std::vector<float> before(fdata(r.s), fdata(r.s) + n * cells);

        D3d12GdnFusePlan fp;
        REQUIRE(d3d12_gdn_fuse_plan(r.gf, r.gdn_idx, graph_uses(r.gf), &fp));
        CHECK(fp.get_rows == r.states_all);
        CHECK(fp.cpy == r.cpy);
        CHECK(fp.states == r.states);
        CHECK(fp.ids == r.s_copy);
        CHECK(fp.dst == r.cpy_dst);

        compute(c.ctx, r.gf);

        D3d12GdnPlan p;
        REQUIRE(d3d12_gdn_plan(r.gdn, &p));
        p.params.k_kda |= kD3d12GdnFused;
        std::vector<float> out = poisoned(r.gdn);
        std::vector<float> cache = before;
        d3d12_gdn_emulate(p.params, p.groups_x, p.groups_y, p.groups_z, fdata(p.q), fdata(p.k),
                          fdata(p.v), fdata(p.g), fdata(p.beta), before.data() + cs.src_row * n,
                          out.data(), cache.data() + cs.kv * n);
        // The attention rows as the CPU's; the op's state rows stay unwritten.
        const std::size_t attn = static_cast<std::size_t>(128 * cs.hv * cs.T);
        const double e_attn = d3d12_ssm_max_err(out.data(), fdata(r.gdn), attn);
        CAPTURE(e_attn);
        CHECK(e_attn <= 1e-4);
        std::size_t touched = 0;
        for (std::size_t i = attn; i < out.size(); ++i)
            if (std::memcmp(&out[i], &kPoison, sizeof(float)) != 0)
                ++touched;
        CHECK(touched == 0);
        // Cache row kv holds the CPU's new state, every other row is as it was.
        const double e_state = d3d12_ssm_max_err(cache.data() + cs.kv * n, fdata(r.s) + cs.kv * n, n);
        CAPTURE(e_state);
        CHECK(e_state <= 1e-4);
        for (int row = 0; row < cells; ++row) {
            if (row == cs.kv)
                continue;
            CAPTURE(row);
            CHECK(std::memcmp(cache.data() + row * n, before.data() + row * n, n * sizeof(float)) == 0);
            CHECK(std::memcmp(fdata(r.s) + row * n, before.data() + row * n, n * sizeof(float)) == 0);
        }
    }
}

TEST_CASE("ggml_d3d12 ssm: round trips the fused delta net does not take") {
    const GdnBreak breaks[] = {GdnBreak::gather_reader, GdnBreak::cache_reader, GdnBreak::whole_reader,
                               GdnBreak::k2};
    for (GdnBreak b : breaks) {
        CAPTURE(static_cast<int>(b));
        Ctx c(16u << 20, true);
        GdnRoundTrip r = gdn_round_trip(c.ctx, nullptr, 2, 2, 1, false, 3, 1, 1, b);
        REQUIRE(r.gdn_idx >= 0);
        D3d12GdnFusePlan fp;
        CHECK_FALSE(d3d12_gdn_fuse_plan(r.gf, r.gdn_idx, graph_uses(r.gf), &fp));
    }
    {
        // A graph output among the views, and a split's partial use counts.
        Ctx c(16u << 20, true);
        GdnRoundTrip r = gdn_round_trip(c.ctx, nullptr, 2, 2, 1, false, 3, 1, 1, GdnBreak::none);
        D3d12GdnFusePlan fp;
        REQUIRE(d3d12_gdn_fuse_plan(r.gf, r.gdn_idx, graph_uses(r.gf), &fp));
        auto more = [&](const ggml_tensor* t) { return graph_uses(r.gf)(t) + (t == r.states_all); };
        CHECK_FALSE(d3d12_gdn_fuse_plan(r.gf, r.gdn_idx, more, &fp));
        ggml_set_output(r.new_state);
        CHECK_FALSE(d3d12_gdn_fuse_plan(r.gf, r.gdn_idx, graph_uses(r.gf), &fp));
    }
}

TEST_CASE("ggml_d3d12 ssm: strided rows and copies") {
    Ctx c(16u << 20, false);
    ggml_context* ctx = c.ctx;
    const ggml_type f = GGML_TYPE_F32;
    std::mt19937 rng(5u);
    ggml_tensor* big = ggml_new_tensor_2d(ctx, f, 300, 5);
    fill(big, &rng, -4.f, 4.f);
    // 290 of 300 floats per row: rows evenly spaced but not packed, and 290
    // columns leave the second group 34 of 256 threads.
    ggml_tensor* rows = ggml_view_2d(ctx, big, 290, 5, big->nb[1], ggml_row_size(f, 3));
    ggml_tensor* sg = ggml_sigmoid(ctx, rows);
    ggml_tensor* sp = ggml_softplus(ctx, ggml_scale(ctx, big, 8.f)); // past 20: x itself
    ggml_tensor* packed = ggml_new_tensor_1d(ctx, f, 24);
    fill(packed, &rng, -1.f, 1.f);
    ggml_tensor* holder = ggml_new_tensor_2d(ctx, f, 8, 8);
    fill(holder, &rng, -1.f, 1.f);
    ggml_tensor* strided_dst = ggml_view_2d(ctx, holder, 3, 8, holder->nb[1], ggml_row_size(f, 2));
    ggml_tensor* into_strided = ggml_cpy(ctx, packed, strided_dst); // src contiguous, dst not
    ggml_tensor* cont = ggml_cont(ctx, ggml_transpose(ctx, rows));  // dst contiguous, src not
    ggml_tensor* cat1 = ggml_concat(ctx, ggml_new_tensor_3d(ctx, f, 4, 2, 3),
                                    ggml_new_tensor_3d(ctx, f, 4, 5, 3), 1);
    fill(cat1->src[0], &rng, -1.f, 1.f);
    fill(cat1->src[1], &rng, -1.f, 1.f);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    for (ggml_tensor* t : {sg, sp, into_strided, cont, cat1})
        ggml_build_forward_expand(gf, t);
    compute(ctx, gf);

    D3d12SsmUnaryPlan u;
    REQUIRE(d3d12_ssm_unary_plan(sg, &u));
    CHECK(u.params.src_row_stride == 300);
    CHECK(u.params.dst_row_stride == 290);
    CHECK(u.groups_x == 2);
    CHECK(u.groups_y == 5);
    check_unary(sg, 1e-5);
    check_unary(sp->src[0], 0.0);
    check_unary(sp, 1e-5);
    check_copy(into_strided);
    check_copy(cont);
    D3d12SsmCopyPlan cp;
    REQUIRE(d3d12_ssm_copy_plan(cat1, &cp));
    REQUIRE(cp.copies.size() == 2);
    CHECK(cp.copies[1].dst_offset == 2 * cat1->nb[1]);
    check_copy(cat1);
}
