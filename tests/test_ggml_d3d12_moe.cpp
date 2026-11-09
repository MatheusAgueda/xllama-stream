// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "../src/bridge/ggml_d3d12_moe_emu.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace xllama;

namespace {

struct MoeCase {
    int ne0, n, n_tok;
    int n_expert;   // > 0: the weights come from a router as in build_moe_ffn, else random
    bool w_one_col; // random weights shared by a token's experts, [1, 1, T]
    int n_summed;   // views the ADD chain sums (n for llama's chain)
};

// The tail of llama-graph.cpp build_moe_ffn (with norm_w, softmax gating) and
// of qwen35moe.cpp build_layer_ffn, node for node and in the same expand order.
struct MoeGraph {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor* moe_out = nullptr; // A_(n_summed - 1)
    ggml_tensor* before_last = nullptr;
    ggml_tensor* gated = nullptr;   // G
    ggml_tensor* ffn_out = nullptr; // O
    ggml_tensor* post = nullptr;    // P
    MoeGraph() = default;
    MoeGraph(const MoeGraph&) = delete;
    MoeGraph& operator=(const MoeGraph&) = delete;
    ~MoeGraph() {
        if (ctx)
            ggml_free(ctx);
    }
};

void fill(ggml_tensor* t, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    auto* f = static_cast<float*>(t->data);
    for (std::int64_t i = 0; i < ggml_nelements(t); ++i)
        f[i] = d(rng);
}

void build(MoeGraph& g, const MoeCase& c, bool no_alloc) {
    const std::size_t big = sizeof(float) * static_cast<std::size_t>(c.ne0) * c.n * c.n_tok;
    ggml_init_params ip{};
    ip.mem_size = (no_alloc ? 0 : 4 * big + 64 * sizeof(float) * (c.n_expert + c.n + 4) * c.n_tok) +
                  256 * ggml_tensor_overhead() + ggml_graph_overhead() + (4u << 20);
    ip.no_alloc = no_alloc;
    g.ctx = ggml_init(ip);
    REQUIRE(g.ctx != nullptr);
    std::mt19937 rng(77u + static_cast<unsigned>(c.ne0 * 3 + c.n * 5 + c.n_tok * 7 + c.n_expert));
    ggml_context* ctx = g.ctx;
    g.gf = ggml_new_graph(ctx);

    ggml_tensor* e = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.ne0, c.n, c.n_tok);
    ggml_tensor* weights;
    ggml_tensor* logits = nullptr;
    if (c.n_expert > 0) {
        logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.n_expert, c.n_tok);
        ggml_tensor* probs = ggml_soft_max(ctx, logits);
        ggml_tensor* selected = ggml_argsort_top_k(ctx, probs, c.n);
        probs = ggml_reshape_3d(ctx, probs, 1, c.n_expert, c.n_tok);
        weights = ggml_get_rows(ctx, probs, selected); // [1, n, T]
        weights = ggml_reshape_2d(ctx, weights, c.n, c.n_tok);
        ggml_tensor* weights_sum = ggml_sum_rows(ctx, weights);
        weights = ggml_div(ctx, weights, weights_sum);
        weights = ggml_reshape_3d(ctx, weights, 1, c.n, c.n_tok);
    } else {
        weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, c.w_one_col ? 1 : c.n, c.n_tok);
    }
    ggml_tensor* experts = ggml_mul(ctx, e, weights);
    ggml_build_forward_expand(g.gf, experts);
    std::vector<ggml_tensor*> views(static_cast<std::size_t>(c.n));
    for (int i = 0; i < c.n; ++i) {
        views[static_cast<std::size_t>(i)] =
            ggml_view_2d(ctx, experts, c.ne0, c.n_tok, experts->nb[2], i * experts->nb[1]);
        ggml_build_forward_expand(g.gf, views[static_cast<std::size_t>(i)]);
    }
    ggml_tensor* moe_out = views[0];
    for (int i = 1; i < c.n_summed; ++i) {
        g.before_last = moe_out;
        moe_out = ggml_add(ctx, moe_out, views[static_cast<std::size_t>(i)]);
        ggml_build_forward_expand(g.gf, moe_out);
    }
    g.moe_out = moe_out;

    ggml_tensor* shexp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, c.n_tok);
    ggml_tensor* gate_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, c.n_tok);
    ggml_tensor* residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, c.n_tok);
    ggml_tensor* gate = ggml_sigmoid(ctx, gate_in);
    g.gated = ggml_mul(ctx, shexp, gate);
    g.ffn_out = ggml_add(ctx, moe_out, g.gated);
    g.post = ggml_add(ctx, g.ffn_out, residual);
    ggml_build_forward_expand(g.gf, g.post);
    if (no_alloc)
        return;

    fill(e, rng, -1.f, 1.f);
    if (logits)
        fill(logits, rng, -4.f, 4.f);
    else
        fill(weights, rng, 0.01f, 0.5f);
    fill(shexp, rng, -1.f, 1.f);
    fill(gate_in, rng, -3.f, 3.f);
    fill(residual, rng, -2.f, 2.f);
}

int graph_index(ggml_cgraph* gf, const ggml_tensor* t) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i)
        if (ggml_graph_node(gf, i) == t)
            return i;
    return -1;
}

// The plan's nodes are graph nodes in graph order, the output last.
bool nodes_in_order(ggml_cgraph* gf, const std::vector<const ggml_tensor*>& nodes,
                    const ggml_tensor* out) {
    int prev = -1;
    for (const ggml_tensor* t : nodes) {
        const int i = graph_index(gf, t);
        if (i <= prev)
            return false;
        prev = i;
    }
    return !nodes.empty() && nodes.back() == out;
}

// Bit equality of a [ne0, rows] result written with dst_stride floats per row.
bool same_bits(const ggml_tensor* ref, const std::vector<float>& got, std::uint32_t stride) {
    const std::int64_t rows = ggml_nrows(ref);
    for (std::int64_t r = 0; r < rows; ++r) {
        const auto* row = reinterpret_cast<const char*>(ref->data) + r * ref->nb[1];
        if (std::memcmp(row, got.data() + r * stride, sizeof(float) * ref->ne[0]) != 0)
            return false;
    }
    return true;
}

void compute(MoeGraph& g) {
    ggml_cpu_init();
    REQUIRE(ggml_graph_compute_with_ctx(g.ctx, g.gf, 1) == GGML_STATUS_SUCCESS);
}

const float* fdata(const ggml_tensor* t) {
    return static_cast<const float*>(t->data);
}

} // namespace

TEST_CASE("ggml_d3d12 moe: constants from build_moe_ffn and qwen35moe nodes") {
    // Qwen3.6-35B-A3B at decode: 2048-wide, 8 of 256 experts, one token.
    {
        MoeGraph g;
        build(g, {2048, 8, 1, 256, false, 8}, true);
        D3d12MoeCombinePlan p;
        REQUIRE(d3d12_moe_combine_plan(g.moe_out, &p));
        CHECK(p.params.ne0 == 2048);
        CHECK(p.params.n == 8);
        CHECK(p.params.e_expert_stride == 2048);
        CHECK(p.params.e_token_stride == 2048 * 8);
        CHECK(p.params.w_expert_stride == 1);
        CHECK(p.params.w_token_stride == 0);
        CHECK(p.params.dst_token_stride == 2048);
        CHECK(p.params.pad0 == 0);
        CHECK(p.groups_x == 8);
        CHECK(p.groups_y == 1);
        CHECK(p.nodes.size() == 1 + 8 + 7);
        CHECK(nodes_in_order(g.gf, p.nodes, g.moe_out));
        CHECK(p.e == g.moe_out->src[1]->view_src->src[0]);

        D3d12MoeTailPlan t;
        REQUIRE(d3d12_moe_tail_plan(g.post, &t));
        CHECK(t.params.mode == 2);
        CHECK(t.params.ne0 == 2048);
        CHECK(t.groups_x == 8);
        CHECK(t.groups_y == 1);
        CHECK(t.m == g.moe_out);
        CHECK(t.q == g.post->src[1]);
        CHECK(t.s == g.gated->src[0]);
        CHECK(t.g == g.gated->src[1]);
        CHECK(nodes_in_order(g.gf, t.nodes, g.post));
        CHECK(t.nodes.size() == 3);
    }
    // A 4-token ubatch: the weights are a [1, n, T] reshape of the normalized [n, T].
    {
        MoeGraph g;
        build(g, {2048, 8, 4, 256, false, 8}, true);
        D3d12MoeCombinePlan p;
        REQUIRE(d3d12_moe_combine_plan(g.moe_out, &p));
        CHECK(p.params.w_expert_stride == 1);
        CHECK(p.params.w_token_stride == 8);
        CHECK(p.params.e_token_stride == 2048 * 8);
        CHECK(p.params.dst_token_stride == 2048);
        CHECK(p.groups_y == 4);
        D3d12MoeTailPlan t;
        REQUIRE(d3d12_moe_tail_plan(g.post, &t));
        CHECK(t.params.s_row_stride == 2048);
        CHECK(t.params.g_row_stride == 1);
        CHECK(t.params.m_row_stride == 2048);
        CHECK(t.params.q_row_stride == 2048);
        CHECK(t.params.dst_row_stride == 2048);
        CHECK(t.groups_y == 4);
        REQUIRE(d3d12_moe_tail_plan(g.ffn_out, &t));
        CHECK(t.params.mode == 1);
        CHECK(t.q == nullptr);
        CHECK(t.nodes.size() == 2);
        REQUIRE(d3d12_moe_tail_plan(g.gated, &t));
        CHECK(t.params.mode == 0);
        CHECK(t.m == nullptr);
        CHECK(t.nodes.size() == 1);
    }
}

TEST_CASE("ggml_d3d12 moe: nodes the kernels do not take") {
    D3d12MoeCombinePlan p;
    D3d12MoeTailPlan t;
    CHECK_FALSE(d3d12_moe_combine_plan(nullptr, &p));
    CHECK_FALSE(d3d12_moe_tail_plan(nullptr, &t));
    {
        MoeGraph g;
        build(g, {2048, 8, 2, 0, false, 8}, true);
        CHECK_FALSE(d3d12_moe_combine_plan(g.before_last, &p)); // A_6 of 7: not the chain's end
        CHECK_FALSE(d3d12_moe_combine_plan(g.post, &p));        // not a chain at all
        CHECK_FALSE(d3d12_moe_tail_plan(g.moe_out, &t));        // an ADD of views, no gate
        CHECK_FALSE(d3d12_moe_tail_plan(g.post->src[1], &t));   // a leaf
    }
    {
        MoeGraph g; // the chain stops one expert short of W's 8
        build(g, {2048, 8, 2, 0, false, 7}, true);
        CHECK_FALSE(d3d12_moe_combine_plan(g.moe_out, &p));
    }
    {
        MoeGraph g; // 70000 tokens: past 65535 groups in y
        build(g, {256, 2, 70000, 0, false, 2}, true);
        CHECK_FALSE(d3d12_moe_combine_plan(g.moe_out, &p));
        CHECK_FALSE(d3d12_moe_tail_plan(g.post, &t));
    }
    {
        ggml_init_params ip{};
        ip.mem_size = 16 * ggml_tensor_overhead() + (1u << 20);
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        REQUIRE(ctx != nullptr);
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 3);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 3);
        ggml_tensor* row = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        CHECK_FALSE(d3d12_moe_tail_plan(ggml_mul(ctx, a, b), &t));   // no broadcast: elt's MUL
        CHECK_FALSE(d3d12_moe_tail_plan(ggml_mul(ctx, a, row), &t)); // a norm weight, not a gate
        CHECK_FALSE(d3d12_moe_tail_plan(ggml_add(ctx, a, b), &t));
        ggml_tensor* h = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 3);
        ggml_tensor* g1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 3);
        CHECK_FALSE(
            d3d12_moe_tail_plan(ggml_mul(ctx, h, ggml_cast(ctx, g1, GGML_TYPE_F16)), &t)); // f16
        ggml_free(ctx);
    }
}

TEST_CASE("ggml_d3d12 moe: kernel emulation equals ggml's CPU ops bit for bit") {
    const MoeCase cases[] = {
        {2048, 8, 1, 256, false, 8}, // Qwen3.6-35B-A3B decode
        {2048, 8, 4, 256, false, 8}, // a 4-token ubatch
        {300, 2, 3, 0, false, 2},    // 300 wide: the second group stores 44 of 256
        {512, 4, 5, 0, true, 4},     // one weight per token for all its experts
        {1, 3, 2, 0, false, 3},      // one column
    };
    for (const auto& c : cases) {
        CAPTURE(c.ne0);
        CAPTURE(c.n);
        CAPTURE(c.n_tok);
        CAPTURE(c.n_expert);
        MoeGraph g;
        build(g, c, false);
        compute(g);

        D3d12MoeCombinePlan p;
        REQUIRE(d3d12_moe_combine_plan(g.moe_out, &p));
        constexpr float kPoison = -7.f;
        const std::size_t span =
            static_cast<std::size_t>(p.params.dst_token_stride) * (c.n_tok - 1) + c.ne0;
        std::vector<float> moe(span + 16, kPoison);
        d3d12_moe_combine_emulate(p.params, p.groups_x, p.groups_y, fdata(p.e), fdata(p.w),
                                  moe.data());
        CHECK(same_bits(g.moe_out, moe, p.params.dst_token_stride));
        for (std::size_t i = span; i < moe.size(); ++i)
            CHECK(moe[i] == kPoison);

        for (const ggml_tensor* node : {g.post, g.ffn_out, g.gated}) {
            D3d12MoeTailPlan t;
            REQUIRE(d3d12_moe_tail_plan(node, &t));
            std::vector<float> out(
                static_cast<std::size_t>(t.params.dst_row_stride) * (c.n_tok - 1) + c.ne0 + 16,
                kPoison);
            d3d12_moe_tail_emulate(t.params, t.groups_x, t.groups_y, fdata(t.s), fdata(t.g),
                                   t.m ? fdata(t.m) : nullptr, t.q ? fdata(t.q) : nullptr,
                                   out.data());
            CAPTURE(t.params.mode);
            CHECK(same_bits(node, out, t.params.dst_row_stride));
        }
    }
}
