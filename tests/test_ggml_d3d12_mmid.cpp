// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "../src/bridge/ggml_d3d12_mmid_emu.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace xllama;

namespace {

struct MmidCase {
    int k, n, n_as, n_ids, ne11, n_tok;
    int ids_width; // ids is a view of an [ids_width, n_tok] tensor
};

struct MmidGraph {
    ggml_context* ctx = nullptr;
    ggml_tensor* out = nullptr;
    MmidGraph() = default;
    MmidGraph(const MmidGraph&) = delete;
    MmidGraph& operator=(const MmidGraph&) = delete;
    ~MmidGraph() {
        if (ctx)
            ggml_free(ctx);
    }
};

// ggml_mul_mat_id(as, b, ids) with random IQ4_XS or IQ3_S experts (any block
// bytes are a valid block of either once d is a finite half), random activations and
// distinct experts per token; the ids view's unused columns hold other valid
// experts, so reading them changes the result. no_alloc builds shapes only.
void build(MmidGraph& g, const MmidCase& c, ggml_type wtype, bool no_alloc) {
    const std::size_t w_bytes = ggml_row_size(wtype, c.k) * c.n * c.n_as;
    const std::size_t b_bytes = sizeof(float) * c.k * c.ne11 * c.n_tok;
    const std::size_t out_bytes = sizeof(float) * c.n * c.n_ids * c.n_tok;
    ggml_init_params ip{};
    ip.mem_size = (no_alloc ? 0 : w_bytes + 2 * b_bytes + 2 * out_bytes) +
                  sizeof(std::int32_t) * c.ids_width * c.n_tok + 8 * ggml_tensor_overhead() +
                  ggml_graph_overhead() + (4u << 20);
    ip.no_alloc = no_alloc;
    g.ctx = ggml_init(ip);
    REQUIRE(g.ctx != nullptr);
    ggml_tensor* as = ggml_new_tensor_3d(g.ctx, wtype, c.k, c.n, c.n_as);
    ggml_tensor* b = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, c.k, c.ne11, c.n_tok);
    ggml_tensor* ids_all = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, c.ids_width, c.n_tok);
    ggml_tensor* ids = ggml_view_2d(g.ctx, ids_all, c.n_ids, c.n_tok, ids_all->nb[1], 0);
    g.out = ggml_mul_mat_id(g.ctx, as, b, ids);
    if (no_alloc)
        return;

    std::mt19937 rng(1234u + static_cast<unsigned>(c.k + c.n * 3 + c.n_as * 5 + c.n_tok));
    auto* wb = static_cast<std::uint8_t*>(as->data);
    REQUIRE(w_bytes % 4 == 0); // IQ4_XS rows, and IQ3_S rows of an even block count
    for (std::size_t i = 0; i < w_bytes; i += 4) {
        const std::uint32_t v = rng();
        std::memcpy(wb + i, &v, sizeof(v));
    }
    std::uniform_real_distribution<float> scale(1e-3f, 1e-2f);
    const std::size_t blocks = w_bytes / ggml_type_size(wtype);
    for (std::size_t i = 0; i < blocks; ++i) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(scale(rng));
        std::memcpy(wb + i * ggml_type_size(wtype), &d, sizeof(d)); // d leads every block
    }
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    auto* bf = static_cast<float*>(b->data);
    for (std::int64_t i = 0; i < ggml_nelements(b); ++i)
        bf[i] = uni(rng);
    auto* idv = static_cast<std::int32_t*>(ids_all->data);
    std::vector<std::int32_t> perm(static_cast<std::size_t>(c.n_as));
    for (int t = 0; t < c.n_tok; ++t) {
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i < c.ids_width; ++i)
            idv[t * c.ids_width + i] = perm[static_cast<std::size_t>(i % c.n_as)];
    }
}

struct MmidRun {
    bool planned = false;
    D3d12MmidPlan plan;
    double rel_err[2] = {1e9, 1e9}; // 64 and 128 threads
    bool ran[2] = {false, false};   // a blob narrower than the plan's row sets is skipped
    bool tail_untouched = true;
};

// The emulated kernel on the plan's constants against ggml's CPU mul_mat_id,
// which quantizes src1 with the same from_float and sums with vec_dot.
MmidRun run(const MmidCase& c, ggml_type wtype, const D3d12GeomReq& req = {}) {
    MmidRun r;
    MmidGraph g;
    build(g, c, wtype, false);
    r.planned = d3d12_mmid_plan(g.out, &r.plan, req);
    if (!r.planned)
        return r;
    ggml_cpu_init();
    ggml_cgraph* gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(gf, g.out);
    REQUIRE(ggml_graph_compute_with_ctx(g.ctx, gf, 1) == GGML_STATUS_SUCCESS);
    const auto* ref = static_cast<const float*>(g.out->data);
    const std::size_t n_out = static_cast<std::size_t>(ggml_nelements(g.out));

    const ggml_tensor* as = g.out->src[0];
    const ggml_tensor* ids = g.out->src[2];
    std::vector<std::uint8_t> xq(r.plan.x_bytes);
    d3d12_mmid_quantize_activations(g.out, g.out->src[1]->data, xq.data());
    const std::uint32_t blobs[2] = {static_cast<std::uint32_t>(kD3d12MmvThreadsShort),
                                    static_cast<std::uint32_t>(kD3d12MmvThreadsLong)};
    for (int bi = 0; bi < 2; ++bi) {
        if (blobs[bi] / 16u < r.plan.geom.rsl)
            continue;
        r.ran[bi] = true;
        constexpr std::size_t kTail = 16;
        std::vector<float> y(n_out + kTail, -7.f);
        d3d12_mmid_emulate(wtype, r.plan.params, r.plan.groups_x, r.plan.groups_y, blobs[bi],
                           static_cast<const std::uint8_t*>(as->data),
                           static_cast<const std::uint8_t*>(ids->data), xq.data(), y.data());
        double max_ref = 0.0, max_diff = 0.0;
        for (std::size_t i = 0; i < n_out; ++i) {
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[i])));
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(ref[i]) - y[i]));
        }
        r.rel_err[bi] = max_ref > 0.0 ? max_diff / max_ref : max_diff;
        for (std::size_t i = n_out; i < y.size(); ++i)
            r.tail_untouched = r.tail_untouched && y[i] == -7.f;
    }
    return r;
}

} // namespace

TEST_CASE("ggml_d3d12 mmid: constants from a MUL_MAT_ID node") {
    // Qwen3.6-35B-A3B gate/up at decode: 2048 -> 512, 8 of 41 slots, ids a view
    // of the [256, T] argsort as without the expert cache.
    {
        MmidGraph g;
        build(g, {2048, 512, 41, 8, 1, 3, 256}, GGML_TYPE_IQ4_XS, true);
        D3d12MmidPlan p;
        REQUIRE(d3d12_mmid_plan(g.out, &p));
        const std::uint32_t row = 8 * 136, xq_row = 8 * 292;
        CHECK(p.params.n == 512);
        CHECK(p.params.k_dim == 2048);
        CHECK(p.params.nchunk == 8);
        CHECK(p.params.w_row_bytes == row);
        CHECK(p.params.expert_stride == 512 * row);
        CHECK(p.params.n_ids == 8);
        CHECK(p.params.ids_stride == 256 * 4);
        CHECK(p.params.x_stride == xq_row);
        CHECK(p.params.x_col_stride == 0);
        CHECK(p.params.y_expert_stride == 512);
        CHECK(p.params.y_stride == 512 * 8);
        CHECK(p.params.pad0 == 0);
        CHECK(p.params.pad1 == 0);
        CHECK(p.groups_x == 128);
        CHECK(p.groups_y == 24);
        CHECK(p.threads == static_cast<std::uint32_t>(d3d12_mm_threads(2048)));
        CHECK(p.x_bytes == 3u * xq_row);
        CHECK(p.ids_bytes == 2u * 256 * 4 + 8 * 4);
    }
    // down: 512 -> 2048, one activation column per expert.
    {
        MmidGraph g;
        build(g, {512, 2048, 41, 8, 8, 2, 8}, GGML_TYPE_IQ4_XS, true);
        D3d12MmidPlan p;
        REQUIRE(d3d12_mmid_plan(g.out, &p));
        CHECK(p.params.nchunk == 2);
        CHECK(p.params.x_col_stride == 2 * 292);
        CHECK(p.params.x_stride == 8 * 2 * 292);
        CHECK(p.params.ids_stride == 8 * 4);
        CHECK(p.groups_x == 512);
        CHECK(p.groups_y == 16);
        CHECK(p.x_bytes == 2u * 8 * 2 * 292);
    }
    // Qwen3.6-35B-A3B i1-IQ3_S: 110-byte blocks, gate/up rows 880 B and down
    // rows 220 B, both 450560 B per expert.
    {
        MmidGraph g;
        build(g, {2048, 512, 41, 8, 1, 3, 256}, GGML_TYPE_IQ3_S, true);
        D3d12MmidPlan p;
        REQUIRE(d3d12_mmid_plan(g.out, &p));
        CHECK(p.params.w_row_bytes == 8 * 110);
        CHECK(p.params.expert_stride == 450560);
        CHECK(p.params.x_stride == 8 * 292);
        CHECK(p.groups_x == 128);
        CHECK(p.groups_y == 24);
    }
    {
        MmidGraph g;
        build(g, {512, 2048, 41, 8, 8, 2, 8}, GGML_TYPE_IQ3_S, true);
        D3d12MmidPlan p;
        REQUIRE(d3d12_mmid_plan(g.out, &p));
        CHECK(p.params.w_row_bytes == 2 * 110);
        CHECK(p.params.expert_stride == 450560);
        CHECK(p.params.x_col_stride == 2 * 292);
    }
}

TEST_CASE("ggml_d3d12 mmid: nodes the kernel does not take") {
    D3d12MmidPlan p;
    CHECK_FALSE(d3d12_mmid_plan(nullptr, &p));
    {
        MmidGraph g; // other expert type
        build(g, {2048, 512, 41, 8, 1, 1, 8}, GGML_TYPE_Q4_K, true);
        CHECK_FALSE(d3d12_mmid_plan(g.out, &p));
    }
    {
        MmidGraph g; // IQ3_S rows of one block (110 B) are not 4-byte multiples
        build(g, {256, 512, 41, 8, 1, 1, 8}, GGML_TYPE_IQ3_S, true);
        CHECK_FALSE(d3d12_mmid_plan(g.out, &p));
    }
    {
        MmidGraph g; // ggml broadcasts 2 columns over 8 ids; the kernel takes 1 or n_ids
        build(g, {2048, 512, 41, 8, 2, 1, 8}, GGML_TYPE_IQ4_XS, true);
        CHECK_FALSE(d3d12_mmid_plan(g.out, &p));
    }
    {
        MmidGraph g; // 8 ids x 8192 tokens = 65536 groups in y
        build(g, {256, 4, 9, 8, 1, 8192, 8}, GGML_TYPE_IQ4_XS, true);
        CHECK_FALSE(d3d12_mmid_plan(g.out, &p));
    }
    {
        MmidGraph g; // 8191 tokens fit
        build(g, {256, 4, 9, 8, 1, 8191, 8}, GGML_TYPE_IQ4_XS, true);
        CHECK(d3d12_mmid_plan(g.out, &p));
    }
    {
        MmidGraph g; // not a MUL_MAT_ID
        build(g, {256, 4, 9, 8, 1, 1, 8}, GGML_TYPE_IQ4_XS, true);
        CHECK_FALSE(d3d12_mmid_plan(g.out->src[1], &p));
    }
}

TEST_CASE("ggml_d3d12 mmid: kernel emulation matches ggml's CPU mul_mat_id") {
    const MmidCase cases[] = {
        // Qwen3.6-35B-A3B experts, 41 slots: gate/up and down at decode, gate/up
        // for a 4-token ubatch.
        {2048, 512, 41, 8, 1, 1, 256},
        {512, 2048, 41, 8, 8, 1, 8},
        {2048, 512, 41, 8, 1, 4, 8},
        // 37 rows: the last group clamps its loads and skips 3 stores; ids
        // view 16 wide, 5 tokens, one column per id.
        {1024, 37, 11, 8, 8, 5, 16},
        // K = 4096 (16 chunks), 4 ids per token, shared columns.
        {4096, 13, 9, 4, 1, 3, 4},
    };
    for (const ggml_type wtype : {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S})
        for (const auto& c : cases) {
            CAPTURE(ggml_type_name(wtype));
            CAPTURE(c.k);
            CAPTURE(c.n);
            CAPTURE(c.ne11);
            CAPTURE(c.n_tok);
            const MmidRun r = run(c, wtype);
            REQUIRE(r.planned);
            CHECK(r.rel_err[0] <= kD3d12SelftestRelTol);
            CHECK(r.rel_err[1] <= kD3d12SelftestRelTol);
            CHECK(r.tail_untouched);
        }
}

TEST_CASE("ggml_d3d12 mmid: IQ3_S group geometry") {
    // The plan: auto fills only the lanes a short K leaves idle; requests clamp
    // to a power of two within the blob's lanes; rt to [1, kD3d12GeomMaxRt].
    struct PlanCase {
        int k, n;
        D3d12GeomReq req;
        std::uint32_t rsl, rt, groups_x;
    };
    const PlanCase plans[] = {
        {512, 2048, {0, 1}, 2, 1, 256},  // down: 2 chunks on 4 lanes
        {2048, 512, {0, 1}, 1, 1, 128},  // gate/up: 8 chunks fill the lanes
        {512, 2048, {16, 1}, 4, 1, 128}, // past the 64-thread blob's 4 lanes
        {4096, 512, {16, 1}, 8, 1, 16},  // 128 threads: 8 lanes
        {2048, 512, {3, 2}, 2, 2, 32},   // 3 -> 2
        {2048, 512, {1, 100}, 1, 64, 2}, // rt clamped
        {512, 2048, {0, 4}, 2, 4, 64},
    };
    for (const auto& pc : plans) {
        CAPTURE(pc.k);
        CAPTURE(pc.req.rsl);
        CAPTURE(pc.req.rt);
        MmidGraph g;
        build(g, {pc.k, pc.n, 41, 8, pc.k == 512 ? 8 : 1, 1, 8}, GGML_TYPE_IQ3_S, true);
        D3d12MmidPlan p;
        REQUIRE(d3d12_mmid_plan(g.out, &p, pc.req));
        CHECK(p.geom.rsl == pc.rsl);
        CHECK(p.geom.rt == pc.rt);
        CHECK(p.groups_x == pc.groups_x);
        CHECK(p.params.pad0 == (pc.rsl | (pc.rt << 8)));
    }
    {
        MmidGraph g; // IQ4_XS has no geometry: pad0 stays 0
        build(g, {512, 2048, 41, 8, 8, 1, 8}, GGML_TYPE_IQ4_XS, true);
        D3d12MmidPlan p;
        REQUIRE(d3d12_mmid_plan(g.out, &p, {4, 3}));
        CHECK(p.params.pad0 == 0);
        CHECK(p.groups_x == 512);
    }
    // The emulated kernel under each geometry against the CPU: wide groups, tile
    // loops and tails no geometry divides (37, 13 rows).
    struct EmuCase {
        MmidCase c;
        D3d12GeomReq req;
    };
    const EmuCase emus[] = {
        {{512, 2048, 41, 8, 8, 1, 8}, {0, 1}}, {{512, 2048, 41, 8, 8, 2, 8}, {0, 4}},
        {{2048, 512, 41, 8, 1, 4, 8}, {4, 2}}, {{1024, 37, 11, 8, 8, 5, 16}, {2, 3}},
        {{4096, 13, 9, 4, 1, 3, 4}, {8, 1}},   {{4096, 13, 9, 4, 1, 3, 4}, {2, 5}},
    };
    for (const auto& e : emus) {
        CAPTURE(e.c.k);
        CAPTURE(e.c.n);
        CAPTURE(e.req.rsl);
        CAPTURE(e.req.rt);
        const MmidRun r = run(e.c, GGML_TYPE_IQ3_S, e.req);
        REQUIRE(r.planned);
        const int own = r.plan.threads == static_cast<std::uint32_t>(kD3d12MmvThreadsLong) ? 1 : 0;
        REQUIRE(r.ran[own]);
        for (int bi = 0; bi < 2; ++bi)
            if (r.ran[bi])
                CHECK(r.rel_err[bi] <= kD3d12SelftestRelTol);
        CHECK(r.tail_untouched);
    }
}

TEST_CASE("ggml_d3d12: IQ3_S geometry knobs") {
    CHECK(d3d12_geom_parse(nullptr, 1) == 1);
    CHECK(d3d12_geom_parse("", 1) == 1);
    CHECK(d3d12_geom_parse("auto", 1) == 0);
    CHECK(d3d12_geom_parse("0", 1) == 0);
    CHECK(d3d12_geom_parse("4", 1) == 4);
    CHECK(d3d12_geom_parse("255", 1) == 255);
    CHECK(d3d12_geom_parse("256", 1) == 1);
    CHECK(d3d12_geom_parse("4x", 1) == 1);
    CHECK(d3d12_geom_parse("-2", 1) == 1);
    const D3d12Geom one = d3d12_geom_for({}, 8, 64);
    CHECK(one.rsl == 1);
    CHECK(one.rt == 1);
    CHECK(d3d12_geom_pack(one) == 0x101u);
    CHECK(d3d12_geom_rows(one) == static_cast<std::uint32_t>(kD3d12MmvRows));
    CHECK(d3d12_geom_for({0, 1}, 2, 64).rsl == 2);
    CHECK(d3d12_geom_for({0, 1}, 1, 128).rsl == 8);
    CHECK(d3d12_geom_for({0, 1}, 3, 64).rsl == 1); // 4 / 3 -> 1
    CHECK(d3d12_geom_for({0, 1}, 16, 128).rsl == 1);
    CHECK(d3d12_geom_for({6, 0}, 8, 128).rsl == 4);
    CHECK(d3d12_geom_for({6, 0}, 8, 128).rt == 1);
    CHECK(d3d12_geom_rows(d3d12_geom_for({2, 3}, 8, 64)) ==
          static_cast<std::uint32_t>(kD3d12MmvRows) * 6u);
}
