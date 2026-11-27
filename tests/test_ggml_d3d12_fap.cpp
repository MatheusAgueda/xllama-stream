// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "../src/bridge/ggml_d3d12_fap_emu.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace xllama;

namespace {

struct Ctx {
    ggml_context* ctx = nullptr;
    Ctx(std::size_t data_bytes, bool no_alloc) {
        ggml_init_params ip{};
        ip.mem_size = (no_alloc ? 0 : data_bytes) + 64 * ggml_tensor_overhead() +
                      ggml_graph_overhead() + (1u << 20);
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

// One attention layer as llama-graph.cpp builds it with flash attention: the
// f16 K / V caches [D * Hkv, kv_size], their views (llama-kv-cache.cpp get_k /
// get_v) permuted to [D, n_kv, Hkv], the roped Q [D, H, N] permuted to
// [D, N, H], the f16 mask [n_kv, N].
struct Shape {
    int d = 256, heads = 16, kv_heads = 2, n = 64, n_kv = 256, kv_size = 256;
    int p0 = 0;           // position of the ubatch's first token (cells hold positions 0..)
    bool mask = true;     // causal mask; false: no mask
    int blind_token = -1; // a token that sees nothing
    int lone_token = -1;  // a token that sees cell 0 only
};

struct Layer {
    ggml_tensor* kc = nullptr;
    ggml_tensor* vc = nullptr;
    ggml_tensor* q_cur = nullptr;
    ggml_tensor* mask = nullptr;
    ggml_tensor* out = nullptr;
};

Layer build(ggml_context* ctx, const Shape& s, float scale = 0.f, float max_bias = 0.f,
            float softcap = 0.f) {
    Layer l;
    const std::int64_t gqa = static_cast<std::int64_t>(s.d) * s.kv_heads;
    l.kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, gqa, s.kv_size);
    l.vc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, gqa, s.kv_size);
    auto view = [&](ggml_tensor* c) {
        ggml_tensor* t =
            ggml_view_4d(ctx, c, s.d, s.kv_heads, s.n_kv, 1, ggml_row_size(c->type, s.d),
                         ggml_row_size(c->type, gqa), ggml_row_size(c->type, gqa * s.kv_size), 0);
        return ggml_permute(ctx, t, 0, 2, 1, 3);
    };
    l.q_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.d, s.heads, s.n);
    ggml_tensor* q = ggml_permute(ctx, l.q_cur, 0, 2, 1, 3);
    if (s.mask)
        l.mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s.n_kv, s.n, 1, 1);
    if (scale == 0.f)
        scale = 1.f / std::sqrt(static_cast<float>(s.d));
    l.out = ggml_flash_attn_ext(ctx, q, view(l.kc), view(l.vc), l.mask, scale, max_bias, softcap);
    return l;
}

void fill(ggml_tensor* t, std::mt19937* rng) {
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const std::int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        auto* f = static_cast<float*>(t->data);
        for (std::int64_t i = 0; i < n; ++i)
            f[i] = d(*rng);
    } else {
        auto* h = static_cast<ggml_fp16_t*>(t->data);
        for (std::int64_t i = 0; i < n; ++i)
            h[i] = ggml_fp32_to_fp16(d(*rng));
    }
}

void fill_mask(ggml_tensor* m, const Shape& s) {
    auto* h = static_cast<ggml_fp16_t*>(m->data);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    for (int i = 0; i < s.n; ++i)
        for (int j = 0; j < s.n_kv; ++j) {
            bool seen = j <= s.p0 + i;
            if (i == s.blind_token)
                seen = false;
            if (i == s.lone_token)
                seen = j == 0;
            h[static_cast<std::size_t>(i) * s.n_kv + j] = seen ? zero : ninf;
        }
}

bool seen(const Shape& s, int i, int j) {
    if (!s.mask)
        return true;
    if (i == s.blind_token)
        return false;
    if (i == s.lone_token)
        return j == 0;
    return j <= s.p0 + i;
}

// softmax(q k^T * scale + mask) v in double, from the cache rows.
std::vector<double> reference(const Layer& l, const Shape& s, float scale) {
    const int g = s.heads / s.kv_heads;
    const auto* q = static_cast<const float*>(l.q_cur->data);
    const auto* kc = static_cast<const ggml_fp16_t*>(l.kc->data);
    const auto* vc = static_cast<const ggml_fp16_t*>(l.vc->data);
    const std::size_t gqa = static_cast<std::size_t>(s.d) * s.kv_heads;
    std::vector<double> out(static_cast<std::size_t>(s.n) * s.heads * s.d, 0.0);
    std::vector<double> w(s.n_kv);
    for (int i = 0; i < s.n; ++i)
        for (int h = 0; h < s.heads; ++h) {
            const int hk = h / g;
            const float* qr = q + (static_cast<std::size_t>(i) * s.heads + h) * s.d;
            double mx = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < s.n_kv; ++j) {
                w[j] = -std::numeric_limits<double>::infinity();
                if (!seen(s, i, j))
                    continue;
                const ggml_fp16_t* kr = kc + j * gqa + static_cast<std::size_t>(hk) * s.d;
                double dot = 0.0;
                for (int e = 0; e < s.d; ++e)
                    dot += static_cast<double>(qr[e]) * ggml_fp16_to_fp32(kr[e]);
                w[j] = dot * scale;
                mx = std::max(mx, w[j]);
            }
            double sum = 0.0;
            for (int j = 0; j < s.n_kv; ++j) {
                w[j] = std::isinf(w[j]) ? 0.0 : std::exp(w[j] - mx);
                sum += w[j];
            }
            double* o = out.data() + (static_cast<std::size_t>(i) * s.heads + h) * s.d;
            if (sum == 0.0)
                continue;
            for (int j = 0; j < s.n_kv; ++j) {
                if (w[j] == 0.0)
                    continue;
                const ggml_fp16_t* vr = vc + j * gqa + static_cast<std::size_t>(hk) * s.d;
                for (int e = 0; e < s.d; ++e)
                    o[e] += w[j] / sum * ggml_fp16_to_fp32(vr[e]);
            }
        }
    return out;
}

struct Errors {
    double vs_ref = 0.0; // emulation vs the double reference
    double vs_cpu = 0.0; // emulation vs ggml's CPU op
    double ref_max = 0.0;
};

// chunk > 0: the split and reduce dispatches (d3d12_fap_emulate_chunked)
// instead of the one-pass kernel.
Errors run(const Shape& s, unsigned seed, std::uint32_t chunk = 0) {
    const std::size_t bytes = 4 * static_cast<std::size_t>(s.d) * s.kv_heads * s.kv_size +
                              4 * static_cast<std::size_t>(s.d) * s.heads * s.n * 2 +
                              2 * static_cast<std::size_t>(s.n_kv) * s.n + (16u << 20);
    Ctx c(bytes, false);
    const float scale = 1.f / std::sqrt(static_cast<float>(s.d));
    Layer l = build(c.ctx, s, scale);
    std::mt19937 rng(seed);
    fill(l.kc, &rng);
    fill(l.vc, &rng);
    fill(l.q_cur, &rng);
    if (l.mask)
        fill_mask(l.mask, s);

    D3d12FapPlan p;
    REQUIRE(d3d12_fap_plan(l.out, &p));
    std::vector<float> emu(static_cast<std::size_t>(ggml_nelements(l.out)), -7.f);
    const auto* qd = static_cast<const float*>(p.q->data);
    const auto* kd = static_cast<const ggml_fp16_t*>(p.k->data);
    const auto* vd = static_cast<const ggml_fp16_t*>(p.v->data);
    const auto* md = p.mask ? static_cast<const ggml_fp16_t*>(p.mask->data) : nullptr;
    if (chunk)
        d3d12_fap_emulate_chunked(p.params, chunk, p.groups_x, p.groups_y, qd, kd, vd, md,
                                  emu.data());
    else
        d3d12_fap_emulate(p.params, p.groups_x, p.groups_y, qd, kd, vd, md, emu.data());

    ggml_cgraph* gf = ggml_new_graph(c.ctx);
    ggml_build_forward_expand(gf, l.out);
    ggml_cpu_init();
    REQUIRE(ggml_graph_compute_with_ctx(c.ctx, gf, 2) == GGML_STATUS_SUCCESS);
    const auto* cpu = static_cast<const float*>(l.out->data);

    const std::vector<double> ref = reference(l, s, scale);
    Errors e;
    for (std::size_t i = 0; i < emu.size(); ++i) {
        REQUIRE(std::isfinite(emu[i]));
        e.vs_ref = std::max(e.vs_ref, std::fabs(emu[i] - ref[i]));
        e.vs_cpu = std::max(e.vs_cpu, static_cast<double>(std::fabs(emu[i] - cpu[i])));
        e.ref_max = std::max(e.ref_max, std::fabs(ref[i]));
    }
    // dst is [D, H, N]: token i, head h at (i * H + h) * D
    auto row = [&](int i, int h) {
        return emu.data() + (static_cast<std::size_t>(i) * s.heads + h) * s.d;
    };
    if (s.blind_token >= 0)
        for (int h = 0; h < s.heads; ++h)
            for (int x = 0; x < s.d; ++x)
                CHECK(row(s.blind_token, h)[x] == 0.f);
    if (s.lone_token >= 0) {
        const auto* vc = static_cast<const ggml_fp16_t*>(l.vc->data);
        const int g = s.heads / s.kv_heads;
        for (int h = 0; h < s.heads; ++h)
            for (int x = 0; x < s.d; ++x)
                CHECK(row(s.lone_token, h)[x] == ggml_fp16_to_fp32(vc[(h / g) * s.d + x]));
    }
    MESSAGE("d=" << s.d << " heads=" << s.heads << "/" << s.kv_heads << " n=" << s.n
                 << " n_kv=" << s.n_kv << " chunk=" << chunk << ": vs double " << e.vs_ref
                 << ", vs ggml cpu " << e.vs_cpu << ", |ref| max " << e.ref_max);
    return e;
}

} // namespace

TEST_CASE("ggml_d3d12 fap: constants from a qwen3.6 full-attention layer") {
    Ctx c(0, true);
    Shape s;
    s.d = 256;
    s.heads = 16;
    s.kv_heads = 2;
    s.n = 1024;
    s.n_kv = 13312;
    s.kv_size = 16384;
    Layer l = build(c.ctx, s, 0.0625f);
    D3d12FapPlan p;
    REQUIRE(d3d12_fap_plan(l.out, &p));
    CHECK(p.params.d_g == (256u | 8u << 16));
    CHECK(p.params.n_kv == 13312u);
    CHECK(p.params.n_head == (1024u | 16u << 16));
    CHECK(p.params.q_token_stride == 16u * 256u);
    CHECK(p.params.q_head_stride == 256u);
    CHECK(p.params.k_cell_stride == 256u); // 512 halves a cell
    CHECK(p.params.k_head_stride == 128u);
    CHECK(p.params.v_cell_stride == 256u);
    CHECK(p.params.v_head_stride == 128u);
    CHECK(p.params.mask_row_stride == 13312u);
    CHECK(p.params.scale_bits == fap_detail::bits(0.0625f));
    CHECK(p.params.flags == kD3d12FapMask);
    CHECK(p.groups_x == 512u); // 2 tokens of 8 heads a group
    CHECK(p.groups_y == 2u);
    CHECK(p.mask == l.mask);
}

TEST_CASE("ggml_d3d12 fap: nodes the kernel does not take") {
    Ctx c(0, true);
    Shape s;
    s.n = 64;
    s.n_kv = 128;
    s.kv_size = 128;
    D3d12FapPlan p;
    CHECK_FALSE(d3d12_fap_plan(build(c.ctx, s, 0.f, 8.f).out, &p));       // ALiBi
    CHECK_FALSE(d3d12_fap_plan(build(c.ctx, s, 0.f, 0.f, 30.f).out, &p)); // softcap
    {
        Layer l = build(c.ctx, s);
        ggml_tensor* sinks = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, s.heads);
        ggml_flash_attn_ext_add_sinks(l.out, sinks);
        CHECK_FALSE(d3d12_fap_plan(l.out, &p));
    }
    {
        Shape t = s;
        t.heads = 6; // G = 3
        CHECK_FALSE(d3d12_fap_plan(build(c.ctx, t).out, &p));
        t.heads = 64; // G = 32
        CHECK_FALSE(d3d12_fap_plan(build(c.ctx, t).out, &p));
        t = s;
        t.d = 512;
        CHECK_FALSE(d3d12_fap_plan(build(c.ctx, t).out, &p));
        t.d = 130; // D / 2 odd
        CHECK_FALSE(d3d12_fap_plan(build(c.ctx, t).out, &p));
    }
    {
        // f32 K / V (a cache type the kernel does not read)
        ggml_tensor* q = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 128, 64, 4);
        ggml_tensor* k = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 128, 128, 4);
        ggml_tensor* v = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 128, 128, 4);
        ggml_tensor* m = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, 128, 64);
        CHECK_FALSE(d3d12_fap_plan(ggml_flash_attn_ext(c.ctx, q, k, v, m, 0.1f, 0.f, 0.f), &p));
        // a mask per head
        ggml_tensor* kh = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, 128, 128, 4);
        ggml_tensor* vh = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, 128, 128, 4);
        ggml_tensor* mh = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, 128, 64, 4);
        CHECK_FALSE(d3d12_fap_plan(ggml_flash_attn_ext(c.ctx, q, kh, vh, mh, 0.1f, 0.f, 0.f), &p));
        CHECK(d3d12_fap_plan(ggml_flash_attn_ext(c.ctx, q, kh, vh, m, 0.1f, 0.f, 0.f), &p));
    }
    CHECK_FALSE(d3d12_fap_plan(ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 4), &p));
}

TEST_CASE("ggml_d3d12 fap: emulation matches a double reference and ggml's CPU op") {
    // tolerances: f32 accumulation over <= 300 positions vs double; the CPU
    // op rounds Q to f16 for the dot (and its untiled path accumulates V in
    // f16), so it is a looser bound
    constexpr double kRef = 2e-5, kCpu = 4e-3;
    SUBCASE("qwen3.6 heads, a ubatch past 150 cached tokens, n and n_kv off the tiles") {
        Shape s;
        s.n = 67;
        s.p0 = 150;
        s.n_kv = 230;
        s.kv_size = 256;
        const Errors e = run(s, 1);
        CHECK(e.vs_ref < kRef);
        CHECK(e.vs_cpu < kCpu);
    }
    SUBCASE("one head per kv head, D 128, first ubatch") {
        Shape s;
        s.d = 128;
        s.heads = 4;
        s.kv_heads = 4;
        s.n = 33;
        s.n_kv = 40;
        s.kv_size = 64;
        const Errors e = run(s, 2);
        CHECK(e.vs_ref < kRef);
        CHECK(e.vs_cpu < kCpu);
    }
    SUBCASE("16 heads on one kv head, D 64, no mask") {
        Shape s;
        s.d = 64;
        s.heads = 16;
        s.kv_heads = 1;
        s.n = 5;
        s.n_kv = 37;
        s.kv_size = 48;
        s.mask = false;
        const Errors e = run(s, 3);
        CHECK(e.vs_ref < kRef);
        CHECK(e.vs_cpu < kCpu);
    }
    SUBCASE("a token that sees nothing and one that sees a single cell") {
        Shape s;
        s.n = 20;
        s.p0 = 40;
        s.n_kv = 64;
        s.kv_size = 64;
        s.blind_token = 3;
        s.lone_token = 4;
        const Errors e = run(s, 4);
        CHECK(e.vs_ref < kRef);
    }
}

// the bridge's split-K scratch (ensure_fad); namespace scope: GCC rejects a
// local in a lambda's default argument
constexpr std::size_t kFadScratch = std::size_t(4) << 20;

TEST_CASE("ggml_d3d12 fap: split-K chunks for decode, none for prefill") {
    Ctx c(0, true);
    auto chunk_of = [&](int n, int n_kv, std::size_t scratch = kFadScratch) {
        Shape s;
        s.n = n;
        s.n_kv = n_kv;
        s.kv_size = n_kv;
        D3d12FapPlan p;
        REQUIRE(d3d12_fap_plan(build(c.ctx, s).out, &p));
        return d3d12_fap_chunk(p, scratch);
    };
    // one decode token: 2 groups (one per kv head) -> 64 chunks of 256 at 16K
    CHECK(chunk_of(1, 16384) == 256u);
    CHECK(d3d12_fap_chunks(16384, 256) == 64u);
    // the chunk floor of 64 positions bounds the count at short contexts
    CHECK(chunk_of(1, 1024) == 64u);
    CHECK(chunk_of(1, 300) == 80u); // 4 chunks, the last one 60 positions
    CHECK(d3d12_fap_chunks(300, 80) == 4u);
    CHECK(chunk_of(1, 100) == 0u); // one chunk: the one-pass kernel
    // eight draft tokens: 8 groups -> 16 chunks
    CHECK(chunk_of(8, 8192) == 512u);
    // the scratch bounds the chunks: 8 tokens x 16 heads x 260 floats = 130 KiB a chunk
    CHECK(chunk_of(8, 8192, std::size_t(1) << 20) == 1184u); // 7 chunks fit
    CHECK(d3d12_fap_chunks(8192, 1184) == 7u);
    CHECK(chunk_of(8, 8192, 100000) == 0u);
    // prefill ubatches have groups enough
    CHECK(chunk_of(64, 4096) == 0u);
    CHECK(chunk_of(1024, 13312) == 0u);
}

TEST_CASE("ggml_d3d12 fap: split-K emulation matches a double reference and ggml's CPU op") {
    // the chunks' partials are combined in f32 after an exp rescale: the same
    // bounds as the one-pass kernel over a longer range
    constexpr double kRef = 5e-5, kCpu = 4e-3;
    SUBCASE("a decode token at position 700 of a 1024-cell window (masked tail chunks)") {
        Shape s;
        s.n = 1;
        s.p0 = 700;
        s.n_kv = 1024;
        s.kv_size = 1024;
        const Errors e = run(s, 11, 64);
        CHECK(e.vs_ref < kRef);
        CHECK(e.vs_cpu < kCpu);
    }
    SUBCASE("a decode token over 4096 cells, uneven last chunk") {
        Shape s;
        s.n = 1;
        s.p0 = 4000;
        s.n_kv = 4000 + 1;
        s.kv_size = 4096;
        const Errors e = run(s, 12, 256);
        CHECK(e.vs_ref < kRef);
        CHECK(e.vs_cpu < kCpu);
    }
    SUBCASE("three draft tokens, a blind one and one that sees cell 0 only") {
        Shape s;
        s.n = 3;
        s.p0 = 500;
        s.n_kv = 512;
        s.kv_size = 512;
        s.blind_token = 1;
        s.lone_token = 2;
        const Errors e = run(s, 13, 64);
        CHECK(e.vs_ref < kRef);
    }
    SUBCASE("16 heads on one kv head, D 64, no mask") {
        Shape s;
        s.d = 64;
        s.heads = 16;
        s.kv_heads = 1;
        s.n = 1;
        s.n_kv = 333;
        s.kv_size = 336;
        s.mask = false;
        const Errors e = run(s, 14, 80);
        CHECK(e.vs_ref < kRef);
        CHECK(e.vs_cpu < kCpu);
    }
}
