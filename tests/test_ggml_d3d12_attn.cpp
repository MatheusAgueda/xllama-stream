// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "../src/bridge/ggml_d3d12_attn_emu.h"

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

constexpr float kPoison = -7.f;
constexpr std::uint16_t kHalfNegInf = 0xfc00;

struct Ctx {
    ggml_context* ctx = nullptr;
    explicit Ctx(std::size_t data_bytes) {
        ggml_init_params ip{};
        ip.mem_size =
            data_bytes + 512 * ggml_tensor_overhead() + 8 * ggml_graph_overhead() + (8u << 20);
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
    std::uniform_real_distribution<float> d(lo, hi);
    auto* f = static_cast<float*>(t->data);
    for (std::int64_t i = 0; i < ggml_nelements(t); ++i)
        f[i] = d(*rng);
}

// Uniform values rounded to f16 as ggml rounds them.
void fill_f16(ggml_tensor* t, std::mt19937* rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    auto* h = static_cast<ggml_fp16_t*>(t->data);
    for (std::int64_t i = 0; i < ggml_nelements(t); ++i)
        h[i] = ggml_fp32_to_fp16(d(*rng));
}

const float* fdata(const ggml_tensor* t) {
    return static_cast<const float*>(t->data);
}

const std::uint16_t* hdata(const ggml_tensor* t) {
    return static_cast<const std::uint16_t*>(t->data);
}

void compute(ggml_context* ctx, ggml_tensor* node) {
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, node);
    ggml_cpu_init();
    REQUIRE(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS);
}

// t's span (ggml_nbytes from t->data) as floats, before the CPU writes it.
std::vector<float> span_copy(const ggml_tensor* t) {
    std::vector<float> v(ggml_nbytes(t) / sizeof(float) + 1, kPoison);
    std::memcpy(v.data(), t->data, ggml_nbytes(t));
    return v;
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
// node: bit for bit when tol is 0, else d3d12_attn_max_err <= tol; the floats
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
        const double e = d3d12_attn_max_err(g.data(), r.data(), g.size());
        CAPTURE(e);
        CHECK(e <= tol);
    }
    std::size_t touched = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (!mine[i] && std::memcmp(&got[i], &kPoison, sizeof(float)) != 0)
            ++touched;
    CHECK(touched == 0);
}

bool same_bits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

// ---- rope -------------------------------------------------------------------

// The four position rows (t, h, w, e) of an mrope over n tokens.
ggml_tensor* mrope_pos(ggml_context* ctx, int n) {
    ggml_tensor* p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
    auto* v = static_cast<std::int32_t*>(p->data);
    for (int i = 0; i < n; ++i) {
        v[i] = 1000 + i;
        v[n + i] = 2000 + 3 * i;
        v[2 * n + i] = 3000 + 7 * i;
        v[3 * n + i] = 17 + i;
    }
    return p;
}

ggml_tensor* text_pos(ggml_context* ctx, int n, int first) {
    ggml_tensor* p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    auto* v = static_cast<std::int32_t*>(p->data);
    for (int i = 0; i < n; ++i)
        v[i] = first + i;
    return p;
}

// The node's plan and emulation against the CPU (inputs already filled).
D3d12AttnRopePlan check_rope(ggml_context* ctx, ggml_tensor* node, double tol) {
    D3d12AttnRopePlan plan;
    REQUIRE(d3d12_attn_rope_plan(node, &plan));
    REQUIRE(plan.src == node->src[0]);
    REQUIRE(plan.groups_x > 0);
    const std::vector<float> src = span_copy(plan.src);
    std::vector<float> got = poisoned(node);
    d3d12_attn_rope_emulate(plan.params, plan.groups_x, src.data(),
                            static_cast<const std::int32_t*>(plan.pos->data),
                            plan.ff ? fdata(plan.ff) : nullptr, got.data());
    compute(ctx, node);
    check_out(node, got, tol);
    return plan;
}

// ---- flash attention ----------------------------------------------------------

struct FaCase {
    int head = 256;
    int n_head = 16;
    int n_head_kv = 2;
    int n_tok = 1;
    int n_kv = 256;
};

struct FaGraph {
    ggml_tensor* kc = nullptr; // [head * n_head_kv, n_kv] f16 cells, as the kv cache
    ggml_tensor* vc = nullptr;
    ggml_tensor* qf = nullptr; // [head, n_head, n_tok]
    ggml_tensor* mask = nullptr;
    ggml_tensor* node = nullptr;
    float scale = 0.f;
};

// The layer's flash attention as llama-graph builds it: q permuted to
// [head, n_tok, n_head], k and v [head, n_kv, n_head_kv] views of the cells,
// an f16 mask [n_kv, n_tok] in which token i sees the cells up to
// n_kv - n_tok + i.
FaGraph fa_graph(ggml_context* ctx, const FaCase& fc, std::mt19937* rng, float max_bias = 0.f,
                 float softcap = 0.f) {
    FaGraph g;
    const std::int64_t cell = static_cast<std::int64_t>(fc.head) * fc.n_head_kv;
    g.kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cell, fc.n_kv);
    g.vc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cell, fc.n_kv);
    g.qf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, fc.head, fc.n_head, fc.n_tok);
    g.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, fc.n_kv, fc.n_tok);
    fill_f16(g.kc, rng, -1.f, 1.f);
    fill_f16(g.vc, rng, -1.f, 1.f);
    fill(g.qf, rng, -3.f, 3.f);
    auto* m = static_cast<std::uint16_t*>(g.mask->data);
    for (int t = 0; t < fc.n_tok; ++t)
        for (int p = 0; p < fc.n_kv; ++p)
            m[t * fc.n_kv + p] = p <= fc.n_kv - fc.n_tok + t ? 0 : kHalfNegInf;
    const std::size_t hs = ggml_row_size(GGML_TYPE_F16, fc.head);
    ggml_tensor* k = ggml_permute(
        ctx, ggml_view_3d(ctx, g.kc, fc.head, fc.n_head_kv, fc.n_kv, hs, g.kc->nb[1], 0), 0, 2, 1,
        3);
    ggml_tensor* v = ggml_permute(
        ctx, ggml_view_3d(ctx, g.vc, fc.head, fc.n_head_kv, fc.n_kv, hs, g.vc->nb[1], 0), 0, 2, 1,
        3);
    ggml_tensor* q = ggml_permute(ctx, g.qf, 0, 2, 1, 3);
    g.scale = 1.f / std::sqrt(static_cast<float>(fc.head));
    g.node = ggml_flash_attn_ext(ctx, q, k, v, g.mask, g.scale, max_bias, softcap);
    return g;
}

// Softmax attention in double from the f16 k, v and mask and the f32 q;
// laid out like the node, [head, n_head, n_tok].
std::vector<float> fa_ref(const FaCase& fc, const FaGraph& g) {
    const int grp = fc.n_head / fc.n_head_kv;
    const std::int64_t cell = static_cast<std::int64_t>(fc.head) * fc.n_head_kv;
    std::vector<float> out(static_cast<std::size_t>(fc.n_tok) * fc.n_head * fc.head, 0.f);
    std::vector<double> s(static_cast<std::size_t>(fc.n_kv)), acc(fc.head);
    for (int t = 0; t < fc.n_tok; ++t)
        for (int h = 0; h < fc.n_head; ++h) {
            const int hk = h / grp;
            const float* q = fdata(g.qf) + (static_cast<std::size_t>(t) * fc.n_head + h) * fc.head;
            double mx = -std::numeric_limits<double>::infinity();
            for (int p = 0; p < fc.n_kv; ++p) {
                const float m = ggml_fp16_to_fp32(hdata(g.mask)[t * fc.n_kv + p]);
                if (std::isinf(m) && m < 0) {
                    s[p] = -std::numeric_limits<double>::infinity();
                    continue;
                }
                double dot = 0.0;
                for (int d = 0; d < fc.head; ++d)
                    dot += static_cast<double>(q[d]) *
                           ggml_fp16_to_fp32(hdata(g.kc)[p * cell + hk * fc.head + d]);
                s[p] = dot * g.scale + m;
                mx = std::max(mx, s[p]);
            }
            if (std::isinf(mx))
                continue;
            std::fill(acc.begin(), acc.end(), 0.0);
            double sum = 0.0;
            for (int p = 0; p < fc.n_kv; ++p) {
                if (std::isinf(s[p]))
                    continue;
                const double w = std::exp(s[p] - mx);
                sum += w;
                for (int d = 0; d < fc.head; ++d)
                    acc[d] += w * ggml_fp16_to_fp32(hdata(g.vc)[p * cell + hk * fc.head + d]);
            }
            for (int d = 0; d < fc.head; ++d)
                out[(static_cast<std::size_t>(t) * fc.n_head + h) * fc.head + d] =
                    static_cast<float>(acc[d] / sum);
        }
    return out;
}

// KERNEL 3, and KERNEL 4 when the plan splits, on the host.
std::vector<float> fa_emulate(const D3d12AttnFaPlan& plan, const ggml_tensor* node) {
    std::vector<float> got = poisoned(node);
    std::vector<float> partials(static_cast<std::size_t>(plan.scratch_floats), kPoison);
    d3d12_attn_fa_emulate(plan.params, plan.groups_x, plan.groups_y, plan.groups_z, fdata(plan.q),
                          hdata(plan.k), hdata(plan.v), plan.mask ? hdata(plan.mask) : nullptr,
                          partials.data(), got.data());
    if (plan.scratch_floats > 0)
        d3d12_attn_fa_reduce_emulate(plan.reduce, plan.reduce_groups_x, plan.reduce_groups_y,
                                     partials.data(), got.data());
    return got;
}

constexpr std::uint32_t kForceDirect = 1u << 20; // more tiles per chunk than any case has

} // namespace

TEST_CASE("d3d12 attn sincos: within 2e-7 of sin and cos up to 411000 rad, odd and even") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(0.f, 411000.f);
    std::vector<float> xs;
    for (int i = 0; i < 200000; ++i)
        xs.push_back(d(rng));
    for (int i = -4000; i <= 4000; ++i)
        xs.push_back(static_cast<float>(i) * 0.01f);
    for (int k = 1; k < 261000; k += 13) {
        const float x = static_cast<float>(k * 1.5707963267948966);
        xs.push_back(x);
        xs.push_back(std::nextafter(x, 1e9f));
        xs.push_back(std::nextafter(x, 0.f));
    }
    double es = 0.0, ec = 0.0;
    std::size_t asym = 0;
    for (float x : xs) {
        float s = 0.f, c = 0.f, sn = 0.f, cn = 0.f;
        d3d12_attn_sincos(x, &s, &c);
        d3d12_attn_sincos(-x, &sn, &cn);
        es = std::max(es, std::fabs(s - std::sin(static_cast<double>(x))));
        ec = std::max(ec, std::fabs(c - std::cos(static_cast<double>(x))));
        if (x != 0.f && (!same_bits(sn, -s) || !same_bits(cn, c)))
            ++asym;
    }
    CHECK(es <= 2e-7);
    CHECK(ec <= 2e-7);
    CHECK(asym == 0);
}

TEST_CASE("d3d12 attn f16: both conversions match ggml's, ties to even, overflow, subnormals") {
    std::size_t bad = 0;
    for (std::uint32_t h = 0; h < 65536; ++h) {
        const float a = d3d12_attn_f16_to_f32(static_cast<std::uint16_t>(h));
        const float b = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(h));
        if (std::isnan(a) || std::isnan(b)) {
            bad += std::isnan(a) != std::isnan(b);
            continue;
        }
        bad += !same_bits(a, b);
        bad += d3d12_attn_f32_to_f16(a) != h;
    }
    CHECK(bad == 0);

    // Halfway between neighbouring halves, and one float either side.
    std::size_t ties = 0;
    for (std::uint32_t h = 0; h < 0x7c00; ++h) {
        const float a = d3d12_attn_f16_to_f32(static_cast<std::uint16_t>(h));
        const float b =
            h == 0x7bff ? 65536.f : d3d12_attn_f16_to_f32(static_cast<std::uint16_t>(h + 1));
        const float mid = (a + b) / 2;
        for (float x : {mid, std::nextafter(mid, 0.f), std::nextafter(mid, 1e30f)})
            for (float y : {x, -x})
                ties += d3d12_attn_f32_to_f16(y) != ggml_fp32_to_fp16(y);
    }
    CHECK(ties == 0);

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> m(1.f, 2.f);
    std::uniform_int_distribution<int> e(-30, 17);
    std::vector<float> xs = {0.f,
                             -0.f,
                             1e-8f,
                             3e-8f,
                             6e-8f,
                             2.98023224e-8f,
                             5.96046448e-8f,
                             1e-40f,
                             1e30f,
                             70000.f,
                             65519.f,
                             65520.f,
                             std::numeric_limits<float>::min(),
                             std::numeric_limits<float>::infinity()};
    for (int i = 0; i < 200000; ++i)
        xs.push_back((i & 1 ? -1.f : 1.f) * std::ldexp(m(rng), e(rng)));
    std::size_t rnd = 0;
    for (float x : xs)
        for (float y : {x, -x})
            rnd += d3d12_attn_f32_to_f16(y) != ggml_fp32_to_fp16(y);
    CHECK(rnd == 0);
}

TEST_CASE("d3d12 attn rope: qwen35moe's imrope of q (a strided view) and of k in place") {
    Ctx c(32u << 20);
    std::mt19937 rng(21);
    const int n = 5;
    int sections[GGML_MROPE_SECTIONS] = {11, 11, 10, 0};

    // q and its gate interleaved per head, as qwen35moe's q projection.
    ggml_tensor* full = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 512, 16, n);
    fill(full, &rng, -1.f, 1.f);
    ggml_tensor* q = ggml_view_3d(c.ctx, full, 256, 16, n, full->nb[1], full->nb[2], 0);
    ggml_tensor* qr =
        ggml_rope_multi(c.ctx, q, mrope_pos(c.ctx, n), nullptr, 64, sections, GGML_ROPE_TYPE_IMROPE,
                        262144, 1e7f, 1.f, 0.f, 1.f, 32.f, 1.f);
    const D3d12AttnRopePlan p = check_rope(c.ctx, qr, 2e-6);
    CHECK(p.params.ne0_dims == (256u | 64u << 16));
    CHECK(p.params.offs_mode == (kD3d12RopeImrope << 16));
    CHECK(p.params.sections == (11u | 11u << 8 | 10u << 16));
    CHECK(p.params.src_row_stride == 512);
    CHECK(p.params.src_plane_stride == 512 * 16);
    CHECK(p.params.dst_row_stride == 256);
    CHECK(p.params.corr_dims == 0);
    CHECK(p.groups_x == 128 * 16 * n / 256);

    ggml_tensor* k = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 256, 2, n);
    fill(k, &rng, -1.f, 1.f);
    ggml_tensor* kr =
        ggml_rope_multi_inplace(c.ctx, k, mrope_pos(c.ctx, n), nullptr, 64, sections,
                                GGML_ROPE_TYPE_IMROPE, 262144, 1e7f, 1.f, 0.f, 1.f, 32.f, 1.f);
    REQUIRE(kr->data == k->data);
    check_rope(c.ctx, kr, 2e-6);

    // VISION is not a mode the kernel has.
    ggml_tensor* vr =
        ggml_rope_multi(c.ctx, k, mrope_pos(c.ctx, n), nullptr, 64, sections, GGML_ROPE_TYPE_IMROPE,
                        262144, 1e7f, 1.f, 0.f, 1.f, 32.f, 1.f);
    vr->op_params[2] = GGML_ROPE_TYPE_VISION;
    D3d12AttnRopePlan none;
    CHECK_FALSE(d3d12_attn_rope_plan(vr, &none));
}

TEST_CASE("d3d12 attn rope: mrope, neox with an offset, normal with freq factors") {
    Ctx c(16u << 20);
    std::mt19937 rng(22);

    int sections[GGML_MROPE_SECTIONS] = {16, 24, 24, 0};
    ggml_tensor* x = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 256, 4, 3);
    fill(x, &rng, -1.f, 1.f);
    check_rope(c.ctx,
               ggml_rope_multi(c.ctx, x, mrope_pos(c.ctx, 3), nullptr, 128, sections,
                               GGML_ROPE_TYPE_MROPE, 4096, 1e6f, 1.f, 0.f, 1.f, 32.f, 1.f),
               2e-6);

    ggml_tensor* y = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 128, 3, 4);
    fill(y, &rng, -1.f, 1.f);
    ggml_tensor* yr = ggml_rope_set_offset(ggml_rope_ext(c.ctx, y, text_pos(c.ctx, 4, 40), nullptr,
                                                         32, GGML_ROPE_TYPE_NEOX, 4096, 10000.f,
                                                         1.f, 0.f, 1.f, 32.f, 1.f),
                                           16);
    const D3d12AttnRopePlan py = check_rope(c.ctx, yr, 2e-6);
    CHECK((py.params.offs_mode & 0xffffu) == 16);

    ggml_tensor* z = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 128, 2, 3);
    ggml_tensor* ff = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 32);
    fill(z, &rng, -1.f, 1.f);
    fill(ff, &rng, 1.f, 4.f);
    const D3d12AttnRopePlan pz =
        check_rope(c.ctx,
                   ggml_rope_ext(c.ctx, z, text_pos(c.ctx, 3, 7), ff, 64, GGML_ROPE_TYPE_NORMAL,
                                 4096, 10000.f, 1.f, 0.f, 1.f, 32.f, 1.f),
                   2e-6);
    CHECK(pz.params.offs_mode == (kD3d12RopeNormal << 16 | 1u << 24));
}

TEST_CASE("d3d12 attn rope: yarn, with its ramp and with n_ctx_orig 0 (high = -inf)") {
    Ctx c(16u << 20);
    std::mt19937 rng(23);
    // Small positions: the CPU may fuse the ramp's mix into an fma, a rounding
    // the angle carries into sin and cos, so keep the angles small.
    ggml_tensor* x = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 128, 2, 6);
    fill(x, &rng, -1.f, 1.f);
    const D3d12AttnRopePlan p =
        check_rope(c.ctx,
                   ggml_rope_ext(c.ctx, x, text_pos(c.ctx, 6, 0), nullptr, 128, GGML_ROPE_TYPE_NEOX,
                                 4096, 10000.f, 0.25f, 1.f, 1.f, 32.f, 1.f),
                   2e-6);
    float dims[2] = {};
    ggml_rope_yarn_corr_dims(128, 4096, 10000.f, 32.f, 1.f, dims);
    CHECK(p.params.corr_dims ==
          (static_cast<std::uint32_t>(dims[0]) | static_cast<std::uint32_t>(dims[1]) << 16));
    CHECK(dims[0] > 0.f);
    CHECK(dims[1] > dims[0]);

    const D3d12AttnRopePlan q =
        check_rope(c.ctx,
                   ggml_rope_ext(c.ctx, x, text_pos(c.ctx, 6, 0), nullptr, 128, GGML_ROPE_TYPE_NEOX,
                                 0, 10000.f, 0.25f, 1.f, 1.f, 32.f, 1.f),
                   2e-6);
    CHECK(q.params.corr_dims == 0x80000000u);
}

TEST_CASE("d3d12 attn set_rows: f16 cells bit for bit from a strided source by i64 rows") {
    Ctx c(8u << 20);
    std::mt19937 rng(31);
    ggml_tensor* cache = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, 512, 64);
    fill_f16(cache, &rng, -1.f, 1.f);
    ggml_tensor* full = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 1024, 5);
    fill(full, &rng, -2.f, 2.f);
    const float specials[] = {70000.f,        -70000.f,       65519.f, 65520.f, 1e-8f,
                              3e-8f,          6e-8f,          -6e-8f,  0.f,     -0.f,
                              2.98023224e-8f, 5.96046448e-8f, 1e-40f,  1e30f};
    std::memcpy(full->data, specials, sizeof(specials));
    ggml_tensor* src = ggml_view_2d(c.ctx, full, 512, 5, full->nb[1], 0);
    ggml_tensor* idx = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I64, 5);
    const std::int64_t rows[] = {3, 17, 0, 63, 40};
    std::memcpy(idx->data, rows, sizeof(rows));
    ggml_tensor* node = ggml_set_rows(c.ctx, cache, src, idx);
    REQUIRE(node->data == cache->data);

    D3d12AttnSetRowsPlan plan;
    REQUIRE(d3d12_attn_set_rows_plan(node, &plan));
    CHECK(plan.params.flags == (kD3d12SetRowsF16 | kD3d12SetRowsI64));
    CHECK(plan.params.src_row_stride == 1024);
    CHECK(plan.params.dst_row_stride == 512);
    CHECK(plan.params.dst_rows == 64);
    CHECK(plan.groups_x == 1);
    CHECK(plan.groups_y == 5);

    const std::size_t nb = ggml_nbytes(cache);
    std::vector<unsigned char> got(nb);
    std::memcpy(got.data(), cache->data, nb);
    const std::vector<float> s = span_copy(src);
    d3d12_attn_set_rows_emulate(plan.params, plan.groups_x, plan.groups_y, s.data(),
                                static_cast<const std::uint32_t*>(idx->data), got.data());
    std::vector<unsigned char> before = got;
    compute(c.ctx, node);
    CHECK(std::memcmp(got.data(), cache->data, nb) == 0);

    // An i64 row with a high word, or past the cache, is skipped.
    std::memcpy(before.data(), cache->data, nb);
    std::vector<std::uint32_t> words(10);
    std::memcpy(words.data(), rows, sizeof(rows));
    words[3] = 1;  // row 17's high word
    words[6] = 64; // row 63 -> 64, past the cache
    std::vector<float> fresh(s.size());
    for (std::size_t i = 0; i < fresh.size(); ++i)
        fresh[i] = s[i] + 0.5f;
    std::vector<unsigned char> skip = before;
    d3d12_attn_set_rows_emulate(plan.params, plan.groups_x, plan.groups_y, fresh.data(),
                                words.data(), skip.data());
    const std::size_t row_bytes = 512 * sizeof(std::uint16_t);
    for (int r = 0; r < 64; ++r) {
        const bool written = r == 3 || r == 0 || r == 40;
        CAPTURE(r);
        CHECK((std::memcmp(skip.data() + r * row_bytes, before.data() + r * row_bytes, row_bytes) !=
               0) == written);
    }
}

TEST_CASE("d3d12 attn set_rows: f32 rows by i32 rows; odd f16 rows are not taken") {
    Ctx c(4u << 20);
    std::mt19937 rng(32);
    ggml_tensor* dst = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 64, 10);
    ggml_tensor* src = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 64, 3);
    fill(dst, &rng, -1.f, 1.f);
    fill(src, &rng, -1.f, 1.f);
    ggml_tensor* idx = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, 3);
    const std::int32_t rows[] = {9, 0, 4};
    std::memcpy(idx->data, rows, sizeof(rows));
    ggml_tensor* node = ggml_set_rows(c.ctx, dst, src, idx);

    D3d12AttnSetRowsPlan plan;
    REQUIRE(d3d12_attn_set_rows_plan(node, &plan));
    CHECK(plan.params.flags == 0);
    CHECK(plan.params.idx_stride == 1);
    const std::size_t nb = ggml_nbytes(dst);
    std::vector<unsigned char> got(nb);
    std::memcpy(got.data(), dst->data, nb);
    d3d12_attn_set_rows_emulate(plan.params, plan.groups_x, plan.groups_y, fdata(src),
                                static_cast<const std::uint32_t*>(idx->data), got.data());
    compute(c.ctx, node);
    CHECK(std::memcmp(got.data(), dst->data, nb) == 0);

    ggml_tensor* odd = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, 7, 4);
    ggml_tensor* osrc = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 7, 2);
    ggml_tensor* oidx = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I64, 2);
    CHECK_FALSE(d3d12_attn_set_rows_plan(ggml_set_rows(c.ctx, odd, osrc, oidx), &plan));
}

TEST_CASE("d3d12 attn fa: every split of the kv range against double and the CPU") {
    // qwen35moe's layer (head 256, 16 q heads on 2 kv heads) at decode and
    // small batches, then 2, 1 and 16 q heads per kv head.
    const FaCase cases[] = {
        {256, 16, 2, 1, 256}, {256, 16, 2, 1, 300}, {256, 16, 2, 5, 300}, {256, 16, 2, 5, 512},
        {256, 16, 2, 3, 70},  {64, 8, 4, 2, 100},   {128, 4, 4, 3, 200},  {128, 16, 1, 2, 600},
    };
    for (const FaCase& fc : cases) {
        CAPTURE(fc.head);
        CAPTURE(fc.n_head);
        CAPTURE(fc.n_head_kv);
        CAPTURE(fc.n_tok);
        CAPTURE(fc.n_kv);
        Ctx c(32u << 20);
        std::mt19937 rng(41 + fc.n_kv + fc.n_tok);
        const FaGraph g = fa_graph(c.ctx, fc, &rng);
        const std::vector<float> ref = fa_ref(fc, g);
        compute(c.ctx, g.node);
        const std::uint32_t grp = static_cast<std::uint32_t>(fc.n_head / fc.n_head_kv);
        const std::uint32_t tiles =
            (static_cast<std::uint32_t>(fc.n_kv) + 256 / grp - 1) / (256 / grp);
        for (std::uint32_t ct : {0u, 1u, 3u, kForceDirect}) {
            CAPTURE(ct);
            D3d12AttnFaPlan plan;
            REQUIRE(d3d12_attn_fa_plan(g.node, &plan, ct));
            const std::uint32_t n_chunks = plan.params.tokens_chunks >> 16;
            CHECK(plan.groups_x == n_chunks);
            CHECK(plan.groups_y == static_cast<std::uint32_t>(fc.n_head_kv));
            CHECK(plan.groups_z == static_cast<std::uint32_t>(fc.n_tok));
            CHECK(n_chunks * plan.params.chunk_len >= static_cast<std::uint32_t>(fc.n_kv));
            CHECK((n_chunks - 1) * plan.params.chunk_len < static_cast<std::uint32_t>(fc.n_kv));
            if (ct == 1)
                CHECK(n_chunks == tiles);
            if (ct == kForceDirect)
                CHECK(n_chunks == 1);
            const bool direct = (plan.params.heads_flags & kD3d12FaDirect) != 0;
            CHECK(direct == (n_chunks == 1));
            CHECK((plan.scratch_floats == 0) == direct);
            CHECK((plan.params.heads_flags & kD3d12FaMask) != 0);
            const std::vector<float> got = fa_emulate(plan, g.node);
            const double e_ref = d3d12_attn_max_err(got.data(), ref.data(), ref.size());
            CAPTURE(e_ref);
            CHECK(e_ref <= 2e-6);
            // The CPU rounds q to f16 and sums v in f16.
            check_out(g.node, got, 4e-3);
        }
    }
}

TEST_CASE("d3d12 attn fa: decode splits the kv range; a batch of 64 groups does not") {
    Ctx c(32u << 20);
    std::mt19937 rng(43);
    const FaGraph decode = fa_graph(c.ctx, {256, 16, 2, 1, 4096}, &rng);
    D3d12AttnFaPlan plan;
    REQUIRE(d3d12_attn_fa_plan(decode.node, &plan));
    CHECK(plan.groups_x * plan.groups_y >= 64);
    CHECK(plan.scratch_floats ==
          static_cast<std::uint64_t>(16) * plan.groups_x * (256 + kD3d12AttnPartialHeader));
    CHECK(plan.reduce_groups_x == 16);
    CHECK(plan.reduce_groups_y == 1);

    const FaGraph batch = fa_graph(c.ctx, {256, 16, 2, 32, 4096}, &rng);
    REQUIRE(d3d12_attn_fa_plan(batch.node, &plan));
    CHECK(plan.groups_x == 1);
    CHECK(plan.scratch_floats == 0);
}

TEST_CASE("d3d12 attn fa: one visible cell per token gives that cell's v bit for bit") {
    const FaCase fc{256, 16, 2, 5, 300};
    Ctx c(16u << 20);
    std::mt19937 rng(44);
    const FaGraph g = fa_graph(c.ctx, fc, &rng);
    const int seen[] = {0, 299, 137, 64, -1}; // the last token sees nothing: zeros
    auto* m = static_cast<std::uint16_t*>(g.mask->data);
    for (int t = 0; t < fc.n_tok; ++t)
        for (int p = 0; p < fc.n_kv; ++p)
            m[t * fc.n_kv + p] = p == seen[t] ? 0 : kHalfNegInf;
    compute(c.ctx, g.node);
    for (std::uint32_t ct : {0u, 1u, 3u, kForceDirect}) {
        CAPTURE(ct);
        D3d12AttnFaPlan plan;
        REQUIRE(d3d12_attn_fa_plan(g.node, &plan, ct));
        const std::vector<float> got = fa_emulate(plan, g.node);
        check_out(g.node, got, 0.0);
        std::size_t off = 0;
        for (int t = 0; t < fc.n_tok; ++t)
            for (int h = 0; h < fc.n_head; ++h) {
                const std::size_t base = (static_cast<std::size_t>(t) * fc.n_head + h) * fc.head;
                const std::size_t cell =
                    static_cast<std::size_t>(seen[t] < 0 ? 0 : seen[t]) * fc.head * fc.n_head_kv +
                    static_cast<std::size_t>(h / 8) * fc.head;
                for (int d = 0; d < fc.head; ++d) {
                    const float want = seen[t] < 0 ? 0.f : ggml_fp16_to_fp32(hdata(g.vc)[cell + d]);
                    off += !same_bits(got[base + d], want);
                }
            }
        CHECK(off == 0);
    }
}

TEST_CASE("d3d12 attn fa: the plan takes no alibi, softcap, odd groups or wide heads") {
    Ctx c(16u << 20);
    std::mt19937 rng(45);
    D3d12AttnFaPlan plan;
    CHECK_FALSE(d3d12_attn_fa_plan(fa_graph(c.ctx, {64, 8, 2, 1, 32}, &rng, 0.5f).node, &plan));
    CHECK_FALSE(
        d3d12_attn_fa_plan(fa_graph(c.ctx, {64, 8, 2, 1, 32}, &rng, 0.f, 30.f).node, &plan));
    CHECK_FALSE(d3d12_attn_fa_plan(fa_graph(c.ctx, {64, 6, 2, 1, 32}, &rng).node, &plan));
    CHECK_FALSE(d3d12_attn_fa_plan(fa_graph(c.ctx, {64, 32, 1, 1, 32}, &rng).node, &plan));
    CHECK_FALSE(d3d12_attn_fa_plan(fa_graph(c.ctx, {20, 4, 2, 1, 32}, &rng).node, &plan));
    CHECK_FALSE(d3d12_attn_fa_plan(fa_graph(c.ctx, {512, 4, 2, 1, 32}, &rng).node, &plan));
    CHECK(d3d12_attn_fa_plan(fa_graph(c.ctx, {64, 8, 2, 1, 32}, &rng).node, &plan));
}
