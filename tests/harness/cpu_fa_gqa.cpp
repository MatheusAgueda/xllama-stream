// Host harness for GGML_CPU_FA_GQA (patches 0024, 0025) on the decode shape of Ornith/Qwen3.5-35B-A3B (16 query
// heads, 2 K/V heads, DK = DV = 256, f16 K/V in llama.cpp's cache layout), each layer with its own KV so it is cold
// like on the console. The variable is read per call, so the three paths run in one process:
//   0 = per-head loop · 1 = grouped chunk, must equal 0 byte for byte · 2 = fused chunk (f32 Q and accumulators),
//   whose error against an f64 reference must not exceed the per-head loop's (max and rms).
// Then a thread sweep (1/2/3/6 threads at 10K and 16K rows) and a single-thread breakdown of one (head, row) pair of
// the per-head loop into its f16 dot, f16 mad and expf. Not in the CMake build.
// From a directory holding this file, a CMakeLists.txt with
//   cmake_minimum_required(VERSION 3.14) / project(h24 C CXX) / set(CMAKE_CXX_STANDARD 17)
//   set(GGML_METAL OFF CACHE BOOL "" FORCE) (same for GGML_BLAS, GGML_OPENMP, BUILD_SHARED_LIBS)
//   add_subdirectory(${GGML_SRC} ggml) / add_executable(h24 cpu_fa_gqa.cpp)
//   target_include_directories(h24 PRIVATE ${GGML_SRC}/src ${GGML_SRC}/src/ggml-cpu)
//   target_compile_options(h24 PRIVATE -mavx2 -mfma -mf16c)
//   target_link_libraries(h24 PRIVATE ggml ggml-cpu ggml-base)
// and GGML_SRC = the patched llama.cpp/ggml, configured -DCMAKE_BUILD_TYPE=Release and x86 like the console
// (-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF):
//   ./h24 [layers=6] [reps=5]                          -> ALL_PASS (exit 1 on a differing byte or a less exact fused)
#include "ggml.h"
#include "ggml-cpu.h"
#include "vec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static const int DK = 256, DV = 256, NH = 16, NHKV = 2;

struct Case { int nkv; int nth; float max_bias; float softcap; bool sinks; int holes; bool sweep; };
struct Err  { double max_abs = 0, sq = 0, ref_sq = 0; double rms() const { return ref_sq > 0 ? sqrt(sq/ref_sq) : 0; } };

static double run(ggml_cgraph * gf, ggml_cplan & cp, int reps) {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) ggml_graph_compute(gf, &cp);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

// f64 attention for one layer, from the f32 Q and the f16 K/V/mask exactly as stored
static void reference(const Case & c, float scale, const float * q, const ggml_fp16_t * kd, const ggml_fp16_t * vd,
                      const ggml_fp16_t * md, const float * sinks, double * out) {
    const int G = NH/NHKV;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(NH));
    const double m0 = pow(2.0, -(c.max_bias) / n_head_log2), m1 = pow(2.0, -(c.max_bias/2.0) / n_head_log2);
    std::vector<float> kf((size_t) c.nkv*DK*NHKV), vf((size_t) c.nkv*DV*NHKV), mf(c.nkv);
    for (size_t i = 0; i < kf.size(); ++i) kf[i] = ggml_fp16_to_fp32(kd[i]);
    for (size_t i = 0; i < vf.size(); ++i) vf[i] = ggml_fp16_to_fp32(vd[i]);
    for (int i = 0; i < c.nkv; ++i) mf[i] = ggml_fp16_to_fp32(md[i]);
    std::vector<double> s(c.nkv);
    for (int h = 0; h < NH; ++h) {
        const int kvh = h/G;
        const double slope = c.max_bias > 0 ? (h < (int) n_head_log2 ? pow(m0, h + 1) : pow(m1, 2*(h - n_head_log2) + 1)) : 1.0;
        double M = -INFINITY;
        for (int i = 0; i < c.nkv; ++i) {
            if (mf[i] == -INFINITY) { s[i] = -INFINITY; continue; }
            const float * kr = kf.data() + (size_t) i*DK*NHKV + kvh*DK;
            double d = 0;
            for (int j = 0; j < DK; ++j) d += (double) q[h*DK + j]*kr[j];
            d = c.softcap != 0 ? c.softcap*tanh(d*scale/c.softcap) : d*scale;
            s[i] = d + slope*mf[i];
            M = std::max(M, s[i]);
        }
        if (sinks) M = std::max(M, (double) sinks[h]);
        double S = sinks ? exp(sinks[h] - M) : 0.0;
        double * o = out + h*DV;
        std::fill(o, o + DV, 0.0);
        for (int i = 0; i < c.nkv; ++i) {
            if (s[i] == -INFINITY) continue;
            const double w = exp(s[i] - M);
            const float * vr = vf.data() + (size_t) i*DV*NHKV + kvh*DV;
            S += w;
            for (int d = 0; d < DV; ++d) o[d] += w*vr[d];
        }
        for (int d = 0; d < DV; ++d) o[d] /= S;
    }
}

static void add_err(Err & e, const float * got, const double * ref, int n) {
    for (int i = 0; i < n; ++i) {
        const double d = fabs((double) got[i] - ref[i]);
        e.max_abs = std::max(e.max_abs, std::isfinite(d) ? d : INFINITY);
        e.sq += d*d;
        e.ref_sq += ref[i]*ref[i];
    }
}

// single-thread cost of one (head, row) pair of the per-head loop, by part, on 2048 rows of one K/V head held in L2/L3
static void breakdown(int reps) {
    const int R = 2048, G = NH/NHKV;
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<ggml_fp16_t> K((size_t) R*DK), V((size_t) R*DV), Q16((size_t) G*DK), A16((size_t) G*DV, ggml_fp32_to_fp16(0.0f));
    std::vector<float> x((size_t) R*G);
    for (auto & h : K) h = ggml_fp32_to_fp16(nd(rng)*0.7f);
    for (auto & h : V) h = ggml_fp32_to_fp16(nd(rng));
    for (auto & h : Q16) h = ggml_fp32_to_fp16(nd(rng));
    for (auto & f : x) f = -fabsf(nd(rng))*4.0f;
    volatile float sink = 0;
    auto per_pair = [&](auto && body) {
        body();
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) body();
        return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / ((double) reps*R*G);
    };
    const double t_dot = per_pair([&] {
        float acc = 0;
        for (int ic = 0; ic < R; ++ic)
            for (int g = 0; g < G; ++g) { float s; ggml_vec_dot_f16(DK, &s, 0, K.data() + (size_t) ic*DK, 0, Q16.data() + g*DK, 0, 1); acc += s; }
        sink = sink + acc;
    });
    const double t_mad = per_pair([&] {
        for (int ic = 0; ic < R; ++ic)
            for (int g = 0; g < G; ++g) ggml_vec_mad_f16(DV, A16.data() + g*DV, V.data() + (size_t) ic*DV, 1e-4f);
    });
    const double t_exp = per_pair([&] {
        float acc = 0;
        for (size_t i = 0; i < x.size(); ++i) acc += expf(x[i]);
        sink = sink + acc;
    });
    printf("breakdown 1 thread, per (head, row) pair of the per-head loop: f16 dot %.2f ns  f16 mad %.2f ns  expf %.2f ns  "
           "sum %.2f ns\n", t_dot, t_mad, t_exp, t_dot + t_mad + t_exp);
}

int main(int argc, char ** argv) {
    const int layers = argc > 1 ? atoi(argv[1]) : 6;  // separate KV per layer, so a layer's KV is cold like on the console
    const int reps   = argc > 2 ? atoi(argv[2]) : 5;
    const Case cases[] = {
        {512, 6, 0, 0, false, 0, false}, {1000, 6, 0, 0, false, 7, false}, {4096, 6, 0, 0, false, 0, false},
        {10000, 6, 0, 0, false, 31, false}, {16384, 6, 0, 0, false, 0, false}, {520, 30, 0, 0, false, 3, false},
        {4096, 6, 8.0f, 0, false, 5, false}, {4096, 6, 0, 30.0f, false, 0, false}, {4096, 6, 0, 0, true, 9, false},
        {10000, 1, 0, 0, false, 0, true}, {10000, 2, 0, 0, false, 0, true}, {10000, 3, 0, 0, false, 0, true},
        {16384, 1, 0, 0, false, 0, true}, {16384, 2, 0, 0, false, 0, true}, {16384, 3, 0, 0, false, 0, true},
        {16384, 6, 0, 0, false, 0, true},
    };
    const char * mode_env[3] = {"0", "1", "2"};
    const float scale = 1.0f/sqrtf((float) DK);
    int bad = 0;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const Case & c : cases) {
        const size_t kv_bytes = (size_t) layers * 2 * c.nkv * DK * NHKV * sizeof(ggml_fp16_t);
        ggml_init_params ip = { kv_bytes + (size_t) layers * (64 << 10) + (size_t) c.nkv * 8 * layers + (64u << 20), NULL, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4 * layers + 16, false);
        std::vector<ggml_tensor *> outs, qs, kcs, vcs, ms, ss;
        for (int l = 0; l < layers; ++l) {
            ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, DK * NHKV, c.nkv);
            ggml_tensor * vc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, DV * NHKV, c.nkv);
            ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, 1, NH, 1);
            ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.nkv, 1, 1, 1);
            ggml_fp16_t * kd = (ggml_fp16_t *) kc->data, * vd = (ggml_fp16_t *) vc->data, * md = (ggml_fp16_t *) m->data;
            for (int64_t i = 0; i < ggml_nelements(kc); ++i) kd[i] = ggml_fp32_to_fp16(nd(rng) * 0.7f);
            for (int64_t i = 0; i < ggml_nelements(vc); ++i) vd[i] = ggml_fp32_to_fp16(nd(rng));
            for (int64_t i = 0; i < ggml_nelements(q); ++i) ((float *) q->data)[i] = nd(rng);
            for (int i = 0; i < c.nkv; ++i) md[i] = ggml_fp32_to_fp16(0.0f);
            for (int h = 0; h < c.holes; ++h) md[rng() % c.nkv] = ggml_fp32_to_fp16(-INFINITY);
            if (c.holes) md[c.nkv - 1] = ggml_fp32_to_fp16(-INFINITY);
            ggml_tensor * k = ggml_view_3d(ctx, kc, DK, c.nkv, NHKV, kc->nb[1], DK * sizeof(ggml_fp16_t), 0);
            ggml_tensor * v = ggml_view_3d(ctx, vc, DV, c.nkv, NHKV, vc->nb[1], DV * sizeof(ggml_fp16_t), 0);
            ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, scale, c.max_bias, c.softcap);
            ggml_tensor * s = NULL;
            if (c.sinks) {
                s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NH);
                for (int h = 0; h < NH; ++h) ((float *) s->data)[h] = nd(rng) * 2.0f;
                ggml_flash_attn_ext_add_sinks(o, s);
            }
            ggml_build_forward_expand(gf, o);
            outs.push_back(o); qs.push_back(q); kcs.push_back(kc); vcs.push_back(vc); ms.push_back(m); ss.push_back(s);
        }
        ggml_threadpool_params tpp = ggml_threadpool_params_default(c.nth);
        ggml_threadpool * tp = ggml_threadpool_new(&tpp);
        ggml_cplan cp = ggml_graph_plan(gf, c.nth, tp);
        std::vector<uint8_t> work(cp.work_size);
        cp.work_data = work.data();

        const size_t ob = ggml_nbytes(outs[0]);
        std::vector<uint8_t> got[3];
        double t[3];
        for (int mo = 0; mo < 3; ++mo) {
            setenv("GGML_CPU_FA_GQA", mode_env[mo], 1);
            for (int l = 0; l < layers; ++l) memset(outs[l]->data, 0x7f, ob);
            ggml_graph_compute(gf, &cp);
            got[mo].resize(ob * layers);
            for (int l = 0; l < layers; ++l) memcpy(got[mo].data() + l * ob, outs[l]->data, ob);
            t[mo] = run(gf, cp, reps);
        }
        const bool same = memcmp(got[0].data(), got[1].data(), got[0].size()) == 0;
        size_t diff = 0;
        for (size_t i = 0; i < got[0].size(); ++i) diff += got[0][i] != got[1][i];
        bool fused_ok = true;
        char errs[160] = "";
        if (!c.sweep) {
            Err e[3];
            std::vector<double> ref((size_t) NH * DV);
            for (int l = 0; l < layers; ++l) {
                reference(c, scale, (const float *) qs[l]->data, (const ggml_fp16_t *) kcs[l]->data,
                          (const ggml_fp16_t *) vcs[l]->data, (const ggml_fp16_t *) ms[l]->data,
                          ss[l] ? (const float *) ss[l]->data : NULL, ref.data());
                for (int mo = 0; mo < 3; ++mo) add_err(e[mo], (const float *) (got[mo].data() + l * ob), ref.data(), NH * DV);
            }
            fused_ok = e[2].max_abs <= e[0].max_abs && e[2].rms() <= e[0].rms();
            snprintf(errs, sizeof errs, "  err max %.2e/%.2e rms %.2e/%.2e (old/fused)%s", e[0].max_abs, e[2].max_abs,
                     e[0].rms(), e[2].rms(), fused_ok ? "" : "  FUSED-LESS-EXACT");
        }
        printf("nkv=%5d nth=%2d bias=%g softcap=%g sinks=%d holes=%2d  %s (%zu bytes)  old %.3f  gqa %.3f  fused %.3f ms  "
               "x%.2f x%.2f%s%s  (%d layers)\n",
               c.nkv, c.nth, c.max_bias, c.softcap, c.sinks, c.holes, same ? "BIT-EXACT" : "DIFFERS", diff,
               t[0], t[1], t[2], t[0] / t[1], t[0] / t[2], c.sweep ? "  sweep" : "", errs, layers);
        fflush(stdout);
        bad += !same || !fused_ok;
        ggml_threadpool_free(tp);
        ggml_free(ctx);
    }
    breakdown(reps * 4);
    printf(bad ? "FAIL %d\n" : "ALL_PASS\n", bad);
    return bad ? 1 : 0;
}
