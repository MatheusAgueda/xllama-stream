// Host harness for GGML_CPU_FA_GQA (patch 0024): the grouped-query split-KV flash-attention chunk against the
// per-head path it replaces, byte for byte, on the decode shape of Ornith/Qwen3.5-35B-A3B (16 query heads, 2 K/V
// heads, DK = DV = 256, f16 K/V in llama.cpp's cache layout), each layer with its own KV so it is cold like on the
// console. The variable is read per call, so both paths run in one process. Not in the CMake build.
// From a directory holding this file, a CMakeLists.txt with
//   cmake_minimum_required(VERSION 3.14) / project(h24 C CXX) / set(CMAKE_CXX_STANDARD 17)
//   set(GGML_METAL OFF CACHE BOOL "" FORCE) (same for GGML_BLAS, GGML_OPENMP, BUILD_SHARED_LIBS)
//   add_subdirectory(${GGML_SRC} ggml) / add_executable(h24 cpu_fa_gqa.cpp)
//   target_link_libraries(h24 PRIVATE ggml ggml-cpu ggml-base)
// and GGML_SRC = the patched llama.cpp/ggml (x86 like the console: -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON
// -DGGML_F16C=ON -DGGML_AVX512=OFF):
//   ./h24 [layers=10] [reps=5]                                  -> ALL_BIT_EXACT (exit 1 on any differing byte)
#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct Case { int nkv; int nth; float max_bias; float softcap; bool sinks; int holes; };

static double run(ggml_cgraph * gf, ggml_cplan & cp, int reps) {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) ggml_graph_compute(gf, &cp);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

int main(int argc, char ** argv) {
    const int layers = argc > 1 ? atoi(argv[1]) : 10;  // separate KV per layer, so a layer's KV is cold like on the console
    const int reps   = argc > 2 ? atoi(argv[2]) : 5;
    const int DK = 256, DV = 256, NH = 16, NHKV = 2;
    const Case cases[] = {
        {512, 6, 0, 0, false, 0}, {1000, 6, 0, 0, false, 7}, {4096, 6, 0, 0, false, 0}, {10000, 6, 0, 0, false, 31},
        {16384, 6, 0, 0, false, 0}, {10000, 3, 0, 0, false, 0}, {520, 30, 0, 0, false, 3}, {4096, 6, 8.0f, 0, false, 5},
        {4096, 6, 0, 30.0f, false, 0}, {4096, 6, 0, 0, true, 9},
    };
    int bad = 0;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const Case & c : cases) {
        const size_t kv_bytes = (size_t) layers * 2 * c.nkv * DK * NHKV * sizeof(ggml_fp16_t);
        ggml_init_params ip = { kv_bytes + (size_t) layers * (64 << 10) + (size_t) c.nkv * 8 * layers + (64u << 20), NULL, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4 * layers + 16, false);
        std::vector<ggml_tensor *> outs;
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
            ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf((float) DK), c.max_bias, c.softcap);
            if (c.sinks) {
                ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NH);
                for (int h = 0; h < NH; ++h) ((float *) s->data)[h] = nd(rng) * 2.0f;
                ggml_flash_attn_ext_add_sinks(o, s);
            }
            ggml_build_forward_expand(gf, o);
            outs.push_back(o);
        }
        ggml_threadpool_params tpp = ggml_threadpool_params_default(c.nth);
        ggml_threadpool * tp = ggml_threadpool_new(&tpp);
        ggml_cplan cp = ggml_graph_plan(gf, c.nth, tp);
        std::vector<uint8_t> work(cp.work_size);
        cp.work_data = work.data();

        const size_t ob = ggml_nbytes(outs[0]);
        std::vector<uint8_t> ref(ob * layers), got(ob * layers);
        setenv("GGML_CPU_FA_GQA", "0", 1);
        ggml_graph_compute(gf, &cp);
        for (int l = 0; l < layers; ++l) memcpy(ref.data() + l * ob, outs[l]->data, ob);
        const double t_old = run(gf, cp, reps);
        setenv("GGML_CPU_FA_GQA", "1", 1);
        for (int l = 0; l < layers; ++l) memset(outs[l]->data, 0x7f, ob);
        ggml_graph_compute(gf, &cp);
        for (int l = 0; l < layers; ++l) memcpy(got.data() + l * ob, outs[l]->data, ob);
        const double t_new = run(gf, cp, reps);
        const bool same = memcmp(ref.data(), got.data(), ref.size()) == 0;
        size_t diff = 0;
        for (size_t i = 0; i < ref.size(); ++i) diff += ref[i] != got[i];
        bool finite = true;
        for (size_t i = 0; i < ref.size() / 4; ++i) finite &= std::isfinite(((float *) ref.data())[i]);
        printf("nkv=%5d nth=%2d bias=%g softcap=%g sinks=%d holes=%2d  %s (%zu bytes differ, finite=%d)  "
               "old %.3f ms  gqa %.3f ms  x%.2f  (%d layers)\n",
               c.nkv, c.nth, c.max_bias, c.softcap, c.sinks, c.holes, same ? "BIT-EXACT" : "DIFFERS", diff, finite,
               t_old, t_new, t_old / t_new, layers);
        bad += !same || !finite;
        ggml_threadpool_free(tp);
        ggml_free(ctx);
    }
    printf(bad ? "FAIL %d\n" : "ALL_BIT_EXACT\n", bad);
    return bad ? 1 : 0;
}
