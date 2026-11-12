// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// ggml backend "d3d12" — GGUF GPU decode D2 (docs/gguf-gpu-decode.md).
//
// A GPU-type ggml device, registered at runtime, that runs MUL_MAT with
// Q4_0 / Q4_K / Q6_K / IQ3_S weights, GET_ROWS from a Q6_K weight and the f32
// adds, norms and SwiGLU between them on our D3D12 compute shaders; matmul
// inputs are quantized to q8 the way the CPU backend does (q8_K on the GPU).
// Two buffer types:
//   D3D12_Weights  DEFAULT heap, holds matmul weights (exposed as an extra buft)
//   D3D12_Host     CUSTOM WRITE_BACK heap, is_host — the device default buft, so
//                  the scheduler's activations are CPU-visible: CPU<->GPU copies
//                  are memcpy and the CPU reads results in place.
// supports_op accepts an op only when its weight already lives in
// D3D12_Weights, which steers llama.cpp's per-weight buft probe past the host
// buft; every other op and weight falls back to the CPU.
//
// This header is WinRT- and D3D12-free: the pure rules below are host-tested on
// Linux, where ggml_d3d12_register() returns false.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace xllama {

// Rows per thread group and threads per group of every mmv kernel
// (shaders/ggml_d3d12_mmv_*.hlsl).
inline constexpr int kD3d12MmvRows = 4;
// Threads per group: 64 or 128, chosen per matmul by d3d12_mm_threads(). 16
// threads share one 256-element chunk, so threads/16 chunks are in flight.
inline constexpr int kD3d12MmvThreadsShort = 64;
inline constexpr int kD3d12MmvThreadsLong = 128;
inline constexpr int kD3d12LongKChunks = 16; // K >= 4096 → 128 threads
// Every kernel walks K in 256-element chunks (one Q4_K/Q6_K super-block, eight
// Q4_0 blocks).
inline constexpr int kD3d12Chunk = 256;
inline constexpr std::uint32_t kD3d12MaxGroups = 65535;

// Weight types the backend runs.
bool d3d12_weight_type_supported(ggml_type t);

// What supports_op sees, as plain data (no ggml_tensor needed in tests).
struct D3d12MatmulDesc {
    ggml_type src0_type = GGML_TYPE_F32;
    std::int64_t ne00 = 0, ne01 = 0, ne02 = 1, ne03 = 1; // weight [K, N]
    std::int64_t ne10 = 0, ne11 = 0, ne12 = 1, ne13 = 1; // activations [K, M]
    ggml_type src1_type = GGML_TYPE_F32;
    ggml_type dst_type = GGML_TYPE_F32;
    bool src0_contiguous = true;
    bool src1_contiguous = true;
    bool src0_in_weight_buffer = false; // weight lives in a D3D12_Weights buffer
};

// The MUL_MAT rules of supports_op.
bool d3d12_mm_supported(const D3d12MatmulDesc& d);

struct D3d12Dispatch {
    std::uint32_t groups_x = 0; // ceil(N / kD3d12MmvRows)
    std::uint32_t groups_y = 0; // one per activation column (0: nothing to run)
    bool ok = false;            // within the 65535 group limit
};

D3d12Dispatch d3d12_mm_dispatch(std::int64_t n, std::int64_t ncols);

// Kernel width for a K: 128 threads (8 chunks in flight) once K has at least
// kD3d12LongKChunks chunks, else 64 — measured both ways in D2a runs 1 and 2.
int d3d12_mm_threads(std::int64_t k);

// --- IQ3_S / IQ2_S group geometry (shaders/ggml_d3d12_geom.hlsli) ---
// The IQ3_S and IQ2_S mmv and mmid kernels read their group shape from pad0 (constant 6):
// rsl row sets side by side in the group's threads/16 chunk lanes and rt row
// tiles the group walks in turn, so group gid.x covers rows
// [gid.x * R, (gid.x + 1) * R) with R = kD3d12MmvRows * rsl * rt. rsl = rt = 1
// is the plain layout (and pad0 = 0 reads as that). rsl > 1 puts the idle chunk
// lanes of a short K (ffn_down_exps K = 512: 2 chunks, 4 lanes) on more rows; rt
// > 1 spreads the group's 2 KiB grid preload over more rows.
struct D3d12Geom {
    std::uint32_t rsl = 1;
    std::uint32_t rt = 1;
};

// What a backend asks for (XLLAMA_D3D12_IQ3S_RSL / _RT): rsl 0 = auto (the
// idle lanes of a K shorter than the lanes, nothing else), rt as given.
struct D3d12GeomReq {
    std::uint32_t rsl = 1;
    std::uint32_t rt = 1;
};

inline constexpr std::uint32_t kD3d12GeomMaxRt = 64;

// The geometry for a matmul of nchunk chunks on a `threads`-wide blob: row sets
// split the threads/16 chunk lanes, so rsl is a power of two <= threads/16 (the
// shader clamps to the same bound; an XLLAMA_D3D12_SHADER_DIR blob of an IQ3_S
// slot keeps this contract), rt is clamped to [1, kD3d12GeomMaxRt].
D3d12Geom d3d12_geom_for(const D3d12GeomReq& req, std::uint32_t nchunk, std::uint32_t threads);
inline std::uint32_t d3d12_geom_pack(const D3d12Geom& g) {
    return (g.rsl & 0xffu) | ((g.rt & 0xffu) << 8);
}
inline std::uint32_t d3d12_geom_rows(const D3d12Geom& g) {
    return static_cast<std::uint32_t>(kD3d12MmvRows) * g.rsl * g.rt;
}
// The weight types whose mmv and mmid kernels read that geometry: IQ3_S and
// IQ2_S (both walk ggml_d3d12_geom.hlsli); the others take pad0 = 0.
inline bool d3d12_geom_type(ggml_type t) {
    return t == GGML_TYPE_IQ3_S || t == GGML_TYPE_IQ2_S;
}
// "auto", "0" or a count; anything else (or null) leaves `fallback`.
std::uint32_t d3d12_geom_parse(const char* s, std::uint32_t fallback);

// Activations go to the GPU the way the CPU backend feeds its own vec_dot:
// quantized by ggml's from_float of the weight's vec_dot type (Q4_0 → q8_0,
// Q4_K / Q6_K → q8_K), one row of ggml blocks per column. The kernels then
// sum integers exactly, so GPU and CPU differ only in float summation order
// (#312: f32 activations changed greedy text).
ggml_type d3d12_activation_type(ggml_type weight);
void d3d12_quantize_activations(ggml_type weight, const float* x, std::size_t x_stride, int k,
                                int ncols, std::uint8_t* out);

// Host emulation of the shaders' per-thread lane mapping, unaligned loads and
// algebra, on activations quantized as above: y[c*y_stride + r] ≈ W[r,:] · x_c.
// `w` holds N rows of `w_row_bytes` packed ggml blocks. Tests and the console
// selftest compare against the CPU backend's vec_dot.
void d3d12_mmv_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes, const float* x,
                       std::size_t x_stride, float* y, std::size_t y_stride, int n, int k,
                       int ncols);

// --- GET_ROWS on a weight in D3D12_Weights (#309) ---
// A tied model reuses token_embd as the lm_head; llama.cpp then keeps the
// input copy on the CPU and duplicates the tensor into D3D12_Weights. With
// GET_ROWS on the GPU the embedding can live in D3D12_Weights once.

// Embedding types the GET_ROWS kernel dequantizes (the catalogue's tied
// embeddings are Q6_K: Coder-3B, LFM2.5-1.2B, LFM2.5-350M).
bool d3d12_get_rows_type_supported(ggml_type t);

struct D3d12GetRowsDesc {
    ggml_type src0_type = GGML_TYPE_F32;
    std::int64_t ne00 = 0, ne01 = 0, ne02 = 1, ne03 = 1; // weight [K, rows]
    ggml_type src1_type = GGML_TYPE_I32;
    std::int64_t ne10 = 0, ne11 = 1, ne12 = 1; // row ids
    ggml_type dst_type = GGML_TYPE_F32;
    bool src0_contiguous = true;
    bool src1_contiguous = true; // ids are staged with one memcpy
    bool src0_in_weight_buffer = false;
};

bool d3d12_get_rows_supported(const D3d12GetRowsDesc& d);

// Host emulation of shaders/ggml_d3d12_get_rows_q6_k.hlsl: y[i*y_stride + j] =
// dequantized W[ids[i], j]. Tests compare it with ggml's to_float bit for bit.
void d3d12_get_rows_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes,
                            const std::int32_t* ids, int n_ids, int k, float* y,
                            std::size_t y_stride);

// Whether llama_gpu.h places a model's token_embd in D3D12_Weights: only a
// tied embedding (no output.weight) of a type GET_ROWS supports, so the one
// copy serves both the input lookup and the lm_head.
bool d3d12_place_tied_embedding(ggml_type embd_type, bool has_output_weight);

// --- f32 ops between the matmuls (shaders/ggml_d3d12_elt.hlsl) ---
// A decoder layer's residual adds, RMS norms with their weight and SwiGLU run
// on the GPU next to the matmuls around them, and matmul inputs are quantized
// to q8_K there too, so a layer's output projection, FFN and the next layer's
// input projections are one submission instead of four.

// Narrower rows stay on the CPU: per-head q/k norms (head_dim 128-256) and the
// small linear-attention ops sit between CPU ops, and moving them would only
// add CPU <-> GPU switches.
inline constexpr std::int64_t kD3d12EltMinRow = 1024;

// Shape rule of every elementwise op: 2-D, rows of ne0 >= kD3d12EltMinRow
// floats, one thread group per row (ne1 <= 65535).
bool d3d12_elt_rows_supported(std::int64_t ne0, std::int64_t ne1, std::int64_t ne2,
                              std::int64_t ne3);

// The norm weights whose MUL the backend runs, by GGUF name: a layer's
// attn_norm, post_attention_norm and ffn_norm, and output_norm. Their buft
// probe then lands them in D3D12_Host (GPU- and CPU-visible).
bool d3d12_norm_weight_name(const char* name);

// Host emulation of the q8_K kernel (KERNEL 3): k floats (k % 256 == 0) to
// k / 256 block_q8_K, with the shader's per-thread absmax and tree reduction.
// Tests compare it with ggml's from_float (quantize_row_q8_K_ref) bit for bit.
void d3d12_quant_q8k_emulate(const float* x, int k, std::uint8_t* out);

// RN(n / |b|) for an integer 1 <= n < 2^24 and a normal float b, by integer
// long division — the kernel's stand-in for fdiv (2.5 ULP on D3D), so its
// scales equal the CPU's -127/max and 1/iscale bit for bit.
float d3d12_div_rn(std::uint32_t n, float b);

// The D3D12_Weights buffer type, for llama's tensor_buft_overrides. Null
// until ggml_d3d12_register() succeeds (always on non-Windows).
ggml_backend_buffer_type_t ggml_d3d12_weights_buft();

// Register the backend with ggml (idempotent). False when D3D12 is unavailable
// (always on non-Windows). The device appears as a GPU device named "D3D12".
bool ggml_d3d12_register();

// --- Console selftest (d3d12be.flag) ---

struct D3d12SelftestRow {
    std::string type; // q4_0 | q4_k | q6_k
    int n = 0, k = 0, ncols = 0;
    double rel_err = 0.0; // max |gpu - ref| / max |ref|
    double gpu_ms = 0.0;  // GPU timestamp time, median of 5 timed computes (ms)
    double packed_gbs = 0.0;
    bool ok = false;
    bool d3d12_ran = false;
    std::string error;
};

// Against the CPU backend (q8 activations + ggml vec_dot): only float
// summation order may differ (#312, predeclared gate A).
inline constexpr double kD3d12SelftestRelTol = 1e-5;

const char* d3d12_selftest_csv_header();
std::string format_d3d12_selftest_row(const D3d12SelftestRow& r, const char* host_label);

// Windows: run every type x shape through the backend and append rows.
// Non-Windows: one row with d3d12_ran=false.
void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out);

// Largest single D3D12_Weights buffer llama.cpp may ask for (0 = no limit).
// Above it ggml splits a context's weights into several buffers; a tensor
// larger than the limit still gets one buffer of its own.
void ggml_d3d12_set_weights_max_buffer(std::size_t bytes);

// llama's decode CPU threadpool (n_threads), for GPU routing's host side
// (XLLAMA_D3D12_GPUROUTE): a miss's nodes run on the same, still spinning
// workers as the CPU splits. Null before that pool is freed.
void ggml_d3d12_set_host_threadpool(ggml_threadpool_t tp, int n_threads);

// Keep output.weight in a CPU buffer when layers go to D3D12 (UWP knob
// LocalState\gguf_output_cpu.txt). Read by apply_gguf_gpu_layers.
void gguf_set_output_on_cpu(bool on);
bool gguf_output_on_cpu();

// GPU memory probe (gpumem.flag): single-allocation ceilings per heap kind,
// cumulative ceilings with and without the heap filled first, and a reserved
// (tiled) buffer backed by 64 MiB heaps. One line per result through emit.
void run_d3d12_memprobe(const std::function<void(const std::string&)>& emit);

} // namespace xllama
