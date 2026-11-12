// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The one place that turns an n_gpu_layers request into llama.cpp params for
// the d3d12 backend (docs/gguf-gpu-decode.md, D2b). Used by run_inference_llama
// (CLI, headless bench) and the persistent LlamaSession (GUI, LAN API).
#pragma once

#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "xllama/ggml_d3d12.h"
#include "xllama/platform.h"
#include "xllama/speculative.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace xllama {

// llama/ggml log lines worth keeping in xllama.log on every GGUF load, CPU or
// d3d12: placement and buffer sizes, graph splits, the expert cache's hit/read
// stats (patch 0004, "expert cache (running|final)"), the scheduler's per-split
// profile (patch 0011, GGML_SCHED_PROFILE: "sched profile" / "sched split"), and
// every warning or error.
// Before this was installed for all loads, a CPU-only load's warnings went to
// llama's default stderr logger, which UWP never shows (found in a community
// fork, SamirCharleston/xllama). Runs on llama.cpp's threads through a C
// callback, so nothing may escape it.
inline void llama_bridge_log(ggml_log_level level, const char* text, void*) noexcept {
    if (!text)
        return;
    const bool keep = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR ||
                      std::strstr(text, "buffer size") || std::strstr(text, "offload") ||
                      std::strstr(text, "graph splits") || std::strstr(text, "splits = ") ||
                      std::strstr(text, "D3D12") || std::strstr(text, "expert cache (") ||
                      std::strstr(text, "sched profile") || std::strstr(text, "sched split");
    if (!keep)
        return;
    try {
        log_output(std::string("[llama] ") + text);
    } catch (...) {
    }
}

// llama_log_set writes process-global llama and ggml state without a lock, so
// it is set exactly once, before the first model load on any path (session,
// run_inference, device train), and never swapped afterwards.
inline void install_llama_log() {
    static std::once_flag once;
    std::call_once(once, [] { llama_log_set(llama_bridge_log, nullptr); });
}

// Working set at a load stage, for every GGUF load: the CPU and d3d12 runs log
// the same stages, so a peak difference can be attributed (#309).
inline void log_gguf_ws(const char* stage) {
    log_output(std::string("[xllama] ws ") + stage + ": " + std::to_string(working_set_mb()) +
               " MB\n");
}

// token_embd's type and whether the model has its own output.weight, read from
// the GGUF header only (no tensor data). False when the header cannot be read.
inline bool gguf_embedding_info(const std::string& path, ggml_type* embd_type, bool* has_output) {
    gguf_init_params gp = {};
    gp.no_alloc = true;
    gp.ctx = nullptr;
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g)
        return false;
    const int64_t embd = gguf_find_tensor(g, "token_embd.weight");
    const bool ok = embd >= 0;
    if (ok) {
        *embd_type = gguf_get_tensor_type(g, embd);
        *has_output = gguf_find_tensor(g, "output.weight") >= 0;
    }
    gguf_free(g);
    return ok;
}

// Where a GGUF's weights live. Default: read into the heap — on UWP this is
// explicit, because patch 0002 makes llama_mmap SUPPORTED in the AppContainer
// and the pin's AUTO would otherwise map every model. stream: map the file
// (CreateFileMappingFromApp there) and keep every CPU weight in the mapping, no
// repack copy, so pages are read from storage on first touch and the memory
// manager can drop them again: a MoE larger than the title's RAM budget loads
// and each token reads only the experts it routes to.
// expert_slots > 0 (llama.cpp patch 0004, wins over stream): the weights are
// read into the heap except the MoE experts, of which only expert_slots per
// layer stay resident; the others are read from the file when routed to. The
// heap is fixed-size, unlike a mapping, which the AppContainer caps per process.
// expert_swap > 0: cache-aware routing (llama_model_params::expert_swap_ratio).
// expert_prefetch / expert_major / expert_prefetch_ahead: the cache reads them
// from the environment when the model loads (LLAMA_EXPERT_PREFETCH,
// LLAMA_EXPERT_MAJOR, LLAMA_EXPERT_PREFETCH_AHEAD); set every load so a reload
// with other values takes them.
inline void put_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value); // "" removes it
#else
    if (*value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

// The catalogue "env" of the current load (SessionParams::env): a name it lists
// wins over the value set_env_int would put, and the next load unsets it.
struct EnvOverrides {
    std::mutex mtx;
    std::vector<std::pair<std::string, std::string>> vars;
};
inline EnvOverrides& env_overrides() {
    static EnvOverrides o;
    return o;
}

inline void apply_env_overrides(const std::vector<std::pair<std::string, std::string>>& env) {
    auto& o = env_overrides();
    std::lock_guard<std::mutex> lock(o.mtx);
    for (const auto& kv : o.vars)
        put_env(kv.first.c_str(), "");
    o.vars = env;
    for (const auto& kv : env) {
        put_env(kv.first.c_str(), kv.second.c_str());
        log_output("[xllama] env " + kv.first + "=" + kv.second + "\n");
    }
}

inline void set_env_int(const char* name, int v) {
    auto& o = env_overrides();
    std::lock_guard<std::mutex> lock(o.mtx);
    for (const auto& kv : o.vars)
        if (kv.first == name)
            return;
    put_env(name, std::to_string(v).c_str());
}

inline void set_env_str(const char* name, const char* v) {
    auto& o = env_overrides();
    std::lock_guard<std::mutex> lock(o.mtx);
    for (const auto& kv : o.vars)
        if (kv.first == name)
            return;
    put_env(name, v);
}

inline void apply_gguf_residency(bool stream, llama_model_params& mparams, int expert_slots = 0,
                                 float expert_swap = 0.0f, int expert_prefetch = 0,
                                 int expert_major = 0, bool expert_prefetch_ahead = false) {
    // patch 0016: the tensors named here are read through a mapping of their span only; every load sets it again
    set_env_str("LLAMA_LAZY_TENSORS", "");
    if (expert_slots > 0) {
#ifdef LLAMA_HAS_EXPERT_SLOTS
        // XLLAMA_EXPERT_MMAP=1 (a manifest env.* entry): token_embd and output.weight become lazy tensors (on the
        // CPU), file-backed through one view of the span that holds them, instead of private heap copies. The
        // Qwen3.5-122B-A10B IQ3_S load reached 4810 MB with those 909 MiB copied and died allocating its 245 MiB of
        // KV + recurrent state. A whole-file view (LLAMA_LOAD_MODE_MMAP) of its 52.9 GB is refused by the title's
        // mapped-view ceiling (MapViewOfFileFromApp: not enough memory resources, measured 2026-11-12), so the load
        // stays NONE and only the lazy span is mapped: 920 MiB at the start of that file, under the 1.0 GiB that maps.
        const char* em = std::getenv("XLLAMA_EXPERT_MMAP");
        const bool mapped = em && em[0] == '1';
        mparams.load_mode = LLAMA_LOAD_MODE_NONE;
        if (mapped)
            set_env_str("LLAMA_LAZY_TENSORS", "token_embd.weight,output.weight");
        mparams.n_expert_slots = static_cast<uint32_t>(expert_slots);
        mparams.expert_swap_ratio = expert_swap;
        set_env_int("LLAMA_EXPERT_PREFETCH", expert_prefetch);
        set_env_int("LLAMA_EXPERT_MAJOR", expert_major);
        set_env_int("LLAMA_EXPERT_PREFETCH_AHEAD", expert_prefetch_ahead ? 1 : 0);
        log_output(std::string("[xllama] gguf weights: ") + (mapped ? "heap, token_embd + output lazy (span mapped)" : "heap") +
                   ", expert cache of " + std::to_string(expert_slots) +
                   " experts per layer, swap ratio " + std::to_string(expert_swap) + ", prefetch " +
                   std::to_string(expert_prefetch) + (expert_prefetch_ahead ? " ahead" : "") +
                   ", expert-major " + std::to_string(expert_major) + "\n");
        return;
#else
        (void)expert_swap;
        (void)expert_prefetch;
        (void)expert_major;
        (void)expert_prefetch_ahead;
        log_output("[xllama] gguf expert cache: this llama.cpp lacks patch 0004, ignored\n");
#endif
    }
    if (stream) {
        mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
        mparams.use_extra_bufts = false;
        log_output("[xllama] gguf weights: streamed (mmap, no repack)\n");
        return;
    }
#ifdef XLLAMA_UWP
    mparams.load_mode = LLAMA_LOAD_MODE_NONE;
#endif
}

// XLLAMA_EXPERT_USED=<k> (a manifest env.* entry): route each token to k experts instead of the
// GGUF's <arch>.expert_used_count, through a llama.cpp kv override (the router keeps its top k and
// renormalises their weights over them). With the expert cache a decode reads ~k/K of the expert
// bytes it would; quality is the price, so it is measured per model, never a default. Ignored
// unless 0 < k < the file's count; every load sets mparams.kv_overrides again.
inline void apply_expert_used_override(llama_model_params& mparams, const std::string& path) {
    static llama_model_kv_override ovr[2];
    mparams.kv_overrides = nullptr;
    const char* s = std::getenv("XLLAMA_EXPERT_USED");
    const int k = s && *s ? std::atoi(s) : 0;
    if (k <= 0)
        return;
    gguf_init_params gp = {};
    gp.no_alloc = true;
    gp.ctx = nullptr;
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g)
        return;
    std::string key;
    int64_t have = -1;
    const int64_t ak = gguf_find_key(g, "general.architecture");
    if (ak >= 0 && gguf_get_kv_type(g, ak) == GGUF_TYPE_STRING) {
        key = std::string(gguf_get_val_str(g, ak)) + ".expert_used_count";
        const int64_t uk = gguf_find_key(g, key.c_str());
        if (uk >= 0 && gguf_get_kv_type(g, uk) == GGUF_TYPE_UINT32)
            have = gguf_get_val_u32(g, uk);
        else if (uk >= 0 && gguf_get_kv_type(g, uk) == GGUF_TYPE_INT32)
            have = gguf_get_val_i32(g, uk);
    }
    gguf_free(g);
    if (have <= 0 || k >= have || key.size() >= sizeof(ovr[0].key)) {
        log_output("[xllama] expert used " + std::to_string(k) + " ignored (" + (key.empty() ? "no arch" : key) +
                   " = " + std::to_string(have) + ")\n");
        return;
    }
    ovr[0] = {};
    ovr[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    std::memcpy(ovr[0].key, key.c_str(), key.size() + 1);
    ovr[0].val_i64 = k;
    ovr[1] = {}; // key[0] == 0 ends the list
    mparams.kv_overrides = ovr;
    log_output("[xllama] expert used " + std::to_string(k) + " of " + std::to_string(have) + " (" + key + ")\n");
}

// Sets mparams.devices / n_gpu_layers and returns the layers actually
// offloaded: 0 when none were asked for or the device is unavailable, in
// which case the model loads exactly as before (CPU, no GPU device listed).
inline int apply_gguf_gpu_layers(int requested, llama_model_params& mparams,
                                 const std::string& model_path) {
    static ggml_backend_dev_t no_devices[] = {nullptr};
    mparams.n_gpu_layers = 0;
    mparams.devices = no_devices;
    if (requested <= 0)
        return 0;
    log_gguf_ws("before d3d12");
    if (!ggml_d3d12_register()) {
        log_output("[xllama] gguf gpu layers: d3d12 backend unavailable, using the CPU\n");
        return 0;
    }
    static ggml_backend_dev_t d3d12[] = {ggml_backend_dev_by_name("D3D12"), nullptr};
    if (!d3d12[0]) {
        log_output("[xllama] gguf gpu layers: no D3D12 device registered, using the CPU\n");
        return 0;
    }
    mparams.devices = d3d12;
    mparams.n_gpu_layers = requested;
    // D3D12_Host is also the device's host buft, so llama.cpp puts the CPU
    // backend's compute buffer there and splits share activations in place.
    // Keep it out of the CPU *weight* list: weights left on the CPU stay in the
    // CPU / repack bufts exactly as on a CPU-only load.
    mparams.no_host = true;
    // A tied token_embd also serves as the lm_head: llama.cpp would keep the
    // input copy on the CPU and duplicate it into D3D12_Weights (244 MiB on
    // Coder-3B). Placing it in D3D12_Weights, where GET_ROWS also runs, leaves
    // one copy: the duplicate finds the original in the same buffer context.
    ggml_type embd_type = GGML_TYPE_COUNT;
    bool has_output = true;
    static llama_model_tensor_buft_override overrides[4] = {};
    int n_overrides = 0;
    const bool have_info = gguf_embedding_info(model_path, &embd_type, &has_output);
    if (have_info && d3d12_place_tied_embedding(embd_type, has_output)) {
        overrides[n_overrides++] = {"^token_embd\\.weight$", ggml_d3d12_weights_buft()};
        log_output("[xllama] gguf gpu layers: tied token_embd in D3D12_Weights\n");
    }
    // n_gpu_layers >= 1 always offloads output.weight first, and on a large
    // vocabulary it is the single biggest tensor (Qwen3.5: 248K x 4096 Q6_K =
    // 796 MiB, over what one Series X allocation takes). Knob: keep it on the CPU.
    if (have_info && has_output && gguf_output_on_cpu()) {
        overrides[n_overrides++] = {"^output\\.weight$", ggml_backend_cpu_buffer_type()};
        log_output("[xllama] gguf gpu layers: output.weight kept on the CPU\n");
    }
    // The small f32 weights of the linear-attention layers (qwen35 / qwen35moe:
    // conv kernel, A, dt bias, gated-norm weight; ~4 MiB on the 35B) go to
    // D3D12_Host, which both the delta-net kernels and the CPU read: a layer's
    // whole chain can then run in one D3D12 split, and the CPU still runs any op
    // the device declines. An override skips llama.cpp's supports_op probe,
    // which these ops' other operands (activations) would fail at load time.
    overrides[n_overrides++] = {"^blk\\.[0-9]+\\.ssm_(a|dt\\.bias|norm\\.weight|conv1d\\.weight)$",
                                ggml_backend_dev_buffer_type(d3d12[0])};
    overrides[n_overrides] = {nullptr, nullptr};
    if (n_overrides > 0)
        mparams.tensor_buft_overrides = overrides;
    log_gguf_ws("after d3d12 init");
    log_output("[xllama] gguf gpu layers: " + std::to_string(requested) + " on D3D12\n");
    return requested;
}

// Most logits rows a context will request per ubatch, for n_outputs_max: one
// for prefill and decode, 1 + draft for a prompt-lookup verify batch
// (decode_loop.h), unlimited (0 = llama.cpp's default) for embeddings, which
// output every token. llama.cpp asserts in output_reserve when a batch asks
// for more, so this must cover every batch the context will see.
inline uint32_t gguf_gpu_outputs_max(bool embeddings, bool prompt_lookup) {
    if (embeddings)
        return 0;
    return 1u + (prompt_lookup ? static_cast<uint32_t>(kSpecDraftKDefault) : 0u);
}

// KV cache and attention stay on the CPU in ordinary memory: the backend only
// runs weight matmuls. Flash Attention AUTO would resolve to disabled here:
// llama.cpp compares each layer's device (D3D12) with the FA node's (the CPU,
// beside the KV cache), and any mismatch turns FA off. The CPU supports FA and
// the CPU-only path resolves AUTO to enabled, so the GPU path asks for it
// explicitly and keeps the same attention (#310).
// n_outputs_max also sizes the compute reserve: llama.cpp
// reserves the prefill graph for that many logits rows (n_ubatch by default:
// 512 x 151936 x 4 B = 297 MiB on Coder-3B), and on the GPU path the reserve
// is a committed D3D12_Host buffer, where the CPU's malloc'd one only counts
// the pages it touches (#309).
inline void apply_gguf_gpu_context(int applied_layers, llama_context_params& cparams,
                                   uint32_t outputs_max) {
    // The recurrent states of linear-attention layers (qwen35 / qwen35moe) go
    // to the device buffer, CPU-visible D3D12_Host, which the delta-net kernels
    // read and write in place (llama.cpp patch 0010; the CPU uses it as well
    // where they cannot run). Set on every context so a CPU reload clears it.
    set_env_int("LLAMA_RS_ON_DEVICE", applied_layers > 0 ? 1 : 0);
    if (applied_layers > 0) {
        cparams.offload_kqv = false;
        if (cparams.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO)
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (outputs_max > 0)
            cparams.n_outputs_max = outputs_max;
    }
}

// Persistent CPU threadpools for a context that alternates CPU and d3d12
// splits. Without one attached, ggml-cpu builds and joins a disposable pool on
// every graph_compute (ggml-cpu.c) — once per token on a CPU-only graph, but
// ~50-70 times per token once matmuls run on the GPU: lfm25-350m decoded at
// 26.7 tok/s with 6 threads and 66.9 with 1 (D2b smoke, CI 1.6.0.1130). The
// CPU-only path keeps llama's default so its baseline does not move. Must
// outlive every context it is attached to (declare it before the context).
class GgufCpuThreadpools {
  public:
    GgufCpuThreadpools() = default;
    GgufCpuThreadpools(const GgufCpuThreadpools&) = delete;
    GgufCpuThreadpools& operator=(const GgufCpuThreadpools&) = delete;
    ~GgufCpuThreadpools() {
        reset();
    }

    void attach(int applied_layers, llama_context* ctx, int n_threads, int n_threads_batch) {
        if (applied_layers <= 0 || !ctx || n_threads <= 0)
            return;
        if (!tp_) {
            ggml_threadpool_params p = ggml_threadpool_params_default(n_threads);
            tune(p);
            tp_ = ggml_threadpool_new(&p);
            if (n_threads_batch > 0 && n_threads_batch != n_threads) {
                ggml_threadpool_params pb = ggml_threadpool_params_default(n_threads_batch);
                tune(pb);
                tp_batch_ = ggml_threadpool_new(&pb);
            }
            log_output("[xllama] cpu threadpool: " + std::to_string(n_threads) + " threads (batch " +
                       std::to_string(tp_batch_ ? n_threads_batch : n_threads) + "), poll " +
                       std::to_string(p.poll) + ", prio " + std::to_string(int(p.prio)) + "\n");
            if (tp_)
                ggml_d3d12_set_host_threadpool(tp_, n_threads);
        }
        if (tp_)
            llama_attach_threadpool(ctx, tp_, tp_batch_ ? tp_batch_ : tp_);
    }

    void reset() {
        if (tp_batch_)
            ggml_threadpool_free(tp_batch_);
        if (tp_) {
            ggml_d3d12_set_host_threadpool(nullptr, 0);
            ggml_threadpool_free(tp_);
        }
        tp_ = tp_batch_ = nullptr;
    }

  private:
    // XLLAMA_CPU_POLL (0..100, ggml's default 50): how long the workers spin
    // before sleeping between graphs; XLLAMA_CPU_PRIO (-1..3, GGML_SCHED_PRIO_*).
    // Both from the catalogue "env".
    // With XLLAMA_D3D12_GPUROUTE=1 a miss's nodes run on this pool too
    // (ggml_d3d12_set_host_threadpool). Only XLLAMA_D3D12_GPUROUTE_SHARED=0
    // puts them on a pool of their own, which the workers that spin after a CPU
    // split starve: 107 ms per miss at poll 50 (console, 11/11), so there the
    // poll defaults to 0.
    static void tune(ggml_threadpool_params& p) {
        const char* r = std::getenv("XLLAMA_D3D12_GPUROUTE");
        const char* sh = std::getenv("XLLAMA_D3D12_GPUROUTE_SHARED");
        if (r && *r && *r != '0' && sh && *sh == '0')
            p.poll = 0;
        if (const char* s = std::getenv("XLLAMA_CPU_POLL"); s && *s)
            p.poll = static_cast<uint32_t>(std::clamp(std::atoi(s), 0, 100));
        if (const char* s = std::getenv("XLLAMA_CPU_PRIO"); s && *s)
            p.prio = static_cast<ggml_sched_priority>(std::clamp(std::atoi(s), -1, 3));
    }

    ggml_threadpool_t tp_ = nullptr;
    ggml_threadpool_t tp_batch_ = nullptr;
};

} // namespace xllama
