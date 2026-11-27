// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include "xllama/inference_progress.h"
#include "xllama/sampling.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

/// The shipping context size.
///
/// Every surface that opens a context (chat UI, LAN API, Session default, CLI
/// default) reads this constant. The trimmer budget `kMaxPromptTokens`
/// (routing_policy.h) is sized against it.
inline constexpr int kDefaultNCtx = 2048;

// ---------------------------------------------------------------------------
// Inference configuration
// ---------------------------------------------------------------------------

/// Configuration for a single inference call.
///
/// Used by the CLI, benchmarks, and the `run_inference()` function. Most fields
/// have defaults so callers only set what they want to change.
struct InferenceParams {
    /// Path to model. Linux: absolute path; UWP: filename in LocalFolder.
    std::string model_path;

    /// Input prompt text.
    std::string prompt;

    /// Max tokens to generate (default: 128).
    int n_predict = 128;

    /// Context size (default: kDefaultNCtx = 2048).
    int n_ctx = kDefaultNCtx;

    /// Override for the engine's max_length.
    ///   0  = derive as min(n_ctx, n_prompt + n_predict) (bench default)
    ///  -1  = saturate to n_ctx (what Session::generate ships)
    ///  >0  = explicit, clamped to (n_prompt+1, n_ctx]
    int max_length_override = 0;

    /// Thread count (0 = auto-detect).
    int n_threads = 0;

    /// Which repetition of a repeated bench this run is (0 = not a bench).
    /// Written verbatim into the bench CSV's run_index column.
    int run_index = 0;

    /// llama.cpp prefill batching (GGUF path only; 0 = default).
    int n_batch = 0;
    int n_ubatch = 0;

    /// Quantize KV cache to q8_0 (GGUF path only).
    bool kv_q8 = false;

    /// GGUF layers whose weight matmuls run on the d3d12 backend (0 = CPU,
    /// docs/gguf-gpu-decode.md). Ignored where D3D12 is unavailable.
    int n_gpu_layers = 0;

    /// Bench only: decode exactly n_predict tokens, through end-of-generation
    /// (stop sequences should be cleared too). Default off.
    bool ignore_eog = false;

    /// Sampling defaults from `xllama/sampling.h`.
    float temperature = sampling_defaults::kTemperature;
    float top_p = sampling_defaults::kTopP;
    int top_k = sampling_defaults::kTopK;
    float repetition_penalty = sampling_defaults::kRepetitionPenalty;
    uint32_t seed = sampling_defaults::kSeed;

    /// Deterministic argmax decode (prerequisite for cross-backend logit parity).
    bool greedy = false;

    /// Build a `SamplingConfig` from the sampling fields.
    SamplingConfig sampling() const {
        return SamplingConfig{temperature, top_p, top_k, repetition_penalty, seed, greedy};
    }

    /// System message for chat template (empty = built-in default).
    std::string system_prompt;

    /// Dump last prefill-token logit vector to this path (+ `.json` sidecar).
    std::string dump_logits_path;

    /// Stop strings; generation ends when output ends with any of these.
    std::vector<std::string> stop_sequences;

    /// Wrap `prompt` with the model's chat template before inference.
    bool chat_template = false;

    /// GGUF LoRA adapter path (llama.cpp path only; empty = off).
    std::string lora_path;
    float lora_scale = 1.0f;

    /// Draft-free prompt-lookup speculative decoding (GGUF path only; default OFF).
    bool prompt_lookup = false;

    /// Probe CPU memory bandwidth and exit (no model load).
    bool run_membw = false;

    /// Probe disk (NVMe) read bandwidth and exit (no model load).
    bool run_diskbw = false;

    /// Probe GPU STREAM bandwidth and exit (D3D12; non-Windows reports unavailable).
    bool run_gpubw = false;

    /// Probe Q4_K GEMV density and exit (D3D12 CS; not a backend).
    bool run_gpugemv = false;

    /// GGUF GPU decode probe D1: print the cost-model projection and exit.
    bool run_gpustep = false;

    /// Evaluate this gpustep CSV against the D1 gates and exit (empty = off).
    std::string gpustep_verdict_csv;

    /// Probe heap ceiling and exit (no model load).
    bool run_ramceil = false;

    /// Validate a TrainingJob JSON and exit (no model).
    bool run_validate_train_job = false;

    /// Shell out to the host training runner (from a TrainingJob JSON).
    bool run_train_job = false;

    /// Print training capability matrix and exit.
    bool run_training_capabilities = false;

    /// Run embedding smoke test and exit (load model, embed inputs, report metrics).
    bool run_embed = false;

    /// Embedding input strings for --embed mode.
    std::vector<std::string> embed_inputs;

    /// Requested embedding dimensions for --embed mode (0 = native).
    int embed_dimensions = 0;

    /// Path to TrainingJob JSON (for --train-job / --validate-train-job).
    std::string train_job_path;

    /// UI callback: status changes (e.g. "loading model"). Called from inference thread.
    std::function<void(const std::string&)> on_status;

    /// UI callback: per-token text piece. Called from inference thread; copy before return.
    std::function<void(std::string_view)> on_token;

    /// Stream generated pieces to stdout (interactive CLI; off by default).
    bool echo_stdout = false;

    /// Atomic flag set from UI thread to request early termination.
    std::atomic<bool>* abort_flag = nullptr;

    /// Decode progress log cadence (tokens). 0 = disable token cadence.
    int decode_log_every_n = kDefaultDecodeLogEveryN;
    /// Decode progress log cadence (milliseconds). 0 = disable time cadence.
    int decode_log_every_ms = kDefaultDecodeLogEveryMs;
    /// Stall watchdog (seconds). 0 = disable. See inference_progress.h.
    int stall_timeout_s = kDefaultInferStallTimeoutS;

    InferenceProgressConfig progress() const {
        return InferenceProgressConfig{decode_log_every_n, decode_log_every_ms, stall_timeout_s};
    }
};

/// Resolve the max_length for inference.
///
/// @param n_ctx        Session context size.
/// @param n_prompt     Number of prompt tokens.
/// @param n_predict    Max tokens to generate.
/// @param override_v   Override value (<0=saturate to n_ctx, 0=derive, >0=explicit).
/// @return             The effective max_length.
inline int resolve_max_length(int n_ctx, int n_prompt, int n_predict, int override_v) {
    if (override_v < 0)
        return n_ctx;
    if (override_v > 0)
        return std::clamp(override_v, n_prompt + 1, n_ctx);
    // 64-bit: an INT_MAX n_predict (CLI, bench) must saturate to n_ctx (F01).
    return static_cast<int>(std::min<int64_t>(n_ctx, int64_t{n_prompt} + n_predict));
}

/// One reply token as scored by the decode loop (GenerateParams::logprobs).
struct TokenLogprob {
    int32_t id = 0;
    /// log p(id) under the raw logits (temperature 1, no sampler stage).
    float logprob = 0.0f;
    /// The id was the logits' argmax (greedy would have picked it).
    bool argmax = false;
    /// The token's text, special tokens rendered.
    std::string piece;
};

/// The expert cache's counters over one request (llama.cpp patch 0027): the difference of two
/// snapshots of its cumulative counters, taken as the request starts and ends. `present` is false
/// when the model has no expert cache; the reply then has no "expert" object.
struct ExpertStats {
    bool present = false;
    uint64_t hits = 0;     ///< Experts found in their slots.
    uint64_t misses = 0;   ///< Experts read from the file.
    uint64_t swaps = 0;    ///< Cache-aware routing: residents that took a missing expert's place.
    uint64_t em_swaps = 0; ///< The same in expert-major (prompt) ubatches.
    uint64_t bytes = 0;    ///< Bytes read by the plan and load ops.
    uint64_t read_us = 0;  ///< Wall time of the load ops.
    uint64_t em_reads = 0; ///< Expert tensors read by expert-major ubatches.
    uint64_t em_bytes = 0; ///< Their bytes.
    /// FNV-1a 64 over the FINAL expert ids of every (token, layer) of the request (patch 0028);
    /// not a difference: the bridge restarts it at the request's start. 0 when the patch is absent.
    uint64_t route_hash = 0;
    uint64_t route_n = 0; ///< The (token, layer) pairs it covers.
    bool has_route = false;

    /// This snapshot minus `before` (a counter that went down reads 0).
    ExpertStats since(const ExpertStats& before) const {
        const auto d = [](uint64_t now, uint64_t then) { return now >= then ? now - then : 0; };
        ExpertStats r;
        r.present = present && before.present;
        r.hits = d(hits, before.hits);
        r.misses = d(misses, before.misses);
        r.swaps = d(swaps, before.swaps);
        r.em_swaps = d(em_swaps, before.em_swaps);
        r.bytes = d(bytes, before.bytes);
        r.read_us = d(read_us, before.read_us);
        r.em_reads = d(em_reads, before.em_reads);
        r.em_bytes = d(em_bytes, before.em_bytes);
        r.route_hash = route_hash; // a running chain, restarted per request: copied, not subtracted
        r.route_n = route_n;
        r.has_route = has_route;
        return r;
    }
};

/// B3 (Session::generate_batch): the shared prefill one result took part in.
/// k == 0 = not batched (every other path), and nothing else is meaningful.
struct BatchPrefillInfo {
    int k = 0;                // requests in the batch
    int index = 0;            // this result's position in it
    int n_tokens = 0;         // tokens the shared prefill decoded: k x (shortest prompt - 1)
    int n_shared_decodes = 0; // llama_decode calls of the shared prefill
    double ms = 0.0;          // wall time of the shared prefill
    ExpertStats expert;       // expert-cache counters of the shared prefill
    int own_n = 0;            // this request's own pass: its prompt tokens after the shared part
    int own_decodes = 0;      // llama_decode calls of that own pass
};

/// Result of an inference call.
struct InferenceResult {
    /// Whether generation succeeded.
    bool success = false;

    /// Model load time (ms).
    double t_load_ms = 0.0;

    /// Prefill time (ms).
    double t_p_eval_ms = 0.0;

    /// Decode time (ms).
    double t_eval_ms = 0.0;

    /// Time from the start of prefill until the first generated token is ready.
    /// This is the interactive time-to-first-token (TTFT), in milliseconds.
    double t_first_token_ms = 0.0;

    /// Number of prefill tokens evaluated. Smaller than n_prompt when a session
    /// reuses a resident KV prefix (#170a); use it for prefill throughput.
    int n_p_eval = 0;

    /// Length of the prompt passed to this call, in tokens, whatever was reused.
    /// This is what an API reports as usage.prompt_tokens (#335).
    int n_prompt = 0;

    /// Prompt tokens restored from the session's on-disk prefix cache
    /// (XLLAMA_PREFIX_CACHE_MIB) instead of prefilled; 0 when unused.
    int n_prefix_cache_tokens = 0;

    /// Number of decode tokens generated.
    int n_eval = 0;

    /// True if stopped on a stop sequence; false if capped by n_predict/EOS.
    bool ended_with_stop = false;

    /// Max length actually requested of the engine.
    /// On DirectML this controls prefill throughput. 0 = N/A (GGUF).
    int max_length = 0;

    /// Peak working set size (MB).
    size_t peak_ws_mb = 0;

    /// Per-process GPU memory usage after model load (0 = N/A).
    size_t gpu_mem_mb = 0;

    /// OS-granted GPU budget for this process (0 = N/A).
    size_t gpu_budget_mb = 0;

    /// GGUF layers actually placed on the d3d12 backend (0 = CPU run).
    int gpu_layers = 0;

    /// Speculative drafts generated (0 when prompt_lookup is off).
    int n_drafted = 0;

    /// Speculative drafts accepted.
    int n_spec_accepted = 0;

    /// Thinking budget armed (GenerateParams::think_budget): generated tokens up
    /// to and including "</think>" (all of them when it never closed), and
    /// whether the budget wrote the close. 0 / false when no budget was armed.
    int n_think = 0;
    bool think_forced = false;

    /// Generated text output.
    std::string output_text;

    /// Error description on failure.
    std::string error_msg;

    /// True when the stall watchdog ended the run (see inference_progress.h).
    bool stalled = false;

    /// Expert-cache counters of this call (GGUF MoE with an expert cache only).
    ExpertStats expert;

    /// Per reply token, when the call asked for them (GenerateParams::logprobs
    /// or forced_tokens). Empty otherwise.
    std::vector<TokenLogprob> logprobs;

    /// Set by Session::generate_batch only. t_p_eval_ms then = batch.ms + this
    /// request's own pass (its prompt after the shared part); expert = this request's own counters.
    BatchPrefillInfo batch;
};

} // namespace xllama
