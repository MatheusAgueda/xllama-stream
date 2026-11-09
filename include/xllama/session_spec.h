// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// One resolver from catalogue data to session load parameters (ADR 0007, audit
// F03). GUI chat, LAN chat, embeddings and pull all build their session here, so
// a model loads the same way whichever surface asks first, and the hub can tell
// whether the resident session is the one being asked for. WinRT-free: the UWP
// side only copies its wide-string manifest entry into CatalogueLoad.
#pragma once

#include "xllama/routing_policy.h" // resolve_n_ctx, resolve_gguf_gpu_layers
#include "xllama/session.h"

#include <algorithm>
#include <string>

namespace xllama {

// The catalogue fields that decide how a model is loaded.
struct CatalogueLoad {
    bool found = false; // false: not in the catalogue, load with the defaults
    bool gguf = false;  // kind "gguf" (llama.cpp); else ORT GenAI
    int n_ctx = 0;      // raw catalogue value, clamped by resolve_n_ctx
    int gpu_layers = 0; // catalogue default; the operator knob may override it
    std::string lora;   // adapter file relative to the model directory
    float lora_scale = 1.0f;
    bool stream = false;      // "stream": weights stay file-backed (SessionParams::stream_weights)
    int expert_slots = 0;     // "expert_slots": experts kept per MoE layer (SessionParams)
    float expert_swap = 0.0f; // "expert_swap": cache-aware routing ratio (SessionParams)
    int expert_prefetch = 0;  // "expert_prefetch": experts read ahead per token (SessionParams)
    int expert_major = 0;     // "expert_major": largest expert-major prompt ubatch (SessionParams)
    bool expert_prefetch_ahead = false; // "expert_prefetch_ahead": predict a layer early
    int n_ubatch = 0;                   // "n_ubatch": physical prefill chunk (0 = llama default)
    std::vector<std::pair<std::string, std::string>> env; // "env": {NAME: value} (SessionParams)
};

// A catalogue "env" name: LLAMA_*, GGML_* or XLLAMA_*, upper case, digits, '_'.
inline bool catalogue_env_name(const std::string& n) {
    if (n.rfind("LLAMA_", 0) != 0 && n.rfind("GGML_", 0) != 0 && n.rfind("XLLAMA_", 0) != 0)
        return false;
    return std::all_of(n.begin(), n.end(), [](char ch) {
        return (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_';
    });
}

// The usable entries of a catalogue "env", sorted by name (the last of a
// repeated name wins), values without a NUL.
inline std::vector<std::pair<std::string, std::string>>
resolve_env(const std::vector<std::pair<std::string, std::string>>& env) {
    std::vector<std::pair<std::string, std::string>> out;
    for (const auto& kv : env) {
        if (!catalogue_env_name(kv.first) || kv.second.find('\0') != std::string::npos)
            continue;
        auto it = std::find_if(out.begin(), out.end(),
                               [&](const auto& o) { return o.first == kv.first; });
        if (it != out.end())
            it->second = kv.second;
        else
            out.push_back(kv);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// `model_dir` is where the model's files live (resolve_model_path on the bare
// name); the adapter is resolved against it. `gpu_layers_knob` is the operator
// override read by the caller (-1 = none, see resolve_gguf_gpu_layers).
inline SessionParams resolve_session_params(const std::string& model, const CatalogueLoad& c,
                                            const std::string& model_dir, int gpu_layers_knob) {
    SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = kDefaultNCtx;
    if (!c.found)
        return sp;
    sp.n_ctx = resolve_n_ctx(c.n_ctx);
    if (!c.gguf)
        return sp;
    sp.backend = Backend::LlamaCpp;
    sp.stream_weights = c.stream;
    sp.expert_slots = c.expert_slots > 0 ? c.expert_slots : 0;
    sp.expert_swap =
        sp.expert_slots > 0 && c.expert_swap > 0.0f && c.expert_swap <= 1.0f ? c.expert_swap : 0.0f;
    sp.expert_prefetch = sp.expert_slots > 0 ? std::clamp(c.expert_prefetch, 0, 64) : 0;
    sp.expert_major = sp.expert_slots > 0 ? std::clamp(c.expert_major, 0, 65536) : 0;
    sp.expert_prefetch_ahead = sp.expert_prefetch > 0 && c.expert_prefetch_ahead;
    // llama.cpp clamps n_ubatch to n_batch (default 2048), and the expert cache
    // to max(slots / experts used, expert_major).
    sp.n_ubatch = c.n_ubatch > 0 ? std::clamp(c.n_ubatch, 32, 2048) : 0;
    sp.env = resolve_env(c.env);
    // A streamed model is larger than any GPU budget: d3d12 layers would copy
    // its experts into video memory, so the knob does not apply to it. The
    // expert cache keeps its experts in the CPU cache whatever the layer device,
    // so an expert-cached model (which wins over stream) may offload the rest.
    const bool cpu_only = c.stream && sp.expert_slots == 0;
    sp.n_gpu_layers = cpu_only ? 0 : resolve_gguf_gpu_layers(c.gpu_layers, gpu_layers_knob);
    if (!c.lora.empty()) {
        sp.lora_path = model_dir;
        if (!sp.lora_path.empty() && sp.lora_path.back() != '\\' && sp.lora_path.back() != '/')
            sp.lora_path.push_back('\\');
        sp.lora_path += c.lora;
        sp.lora_scale = c.lora_scale;
    }
    return sp;
}

// Two parameter sets load the same thing into RAM. Sampling is per generate and
// threads/batch are tuning knobs, so neither is part of the load identity.
inline bool same_load(const SessionParams& a, const SessionParams& b) {
    return a.model_path == b.model_path && a.backend == b.backend && a.n_ctx == b.n_ctx &&
           a.n_gpu_layers == b.n_gpu_layers && a.lora_path == b.lora_path &&
           a.lora_scale == b.lora_scale && a.kv_q8 == b.kv_q8 &&
           a.stream_weights == b.stream_weights && a.expert_slots == b.expert_slots &&
           a.expert_swap == b.expert_swap && a.expert_prefetch == b.expert_prefetch &&
           a.expert_major == b.expert_major && a.expert_prefetch_ahead == b.expert_prefetch_ahead &&
           a.env == b.env;
}

} // namespace xllama
