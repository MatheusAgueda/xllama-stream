// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/session_hub.h"
#include "xllama/session_spec.h"

#include <cstdlib>
#include <string>

using xllama::CatalogueLoad;
using xllama::resolve_session_params;
using xllama::same_load;

TEST_CASE("resolve_session_params: one resolver for every surface (ADR 0007)") {
    // Not in the catalogue: defaults, backend left to the loader.
    const auto unknown = resolve_session_params("custom", CatalogueLoad{}, "D:\\m\\custom", -1);
    CHECK(unknown.model_path == "custom");
    CHECK(unknown.n_ctx == xllama::kDefaultNCtx);
    CHECK(unknown.backend == xllama::Backend::Auto);
    CHECK(unknown.lora_path.empty());

    // An ORT entry: catalogue context, no llama.cpp fields.
    CatalogueLoad ort;
    ort.found = true;
    ort.n_ctx = 3072;
    ort.lora = "ignored.gguf";
    const auto o = resolve_session_params("smollm2", ort, "D:\\m\\smollm2", 4);
    CHECK(o.n_ctx == 3072);
    CHECK(o.n_gpu_layers == 0);
    CHECK(o.lora_path.empty());

    // A GGUF entry with an adapter: the GUI and the API used to disagree here (F03).
    CatalogueLoad g;
    g.found = true;
    g.gguf = true;
    g.n_ctx = 1; // clamped like every session open
    g.gpu_layers = 32;
    g.lora = "adapter.gguf";
    g.lora_scale = 0.5f;
    const auto a = resolve_session_params("smollm2-lora-rt", g, "D:\\m\\smollm2-lora-rt", -1);
    CHECK(a.backend == xllama::Backend::LlamaCpp);
    CHECK(a.n_ctx == xllama::kMinSessionNCtx);
    CHECK(a.n_gpu_layers == 32);
    CHECK(a.lora_path == "D:\\m\\smollm2-lora-rt\\adapter.gguf");
    CHECK(a.lora_scale == 0.5f);
    // A trailing separator is not doubled; the operator knob wins, 0 included.
    CHECK(resolve_session_params("x", g, "D:\\m\\x\\", 0).lora_path == "D:\\m\\x\\adapter.gguf");
    CHECK(resolve_session_params("x", g, "D:\\m\\x\\", 0).n_gpu_layers == 0);

    // A streamed GGUF stays on the CPU whatever the catalogue or the knob asks:
    // d3d12 layers would copy its experts into video memory.
    CatalogueLoad s = g;
    s.stream = true;
    const auto st = resolve_session_params("moe", s, "D:\\m\\moe", 16);
    CHECK(st.stream_weights);
    CHECK(st.n_gpu_layers == 0);
    CHECK_FALSE(a.stream_weights);
    // An ORT entry never streams (the field is GGUF-only).
    CatalogueLoad os = ort;
    os.stream = true;
    CHECK_FALSE(resolve_session_params("smollm2", os, "D:\\m\\smollm2", -1).stream_weights);

    // An expert-cached GGUF keeps its experts in the CPU cache, so the rest of
    // its layers may go to d3d12, even with "stream" set (the cache wins); a
    // negative count is off.
    CatalogueLoad ec = g;
    ec.expert_slots = 32;
    ec.stream = true;
    const auto ect = resolve_session_params("moe", ec, "D:\\m\\moe", 16);
    CHECK(ect.expert_slots == 32);
    CHECK(ect.n_gpu_layers == 16);
    CHECK(resolve_session_params("moe", ec, "D:\\m\\moe", -1).n_gpu_layers == 32);
    CHECK(a.expert_slots == 0);
    ec.expert_slots = -4;
    CHECK(resolve_session_params("moe", ec, "D:\\m\\moe", -1).expert_slots == 0);
    CHECK(resolve_session_params("moe", ec, "D:\\m\\moe", 16).n_gpu_layers == 0); // stream again
    os.expert_slots = 32;
    CHECK(resolve_session_params("smollm2", os, "D:\\m\\smollm2", -1).expert_slots == 0);

    // The swap ratio needs the cache and lies in (0, 1].
    CatalogueLoad sw = g;
    sw.expert_slots = 32;
    sw.expert_swap = 0.5f;
    CHECK(resolve_session_params("moe", sw, "D:\\m\\moe", -1).expert_swap == 0.5f);
    sw.expert_swap = 1.5f;
    CHECK(resolve_session_params("moe", sw, "D:\\m\\moe", -1).expert_swap == 0.0f);
    sw.expert_swap = -0.5f;
    CHECK(resolve_session_params("moe", sw, "D:\\m\\moe", -1).expert_swap == 0.0f);
    sw.expert_swap = 0.5f;
    sw.expert_slots = 0;
    CHECK(resolve_session_params("moe", sw, "D:\\m\\moe", -1).expert_swap == 0.0f);

    // Prefetch and expert-major need the cache; prefetch is capped at 64.
    CatalogueLoad pf = g;
    pf.expert_slots = 32;
    pf.expert_prefetch = 8;
    pf.expert_major = 256;
    CHECK(resolve_session_params("moe", pf, "D:\\m\\moe", -1).expert_prefetch == 8);
    CHECK(resolve_session_params("moe", pf, "D:\\m\\moe", -1).expert_major == 256);
    pf.expert_prefetch = 500;
    pf.expert_major = -3;
    CHECK(resolve_session_params("moe", pf, "D:\\m\\moe", -1).expert_prefetch == 64);
    CHECK(resolve_session_params("moe", pf, "D:\\m\\moe", -1).expert_major == 0);
    pf.expert_slots = 0;
    pf.expert_prefetch = 8;
    CHECK(resolve_session_params("moe", pf, "D:\\m\\moe", -1).expert_prefetch == 0);
}

TEST_CASE("same_load: an adapter, context or GPU split is a different model in RAM") {
    CatalogueLoad g;
    g.found = true;
    g.gguf = true;
    g.lora = "adapter.gguf";
    const auto base = resolve_session_params("m", g, "D:\\m", -1);
    auto other = base;
    CHECK(same_load(base, other));
    other.n_threads = 6; // a tuning knob, not the load
    other.n_batch = 256;
    CHECK(same_load(base, other));
    for (int field = 0; field < 10; ++field) {
        auto changed = base;
        switch (field) {
        case 0:
            changed.lora_path.clear();
            break;
        case 1:
            changed.lora_scale = 0.25f;
            break;
        case 2:
            changed.n_ctx = 1024;
            break;
        case 3:
            changed.n_gpu_layers = 8;
            break;
        case 4:
            changed.model_path = "other";
            break;
        case 5:
            changed.stream_weights = true; // mapped vs heap: a different load
            break;
        case 6:
            changed.expert_slots = 32; // a different set of weights in RAM
            break;
        case 7:
            changed.expert_swap = 0.5f; // the cache is built at load with its routing
            break;
        case 8:
            changed.expert_prefetch = 8; // read from the environment at load
            break;
        case 9:
            changed.expert_major = 256;
            break;
        }
        CHECK_MESSAGE(!same_load(base, changed), field);
    }
}

TEST_CASE("SessionHub reuses only the same load (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;
    xllama::SessionHub hub;
    std::lock_guard<std::mutex> lock(hub.mtx);
    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    std::string err;
    xllama::Session* first = hub.ensure_locked("m", sp, &err);
    REQUIRE_MESSAGE(first, err);
    const uint64_t gen = hub.generation;
    CHECK(hub.ensure_locked("m", sp, &err) == first);
    CHECK(hub.generation == gen);
    sp.n_ctx = 1024; // the embedding num_ctx case: a different load
    REQUIRE(hub.ensure_locked("m", sp, &err) != nullptr);
    CHECK(hub.generation > gen);
    CHECK(hub.session->context_length() == 1024);
    const uint64_t gen2 = hub.generation;
    hub.invalidate_locked("other"); // another model's files changed: no effect
    CHECK(hub.generation == gen2);
    hub.invalidate_locked("m"); // this model's files changed: stale even if equal
    CHECK(hub.session == nullptr);
}
