// SPDX-License-Identifier: MIT
//
// Expert cache (llama.cpp patch 0004, llama_model_params::n_expert_slots): a MoE
// run with fewer resident experts per layer than the model has must decode
// exactly like the same model with every expert in RAM (same greedy tokens,
// same logits after the prompt) while really reading experts from the file
// (misses > 0 in the cache's final log line).
//
// Opt-in: XLLAMA_EC_TEST_MODEL = a small MoE GGUF whose experts are not all
// used per token (CI: ggml-org/stories15M_MOE Q8_0, 4 experts, 2 used —
// .github/workflows/expert-cache-host.yml). SKIPs when unset; compiled only
// against a llama.cpp with the patch (LLAMA_HAS_EXPERT_SLOTS), so the unpatched
// build-linux job sees an empty file.

#include <doctest/doctest.h>

#include "llama.h"

#ifdef LLAMA_HAS_EXPERT_SLOTS

    #include <cmath>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <mutex>
    #include <string>
    #include <vector>

namespace {

struct EcLog {
    std::mutex mtx;
    std::string final_line;
};

void ec_log_cb(ggml_log_level, const char* text, void* user_data) {
    if (!text || !std::strstr(text, "expert cache (final)"))
        return;
    auto* log = static_cast<EcLog*>(user_data);
    std::lock_guard<std::mutex> lock(log->mtx);
    log->final_line = text;
}

// The "N misses" count of a stats line, or -1 when it is not one.
long long misses_of(const std::string& line) {
    const char* p = std::strstr(line.c_str(), " hits, ");
    unsigned long long misses = 0;
    if (!p || std::sscanf(p, " hits, %llu misses", &misses) != 1)
        return -1;
    return static_cast<long long>(misses);
}

struct EcRun {
    bool ok = false;
    uint32_t n_ubatch = 0;
    std::vector<llama_token> tokens;
    std::vector<float> logits; // after the prompt
};

// used_override > 0 replaces the model's experts-per-token, which is what bounds
// n_ubatch under the cache (n_slots / used): used 1 makes multi-token ubatches.
EcRun run_greedy(const char* path, uint32_t slots, uint32_t n_ubatch, int64_t used_override,
                 int n_gen) {
    EcRun r;
    llama_model_kv_override kv[2] = {};
    if (used_override > 0) {
        char arch[64] = {};
        // The key is per architecture ("llama.expert_used_count"); read it from a
        // vocab-only load so the test works for any MoE arch.
        llama_model_params vp = llama_model_default_params();
        vp.vocab_only = true;
        llama_model* meta = llama_model_load_from_file(path, vp);
        if (!meta)
            return r;
        llama_model_meta_val_str(meta, "general.architecture", arch, sizeof(arch));
        llama_model_free(meta);
        kv[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::snprintf(kv[0].key, sizeof(kv[0].key), "%s.expert_used_count", arch);
        kv[0].val_i64 = used_override;
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.n_expert_slots = slots;
    if (used_override > 0)
        mp.kv_overrides = kv;
    llama_model* model = llama_model_load_from_file(path, mp);
    if (!model)
        return r;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 64;
    cp.n_ubatch = n_ubatch;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (ctx) {
        r.n_ubatch = llama_n_ubatch(ctx);
        const llama_vocab* vocab = llama_model_get_vocab(model);
        const std::string prompt = "Once upon a time, there was a little girl named Lily who";
        std::vector<llama_token> toks(128);
        const int32_t n =
            llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), toks.data(),
                           static_cast<int32_t>(toks.size()), true, false);
        if (n > 0) {
            toks.resize(static_cast<size_t>(n));
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            r.tokens.reserve(static_cast<size_t>(n_gen));
            bool ok = llama_decode(ctx, llama_batch_get_one(toks.data(), n)) == 0;
            for (int i = 0; ok && i < n_gen; ++i) {
                const float* lg = llama_get_logits_ith(ctx, -1);
                if (i == 0)
                    r.logits.assign(lg, lg + n_vocab);
                llama_token best = 0;
                for (llama_token v = 1; v < n_vocab; ++v)
                    if (lg[v] > lg[best])
                        best = v;
                r.tokens.push_back(best);
                ok = llama_decode(ctx, llama_batch_get_one(&r.tokens.back(), 1)) == 0;
            }
            r.ok = ok;
        }
        llama_free(ctx);
    }
    llama_model_free(model); // the cache logs its final stats here
    return r;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        d = std::fmax(d, std::fabs(a[i] - b[i]));
    return d;
}

} // namespace

TEST_CASE("expert cache: decodes like the resident model (opt-in: XLLAMA_EC_TEST_MODEL)") {
    const char* path = std::getenv("XLLAMA_EC_TEST_MODEL");
    if (!path)
        return;
    llama_backend_init();
    EcLog log;
    llama_log_set(ec_log_cb, &log);

    struct Case {
        uint32_t slots;
        int64_t used; // 0 = the model's own
    };
    // 2 slots with 2 used: every token's experts evict the previous token's.
    // 3 slots: a partial LRU, hits and misses mixed. used 1: ubatches of 2 and 3
    // tokens whose experts must all be resident at once.
    const Case cases[] = {{2, 0}, {3, 0}, {2, 1}, {3, 1}};
    for (const Case& c : cases) {
        CAPTURE(c.slots);
        CAPTURE(c.used);
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        const EcRun cached = run_greedy(path, c.slots, 512, c.used, 24);
        REQUIRE(cached.ok);
        CHECK(cached.n_ubatch >= 1);
        CHECK(cached.n_ubatch < 64);
        std::string final_line;
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            final_line = log.final_line;
        }
        MESSAGE(final_line);
        CHECK(misses_of(final_line) > 0);

        // Same ubatch size: the CPU kernels differ between batch shapes.
        const EcRun full = run_greedy(path, 0, cached.n_ubatch, c.used, 24);
        REQUIRE(full.ok);
        CHECK(full.n_ubatch == cached.n_ubatch);
        CHECK(cached.tokens == full.tokens);
        REQUIRE(cached.logits.size() == full.logits.size());
        CHECK(max_abs_diff(cached.logits, full.logits) <= 1e-4f);
    }

    // As many slots as experts: no cache, no ubatch cap, no stats line.
    {
        std::lock_guard<std::mutex> lock(log.mtx);
        log.final_line.clear();
    }
    const EcRun all = run_greedy(path, 64, 64, 0, 4);
    REQUIRE(all.ok);
    CHECK(all.n_ubatch == 64);
    {
        std::lock_guard<std::mutex> lock(log.mtx);
        CHECK(log.final_line.empty());
    }

    llama_log_set(nullptr, nullptr);
}

#endif // LLAMA_HAS_EXPERT_SLOTS
