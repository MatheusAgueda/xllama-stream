// SPDX-License-Identifier: MIT
//
// Expert cache (llama.cpp patch 0004, llama_model_params::n_expert_slots): a MoE
// run with fewer resident experts per layer than the model has must decode
// exactly like the same model with every expert in RAM (same greedy tokens,
// same logits after the prompt) while really reading experts from the file
// (misses > 0 in the cache's final log line).
//
// With a swap ratio (cache-aware routing) the same run reads fewer experts.
//
// Opt-in: XLLAMA_EC_TEST_MODEL = a small MoE GGUF whose experts are not all
// used per token (CI: ggml-org/stories15M_MOE Q8_0, 4 experts, 2 used —
// .github/workflows/expert-cache-host.yml). SKIPs when unset; compiled only
// against a llama.cpp with the patch (LLAMA_HAS_EXPERT_SLOTS), so the unpatched
// build-linux job sees an empty file.
//
// "expert-cache bench" (opt-in: XLLAMA_EC_BENCH_MODEL = a real MoE GGUF) is a
// measurement, not a check: decode speed, hit rate and read time per swap
// ratio, the agreement of each ratio's tokens with exact routing, and the
// exact run's routing as a trace for offline cache-policy simulation
// (.github/workflows/expert-cache-bench.yml).

#include <doctest/doctest.h>

#include "llama.h"

#ifdef LLAMA_HAS_EXPERT_SLOTS

    #include <chrono>
    #include <cmath>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <mutex>
    #include <string>
    #include <vector>

    #ifdef __linux__
        #include <fcntl.h>
        #include <unistd.h>
    #endif

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

// The "N swaps" count of a stats line, or -1 when it is not one.
long long swaps_of(const std::string& line) {
    const char* p = std::strstr(line.c_str(), " hit), ");
    unsigned long long swaps = 0;
    if (!p || std::sscanf(p, " hit), %llu swaps", &swaps) != 1)
        return -1;
    return static_cast<long long>(swaps);
}

// The "prefetch N read" count of a stats line, or -1 when prefetch is off.
long long prefetched_of(const std::string& line) {
    const char* p = std::strstr(line.c_str(), "; prefetch ");
    unsigned long long reads = 0;
    if (!p || std::sscanf(p, "; prefetch %llu read", &reads) != 1)
        return -1;
    return static_cast<long long>(reads);
}

// The "waited for N (P parts read by the plan)" counts of a stats line: the
// reads in flight plans waited for and the parts of them they read themselves,
// both -1 when prefetch is off.
struct WaitStats {
    long long waits = -1;
    long long helped = -1;
};

WaitStats waits_of(const std::string& line) {
    WaitStats w;
    const char* p = std::strstr(line.c_str(), " ms waited for ");
    unsigned long long n = 0, h = 0;
    if (p && std::sscanf(p, " ms waited for %llu (%llu parts read by the plan)", &n, &h) == 2) {
        w.waits = static_cast<long long>(n);
        w.helped = static_cast<long long>(h);
    }
    return w;
}

// The "expert-major N ubatches, R read (M MiB), H resident, K kept, T ms, F
// resident read from the file (mode" counts of a stats line, all -1 (mode
// empty) when no ubatch ran expert-major.
struct EmStats {
    long long ubatches = -1;
    long long reads = -1;
    long long resident = -1;
    long long kept = -1;
    long long from_file = -1;
    std::string mode;
};

EmStats em_of(const std::string& line) {
    EmStats s;
    const char* p = std::strstr(line.c_str(), "; expert-major ");
    unsigned long long u = 0, r = 0, h = 0, k = 0, f = 0;
    double mib = 0.0, ms = 0.0;
    char mode[8] = "";
    if (p && std::sscanf(
                 p, "; expert-major %llu ubatches, %llu read (%lf MiB), %llu resident, %llu kept",
                 &u, &r, &mib, &h, &k) == 5) {
        s.ubatches = static_cast<long long>(u);
        s.reads = static_cast<long long>(r);
        s.resident = static_cast<long long>(h);
        s.kept = static_cast<long long>(k);
    }
    if (p && std::sscanf(p,
                         "; expert-major %llu ubatches, %llu read (%lf MiB), %llu resident, %llu "
                         "kept, %lf ms, %llu resident read from the file (%7[a-z]",
                         &u, &r, &mib, &h, &k, &ms, &f, mode) == 8) {
        s.from_file = static_cast<long long>(f);
        s.mode = mode;
    }
    return s;
}

// The "reads D direct, S staged, B buffered" counts of a stats line, all -1
// when it is not one.
struct IoStats {
    long long direct = -1;
    long long staged = -1;
    long long buffered = -1;
};

IoStats io_of(const std::string& line) {
    IoStats s;
    const char* p = std::strstr(line.c_str(), "; reads ");
    unsigned long long d = 0, st = 0, b = 0;
    if (p && std::sscanf(p, "; reads %llu direct, %llu staged, %llu buffered", &d, &st, &b) == 3) {
        s.direct = static_cast<long long>(d);
        s.staged = static_cast<long long>(st);
        s.buffered = static_cast<long long>(b);
    }
    return s;
}

// An env knob read at model load: k >= 0 sets it, k < 0 restores the default.
void set_env_k(const char* name, int k) {
    #ifdef _WIN32
    _putenv_s(name, k >= 0 ? std::to_string(k).c_str() : "");
    #else
    if (k >= 0)
        setenv(name, std::to_string(k).c_str(), 1);
    else
        unsetenv(name);
    #endif
}

// A string env knob read at model load: nullptr restores the default.
void set_env_s(const char* name, const char* value) {
    #ifdef _WIN32
    _putenv_s(name, value ? value : "");
    #else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
    #endif
}

// LLAMA_EXPERT_PREFETCH: predicted experts per token read ahead (default 0 = off).
void set_prefetch(int k) {
    set_env_k("LLAMA_EXPERT_PREFETCH", k);
}

// LLAMA_EXPERT_MAJOR: largest expert-major ubatch (default 0 = off).
void set_expert_major(int n) {
    set_env_k("LLAMA_EXPERT_MAJOR", n);
}

struct EcRun {
    bool ok = false;
    uint32_t n_ubatch = 0;
    std::vector<llama_token> tokens;
    std::vector<float> logits; // after the prompt
};

// used_override > 0 replaces the model's experts-per-token, which is what bounds
// n_ubatch under the cache (n_slots / used): used 1 makes multi-token ubatches.
// split > 0 decodes the prompt's first `split` tokens on their own first.
EcRun run_greedy(const char* path, uint32_t slots, uint32_t n_ubatch, int64_t used_override,
                 int n_gen, float swap = 0.0f, int split = 0) {
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
    mp.expert_swap_ratio = swap;
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
            const int32_t head = split > 0 && split < n ? split : 0;
            bool ok = head == 0 || llama_decode(ctx, llama_batch_get_one(toks.data(), head)) == 0;
            ok = ok && llama_decode(ctx, llama_batch_get_one(toks.data() + head, n - head)) == 0;
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

// ---- expert-cache bench ----

int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    return v && *v ? std::atoi(v) : def;
}

// One bench run: a swap ratio, a prefetch width (ahead: from the previous
// layer, LLAMA_EXPERT_PREFETCH_AHEAD) and the largest expert-major ubatch (-1 =
// the cache's default).
struct BenchCfg {
    float swap = 0.0f;
    int pf = -1;
    bool ahead = false;
    int em = -1;
};

// "ratio[:k[a][:em]],..." with empty fields for the default, e.g.
// "0,0:8,0:8a,0::512".
std::vector<BenchCfg> env_configs(const char* name, const char* def) {
    const char* v = std::getenv(name);
    const std::string s = v && *v ? v : def;
    std::vector<BenchCfg> out;
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string item = s.substr(pos, comma == std::string::npos ? comma : comma - pos);
        if (!item.empty()) {
            BenchCfg c;
            char* end = nullptr;
            c.swap = std::strtof(item.c_str(), &end);
            int* fields[] = {&c.pf, &c.em};
            for (int* f : fields) {
                if (!end || *end != ':')
                    break;
                const char* at = end + 1;
                const long k = std::strtol(at, &end, 10);
                if (end != at)
                    *f = static_cast<int>(k);
                if (f == &c.pf && *end == 'a') {
                    c.ahead = true;
                    ++end;
                }
            }
            out.push_back(c);
        }
        if (comma == std::string::npos)
            break;
        pos = comma + 1;
    }
    return out;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void log_softmax(const float* lg, int32_t n, std::vector<float>& out) {
    out.resize(static_cast<size_t>(n));
    float mx = lg[0];
    for (int32_t v = 1; v < n; ++v)
        mx = std::fmax(mx, lg[v]);
    double sum = 0.0;
    for (int32_t v = 0; v < n; ++v)
        sum += std::exp(static_cast<double>(lg[v] - mx));
    const float lse = mx + static_cast<float>(std::log(sum));
    for (int32_t v = 0; v < n; ++v)
        out[static_cast<size_t>(v)] = lg[v] - lse;
}

// Drops the model file's clean pages, so every ratio starts from the same cold
// page cache and its reads cost what they would on first touch.
void drop_page_cache(const char* path) {
    #ifdef __linux__
    const int fd = open(path, O_RDONLY);
    if (fd >= 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
    #else
    (void)path;
    #endif
}

// Routing trace: one line per layer and ubatch, "layer n_tok" and then the
// n_tok x n_used expert ids, read through the strides (argsort_top_k's result
// is a view). f == nullptr still observes, so every ratio pays the same split
// overhead.
struct EcTrace {
    FILE* f = nullptr;
};

bool ec_trace_cb(ggml_tensor* t, bool ask, void* user_data) {
    static constexpr char kTopk[] = "ffn_moe_topk-";
    const bool topk = std::strncmp(t->name, kTopk, sizeof(kTopk) - 1) == 0;
    if (ask)
        return topk;
    auto* tr = static_cast<EcTrace*>(user_data);
    if (!topk || !tr->f || t->type != GGML_TYPE_I32 || !t->buffer ||
        !ggml_backend_buffer_is_host(t->buffer))
        return true;
    std::fprintf(tr->f, "%s %lld", t->name + sizeof(kTopk) - 1, static_cast<long long>(t->ne[1]));
    const auto* base = static_cast<const char*>(t->data);
    for (int64_t j = 0; j < t->ne[1]; ++j)
        for (int64_t i = 0; i < t->ne[0]; ++i)
            std::fprintf(tr->f, " %d",
                         *reinterpret_cast<const int32_t*>(base + i * t->nb[0] + j * t->nb[1]));
    std::fputc('\n', tr->f);
    return true;
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
    // tokens whose experts must all be resident at once. The cached run reads
    // ahead (pre-gating prefetch, 2 predicted experts per token): its reads change
    // which experts are resident, never the routing.
    long long prefetched = 0;
    long long prefetched_ahead = 0;
    long long helped = 0;
    const Case cases[] = {{2, 0}, {3, 0}, {2, 1}, {3, 1}};
    for (const Case& c : cases) {
        CAPTURE(c.slots);
        CAPTURE(c.used);
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        set_prefetch(2);
        const EcRun cached = run_greedy(path, c.slots, 512, c.used, 24);
        set_prefetch(-1);
        REQUIRE(cached.ok);
        CHECK(cached.n_ubatch >= 1);
        CHECK(cached.n_ubatch < 64);
        std::string final_line;
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            final_line = log.final_line;
        }
        MESSAGE(final_line);
        const long long misses = misses_of(final_line);
        CHECK(misses > 0);
        CHECK(swaps_of(final_line) == 0);
        CHECK(prefetched_of(final_line) >= 0);
        prefetched += prefetched_of(final_line);
        // A plan waiting for a read in flight reads its queued parts (the
        // default): never more parts than reads it waited for times 6 (2 halves
        // of each of the 3 expert tensors).
        CHECK(final_line.find(", help") != std::string::npos);
        const WaitStats waits = waits_of(final_line);
        CHECK(waits.waits >= 0);
        CHECK(waits.helped >= 0);
        CHECK(waits.helped <= 6 * waits.waits);
        helped += waits.helped;

        // Prefetch off (the default): the plan alone, no prefetch stats.
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        const EcRun plain = run_greedy(path, c.slots, 512, c.used, 24);
        REQUIRE(plain.ok);
        CHECK(plain.n_ubatch == cached.n_ubatch);
        long long plain_misses = -1;
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            MESSAGE(log.final_line);
            plain_misses = misses_of(log.final_line);
            CHECK(plain_misses > 0);
            CHECK(prefetched_of(log.final_line) == -1);
            const IoStats io = io_of(log.final_line);
            CHECK(io.direct >= 0);
            CHECK(io.direct + io.staged + io.buffered > 0);
    #ifdef __linux__
            // O_DIRECT on the runner's file system: the reads bypass the page cache
            CHECK(io.direct + io.staged > 0);
    #endif
        }

        // Buffered whole reads (LLAMA_EXPERT_DIRECT=0, LLAMA_EXPERT_SPLIT=0):
        // the same bytes reach the slots, so the same tokens and logits bit for bit.
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        set_env_k("LLAMA_EXPERT_DIRECT", 0);
        set_env_k("LLAMA_EXPERT_SPLIT", 0);
        const EcRun buffered = run_greedy(path, c.slots, 512, c.used, 24);
        set_env_k("LLAMA_EXPERT_DIRECT", -1);
        set_env_k("LLAMA_EXPERT_SPLIT", -1);
        REQUIRE(buffered.ok);
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            MESSAGE(log.final_line);
            const IoStats io = io_of(log.final_line);
            CHECK(io.direct == 0);
            CHECK(io.staged == 0);
            CHECK(io.buffered > 0);
            CHECK(misses_of(log.final_line) == plain_misses);
        }
        CHECK(buffered.tokens == plain.tokens);
        CHECK(buffered.logits == plain.logits);

        // Unbuffered reads with 8 requests in flight (LLAMA_EXPERT_IO_QD=8, an
        // overlapped handle on Windows; elsewhere the knob does nothing): the
        // same bytes reach the slots, so the same tokens and logits bit for bit.
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        set_env_k("LLAMA_EXPERT_IO_QD", 8);
        const EcRun queued = run_greedy(path, c.slots, 512, c.used, 24);
        set_env_k("LLAMA_EXPERT_IO_QD", -1);
        REQUIRE(queued.ok);
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            MESSAGE(log.final_line);
            const IoStats io = io_of(log.final_line);
            CHECK(io.direct + io.staged + io.buffered > 0);
            CHECK(misses_of(log.final_line) == plain_misses);
    #ifdef _WIN32
            CHECK(log.final_line.find(", qd 8") != std::string::npos);
            // NTFS on the runner: the overlapped reads must not fall back
            CHECK(io.direct + io.staged > 0);
    #else
            CHECK(log.final_line.find(", qd ") == std::string::npos);
    #endif
        }
        CHECK(queued.tokens == plain.tokens);
        CHECK(queued.logits == plain.logits);

        // Prefetch jobs read whole by one I/O thread (LLAMA_EXPERT_SPLIT=0), where
        // the cached run above split each expert in parts over all of them: the
        // reads change which experts are resident, never the routing.
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        set_prefetch(2);
        set_env_k("LLAMA_EXPERT_SPLIT", 0);
        const EcRun whole = run_greedy(path, c.slots, 512, c.used, 24);
        set_env_k("LLAMA_EXPERT_SPLIT", -1);
        set_prefetch(-1);
        REQUIRE(whole.ok);
        CHECK(whole.n_ubatch == cached.n_ubatch);
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            MESSAGE(log.final_line);
            CHECK(prefetched_of(log.final_line) >= 0);
            CHECK(swaps_of(log.final_line) == 0);
        }
        CHECK(whole.tokens == cached.tokens);

        // Prefetch ahead (LLAMA_EXPERT_PREFETCH_AHEAD=1): layer il+1's experts
        // predicted from layer il's post-attention residual and queued right after
        // layer il's load op. Still only residency changes, never the routing.
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        set_prefetch(2);
        set_env_k("LLAMA_EXPERT_PREFETCH_AHEAD", 1);
        const EcRun ahead = run_greedy(path, c.slots, 512, c.used, 24);
        set_env_k("LLAMA_EXPERT_PREFETCH_AHEAD", -1);
        set_prefetch(-1);
        REQUIRE(ahead.ok);
        CHECK(ahead.n_ubatch == cached.n_ubatch);
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            MESSAGE(log.final_line);
            CHECK(prefetched_of(log.final_line) >= 0);
            CHECK(log.final_line.find(", ahead") != std::string::npos);
            CHECK(swaps_of(log.final_line) == 0);
            prefetched_ahead += prefetched_of(log.final_line);
        }
        CHECK(ahead.tokens == cached.tokens);

        // A plan that only waits for the I/O threads (LLAMA_EXPERT_PREFETCH_HELP=0)
        // reads the same experts: the same tokens, no part read by a plan.
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        set_prefetch(2);
        set_env_k("LLAMA_EXPERT_PREFETCH_HELP", 0);
        const EcRun waiting = run_greedy(path, c.slots, 512, c.used, 24);
        set_env_k("LLAMA_EXPERT_PREFETCH_HELP", -1);
        set_prefetch(-1);
        REQUIRE(waiting.ok);
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            MESSAGE(log.final_line);
            CHECK(prefetched_of(log.final_line) >= 0);
            CHECK(log.final_line.find(", help") == std::string::npos);
            CHECK(waits_of(log.final_line).helped == 0);
        }
        CHECK(waiting.tokens == cached.tokens);

        // Cache-aware routing: a resident expert stands in for a missing one
        // with at least half its probability, so the same run reads less.
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        const EcRun swapped = run_greedy(path, c.slots, 512, c.used, 24, 0.5f);
        REQUIRE(swapped.ok);
        CHECK(swapped.n_ubatch == cached.n_ubatch);
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            MESSAGE(log.final_line);
            CHECK(swaps_of(log.final_line) > 0);
            CHECK(misses_of(log.final_line) >= 0);
            CHECK(misses_of(log.final_line) < plain_misses);
        }

        // Swap tail (LLAMA_EXPERT_SWAP_TAIL): the weakest ranks of the top-k take
        // any resident stand-in, the top rank keeps the ratio test (the knob is
        // clamped to used - 1, a no-op at used 1). Same ubatches, the run holds
        // together and still reads less than exact routing.
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        set_env_k("LLAMA_EXPERT_SWAP_TAIL", 64);
        const EcRun tailed = run_greedy(path, c.slots, 512, c.used, 24, 0.5f);
        set_env_k("LLAMA_EXPERT_SWAP_TAIL", -1);
        REQUIRE(tailed.ok);
        CHECK(tailed.n_ubatch == cached.n_ubatch);
        CHECK(tailed.tokens.size() == swapped.tokens.size());
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            MESSAGE(log.final_line);
            CHECK(swaps_of(log.final_line) > 0);
            CHECK(misses_of(log.final_line) >= 0);
            CHECK(misses_of(log.final_line) < plain_misses);
        }

        // Routing trace (LLAMA_EXPERT_TRACE): a ratio above 1 never swaps, so the
        // run is exact routing through the select op - same tokens and misses as
        // plain - and the file holds one "ECT1" record per layer and ubatch with
        // each token's ranking, best first, before any swap.
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        const std::string trace_path = std::string(path) + ".ectrace";
        std::remove(trace_path.c_str());
        set_env_s("LLAMA_EXPERT_TRACE", trace_path.c_str());
        const EcRun traced = run_greedy(path, c.slots, 512, c.used, 24, 1000.0f);
        set_env_s("LLAMA_EXPERT_TRACE", nullptr);
        REQUIRE(traced.ok);
        CHECK(traced.tokens == plain.tokens);
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            MESSAGE(log.final_line);
            CHECK(swaps_of(log.final_line) == 0);
            CHECK(misses_of(log.final_line) == plain_misses);
        }
        {
            FILE* f = std::fopen(trace_path.c_str(), "rb");
            REQUIRE(f != nullptr);
            long long records = 0, bad = 0, token_layers = 0;
            int32_t head[5];
            while (std::fread(head, sizeof(head), 1, f) == 1) {
                records++;
                const int32_t n_tok = head[2], k = head[3];
                if (head[0] != 0x31544345 || head[1] < 0 || n_tok < 1 || k < 1 || k > 32 ||
                    head[4] < 1 || head[4] > k) {
                    bad++;
                    break;
                }
                std::vector<int16_t> ids((size_t)k);
                std::vector<float> sc((size_t)k);
                for (int32_t t = 0; t < n_tok; ++t) {
                    if (std::fread(ids.data(), sizeof(int16_t), (size_t)k, f) != (size_t)k ||
                        std::fread(sc.data(), sizeof(float), (size_t)k, f) != (size_t)k) {
                        bad++;
                        break;
                    }
                    for (int32_t j = 0; j < k; ++j) {
                        if (ids[(size_t)j] < 0 || (j > 0 && sc[(size_t)j] > sc[(size_t)j - 1]))
                            bad++;
                    }
                    token_layers++;
                }
            }
            std::fclose(f);
            std::remove(trace_path.c_str());
            MESSAGE("trace records " << records << ", token-layers " << token_layers);
            CHECK(records > 0);
            CHECK(token_layers >= records);
            CHECK(bad == 0);
        }

        // Both: the prefetch predicts the swapped routing (stand-ins kept, only
        // the experts still missing read ahead). Its reads change the residency
        // and so the routing; the run must hold together and report both.
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        set_prefetch(2);
        const EcRun both = run_greedy(path, c.slots, 512, c.used, 24, 0.5f);
        set_prefetch(-1);
        REQUIRE(both.ok);
        CHECK(both.n_ubatch == cached.n_ubatch);
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            MESSAGE(log.final_line);
            CHECK(misses_of(log.final_line) >= 0);
            CHECK(swaps_of(log.final_line) >= 0);
            CHECK(prefetched_of(log.final_line) >= 0);
        }

        // Same ubatch size: the CPU kernels differ between batch shapes.
        const EcRun full = run_greedy(path, 0, cached.n_ubatch, c.used, 24);
        REQUIRE(full.ok);
        CHECK(full.n_ubatch == cached.n_ubatch);
        CHECK(cached.tokens == full.tokens);
        CHECK(plain.tokens == full.tokens);
        REQUIRE(cached.logits.size() == full.logits.size());
        REQUIRE(plain.logits.size() == full.logits.size());
        CHECK(max_abs_diff(cached.logits, full.logits) <= 1e-4f);
        CHECK(max_abs_diff(plain.logits, full.logits) <= 1e-4f);
        REQUIRE(whole.logits.size() == full.logits.size());
        CHECK(max_abs_diff(whole.logits, full.logits) <= 1e-4f);
        REQUIRE(ahead.logits.size() == full.logits.size());
        CHECK(max_abs_diff(ahead.logits, full.logits) <= 1e-4f);
    }
    // The predictions really read ahead somewhere (layers 1+ of each case), in
    // both placements.
    CHECK(prefetched > 0);
    CHECK(prefetched_ahead > 0);
    // Whether a plan found a read in flight with parts still queued depends on
    // the runner's timing: reported, not checked.
    MESSAGE("prefetch parts read by waiting plans: " << helped);

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

TEST_CASE(
    "expert cache: expert-major prompts match the resident model (opt-in: XLLAMA_EC_TEST_MODEL)") {
    const char* path = std::getenv("XLLAMA_EC_TEST_MODEL");
    if (!path)
        return;
    llama_backend_init();
    EcLog log;
    llama_log_set(ec_log_cb, &log);

    struct Case {
        uint32_t slots;
        int64_t used; // 0 = the model's own
        int split;    // prompt tokens decoded through the slots first
    };
    // LLAMA_EXPERT_MAJOR 64 lifts the ubatch cap to 64: the prompt's ubatch is
    // larger than the slots take and runs expert-major, the decode goes through
    // the slots. split: the first tokens on their own (slot path) leave their
    // experts resident, so the expert-major ubatch after them computes some from
    // their slots instead of reading them. Without split every slot is free, so
    // the ubatch reads some experts into them (kept) and the decode starts from
    // their data: matching the resident model proves all its tensors got them.
    const Case cases[] = {{2, 0, 0}, {2, 0, 1}, {3, 1, 2}};
    for (const Case& c : cases) {
        CAPTURE(c.slots);
        CAPTURE(c.used);
        CAPTURE(c.split);
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        set_expert_major(64);
        const EcRun em = run_greedy(path, c.slots, 512, c.used, 24, 0.0f, c.split);
        set_expert_major(-1);
        REQUIRE(em.ok);
        CHECK(em.n_ubatch == 64);
        std::string final_line;
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            final_line = log.final_line;
        }
        MESSAGE(final_line);
        const EmStats s = em_of(final_line);
        CHECK(s.ubatches == 1);
        CHECK(s.reads > 0);
        if (c.split > 0)
            CHECK(s.resident > 0);
        else
            CHECK(s.kept > 0);
        CHECK(s.kept <= s.reads);
        CHECK(misses_of(final_line) > 0); // the decode
        // auto (the default): any resident expert may come from the file
        CHECK(s.mode == "auto");
        CHECK(s.from_file >= 0);
        CHECK(s.from_file <= s.resident);

        // Resident experts read from the file (LLAMA_EXPERT_EM_RESIDENT=file)
        // or copied from their slots (=slot): the same bytes reach the stage,
        // so the same tokens and logits bit for bit.
        for (const char* mode : {"file", "slot"}) {
            CAPTURE(mode);
            {
                std::lock_guard<std::mutex> lock(log.mtx);
                log.final_line.clear();
            }
            set_expert_major(64);
            set_env_s("LLAMA_EXPERT_EM_RESIDENT", mode);
            const EcRun forced = run_greedy(path, c.slots, 512, c.used, 24, 0.0f, c.split);
            set_env_s("LLAMA_EXPERT_EM_RESIDENT", nullptr);
            set_expert_major(-1);
            REQUIRE(forced.ok);
            std::string forced_line;
            {
                std::lock_guard<std::mutex> lock(log.mtx);
                forced_line = log.final_line;
            }
            MESSAGE(forced_line);
            const EmStats f = em_of(forced_line);
            CHECK(f.mode == mode);
            CHECK(f.resident == s.resident);
            CHECK(f.kept == s.kept);
            CHECK(f.from_file == (std::strcmp(mode, "file") == 0 ? f.resident : 0));
            CHECK(forced.tokens == em.tokens);
            CHECK(forced.logits == em.logits);
        }

        const EcRun full = run_greedy(path, 0, 64, c.used, 24, 0.0f, c.split);
        REQUIRE(full.ok);
        CHECK(full.n_ubatch == 64);
        CHECK(em.tokens == full.tokens);
        REQUIRE(em.logits.size() == full.logits.size());
        CHECK(max_abs_diff(em.logits, full.logits) <= 1e-4f);
    }

    llama_log_set(nullptr, nullptr);
}

// A measurement, not a check (the hyphen keeps it out of "expert cache*"): per
// config (swap ratio[:prefetch width, "a" for ahead[:largest expert-major
// ubatch]]), prefill and decode speed and the cache's final stats; for every
// config after the first, the top-1 agreement and the mean KL against the first
// config's greedy run, teacher-forced on its tokens so a diverging continuation
// cannot hide or inflate the difference. Prefetch never changes the routing: "0"
// against "0:8" or "0:8a" must agree 100% with KL 0; expert-major ("0::512")
// only changes the summation
// order. Env: XLLAMA_EC_BENCH_SLOTS (32), _SWAPS ("0,0:8,0.7,0.7:8"), _TOKENS
// (64), _THREADS (4), _LONG (1: a fourth prompt of ~350 tokens, where
// expert-major prefill pays off), _TRACE (path: the first config's routing,
// for offline cache-policy simulation).
TEST_CASE("expert-cache bench (opt-in: XLLAMA_EC_BENCH_MODEL)") {
    const char* path = std::getenv("XLLAMA_EC_BENCH_MODEL");
    if (!path)
        return;
    const auto slots = static_cast<uint32_t>(env_int("XLLAMA_EC_BENCH_SLOTS", 32));
    const int n_gen = env_int("XLLAMA_EC_BENCH_TOKENS", 64);
    const int n_threads = env_int("XLLAMA_EC_BENCH_THREADS", 4);
    const std::vector<BenchCfg> configs = env_configs("XLLAMA_EC_BENCH_SWAPS", "0,0:8,0.7,0.7:8");
    const char* trace_path = std::getenv("XLLAMA_EC_BENCH_TRACE");
    REQUIRE(!configs.empty());
    REQUIRE(n_gen > 0);

    // Chat format with an empty think block: answers, not reasoning, in n_gen tokens.
    static const char* const prompts[] = {
        "Explica em português de Portugal, em três frases, como funciona uma bomba de calor.",
        "Write a Python function that reads a CSV file and returns the mean of its \"price\" "
        "column.",
        "A train leaves at 14:35 and arrives at 17:10 after a 20 minute stop. For how long was it "
        "moving? Answer briefly.",
        "Summarise the following notes in two sentences.\n\n"
        "The workshop moved to the new building in March. The old site had one loading bay, so "
        "deliveries queued in the street every morning and the drivers often waited more than an "
        "hour. The new building has three bays and a covered yard, which lets two trucks unload "
        "while a third one is loaded. Because the yard is covered, the crew no longer stops work "
        "when it rains, and the forklifts stay dry overnight. The move took eleven days: four to "
        "pack, two to transport the machines and five to install and test them. The paint booth "
        "was the slowest part, since its extraction duct had to be rebuilt to fit the lower "
        "ceiling "
        "of the new hall. During the move the team kept a small stock of finished units at a "
        "rented garage, so that no customer order was delayed. Electricity costs went up by about "
        "a tenth, mostly because of the larger heated area, but the time saved on deliveries more "
        "than pays for it. Next year the plan is to add solar panels on the south roof and to "
        "replace the two oldest compressors, which are noisy and use more power than the newer "
        "ones. The staff asked for a proper break room and bicycle parking, and both are planned "
        "for the summer. The manager also wants a second paint booth if orders keep growing at "
        "the current rate, but that depends on a permit from the town hall, which can take months.",
    };
    constexpr size_t n_all = sizeof(prompts) / sizeof(prompts[0]);
    const size_t n_prompts = n_all - (env_int("XLLAMA_EC_BENCH_LONG", 0) > 0 ? 0 : 1);

    llama_backend_init();
    EcLog log;
    llama_log_set(ec_log_cb, &log);
    EcTrace trace;

    std::vector<llama_token> ref_tokens[n_all];
    std::vector<std::vector<float>> ref_logp[n_all];
    std::vector<float> logp;

    for (size_t r = 0; r < configs.size(); ++r) {
        const bool reference = r == 0;
        const BenchCfg& cfg = configs[r];
        char label[64];
        int at = std::snprintf(label, sizeof(label), "ratio %.2f", cfg.swap);
        if (cfg.pf >= 0)
            at += std::snprintf(label + at, sizeof(label) - static_cast<size_t>(at), " pf %d%s",
                                cfg.pf, cfg.ahead ? " ahead" : "");
        if (cfg.em >= 0)
            std::snprintf(label + at, sizeof(label) - static_cast<size_t>(at), " em %d", cfg.em);
        CAPTURE(label);
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            log.final_line.clear();
        }
        drop_page_cache(path);
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.n_expert_slots = slots;
        mp.expert_swap_ratio = cfg.swap;
        set_prefetch(cfg.pf);
        set_env_k("LLAMA_EXPERT_PREFETCH_AHEAD", cfg.ahead ? 1 : -1);
        set_expert_major(cfg.em);
        auto t_load = std::chrono::steady_clock::now();
        llama_model* model = llama_model_load_from_file(path, mp);
        set_prefetch(-1);
        set_env_k("LLAMA_EXPERT_PREFETCH_AHEAD", -1);
        set_expert_major(-1);
        REQUIRE(model);
        const double load_s = seconds_since(t_load);

        if (reference && trace_path) {
            trace.f = std::fopen(trace_path, "w");
            CHECK(trace.f);
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 1024;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_threads = n_threads;
        cp.n_threads_batch = n_threads;
        if (trace_path) {
            cp.cb_eval = ec_trace_cb;
            cp.cb_eval_user_data = &trace;
        }
        llama_context* ctx = llama_init_from_model(model, cp);
        REQUIRE(ctx);
        const llama_vocab* vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        std::printf("[ec-bench] %s: load %.1f s, %u slots, n_ubatch %u, %d threads\n", label,
                    load_s, slots, llama_n_ubatch(ctx), n_threads);

        int gen_total = 0;
        double gen_s_total = 0.0;
        double kl_sum = 0.0;
        int agree_total = 0;
        int steps_total = 0;
        for (size_t p = 0; p < n_prompts; ++p) {
            llama_memory_clear(llama_get_memory(ctx), true);
            if (trace.f)
                std::fprintf(trace.f, "# prompt %zu\n", p);
            const std::string text = std::string("<|im_start|>user\n") + prompts[p] +
                                     "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            std::vector<llama_token> toks(text.size() + 16);
            const int32_t n =
                llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), toks.data(),
                               static_cast<int32_t>(toks.size()), true, true);
            REQUIRE(n > 0);
            toks.resize(static_cast<size_t>(n));

            auto t0 = std::chrono::steady_clock::now();
            REQUIRE(llama_decode(ctx, llama_batch_get_one(toks.data(), n)) == 0);
            const double prompt_s = seconds_since(t0);

            // The reference generates; the other configs replay its tokens.
            const int steps = reference ? n_gen : static_cast<int>(ref_tokens[p].size());
            int agree = 0;
            int decoded = 0;
            double kl = 0.0;
            std::string out;
            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < steps; ++i) {
                const float* lg = llama_get_logits_ith(ctx, -1);
                llama_token best = 0;
                for (llama_token v = 1; v < n_vocab; ++v)
                    if (lg[v] > lg[best])
                        best = v;
                llama_token next = best;
                if (reference) {
                    ref_tokens[p].push_back(best);
                    log_softmax(lg, n_vocab, logp);
                    ref_logp[p].push_back(logp);
                    char piece[256];
                    const int32_t len =
                        llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, false);
                    if (len > 0)
                        out.append(piece, static_cast<size_t>(len));
                } else {
                    next = ref_tokens[p][static_cast<size_t>(i)];
                    agree += best == next;
                    log_softmax(lg, n_vocab, logp);
                    const std::vector<float>& ref = ref_logp[p][static_cast<size_t>(i)];
                    double d = 0.0;
                    for (int32_t v = 0; v < n_vocab; ++v) {
                        const auto k = static_cast<size_t>(v);
                        d += std::exp(static_cast<double>(ref[k])) * (ref[k] - logp[k]);
                    }
                    kl += d;
                }
                if (llama_vocab_is_eog(vocab, next))
                    break;
                REQUIRE(llama_decode(ctx, llama_batch_get_one(&next, 1)) == 0);
                ++decoded;
            }
            const double gen_s = seconds_since(t0);
            gen_total += decoded;
            gen_s_total += gen_s;
            std::printf("[ec-bench] %s prompt %zu: prefill %d tok %.2f s (%.2f tok/s), "
                        "decode %d tok %.2f s (%.2f tok/s)",
                        label, p, n, prompt_s, n / prompt_s, decoded, gen_s,
                        decoded > 0 ? decoded / gen_s : 0.0);
            if (reference) {
                std::printf("\n[ec-bench] text %zu: %s\n", p, out.c_str());
            } else {
                const int scored = static_cast<int>(ref_tokens[p].size());
                agree_total += agree;
                steps_total += scored;
                kl_sum += kl;
                std::printf(", top-1 %d/%d, mean KL %.5f\n", agree, scored,
                            scored > 0 ? kl / scored : 0.0);
            }
            std::fflush(stdout);
        }
        llama_free(ctx);
        if (trace.f) {
            std::fclose(trace.f);
            trace.f = nullptr;
        }
        llama_model_free(model); // the cache logs its final stats here
        std::string final_line;
        {
            std::lock_guard<std::mutex> lock(log.mtx);
            final_line = log.final_line;
        }
        std::printf("[ec-bench] %s total: decode %.2f tok/s", label,
                    gen_s_total > 0.0 ? gen_total / gen_s_total : 0.0);
        if (!reference)
            std::printf(", top-1 %.1f%%, mean KL %.5f",
                        steps_total > 0 ? 100.0 * agree_total / steps_total : 0.0,
                        steps_total > 0 ? kl_sum / steps_total : 0.0);
        std::printf("\n[ec-bench] %s %s", label,
                    final_line.empty() ? "no cache stats\n" : final_line.c_str());
        std::fflush(stdout);
        CHECK(swaps_of(final_line) >= 0);
    }

    llama_log_set(nullptr, nullptr);
}

#endif // LLAMA_HAS_EXPERT_SLOTS
