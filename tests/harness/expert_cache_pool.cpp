// Host harness for LLAMA_EXPERT_POOL (patch 0023): the real ec_pool_claim on constructed layer state.
// The pool shares one slot allocation across every layer (shrink-time view_src aliasing + bind-time view
// clear, a D3D12/allocator property proved on-device, not here). Its host-side control flow is
// ec_pool_claim, which hands the shared slots to one layer at a time and evicts the layer before it.
// This harness drives that directly and checks the invariants a wrong claim would break:
//   - the first claim (owner was null) takes ownership and evicts nothing;
//   - claiming the current owner is a no-op;
//   - claiming a new layer takes ownership and FULLY evicts the previous owner, leaving the new one's
//     own residents untouched;
//   - with the pool off a claim does nothing at all (the env-gate: a non-pool model is never touched).
//
// Not in the CMake build (it compiles llama.cpp's source into itself and leaves the ggml symbols it
// never calls unresolved). From a llama.cpp tree with patches/ applied (L):
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -I$L/include -I$L/ggml/include -I$L/src \
//       expert_cache_pool.cpp -o h23 -Wl,--unresolved-symbols=ignore-all   (macOS: -Wl,-undefined,dynamic_lookup)
//   ./h23 [cases]                                               -> HARNESS_OK
#include "llama-expert-cache.cpp"

#include <chrono>
#include <cstdarg>
#include <memory>
#include <random>

extern "C" {
int64_t ggml_time_us(void) {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
FILE * ggml_fopen(const char * fname, const char * mode) {
    return fopen(fname, mode);
}
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ABORT %s:%d ", file, line);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    abort();
}
}

void llama_log_internal(ggml_log_level, const char *, ...) {}

// a layer with n_res random experts resident in its first slots
static std::unique_ptr<ec_layer> make_layer(llama_expert_cache * C, int32_t id, int32_t ne, int32_t ns,
                                            int32_t n_res, std::mt19937 & rng) {
    auto L = std::make_unique<ec_layer>();
    L->cache = C;
    L->id    = id;
    L->slot_of.assign(ne, -1);
    L->expert_of.assign(ns, -1);
    L->used.assign(ns, 0);
    std::vector<int32_t> all(ne);
    for (int32_t e = 0; e < ne; ++e) {
        all[e] = e;
    }
    std::shuffle(all.begin(), all.end(), rng);
    for (int32_t i = 0; i < n_res && i < ns; ++i) {
        L->slot_of[all[i]] = i;
        L->expert_of[i]    = all[i];
        L->used[i]         = 1;
    }
    return L;
}

static bool empty_layer(const ec_layer & L) {
    for (int32_t x : L.expert_of) {
        if (x != -1) return false;
    }
    for (int32_t x : L.slot_of) {
        if (x != -1) return false;
    }
    return true;
}

static int32_t n_resident(const ec_layer & L) {
    int32_t n = 0;
    for (int32_t x : L.expert_of) {
        if (x >= 0) ++n;
    }
    return n;
}

int main(int argc, char ** argv) {
    const int n_cases = argc > 1 ? atoi(argv[1]) : 20000;
    std::mt19937 rng(23);
    long bad = 0;
    for (int ci = 0; ci < n_cases; ++ci) {
        const int32_t ne_opts[] = { 8, 16, 64, 128, 256 };
        const int32_t ne = ne_opts[rng() % 5];
        const int32_t k  = 1 + (int32_t) (rng() % (uint32_t) std::min(ne, 8));
        const int32_t ns = k + (int32_t) (rng() % (uint32_t) (ne - k + 1)); // the layer's slots

        auto C = std::make_unique<llama_expert_cache>();
        C->pool          = true;
        C->n_slots       = (uint32_t) ns;
        C->n_expert_used = (uint32_t) k;

        const int32_t res_a = (int32_t) (rng() % (uint32_t) (ns + 1));
        const int32_t res_b = (int32_t) (rng() % (uint32_t) (ns + 1));
        auto A = make_layer(C.get(), 0, ne, ns, res_a, rng);
        auto B = make_layer(C.get(), 1, ne, ns, res_b, rng);
        const int32_t na = n_resident(*A);
        const int32_t nb = n_resident(*B);

        // 1. first claim (owner was null): takes ownership, evicts nothing
        ec_pool_claim(*A);
        if (C->pool_owner != A.get()) {
            fprintf(stderr, "case %d: claim A, owner not A\n", ci); bad++;
        }
        if (n_resident(*A) != na) {
            fprintf(stderr, "case %d: claim A evicted its own experts\n", ci); bad++;
        }

        // 2. claiming the current owner is a no-op
        ec_pool_claim(*A);
        if (C->pool_owner != A.get() || n_resident(*A) != na) {
            fprintf(stderr, "case %d: re-claim A changed state\n", ci); bad++;
        }

        // 3. claim B: takes ownership, A fully evicted, B untouched
        ec_pool_claim(*B);
        if (C->pool_owner != B.get()) {
            fprintf(stderr, "case %d: claim B, owner not B\n", ci); bad++;
        }
        if (!empty_layer(*A)) {
            fprintf(stderr, "case %d: claim B left %d of A resident\n", ci, n_resident(*A)); bad++;
        }
        if (n_resident(*B) != nb) {
            fprintf(stderr, "case %d: claim B evicted its own experts\n", ci); bad++;
        }

        // 4. claim A again: B evicted, A owns (A was emptied in step 3)
        ec_pool_claim(*A);
        if (C->pool_owner != A.get() || !empty_layer(*B)) {
            fprintf(stderr, "case %d: re-claim A did not evict B\n", ci); bad++;
        }

        // 5. env-gate: with the pool off, a claim must do nothing (a non-pool model is untouched)
        auto D = std::make_unique<llama_expert_cache>(); // pool defaults false
        D->n_slots = (uint32_t) ns; D->n_expert_used = (uint32_t) k;
        auto E = make_layer(D.get(), 0, ne, ns, res_a, rng);
        const int32_t ne_res = n_resident(*E);
        ec_pool_claim(*E);
        if (D->pool_owner != nullptr || n_resident(*E) != ne_res) {
            fprintf(stderr, "case %d: pool off, claim changed state\n", ci); bad++;
        }
    }
    if (bad) {
        fprintf(stderr, "HARNESS_FAIL %ld\n", bad);
        return 1;
    }
    printf("HARNESS_OK %d cases\n", n_cases);
    return 0;
}
