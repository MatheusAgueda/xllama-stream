// Host harness for LLAMA_EXPERT_EM_SWAP (patch 0022): the real ec_select on random router scores and
// residency. Per case it checks that an expert-major ubatch (n_tokens > llama_expert_cache_max_ubatch)
// routes at em_swap, with no tail and as a batch (a missing expert some token keeps counts as resident
// for the others), that a slot-path ubatch still routes at swap_ratio / swap_tail / swap_batch, that
// every row against an independent reference, that the expert-major reads are a subset of the exact
// routing's, and that the swaps land in em_swaps or swaps by the ubatch's kind.
//
// Not in the CMake build (it compiles llama.cpp's source into itself and leaves the ggml symbols it
// never calls unresolved). From a llama.cpp tree with patches/ applied (L):
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -I$L/include -I$L/ggml/include -I$L/src \
//       expert_cache_em_swap.cpp -o h22 -Wl,--unresolved-symbols=ignore-all   (macOS: -Wl,-undefined,dynamic_lookup)
//   ./h22 [cases]                                               -> HARNESS_OK
#include "llama-expert-cache.cpp"

#include <chrono>
#include <cstdarg>
#include <random>
#include <set>

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

// the reference: rank by (score desc, id asc); the weakest missing expert of the top-k takes the
// strongest resident outside it while that one scores > 0 and >= r x the missing one's (r = 0 in the
// weakest `tail` ranks); the first failure ends the token
static std::vector<int32_t> ref_row(const std::vector<int32_t> & res, const float * s, int32_t ne, int32_t k,
                                    float ratio, int32_t tail, int32_t & swaps) {
    std::vector<int32_t> rk(ne);
    for (int32_t e = 0; e < ne; ++e) {
        rk[e] = e;
    }
    std::stable_sort(rk.begin(), rk.end(), [s](int32_t a, int32_t b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
    std::vector<int32_t> row(rk.begin(), rk.begin() + k);
    std::vector<int32_t> cand;
    for (int32_t i = k; i < ne; ++i) {
        if (res[rk[i]] >= 0) {
            cand.push_back(rk[i]);
        }
    }
    size_t c = 0;
    swaps = 0;
    for (int32_t j = k - 1; j >= 0; --j) {
        if (res[row[j]] >= 0) {
            continue;
        }
        if (c == cand.size()) {
            break;
        }
        const float r = k - j <= tail ? 0.0f : ratio;
        const int32_t m = row[j];
        if (!(s[cand[c]] > 0.0f) || s[cand[c]] < r*s[m]) {
            break;
        }
        row[j] = cand[c++];
        swaps++;
    }
    return row;
}

int main(int argc, char ** argv) {
    const int n_cases = argc > 1 ? atoi(argv[1]) : 20000;
    std::mt19937 rng(22);
    long bad = 0, n_em = 0, n_slot = 0, n_em_sw = 0;
    uint64_t reads_exact = 0, reads_slots = 0, reads_swap = 0;
    for (int ci = 0; ci < n_cases; ++ci) {
        const int32_t ne_opts[] = { 8, 16, 64, 128, 256 };
        const int32_t ne = ne_opts[rng() % 5];
        const int32_t k  = 1 + (int32_t) (rng() % (uint32_t) std::min(ne, 8));
        const int32_t ns = k + (int32_t) (rng() % (uint32_t) (ne - k + 1)); // the layer's slots
        const float ratios[] = { 0.25f, 0.5f, 1.0f };

        auto C = std::make_unique<llama_expert_cache>();
        C->n_slots       = (uint32_t) ns;
        C->n_expert_used = (uint32_t) k;
        C->swap_ratio    = rng() % 4 == 0 ? 0.0f : ratios[rng() % 3];
        C->swap_tail     = C->swap_ratio > 0.0f ? rng() % 3 : 0;
        C->swap_batch    = C->swap_ratio > 0.0f && rng() % 2 == 0;
        C->em_max        = rng() % 5 == 0 ? 0 : 512;
        C->em_swap       = C->em_max > 0 ? ratios[rng() % 3] : 0.0f;
        const int32_t mu = std::max(1, ns / k); // llama_expert_cache_max_ubatch with no layer registered
        const int32_t nt = 1 + (int32_t) (rng() % (uint32_t) (3*mu + 3));
        const bool    em = C->em_max > 0 && nt > mu;
        if ((uint32_t) mu != llama_expert_cache_max_ubatch(*C)) {
            fprintf(stderr, "case %d: max_ubatch %u, expected %d\n", ci, llama_expert_cache_max_ubatch(*C), mu);
            bad++;
            continue;
        }
        const float   ratio = em ? C->em_swap : C->swap_ratio;
        const int32_t tail  = em ? 0 : (int32_t) C->swap_tail;
        const bool    batch = (em || C->swap_batch) && nt > 1;
        if (!(ratio > 0.0f)) {
            continue; // build_select makes no select op then
        }
        em ? n_em++ : n_slot++;

        auto L = std::make_unique<ec_layer>();
        L->cache = C.get();
        L->id    = 0;
        L->slot_of.assign(ne, -1);
        L->expert_of.assign(ns, -1);
        std::vector<int32_t> all(ne);
        for (int32_t e = 0; e < ne; ++e) {
            all[e] = e;
        }
        std::shuffle(all.begin(), all.end(), rng);
        const int32_t n_res = (int32_t) (rng() % (uint32_t) (ns + 1)); // some slots empty
        for (int32_t i = 0; i < n_res; ++i) {
            L->slot_of[all[i]] = i;
            L->expert_of[i]    = all[i];
        }

        // scores: softmax-like positives, coarse so that ties happen, some zeros
        std::vector<float> sc((size_t) ne*nt);
        for (auto & v : sc) {
            v = rng() % 17 == 0 ? 0.0f : (float) (1 + rng() % 40) / 40.0f;
        }
        std::vector<int32_t> out((size_t) k*nt, -7);
        ggml_tensor ts;
        ggml_tensor td;
        memset(&ts, 0, sizeof(ts));
        memset(&td, 0, sizeof(td));
        ts.type  = GGML_TYPE_F32;
        ts.data  = sc.data();
        ts.ne[0] = ne; ts.ne[1] = nt; ts.ne[2] = 1; ts.ne[3] = 1;
        ts.nb[0] = sizeof(float); ts.nb[1] = (size_t) ne*sizeof(float);
        td.type  = GGML_TYPE_I32;
        td.data  = out.data();
        td.ne[0] = k; td.ne[1] = nt; td.ne[2] = 1; td.ne[3] = 1;
        td.nb[0] = sizeof(int32_t); td.nb[1] = (size_t) k*sizeof(int32_t);
        td.src[0] = &ts;

        ec_select(&td, 0, 1, L.get());

        // the reference: first pass against the residents (expert-major: plus the reads of the tokens
        // before), its kept missing experts resident for all; rd_slots = the reads of a first pass
        // against the residents alone, which the expert-major one may never exceed
        std::vector<int32_t> res = L->slot_of;
        std::set<int32_t> rd_slots;
        int32_t sw = 0;
        if (batch) {
            for (int32_t t = 0; t < nt; ++t) {
                const float * s = sc.data() + (size_t) t*ne;
                for (const int32_t e : ref_row(em ? res : L->slot_of, s, ne, k, ratio, tail, sw)) {
                    res[e] = std::max(res[e], 0);
                }
                for (const int32_t e : ref_row(L->slot_of, s, ne, k, ratio, tail, sw)) {
                    if (L->slot_of[e] < 0) {
                        rd_slots.insert(e);
                    }
                }
            }
        }
        uint64_t sw_all = 0;
        std::set<int32_t> rd_exact, rd_swap;
        const std::vector<int32_t> none(ne, -1); // no residents: the exact top-k
        for (int32_t t = 0; t < nt; ++t) {
            const float * s = sc.data() + (size_t) t*ne;
            const std::vector<int32_t> want = ref_row(res, s, ne, k, ratio, tail, sw);
            sw_all += (uint64_t) sw;
            int32_t dummy = 0;
            for (const int32_t e : ref_row(none, s, ne, k, ratio, 0, dummy)) {
                if (L->slot_of[e] < 0) {
                    rd_exact.insert(e);
                }
            }
            std::set<int32_t> uniq;
            for (int32_t j = 0; j < k; ++j) {
                const int32_t e = out[(size_t) t*k + j];
                uniq.insert(e);
                if (e != want[j]) {
                    if (bad < 10) {
                        fprintf(stderr, "case %d (%s, ne %d, k %d, ns %d, nt %d, ratio %.2f, tail %d, batch %d) token %d rank %d: %d, expected %d\n",
                                ci, em ? "em" : "slot", ne, k, ns, nt, ratio, tail, (int) batch, t, j, e, want[j]);
                    }
                    bad++;
                }
                if (e >= 0 && e < ne && L->slot_of[e] < 0) {
                    rd_swap.insert(e);
                }
            }
            if ((int32_t) uniq.size() != k) {
                fprintf(stderr, "case %d token %d: %zu distinct experts of %d\n", ci, t, uniq.size(), k);
                bad++;
            }
        }
        // the routing reads no expert the exact one would not (swaps only bring residents in)
        for (const int32_t e : rd_swap) {
            if (!rd_exact.count(e)) {
                fprintf(stderr, "case %d: expert %d read, not in the exact routing's reads\n", ci, e);
                bad++;
            }
        }
        if (em) {
            for (const int32_t e : rd_swap) {
                if (!rd_slots.count(e)) {
                    fprintf(stderr, "case %d: expert %d read, not read against the residents alone\n", ci, e);
                    bad++;
                }
            }
            reads_exact += rd_exact.size();
            reads_slots += rd_slots.size();
            reads_swap  += rd_swap.size();
            n_em_sw     += sw_all > 0;
        }
        const uint64_t got_em = C->em_swaps.load(), got_sl = C->swaps.load();
        if ((em ? got_em : got_sl) != sw_all || (em ? got_sl : got_em) != 0) {
            fprintf(stderr, "case %d (%s): em_swaps %llu swaps %llu, expected %llu in %s\n", ci, em ? "em" : "slot",
                    (unsigned long long) got_em, (unsigned long long) got_sl, (unsigned long long) sw_all, em ? "em_swaps" : "swaps");
            bad++;
        }
        C->em_rounds = 0; // no stats line from the destructor
        C->n_rounds  = 0;
    }
    printf("cases %d: %ld expert-major (%ld with swaps), %ld slot path; expert-major reads %llu exact, %llu against the "
           "residents alone, %llu (%.1f%% of exact)\n",
           n_cases, n_em, n_em_sw, n_slot, (unsigned long long) reads_exact, (unsigned long long) reads_slots,
           (unsigned long long) reads_swap, reads_exact ? 100.0*reads_swap/reads_exact : 0.0);
    if (bad > 0 || n_em == 0 || n_slot == 0 || n_em_sw == 0) {
        printf("HARNESS_FAILED %ld\n", bad);
        return 1;
    }
    printf("HARNESS_OK\n");
    return 0;
}
