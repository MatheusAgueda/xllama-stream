// Host harness for LLAMA_EXPERT_EM_RUN (patch 0020): the real ec_em_gpu_queue, ec_em_runs,
// ec_read_run and the reader threads of ec_em_mm_gpu on fake expert tensors backed by a patterned
// file. A simulated thread 0 waits for each chunk as the real one does and checks that every staged
// expert (and every round's slot) holds its file bytes, that no other byte of the staging buffer or
// the slots changed, and that the dispatch base/stride the real thread 0 would use address them.
// Runs with spans off (0) and on, buffered and unbuffered (O_DIRECT where the system has it).
//
// Not in the CMake build (it compiles llama.cpp's source into itself and leaves the ggml symbols it
// never calls unresolved). From a llama.cpp tree with patches/ applied (L):
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -I$L/include -I$L/ggml/include -I$L/src \
//       expert_cache_em_run.cpp -o h20 -Wl,--unresolved-symbols=ignore-all   (macOS: -Wl,-undefined,dynamic_lookup)
//   ./h20 <scratch file on a disk, not tmpfs> [reps]          -> HARNESS_OK
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
static char * g_stage = nullptr;
void * ggml_backend_buffer_get_base(ggml_backend_buffer_t) {
    return g_stage;
}
}

void llama_log_internal(ggml_log_level, const char *, ...) {}

static constexpr size_t A = EC_DIRECT_ALIGN;
static constexpr uint8_t SENT = 0xA5;

static uint8_t pat(size_t x) {
    uint64_t v = (uint64_t) x*0x9E3779B97F4A7C15ull;
    v ^= v >> 29;
    v *= 0xBF58476D1CE4E5B9ull;
    v ^= v >> 32;
    const uint8_t b = (uint8_t) v;
    return b == SENT ? (uint8_t) 0x5A : b;
}

static std::vector<uint8_t> g_file;

static bool file_bytes(const char * p, size_t n, size_t offs) {
    return memcmp(p, g_file.data() + offs, n) == 0;
}

static bool all_sent(const char * p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if ((uint8_t) p[i] != SENT) {
            return false;
        }
    }
    return true;
}

static char * aligned(size_t n, std::vector<char> & keep) {
    keep.assign(n + 2*A, (char) SENT);
    return keep.data() + (A - (uintptr_t) keep.data() % A) % A;
}

struct Case {
    size_t   eb;
    size_t   off;   // the tensor's file offset
    bool     pad;
    uint32_t run;
    uint32_t chunk;
    int      nth;
    bool     direct;
    int32_t  n_expert;
};

struct Out {
    long     bad = 0;
    uint64_t reads = 0, spans = 0, span_e = 0, direct = 0, staged = 0, buffered = 0;
    int32_t  max_len = 0;
};

static void one(const Case & k, const char * path, std::mt19937 & rng, Out & O) {
    auto C = std::make_unique<llama_expert_cache>();
    C->em_gpu_chunk = k.chunk;
    C->em_run       = k.run;
    C->em_gpu_half  = (size_t) k.chunk*(GGML_PAD(k.eb, 256) + A) + 3*A;
    const size_t n_stage = 2*C->em_gpu_half + 4*A;
    std::vector<char> stage_keep;
    g_stage = aligned(n_stage, stage_keep);
    memset(g_stage, SENT, n_stage);
    C->em_gpu_stage = (ggml_backend_buffer_t) &stage_keep; // only ever handed to the stub
    C->io_direct = k.direct;
    C->files.emplace_back(new ec_file());
    C->files[0]->path   = path;
    C->files[0]->direct = k.direct;
    C->files[0]->qd     = 1;

    const int32_t ne = k.n_expert;
    // the slots: one per expert, padded like ec_buft's tensors (the data at the offset's remainder)
    const size_t slot_nb2 = k.pad ? k.eb + A : k.eb;
    std::vector<char> slot_keep;
    char * slots0 = aligned((size_t) ne*slot_nb2 + 2*A, slot_keep);
    const size_t n_slots_b = (size_t) ne*slot_nb2 + 2*A;
    memset(slots0, SENT, n_slots_b);
    ggml_tensor wt;
    memset(&wt, 0, sizeof(wt));
    wt.data  = slots0 + A + (k.pad ? k.off % A : 0);
    wt.nb[2] = slot_nb2;
    C->tensors.push_back({ &wt, 0, k.off, 0, k.eb, k.pad });
    const ec_tensor & T = C->tensors[0];

    ec_layer L;
    L.cache = C.get();
    L.id    = 0;
    L.slot_of.assign(ne, -1);
    L.em_slot.assign(ne, -1);

    ec_em E;
    E.cache  = C.get();
    E.layer  = &L;
    E.tensor = 0;
    // the ubatch's experts in a random order, some given slots this round, some resident
    std::vector<int32_t> all(ne);
    for (int32_t e = 0; e < ne; ++e) {
        all[e] = e;
    }
    std::shuffle(all.begin(), all.end(), rng);
    const int32_t n_list = 1 + (int32_t) (rng() % (uint32_t) ne);
    E.list.assign(all.begin(), all.begin() + n_list);
    const uint32_t p_kept = rng() % 4, p_res = rng() % 4; // of 8
    for (int32_t e : E.list) {
        const uint32_t r = rng() % 8;
        if (r < p_kept) {
            L.slot_of[e] = L.em_slot[e] = e;
        } else if (r < p_kept + p_res) {
            L.slot_of[e] = e;
        }
    }

    ec_em_gpu_queue(*C, E, L);
    const int32_t nq = (int32_t) E.gq.size();

    // the queue: the round's ones first, then every missing one exactly once
    {
        std::multiset<int32_t> want, got(E.gq.begin(), E.gq.end());
        for (int32_t e : E.list) {
            if (L.em_slot[e] >= 0) {
                want.insert(e);
            }
        }
        for (int32_t e : E.list) {
            if (L.slot_of[e] < 0) {
                want.insert(e);
            }
        }
        for (int32_t i = 0; i < E.n_kept; ++i) {
            if (L.em_slot[E.gq[i]] < 0) {
                O.bad++;
            }
        }
        if (want != got) {
            O.bad++;
        }
    }
    // the spans: the queue in order, each consecutive, inside one chunk, at most run long, and cut
    // only where one of those forces it
    const bool runs = ec_em_run_ok(*C, T);
    if (runs != (k.run >= 2 && k.eb % A == 0)) {
        O.bad++;
    }
    if (runs) {
        int32_t at = 0;
        for (int32_t r = 0; r < E.n_run; ++r) {
            const int32_t i = E.grun[2*r], len = E.grun[2*r + 1];
            if (i != at || len < 1 || len > (int32_t) k.run || i + len > nq) {
                O.bad++;
                break;
            }
            if (i < E.n_kept && len != 1) {
                O.bad++;
            }
            for (int32_t j = 1; j < len; ++j) {
                if (E.gq[i + j] != E.gq[i + j - 1] + 1 || (i + j - E.n_kept)/(int32_t) k.chunk != (i - E.n_kept)/(int32_t) k.chunk) {
                    O.bad++;
                }
            }
            const int32_t nx = i + len;
            if (i >= E.n_kept && nx < nq && len < (int32_t) k.run && E.gq[nx] == E.gq[nx - 1] + 1 &&
                (nx - E.n_kept)/(int32_t) k.chunk == (i - E.n_kept)/(int32_t) k.chunk) {
                O.bad++; // cut without a reason
            }
            O.max_len = std::max(O.max_len, len);
            at = nx;
        }
        if (at != nq) {
            O.bad++;
        }
        for (int32_t i = E.n_kept + 1; i < nq; ++i) {
            if (E.gq[i] <= E.gq[i - 1]) {
                O.bad++;
            }
        }
    } else if (E.n_run != 0) {
        O.bad++;
    }

    // readers 1..nth-1 (the real code), thread 0 simulated
    std::vector<uint64_t> r_reads(k.nth, 0), r_hits(k.nth, 0), r_fb(k.nth, 0), r_fus(k.nth, 0);
    std::vector<std::thread> th;
    for (int ith = 1; ith < k.nth; ++ith) {
        th.emplace_back([&, ith] { ec_em_mm_gpu(nullptr, ith, E, r_reads[ith], r_hits[ith], r_fb[ith], r_fus[ith]); });
    }
    auto wait_for = [&](int32_t at, int32_t n) {
        int s = 0;
        while (E.gread[at].load(std::memory_order_acquire) < n) {
            ec_spin(s);
        }
    };
    wait_for(0, E.n_kept);
    const int32_t chunk = (int32_t) k.chunk;
    for (int32_t c = 0; c < E.n_chunk; ++c) {
        const int32_t i0 = E.n_kept + c*chunk;
        const int32_t n  = std::min(chunk, nq - i0);
        wait_for(1 + c, n);
        size_t nb2 = 0;
        char * base = ec_em_gpu_pos(*C, T, c & 1, 0, &nb2);
        if (runs && nb2 != k.eb) {
            O.bad++;
        }
        char * h0 = g_stage + (size_t) (c & 1)*C->em_gpu_half;
        if (base < h0 || base + (size_t) n*nb2 > h0 + C->em_gpu_half) {
            O.bad++;
        }
        for (int32_t j = 0; j < n; ++j) {
            if (!file_bytes(base + (size_t) j*nb2, k.eb, k.off + (size_t) E.gq[i0 + j]*k.eb)) {
                O.bad++;
            }
        }
        // nothing else in the half moved: packed spans spill nowhere; padded single reads only
        // into the gaps (EC_DIRECT_ALIGN after each expert, and before the first)
        if (runs || !k.pad) {
            if (!all_sent(h0, (size_t) (base - h0)) ||
                !all_sent(base + (size_t) n*nb2, (size_t) (h0 + C->em_gpu_half - (base + (size_t) n*nb2)))) {
                O.bad++;
            }
            for (int32_t j = 0; j + 1 < n && nb2 > k.eb; ++j) {
                if (!all_sent(base + (size_t) j*nb2 + k.eb, nb2 - k.eb)) {
                    O.bad++;
                }
            }
        } else {
            const char * lo = base - A;
            const char * hi = base + (size_t) (n - 1)*nb2 + k.eb + A;
            if (lo < h0 || !all_sent(h0, (size_t) (lo - h0)) || !all_sent(hi, (size_t) (h0 + C->em_gpu_half - hi))) {
                O.bad++;
            }
        }
        memset(h0, SENT, C->em_gpu_half); // the next chunk of this half starts from the sentinel
        E.gdisp.store(c + 1, std::memory_order_release);
    }
    for (auto & t : th) {
        t.join();
    }
    if (!all_sent(g_stage + 2*C->em_gpu_half, n_stage - 2*C->em_gpu_half)) {
        O.bad++;
    }
    // the round's slots hold their experts, the other slots untouched
    for (int32_t e = 0; e < ne; ++e) {
        char * s = (char *) wt.data + (size_t) e*slot_nb2;
        if (L.em_slot[e] >= 0) {
            if (!file_bytes(s, k.eb, k.off + (size_t) e*k.eb)) {
                O.bad++;
            }
        } else if (!all_sent(s, k.eb)) {
            O.bad++;
        }
    }
    uint64_t rd = 0;
    for (int ith = 1; ith < k.nth; ++ith) {
        rd += r_reads[ith];
    }
    if (rd != (uint64_t) nq) {
        O.bad++;
    }
    O.reads  += rd;
    O.spans  += C->em_run_n.load();
    O.span_e += C->em_run_e.load();
    if (runs && C->em_run_e.load() != (uint64_t) (nq - E.n_kept)) {
        O.bad++;
    }
    O.direct   += C->files[0]->n_direct.load();
    O.staged   += C->files[0]->n_staged.load();
    O.buffered += C->files[0]->n_buffered.load();
    C->em_gpu_stage = nullptr; // the stub's handle: nothing for gpu_free to free
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "h20_experts.bin";
    const int    reps = argc > 2 ? atoi(argv[2]) : 40;
    const int32_t NE  = 64;
    const size_t  MAX_EB = 4*A + 512;
    const size_t  MAX_OFF = 2*A + 1056;
    g_file.resize(MAX_OFF + (size_t) NE*MAX_EB + 2*A);
    for (size_t x = 0; x < g_file.size(); ++x) {
        g_file[x] = pat(x);
    }
    {
        FILE * f = fopen(path, "wb");
        if (f == nullptr || fwrite(g_file.data(), 1, g_file.size(), f) != g_file.size()) {
            fprintf(stderr, "cannot write %s\n", path);
            return 2;
        }
        fclose(f);
    }
    bool have_direct = false;
    {
        ec_direct d;
        have_direct = ec_direct_open(path, d, 1);
        ec_direct_close(d);
    }

    std::mt19937 rng(20261112);
    long bad = 0, n_case = 0;
    Out tot_run, tot_one;
    const size_t ebs[]  = { 3*A, A, 2*A, 3*A + 512 };
    const size_t offs[] = { 0, 1056, A + 32, 2*A + 1056 };
    const uint32_t runs[]   = { 0, 2, 4, 16 };
    const uint32_t chunks[] = { 1, 4, 16 };
    const int nths[] = { 2, 3, 6 };
    for (size_t eb : ebs) for (size_t off : offs) for (int pad = 0; pad < 2; ++pad) for (uint32_t run : runs)
    for (uint32_t ch : chunks) for (int nth : nths) for (int dio = 0; dio < (have_direct ? 2 : 1); ++dio) {
        if (pad && eb % A != 0) {
            continue; // ec_buft pads only experts of whole blocks
        }
        Case k { eb, off, pad == 1, run, ch, nth, dio == 1, NE };
        for (int r = 0; r < reps; ++r) {
            Out O;
            one(k, path, rng, O);
            if (O.bad) {
                if (bad < 20) {
                    printf("BAD eb %zu off %zu pad %d run %u chunk %u nth %d direct %d rep %d: %ld\n", eb, off, pad, run, ch, nth, dio, r, O.bad);
                }
                bad += O.bad;
            }
            Out & t = run >= 2 && eb % A == 0 ? tot_run : tot_one;
            t.reads += O.reads; t.spans += O.spans; t.span_e += O.span_e; t.max_len = std::max(t.max_len, O.max_len);
            t.direct += O.direct; t.staged += O.staged; t.buffered += O.buffered;
            n_case++;
        }
    }
    printf("cases %ld, O_DIRECT %s\n", n_case, have_direct ? "yes" : "no");
    printf("spans on:  %" PRIu64 " experts read in %" PRIu64 " spans (mean %.2f, max %d); reads %" PRIu64 " direct, %" PRIu64 " staged, %" PRIu64 " buffered\n",
            tot_run.reads, tot_run.spans, tot_run.spans ? (double) tot_run.span_e/tot_run.spans : 0.0, tot_run.max_len, tot_run.direct, tot_run.staged, tot_run.buffered);
    printf("spans off: %" PRIu64 " experts; reads %" PRIu64 " direct, %" PRIu64 " staged, %" PRIu64 " buffered\n",
            tot_one.reads, tot_one.direct, tot_one.staged, tot_one.buffered);
    remove(path);
    if (bad) {
        printf("HARNESS_FAIL %ld\n", bad);
        return 1;
    }
    printf("HARNESS_OK\n");
    return 0;
}
