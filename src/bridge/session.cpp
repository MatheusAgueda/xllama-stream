// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// xllama::Session — persistent model session for multi-turn applications.
// ORT GenAI path (UWP): model + tokenizer loaded once, reused per generate().
// llama.cpp path (Linux): model kept alive; context rebuilt per generate().

#include "xllama/session.h"
#include "xllama/chat_prompt.h"
#include "xllama/inference_params.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/prefix_cache.h"
#include "xllama/routing_policy.h"
#include "xllama/session_hub.h"
#include "xllama/utf8_utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>

// Backend availability. A build defines XLLAMA_USE_ORT when ORT GenAI is linked.
// llama.cpp is available on Linux (always) and on the UWP llamacpp variant
// (XllamaBackend=llamacpp → XLLAMA_USE_ORT undefined). The unified UWP build
// defines BOTH explicitly; this shim derives the llama flag so exactly one path
// matches every current single-backend build without touching the build files.
#if !defined(XLLAMA_USE_LLAMA)
    #if defined(XLLAMA_LINUX) || !defined(XLLAMA_USE_ORT)
        #define XLLAMA_USE_LLAMA 1
    #endif
#endif

#if !defined(XLLAMA_USE_ORT) && !defined(XLLAMA_USE_LLAMA)
    #error "no inference backend compiled (define XLLAMA_USE_ORT and/or XLLAMA_USE_LLAMA)"
#endif

namespace xllama {
namespace detail {
// Backend factories, each defined in its own #ifdef block below. The public
// Session::create dispatches to one of these.
#ifdef XLLAMA_USE_ORT
std::unique_ptr<Session> create_ort(const SessionParams& sp, std::string* err);
#endif
#ifdef XLLAMA_USE_LLAMA
std::unique_ptr<Session> create_llama(const SessionParams& sp, std::string* err);
#endif
} // namespace detail
} // namespace xllama

// ---------------------------------------------------------------------------
// ONNX Runtime GenAI path (UWP / Xbox Series S)
// ---------------------------------------------------------------------------
#ifdef XLLAMA_USE_ORT

// clang-format off
#include <windows.h>
#include <eh.h>
#include "ort_genai_c.h"

#include "decode_loop_ort.h"
#include "ort_common.h"
#include "ort_sampling.h" // shared ORT search-param builder (#125); needs ort_genai_c.h
#include "xllama/ort_raii.h"
// clang-format on

namespace xllama {

class OrtSession final : public Session {
  public:
    OgaModelPtr m_model;
    OgaTokenizerPtr m_tok;
    int m_n_ctx;

    // Persistent chat state for KV-cache reuse (continuous decoding). Kept alive
    // between generate() calls when GenerateParams::reuse_kv is set. m_chat_params
    // must outlive m_chat_gen.
    OgaGeneratorParamsPtr m_chat_params;
    OgaGeneratorPtr m_chat_gen;
    OgaTokenizerStreamPtr m_chat_stream;
    bool m_chat_valid = false;
    // Sampling signature bound into m_chat_gen — the generator can only be reused
    // while these are unchanged (ORT binds sampling at generator creation).
    float m_b_temp = 0.0f, m_b_top_p = 0.0f, m_b_rep = 0.0f;
    int m_b_top_k = 0;

    explicit OrtSession(OgaModelPtr model, OgaTokenizerPtr tok, int n_ctx)
        : m_model(std::move(model)), m_tok(std::move(tok)), m_n_ctx(n_ctx) {}

    void reset_chat_state() {
        m_chat_gen.reset(); // generator before its params
        m_chat_params.reset();
        m_chat_stream.reset();
        m_chat_valid = false;
    }

    bool sampling_matches(const GenerateParams& gp) const {
        return m_chat_valid && m_b_temp == gp.temperature && m_b_top_p == gp.top_p &&
               m_b_top_k == gp.top_k && m_b_rep == gp.repetition_penalty;
    }

    OgaGeneratorParamsPtr make_params(const GenerateParams& gp) {
        OgaGeneratorParams* raw_params = nullptr;
        oga_check(OgaCreateGeneratorParams(m_model.get(), &raw_params), "OgaCreateGeneratorParams");
        OgaGeneratorParamsPtr gparams(raw_params);
        // Session ALWAYS saturates max_length to n_ctx via the shared #130
        // ladder (resolve_max_length, inference_params.h) — the interior band
        // is a measured DirectML valley (§5c; table at the stateless caller).
        // The n_predict bound is enforced by run_decode's cap instead.
        const int max_length = resolve_max_length(m_n_ctx, 0, 0, /*override_v=*/-1);
        oga_check(OgaGeneratorParamsSetSearchNumber(gparams.get(), "max_length",
                                                    static_cast<double>(max_length)),
                  "SetSearchNumber max_length");
        // #125 follow-up: temperature and the greedy guard via the shared helper,
        // so this path and run_inference cannot drift apart again. This is the
        // fix for the greedy gap — make_params used to set the full chain
        // unconditionally, running the repetition penalty before argmax at
        // temperature 0.
        apply_ort_sampling(gparams.get(), gp.sampling());
        return gparams;
    }

    // Decode loop shared by the stateless and chat paths. The caller has already
    // appended the prompt tokens and captured t_prefill_start immediately before
    // AppendTokenSequences; prefill-end is marked after the first
    // GenerateNextToken (in ORT GenAI the prompt prefill runs during the first
    // compute step), so decode timing excludes prefill. n_predict_cap > 0 caps
    // per-turn generation (chat mode); 0 relies on max_length (stateless mode).
    //
    // Consolidated into decode_loop_ort.h to eliminate duplication with
    // run_inference_ort (inference.cpp).
    void run_decode(OgaGenerator* gen, OgaTokenizerStream* stream, const GenerateParams& gp,
                    InferenceResult& res, std::chrono::steady_clock::time_point t_prefill_start,
                    int n_prompt_tok, int n_predict_cap, InferenceWatchdog* wd = nullptr) {
        detail::run_decode_loop_ort(gen, stream, gp, res, t_prefill_start, n_prompt_tok,
                                    n_predict_cap, wd);
    }

    // Legacy stateless turn: fresh generator, full prompt, destroyed at return.
    void generate_stateless(const GenerateParams& gp, InferenceResult& res) {
        if (gp.on_status)
            gp.on_status("tokenizing");

        OgaTokenizerStream* raw_stream = nullptr;
        oga_check(OgaCreateTokenizerStream(m_tok.get(), &raw_stream), "OgaCreateTokenizerStream");
        OgaTokenizerStreamPtr stream(raw_stream);

        OgaSequences* raw_seqs = nullptr;
        oga_check(OgaCreateSequences(&raw_seqs), "OgaCreateSequences");
        OgaSequencesPtr seqs(raw_seqs);
        oga_check(OgaTokenizerEncode(m_tok.get(), gp.prompt.c_str(), seqs.get()),
                  "OgaTokenizerEncode");
        int n_prompt_tok = static_cast<int>(OgaSequencesGetSequenceCount(seqs.get(), 0));

        // #130: request the FULL context as max_length and bound generation with
        // the n_predict cap in run_decode — exactly what generate_chat already
        // does. The obvious alternative, max_length = min(n_ctx, prompt +
        // n_predict), is what this used to do, and on DirectML it is 3-4x slower.
        //
        // Measured 2026-07-21, Series S, one byte-identical 1289-token prompt,
        // only n_predict varied (bench/results/phase12-maxlen-band.csv):
        //
        //   max_length  1297  1400  1545  1650  1801  1950  2048
        //   prefill t/s   515   475   170   215   130   212   611
        //
        // There is a valley in max_length between ~1400 and n_ctx, deepest near
        // 1800, with both edges clean. The prompt never changed, so this is not a
        // prompt-length effect. A control run at n_ctx 3072 holding max_length at
        // 1801 reproduced the slow figure (132 t/s), so n_ctx has no effect of its
        // own — max_length alone controls it. Saturating also costs LESS memory
        // here (1968 MB against 2483 at max_length 1950).
        //
        // The shipping default (n_predict 256) landed at 1545 — inside the valley.
        // Mechanism still unknown; suspected shape-bucketed DML kernel selection.
        OgaGeneratorParamsPtr gparams = make_params(gp);
        OgaGenerator* raw_gen = nullptr;
        oga_check(OgaCreateGenerator(m_model.get(), gparams.get(), &raw_gen), "OgaCreateGenerator");
        OgaGeneratorPtr gen(raw_gen);

        if (gp.on_status)
            gp.on_status("generating");
        InferenceWatchdog wd(gp.progress(), gp.abort_flag);
        wd.enter_prefill(n_prompt_tok);
        auto t0 = std::chrono::steady_clock::now();
        oga_check(OgaGenerator_AppendTokenSequences(gen.get(), seqs.get()), "AppendTokenSequences");
        wd.prefill_done(n_prompt_tok, std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - t0)
                                          .count());
        // Reproduce the old allowance exactly. The previous max_length of
        // min(n_ctx, prompt + n_predict) permitted min(n_ctx - prompt, n_predict)
        // new tokens; max_length is no longer the bound, so the cap has to carry
        // that arithmetic or a long prompt would now generate past the context.
        // Clamped to >= 1 because run_decode treats a cap of 0 as "no cap": a
        // prompt at or beyond the context would otherwise flip from a hard stop
        // to unbounded generation. Such a prompt fails inside ORT first
        // ("input_ids size exceeds max length"), which is the intended error.
        const int n_predict_cap = std::max(1, std::min(m_n_ctx - n_prompt_tok, gp.n_predict));
        run_decode(gen.get(), stream.get(), gp, res, t0, n_prompt_tok, n_predict_cap, &wd);
        if (wd.stalled()) {
            res.success = false;
            res.stalled = true;
            res.error_msg = wd.stall_error();
            log_output(("[xllama] " + res.error_msg + "\n").c_str());
            return;
        }
        log_gen(res, false);
    }

    // Continuation turn: append `gp.prompt` (the new turn's tokens) to the
    // persistent generator; on reset_kv or a sampling change, rebuild it first
    // (then `gp.prompt` is the full context).
    void generate_chat(const GenerateParams& gp, InferenceResult& res) {
        const bool reuse = !gp.reset_kv && sampling_matches(gp);
        if (!reuse) {
            reset_chat_state();
            m_chat_params = make_params(gp);
            OgaGenerator* raw_gen = nullptr;
            oga_check(OgaCreateGenerator(m_model.get(), m_chat_params.get(), &raw_gen),
                      "OgaCreateGenerator(chat)");
            m_chat_gen.reset(raw_gen);
            OgaTokenizerStream* raw_stream = nullptr;
            oga_check(OgaCreateTokenizerStream(m_tok.get(), &raw_stream),
                      "OgaCreateTokenizerStream(chat)");
            m_chat_stream.reset(raw_stream);
            m_b_temp = gp.temperature;
            m_b_top_p = gp.top_p;
            m_b_top_k = gp.top_k;
            m_b_rep = gp.repetition_penalty;
            m_chat_valid = true;
        }

        if (gp.on_status)
            gp.on_status("tokenizing");
        OgaSequences* raw_seqs = nullptr;
        oga_check(OgaCreateSequences(&raw_seqs), "OgaCreateSequences");
        OgaSequencesPtr seqs(raw_seqs);
        oga_check(OgaTokenizerEncode(m_tok.get(), gp.prompt.c_str(), seqs.get()),
                  "OgaTokenizerEncode");
        int n_prompt_tok = static_cast<int>(OgaSequencesGetSequenceCount(seqs.get(), 0));

        // #173: a continuation that cannot fit (KV + delta + >=1 generated token
        // over n_ctx) is predictable from state we already hold — fail before
        // touching the generator instead of letting the append discover it. The
        // throw lands in generate()'s catch, which resets the chat state; the
        // caller sees n_eval == 0 and falls back to a trimmed full prefill.
        if (reuse) {
            const int kv = static_cast<int>(OgaGenerator_GetSequenceCount(m_chat_gen.get(), 0));
            if (kv + n_prompt_tok + 1 > m_n_ctx)
                throw std::runtime_error("context full: kv=" + std::to_string(kv) +
                                         " + delta=" + std::to_string(n_prompt_tok) +
                                         " exceeds n_ctx=" + std::to_string(m_n_ctx));
        }

        if (gp.on_status)
            gp.on_status("generating");
        InferenceWatchdog wd(gp.progress(), gp.abort_flag);
        wd.enter_prefill(n_prompt_tok);
        auto t0 = std::chrono::steady_clock::now();
        oga_check(OgaGenerator_AppendTokenSequences(m_chat_gen.get(), seqs.get()),
                  "AppendTokenSequences(chat)");
        wd.prefill_done(n_prompt_tok, std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - t0)
                                          .count());
        run_decode(m_chat_gen.get(), m_chat_stream.get(), gp, res, t0, n_prompt_tok, gp.n_predict,
                   &wd);
        if (wd.stalled()) {
            res.success = false;
            res.stalled = true;
            res.error_msg = wd.stall_error();
            log_output(("[xllama] " + res.error_msg + "\n").c_str());
            return;
        }

        size_t kv = OgaGenerator_GetSequenceCount(m_chat_gen.get(), 0);
        char lb[192];
        snprintf(lb, sizeof(lb),
                 "[xllama] chat turn: reuse=%d prefill=%d tok decode=%d tok kv_len=%zu\n",
                 reuse ? 1 : 0, res.n_p_eval, res.n_eval, kv);
        log_output(lb);
    }

    void log_gen(const InferenceResult& res, bool chat) {
        double dt =
            (res.n_eval > 0 && res.t_eval_ms > 0) ? res.n_eval / (res.t_eval_ms / 1000.0) : 0.0;
        // Prefill rate too: without it the per-turn TTFT (and the #130
        // cold-vs-warm gap) cannot be reconstructed from the log.
        double pt = (res.n_p_eval > 0 && res.t_p_eval_ms > 0)
                        ? res.n_p_eval / (res.t_p_eval_ms / 1000.0)
                        : 0.0;
        char log_buf[256];
        snprintf(log_buf, sizeof(log_buf),
                 "[xllama] session generate%s: prefill=%.1f tok/s (%d tok, %.0f ms) "
                 "decode=%.1f tok/s n=%d\n",
                 chat ? " (chat)" : "", pt, res.n_p_eval, res.t_p_eval_ms, dt, res.n_eval);
        log_output(log_buf);
    }

    InferenceResult generate(const GenerateParams& gp) override {
        InferenceResult res;
        install_se_translator();
        try {
            if (gp.reuse_kv)
                generate_chat(gp, res);
            else
                generate_stateless(gp, res);
        } catch (const std::exception& e) {
            res.success = false;
            res.error_msg = e.what();
            if (gp.reuse_kv)
                reset_chat_state(); // poisoned generator — force a fresh one next turn
            log_output(("[xllama] session generate error: " + res.error_msg + "\n").c_str());
            if (gp.on_status)
                gp.on_status("error: " + res.error_msg);
        }
        return res;
    }

    int count_tokens(const std::string& prompt) override {
        OgaSequences* raw_seqs = nullptr;
        oga_check(OgaCreateSequences(&raw_seqs), "OgaCreateSequences");
        OgaSequencesPtr seqs(raw_seqs);
        oga_check(OgaTokenizerEncode(m_tok.get(), prompt.c_str(), seqs.get()),
                  "OgaTokenizerEncode");
        return static_cast<int>(OgaSequencesGetSequenceCount(seqs.get(), 0));
    }

    int context_length() const override {
        return m_n_ctx;
    }
};

namespace detail {
std::unique_ptr<Session> create_ort(const SessionParams& sp, std::string* err) {
    install_se_translator();

    const std::string model_dir = resolve_model_path(sp.model_path);
    try {
        // Redirect ORT GenAI internal log messages into xllama.log.
        register_oga_logging();

        OgaModel* raw_model = nullptr;
        oga_check(OgaCreateModel(model_dir.c_str(), &raw_model), "OgaCreateModel");
        OgaModelPtr model(raw_model);

        OgaTokenizer* raw_tok = nullptr;
        oga_check(OgaCreateTokenizer(model.get(), &raw_tok), "OgaCreateTokenizer");
        OgaTokenizerPtr tok(raw_tok);

        log_output("[xllama] Session: model loaded (persistent)\n");
        GpuMemInfo gpu = gpu_mem_info();
        if (gpu.available) {
            char gpu_buf[128];
            snprintf(gpu_buf, sizeof(gpu_buf),
                     "[xllama] gpu-mem post-load: current=%zuMB budget=%zuMB\n", gpu.current_mb,
                     gpu.budget_mb);
            log_output(gpu_buf);
        }
        int n_ctx = sp.n_ctx > 0 ? sp.n_ctx : 2048;

        // #130: DirectML pays a large one-time per-process cost on the first
        // generate (§5e: cold→warm prefill 1.64-1.72×; CPU control 1.00× —
        // suspected lazy kernel compilation). Pay it here, inside the load
        // phase the UI already presents as such, instead of on the user's
        // first turn. The throwaway generator uses max_length = n_ctx exactly
        // like every real turn: the §5c valley is keyed by max_length, so
        // warming a smaller shape could warm the wrong regime. Failure is
        // non-fatal — the session is still usable, just cold.
        if (sp.dml_warmup && model_is_dml(sp.model_path)) {
            try {
                const auto t_w0 = std::chrono::steady_clock::now();
                OgaGeneratorParams* raw_wp = nullptr;
                oga_check(OgaCreateGeneratorParams(model.get(), &raw_wp),
                          "OgaCreateGeneratorParams warmup");
                OgaGeneratorParamsPtr wparams(raw_wp);
                oga_check(OgaGeneratorParamsSetSearchNumber(wparams.get(), "max_length",
                                                            static_cast<double>(n_ctx)),
                          "SetSearchNumber warmup");
                OgaSequences* raw_wseqs = nullptr;
                oga_check(OgaCreateSequences(&raw_wseqs), "OgaCreateSequences warmup");
                OgaSequencesPtr wseqs(raw_wseqs);
                // A ~2-token warm-up left turn-1 at ~76% of the warm prefill
                // rate (validated on-console, PR #158 build 678): a tiny
                // prompt never exercises the long-sequence prefill GEMMs. A
                // few hundred tokens lands in the same regime as real turns.
                std::string wtext;
                wtext.reserve(2048);
                for (int i = 0; i < 256; ++i)
                    wtext += "warmup ";
                oga_check(OgaTokenizerEncode(tok.get(), wtext.c_str(), wseqs.get()),
                          "OgaTokenizerEncode warmup");
                OgaGenerator* raw_wgen = nullptr;
                oga_check(OgaCreateGenerator(model.get(), wparams.get(), &raw_wgen),
                          "OgaCreateGenerator warmup");
                OgaGeneratorPtr wgen(raw_wgen);
                oga_check(OgaGenerator_AppendTokenSequences(wgen.get(), wseqs.get()),
                          "AppendTokenSequences warmup");
                // 3 steps: the first IS the prefill compute; only later ones
                // run the seq=1 decode kernels (turn-1 decode stayed 17% slow
                // with a single step — decode was never warmed).
                for (int i = 0; i < 3; ++i)
                    oga_check(OgaGenerator_GenerateNextToken(wgen.get()),
                              "GenerateNextToken warmup");
                const double w_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - t_w0)
                                        .count();
                char w_buf[96];
                snprintf(w_buf, sizeof(w_buf), "[xllama] Session: DML warm-up %.0f ms\n", w_ms);
                log_output(w_buf);
            } catch (const std::exception& we) {
                log_output(("[xllama] Session: DML warm-up failed (non-fatal): " +
                            std::string(we.what()) + "\n")
                               .c_str());
            }
        }
        return std::make_unique<OrtSession>(std::move(model), std::move(tok), n_ctx);

    } catch (const std::exception& e) {
        if (err)
            *err = e.what();
        log_output(("[xllama] Session::create error: " + std::string(e.what()) + "\n").c_str());
        return nullptr;
    }
}
} // namespace detail

} // namespace xllama

#endif // XLLAMA_USE_ORT

// ---------------------------------------------------------------------------
// llama.cpp path (Linux dev + UWP llamacpp/unified)
// ---------------------------------------------------------------------------
#ifdef XLLAMA_USE_LLAMA

    #include "llama.h"

    #include "decode_loop.h"     // shared prefill + generation loops; needs llama.h
    #include "decode_loop_mtp.h" // MTP self-speculation (sidecar head, patch 0012)
    #include "llama_gpu.h"
    #include "sampler_chain.h" // shared sampler chain (#125); needs llama.h
    #include "xllama/llama_raii.h"

    #include <vector>

namespace xllama {

class LlamaSession final : public Session {
  public:
    LlamaModelPtr m_model;
    LlamaAdapterLoraPtr m_adapter; // optional runtime LoRA (freed before model)
    float m_lora_scale = 1.0f;
    int m_n_ctx;
    int m_n_threads;
    int m_n_batch;  // 0 = llama.cpp default
    int m_n_ubatch; // 0 = llama.cpp default
    // Persistent context across turns: the KV cache lives here so a reuse turn
    // (reuse_kv && !reset_kv) can append only the delta instead of re-prefilling
    // the whole conversation. Created lazily on the first generate().
    GgufCpuThreadpools m_cpu_pools; // declared before m_ctx: must outlive it
    LlamaContextPtr m_ctx;
    // MTP self-speculation (decode_loop_mtp.h): the head's own context, the
    // common/speculative state driving it, and the loop's handle on both.
    // Declared after m_ctx so they go first — the drafter reads the target's
    // hidden rows. m_mtp_n = drafts per round (XLLAMA_MTP_DRAFT, 0 = off).
    int m_mtp_n = 0;
    float m_mtp_p_min = 0.0f;
    LlamaContextPtr m_mtp_ctx;
    common_speculative_ptr m_mtp_spec;
    MtpDrafter m_mtp;
    // #175 decision: sampler state follows the KV lifecycle, mirroring the ORT
    // persistent generator — the penalty window (and the dist RNG) live as long
    // as the conversation, so the tail of the previous reply stays penalized at
    // the start of the next turn. Rebuilt on reset_kv or a sampling change
    // (same_chain guard). The 64-token window itself is a deliberate divergence
    // from ORT's whole-sequence penalty — see sampling.h.
    LlamaSamplerPtr m_sampler;
    SamplingConfig m_sampler_cfg;

    // #170a: the tokens whose KV is currently resident (prefilled + generated,
    // in position order). A full-prompt turn diffs its tokenization against
    // this and rewinds only the divergent tail (llama_memory_seq_rm) instead
    // of clearing — regenerate, an edited last message, and LAN-API requests
    // that extend the previous one re-prefill only the difference. Cleared
    // whenever the KV state stops being trustworthy (decode failure).
    std::vector<llama_token> m_kv_tokens;

    bool m_kv_q8 = false;         // #171: q8_0 KV + forced flash attention
    bool m_prompt_lookup = false; // #210: draft-free speculative decoding
    int m_gpu_layers = 0;         // layers on the d3d12 backend (llama_gpu.h)

    // #169: whether the resident KV supports front-drop eviction + RoPE shift.
    // Known once the lazy context exists. Gated on llama_memory_can_shift —
    // seq_add on an unsupported arch (mrope) is a GGML_ASSERT abort, not an
    // error — and on the model not using SWA, where the resident window is
    // not [0, kv_len) and the shift arithmetic below would lie.
    bool m_can_shift = false;

    // #170c: a hybrid (or recurrent) cache refuses #170a's partial tail erase —
    // the recurrent state has no per-token history — so every LAN-API turn,
    // regenerate and edit re-prefilled from token 0. On the console (Qwen3.6
    // 35B-A3B, Gated DeltaNet + attention) that was 48-53 tokens, 8-10 s, per
    // request. Upstream's server answers with context checkpoints and so does
    // this: snapshot only the recurrent half (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY,
    // 63 MiB there whatever the prompt length) at the points a later prompt
    // tends to diverge after, and when a rewind is refused restore the longest
    // one inside the common prefix — the attention half then rewinds normally.
    // A checkpoint names the history it absorbed, so its validity is a prefix
    // compare against the new prompt, not bookkeeping across clears.
    struct RsCheckpoint {
        std::vector<llama_token> tokens; // the history the state absorbed
        std::vector<uint8_t> state;      // llama_state_seq_get_data_ext, PARTIAL_ONLY
    };
    // 2 x 63 MiB on the console, whose heap has ~0.5 GB free with the model in.
    static constexpr size_t kRsCheckpoints = 2;
    // Heap a new checkpoint must leave free (XLLAMA_RS_CKPT_RESERVE_MB, default 384):
    // the 122B's are 149 MiB each, and the console's heap has less to spare there.
    static size_t rs_ckpt_reserve_mb() {
        static const size_t v = [] {
            const char* s = std::getenv("XLLAMA_RS_CKPT_RESERVE_MB");
            return s && *s ? static_cast<size_t>(std::strtoul(s, nullptr, 10)) : size_t(384);
        }();
        return v;
    }
    // A turn's last 4 tokens are what the next request re-renders differently:
    // Qwen's "<think>\n\n</think>\n\n" generation prompt is dropped from the
    // history (upstream checkpoints at the same offset).
    static constexpr size_t kRsCheckpointTail = 4;
    // A divergence point is kept only when it spares at least this much prefill.
    static constexpr size_t kRsCheckpointMinGain = 16;
    std::vector<RsCheckpoint> m_rs_ckpts; // least recently used first
    bool m_rs_ckpt_enabled = false;       // hybrid or recurrent memory (ensure_ctx)
    // After a #169 shift the recurrent state still holds the evicted history, so
    // a checkpoint taken then carries the approximation the resident cache
    // already serves #170a's pure extensions with — no stricter, no looser.

    // Prefix cache on disk (XLLAMA_PREFIX_CACHE_MIB, off by default). The
    // #170c checkpoints hold the recurrent half only, so they help a request
    // that shares its prefix with the RESIDENT history. The LAN API serves
    // several clients through one session, and their conversations interleave
    // (A, B, A'): B overwrites A's attention KV, and A' re-prefilled from token
    // 0 — 27.8 s for a 1500-token code prompt on the console. At the same
    // split points as the checkpoints, a full-prompt prefill of a request that
    // opts in (GenerateParams::save_prefix: the file holds the prompt's token
    // history) also writes the FULL sequence state (attention + recurrent) to
    // a file, streamed by llama.cpp with no whole-state heap buffer; a later
    // full prompt (opted in or not) restores the longest saved prefix that
    // beats the in-memory paths. Each session writes only its own directory,
    // removed with the session, so a file is never read by another model or
    // context shape.
    uint64_t m_pc_budget = 0; // bytes on disk; 0 = off
    size_t m_pc_min = 256;    // shortest history worth a file (XLLAMA_PREFIX_CACHE_MIN)
    std::string m_pc_dir;     // this session's directory, created by the first save
    std::string m_pc_fp;      // state_fingerprint() the files were written under
    PrefixCacheIndex m_pc;
    std::chrono::seconds m_pc_ttl{1800}; // XLLAMA_PREFIX_CACHE_TTL_S
    std::string m_pc_root;

    explicit LlamaSession(LlamaModelPtr model, LlamaAdapterLoraPtr adapter, float lora_scale,
                          int n_ctx, int n_threads, int n_batch, int n_ubatch, bool kv_q8,
                          bool prompt_lookup, int gpu_layers, int mtp_n = 0, float mtp_p_min = 0.0f)
        : m_model(std::move(model)), m_adapter(std::move(adapter)), m_lora_scale(lora_scale),
          m_n_ctx(n_ctx), m_n_threads(n_threads), m_n_batch(n_batch), m_n_ubatch(n_ubatch),
          m_mtp_n(mtp_n), m_mtp_p_min(mtp_p_min), m_kv_q8(kv_q8), m_prompt_lookup(prompt_lookup),
          m_gpu_layers(gpu_layers) {}

    ~LlamaSession() override {
        drop_prefix_files();
    }

    const void* resident_llama_model() const override {
        return m_model.get();
    }

    // Lazy context creation, shared by generate() and the state-file entry
    // points (#170b needs a context before the first turn). Returns false and
    // sets *err on failure; m_ctx stays null.
    bool ensure_ctx(std::string* err, bool for_embedding = false) {
        if (m_ctx)
            return true;
        {
            llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx = m_n_ctx;
            // XLLAMA_CPU_THREADS_TG (catalogue env, 1..n_threads): threads of a
            // one-token decode graph; with GPU layers its CPU splits are the
            // MoE router + expert reads and a few tiny ops, where every node
            // pays a barrier across the pool.
            const char* tg = std::getenv("XLLAMA_CPU_THREADS_TG");
            cparams.n_threads = tg && *tg ? std::clamp(std::atoi(tg), 1, m_n_threads) : m_n_threads;
            // Prefill (any ubatch > 1 token) runs on n_threads_batch, whose
            // default is GGML_DEFAULT_N_THREADS (4) regardless of n_threads —
            // left unset it caps prefill at 4 threads while decode gets 6 (#168).
            cparams.n_threads_batch = m_n_threads;
            if (m_n_batch > 0)
                cparams.n_batch = static_cast<uint32_t>(m_n_batch);
            if (m_n_ubatch > 0)
                cparams.n_ubatch = static_cast<uint32_t>(m_n_ubatch);
            // MTP: the model must carry the head (patch 0012 sidecar), and the
            // recurrent layers keep a snapshot per draft so a rejected tail rolls
            // back (llama.cpp n_rs_seq; the verify batch is 1 + m_mtp_n rows).
            const bool mtp_on =
                m_mtp_n > 0 && !for_embedding && llama_model_n_layer_nextn(m_model.get()) > 0;
            if (m_mtp_n > 0 && !for_embedding && !mtp_on)
                log_output("[xllama] mtp: XLLAMA_MTP_DRAFT set but the model has no MTP "
                           "layer (XLLAMA_MTP_SIDECAR?) — off\n");
            uint32_t outputs_max = gguf_gpu_outputs_max(for_embedding, m_prompt_lookup);
            if (mtp_on) {
                cparams.n_rs_seq = static_cast<uint32_t>(m_mtp_n);
                outputs_max = std::max(outputs_max, 1u + static_cast<uint32_t>(m_mtp_n));
                // llama-batch asserts n_ubatch > n_rs_seq + 1 (the kept tail): a
                // catalogue n_ubatch below that would abort the first prefill.
                cparams.n_ubatch = std::max(cparams.n_ubatch, cparams.n_rs_seq + 2);
            }
            apply_gguf_gpu_context(m_gpu_layers, cparams, outputs_max);
    #ifdef XLLAMA_UWP
            // The session's decode loop asks for the same rows on the CPU, so
            // the CPU path takes the cap too: the default reserve is n_ubatch
            // logits rows, 512 x 248320 x 4 B = 529 MiB on Qwen3.5, which kept
            // a 4.2 GB IQ3_M from getting a context under the Series X's
            // 4.86 GB heap (2026-11-09).
            if (m_gpu_layers <= 0) {
                if (outputs_max > 0)
                    cparams.n_outputs_max = outputs_max;
            }
    #endif
            if (for_embedding) {
                // Non-causal embedding graphs cannot split a sequence into
                // physical micro-batches (llama.cpp asserts instead of failing).
                cparams.n_batch = std::min(cparams.n_batch, cparams.n_ctx);
                cparams.n_ubatch = cparams.n_batch;
                cparams.embeddings = true;
            }
            const llama_flash_attn_type fa_requested = cparams.flash_attn_type;
            if (m_kv_q8) {
                // #171: quantized V requires flash attention (the pin throws at
                // context creation with FA disabled, and AUTO may resolve to
                // disabled) — force it and fall back below if the arch refuses.
                cparams.type_k = GGML_TYPE_Q8_0;
                cparams.type_v = GGML_TYPE_Q8_0;
                cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            }
            m_ctx.reset(llama_init_from_model(m_model.get(), cparams));
            if (!m_ctx && m_kv_q8) {
                log_output("[xllama] session: q8_0 KV context failed — falling back "
                           "to default cache types (#171)\n");
                cparams.type_k = llama_context_default_params().type_k;
                cparams.type_v = llama_context_default_params().type_v;
                cparams.flash_attn_type = fa_requested;
                m_kv_q8 = false;
                m_ctx.reset(llama_init_from_model(m_model.get(), cparams));
            }
            m_cpu_pools.attach(m_gpu_layers, m_ctx.get(), static_cast<int>(cparams.n_threads),
                               static_cast<int>(cparams.n_threads_batch));
            log_gguf_ws("after context");
            if (!m_ctx) {
                if (err)
                    *err = "failed to create context";
                log_output("[xllama] session: failed to create context\n");
                return false;
            }
            m_can_shift = llama_memory_can_shift(llama_get_memory(m_ctx.get())) &&
                          llama_model_n_swa(m_model.get()) == 0;
            // Logged because it is a per-ARCH capability that decides whether a long
            // chat shifts (#169) or fail-fasts (#173), and docs used to state it from
            // assumption: imrope and SWA architectures cannot shift, and which is
            // which is not something to guess per model. One line makes it a
            // measurement anyone can read off the device log.
            {
                char cb[128];
                snprintf(cb, sizeof(cb),
                         "[xllama] session: can_shift=%d (memory_can_shift=%d n_swa=%d)\n",
                         m_can_shift ? 1 : 0,
                         llama_memory_can_shift(llama_get_memory(m_ctx.get())) ? 1 : 0,
                         (int)llama_model_n_swa(m_model.get()));
                log_output(cb);
            }
            m_rs_ckpt_enabled =
                llama_model_is_hybrid(m_model.get()) || llama_model_is_recurrent(m_model.get());
            m_rs_ckpts.clear();
            // Apply runtime LoRA after context exists (llama_set_adapters_lora is
            // context-scoped). Must re-apply if context is ever recreated.
            if (m_adapter) {
                llama_adapter_lora* arr[1] = {m_adapter.get()};
                float scales[1] = {m_lora_scale};
                if (llama_set_adapters_lora(m_ctx.get(), arr, 1, scales) != 0) {
                    if (err)
                        *err = "llama_set_adapters_lora failed";
                    log_output("[xllama] session: set_adapters_lora failed\n");
                    m_ctx.reset();
                    return false;
                }
                log_output("[xllama] session: runtime LoRA adapter applied\n");
            }
            if (mtp_on)
                init_mtp(cparams);
        }
        return true;
    }

    // The MTP head's context: same model, LLAMA_CONTEXT_TYPE_MTP (llama.cpp
    // builds only the nextn block there, with a plain KV cache of its own), as
    // many positions as the target. One logits row: draft() decodes one token,
    // the catch-up mirror none. A failure leaves the session drafting-free.
    void init_mtp(llama_context_params cparams) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cparams.n_rs_seq = 0;
        cparams.ctx_other = m_ctx.get();
        cparams.n_ctx = llama_n_ctx(m_ctx.get());
        cparams.n_outputs_max = 1;
        cparams.embeddings = false;
        m_mtp_ctx.reset(llama_init_from_model(m_model.get(), cparams));
        if (!m_mtp_ctx) {
            log_output("[xllama] mtp: draft context failed — drafting off (target keeps "
                       "n_rs_seq " +
                       std::to_string(m_mtp_n) + ")\n");
            return;
        }
        m_cpu_pools.attach(m_gpu_layers, m_mtp_ctx.get(), static_cast<int>(cparams.n_threads),
                           static_cast<int>(cparams.n_threads_batch));
        common_params_speculative sp;
        sp.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        sp.draft.n_max = m_mtp_n;
        sp.draft.n_min = 0;
        sp.draft.p_min = m_mtp_p_min;
        sp.draft.backend_sampling = false;
        sp.draft.ctx_tgt = m_ctx.get();
        sp.draft.ctx_dft = m_mtp_ctx.get();
        m_mtp_spec.reset(common_speculative_init(sp, 1));
        if (!m_mtp_spec) {
            log_output("[xllama] mtp: common_speculative_init failed — drafting off\n");
            m_mtp_ctx.reset();
            return;
        }
        m_mtp.ctx_dft = m_mtp_ctx.get();
        m_mtp.spec = m_mtp_spec.get();
        m_mtp.n_max = m_mtp_n;
        m_mtp.broken = false;
        m_mtp.mirror = common_batch(m_ctx.get());
        m_mtp.verify = common_batch(m_ctx.get());
        log_gguf_ws("after mtp context");
        char cb[160];
        snprintf(cb, sizeof(cb),
                 "[xllama] mtp: drafter ready — %d draft/round, p_min %.2f, n_rs_seq %u, "
                 "n_ctx %u\n",
                 m_mtp_n, (double)m_mtp_p_min, llama_n_rs_seq(m_ctx.get()),
                 llama_n_ctx(m_mtp_ctx.get()));
        log_output(cb);
    }

    // #170c: the longest checkpoint whose history is a prefix of
    // tokens[0, limit); m_rs_ckpts.size() when none fits.
    size_t best_rs_checkpoint(const std::vector<llama_token>& tokens, size_t limit) const {
        size_t best = m_rs_ckpts.size();
        for (size_t i = 0; i < m_rs_ckpts.size(); ++i) {
            const std::vector<llama_token>& h = m_rs_ckpts[i].tokens;
            if (h.empty() || h.size() > limit ||
                (best < m_rs_ckpts.size() && h.size() <= m_rs_ckpts[best].tokens.size()))
                continue;
            if (std::equal(h.begin(), h.end(), tokens.begin()))
                best = i;
        }
        return best;
    }

    // #170c: restore the longest checkpoint whose history is a prefix of
    // tokens[0, limit) and erase the attention tail past it. Returns the
    // restored length, 0 when none fits or the restore failed — then the
    // recurrent half may be half-written and the caller must clear the cache.
    size_t restore_rs_checkpoint(llama_context* ctx, const std::vector<llama_token>& tokens,
                                 size_t limit) {
        const size_t best = best_rs_checkpoint(tokens, limit);
        if (best == m_rs_ckpts.size())
            return 0;
        // Most recently used goes last; the restored state is the newest use.
        std::rotate(m_rs_ckpts.begin() + best, m_rs_ckpts.begin() + best + 1, m_rs_ckpts.end());
        const RsCheckpoint& ck = m_rs_ckpts.back();
        const size_t n = ck.tokens.size();
        if (llama_state_seq_set_data_ext(ctx, ck.state.data(), ck.state.size(), 0,
                                         LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0)
            return 0;
        // The recurrent half now ends at n-1, so this erase is the attention
        // tail only — the same one a pure-attention cache does in #170a.
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, static_cast<llama_pos>(n), -1))
            return 0;
        return n;
    }

    // #170c: snapshot the recurrent half after history[0, n). A checkpoint of
    // the same history is only marked used (a restored turn re-reaches it); the
    // least recently used one lends its buffer. An allocation failure drops the
    // checkpoints, never the request.
    void save_rs_checkpoint(llama_context* ctx, const llama_token* history, size_t n) {
        for (size_t i = 0; i < m_rs_ckpts.size(); ++i) {
            const std::vector<llama_token>& h = m_rs_ckpts[i].tokens;
            if (h.size() == n && std::equal(h.begin(), h.end(), history)) {
                std::rotate(m_rs_ckpts.begin() + i, m_rs_ckpts.begin() + i + 1, m_rs_ckpts.end());
                return;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        const size_t size =
            llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (size == 0)
            return;
        RsCheckpoint ck;
        if (m_rs_ckpts.size() >= kRsCheckpoints) {
            ck = std::move(m_rs_ckpts.front());
            m_rs_ckpts.erase(m_rs_ckpts.begin());
        }
        // A new buffer only with heap to spare after it: the prefill and decode that follow
        // allocate too, and a bad_alloc there ends the app (chain94, 122B: 2 x 149 MiB held, the
        // next 2327-token prefill died with 0x8007000E). A lent buffer costs nothing. The
        // smaller of the two headrooms decides: chain96 failed a 149 MiB buffer with avail_phys
        // at 879 MB, so the commit limit is the one heap allocations meet.
        if (ck.state.capacity() < size) {
            const size_t phys = avail_phys_mb();
            const size_t commit = avail_commit_mb();
            const size_t avail = phys == 0 ? commit : commit == 0 ? phys : std::min(phys, commit);
            const size_t need = size / (1024 * 1024) + rs_ckpt_reserve_mb();
            if (avail != 0 && avail < need) {
                char sb[224];
                snprintf(
                    sb, sizeof(sb),
                    "[xllama] session: KV checkpoint at %zu tokens skipped — avail_phys %zu MB, "
                    "commit_avail %zu MB < %zu MB, %zu held (#170c)\n",
                    n, phys, commit, need, m_rs_ckpts.size());
                log_output(sb);
                return;
            }
        }
        try {
            ck.state.resize(size);
            ck.tokens.assign(history, history + n);
        } catch (const std::bad_alloc&) {
            m_rs_ckpts.clear();
            m_rs_ckpts.shrink_to_fit();
            char fb[224];
            snprintf(fb, sizeof(fb),
                     "[xllama] session: KV checkpoint allocation failed — checkpoints dropped "
                     "(#170c; private %zu MB, avail_phys %zu MB, commit_avail %zu MB)\n",
                     private_mb(), avail_phys_mb(), avail_commit_mb());
            log_output(fb);
            return;
        }
        if (llama_state_seq_get_data_ext(ctx, ck.state.data(), size, 0,
                                         LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != size) {
            log_output("[xllama] session: KV checkpoint snapshot failed (#170c)\n");
            return;
        }
        m_rs_ckpts.push_back(std::move(ck));
        char cb[192];
        snprintf(cb, sizeof(cb),
                 "[xllama] session: KV checkpoint at %zu tokens — %.1f MiB in %.1f ms, %zu held "
                 "(#170c)\n",
                 n, (double)size / (1024.0 * 1024.0),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                     .count(),
                 m_rs_ckpts.size());
        log_output(cb);
    }

    // Prefix cache: the knobs, read once at creation (after the catalogue env
    // is applied). XLLAMA_PREFIX_CACHE_DIR names the root; on the console it
    // defaults to LocalState\prefix-cache, on the host the cache is off without
    // it. The first session of a process to use a root wipes it (what an earlier
    // process wrote never reaches a new one); later sessions keep the files of
    // the live ones. Each session then takes its own "pc-<n>" directory.
    void init_prefix_cache() {
        const char* mib = std::getenv("XLLAMA_PREFIX_CACHE_MIB");
        const uint64_t budget = mib && *mib ? std::strtoull(mib, nullptr, 10) << 20 : 0;
        if (budget == 0)
            return;
        const char* dir = std::getenv("XLLAMA_PREFIX_CACHE_DIR");
        std::string root = dir && *dir ? dir : "";
    #ifdef XLLAMA_UWP
        if (root.empty())
            root = resolve_local_path("prefix-cache");
    #endif
        if (root.empty()) {
            log_output("[xllama] session: prefix cache off — XLLAMA_PREFIX_CACHE_DIR unset\n");
            return;
        }
        if (const char* s = std::getenv("XLLAMA_PREFIX_CACHE_MIN"); s && *s)
            m_pc_min = std::max<size_t>(std::strtoul(s, nullptr, 10), kRsCheckpointMinGain);
        m_pc_root = root;
        m_pc_ttl = prefix_cache_ttl(std::getenv("XLLAMA_PREFIX_CACHE_TTL_S"));
        static std::mutex wiped_mu;
        static std::set<std::string> wiped;
        bool first_use = false;
        {
            std::lock_guard<std::mutex> lk(wiped_mu);
            first_use = wiped.insert(root).second;
        }
        if (first_use) {
            const size_t removed = wipe_prefix_cache_files(root);
            log_output("[xllama] session: prefix cache start wipe — " + std::to_string(removed) +
                       " files removed\n");
        }
        static std::atomic<unsigned> serial{0};
        m_pc_dir = root + "/pc-" + std::to_string(++serial);
        m_pc_budget = budget;
        char lb[96];
        snprintf(lb, sizeof(lb), "[xllama] session: prefix cache on — %llu MiB, min %zu tokens, ",
                 static_cast<unsigned long long>(budget >> 20), m_pc_min);
        log_output(lb + m_pc_dir + "\n");
    }

    void drop_prefix_files() {
        m_pc.clear();
        if (m_pc_dir.empty())
            return;
        std::error_code ec;
        std::filesystem::remove_all(m_pc_dir, ec);
    }

    // Prefix cache: write the full state of history[0, n), at a #170c split
    // point of a full-prompt prefill (the cells there are exactly that
    // history). An identical history already on disk is only marked used. A
    // failed write turns the cache off for the session: on the console that
    // is a full disk, and every later request would pay the failing write.
    void save_prefix_cache(llama_context* ctx, const llama_token* history, size_t n) {
        if (m_pc_budget == 0)
            return;
        prune_prefix_cache_files(m_pc_root, m_pc_ttl);
        m_pc.prune(m_pc_dir, m_pc_ttl);
        if (n < m_pc_min || llama_n_rs_seq(ctx) > 0)
            return;
        const std::string fp = state_fingerprint();
        if (fp != m_pc_fp) {
            drop_prefix_files();
            m_pc_fp = fp;
        }
        if (const int i = m_pc.find_exact(history, n); i >= 0) {
            m_pc.touch(static_cast<size_t>(i));
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const std::string name = PrefixCacheIndex::file_name(fp, history, n);
        const std::string path = m_pc_dir + "/" + name;
        const std::string tmp = path + ".tmp";
        std::error_code ec;
        std::filesystem::create_directories(m_pc_dir, ec);
        // A rename, so a crash never leaves a half file under a live name.
        const size_t bytes = ec ? 0 : llama_state_seq_save_file(ctx, tmp.c_str(), 0, history, n);
        if (bytes > 0)
            std::filesystem::rename(tmp, path, ec);
        if (bytes == 0 || ec) {
            std::filesystem::remove(tmp, ec);
            drop_prefix_files();
            m_pc_budget = 0;
            char fb[128];
            snprintf(fb, sizeof(fb),
                     "[xllama] session: prefix cache save at %zu tokens failed — off for this "
                     "session\n",
                     n);
            log_output(fb);
            return;
        }
        PrefixCacheEntry e;
        e.tokens.assign(history, history + n);
        e.file = name;
        e.bytes = bytes;
        for (const std::string& f : m_pc.add(std::move(e), m_pc_budget))
            std::filesystem::remove(m_pc_dir + "/" + f, ec);
        char cb[192];
        snprintf(cb, sizeof(cb),
                 "[xllama] session: prefix cache save — %zu tokens, %.1f MiB in %.1f ms, %zu files "
                 "/ %.1f MiB held\n",
                 n, (double)bytes / (1024.0 * 1024.0),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                     .count(),
                 m_pc.entries().size(), (double)m_pc.total_bytes() / (1024.0 * 1024.0));
        log_output(cb);
    }

    // Prefix cache: on a full prompt, restore the longest saved history that
    // is a prefix of tokens[0, size-1) when it keeps at least
    // kRsCheckpointMinGain more tokens than the in-memory paths would — all of
    // the common prefix on a pure extension, else the best #170c checkpoint
    // inside it. Returns the restored length (m_kv_tokens is that history).
    // 0 leaves the cache untouched, except after a refused load: then the
    // cache is cleared and m_kv_tokens empty, and the request prefills in full.
    size_t restore_prefix_cache(llama_context* ctx, const std::vector<llama_token>& tokens,
                                size_t common) {
        if (m_pc_budget == 0)
            return 0;
        m_pc.prune(m_pc_dir, m_pc_ttl);
        if (m_pc.entries().empty() || tokens.size() < 2 || llama_n_rs_seq(ctx) > 0 ||
            state_fingerprint() != m_pc_fp)
            return 0;
        size_t keep = common;
        if (common < m_kv_tokens.size()) {
            const size_t ck = best_rs_checkpoint(tokens, common);
            keep = ck < m_rs_ckpts.size() ? m_rs_ckpts[ck].tokens.size() : 0;
        }
        const int found = m_pc.find(tokens.data(), tokens.size() - 1, keep + kRsCheckpointMinGain);
        if (found < 0)
            return 0;
        const auto t0 = std::chrono::steady_clock::now();
        const size_t i = static_cast<size_t>(found);
        const PrefixCacheEntry& e = m_pc.entries()[i];
        const std::string path = m_pc_dir + "/" + e.file;
        std::error_code ec;
        const uintmax_t on_disk = std::filesystem::file_size(path, ec);
        std::vector<llama_token> got;
        size_t n_got = 0, read = 0;
        if (!ec && on_disk == e.bytes) {
            got.resize(e.tokens.size());
            llama_memory_clear(llama_get_memory(ctx), true);
            m_kv_tokens.clear();
            read = llama_state_seq_load_file(ctx, path.c_str(), 0, got.data(), got.size(), &n_got);
        }
        if (read != e.bytes || n_got != e.tokens.size() ||
            !std::equal(got.begin(), got.end(), e.tokens.begin())) {
            char fb[160];
            snprintf(fb, sizeof(fb),
                     "[xllama] session: prefix cache restore of %zu tokens refused — file "
                     "dropped, %s\n",
                     e.tokens.size(), got.empty() ? "size mismatch" : "full re-prefill");
            log_output(fb);
            std::filesystem::remove(path, ec);
            m_pc.remove(i);
            if (!got.empty())
                llama_memory_clear(llama_get_memory(ctx), true);
            return 0;
        }
        m_pc.touch(i);
        m_kv_tokens = std::move(got);
        // The file holds the target's cells only: the drafter restarts empty.
        m_mtp.keep_prefix(0);
        char cb[192];
        snprintf(cb, sizeof(cb),
                 "[xllama] session: prefix cache restore — %zu of %zu prompt tokens from disk in "
                 "%.1f ms (in memory: %zu)\n",
                 m_kv_tokens.size(), tokens.size(),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                     .count(),
                 keep);
        log_output(cb);
        return m_kv_tokens.size();
    }

    InferenceResult generate(const GenerateParams& gp) override {
        InferenceResult res;

        if (!ensure_ctx(&res.error_msg))
            return res;
        llama_context* ctx = m_ctx.get();

        // KV-cache reuse: a continuation turn (reuse_kv && !reset_kv) keeps the
        // existing cache and appends the delta; otherwise re-prefill the full
        // prompt — rewinding to the common token prefix when one is resident
        // (#170a) instead of always clearing. add_bos only on a full prompt —
        // a delta must not carry a BOS mid-conversation.
        const bool full_prompt = gp.reset_kv || !gp.reuse_kv;

        const llama_vocab* vocab = llama_model_get_vocab(m_model.get());

        // Scoring: a forced reply is checked before any prefill is paid for.
        std::vector<llama_token> forced;
        if (!gp.forced_tokens.empty()) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            forced.reserve(gp.forced_tokens.size());
            for (const int32_t id : gp.forced_tokens) {
                if (id < 0 || id >= n_vocab) {
                    res.error_msg = "forced token " + std::to_string(id) +
                                    " outside the vocabulary of " + std::to_string(n_vocab);
                    return res;
                }
                forced.push_back(static_cast<llama_token>(id));
            }
        }

        // parse_special=true: every Session caller (chat UI, API endpoint, bench)
        // passes a TEMPLATED prompt, whose <|im_start|>/<|im_end|> markers must
        // become the special ids the model was trained on. As plain text the
        // model sees an alien template and its next-token distribution goes flat
        // (LFM2.5: top-2 gap 0.52 vs 3.91 with specials) — flat enough that the
        // Zen2-vs-desktop kernel jitter (max |Δlogit| ~1.6 measured) flipped the
        // greedy token on-device ("User\n\n<|end|>" instead of a greeting).
        int32_t n_tokens =
            llama_tokenize(vocab, gp.prompt.c_str(), static_cast<int32_t>(gp.prompt.size()),
                           nullptr, 0, full_prompt, true);
        if (n_tokens == INT32_MIN) {
            res.error_msg = "tokenize overflow";
            return res;
        }
        // Size query returns the negated required count (see llama.h) — not an error.
        n_tokens = -n_tokens;

        std::vector<llama_token> tokens(static_cast<size_t>(n_tokens));
        n_tokens = llama_tokenize(vocab, gp.prompt.c_str(), static_cast<int32_t>(gp.prompt.size()),
                                  tokens.data(), n_tokens, full_prompt, true);
        if (n_tokens < 0) {
            res.error_msg = "tokenize failed";
            return res;
        }
        tokens.resize(static_cast<size_t>(n_tokens));

        // #170a: on a full prompt, rewind the resident KV to the common token
        // prefix and prefill only the divergent tail. Capped at size-1 so at
        // least one token is prefilled — the last token's logits seed the first
        // sample. Regenerate keeps everything but the final token; an edited
        // last message keeps everything before the edit; an unrelated prompt
        // diverges immediately and degrades to the old full clear.
        llama_memory_t mem = llama_get_memory(ctx);
        size_t kv_keep = 0;
        // #170c: where a refused rewind diverged, snapped back to just after the
        // last control token before it (a turn marker: <|im_start|>,
        // <|start_header_id|>...). Calls sharing a system prompt diverge inside
        // the user turn, each at its own word ("Which planet is" / "Which
        // planet has"); the turn boundary is what the next one shares too.
        size_t rs_common = 0;
        if (full_prompt) {
            const size_t max_keep = tokens.empty() ? 0 : tokens.size() - 1;
            while (kv_keep < m_kv_tokens.size() && kv_keep < max_keep &&
                   m_kv_tokens[kv_keep] == tokens[kv_keep])
                ++kv_keep;
            // Prefix cache: a saved prefix longer than the in-memory paths keep
            // replaces the resident history; a refused load emptied it.
            const size_t from_disk = restore_prefix_cache(ctx, tokens, kv_keep);
            res.n_prefix_cache_tokens = static_cast<int>(from_disk);
            kv_keep = std::min(kv_keep, m_kv_tokens.size());
            if (from_disk > 0) {
                kv_keep = from_disk;
            } else if (kv_keep > 0) {
                // Pure extension (no divergent tail) needs no removal. When a
                // tail must go, honor seq_rm's result: hybrid caches (LFM2's
                // attn+recurrent) refuse a partial tail erase — the recurrent
                // state cannot be rewound — and mutate NOTHING on failure, so
                // decoding on top would corrupt the output. Degrade to a full
                // clear + re-prefill instead.
                // With MTP the recurrent half keeps n_rs_seq rollback snapshots,
                // and llama.cpp accepts any rollback up to that depth — but the
                // snapshots are only those the LAST ubatch wrote (a one-token
                // decode leaves the older slots stale), so a prefix rewind of
                // 1..n_rs_seq would resume from a wrong state with no error.
                // Only the verify loop rolls back through them; here the
                // checkpoint path stays the one way back, as before MTP.
                if (kv_keep < m_kv_tokens.size() &&
                    (llama_n_rs_seq(ctx) > 0 ||
                     !llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(kv_keep), -1))) {
                    rs_common = kv_keep;
                    while (rs_common > 0 && !llama_vocab_is_control(vocab, tokens[rs_common - 1]))
                        --rs_common;
                    if (rs_common == 0)
                        rs_common = kv_keep; // plain text: no turn marker to snap to
                    const size_t restored = restore_rs_checkpoint(ctx, tokens, kv_keep);
                    char rb[256];
                    if (restored > 0) {
                        snprintf(rb, sizeof(rb),
                                 "[xllama] session: KV checkpoint restore — kept %zu of %zu "
                                 "tokens (common=%zu resident=%zu) (#170c)\n",
                                 restored, tokens.size(), kv_keep, m_kv_tokens.size());
                        kv_keep = restored;
                    } else {
                        snprintf(rb, sizeof(rb),
                                 "[xllama] session: KV rewind unsupported (hybrid cache) — "
                                 "full re-prefill (#170): common=%zu resident=%zu prompt=%zu "
                                 "checkpoints=%zu\n",
                                 kv_keep, m_kv_tokens.size(), tokens.size(), m_rs_ckpts.size());
                        llama_memory_clear(mem, true);
                        kv_keep = 0;
                    }
                    log_output(rb);
                } else {
                    char pb[128];
                    snprintf(pb, sizeof(pb),
                             "[xllama] session: KV prefix reuse — kept %zu of %zu tokens (#170)\n",
                             kv_keep, tokens.size());
                    log_output(pb);
                }
            } else {
                // #216: when a #170b restore just landed, "restored" + full
                // re-prefill means the saved token list and the re-rendered
                // prompt share no common prefix — log enough to diagnose.
                if (!m_kv_tokens.empty()) {
                    char pb[192];
                    snprintf(pb, sizeof(pb),
                             "[xllama] session: KV prefix reuse none — resident %zu tok, "
                             "prompt %zu tok (first mismatch at 0) (#170)\n",
                             m_kv_tokens.size(), tokens.size());
                    log_output(pb);
                }
                llama_memory_clear(mem, true);
            }
            m_kv_tokens.resize(kv_keep);
        }

        // #173: the KV length is exact and free to read (seq_pos_max is -1 on
        // an empty cache). Predict the overflow — KV + delta + at least one
        // generated token — and fail before the decode attempt; the caller
        // retries with the trimmed full prompt. The same arithmetic clamps the
        // generation loop below so hitting the context end is a clean stop,
        // not a mid-generation decode failure.
        int kv_len = full_prompt ? static_cast<int>(kv_keep)
                                 : static_cast<int>(llama_memory_seq_pos_max(mem, 0)) + 1;
        // The slice that is not yet resident: the divergent tail on a full
        // prompt, the whole delta on a continuation.
        llama_token* pf = tokens.data() + (full_prompt ? kv_keep : 0);
        const int n_pf = static_cast<int>(tokens.size()) - (full_prompt ? (int)kv_keep : 0);
        if (n_pf <= 0) {
            // Both callers always render at least one token (a delta carries the
            // turn close + user turn; a full prompt keeps size-1 at most), so
            // this is a programming error, not a user-reachable state. Fail
            // rather than sample on whatever logits the last decode left.
            res.error_msg = "nothing to prefill";
            log_output("[xllama] session generate: nothing to prefill\n");
            return res;
        }
        if (full_prompt && kv_len + n_pf + 1 > m_n_ctx) {
            // A full prompt has nothing older to evict — the UI trimmer
            // (kMaxPromptTokens / max_prompt_tokens_for_n_ctx) owns that, and
            // its char-based estimate can undershoot on dense code or CJK.
            // Fail here: reaching llama_decode with more tokens than the
            // context is not an error code but a GGML_ASSERT abort.
            res.error_msg = "prompt too long: " + std::to_string(n_pf) +
                            " tokens exceed n_ctx=" + std::to_string(m_n_ctx) +
                            " — shorten the message or start a new chat";
            log_output(("[xllama] session generate: " + res.error_msg + "\n").c_str());
            return res;
        }
        if (!full_prompt && kv_len + n_pf + 1 > m_n_ctx) {
            // #169: context shift — evict the oldest tokens past the pinned
            // head (gp.n_keep, the system prompt) and RoPE-shift the survivors
            // down, so a long chat stays in the reuse regime instead of
            // falling back to trimmed ~n_ctx re-prefills every turn. Eviction
            // is upstream-style (server-context.cpp): at least half the
            // movable region, or more when the delta + requested generation
            // need it. The recurrent half of a hybrid cache holds no cells in
            // the evicted range (its absorbed history survives — a documented
            // approximation), so the front-drop seq_rm succeeds where #170a's
            // tail rewind cannot.
            bool shifted = false;
            if (m_can_shift && kv_len > 0 && static_cast<int>(m_kv_tokens.size()) == kv_len) {
                const int keep = std::min(std::max(gp.n_keep, 0), kv_len - 1);
                const int movable = kv_len - keep;
                const int need = kv_len + n_pf + std::max(gp.n_predict, 1) - m_n_ctx;
                const int n_discard = std::min(std::max(need, movable / 2), movable);
                if (n_discard > 0 && kv_len + n_pf + 1 - n_discard <= m_n_ctx &&
                    llama_memory_seq_rm(mem, 0, keep, keep + n_discard)) {
                    llama_memory_seq_add(mem, 0, keep + n_discard, -1, -n_discard);
                    m_kv_tokens.erase(m_kv_tokens.begin() + keep,
                                      m_kv_tokens.begin() + keep + n_discard);
                    kv_len -= n_discard;
                    shifted = true;
                    m_mtp.keep_prefix(0); // its cells sit at the pre-shift positions
                    char sb[160];
                    snprintf(sb, sizeof(sb),
                             "[xllama] session: context shift — evicted %d tokens past "
                             "keep=%d, kv now %d of %d (#169)\n",
                             n_discard, keep, kv_len, m_n_ctx);
                    log_output(sb);
                }
            }
            if (!shifted) {
                // #173 fail-fast: the caller retries with the trimmed full
                // prompt. Reached when the cache cannot shift, the bookkeeping
                // is out of sync, or even a full eviction would not fit.
                res.error_msg = "context full: kv=" + std::to_string(kv_len) +
                                " + delta=" + std::to_string(n_pf) +
                                " exceeds n_ctx=" + std::to_string(m_n_ctx);
                log_output(("[xllama] session generate: " + res.error_msg + "\n").c_str());
                return res;
            }
        }

        // Prefill: positions continue automatically from the KV cache end, so a
        // reuse turn appends after the retained context.
        //
        // Chunked at llama_n_batch: llama_decode does not return an error for an
        // oversized batch, it ASSERTS (GGML_ASSERT(n_tokens_all <= n_batch) in
        // llama-context.cpp → abort, Release included). n_batch defaults to
        // min(n_ctx, 2048) while the trimmer ceiling of a 4096-token coding
        // session is 3846, so a long paste used to kill the process. This is the
        // LOGICAL batch only — the physical ubatch stays 512 (#172 optimum),
        // which is what the prefill rate was measured on.
        InferenceWatchdog wd(gp.progress(), gp.abort_flag);
        if (n_pf >= 256)
            log_gguf_ws("at prefill"); // both pools before a long prefill: what it has to grow into
        wd.enter_prefill(n_pf);
        const auto t_prefill0 = std::chrono::steady_clock::now();
        // MTP: the drafter holds what the target kept, and mirrors every
        // prefill chunk while the target's hidden rows of that chunk are live.
        m_mtp.keep_prefix(static_cast<llama_pos>(kv_len));
        PrefillChunkHook mtp_hook = [this](const llama_token* t, int n, llama_pos pos0) {
            return mtp_mirror_chunk(m_mtp, t, n, pos0);
        };
        const PrefillChunkHook* after_chunk = m_mtp.usable() ? &mtp_hook : nullptr;
        // #170c: split the prefill where checkpoints go — positions in the
        // history this prefill completes: the divergence point a refused
        // rewind fell back from (the next prompt likely shares it: a system
        // prompt, a fixed instruction) and the prompt end minus the re-rendered
        // tail (the next API turn, a regenerate, an edit of the last message).
        const size_t pf_begin = static_cast<size_t>(kv_len);
        const size_t pf_end = pf_begin + static_cast<size_t>(n_pf);
        std::vector<size_t> rs_at;
        std::vector<llama_token> rs_history; // a continuation's resident + delta
        if (m_rs_ckpt_enabled && (full_prompt || m_kv_tokens.size() == pf_begin)) {
            if (rs_common >= pf_begin + kRsCheckpointMinGain &&
                rs_common + kRsCheckpointTail < pf_end)
                rs_at.push_back(rs_common);
            if (pf_end > pf_begin + kRsCheckpointTail)
                rs_at.push_back(pf_end - kRsCheckpointTail);
            if (!full_prompt && !rs_at.empty()) {
                rs_history = m_kv_tokens;
                rs_history.insert(rs_history.end(), tokens.begin(), tokens.end());
            }
        }
        const llama_token* rs_hist = full_prompt ? tokens.data() : rs_history.data();
        bool prefilled = true;
        int n_done = 0;
        for (const size_t at : rs_at) {
            const int seg = static_cast<int>(at - pf_begin) - n_done;
            if (!prefill_chunked(ctx, pf + n_done, seg, &wd, n_done, after_chunk)) {
                prefilled = false;
                break;
            }
            n_done += seg;
            save_rs_checkpoint(ctx, rs_hist, at);
            if (full_prompt && gp.save_prefix)
                save_prefix_cache(ctx, rs_hist, at);
        }
        if (prefilled)
            prefilled = prefill_chunked(ctx, pf + n_done, n_pf - n_done, &wd, n_done, after_chunk);
        if (!prefilled) {
            if (wd.stalled()) {
                res.stalled = true;
                res.error_msg = wd.stall_error();
            } else {
                res.error_msg = "prompt decode failed";
            }
            log_output("[xllama] session generate: " + res.error_msg + "\n");
            // The cache now holds a partial batch — nothing about it is
            // trustworthy, so drop the record AND the cells. Clearing both keeps
            // a continuation turn (which reads the cache length, not
            // m_kv_tokens) from decoding on top of half a prefill.
            m_kv_tokens.clear();
            llama_memory_clear(mem, true);
            m_mtp.keep_prefix(0);
            return res;
        }
        res.t_p_eval_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prefill0)
                .count();
        res.n_p_eval = n_pf;
        res.n_prompt = static_cast<int>(tokens.size());
        wd.prefill_done(n_pf, res.t_p_eval_ms);

        // The resident-token record now covers everything prefilled.
        if (full_prompt)
            m_kv_tokens.assign(tokens.begin(), tokens.end());
        else
            m_kv_tokens.insert(m_kv_tokens.end(), tokens.begin(), tokens.end());
        if (m_mtp.usable())
            common_speculative_begin(m_mtp.spec, 0, m_kv_tokens);

        // Shared with run_inference — see src/bridge/sampler_chain.h and #125.
        // SamplingConfig::is_greedy() also covers temperature 0, where the full
        // chain must not run (the repetition penalty can flip the argmax).
        // #175: reuse the chain on a continuation turn with unchanged sampling,
        // so penalty/RNG state crosses turn boundaries exactly as the KV does.
        const SamplingConfig sc = gp.sampling();
        if (full_prompt || !m_sampler || !same_chain(m_sampler_cfg, sc)) {
            const llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
            m_sampler.reset(llama_sampler_chain_init(sparams));
            add_sampler_stages(m_sampler.get(), sc, vocab);
            m_sampler_cfg = sc;
        }
        llama_sampler* sampler_chain = m_sampler.get();

        // ADR 0005: a tool proposal carries its template's grammar. It lives for
        // this generate only; a grammar that does not parse fails the request.
        LlamaSamplerPtr grammar(make_grammar_sampler(vocab, gp.grammar));
        if (!gp.grammar.empty()) {
            if (!grammar) {
                res.error_msg = "tool grammar did not parse";
                log_output("[xllama] session generate: " + res.error_msg + "\n");
                return res;
            }
            if (!gp.grammar.lazy &&
                !prefill_grammar(grammar.get(), vocab, gp.grammar.generation_prompt)) {
                res.error_msg = "tool grammar rejected the generation prompt";
                log_output("[xllama] session generate: " + res.error_msg + "\n");
                return res;
            }
        }

        const auto t_decode0 = std::chrono::steady_clock::now();

        // Clean stop at the context end (#173): each generated token needs a KV
        // slot, so the loop must not outrun n_ctx - (kv_len + prompt). Clamped
        // at 0 — an oversized full prompt yields a clean zero-token result.
        const int n_predict_eff = std::max(0, std::min(gp.n_predict, m_n_ctx - kv_len - n_pf));

        // The loop itself lives in decode_loop.h, shared with run_inference —
        // see there for why. What stays here is what is genuinely this session's:
        // keeping m_kv_tokens in step with the KV cells, which the #170a prefix
        // diff and the #170b snapshot fingerprint both read.
        // A preserved string counts only when it is one token (llama-server rule).
        std::unordered_set<llama_token> preserved;
        for (const std::string& text : gp.preserved_tokens) {
            llama_token id = 0;
            if (llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), &id, 1,
                               false, true) == 1)
                preserved.insert(id);
        }

        DecodeLoopParams dlp;
        dlp.ctx = ctx;
        dlp.sampler = sampler_chain;
        dlp.vocab = vocab;
        dlp.n_predict = n_predict_eff;
        dlp.stop_sequences = &gp.stop_sequences;
        dlp.preserved_tokens = preserved.empty() ? nullptr : &preserved;
        dlp.grammar = grammar.get();
        dlp.abort_flag = gp.abort_flag;
        dlp.on_token = gp.on_token;
        dlp.on_accepted = [this](llama_token t) { m_kv_tokens.push_back(t); };
        dlp.decode_start = t_decode0;
        // W2: seed + live history is m_kv_tokens (prefill already recorded above).
        dlp.prompt_lookup = m_prompt_lookup;
        dlp.token_history = &m_kv_tokens;
        dlp.watchdog = &wd;
        if (!forced.empty()) {
            dlp.forced = &forced;
            dlp.n_predict = std::min(n_predict_eff, static_cast<int>(forced.size()));
            dlp.stop_sequences = nullptr;
        }
        if (gp.logprobs || !forced.empty()) {
            res.logprobs.reserve(static_cast<size_t>(dlp.n_predict));
            dlp.logprobs = &res.logprobs;
        }
        wd.enter_decode(dlp.n_predict);
        const bool use_mtp = m_mtp.usable() && !dlp.grammar && !dlp.forced && !dlp.logprobs;
        const DecodeLoopResult dlr = use_mtp ? decode_loop_mtp(dlp, m_mtp, res.output_text)
                                             : decode_loop(dlp, res.output_text);
        // A classic turn (grammar, scoring) leaves the drafter without its
        // generated tokens: a gap its attention simply skips (M-RoPE positions
        // only have to move forward, so the next mirror still lands).
        const int n_generated = dlr.n_generated;
        const bool stopped_by_seq = dlr.ended_with_stop;

        res.t_eval_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_decode0)
                .count();
        res.t_first_token_ms =
            dlr.first_token_ms > 0.0 ? res.t_p_eval_ms + dlr.first_token_ms : 0.0;
        res.n_eval = n_generated;
        res.ended_with_stop = stopped_by_seq;
        res.n_drafted = dlr.n_drafted;
        res.n_spec_accepted = dlr.n_accepted;
        if (dlr.rewind_failed) {
            // History and KV are untrustworthy — same class as a decode failure.
            m_kv_tokens.clear();
            if (m_ctx)
                llama_memory_clear(llama_get_memory(m_ctx.get()), /*data=*/true);
            m_mtp.keep_prefix(0);
            res.success = false;
            res.error_msg =
                "speculative KV rewind unsupported (disable prompt_lookup for this model)";
            log_output(("[xllama] " + res.error_msg + "\n").c_str());
            return res;
        }
        if (wd.stalled()) {
            res.success = false;
            res.stalled = true;
            res.error_msg = wd.stall_error();
            log_output(("[xllama] " + res.error_msg + "\n").c_str());
            return res;
        }
        res.success = true;

        char log_buf[320];
        snprintf(log_buf, sizeof(log_buf),
                 "[xllama] session generate: n=%d prefill=%.1fms (%d tok) reuse=%d "
                 "drafted=%d spec_accept=%d\n",
                 n_generated, res.t_p_eval_ms, res.n_p_eval, gp.reuse_kv && !gp.reset_kv,
                 dlr.n_drafted, dlr.n_accepted);
        log_output(log_buf);
        // Where decode time goes: llama_decode vs sampling, and how often llama
        // reused the previous compute graph instead of rebuilding and splitting it
        // (the count is the context's, since it was created).
        snprintf(log_buf, sizeof(log_buf),
                 "[xllama] session decode: %.1f ms in llama_decode, %.1f ms sampling, "
                 "graphs reused %d\n",
                 dlr.decode_ms, dlr.sample_ms, llama_perf_context(m_ctx.get()).n_reused);
        log_output(log_buf);
        if (use_mtp) {
            snprintf(log_buf, sizeof(log_buf),
                     "[xllama] mtp: rounds=%d drafted=%d accepted=%d (%.3f) tok/round=%.3f "
                     "draft=%.1fms mirror=%.1fms%s\n",
                     dlr.n_rounds, dlr.n_drafted, dlr.n_accepted,
                     dlr.n_drafted > 0 ? (double)dlr.n_accepted / dlr.n_drafted : 0.0,
                     dlr.n_rounds > 0 ? (double)n_generated / dlr.n_rounds : 0.0, dlr.draft_ms,
                     dlr.mirror_ms, dlr.mtp_failed ? " FAILED" : "");
            log_output(log_buf);
        }

        return res;
    }

    int count_tokens(const std::string& prompt) override {
        const llama_vocab* vocab = llama_model_get_vocab(m_model.get());
        // parse_special=true to match generate(): counting template markers as
        // plain text overcounts a chat prompt by ~70% (50 vs 29 tokens on a
        // 2-turn ChatML render), which skews the routing threshold.
        int32_t n = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                   nullptr, 0, true, true);
        if (n == INT32_MIN)
            return 0;
        return -n;
    }

    EmbeddingResult embed(const EmbeddingParams& params) override {
        EmbeddingResult result;
        if (params.input.empty()) {
            result.error_msg = "input must not be empty";
            return result;
        }
        std::string ctx_error;
        if (!ensure_ctx(&ctx_error, true)) {
            result.error_msg = "failed to create embedding context: " + ctx_error;
            return result;
        }

        if (llama_model_has_encoder(m_model.get()) && llama_model_has_decoder(m_model.get())) {
            result.error_msg = "encoder-decoder models do not provide supported text embeddings";
            return result;
        }
        const llama_vocab* vocab = llama_model_get_vocab(m_model.get());
        int32_t n =
            llama_tokenize(vocab, params.input.data(), static_cast<int32_t>(params.input.size()),
                           nullptr, 0, true, true);
        if (n == INT32_MIN || n == 0) {
            result.error_msg = "input tokenization failed";
            return result;
        }
        std::vector<llama_token> tokens(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, params.input.data(), static_cast<int32_t>(params.input.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
        if (n <= 0) {
            result.error_msg = "input tokenization failed";
            return result;
        }
        tokens.resize(static_cast<size_t>(n));
        result.n_tokens = n;

        llama_context* ctx = m_ctx.get();
        const enum llama_pooling_type pooling = llama_pooling_type(ctx);
        if (pooling == LLAMA_POOLING_TYPE_RANK) {
            result.error_msg = "ranking GGUF models do not provide text embeddings";
            return result;
        }

        // Append EOS if the GGUF requests it and the sequence does not already end
        // in SEP or EOS. llama.cpp/examples/embedding/embedding.cpp warns when the
        // last token is neither. The tokenizer.ggml.add_eos_token flag is
        // authoritative; we append only when the tokenizer didn't already.
        const llama_token sep_token = llama_vocab_sep(vocab);
        const llama_token eos_token = llama_vocab_eos(vocab);
        if (llama_vocab_get_add_eos(vocab) && !tokens.empty()) {
            const llama_token last = tokens.back();
            if (last != sep_token && last != eos_token) {
                // Only append if it still fits in context after truncation.
                if (static_cast<int>(tokens.size()) < m_n_ctx) {
                    tokens.push_back(eos_token);
                }
            }
        }

        const int max_input_tokens = std::min(m_n_ctx, static_cast<int>(llama_n_batch(ctx)));
        if (static_cast<int>(tokens.size()) > max_input_tokens) {
            if (!params.truncate) {
                result.error_msg = "input exceeds embedding batch/context limit of " +
                                   std::to_string(max_input_tokens) +
                                   " tokens and truncate is false";
                return result;
            }
            // Trim to fit, keeping the tokens that matter for this pooling type.
            const PoolingType pool = static_cast<PoolingType>(static_cast<int>(pooling));
            trim_tokens_for_pooling(tokens, pool, max_input_tokens);
        }
        result.n_tokens = static_cast<int>(tokens.size());

        // Embedding requests share this context with chat. Drop any chat KV and
        // invalidate its token mirror; the next chat turn will prefill cleanly.
        m_kv_tokens.clear();
        llama_memory_clear(llama_get_memory(ctx), true);
        m_mtp.keep_prefix(0);
        llama_set_embeddings(ctx, true);
        struct EmbeddingContextReset {
            llama_context* ctx;
            ~EmbeddingContextReset() {
                llama_memory_clear(llama_get_memory(ctx), true);
                llama_set_embeddings(ctx, false);
            }
        } context_reset{ctx};

        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            result.error_msg = "failed to allocate embedding batch";
            return result;
        }
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[static_cast<size_t>(i)];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == batch.n_tokens - 1) ? 1 : 0;
        }
        const int32_t rc = llama_model_has_encoder(m_model.get()) ? llama_encode(ctx, batch)
                                                                  : llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            result.error_msg =
                "llama.cpp failed to encode the embedding input (code " + std::to_string(rc) + ")";
            return result;
        }

        const int n_embd = llama_model_n_embd_out(m_model.get());
        const float* values = pooling == LLAMA_POOLING_TYPE_NONE
                                  ? llama_get_embeddings_ith(ctx, batch.n_tokens - 1)
                                  : llama_get_embeddings_seq(ctx, 0);
        if (!values || n_embd <= 0) {
            result.error_msg = "model/backend did not produce a sequence embedding";
            return result;
        }
        result.embedding.assign(values, values + n_embd);
        if (!normalize_embedding(result.embedding, params.dimensions, &result.error_msg))
            return result;
        result.success = true;
        return result;
    }

    int context_length() const override {
        return m_n_ctx;
    }

    bool can_context_shift() const override {
        return m_can_shift;
    }

    // --- #170b: KV state files ---------------------------------------------
    //
    // Layout: one text header line, then the resident token ids, then the raw
    // sequence state. Everything the pin does NOT check lives in the header —
    // it validates cache shape only, so a re-quantized build of the same model
    // or the same model with a LoRA applied would load this file and generate
    // silent garbage.

    static constexpr const char* kStateMagic = "xllama-kv";
    static constexpr int kStateVersion = 1;
    // AppContainer dislikes very large single reads: ORT's 1 GB
    // ReadFileIntoBuffer had to come down to 16 MB before errcode 1450
    // (ERROR_NO_SYSTEM_RESOURCES) stopped appearing (uwp-constraints §9), and
    // llama_file's own path would move this blob in 64 MB slices. Stay well
    // under the known-good bound.
    static constexpr size_t kIoChunk = 8u << 20;
    // A sane ceiling for a declared blob size: refuse to allocate for a
    // corrupt or hostile header instead of trying and dying.
    static constexpr size_t kMaxStateBytes = 512u << 20;

    static FILE* open_state_file(const std::string& path, const char* mode) {
    #ifdef XLLAMA_UWP
        const std::string m(mode);
        return _wfopen(utf8_to_wstring(path).c_str(), std::wstring(m.begin(), m.end()).c_str());
    #else
        return std::fopen(path.c_str(), mode);
    #endif
    }

    static bool io_chunked(FILE* fp, void* data, size_t n, bool write) {
        auto* p = static_cast<uint8_t*>(data);
        for (size_t off = 0; off < n;) {
            const size_t want = std::min(kIoChunk, n - off);
            const size_t got =
                write ? std::fwrite(p + off, 1, want, fp) : std::fread(p + off, 1, want, fp);
            if (got != want)
                return false;
            off += want;
        }
        return true;
    }

    // Identity of the weights + context this cache was built on. Computed
    // after ensure_ctx() so it reflects the kv_q8 fallback (#171), not the
    // request.
    std::string state_fingerprint() const {
        char desc[192] = {};
        llama_model_desc(m_model.get(), desc, sizeof(desc) - 1);
        char buf[320];
        snprintf(buf, sizeof(buf), "v%d n_ctx=%d kv_q8=%d lora=%.4f params=%llu %s", kStateVersion,
                 m_n_ctx, m_kv_q8 ? 1 : 0, m_adapter ? m_lora_scale : 0.0f,
                 static_cast<unsigned long long>(llama_model_n_params(m_model.get())), desc);
        return buf;
    }

    bool save_state(const std::string& path, std::string* err) override {
        if (!ensure_ctx(err))
            return false;
        if (m_kv_tokens.empty()) {
            if (err)
                *err = "no resident KV to save";
            return false;
        }

        std::vector<uint8_t> blob;
        size_t n_state = 0;
        try {
            blob.resize(llama_state_seq_get_size(m_ctx.get(), 0));
            n_state = llama_state_seq_get_data(m_ctx.get(), blob.data(), blob.size(), 0);
        } catch (const std::exception& e) {
            if (err)
                *err = std::string("state_seq_get_data threw: ") + e.what();
            return false;
        }
        if (n_state == 0) {
            if (err)
                *err = "llama_state_seq_get_data failed";
            return false;
        }

        // Write to a sibling temp and rename: a half-written file is tens of
        // MB of garbage that the loader can only reject after reading it all,
        // and a crash mid-write must not leave one behind.
        const std::string tmp = path + ".tmp";
        FILE* fp = open_state_file(tmp, "wb");
        if (!fp) {
            if (err)
                *err = "cannot open " + tmp;
            return false;
        }
        char hdr[512];
        const int hn = snprintf(hdr, sizeof(hdr), "%s\t%zu\t%zu\t%s\n", kStateMagic, n_state,
                                m_kv_tokens.size(), state_fingerprint().c_str());
        bool ok = hn > 0 && static_cast<size_t>(hn) < sizeof(hdr) &&
                  std::fwrite(hdr, 1, static_cast<size_t>(hn), fp) == static_cast<size_t>(hn);
        ok = ok && io_chunked(fp, m_kv_tokens.data(), m_kv_tokens.size() * sizeof(llama_token),
                              /*write=*/true);
        ok = ok && io_chunked(fp, blob.data(), n_state, /*write=*/true);
        ok = (std::fclose(fp) == 0) && ok;

        std::error_code ec;
        if (!ok) {
            std::filesystem::remove(tmp, ec);
            if (err)
                *err = "write failed (disk full?)";
            return false;
        }
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            std::filesystem::remove(tmp, ec);
            if (err)
                *err = "rename failed: " + ec.message();
            return false;
        }
        char lb[160];
        snprintf(lb, sizeof(lb), "[xllama] session: KV state saved — %zu tokens, %.1f MB (#170b)\n",
                 m_kv_tokens.size(), (double)n_state / (1024.0 * 1024.0));
        log_output(lb);
        return true;
    }

    bool load_state(const std::string& path, std::string* err) override {
        if (!ensure_ctx(err))
            return false;
        FILE* fp = open_state_file(path, "rb");
        if (!fp) {
            if (err)
                *err = "no state file at " + path;
            return false;
        }
        struct Closer {
            FILE* f;
            ~Closer() {
                if (f)
                    std::fclose(f);
            }
        } closer{fp};

        char hdr[512] = {};
        if (!std::fgets(hdr, sizeof(hdr), fp)) {
            if (err)
                *err = "empty state file";
            return false;
        }
        std::string line(hdr);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
            line.pop_back();
        // magic \t n_state \t n_tokens \t fingerprint(rest of line)
        std::string field[3];
        size_t pos = 0;
        for (auto& f : field) {
            const size_t t = line.find('\t', pos);
            if (t == std::string::npos) {
                if (err)
                    *err = "malformed state header";
                return false;
            }
            f = line.substr(pos, t - pos);
            pos = t + 1;
        }
        if (field[0] != kStateMagic) {
            if (err)
                *err = "not an xllama KV state file";
            return false;
        }
        if (line.substr(pos) != state_fingerprint()) {
            // The common, expected case: model / n_ctx / KV-quant / LoRA
            // changed since the file was written. Not an error worth shouting
            // about — the caller just prefills.
            if (err)
                *err = "state file belongs to a different model or context";
            return false;
        }
        size_t n_state = 0, n_tokens = 0;
        try {
            n_state = std::stoull(field[1]);
            n_tokens = std::stoull(field[2]);
        } catch (const std::exception&) {
            if (err)
                *err = "malformed state header";
            return false;
        }
        if (n_state == 0 || n_state > kMaxStateBytes || n_tokens == 0 ||
            n_tokens > static_cast<size_t>(m_n_ctx)) {
            if (err)
                *err = "state header out of range";
            return false;
        }

        std::vector<llama_token> tokens(n_tokens);
        std::vector<uint8_t> blob(n_state);
        if (!io_chunked(fp, tokens.data(), n_tokens * sizeof(llama_token), /*write=*/false) ||
            !io_chunked(fp, blob.data(), n_state, /*write=*/false)) {
            if (err)
                *err = "state file truncated";
            return false;
        }

        size_t used = 0;
        try {
            used = llama_state_seq_set_data(m_ctx.get(), blob.data(), blob.size(), 0);
        } catch (const std::exception& e) {
            used = 0;
            if (err)
                *err = std::string("state_seq_set_data threw: ") + e.what();
        }
        if (used == 0) {
            // The pin rolls its own read back, but the bookkeeping is ours.
            llama_memory_clear(llama_get_memory(m_ctx.get()), true);
            m_kv_tokens.clear();
            if (err && err->empty())
                *err = "llama_state_seq_set_data rejected the file";
            return false;
        }
        m_kv_tokens = std::move(tokens);
        // The file holds the target's cells only: the drafter restarts empty
        // (drafts read just the new tokens until the history refills).
        m_mtp.keep_prefix(0);
        char lb[160];
        snprintf(lb, sizeof(lb),
                 "[xllama] session: KV state loaded — %zu tokens, %.1f MB (#170b)\n",
                 m_kv_tokens.size(), (double)n_state / (1024.0 * 1024.0));
        log_output(lb);
        return true;
    }
};

namespace detail {
std::unique_ptr<Session> create_llama(const SessionParams& sp, std::string* err) {
    // resolve_model_path yields a FILE path on Linux (a direct .gguf) but the
    // model DIRECTORY on UWP (catalogue layout LocalState\models\<name>\). llama
    // loads a file, so descend into a directory and pick the single .gguf inside.
    // Exclude LoRA basename so dirs with model.gguf + adapter.gguf load the base.
    std::string lora_base;
    if (!sp.lora_path.empty()) {
        lora_base = std::filesystem::path(sp.lora_path).filename().string();
    }
    std::string abs_path = first_gguf_in_dir(resolve_model_path(sp.model_path), lora_base);
    if (abs_path.empty()) {
        if (err)
            *err = "no .gguf file in model dir: " + resolve_model_path(sp.model_path);
        return nullptr;
    }

    llama_model_params mparams = llama_model_default_params();
    // the expert cache (which wins over stream) keeps the experts on the CPU itself
    const bool cpu_only = sp.stream_weights && sp.expert_slots <= 0;
    apply_env_overrides(sp.env); // first: the set_env_int below honour it
    // MTP self-speculation (decode_loop_mtp.h): XLLAMA_MTP_SIDECAR names the
    // head's GGUF (patch 0012 adds its block to this model), XLLAMA_MTP_DRAFT
    // the drafts per round (default 1, 0 = load nothing), XLLAMA_MTP_PMIN the
    // head's confidence below which a round decodes one token (default 0).
    int mtp_n = 0;
    float mtp_p_min = 0.0f;
    if (const char* sc = std::getenv("XLLAMA_MTP_SIDECAR"); sc && *sc) {
        mtp_n = 1;
        if (const char* s = std::getenv("XLLAMA_MTP_DRAFT"); s && *s)
            mtp_n = std::clamp(std::atoi(s), 0, 8);
        if (const char* s = std::getenv("XLLAMA_MTP_PMIN"); s && *s)
            mtp_p_min = std::clamp(static_cast<float>(std::atof(s)), 0.0f, 1.0f);
        mparams.load_mtp = mtp_n > 0;
        log_output("[xllama] mtp: sidecar " + std::string(sc) + ", " + std::to_string(mtp_n) +
                   " draft/round\n");
    }
    const int gpu_layers = apply_gguf_gpu_layers(cpu_only ? 0 : sp.n_gpu_layers, mparams, abs_path);
    apply_gguf_residency(sp.stream_weights, mparams, sp.expert_slots, sp.expert_swap,
                         sp.expert_prefetch, sp.expert_major, sp.expert_prefetch_ahead);
    apply_expert_used_override(mparams, abs_path);

    install_llama_log();
    llama_model* raw_model = llama_model_load_from_file(abs_path.c_str(), mparams);
    log_gguf_ws("after model load");
    if (!raw_model) {
        if (err)
            *err = "failed to load model: " + abs_path;
        return nullptr;
    }

    LlamaAdapterLoraPtr adapter;
    if (!sp.lora_path.empty()) {
        llama_adapter_lora* raw_ad = llama_adapter_lora_init(raw_model, sp.lora_path.c_str());
        if (!raw_ad) {
            llama_model_free(raw_model);
            if (err)
                *err = "failed to load LoRA adapter: " + sp.lora_path;
            return nullptr;
        }
        adapter.reset(raw_ad);
        log_output("[xllama] Session: GGUF LoRA adapter loaded (runtime)\n");
    }

    int n_threads = sp.n_threads > 0 ? sp.n_threads : detect_threads_llama();
    int n_ctx = sp.n_ctx > 0 ? sp.n_ctx : kDefaultNCtx;
    log_output("[xllama] Session: GGUF model loaded via llama.cpp (persistent)\n");
    auto session = std::make_unique<LlamaSession>(
        LlamaModelPtr(raw_model), std::move(adapter), sp.lora_scale, n_ctx, n_threads, sp.n_batch,
        sp.n_ubatch, sp.kv_q8, sp.prompt_lookup, gpu_layers, mtp_n, mtp_p_min);
    session->init_prefix_cache();
    return session;
}
} // namespace detail

} // namespace xllama

#endif // XLLAMA_USE_LLAMA

// ---------------------------------------------------------------------------
// Public factory: dispatch to the compiled backend(s).
// ---------------------------------------------------------------------------
namespace xllama {

std::unique_ptr<Session> Session::create(const SessionParams& sp, std::string* err) {
#if defined(XLLAMA_USE_ORT) && defined(XLLAMA_USE_LLAMA)
    Backend b = sp.backend;
    if (b == Backend::Auto) {
        // Layout-aware: uses suffix fast-path + resolve + directory scan so that
        // bare catalogue names (e.g. "qwen35-0.8b" pointing at a dir with .gguf)
        // are classified correctly. Explicit Backend from manifest (MainPage) or
        // tests still takes precedence.
        b = model_uses_llama_backend(sp.model_path) ? Backend::LlamaCpp : Backend::OrtGenAI;
    }
    return b == Backend::LlamaCpp ? detail::create_llama(sp, err) : detail::create_ort(sp, err);
#elif defined(XLLAMA_USE_ORT)
    return detail::create_ort(sp, err);
#else // XLLAMA_USE_LLAMA
    return detail::create_llama(sp, err);
#endif
}

SessionHub& session_hub() {
    static SessionHub hub;
    return hub;
}

} // namespace xllama
