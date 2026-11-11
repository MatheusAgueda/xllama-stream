#pragma once

// MTP self-speculation (decode_loop with a multi-token-prediction head).
//
// The trunk carries one extra transformer block, trained by the model's authors
// to predict token t+2 from the hidden state of t and the embedding of t+1
// (Qwen3.6 ships it as mtp.*; the xllama GGUF for the heretic IQ3_S trunk has
// none, so it comes from a sidecar GGUF, patch 0012). llama.cpp runs that block
// in a second context of the same model (LLAMA_CONTEXT_TYPE_MTP, own KV for the
// one layer) and common/speculative.cpp's draft-mtp keeps it in step:
//   - every batch the target decodes is mirrored into the drafter (process()),
//     row k paired with the target's pre-norm hidden state of row k-1;
//   - draft() decodes the last committed token there and reads its argmax;
//   - accept(n) picks the hidden row the next draft starts from.
//
// One round = sample x from the target, draft d_1..d_k after x, decode
// [x, d_1..d_k] on the target in ONE batch, keep the drafts the target's own
// sampler agrees with. The batch costs about one forward's fixed overhead and
// dense weights, so every accepted draft is a token that skipped it; expert
// reads are per token either way, so a rejected draft costs one token's
// expert reads and compute. The recurrent (gated delta-net) layers roll the
// rejected tail back through llama.cpp's n_rs_seq snapshots, which is why the
// target context is created with n_rs_seq = k (session.cpp).
//
// Output is the target's samples, so greedy text follows the target. It is not
// bit-equal to the one-token-per-decode path: a multi-row batch reduces in a
// different order (see the correctness note on decode_loop).

#include "decode_loop.h"

#include "common.h"      // common_batch
#include "speculative.h" // common_speculative_*

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

namespace xllama {

struct MtpDrafter {
    llama_context* ctx_dft = nullptr; // LLAMA_CONTEXT_TYPE_MTP context of the same model
    common_speculative* spec = nullptr;
    int n_max = 1;       // drafts per round
    bool broken = false; // a drafter decode or rewind failed; drafting stops for the session
    common_batch mirror; // prefill chunks re-described for process()
    common_batch verify; // the target's verify batch, shared with process()

    bool usable() const {
        return ctx_dft && spec && !broken;
    }

    void fail(const char* what) {
        if (!broken)
            log_output(std::string("[xllama] mtp: ") + what + " — drafting off for this session\n");
        broken = true;
    }

    // The drafter holds exactly the target's first |n_keep| positions. Plain KV
    // (llama.cpp gives the MTP head of a hybrid Qwen a pure attention cache), so
    // a partial removal cannot refuse unless something is badly wrong.
    void keep_prefix(llama_pos n_keep) {
        if (!usable())
            return;
        if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_keep, -1))
            fail("drafter seq_rm refused");
    }
};

// prefill_chunked hook: the target just decoded |n| prompt tokens from |pos0|.
// Never fails the prefill — the target is fine; only drafting stops.
inline bool mtp_mirror_chunk(MtpDrafter& m, const llama_token* tokens, int n, llama_pos pos0) {
    if (!m.usable() || n <= 0)
        return true;
    m.keep_prefix(pos0);
    if (!m.usable())
        return true;
    m.mirror.clear();
    for (int i = 0; i < n; ++i)
        m.mirror.add(tokens[i], pos0 + i, 0, false);
    if (!common_speculative_process(m.spec, m.mirror))
        m.fail("drafter catch-up decode failed (prefill)");
    return true;
}

// decode_loop for a session with an MTP drafter. Callers route here only when
// no grammar, forced reply or logprobs are requested (drafts are accepted into
// the sampler as they are verified, and those paths cannot rewind it).
inline DecodeLoopResult decode_loop_mtp(const DecodeLoopParams& p, MtpDrafter& m,
                                        std::string& output_text) {
    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };

    DecodeLoopResult out;
    llama_memory_t mem = llama_get_memory(p.ctx);
    const llama_pos n_ctx = static_cast<llama_pos>(llama_n_ctx(p.ctx));
    std::vector<llama_token> draft;
    draft.reserve(static_cast<size_t>(std::max(1, m.n_max)));

    bool have_pending =
        false; // the target already sampled the next token (a rejected draft's slot)
    llama_token pending = LLAMA_TOKEN_NULL;

    while (out.n_generated < p.n_predict) {
        if (p.abort_flag && p.abort_flag->load())
            break;
        if (p.watchdog && p.watchdog->aborted())
            break;

        llama_token x;
        if (have_pending) {
            x = pending;
            have_pending = false;
        } else {
            const auto ts0 = clock::now();
            x = llama_sampler_sample(p.sampler, p.ctx, -1);
            out.sample_ms += ms_since(ts0);
        }
        if (detail::stops_at_eog(p, x)) {
            log_output("[xllama] EOG after " + std::to_string(out.n_generated) + " tokens\n");
            break;
        }
        if (out.n_generated == 0 && out.first_token_ms == 0.0 &&
            p.decode_start != clock::time_point{})
            out.first_token_ms = ms_since(p.decode_start);
        if (detail::emit_token(p, x, output_text)) {
            out.ended_with_stop = true;
            ++out.n_generated;
            log_output("[xllama] stop sequence after " + std::to_string(out.n_generated) +
                       " tokens\n");
            break;
        }

        const llama_pos n = llama_memory_seq_pos_max(mem, 0) + 1; // x's position
        int n_draft = m.usable() ? m.n_max : 0;
        n_draft = std::min(n_draft, p.n_predict - out.n_generated - 1);
        n_draft = std::min(n_draft, static_cast<int>(n_ctx - 1 - n));
        draft.clear();
        if (n_draft > 0) {
            const auto td0 = clock::now();
            common_speculative_draft_params& dp = common_speculative_get_draft_params(m.spec, 0);
            dp.drafting = true;
            dp.n_max = n_draft;
            dp.pos0 = n;
            dp.id_last = x;
            dp.prompt = p.token_history;
            dp.result = &draft;
            dp.result_q = nullptr;
            common_speculative_draft(m.spec);
            // The drafter wrote its guesses at n..; the mirror of the verify
            // batch rewrites those positions from the target's hidden states.
            m.keep_prefix(n);
            out.draft_ms += ms_since(td0);
            if (!m.usable())
                draft.clear();
        }

        m.verify.clear();
        m.verify.add(x, n, 0, true);
        for (size_t i = 0; i < draft.size(); ++i)
            m.verify.add(draft[i], n + 1 + static_cast<llama_pos>(i), 0, true);
        {
            const auto t0 = clock::now();
            const int32_t rc = llama_process(p.ctx, LLAMA_PROCESS_TYPE_DECODE, m.verify.get());
            out.decode_ms += ms_since(t0);
            if (rc != 0) {
                log_output("[xllama] decode failed at token, stopping generation\n");
                break;
            }
        }
        ++out.n_rounds;
        if (m.usable()) {
            const auto tm0 = clock::now();
            if (!common_speculative_process(m.spec, m.verify))
                m.fail("drafter catch-up decode failed");
            out.mirror_ms += ms_since(tm0);
        }
        detail::accept_token(p, x);
        ++out.n_generated;

        // Row i of the batch predicts the token after batch[i]: x for i = 0, d_i after.
        int n_acc = 0;
        bool stop = false;
        for (size_t i = 0; i < draft.size(); ++i) {
            if (out.n_generated >= p.n_predict || (p.abort_flag && p.abort_flag->load())) {
                stop = true;
                break;
            }
            const auto ts0 = clock::now();
            const llama_token y = llama_sampler_sample(p.sampler, p.ctx, static_cast<int32_t>(i));
            out.sample_ms += ms_since(ts0);
            if (y != draft[i]) {
                pending = y;
                have_pending = true;
                break;
            }
            // A matched EOG / stop token is in the KV but not kept: the trim
            // below drops it, as the classic path never decodes it.
            if (detail::stops_at_eog(p, y)) {
                log_output("[xllama] EOG after " + std::to_string(out.n_generated) +
                           " tokens (mtp)\n");
                stop = true;
                break;
            }
            if (detail::emit_token(p, y, output_text)) {
                out.ended_with_stop = true;
                ++out.n_generated;
                stop = true;
                break;
            }
            detail::accept_token(p, y);
            ++out.n_generated;
            ++n_acc;
        }

        out.n_drafted += static_cast<int>(draft.size());
        out.n_accepted += n_acc;
        const llama_pos keep = n + 1 + n_acc;
        if (static_cast<size_t>(n_acc) < draft.size()) {
            if (!llama_memory_seq_rm(mem, 0, keep, -1)) {
                log_output("[xllama] mtp: target seq_rm refused (n_rs_seq too small?) — "
                           "aborting generation\n");
                out.rewind_failed = true;
                break;
            }
            m.keep_prefix(keep);
        }
        if (m.usable())
            common_speculative_accept(m.spec, 0, static_cast<uint16_t>(n_acc));
        if (p.watchdog)
            p.watchdog->on_decode_token(out.n_generated);
        if (stop)
            break;
    }
    out.mtp_failed = m.broken;
    return out;
}

} // namespace xllama
