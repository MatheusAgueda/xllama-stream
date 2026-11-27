// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// B3: shared multi-sequence prefill (Session::generate_batch, POST /v1/xllama/batch). The numbers
// and admission decisions live here, WinRT-free, so the host tests pin them; the session reads
// XLLAMA_BATCH_SEQS through batch_seqs_from_env and the API route admits a batch through
// batch_admit before and after tokenizing.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace xllama {

/// Most sequences a batching context is created with. Each one costs a recurrent-state cell on a
/// hybrid model (63 MiB on the console's 35B) and the API route never takes more requests.
constexpr int kMaxBatchSeqs = 8;

/// XLLAMA_BATCH_SEQS: unset, empty, non-numeric or < 2 = 1 (batching off, the context is created
/// exactly as before); above kMaxBatchSeqs = kMaxBatchSeqs.
inline int batch_seqs_from_env(const char* v) {
    if (!v || !*v)
        return 1;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || n < 2)
        return 1;
    return n > kMaxBatchSeqs ? kMaxBatchSeqs : static_cast<int>(n);
}

/// How a batch's prompts are decoded. llama.cpp's split_equal (hybrid memory) grows a ubatch only
/// while every sequence in it still has a token, so sequences of unequal length cost one ubatch
/// per distinct length, each re-paying the expert reads. The shared part therefore takes the same
/// count from every sequence: positions [0, shared) of all k prompts, chunked so each chunk holds
/// `shared_chunks[c]` tokens of every sequence (k x that <= n_batch). The rest of prompt i,
/// positions [shared, len_i), is its own pass, decoded alone in `own_decodes[i]` calls of at most
/// n_batch tokens, the last token carrying the logits the first sample reads.
struct BatchPrefillPlan {
    int shared = 0;                 ///< E = min(len) - 1 (0 when n_batch < k: no shared part)
    std::vector<int> shared_chunks; ///< per-sequence tokens of each shared llama_decode
    std::vector<int> own;           ///< len_i - shared, >= 1
    std::vector<int> own_decodes;   ///< ceil(own_i / n_batch)
    int n_shared_decodes() const {
        return static_cast<int>(shared_chunks.size());
    }
};

/// The plan for prompts of `lengths` tokens (each >= 1) on a context of logical batch n_batch.
/// The shared part takes ceil(E / floor(n_batch / k)) decodes: equal to ceil(k E / n_batch) when k
/// divides n_batch, one more at most otherwise (a chunk of equal counts cannot fill the remainder).
inline BatchPrefillPlan batch_prefill_plan(const std::vector<int>& lengths, int n_batch) {
    BatchPrefillPlan plan;
    if (lengths.empty())
        return plan;
    n_batch = n_batch > 1 ? n_batch : 1;
    const int k = static_cast<int>(lengths.size());
    int shortest = lengths.front();
    for (const int n : lengths)
        shortest = n < shortest ? n : shortest;
    const int share = n_batch / k;
    plan.shared = share > 0 && shortest > 1 ? shortest - 1 : 0;
    for (int done = 0; done < plan.shared; done += share)
        plan.shared_chunks.push_back(plan.shared - done < share ? plan.shared - done : share);
    for (const int n : lengths) {
        const int own = n - plan.shared > 1 ? n - plan.shared : 1;
        plan.own.push_back(own);
        plan.own_decodes.push_back((own + n_batch - 1) / n_batch);
    }
    return plan;
}

enum class BatchAdmit {
    Ok,
    Empty,        // no requests: 400
    TooMany,      // more requests than the session's sequences: 400
    TooManyTokens // the prompts plus the longest reply do not fit the context: 400
};

/// Request count against the session's sequence count (Session::batch_seqs(), 1 = batching off,
/// where only a single request is served).
inline BatchAdmit batch_admit_count(size_t n_requests, int session_seqs) {
    if (n_requests == 0)
        return BatchAdmit::Empty;
    const size_t cap = static_cast<size_t>(session_seqs > 1 ? session_seqs : 1);
    return n_requests > cap ? BatchAdmit::TooMany : BatchAdmit::Ok;
}

/// The summed prompt cap: every request's prompt tokens plus the largest max_tokens must fit
/// n_ctx, since one unified KV holds every prefix while the first reply decodes. Stricter than
/// what the session checks (it lets a later reply use cells an earlier one freed), so an admitted
/// batch never fails there for lack of room.
inline BatchAdmit batch_admit_tokens(const std::vector<int>& prompt_tokens,
                                     const std::vector<int>& max_tokens, int n_ctx) {
    int64_t sum = 0;
    for (const int n : prompt_tokens)
        sum += n > 0 ? n : 0;
    int64_t longest = 0;
    for (const int n : max_tokens)
        longest = n > longest ? n : longest;
    return sum + longest > static_cast<int64_t>(n_ctx) ? BatchAdmit::TooManyTokens : BatchAdmit::Ok;
}

} // namespace xllama
