// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// S1 hardening of the LAN API (uwp/api-server.cpp): the two auth phases, the compute permit and
// hub lock released before the final reply write, the prompt text cap, /api/pull admission order,
// the /v1/preferences quotas, and the exact model spellings. The decisions live in WinRT-free
// headers; the handler's call ORDER is pinned by reading the source (as test_d3d12_env.cpp does
// for patch 0026), because the handler itself only builds in the UWP job.

#include "xllama/api_auth.h"
#include "xllama/api_limits.h"
#include "xllama/api_pull_policy.h"
#include "xllama/batch_policy.h"
#include "xllama/compute_admission.h"
#include "xllama/http_framing.h"
#include "xllama/model_spelling.h"

#include <doctest/doctest.h>

#include <chrono>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

using namespace xllama;

namespace {
const std::string kKey = "0123456789abcdef0123456789ABCDEF";

std::string read_file(const char* path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The text of `src` from the first `from` to the next `to` after it.
std::string slice(const std::string& src, const std::string& from, const std::string& to) {
    const size_t a = src.find(from);
    REQUIRE(a != std::string::npos);
    const size_t b = src.find(to, a + from.size());
    REQUIRE(b != std::string::npos);
    return src.substr(a, b - a);
}
} // namespace

// ---- 1. auth: two phases ---------------------------------------------------------------------

TEST_CASE("auth phase: api-auth.txt picks the phase, anything but 'enforce' is phase 1") {
    CHECK(api_auth_phase_from_text("") == ApiAuthPhase::Observe);
    CHECK(api_auth_phase_from_text(" \r\n") == ApiAuthPhase::Observe);
    CHECK(api_auth_phase_from_text("observe") == ApiAuthPhase::Observe);
    CHECK(api_auth_phase_from_text("enforce") == ApiAuthPhase::Enforce);
    CHECK(api_auth_phase_from_text("  ENFORCE\r\n") == ApiAuthPhase::Enforce);
    CHECK(api_auth_phase_from_text("Enforce # on 2026-12-01") == ApiAuthPhase::Enforce);
    CHECK(api_auth_phase_from_text("enforced") == ApiAuthPhase::Observe);
    CHECK(api_auth_phase_from_text("enforc") == ApiAuthPhase::Observe);
    CHECK(api_auth_phase_from_text("1") == ApiAuthPhase::Observe);
    CHECK(api_auth_phase_from_text("# enforce") == ApiAuthPhase::Observe);
}

TEST_CASE("auth phase 2: a missing, short, empty or wrong Bearer is denied (401)") {
    const ApiAuthPhase p = ApiAuthPhase::Enforce;
    const ApiAuthDecision ok = decide_api_auth(p, kKey, "Bearer " + kKey);
    CHECK(ok.allow);
    CHECK_FALSE(ok.failed);
    const std::string wrong = "Bearer " + kKey.substr(0, kKey.size() - 1) + "x";
    for (const std::string& header :
         {std::string(), std::string("Bearer "), std::string("Bearer short"), wrong, kKey}) {
        const ApiAuthDecision d = decide_api_auth(p, kKey, header);
        CHECK_FALSE(d.allow);
        CHECK(d.failed);
    }
    // Fail closed: api.key missing, or shorter than 32 characters (api_key_from_text gives "").
    const std::string no_key = api_key_from_text(std::string(31, 'k'));
    CHECK(no_key.empty());
    for (const std::string& header :
         {std::string(), std::string("Bearer "), std::string("Bearer ") + std::string(31, 'k'),
          "Bearer " + kKey}) {
        const ApiAuthDecision d = decide_api_auth(p, no_key, header);
        CHECK_FALSE(d.allow);
        CHECK(d.failed);
    }
}

TEST_CASE("auth phase 1: a Bearer failure is counted and still served") {
    const ApiAuthPhase p = ApiAuthPhase::Observe;
    CHECK(decide_api_auth(p, kKey, "Bearer " + kKey).allow);
    CHECK_FALSE(decide_api_auth(p, kKey, "Bearer " + kKey).failed);
    for (const std::string& header :
         {std::string(), std::string("Bearer "), std::string("Bearer short"),
          "Bearer " + kKey.substr(1) + "x"}) {
        const ApiAuthDecision d = decide_api_auth(p, kKey, header);
        CHECK(d.allow);  // served
        CHECK(d.failed); // and counted
    }
    // No usable key at all: everything is served and counted, nothing can match.
    const ApiAuthDecision none = decide_api_auth(p, "", "Bearer anything");
    CHECK(none.allow);
    CHECK(none.failed);
}

TEST_CASE("auth stats: counts per client, bounded, never the key") {
    ApiAuthStats stats;
    stats.record_failure("192.0.2.7", true);
    stats.record_failure("192.0.2.7", true);
    stats.record_failure("192.0.2.8", false);
    ApiAuthStats::Snapshot s = stats.snapshot();
    CHECK(s.failures == 3);
    CHECK(s.served == 2);
    CHECK(s.denied == 1);
    CHECK(s.clients.size() == 2);

    // More distinct clients than kMaxClients: the memory stays bounded, the total stays exact.
    for (size_t i = 0; i < ApiAuthStats::kMaxClients + 10; ++i)
        stats.record_failure("198.51.100." + std::to_string(i), true);
    s = stats.snapshot();
    CHECK(s.clients.size() == ApiAuthStats::kMaxClients);
    CHECK(s.other > 0);
    CHECK(s.failures == 3 + ApiAuthStats::kMaxClients + 10);

    const std::string json = api_auth_stats_json(ApiAuthPhase::Observe, s);
    CHECK(json.find("\"phase\":\"observe\"") != std::string::npos);
    CHECK(json.find("\"failures\":" + std::to_string(s.failures)) != std::string::npos);
    CHECK(json.find("\"192.0.2.7\":2") != std::string::npos);
    CHECK(json.find(kKey) == std::string::npos);
    CHECK(json.find("Bearer") == std::string::npos);
    CHECK(api_auth_stats_json(ApiAuthPhase::Enforce, ApiAuthStats::Snapshot{})
              .find("\"phase\":\"enforce\"") != std::string::npos);

    // A client string with quote, backslash or control bytes cannot break the JSON out.
    ApiAuthStats odd;
    odd.record_failure(std::string("a\"b\\c\n"), true);
    const std::string odd_json = api_auth_stats_json(ApiAuthPhase::Observe, odd.snapshot());
    CHECK(odd_json.find("\"a?b?c?\":1") != std::string::npos);
}

// ---- 2. the permit and hub lock are free before the final reply write ------------------------

TEST_CASE("compute scope: release() frees the hub lock and the permit before the reply write") {
    ComputeJobAdmission admission;
    std::mutex hub;
    ComputeScope scope;
    scope.permit = admission.try_acquire();
    scope.hub = std::unique_lock<std::mutex>(hub, std::try_to_lock);
    REQUIRE(scope.permit);
    REQUIRE(scope.hub.owns_lock());
    CHECK_FALSE(admission.try_acquire()); // held while the work runs

    // The reply write of a client that never reads: whatever it blocks on, it sees the slot free.
    bool slot_free_during_write = false;
    bool hub_free_during_write = false;
    const auto write_reply = [&] {
        slot_free_during_write = static_cast<bool>(admission.try_acquire());
        std::unique_lock<std::mutex> probe(hub, std::try_to_lock);
        hub_free_during_write = probe.owns_lock();
    };
    scope.release();
    write_reply();
    CHECK(slot_free_during_write);
    CHECK(hub_free_during_write);
    scope.release(); // idempotent
}

TEST_CASE("compute scope: a scope that never got the hub still frees its permit") {
    ComputeJobAdmission admission;
    ComputeScope scope;
    scope.permit = admission.try_acquire();
    scope.release();
    CHECK(admission.try_acquire());
}

TEST_CASE("handler order: chat, embeddings and images release before their final write") {
    const std::string src = read_file(API_SERVER_CPP_PATH);
    REQUIRE_FALSE(src.empty());
    const std::string chat =
        slice(src, "req.path == \"/v1/chat/completions\"", "EmbeddingApi embedding_api;");
    const std::string emb =
        slice(src, "EmbeddingApi embedding_api;", "// #118: preference capture");
    const std::string img = slice(src, "req.path == \"/v1/images/generations\"", "// 4x upscale");
    for (const std::string* route : {&chat, &emb, &img}) {
        const size_t release = route->rfind("scope.release();");
        const size_t final_write = route->rfind("write_response(socket, status, json);");
        REQUIRE(release != std::string::npos);
        REQUIRE(final_write != std::string::npos);
        CHECK(release < final_write);
        // No bare permit or lock left that would outlive the write.
        CHECK(route->find("auto compute_permit") == std::string::npos);
        CHECK(route->find("std::unique_lock<std::mutex> lk") == std::string::npos);
    }
}

// ---- 3. prompt text cap (413) ----------------------------------------------------------------

TEST_CASE("prompt cap: over 256 KiB is 413 and never touches the permit") {
    static_assert(kApiMaxPromptTextBytes == 256 * 1024);
    static_assert(kApiMaxPromptTextBytes < kHttpMaxBody);
    ComputeJobAdmission admission;
    int acquire_calls = 0;
    const auto acquire = [&] {
        ++acquire_calls;
        return admission.try_acquire();
    };

    TextAdmission over = admit_text_then_permit(kApiMaxPromptTextBytes + 1, acquire);
    CHECK(over.status == TextAdmitStatus::TooLarge);
    CHECK_FALSE(over.permit);
    CHECK(acquire_calls == 0); // the permit was not even asked for
    CHECK(admit_text_then_permit(kHttpMaxBody, acquire).status == TextAdmitStatus::TooLarge);
    CHECK(acquire_calls == 0);

    // The slot is free: 413 left it free; and while another job holds it, oversize is still 413
    // (not 503) and the holder keeps it.
    auto holder = admission.try_acquire();
    REQUIRE(holder);
    CHECK(admit_text_then_permit(kApiMaxPromptTextBytes + 1, acquire).status ==
          TextAdmitStatus::TooLarge);
    CHECK(acquire_calls == 0);
    CHECK(admit_text_then_permit(10, acquire).status == TextAdmitStatus::Busy);
    CHECK(acquire_calls == 1);
    holder.reset();

    TextAdmission at_cap = admit_text_then_permit(kApiMaxPromptTextBytes, acquire);
    CHECK(at_cap.status == TextAdmitStatus::Admitted);
    CHECK(at_cap.permit);
    CHECK(acquire_calls == 2);
}

TEST_CASE("handler order: the chat route asks for its permit through the text cap") {
    const std::string src = read_file(API_SERVER_CPP_PATH);
    const std::string chat =
        slice(src, "req.path == \"/v1/chat/completions\"", "EmbeddingApi embedding_api;");
    const size_t cap = chat.find("admit_text_then_permit(");
    const size_t direct = chat.find("acquire_compute_job_or_busy()");
    REQUIRE(cap != std::string::npos);
    REQUIRE(direct != std::string::npos);
    CHECK(cap < direct); // the only acquisition is the callback of the capped admission
    CHECK(chat.find("413 Payload Too Large") != std::string::npos);
}

// ---- 4. /api/pull: admission first, zero bytes when busy ------------------------------------

TEST_CASE("pull admission: permit busy is 503 with zero bytes written and the gate released") {
    ComputeJobAdmission admission;
    ApiPullGate gate;
    auto other_job = admission.try_acquire(); // an inference holds the slot
    REQUIRE(other_job);

    size_t bytes_written = 0;
    bool download_started = false;
    PullAdmission a = admit_pull(gate, [&] { return admission.try_acquire(); });
    if (a.status == PullAdmitStatus::Admitted) { // what the handler does after admission
        bytes_written += 64;                     // the "pulling manifest" line
        download_started = true;
    }
    CHECK(a.status == PullAdmitStatus::Busy);
    CHECK(bytes_written == 0);
    CHECK_FALSE(download_started);
    CHECK_FALSE(a.permit);
    CHECK_FALSE(a.gate.owns_lock());
    CHECK(gate.try_acquire().owns_lock()); // a later pull is not locked out by the refused one
    other_job.reset();
}

TEST_CASE("pull admission: gate busy is 409, admitted holds both until release()") {
    ComputeJobAdmission admission;
    ApiPullGate gate;
    PullAdmission first = admit_pull(gate, [&] { return admission.try_acquire(); });
    REQUIRE(first.status == PullAdmitStatus::Admitted);
    CHECK(first.permit);
    CHECK(first.gate.owns_lock());

    int asked = 0;
    PullAdmission second = admit_pull(gate, [&] {
        ++asked;
        return admission.try_acquire();
    });
    CHECK(second.status == PullAdmitStatus::PullInProgress);
    CHECK(asked == 0); // a second pull does not even reach for the permit
    CHECK_FALSE(admission.try_acquire());

    first.release(); // before the final reply write
    CHECK(admission.try_acquire());
    CHECK(gate.try_acquire().owns_lock());
}

TEST_CASE("handler order: handle_pull admits before the stream opens and before the download") {
    const std::string src = read_file(API_SERVER_CPP_PATH);
    const std::string pull =
        slice(src, "void handle_pull(", "// \"200 OK\" -> 200 for the activity record.");
    const size_t admit = pull.find("admit_pull(");
    const size_t stream = pull.find("std::make_unique<PullJsonStream>");
    const size_t download = pull.find("DownloadAsync(");
    REQUIRE(admit != std::string::npos);
    REQUIRE(stream != std::string::npos);
    REQUIRE(download != std::string::npos);
    CHECK(admit < stream);
    CHECK(stream < download);
    // No second, late acquisition after the download.
    CHECK(pull.find("acquire_compute_job_or_busy()", download) == std::string::npos);
    CHECK(pull.find("503 Service Unavailable", admit) < stream);
}

// ---- 5. /v1/preferences quotas ---------------------------------------------------------------

TEST_CASE("preferences quota: a flood stays bounded, the window slides") {
    PreferenceQuota quota;
    const auto t0 = PreferenceQuota::Clock::now();
    int ok = 0;
    int limited = 0;
    for (int i = 0; i < 1000; ++i) {
        const auto v = quota.admit(200, 0, t0 + std::chrono::milliseconds(i));
        ok += v == PreferenceQuota::Verdict::Ok;
        limited += v == PreferenceQuota::Verdict::RateLimited;
    }
    CHECK(ok == kPreferenceMaxPerWindow);
    CHECK(limited == 1000 - kPreferenceMaxPerWindow);
    // A window later it admits again.
    CHECK(quota.admit(200, 0, t0 + kPreferenceWindow + std::chrono::seconds(2)) ==
          PreferenceQuota::Verdict::Ok);
}

TEST_CASE("preferences quota: body size and file size are refused before the rate") {
    PreferenceQuota quota;
    const auto now = PreferenceQuota::Clock::now();
    CHECK(quota.admit(kPreferenceMaxBody, 0, now) == PreferenceQuota::Verdict::Ok);
    CHECK(quota.admit(kPreferenceMaxBody + 1, 0, now) == PreferenceQuota::Verdict::TooLarge);
    CHECK(quota.admit(10, kPreferenceMaxStoreBytes, now) == PreferenceQuota::Verdict::StoreFull);
    CHECK(quota.admit(10, kPreferenceMaxStoreBytes - 10, now) == PreferenceQuota::Verdict::Ok);
    // Refusals for size do not eat the rate budget.
    PreferenceQuota fresh;
    for (int i = 0; i < 100; ++i)
        CHECK(fresh.admit(kPreferenceMaxBody + 1, 0, now) == PreferenceQuota::Verdict::TooLarge);
    CHECK(fresh.admit(10, 0, now) == PreferenceQuota::Verdict::Ok);
}

TEST_CASE("handler order: handle_preferences checks the quota before it parses or writes") {
    const std::string src = read_file(API_SERVER_CPP_PATH);
    const std::string prefs =
        slice(src, "std::string handle_preferences(", "// GET /v1/training/status");
    const size_t quota = prefs.find("quota.admit(");
    const size_t parse = prefs.find("JsonObject::TryParse");
    const size_t write = prefs.find("append_preference_sample_file");
    REQUIRE(quota != std::string::npos);
    CHECK(quota < parse);
    CHECK(quota < write);
    CHECK(prefs.find("acquire_compute_job") ==
          std::string::npos); // no permit: inference is not blocked
}

// ---- 6. model spellings ----------------------------------------------------------------------

namespace {
struct StubEntry {
    std::wstring name;
    std::wstring env; // stands in for ManifestEntry::env
};
} // namespace

TEST_CASE("model spelling: an unknown spelling is not servable (404), none is folded or merged") {
    const std::wstring declared = L"D:\\DevelopmentFiles\\xllama\\models\\qwen35-9b";
    const std::vector<StubEntry> manifest{{L"qwen35-9b", L"A=1"}, {declared, L"A=2"}};
    const auto known = [&](const std::string& model, const std::string& resident = "",
                           const std::string& hint = "") {
        const std::wstring w(model.begin(), model.end());
        return path_spelling_known(model, find_model_spelling(manifest, w) != nullptr, resident,
                                   hint);
    };
    CHECK(known("D:\\DevelopmentFiles\\xllama\\models\\qwen35-9b"));
    // Other spellings of the same file are unknown: case, slash direction, drive letter, a trailing
    // slash.
    CHECK_FALSE(known("d:\\DevelopmentFiles\\xllama\\models\\qwen35-9b"));
    CHECK_FALSE(known("D:/DevelopmentFiles/xllama/models/qwen35-9b"));
    CHECK_FALSE(known("D:\\DevelopmentFiles\\xllama\\models\\qwen35-9b\\"));
    CHECK_FALSE(known("D:\\DevelopmentFiles\\xllama\\models\\Qwen35-9B"));
    CHECK_FALSE(known("E:\\anything\\else"));
    CHECK_FALSE(known(""));
    // The resident model and the operator's model.txt hint are exact matches too.
    CHECK(known("E:\\resident", "E:\\resident"));
    CHECK_FALSE(known("E:\\resident2", "E:\\resident"));
    CHECK(known("E:\\hint", "", "E:\\hint"));
    CHECK_FALSE(known("E:\\hint ", "", "E:\\hint"));
}

TEST_CASE("model spelling: a declared alias keeps its own entry and env") {
    const std::vector<StubEntry> manifest{{L"qwen35-9b", L"XLLAMA_ATTN_GPU=1"},
                                          {L"D:\\m\\qwen35-9b", L"XLLAMA_ATTN_GPU=0"},
                                          {L"d:\\m\\qwen35-9b", L"LLAMA_KV_ON_DEVICE=1"}};
    const StubEntry* bare = find_model_spelling(manifest, L"qwen35-9b");
    const StubEntry* upper = find_model_spelling(manifest, L"D:\\m\\qwen35-9b");
    const StubEntry* lower = find_model_spelling(manifest, L"d:\\m\\qwen35-9b");
    REQUIRE(bare);
    REQUIRE(upper);
    REQUIRE(lower);
    CHECK(bare->env == L"XLLAMA_ATTN_GPU=1");
    CHECK(upper->env == L"XLLAMA_ATTN_GPU=0"); // not merged into the bare id or its case variant
    CHECK(lower->env == L"LLAMA_KV_ON_DEVICE=1");
    CHECK(bare != upper);
    CHECK(upper != lower);
    CHECK(find_model_spelling(manifest, L"QWEN35-9B") == nullptr);
    CHECK(find_model_spelling(manifest, L"D:\\m\\QWEN35-9B") == nullptr);
}

TEST_CASE("model spelling: what counts as a path") {
    CHECK(model_id_is_path("D:\\m\\x"));
    CHECK(model_id_is_path("/mnt/x"));
    CHECK(model_id_is_path("a:b"));
    CHECK_FALSE(model_id_is_path("qwen35-9b"));
    CHECK_FALSE(model_id_is_path(""));
}

TEST_CASE("handler order: model_servable_locked no longer admits any path unconditionally") {
    const std::string src = read_file(API_SERVER_CPP_PATH);
    const std::string fn =
        slice(src, "bool model_servable_locked(", "std::string model_not_found(");
    CHECK(fn.find("path_spelling_known(") != std::string::npos);
    // A bare id is a directory of exactly that name, not whatever the filesystem folds it to.
    CHECK(fn.find("filename().string() != model") != std::string::npos);
}

// ---- 7. B3 batch route: /v1/xllama/batch ----------------------------------------------------

TEST_CASE("batch policy: XLLAMA_BATCH_SEQS parses to 1 unless it is a number of at least 2") {
    CHECK(batch_seqs_from_env(nullptr) == 1);
    CHECK(batch_seqs_from_env("") == 1);
    CHECK(batch_seqs_from_env("1") == 1);
    CHECK(batch_seqs_from_env("0") == 1);
    CHECK(batch_seqs_from_env("-3") == 1);
    CHECK(batch_seqs_from_env("abc") == 1);
    CHECK(batch_seqs_from_env("4x") == 1);
    CHECK(batch_seqs_from_env("2") == 2);
    CHECK(batch_seqs_from_env("4") == 4);
    CHECK(batch_seqs_from_env("999") == kMaxBatchSeqs);
}

// output_reserve asserts max(n_outputs, n_seq_max) <= n_outputs_max: the GPU path's cap of 1 (+
// draft) must rise to the sequence count, and llama.cpp's default (0 = n_batch) must stay.
TEST_CASE("batch policy: the logits-row cap covers every sequence of a batching context") {
    constexpr uint32_t kMax = static_cast<uint32_t>(kMaxBatchSeqs);
    CHECK(outputs_max_for_seqs(1u, 1u) == 1u);
    CHECK(outputs_max_for_seqs(1u, 4u) == 4u);
    CHECK(outputs_max_for_seqs(5u, 4u) == 5u);
    CHECK(outputs_max_for_seqs(4u, 4u) == 4u);
    CHECK(outputs_max_for_seqs(0u, 4u) == 0u);
    CHECK(outputs_max_for_seqs(1u, kMax) == kMax);
}

TEST_CASE("batch policy: empty and over-sequence batches are refused, batching off serves one") {
    CHECK(batch_admit_count(0, 4) == BatchAdmit::Empty);
    CHECK(batch_admit_count(1, 4) == BatchAdmit::Ok);
    CHECK(batch_admit_count(4, 4) == BatchAdmit::Ok);
    CHECK(batch_admit_count(5, 4) == BatchAdmit::TooMany);
    // XLLAMA_BATCH_SEQS unset: one request only.
    CHECK(batch_admit_count(1, 1) == BatchAdmit::Ok);
    CHECK(batch_admit_count(2, 1) == BatchAdmit::TooMany);
    CHECK(batch_admit_count(2, 0) == BatchAdmit::TooMany);
    CHECK(batch_admit_count(0, 1) == BatchAdmit::Empty);
}

TEST_CASE("batch policy: the summed prompts plus the longest reply must fit n_ctx") {
    CHECK(batch_admit_tokens({100, 200, 300}, {50, 100, 10}, 700) == BatchAdmit::Ok);
    CHECK(batch_admit_tokens({100, 200, 300}, {50, 101, 10}, 700) == BatchAdmit::TooManyTokens);
    // Each prompt alone would fit; together they do not.
    CHECK(batch_admit_tokens({400, 400}, {1, 1}, 512) == BatchAdmit::TooManyTokens);
    CHECK(batch_admit_tokens({2000000000, 2000000000}, {1}, 4096) == BatchAdmit::TooManyTokens);
}

// The shared decodes take the same count from every sequence (no split_equal tail);
// the rest of each prompt is its own pass.
TEST_CASE(
    "batch policy: the prefill plan shares equal counts and leaves the rest to each own pass") {
    using V = std::vector<int>;
    SUBCASE("equal lengths: everything but the last token is shared, one decode") {
        const BatchPrefillPlan p = batch_prefill_plan({10, 10, 10}, 2048);
        CHECK(p.shared == 9);
        CHECK(p.shared_chunks == V{9});
        CHECK(p.n_shared_decodes() == 1);
        CHECK(p.own == V{1, 1, 1});
        CHECK(p.own_decodes == V{1, 1, 1});
    }
    SUBCASE("a one-token prompt: E = 0, no shared decode, whole prompts alone") {
        const BatchPrefillPlan p = batch_prefill_plan({1, 7, 12}, 2048);
        CHECK(p.shared == 0);
        CHECK(p.n_shared_decodes() == 0);
        CHECK(p.own == V{1, 7, 12});
        CHECK(p.own_decodes == V{1, 1, 1});
    }
    SUBCASE("four unequal lengths: E = shortest - 1 from each, in order") {
        const BatchPrefillPlan p = batch_prefill_plan({13, 5, 20, 9}, 2048);
        CHECK(p.shared == 4);
        CHECK(p.shared_chunks == V{4});
        CHECK(p.own == V{9, 1, 16, 5});
        CHECK(p.own_decodes == V{1, 1, 1, 1});
        // n_batch 8: 2 per sequence per chunk, ceil(4 * 4 / 8) = 2 decodes.
        const BatchPrefillPlan q = batch_prefill_plan({13, 5, 20, 9}, 8);
        CHECK(q.shared_chunks == V{2, 2});
        CHECK(q.own_decodes == V{2, 1, 2, 1});
    }
    SUBCASE("a remainder above n_batch is chunked on its own sequence") {
        const BatchPrefillPlan p = batch_prefill_plan({3, 40}, 16);
        CHECK(p.shared == 2);
        CHECK(p.shared_chunks == V{2});
        CHECK(p.own == V{1, 38});
        CHECK(p.own_decodes == V{1, 3});
    }
    SUBCASE("n_batch below k: no room for an equal chunk, nothing shared") {
        const BatchPrefillPlan p = batch_prefill_plan({10, 10, 10}, 2);
        CHECK(p.shared == 0);
        CHECK(p.n_shared_decodes() == 0);
        CHECK(p.own_decodes == V{5, 5, 5});
    }
    SUBCASE("k not dividing n_batch: ceil(E / floor(n_batch / k)) decodes") {
        // n_batch 16, k 3: 5 per sequence (15 rows). E = 15 → 3 = ceil(45 / 16);
        // E = 16 → 4, one above ceil(48 / 16) = 3.
        CHECK(batch_prefill_plan({16, 16, 16}, 16).shared_chunks == V{5, 5, 5});
        CHECK(batch_prefill_plan({17, 17, 17}, 16).shared_chunks == V{5, 5, 5, 1});
    }
    SUBCASE("invariants over a sweep") {
        for (const int n_batch : {1, 2, 3, 7, 16, 512})
            for (const V& lengths : {V{1}, V{2, 2}, V{5, 3, 9}, V{30, 31, 29, 100}, V{1, 1000}}) {
                CAPTURE(n_batch);
                const BatchPrefillPlan p = batch_prefill_plan(lengths, n_batch);
                const int k = static_cast<int>(lengths.size());
                int sum = 0;
                for (const int c : p.shared_chunks) {
                    CHECK(c > 0);
                    CHECK(c * k <= n_batch);
                    sum += c;
                }
                CHECK(sum == p.shared);
                REQUIRE(p.own.size() == lengths.size());
                for (size_t i = 0; i < lengths.size(); ++i) {
                    CHECK(p.own[i] == lengths[i] - p.shared);
                    CHECK(p.own[i] >= 1);
                    CHECK(p.own_decodes[i] == (p.own[i] + n_batch - 1) / n_batch);
                }
            }
    }
    CHECK(batch_prefill_plan({}, 512).own.empty());
}

TEST_CASE("handler order: the batch route is behind auth and the text cap, and releases first") {
    const std::string src = read_file(API_SERVER_CPP_PATH);
    REQUIRE_FALSE(src.empty());
    const size_t auth = src.find("if (!auth.allow) {");
    const size_t route_at = src.find("req.path == \"/v1/xllama/batch\"");
    REQUIRE(auth != std::string::npos);
    REQUIRE(route_at != std::string::npos);
    CHECK(auth < route_at); // /health is the only route before the Bearer check
    const std::string route =
        slice(src, "req.path == \"/v1/xllama/batch\"", "req.path == \"/v1/chat/completions\"");
    const size_t cap = route.find("admit_text_then_permit(");
    const size_t direct = route.find("acquire_compute_job_or_busy()");
    const size_t hub = route.find("acquire_hub_or_busy()");
    const size_t live = route.find("generation == g_generation");
    const size_t handler = route.find("handle_batch_locked(");
    REQUIRE(cap != std::string::npos);
    REQUIRE(direct != std::string::npos);
    REQUIRE(hub != std::string::npos);
    REQUIRE(live != std::string::npos);
    REQUIRE(handler != std::string::npos);
    CHECK(cap < direct);
    CHECK(direct < hub);
    CHECK(hub < live);
    CHECK(live < handler);
    CHECK(route.find("413 Payload Too Large") != std::string::npos);
    const size_t release = route.rfind("scope.release();");
    const size_t final_write = route.rfind("write_response(socket, status, json);");
    REQUIRE(release != std::string::npos);
    REQUIRE(final_write != std::string::npos);
    CHECK(release < final_write);
    CHECK(route.find("auto compute_permit") == std::string::npos);
    CHECK(route.find("std::unique_lock<std::mutex> lk") == std::string::npos);
}

TEST_CASE("handler order: handle_batch_locked checks spelling, count, and summed tokens") {
    const std::string src = read_file(API_SERVER_CPP_PATH);
    const std::string fn =
        slice(src, "std::string handle_batch_locked(", "// Serve one accepted socket.");
    const size_t spelling = fn.find("model_servable_locked(model)");
    const size_t ceiling = fn.find("batch_admit_count(k, ::xllama::kMaxBatchSeqs)");
    const size_t load = fn.find("ensure_locked(");
    const size_t seqs = fn.find("batch_admit_count(k, seqs)");
    const size_t plan = fn.find("plan_chat_locked(");
    const size_t summed = fn.find("batch_admit_tokens(");
    const size_t generate = fn.find("generate_batch(");
    for (size_t at : {spelling, ceiling, load, seqs, plan, summed, generate})
        REQUIRE(at != std::string::npos);
    CHECK(spelling < load); // an unknown spelling is 404 before any model load
    CHECK(ceiling < load);
    CHECK(load < seqs);
    CHECK(seqs < plan);
    CHECK(plan < summed); // every prompt fitted (fit_prompt) before the sum
    CHECK(summed < generate);
    CHECK(fn.find("read_chat_limits(") < load); // max_tokens and sampling before the load (F01)
}
