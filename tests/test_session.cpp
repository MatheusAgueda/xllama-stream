// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/path_utils.h"
#include "xllama/session.h"
#include "xllama/session_hub.h"

#include <chrono>
#include <doctest/doctest.h>

#include <climits>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>

TEST_CASE("SessionHub: failed create leaves the hub empty and bumps generation") {
    auto& hub = xllama::session_hub();
    std::lock_guard<std::mutex> lk(hub.mtx);
    const uint64_t g0 = hub.generation;
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/hub-model.gguf";
    std::string err;
    CHECK(hub.ensure_locked("hub-model", sp, &err) == nullptr);
    CHECK(hub.session == nullptr);
    CHECK(hub.model.empty());
    // The attempted swap must invalidate any holder's generation even though
    // creation failed — its old session pointer is gone either way.
    CHECK(hub.generation == g0 + 1);
    hub.reset_locked(); // no-op when already empty
    CHECK(hub.generation == g0 + 1);
}

TEST_CASE("resolve_max_length: the #130 ladder has one home") {
    using xllama::resolve_max_length;
    // override < 0: saturate to n_ctx — what Session always requests.
    CHECK(resolve_max_length(2048, 1289, 256, -1) == 2048);
    CHECK(resolve_max_length(3072, 10, 96, -1) == 3072);
    // override == 0: derive min(n_ctx, prompt + n_predict) — bench default,
    // keeps every historical row's meaning.
    CHECK(resolve_max_length(2048, 1289, 256, 0) == 1545); // the shipping default
                                                           // that landed in the valley
    CHECK(resolve_max_length(2048, 1800, 512, 0) == 2048); // clamped to context
    // override > 0: explicit, clamped to (n_prompt, n_ctx].
    CHECK(resolve_max_length(2048, 1289, 256, 1801) == 1801);
    CHECK(resolve_max_length(2048, 1289, 256, 100) == 1290);  // floor: prompt+1
    CHECK(resolve_max_length(2048, 1289, 256, 9999) == 2048); // ceiling: n_ctx
    // F01: an INT_MAX n_predict saturates instead of overflowing.
    CHECK(resolve_max_length(2048, 1289, INT_MAX, 0) == 2048);
}

TEST_CASE("Session::create rejects non-existent model path") {
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/path/model.gguf";
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    CHECK(s == nullptr);
    CHECK(!err.empty());
}

TEST_CASE("Session::create rejects empty model path") {
    xllama::SessionParams sp;
    sp.model_path = "";
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    CHECK(s == nullptr);
    CHECK(!err.empty());
}

TEST_CASE("model_uses_llama_backend detects .gguf suffix and dir layout") {
    using xllama::model_uses_llama_backend;

    CHECK(model_uses_llama_backend("foo.gguf"));
    CHECK(model_uses_llama_backend("/abs/path/model.Q4_K.gguf"));
    CHECK_FALSE(model_uses_llama_backend("smollm2-360m"));
    CHECK_FALSE(model_uses_llama_backend("some-ort-model-dir"));

    // Temp dir containing a .gguf file (simulates catalogue layout on disk)
    auto tmp = std::filesystem::temp_directory_path() / "xllama-gguf-layout-test";
    std::filesystem::create_directories(tmp);
    {
        std::ofstream f((tmp / "model.gguf").string());
        f << "dummy";
    }
    CHECK(model_uses_llama_backend(tmp.string()));

    // Dir without .gguf should be false
    auto tmp2 = std::filesystem::temp_directory_path() / "xllama-no-gguf";
    std::filesystem::create_directories(tmp2);
    CHECK_FALSE(model_uses_llama_backend(tmp2.string()));

    // Cleanup
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::remove_all(tmp2, ec);
}

TEST_CASE("first_gguf_in_dir prefers model.gguf over adapter.gguf") {
    using xllama::first_gguf_in_dir;
    auto tmp = std::filesystem::temp_directory_path() / "xllama-gguf-prefer-test";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp);
    {
        std::ofstream a((tmp / "adapter.gguf").string());
        a << "adapter";
        std::ofstream m((tmp / "model.gguf").string());
        m << "model";
    }
    const std::string picked = first_gguf_in_dir(tmp.string(), "adapter.gguf");
    CHECK(picked.find("model.gguf") != std::string::npos);
    // Without exclude, model.gguf still preferred by name
    const std::string picked2 = first_gguf_in_dir(tmp.string());
    CHECK(picked2.find("model.gguf") != std::string::npos);
    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("Session::create fails when LoRA path is invalid (llama)") {
    // Real base not required: create fails at model load first if path is junk.
    // Here we only assert empty model + lora still rejects cleanly.
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/base.gguf";
    sp.lora_path = "/nonexistent/adapter.gguf";
    sp.backend = xllama::Backend::LlamaCpp;
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    CHECK(s == nullptr);
    CHECK(!err.empty());
}

TEST_CASE("Session::create Auto with explicit Backend::LlamaCpp on GGUF layout") {
    // Even without real weights the dispatch must select the llama path.
    // We only assert that it does not succeed via the ORT path and that an error is produced.
    auto tmp = std::filesystem::temp_directory_path() / "xllama-session-gguf-auto";
    std::filesystem::create_directories(tmp);
    {
        std::ofstream f((tmp / "mini.gguf").string());
        f << "not-a-real-gguf";
    }

    xllama::SessionParams sp;
    sp.model_path = tmp.string();
    sp.backend = xllama::Backend::Auto; // should resolve to Llama via layout

    std::string err;
    auto s = xllama::Session::create(sp, &err);
    // On Linux this build only has llama; we expect a load failure from the llama code path.
    CHECK(s == nullptr);
    CHECK(!err.empty());

    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
}

// #170a: a full-prompt turn rewinds the resident KV to the common token prefix
// instead of clearing. Opt-in — needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: KV prefix reuse collapses a repeated prefill (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    auto mkgp = [](const std::string& p) {
        xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 12;
        gp.temperature = 0.0f; // greedy: outputs must be reproducible
        return gp;
    };
    const std::string prompt = "The capital of France is";

    const auto r_cold = sess->generate(mkgp(prompt));
    REQUIRE(r_cold.success);
    REQUIRE(r_cold.n_p_eval > 1);
    CHECK(r_cold.n_prompt == r_cold.n_p_eval);

    // Regenerate: identical prompt on the same session. Two legal regimes:
    // pure-attention caches rewind and re-prefill exactly 1 token (the logits
    // source); hybrid caches (LFM2) cannot erase a partial tail — seq_rm
    // returns false — so the session degrades to a full clear + re-prefill.
    // Byte-identical output is the correctness bar in both regimes.
    const auto r_regen = sess->generate(mkgp(prompt));
    REQUIRE(r_regen.success);
    CHECK(r_regen.n_p_eval <= r_cold.n_p_eval);
    // #335: the prompt is as long as before even when only its tail prefilled;
    // usage.prompt_tokens reads n_prompt, not n_p_eval.
    CHECK(r_regen.n_prompt == r_cold.n_prompt);
    CHECK(r_regen.output_text == r_cold.output_text);

    // Extended prompt (the LAN-API shape). This prompt diverges from the
    // resident generated tail right after the original prompt, so it exercises
    // the tail-removal path in both regimes.
    const bool full_rewind = r_regen.n_p_eval == 1;
    const std::string ext = prompt + " Paris, and the capital of Italy is";
    const auto r_ext = sess->generate(mkgp(ext));
    REQUIRE(r_ext.success);
    CHECK(r_ext.n_prompt == sess->count_tokens(ext));

    auto fresh = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(fresh, err);
    const auto r_fresh = fresh->generate(mkgp(ext));
    REQUIRE(r_fresh.success);
    if (full_rewind) {
        // Pure-attention: only the extension past the common prefix re-prefills.
        CHECK(r_ext.n_p_eval < sess->count_tokens(ext));
        // Exact equality is the wrong bar here (it holds for the regenerate
        // case above, where the resident bytes are identical): the resident
        // prefix was accumulated in a different batch shape than the fresh
        // session's single prefill, so the K/V last bits differ and greedy can
        // flip at a NEAR-TIE downstream — inherent to every prefix cache,
        // observed as " Rome." vs " Rome.\n\nThe answer is Paris.". The
        // correctness signal is the leading strong-signal token agreeing.
        REQUIRE(r_fresh.output_text.size() >= 5);
        REQUIRE(r_ext.output_text.size() >= 5);
        CHECK(r_ext.output_text.substr(0, 5) == r_fresh.output_text.substr(0, 5));
    } else if (r_ext.n_p_eval == sess->count_tokens(ext)) {
        // Hybrid (LFM2, Qwen3.5) without a usable checkpoint: the divergent
        // tail cannot be erased — seq_rm refuses — so the session degraded to a
        // full clear + re-prefill, split where a fresh session splits it too
        // (#170c): byte-identical output.
        CHECK(r_ext.output_text == r_fresh.output_text);
    } else {
        // Hybrid with a #170c checkpoint inside the common prefix: the restored
        // state was prefilled in other batch shapes than the fresh session's,
        // so the same near-tie caveat as the pure-attention rewind applies.
        REQUIRE(r_fresh.output_text.size() >= 5);
        REQUIRE(r_ext.output_text.size() >= 5);
        CHECK(r_ext.output_text.substr(0, 5) == r_fresh.output_text.substr(0, 5));
    }
}

// The first characters two greedy runs must share when their prefixes were
// prefilled in different batch shapes (see the near-tie note above).
static void check_same_lead(const std::string& a, const std::string& b) {
    const size_t n = std::min<size_t>(5, std::min(a.size(), b.size()));
    REQUIRE(n >= 3);
    CHECK(a.substr(0, n) == b.substr(0, n));
}

// #170c: a LAN-API turn re-sends the whole conversation with the previous
// generation prompt re-rendered as history, so it diverges 4 tokens before the
// previous prompt's end (Qwen drops "<think>\n\n</think>\n\n" from history). A
// hybrid cache cannot erase that tail — before #170c it re-prefilled from token
// 0 (8-10 s per turn on the console). The recurrent checkpoint at prompt end
// minus 4 makes it prefill only the new turn, like a pure-attention rewind.
// Opt-in — needs a real GGUF with a ChatML template:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: an API turn prefills only the new turn (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 1024;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    auto mkgp = [](const std::string& p) {
        xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 10;
        gp.temperature = 0.0f;
        return gp;
    };
    const std::string sys = "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n";
    const std::string gen = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const std::string q1 = "<|im_start|>user\nName a colour.<|im_end|>\n";
    const std::string turn1 = sys + q1 + gen;
    const auto r1 = sess->generate(mkgp(turn1));
    REQUIRE_MESSAGE(r1.success, r1.error_msg);
    REQUIRE(!r1.output_text.empty());

    const std::string turn2 = sys + q1 + "<|im_start|>assistant\n" + r1.output_text +
                              "<|im_end|>\n<|im_start|>user\nAnd a fruit?<|im_end|>\n" + gen;
    const int n1 = sess->count_tokens(turn1);
    const int n2 = sess->count_tokens(turn2);
    const auto r2 = sess->generate(mkgp(turn2));
    REQUIRE_MESSAGE(r2.success, r2.error_msg);
    CHECK(r2.n_prompt == n2);
    // Everything up to the previous turn's "<|im_start|>assistant\n" stays.
    CHECK(r2.n_p_eval <= n2 - (n1 - 4));

    auto fresh = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(fresh, err);
    const auto r2_fresh = fresh->generate(mkgp(turn2));
    REQUIRE_MESSAGE(r2_fresh.success, r2_fresh.error_msg);
    CHECK(r2_fresh.n_p_eval == n2);
    check_same_lead(r2.output_text, r2_fresh.output_text);

    // Regenerate: the same prompt again. The checkpoint at its end minus 4 holds
    // exactly the state the first run had there, and the last 4 tokens go
    // through in the same batch, so the output is byte-identical.
    const auto r3 = sess->generate(mkgp(turn2));
    REQUIRE_MESSAGE(r3.success, r3.error_msg);
    CHECK(r3.n_p_eval <= 4);
    CHECK(r3.output_text == r2.output_text);
}

// #170c: one-shot API calls that share a system prompt diverge right after it.
// The divergence point of the first refused rewind becomes a checkpoint, so
// the third call prefills only its own question.
TEST_CASE("Session: one-shot calls reuse their shared system prompt (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 1024;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    auto call = [&](const std::string& question) {
        xllama::GenerateParams gp;
        gp.prompt = "<|im_start|>system\nYou answer questions about the solar system in one "
                    "short sentence, with no preamble and no follow-up "
                    "question.<|im_end|>\n<|im_start|>user\n" +
                    question + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        gp.n_predict = 8;
        gp.temperature = 0.0f;
        return std::make_pair(sess->generate(gp), sess->count_tokens(gp.prompt));
    };
    const auto [r1, n1] = call("Which planet is largest?");
    REQUIRE_MESSAGE(r1.success, r1.error_msg);
    const auto [r2, n2] = call("Which planet is hottest?");
    REQUIRE_MESSAGE(r2.success, r2.error_msg);
    const auto [r3, n3] = call("Which planet has rings?");
    REQUIRE_MESSAGE(r3.success, r3.error_msg);
    // The system prompt alone is over 20 tokens; the third call prefills its
    // question and the generation prompt only.
    CHECK(r3.n_p_eval < n3 - 20);
}

// A prompt longer than the logical batch must be prefilled in chunks. This is
// not a "nicer error" test: llama_decode ASSERTS n_tokens <= n_batch (GGML_ABORT,
// Release included), so before the chunking this test process died. n_batch
// defaults to min(n_ctx, 2048), so the regime only exists above the shipping
// context — which is exactly what the coding catalogue opens (n_ctx 4096, ceiling
// 3846 prompt tokens). Opt-in — needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: a prompt past n_batch prefills in chunks (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    // Same regime as a 4096-token coding session with the default 2048 batch,
    // at 1/30th of the tokens: what matters is prompt > n_batch, and n_batch is
    // a knob. A 2100-token prefill in a Debug build takes minutes; this runs in
    // seconds and fails identically (an abort, not a failed assertion) without
    // the chunking.
    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    sp.n_batch = 64;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    // Grow past n_batch with the model's own tokenizer, so the test does not
    // depend on one vocabulary's density.
    std::string prompt = "Summarize the following log.\n";
    int n_prompt = 0;
    for (int i = 0; i < 200 && n_prompt <= 96; ++i) {
        prompt += "line " + std::to_string(i) + ": nothing happened\n";
        if ((i % 4) == 3)
            n_prompt = sess->count_tokens(prompt);
    }
    n_prompt = sess->count_tokens(prompt);
    REQUIRE(n_prompt > sp.n_batch); // past the logical batch: the regime under test
    REQUIRE(n_prompt < 400);        // still inside n_ctx with room to generate

    xllama::GenerateParams gp;
    gp.prompt = prompt;
    gp.n_predict = 8;
    gp.temperature = 0.0f; // greedy: the two sessions below must agree
    const auto r = sess->generate(gp);
    REQUIRE_MESSAGE(r.success, r.error_msg); // before the chunking: SIGABRT here
    CHECK(r.n_p_eval == n_prompt);           // every token was prefilled, not just the first chunk

    // And chunking must not change what the model reads: the same prompt in ONE
    // batch (default n_batch, well above this prompt) sees the same context.
    // Only the leading token is compared — a different batch shape moves the
    // last K/V bits, so greedy can flip at a near-tie downstream (same caveat as
    // the #170a prefix-reuse case above).
    sp.n_batch = 0;
    auto whole = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(whole, err);
    const auto r_whole = whole->generate(gp);
    REQUIRE(r_whole.success);
    CHECK(r_whole.n_p_eval == n_prompt);
    CHECK(r.n_eval == r_whole.n_eval);
    if (!r.output_text.empty() && !r_whole.output_text.empty())
        CHECK(r.output_text.substr(0, 1) == r_whole.output_text.substr(0, 1));
}

// A full prompt that cannot fit the context has nothing to evict — the trimmer
// upstream owns that — so it must fail fast rather than reach the decode, where
// an over-context batch is an abort. Opt-in:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: a prompt past n_ctx fails fast (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    std::string prompt;
    for (int i = 0; i < 400; ++i)
        prompt += "token " + std::to_string(i) + " ";
    REQUIRE(sess->count_tokens(prompt) > 512);

    xllama::GenerateParams gp;
    gp.prompt = prompt;
    gp.n_predict = 8;
    const auto r = sess->generate(gp);
    CHECK_FALSE(r.success);
    CHECK(r.error_msg.find("prompt too long") != std::string::npos);
    CHECK(r.n_eval == 0);
}

// #170b: the resident KV round-trips through a file, so a conversation the
// session no longer holds resumes without re-reading its history. Opt-in —
// needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: KV state round-trips through a file (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    const auto state_path = std::filesystem::temp_directory_path() / "xllama-kv-state.bin";
    std::error_code ec;
    std::filesystem::remove(state_path, ec);

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 2048; // the shipping context: file size scales with it
    std::string err;
    auto a = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(a, err);

    auto mkgp = [](const std::string& p, bool reset) {
        xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 16;
        gp.temperature = 0.0f; // greedy: the two sessions must agree exactly
        gp.reuse_kv = true;
        gp.reset_kv = reset;
        return gp;
    };
    // A conversation of realistic length: re-reading exactly this is the cost
    // the file exists to avoid, and its size is what the eviction policy has
    // to budget for.
    std::string filler;
    for (int i = 0; i < 80; ++i)
        filler += "Item " + std::to_string(i) +
                  ": the harbour master filed a report on tide tables and cargo manifests. ";
    const std::string seed_prompt =
        "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\n" + filler +
        "The capital of France is Paris. The capital of Italy is Rome. Remember "
        "both.<|im_end|>\n<|im_start|>assistant\n";
    const std::string follow = "<|im_end|>\n<|im_start|>user\nWhich city did I name "
                               "first?<|im_end|>\n<|im_start|>assistant\n";

    const auto r_seed = a->generate(mkgp(seed_prompt, /*reset=*/true));
    REQUIRE_MESSAGE(r_seed.success, r_seed.error_msg);

    const auto t0 = std::chrono::steady_clock::now();
    const bool saved = a->save_state(state_path.string(), &err);
    const double save_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    REQUIRE_MESSAGE(saved, err);
    const auto bytes = std::filesystem::file_size(state_path, ec);
    MESSAGE("state file: " << bytes / 1024 << " KiB for " << r_seed.n_p_eval << " prompt tokens ("
                           << bytes / 1024 / (r_seed.n_p_eval ? r_seed.n_p_eval : 1)
                           << " KiB/token), save " << save_ms << " ms, prefill was "
                           << r_seed.t_p_eval_ms << " ms");

    // The continuation the resident session would produce, for comparison.
    const auto r_cont_resident = a->generate(mkgp(follow, /*reset=*/false));
    REQUIRE_MESSAGE(r_cont_resident.success, r_cont_resident.error_msg);

    // A second session that never saw the conversation: restoring the file
    // must put it in the same state, not merely a similar one.
    auto b = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(b, err);
    const auto t1 = std::chrono::steady_clock::now();
    const bool loaded = b->load_state(state_path.string(), &err);
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    REQUIRE_MESSAGE(loaded, err);
    MESSAGE("load " << load_ms << " ms");

    const auto r_cont_restored = b->generate(mkgp(follow, /*reset=*/false));
    REQUIRE_MESSAGE(r_cont_restored.success, r_cont_restored.error_msg);
    // Only the follow-up is prefilled — the history came from the file.
    CHECK(r_cont_restored.n_p_eval == r_cont_resident.n_p_eval);
    CHECK(r_cont_restored.n_p_eval < r_seed.n_p_eval);
    CHECK(r_cont_restored.output_text == r_cont_resident.output_text);

    // A file whose fingerprint does not match must be refused, not fed to the
    // cache: the pin checks shape only, so this is the whole defence against
    // loading another model's KV.
    xllama::SessionParams sp2 = sp;
    sp2.n_ctx = 1024; // same weights, different context configuration
    auto c = xllama::Session::create(sp2, &err);
    REQUIRE_MESSAGE(c, err);
    err.clear();
    CHECK_FALSE(c->load_state(state_path.string(), &err));
    CHECK(!err.empty());
    // Refusal must leave the session usable.
    const auto r_after = c->generate(mkgp("<|im_start|>user\nHi.<|im_end|>\n"
                                          "<|im_start|>assistant\n",
                                          /*reset=*/true));
    CHECK(r_after.success);

    std::filesystem::remove(state_path, ec);
}

// #169: a continuation turn that would overflow n_ctx evicts the oldest
// resident tokens past n_keep (front-drop + RoPE shift) instead of failing,
// so a long chat never leaves the reuse regime. Opt-in — needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE(
    "Session: context shift keeps continuations alive past n_ctx (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 256; // small on purpose: a few turns must overflow it
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    const std::string sys = "<|im_start|>system\nYou are terse.<|im_end|>\n";
    const int n_keep = sess->count_tokens(sys);
    REQUIRE(n_keep > 0);

    // Seed the persistent context (reuse + reset).
    xllama::GenerateParams seed;
    seed.prompt = sys + "<|im_start|>user\nSay ok.<|im_end|>\n<|im_start|>assistant\n";
    seed.n_predict = 32;
    seed.temperature = 0.0f;
    seed.reuse_kv = true;
    seed.reset_kv = true;
    seed.n_keep = n_keep;
    const auto r_seed = sess->generate(seed);
    REQUIRE_MESSAGE(r_seed.success, r_seed.error_msg);

    // Enough continuation turns to overflow n_ctx=256 several times over.
    // Capability decides the contract (known once the lazy context exists):
    // LFM2 and pure-attention archs shift, so every turn must succeed;
    // an arch that cannot shift (Qwen3.5's imrope cache — seq_add would be
    // a GGML_ASSERT abort, which the gate must prevent) keeps the #173
    // fail-fast, so the overflowing turn must fail CLEANLY with n_eval == 0 —
    // the exact signature the UI's full-prefill retry keys on.
    const bool shifts = sess->can_context_shift();
    int total_tokens = sess->count_tokens(seed.prompt) + r_seed.n_eval;
    bool saw_clean_overflow = false;
    for (int i = 0; i < 8; ++i) {
        xllama::GenerateParams cont;
        cont.prompt = "<|im_end|>\n<|im_start|>user\nAnd again, tell me more "
                      "about that, please.<|im_end|>\n<|im_start|>assistant\n";
        cont.n_predict = 48;
        cont.temperature = 0.0f;
        cont.reuse_kv = true;
        cont.reset_kv = false;
        cont.n_keep = n_keep;
        const auto rc = sess->generate(cont);
        if (shifts) {
            REQUIRE_MESSAGE(rc.success, rc.error_msg);
            CHECK(rc.n_p_eval > 0);
        } else if (!rc.success) {
            CHECK(rc.n_eval == 0);
            CHECK(rc.error_msg.find("context full") != std::string::npos);
            saw_clean_overflow = true;
            break;
        }
        total_tokens += rc.n_p_eval + rc.n_eval;
    }
    if (shifts) {
        // The conversation genuinely outgrew the context — the loop above
        // only proves the point if eviction actually had to happen.
        CHECK(total_tokens > sp.n_ctx);
    } else {
        CHECK(saw_clean_overflow);
    }
}

// Snapshot restoration is not a promise that a hybrid recurrent state can
// rewind to edited/re-rendered history. Compare its fallback with a fresh run.
TEST_CASE("Session: divergent restored snapshot matches cold prefill (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    const auto path = std::filesystem::temp_directory_path() / "xllama-divergent-state.bin";
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 512;
    sp.n_threads = 2;
    std::string error;
    auto source = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(source, error);
    xllama::GenerateParams gp;
    gp.prompt = "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n";
    gp.n_predict = 12;
    gp.temperature = 0;
    REQUIRE(source->generate(gp).success);
    REQUIRE_MESSAGE(source->save_state(path.string(), &error), error);
    source.reset();
    auto restored = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(restored, error);
    REQUIRE_MESSAGE(restored->load_state(path.string(), &error), error);
    gp.prompt += "Edited assistant reply.<|im_end|>\n<|im_start|>user\nSay goodbye."
                 "<|im_end|>\n<|im_start|>assistant\n";
    const auto actual = restored->generate(gp);
    REQUIRE_MESSAGE(actual.success, actual.error_msg);
    restored.reset();
    auto cold = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(cold, error);
    const auto expected = cold->generate(gp);
    REQUIRE_MESSAGE(expected.success, expected.error_msg);
    CHECK(actual.n_p_eval <= expected.n_p_eval);
    if (actual.n_p_eval == expected.n_p_eval)
        CHECK(actual.output_text == expected.output_text);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// Scoring: a greedy reply with its log-probabilities, then the same ids forced
// back on a cold session. The forced reply runs the same per-token decode with
// only the sampling step replaced, so it must reproduce the reply, its ids and
// its log-probabilities; a forced non-argmax id scores below the argmax.
TEST_CASE("Session: a forced reply reproduces the greedy logprobs (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 512;
    sp.n_threads = 2;
    std::string error;
    auto first = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(first, error);
    xllama::GenerateParams gp;
    gp.prompt = "<|im_start|>user\nCount from one to five in words.<|im_end|>\n"
                "<|im_start|>assistant\n";
    gp.n_predict = 16;
    gp.temperature = 0;
    gp.logprobs = true;
    const auto greedy = first->generate(gp);
    REQUIRE_MESSAGE(greedy.success, greedy.error_msg);
    REQUIRE(greedy.n_eval > 1);
    REQUIRE(greedy.logprobs.size() == static_cast<size_t>(greedy.n_eval));
    for (const auto& t : greedy.logprobs) {
        CHECK(t.logprob <= 0.0f);
        CHECK(t.argmax);
    }
    first.reset();

    auto cold = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(cold, error);
    xllama::GenerateParams fp = gp;
    fp.logprobs = false; // implied by forced_tokens
    for (const auto& t : greedy.logprobs)
        fp.forced_tokens.push_back(t.id);
    fp.n_predict = static_cast<int>(fp.forced_tokens.size()) + 8; // the ids end it
    const auto forced = cold->generate(fp);
    REQUIRE_MESSAGE(forced.success, forced.error_msg);
    CHECK(forced.n_eval == greedy.n_eval);
    CHECK(forced.output_text == greedy.output_text);
    REQUIRE(forced.logprobs.size() == greedy.logprobs.size());
    for (size_t i = 0; i < forced.logprobs.size(); ++i) {
        CAPTURE(i);
        CHECK(forced.logprobs[i].id == greedy.logprobs[i].id);
        CHECK(forced.logprobs[i].argmax);
        CHECK(forced.logprobs[i].logprob ==
              doctest::Approx(greedy.logprobs[i].logprob).epsilon(1e-4));
    }

    auto other = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(other, error);
    xllama::GenerateParams op = gp;
    op.forced_tokens = {greedy.logprobs[0].id + 1};
    const auto off = other->generate(op);
    REQUIRE_MESSAGE(off.success, off.error_msg);
    REQUIRE(off.logprobs.size() == 1);
    CHECK_FALSE(off.logprobs[0].argmax);
    CHECK(off.logprobs[0].logprob < greedy.logprobs[0].logprob);

    op.forced_tokens = {-1};
    const auto bad = other->generate(op);
    CHECK_FALSE(bad.success);
    CHECK(bad.error_msg.find("outside the vocabulary") != std::string::npos);
}

// --- Prefix cache on disk (XLLAMA_PREFIX_CACHE_MIB) ---------------------------

// The *.kv files under a prefix-cache root, with their sizes.
static std::map<std::string, uintmax_t> prefix_files(const std::filesystem::path& root) {
    std::map<std::string, uintmax_t> out;
    std::error_code ec;
    for (std::filesystem::recursive_directory_iterator it(root, ec), end; !ec && it != end;
         it.increment(ec))
        if (it->path().extension() == ".kv")
            out[it->path().string()] = std::filesystem::file_size(it->path(), ec);
    return out;
}

static xllama::GenerateParams pc_greedy(const std::string& prompt) {
    xllama::GenerateParams gp;
    gp.prompt = prompt;
    gp.n_predict = 12;
    gp.temperature = 0.0f;
    gp.save_prefix = true;
    return gp;
}

// Two unrelated API conversations, each a ~300-token first turn (past
// XLLAMA_PREFIX_CACHE_MIN) and a second turn that re-sends the first.
struct PcConversation {
    std::string sys, user;
    std::string turn1() const {
        return sys + "<|im_start|>user\n" + user +
               "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }
    std::string turn2(const std::string& reply) const {
        return sys + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" + reply +
               "<|im_end|>\n<|im_start|>user\nNow say it in five words.<|im_end|>\n"
               "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }
};

static PcConversation pc_conversation(const std::string& role, const std::string& thing) {
    PcConversation c;
    c.sys = "<|im_start|>system\nYou are " + role + ". Answer in one sentence.<|im_end|>\n";
    c.user = "Summarise these notes.\n";
    for (int i = 1; i <= 12; ++i)
        c.user += "Note " + std::to_string(i) + ": the " + thing + " checked on day " +
                  std::to_string(i * 3) + " showed no damage and the crew signed the log.\n";
    return c;
}

static std::vector<std::pair<std::string, std::string>> pc_env(const std::filesystem::path& root,
                                                               uint64_t mib) {
    return {{"XLLAMA_PREFIX_CACHE_MIB", std::to_string(mib)},
            {"XLLAMA_PREFIX_CACHE_MIN", "64"},
            {"XLLAMA_PREFIX_CACHE_DIR", root.string()}};
}

// The case the cache exists for: API clients interleave on one session (A, B,
// A'), B overwrites A's attention KV, and A' used to re-prefill from token 0.
// With the cache A' restores A's prefix from disk. The reference is the same
// A' served by the #170c checkpoint while A is still resident: the same state
// at the same position and the same batch shapes after it, so the generation
// must be byte-identical (the silent-corruption check).
TEST_CASE("Session: an interleaved API turn restores its prefix from disk (opt-in: "
          "XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    const auto root = std::filesystem::temp_directory_path() / "xllama-prefix-cache-a";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    const PcConversation a = pc_conversation("a logistics clerk", "mobile home in bay");
    const PcConversation b = pc_conversation("a harbour inspector", "container on quay");

    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 1024;
    std::string err;
    auto ref = xllama::Session::create(sp, &err); // no knobs: the cache is off
    REQUIRE_MESSAGE(ref, err);
    const auto ra = ref->generate(pc_greedy(a.turn1()));
    REQUIRE_MESSAGE(ra.success, ra.error_msg);
    REQUIRE(!ra.output_text.empty());
    const std::string a2 = a.turn2(ra.output_text);
    const auto ra2 = ref->generate(pc_greedy(a2));
    REQUIRE_MESSAGE(ra2.success, ra2.error_msg);
    CHECK(ra2.n_prefix_cache_tokens == 0);
    CHECK(ra2.n_p_eval < ref->count_tokens(a2) - 200); // the #170c restore did run
    CHECK_FALSE(std::filesystem::exists(root));
    ref.reset();

    sp.env = pc_env(root, 256);
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);
    const auto pa = sess->generate(pc_greedy(a.turn1()));
    REQUIRE_MESSAGE(pa.success, pa.error_msg);
    CHECK(pa.output_text == ra.output_text);
    CHECK(prefix_files(root).size() == 1);
    const auto pb = sess->generate(pc_greedy(b.turn1()));
    REQUIRE_MESSAGE(pb.success, pb.error_msg);
    CHECK(pb.n_prefix_cache_tokens == 0);
    CHECK(prefix_files(root).size() == 2);

    const auto pa2 = sess->generate(pc_greedy(a2));
    REQUIRE_MESSAGE(pa2.success, pa2.error_msg);
    CHECK(pa2.n_prefix_cache_tokens > 0);
    CHECK(pa2.n_prefix_cache_tokens + pa2.n_p_eval == pa2.n_prompt);
    CHECK(pa2.n_p_eval == ra2.n_p_eval);
    CHECK(pa2.output_text == ra2.output_text);

    sess.reset(); // the session's files go with it
    CHECK(prefix_files(root).empty());
    std::filesystem::remove_all(root, ec);
}

// The file holds the prompt's token history, so with the cache on only a
// request that opts in (GenerateParams::save_prefix, the API's
// "xllama_prefix_cache": true) writes one. Any request may restore it.
TEST_CASE("Session: the disk prefix cache writes only requests that opt in (opt-in: "
          "XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    const auto root = std::filesystem::temp_directory_path() / "xllama-prefix-cache-optin";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    const PcConversation a = pc_conversation("a logistics clerk", "mobile home in bay");
    const PcConversation b = pc_conversation("a harbour inspector", "container on quay");
    const auto unsaved = [](const std::string& prompt) {
        xllama::GenerateParams gp = pc_greedy(prompt);
        gp.save_prefix = false;
        return gp;
    };

    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 1024;
    sp.env = pc_env(root, 256);
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);
    const auto qa = sess->generate(unsaved(a.turn1()));
    REQUIRE_MESSAGE(qa.success, qa.error_msg);
    const auto qb = sess->generate(unsaved(b.turn1()));
    REQUIRE_MESSAGE(qb.success, qb.error_msg);
    CHECK(prefix_files(root).empty());
    const std::string a2 = a.turn2(qa.output_text);
    const auto qa2 = sess->generate(unsaved(a2));
    REQUIRE_MESSAGE(qa2.success, qa2.error_msg);
    CHECK(qa2.n_prefix_cache_tokens == 0);
    CHECK(prefix_files(root).empty());

    // Opted in: one file per split point of this prefill (CI: two here, after
    // the resident a2; one from an empty session in the case above).
    REQUIRE(sess->generate(pc_greedy(a.turn1())).success);
    const auto saved = prefix_files(root);
    CHECK(!saved.empty());
    REQUIRE(sess->generate(unsaved(b.turn1())).success);
    const auto ra2 = sess->generate(unsaved(a2)); // restores without opting in
    REQUIRE_MESSAGE(ra2.success, ra2.error_msg);
    CHECK(ra2.n_prefix_cache_tokens > 0);
    CHECK(ra2.n_prefix_cache_tokens + ra2.n_p_eval == ra2.n_prompt);
    CHECK(prefix_files(root) == saved); // nothing written, nothing dropped

    sess.reset();
    std::filesystem::remove_all(root, ec);
}

// What can go wrong around the cache must cost a prefill, never a request or a
// wrong state: a file whose content was altered (the load runs and is refused
// on its token history), a truncated file (refused before loading), another
// session with another context shape (never sees these files), and a budget
// that holds one file (the older one is evicted; disk use stays within it).
TEST_CASE("Session: the disk prefix cache drops bad files and keeps its budget (opt-in: "
          "XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    const auto root = std::filesystem::temp_directory_path() / "xllama-prefix-cache-b";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    const PcConversation a = pc_conversation("a logistics clerk", "mobile home in bay");
    const PcConversation b = pc_conversation("a harbour inspector", "container on quay");

    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 1024;
    sp.env = pc_env(root, 256);
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);
    const auto ra = sess->generate(pc_greedy(a.turn1()));
    REQUIRE_MESSAGE(ra.success, ra.error_msg);
    const auto after_a = prefix_files(root);
    REQUIRE(after_a.size() == 1);
    const std::string file_a = after_a.begin()->first;
    const uintmax_t size_a = after_a.begin()->second;
    const auto rb = sess->generate(pc_greedy(b.turn1()));
    REQUIRE_MESSAGE(rb.success, rb.error_msg);
    auto after_b = prefix_files(root);
    REQUIRE(after_b.size() == 2);
    after_b.erase(file_a);
    const std::string file_b = after_b.begin()->first;
    const uintmax_t size_b = after_b.begin()->second;
    const std::string a2 = a.turn2(ra.output_text);
    const std::string b2 = b.turn2(rb.output_text);

    {
        xllama::SessionParams sp2 = sp;
        sp2.n_ctx = 512;
        auto other = xllama::Session::create(sp2, &err);
        REQUIRE_MESSAGE(other, err);
        const auto r = other->generate(pc_greedy(a2));
        REQUIRE_MESSAGE(r.success, r.error_msg);
        CHECK(r.n_prefix_cache_tokens == 0);
    }
    CHECK(std::filesystem::exists(file_a));

    {
        // The first token id, right after the 12-byte header.
        std::fstream f(file_a, std::ios::in | std::ios::out | std::ios::binary);
        REQUIRE(f.is_open());
        const int32_t bad = -7;
        f.seekp(12);
        f.write(reinterpret_cast<const char*>(&bad), sizeof(bad));
    }
    const auto r_alt = sess->generate(pc_greedy(a2));
    REQUIRE_MESSAGE(r_alt.success, r_alt.error_msg);
    CHECK(r_alt.n_prefix_cache_tokens == 0);
    CHECK(r_alt.n_p_eval == sess->count_tokens(a2));
    CHECK_FALSE(std::filesystem::exists(file_a));

    std::filesystem::resize_file(file_b, size_b / 2, ec);
    REQUIRE_FALSE(ec);
    const auto r_cut = sess->generate(pc_greedy(b2));
    REQUIRE_MESSAGE(r_cut.success, r_cut.error_msg);
    CHECK(r_cut.n_prefix_cache_tokens == 0);
    CHECK(r_cut.n_p_eval == sess->count_tokens(b2));
    CHECK_FALSE(std::filesystem::exists(file_b));
    sess.reset();

    // Room for the larger of the two first-turn files, not for both.
    const uint64_t mib = (size_a + size_b - 1) >> 20;
    REQUIRE((mib << 20) >= std::max(size_a, size_b));
    sp.env = pc_env(root, mib);
    auto small = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(small, err);
    REQUIRE(small->generate(pc_greedy(a.turn1())).success);
    REQUIRE(small->generate(pc_greedy(b.turn1())).success);
    auto held = prefix_files(root);
    CHECK(held.size() == 1);
    uint64_t total = 0;
    for (const auto& kv : held)
        total += kv.second;
    CHECK(total <= (mib << 20));
    const auto r_evicted = small->generate(pc_greedy(a2));
    REQUIRE_MESSAGE(r_evicted.success, r_evicted.error_msg);
    CHECK(r_evicted.n_prefix_cache_tokens == 0);
    total = 0;
    for (const auto& kv : prefix_files(root))
        total += kv.second;
    CHECK(total <= (mib << 20));
    small.reset();
    std::filesystem::remove_all(root, ec);
}
