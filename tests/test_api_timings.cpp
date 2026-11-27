// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/api_timings.h"

#include <doctest/doctest.h>

#include <chrono>
#include <limits>

TEST_CASE("api timings: zeros are emitted as numbers") {
    CHECK(xllama::api_timings_json({}) ==
          R"({"prefill_ms":0.0,"decode_ms":0.0,"prompt_n":0,"predicted_n":0})");
}

TEST_CASE("api timings: tenths of milliseconds and evaluated prompt tokens") {
    xllama::InferenceResult result;
    result.t_p_eval_ms = 12.34;
    result.t_eval_ms = 56.78;
    result.n_prompt = 1024;
    result.n_p_eval = 24;
    result.n_eval = 8;
    CHECK(xllama::api_timings_json(result) ==
          R"({"prefill_ms":12.3,"decode_ms":56.8,"prompt_n":24,"predicted_n":8})");
}

TEST_CASE("api timings: a non-finite time is emitted as 0, never as a word JSON cannot parse") {
    xllama::InferenceResult result;
    result.t_p_eval_ms = std::numeric_limits<double>::infinity();
    result.t_eval_ms = std::numeric_limits<double>::quiet_NaN();
    CHECK(xllama::api_timings_json(result) ==
          R"({"prefill_ms":0.0,"decode_ms":0.0,"prompt_n":0,"predicted_n":0})");
}

TEST_CASE("api timings: an expert object only when the model has an expert cache") {
    xllama::InferenceResult result;
    result.n_eval = 3;
    CHECK(xllama::api_timings_json(result).find("expert") == std::string::npos);

    result.expert.present = true;
    result.expert.hits = 90;
    result.expert.misses = 10;
    result.expert.swaps = 4;
    result.expert.em_swaps = 1;
    result.expert.bytes = 123456789012ULL;
    result.expert.read_us = 4500;
    result.expert.em_reads = 6;
    result.expert.em_bytes = 600;
    CHECK(xllama::api_timings_json(result) ==
          R"({"prefill_ms":0.0,"decode_ms":0.0,"prompt_n":0,"predicted_n":3,"expert":{"hits":90,)"
          R"("misses":10,"swaps":4,"em_swaps":1,"bytes":123456789012,"read_us":4500,)"
          R"("em_reads":6,"em_bytes":600}})");
}

TEST_CASE("expert stats: since() is a per-request delta and never goes negative") {
    xllama::ExpertStats begin;
    begin.present = true;
    begin.hits = 100;
    begin.misses = 20;
    begin.read_us = 7;
    xllama::ExpertStats end = begin;
    end.hits = 160;
    end.misses = 25;
    end.swaps = 3;
    end.read_us = 5; // a counter that went down reads 0
    const xllama::ExpertStats d = end.since(begin);
    CHECK(d.present);
    CHECK(d.hits == 60);
    CHECK(d.misses == 5);
    CHECK(d.swaps == 3);
    CHECK(d.read_us == 0);
    // a snapshot without a cache has nothing to subtract from
    CHECK_FALSE(end.since(xllama::ExpertStats{}).present);
}

TEST_CASE("api timings: the route hash is a 16-digit hex string, copied and not subtracted") {
    xllama::ExpertStats begin;
    begin.present = true;
    begin.has_route = true; // the bridge restarts the chain before taking `begin`: route 0, n 0
    xllama::ExpertStats end = begin;
    end.route_hash = 0xcbf29ce484222325ULL; // a value past 2^53: a double would lose its low bits
    end.route_n = 160;
    const xllama::ExpertStats d = end.since(begin);
    CHECK(d.has_route);
    CHECK(d.route_hash == 0xcbf29ce484222325ULL);
    CHECK(d.route_n == 160);

    xllama::InferenceResult result;
    result.expert = d;
    const std::string json = xllama::api_timings_json(result);
    CHECK(json.find(R"("route":"cbf29ce484222325","route_n":160}})") != std::string::npos);
    // small values keep their leading zeros
    result.expert.route_hash = 0xabc;
    CHECK(xllama::api_timings_json(result).find(R"("route":"0000000000000abc")") !=
          std::string::npos);
    // a library without patch 0028 has no route: the object stays as it was
    result.expert.has_route = false;
    CHECK(xllama::api_timings_json(result).find("route") == std::string::npos);
}

TEST_CASE("api timings: request stamps in ms since accept, unset ones left out") {
    using Clock = std::chrono::steady_clock;
    using ms = std::chrono::milliseconds;
    const Clock::time_point t0 = Clock::time_point{} + std::chrono::seconds(100);
    xllama::RequestStamps stamps;
    stamps.begin(t0);
    xllama::InferenceResult result;

    // no permit yet: no object
    CHECK(xllama::api_timings_json(result, &stamps).find("stamps") == std::string::npos);

    stamps.permit(t0 + ms(12));
    CHECK(xllama::api_timings_json(result, &stamps) ==
          R"({"prefill_ms":0.0,"decode_ms":0.0,"prompt_n":0,"predicted_n":0,)"
          R"("stamps":{"permit_ms":12.0}})");

    stamps.first(t0 + ms(800));
    stamps.first(t0 + ms(900)); // only the first call counts
    stamps.last(t0 + ms(2500));
    CHECK(xllama::api_timings_json(result, &stamps) ==
          R"({"prefill_ms":0.0,"decode_ms":0.0,"prompt_n":0,"predicted_n":0,)"
          R"("stamps":{"permit_ms":12.0,"first_ms":800.0,"last_ms":2500.0}})");

    // no stamps, the old reply
    CHECK(xllama::api_timings_json(result, nullptr) == xllama::api_timings_json(result));
}

TEST_CASE("request stamps: a call without a token callback derives the first token") {
    using Clock = std::chrono::steady_clock;
    using ms = std::chrono::milliseconds;
    const Clock::time_point t0 = Clock::time_point{} + std::chrono::seconds(5);
    xllama::RequestStamps stamps;
    stamps.begin(t0);
    stamps.generated(t0 + ms(100), t0 + ms(1000), 400.0);
    CHECK(stamps.first_ms == doctest::Approx(500.0));
    CHECK(stamps.last_ms == doctest::Approx(1000.0));

    // a measured first token stays
    xllama::RequestStamps streamed;
    streamed.begin(t0);
    streamed.first(t0 + ms(450));
    streamed.generated(t0 + ms(100), t0 + ms(1000), 400.0);
    CHECK(streamed.first_ms == doctest::Approx(450.0));

    // no first token (zero tokens generated): nothing derived
    xllama::RequestStamps none;
    none.begin(t0);
    none.generated(t0, t0 + ms(10), 0.0);
    CHECK(none.first_ms < 0.0);
    CHECK(none.last_ms == doctest::Approx(10.0));
}

TEST_CASE("api timings: a batch object only for a batched result (B3)") {
    xllama::InferenceResult result;
    result.t_p_eval_ms = 30.0;
    result.t_eval_ms = 5.0;
    result.n_p_eval = 7;
    result.n_eval = 2;
    // Not batched: byte-identical to the pre-B3 reply.
    CHECK(xllama::api_timings_json(result) ==
          R"({"prefill_ms":30.0,"decode_ms":5.0,"prompt_n":7,"predicted_n":2})");

    result.batch.k = 3;
    result.batch.index = 1;
    result.batch.n_tokens = 18;
    result.batch.ms = 25.04;
    CHECK(xllama::api_timings_json(result) ==
          R"({"prefill_ms":30.0,"decode_ms":5.0,"prompt_n":7,"predicted_n":2,)"
          R"("batch":{"k":3,"index":1,"prefill_ms":25.0,"prefill_n":18}})");

    result.batch.expert.present = true;
    result.batch.expert.hits = 4;
    result.batch.expert.misses = 1;
    result.batch.expert.bytes = 100;
    result.batch.expert.read_us = 9;
    result.batch.ms = std::numeric_limits<double>::quiet_NaN();
    CHECK(xllama::api_timings_json(result) ==
          R"({"prefill_ms":30.0,"decode_ms":5.0,"prompt_n":7,"predicted_n":2,)"
          R"("batch":{"k":3,"index":1,"prefill_ms":0.0,"prefill_n":18,)"
          R"("expert":{"hits":4,"misses":1,"bytes":100,"read_us":9}}})");
}
