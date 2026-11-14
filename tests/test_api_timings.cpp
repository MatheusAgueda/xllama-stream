// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/api_timings.h"

#include <doctest/doctest.h>

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
