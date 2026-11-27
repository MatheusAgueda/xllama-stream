// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/api_policy.h"
#include "xllama/force_ids.h"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

// F01: the audit sent max_tokens -1, 1.5 and 2147483647 and got 200 for all three.

TEST_CASE("validate_max_tokens: whole numbers from 1 to n_ctx - 1") {
    using xllama::validate_max_tokens;
    CHECK(validate_max_tokens("max_tokens", 1, 2048).value == 1);
    CHECK(validate_max_tokens("max_tokens", 2047, 2048).value == 2047);
    CHECK(validate_max_tokens("max_tokens", 2047, 2048).error.empty());

    CHECK(validate_max_tokens("max_tokens", -1, 2048).error == "max_tokens must be at least 1");
    CHECK(validate_max_tokens("max_tokens", 0, 2048).error == "max_tokens must be at least 1");
    CHECK(validate_max_tokens("max_tokens", 1.5, 2048).error ==
          "max_tokens must be a whole number");
    CHECK(validate_max_tokens("max_completion_tokens", 2147483647.0, 2048).error ==
          "max_completion_tokens must be less than the model context (n_ctx 2048)");
    CHECK(validate_max_tokens("max_tokens", 2048, 2048).error ==
          "max_tokens must be less than the model context (n_ctx 2048)");
    CHECK_FALSE(validate_max_tokens("max_tokens", std::numeric_limits<double>::infinity(), 2048)
                    .error.empty());
    CHECK_FALSE(validate_max_tokens("max_tokens", std::nan(""), 2048).error.empty());
}

TEST_CASE("validate_api_number: range before any cast") {
    using xllama::validate_api_number;
    CHECK(validate_api_number("temperature", 0.7, 0, 2, false).empty());
    CHECK(validate_api_number("temperature", 2.5, 0, 2, false) ==
          "temperature must be between 0 and 2");
    CHECK(validate_api_number("top_p", -0.1, 0, 1, false) == "top_p must be between 0 and 1");
    CHECK(validate_api_number("seed", 42, -2147483648.0, 4294967295.0, true).empty());
    CHECK(validate_api_number("seed", -1, -2147483648.0, 4294967295.0, true).empty());
    CHECK(validate_api_number("seed", 1.5, -2147483648.0, 4294967295.0, true) ==
          "seed must be a whole number");
    CHECK(validate_api_number("seed", 1e20, -2147483648.0, 4294967295.0, true) ==
          "seed must be between -2147483648 and 4294967295");
    CHECK(validate_api_number("temperature", std::nan(""), 0, 2, false) ==
          "temperature must be a finite number");
}

TEST_CASE("clamp_api_int: clamped on the double, never an out-of-range cast") {
    using xllama::clamp_api_int;
    CHECK(clamp_api_int(3, 1, 4) == 3);
    CHECK(clamp_api_int(0, 1, 4) == 1);
    CHECK(clamp_api_int(1e300, 1, 4) == 4);
    CHECK(clamp_api_int(-1e300, 1, 4) == 1);
    CHECK(clamp_api_int(std::nan(""), 1, 4) == 1);
    CHECK(clamp_api_int(2.9, 1, 4) == 2);
}

// "xllama_force_ids" (research API): the elements the WinRT layer saw, checked here.

namespace {
std::vector<xllama::ForceIdElement> nums(std::initializer_list<double> values) {
    std::vector<xllama::ForceIdElement> out;
    for (double v : values)
        out.push_back({true, v});
    return out;
}
} // namespace

TEST_CASE("force_ids: a valid list passes through in order") {
    using xllama::ForceIdsError;
    const auto r = xllama::parse_force_ids(true, nums({0, 15, 151645, 2147483647.0}), 8);
    CHECK(r.error == ForceIdsError::None);
    CHECK(r.ids == std::vector<int32_t>{0, 15, 151645, 2147483647});
    // Exactly n_predict ids is allowed; the budget is a ceiling.
    CHECK(xllama::parse_force_ids(true, nums({1, 2, 3}), 3).error == ForceIdsError::None);
}

TEST_CASE("force_ids: an empty list is valid and forces nothing") {
    const auto r = xllama::parse_force_ids(true, {}, 1);
    CHECK(r.error == xllama::ForceIdsError::None);
    CHECK(r.ids.empty());
}

TEST_CASE("force_ids: only whole non-negative numbers up to INT32_MAX are ids") {
    using xllama::ForceIdsError;
    using xllama::parse_force_ids;
    CHECK(parse_force_ids(true, nums({1.5}), 8).error == ForceIdsError::BadId);
    CHECK(parse_force_ids(true, nums({-1}), 8).error == ForceIdsError::BadId);
    CHECK(parse_force_ids(true, nums({-0.5}), 8).error == ForceIdsError::BadId);
    CHECK(parse_force_ids(true, nums({2147483648.0}), 8).error == ForceIdsError::BadId);
    CHECK(parse_force_ids(true, nums({1e300}), 8).error == ForceIdsError::BadId);
    CHECK(parse_force_ids(true, nums({std::nan("")}), 8).error == ForceIdsError::BadId);
    CHECK(parse_force_ids(true, nums({std::numeric_limits<double>::infinity()}), 8).error ==
          ForceIdsError::BadId);
    // 2.0 is whole, whatever its text said.
    CHECK(parse_force_ids(true, nums({2.0}), 8).ids == std::vector<int32_t>{2});
}

TEST_CASE("force_ids: a non-number element is refused, and the first bad one stops the list") {
    using xllama::ForceIdElement;
    using xllama::ForceIdsError;
    // "abc", true, null and nested arrays all reach the checker as is_number = false.
    const std::vector<ForceIdElement> text = {{true, 1}, {false, 0.0}, {true, 2}};
    const auto r = xllama::parse_force_ids(true, text, 8);
    CHECK(r.error == ForceIdsError::BadId);
    CHECK(r.ids.empty()); // nothing half-parsed leaks to the caller
    // A bad id wins over an oversize list: the element check runs first.
    CHECK(xllama::parse_force_ids(true, nums({1, 2, -1}), 1).error == ForceIdsError::BadId);
}

TEST_CASE("force_ids: more ids than max_tokens is TooMany, naming the count") {
    using xllama::ForceIdsError;
    const auto r = xllama::parse_force_ids(true, nums({1, 2, 3}), 2);
    CHECK(r.error == ForceIdsError::TooMany);
    CHECK(r.ids.size() == 3);
    CHECK(xllama::force_ids_message(r.error, r.ids.size()) ==
          "max_tokens must cover the 3 forced ids");
    CHECK(xllama::parse_force_ids(true, nums({1}), 0).error == ForceIdsError::TooMany);
}

TEST_CASE("force_ids: a field that is not an array is NotArray, elements ignored") {
    using xllama::ForceIdsError;
    // "xllama_force_ids": 5, "5", {...}, null, true: the JSON layer passes is_array = false.
    const auto r = xllama::parse_force_ids(false, nums({1, 2}), 8);
    CHECK(r.error == ForceIdsError::NotArray);
    CHECK(r.ids.empty());
    CHECK(xllama::force_ids_message(r.error, 0) ==
          "'xllama_force_ids' must be an array of token ids");
    CHECK(xllama::force_ids_message(ForceIdsError::BadId, 0) ==
          "'xllama_force_ids' must hold whole non-negative numbers");
    CHECK(xllama::force_ids_message(ForceIdsError::None, 0).empty());
}
