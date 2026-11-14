// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/api_auth.h"

#include <doctest/doctest.h>

TEST_CASE("api auth: LocalState key is trimmed and needs 32 characters") {
    CHECK(xllama::api_key_from_text("").empty());
    CHECK(xllama::api_key_from_text(" \t\r\n").empty());
    CHECK(xllama::api_key_from_text(std::string(31, 'k')).empty());
    const std::string key(32, 'k');
    CHECK(xllama::api_key_from_text(" \t" + key + "\r\n\t ") == key);
    CHECK(xllama::api_key_from_text(key + "k") == key + "k");
}

TEST_CASE("api auth: bearer scheme is case insensitive and key is exact") {
    const std::string key = "0123456789abcdef0123456789ABCDEF";
    for (const char* scheme : {"Bearer ", "bearer ", "BEARER ", "bEaReR "})
        CHECK(xllama::bearer_matches(std::string(scheme) + key, key));
    for (size_t i = 0; i < key.size(); ++i) {
        std::string wrong = key;
        wrong[i] ^= 1;
        CHECK_FALSE(xllama::bearer_matches("Bearer " + wrong, key));
    }
    for (const std::string& bad :
         {key, "Basic " + key, "Bearer  " + key, "Bearer\t" + key, " Bearer " + key,
          "Bearer " + key + " ", "Bearer " + key.substr(1), "Bearer " + key + "x"})
        CHECK_FALSE(xllama::bearer_matches(bad, key));
    CHECK_FALSE(xllama::bearer_matches("", key));
    CHECK_FALSE(xllama::bearer_matches("Bearer ", ""));
    CHECK_FALSE(xllama::bearer_matches("", ""));
}
