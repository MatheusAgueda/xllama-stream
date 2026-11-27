// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The environment parses that gate the GPU attention path (include/xllama/d3d12_env.h): the
// XLLAMA_ATTN_GPU flag, XLLAMA_FA_DECODE_KV, and LLAMA_KV_ON_DEVICE of llama.cpp patch 0026. They
// document the exact semantics: a test that fails here means a behaviour change.

#include <doctest/doctest.h>

#include "xllama/d3d12_env.h"

#include <fstream>
#include <iterator>
#include <string>

TEST_CASE("env_flag_on (XLLAMA_ATTN_GPU): the first character decides") {
    using xllama::env_flag_on;
    CHECK_FALSE(env_flag_on(nullptr)); // unset
    CHECK_FALSE(env_flag_on(""));
    CHECK_FALSE(env_flag_on("0"));
    CHECK(env_flag_on("1"));
    CHECK_FALSE(env_flag_on("01")); // a leading 0 is off whatever follows
    CHECK(env_flag_on("10"));
    CHECK(env_flag_on("yes")); // any other first character is on, "false" included
    CHECK(env_flag_on("false"));
    CHECK(env_flag_on(" 0")); // no trimming: a blank is a character
}

TEST_CASE("fa_decode_kv_from_env (XLLAMA_FA_DECODE_KV): positions from 1 to 0x7fffffff, else off") {
    using xllama::fa_decode_kv_from_env;
    CHECK(fa_decode_kv_from_env(nullptr, true) == 0); // unset
    CHECK(fa_decode_kv_from_env("", true) == 0);
    CHECK(fa_decode_kv_from_env("0", true) == 0);
    CHECK(fa_decode_kv_from_env("1", true) == 1);
    CHECK(fa_decode_kv_from_env("1280", true) == 1280);
    CHECK(fa_decode_kv_from_env("2147483647", true) == 0x7fffffffLL);
    CHECK(fa_decode_kv_from_env("2147483648", true) == 0); // one past the range
    CHECK(fa_decode_kv_from_env("99999999999", true) == 0);
    CHECK(fa_decode_kv_from_env("99999999999999999999999", true) == 0); // strtoll saturates
    CHECK(fa_decode_kv_from_env("-5", true) == 0);
    CHECK(fa_decode_kv_from_env("abc", true) == 0);
}

TEST_CASE("fa_decode_kv_from_env: strtoll leniency is kept, and the kernels gate it") {
    using xllama::fa_decode_kv_from_env;
    CHECK(fa_decode_kv_from_env("1280abc", true) == 1280); // trailing text ignored
    CHECK(fa_decode_kv_from_env(" 64", true) == 64);       // leading blank skipped
    CHECK(fa_decode_kv_from_env("+64", true) == 64);
    CHECK(fa_decode_kv_from_env("1e3", true) == 1);  // not a float: stops at the 'e'
    CHECK(fa_decode_kv_from_env("0x40", true) == 0); // base 10: "0" then junk
    // Without the flash-attention kernels (fap_ok false) every value is off.
    CHECK(fa_decode_kv_from_env("1280", false) == 0);
    CHECK(fa_decode_kv_from_env("1", false) == 0);
    CHECK(fa_decode_kv_from_env(nullptr, false) == 0);
}

TEST_CASE("kv_on_device_from_env (LLAMA_KV_ON_DEVICE): parses like XLLAMA_ATTN_GPU") {
    using xllama::env_flag_on;
    using xllama::kv_on_device_from_env;
    // The values production sets (tools/fap_*.py, sxs logs, LocalState manifests): "1", "0", unset.
    CHECK(kv_on_device_from_env("1"));
    CHECK_FALSE(kv_on_device_from_env("0"));
    CHECK_FALSE(kv_on_device_from_env(nullptr));
    // The old quirk is gone: set-but-empty and "00" are off, as for env_flag_on.
    CHECK_FALSE(kv_on_device_from_env(""));
    CHECK_FALSE(kv_on_device_from_env("00"));
    CHECK_FALSE(kv_on_device_from_env("01"));
    CHECK(kv_on_device_from_env("10"));
    CHECK(kv_on_device_from_env("yes"));
    CHECK(kv_on_device_from_env("false")); // first character only, as XLLAMA_ATTN_GPU
    for (const char* v : {static_cast<const char*>(nullptr), "", "0", "00", "01", "1", "10", "yes",
                          "false", " 0", "x"})
        CHECK(kv_on_device_from_env(v) == env_flag_on(v));
}

#ifdef PATCH_0026_PATH
TEST_CASE("kv_on_device_from_env mirrors the expression in patch 0026") {
    std::ifstream in(PATCH_0026_PATH, std::ios::binary);
    REQUIRE(in.good());
    const std::string patch((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // The patch cannot include xllama headers; if either line changes, kv_on_device_from_env and
    // its cases above have to change with it.
    CHECK(patch.find("getenv(\"LLAMA_KV_ON_DEVICE\")") != std::string::npos);
    CHECK(patch.find("offload || (kv_dev && *kv_dev && *kv_dev != '0')") != std::string::npos);
    CHECK(patch.find("strcmp(kv_dev") == std::string::npos); // the old whole-text test is gone
}
#endif
