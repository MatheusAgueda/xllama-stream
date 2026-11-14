// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// PrefixCacheIndex: the bookkeeping of the on-disk prefix cache, model-free.
// The session-level behaviour (restore equivalence, corruption, budget on a
// real model) is in test_session.cpp, opt-in via XLLAMA_TEST_MODEL.

#include "xllama/prefix_cache.h"

#include <doctest/doctest.h>

#include <string>
#include <vector>

using xllama::PrefixCacheEntry;
using xllama::PrefixCacheIndex;

namespace {

PrefixCacheEntry entry(std::vector<int32_t> tokens, uint64_t bytes) {
    PrefixCacheEntry e;
    e.file = PrefixCacheIndex::file_name("fp", tokens.data(), tokens.size());
    e.tokens = std::move(tokens);
    e.bytes = bytes;
    return e;
}

} // namespace

TEST_CASE("PrefixCacheIndex: file names hash the fingerprint and the tokens") {
    const std::vector<int32_t> a = {1, 2, 3};
    const std::vector<int32_t> b = {1, 2, 4};
    const std::string na = PrefixCacheIndex::file_name("fp", a.data(), a.size());
    CHECK(na.size() == 16 + 3);
    CHECK(na.substr(16) == ".kv");
    CHECK(na.find_first_not_of("0123456789abcdef") == 16);
    CHECK(na == PrefixCacheIndex::file_name("fp", a.data(), a.size()));
    CHECK(na != PrefixCacheIndex::file_name("fp", b.data(), b.size()));
    CHECK(na != PrefixCacheIndex::file_name("fp2", a.data(), a.size()));
    CHECK(na != PrefixCacheIndex::file_name("fp", a.data(), 2));
}

TEST_CASE("PrefixCacheIndex: find returns the longest prefix within the bounds") {
    PrefixCacheIndex idx;
    CHECK(idx.add(entry({1, 2}, 10), 1000).empty());
    CHECK(idx.add(entry({1, 2, 3, 4}, 10), 1000).empty());
    CHECK(idx.add(entry({1, 2, 3, 4, 5, 6}, 10), 1000).empty());
    CHECK(idx.add(entry({9, 9, 9}, 10), 1000).empty());
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7};

    const int all = idx.find(prompt.data(), prompt.size(), 0);
    REQUIRE(all >= 0);
    CHECK(idx.entries()[static_cast<size_t>(all)].tokens.size() == 6);
    // max_len: the prompt's last token must stay to prefill.
    const int capped = idx.find(prompt.data(), 5, 0);
    REQUIRE(capped >= 0);
    CHECK(idx.entries()[static_cast<size_t>(capped)].tokens.size() == 4);
    // min_len: shorter than what the in-memory path keeps is no gain.
    CHECK(idx.find(prompt.data(), 5, 5) == -1);
    // A divergent prompt matches only up to where it diverges.
    const std::vector<int32_t> other = {1, 2, 3, 8, 5, 6, 7};
    const int short_one = idx.find(other.data(), other.size(), 0);
    REQUIRE(short_one >= 0);
    CHECK(idx.entries()[static_cast<size_t>(short_one)].tokens.size() == 2);

    const std::vector<int32_t> exact = {1, 2, 3, 4};
    const int e = idx.find_exact(exact.data(), exact.size());
    REQUIRE(e >= 0);
    CHECK(idx.entries()[static_cast<size_t>(e)].tokens == exact);
    CHECK(idx.find_exact(exact.data(), 3) == -1);
}

TEST_CASE("PrefixCacheIndex: eviction keeps the budget, least recently used first") {
    PrefixCacheIndex idx;
    const PrefixCacheEntry a = entry({1}, 40);
    const PrefixCacheEntry b = entry({2}, 40);
    const PrefixCacheEntry c = entry({3}, 40);
    CHECK(idx.add(a, 100).empty());
    CHECK(idx.add(b, 100).empty());
    CHECK(idx.total_bytes() == 80);

    // a used again: b is now the oldest and goes when c needs the room.
    const std::vector<int32_t> ta = {1};
    idx.touch(static_cast<size_t>(idx.find_exact(ta.data(), 1)));
    const std::vector<std::string> evicted = idx.add(c, 100);
    REQUIRE(evicted.size() == 1);
    CHECK(evicted[0] == b.file);
    CHECK(idx.total_bytes() == 80);
    CHECK(idx.entries().front().file == a.file);
    CHECK(idx.entries().back().file == c.file);

    // An entry larger than the whole budget evicts everything, itself included.
    const PrefixCacheEntry big = entry({4}, 500);
    const std::vector<std::string> all = idx.add(big, 100);
    CHECK(all.size() == 3);
    CHECK(all.back() == big.file);
    CHECK(idx.entries().empty());
    CHECK(idx.total_bytes() == 0);
}

TEST_CASE("PrefixCacheIndex: a rewrite under the same name replaces the entry") {
    PrefixCacheIndex idx;
    PrefixCacheEntry first = entry({5, 6}, 30);
    PrefixCacheEntry again = first;
    again.bytes = 50;
    CHECK(idx.add(first, 1000).empty());
    CHECK(idx.add(again, 1000).empty());
    REQUIRE(idx.entries().size() == 1);
    CHECK(idx.total_bytes() == 50);

    CHECK(idx.remove(0) == first.file);
    CHECK(idx.entries().empty());
    CHECK(idx.total_bytes() == 0);

    CHECK(idx.add(entry({7}, 10), 1000).empty());
    idx.clear();
    CHECK(idx.entries().empty());
    CHECK(idx.total_bytes() == 0);
}
