// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// PrefixCacheIndex: the bookkeeping of the on-disk prefix cache, model-free.
// The session-level behaviour (restore equivalence, corruption, budget on a
// real model) is in test_session.cpp, opt-in via XLLAMA_TEST_MODEL.

#include "xllama/prefix_cache.h"

#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using xllama::PrefixCacheEntry;
using xllama::PrefixCacheIndex;

namespace {

struct PrefixDir {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("xllama-prefix-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    PrefixDir() {
        std::filesystem::create_directories(path);
    }
    ~PrefixDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    void put(const std::filesystem::path& name, std::chrono::seconds age) const {
        const auto file = path / name;
        std::filesystem::create_directories(file.parent_path());
        std::ofstream(file, std::ios::binary) << "cached state";
        std::filesystem::last_write_time(file, std::filesystem::file_time_type::clock::now() - age);
    }
};

PrefixCacheEntry entry(std::vector<int32_t> tokens, uint64_t bytes) {
    PrefixCacheEntry e;
    e.file = PrefixCacheIndex::file_name("fp", tokens.data(), tokens.size());
    e.tokens = std::move(tokens);
    e.bytes = bytes;
    return e;
}

} // namespace

TEST_CASE("prefix cache: TTL defaults to 30 minutes and accepts nonnegative seconds") {
    CHECK(xllama::prefix_cache_ttl(nullptr).count() == 1800);
    for (const char* invalid : {"", "-1", "12s", " 12", "2147483648"})
        CHECK(xllama::prefix_cache_ttl(invalid).count() == 1800);
    CHECK(xllama::prefix_cache_ttl("60").count() == 60);
    CHECK(xllama::prefix_cache_ttl("0").count() == 0);
}

TEST_CASE("prefix cache: pruning removes expired files and leaves fresh files") {
    const PrefixDir dir;
    dir.put("pc-1/old.kv", std::chrono::seconds(1801));
    dir.put("pc-1/fresh.kv", std::chrono::seconds(0));
    dir.put("orphan.tmp", std::chrono::seconds(1801));
    CHECK(xllama::prune_prefix_cache_files(dir.path, std::chrono::seconds(1800)) == 2);
    CHECK_FALSE(std::filesystem::exists(dir.path / "pc-1/old.kv"));
    CHECK_FALSE(std::filesystem::exists(dir.path / "orphan.tmp"));
    CHECK(std::filesystem::exists(dir.path / "pc-1/fresh.kv"));
    CHECK(xllama::prune_prefix_cache_files(dir.path, std::chrono::seconds(1800)) == 0);
}

TEST_CASE("prefix cache: expired lookup entries are deleted before selecting a fresh prefix") {
    const PrefixDir dir;
    PrefixCacheIndex idx;
    const auto fresh = entry({1, 2}, 10);
    const auto old = entry({1, 2, 3}, 20);
    const auto missing = entry({1, 2, 3, 4}, 30);
    idx.add(fresh, 1000);
    idx.add(old, 1000);
    idx.add(missing, 1000);
    dir.put(fresh.file, std::chrono::seconds(0));
    dir.put(old.file, std::chrono::seconds(61));
    idx.prune(dir.path, std::chrono::seconds(60));
    CHECK_FALSE(std::filesystem::exists(dir.path / old.file));
    CHECK(std::filesystem::exists(dir.path / fresh.file));
    CHECK(idx.total_bytes() == 10);
    REQUIRE(idx.entries().size() == 1);
    const std::vector<int32_t> prompt{1, 2, 3, 4, 5};
    CHECK(idx.find(prompt.data(), prompt.size(), 0) == 0);
    CHECK(idx.entries()[0].tokens == fresh.tokens);
    CHECK(idx.find_exact(old.tokens.data(), old.tokens.size()) == -1);
}

TEST_CASE("prefix cache: every start wipe empties the root and counts files only") {
    const PrefixDir dir;
    dir.put("pc-1/nested/old.kv", std::chrono::seconds(1801));
    dir.put("pc-2/fresh.kv", std::chrono::seconds(0));
    dir.put("unrelated.tmp", std::chrono::seconds(0));
    CHECK(xllama::wipe_prefix_cache_files(dir.path) == 3);
    CHECK(std::filesystem::is_empty(dir.path));
    dir.put("another.kv", std::chrono::seconds(0));
    CHECK(xllama::wipe_prefix_cache_files(dir.path) == 1);
    CHECK(std::filesystem::is_empty(dir.path));
    CHECK(xllama::wipe_prefix_cache_files(dir.path / "missing") == 0);
}

TEST_CASE("prefix cache: a start wipe does not follow a directory symlink") {
    const PrefixDir root;
    const PrefixDir outside;
    outside.put("keep.kv", std::chrono::seconds(0));
    std::error_code ec;
    std::filesystem::create_directory_symlink(outside.path, root.path / "linked", ec);
    if (ec)
        return; // some Windows host setups do not grant symlink creation
    CHECK(xllama::wipe_prefix_cache_files(root.path) == 1);
    CHECK(std::filesystem::is_empty(root.path));
    CHECK(std::filesystem::exists(outside.path / "keep.kv"));
}

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
