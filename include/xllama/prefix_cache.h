// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// xllama::PrefixCacheIndex — bookkeeping of the on-disk prefix cache of a
// hybrid LlamaSession (XLLAMA_PREFIX_CACHE_MIB). The I/O itself
// (llama_state_seq_save_file / load_file) stays in src/bridge/session.cpp.
#pragma once

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

inline std::chrono::seconds prefix_cache_ttl(const char* value) {
    constexpr std::chrono::seconds fallback{1800};
    if (!value || !*value)
        return fallback;
    const std::string_view text(value);
    int seconds = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), seconds);
    return parsed.ec == std::errc() && parsed.ptr == text.data() + text.size() && seconds >= 0
               ? std::chrono::seconds(seconds)
               : fallback;
}

inline bool prefix_cache_file_expired(const std::filesystem::path& path, std::chrono::seconds ttl) {
    std::error_code ec;
    const auto written = std::filesystem::last_write_time(path, ec);
    return ec || std::filesystem::file_time_type::clock::now() - written > ttl;
}

// No TTL means a start wipe. Do not follow directory symlinks out of the root.
inline size_t remove_prefix_cache_files(const std::filesystem::path& root,
                                        std::optional<std::chrono::seconds> ttl) {
    std::error_code ec;
    std::vector<std::filesystem::path> files, dirs;
    for (std::filesystem::recursive_directory_iterator
             it(root, std::filesystem::directory_options::skip_permission_denied, ec),
         end;
         !ec && it != end; it.increment(ec)) {
        std::error_code status_ec;
        const auto status = it->symlink_status(status_ec);
        if (status_ec)
            continue;
        if (std::filesystem::is_directory(status)) {
            if (!ttl)
                dirs.push_back(it->path());
        } else if (!ttl || prefix_cache_file_expired(it->path(), *ttl)) {
            files.push_back(it->path());
        }
    }
    size_t removed = 0;
    for (const auto& path : files)
        if (std::filesystem::remove(path, ec))
            ++removed;
    for (auto it = dirs.rbegin(); it != dirs.rend(); ++it)
        std::filesystem::remove(*it, ec);
    return removed;
}

inline size_t wipe_prefix_cache_files(const std::filesystem::path& root) {
    return remove_prefix_cache_files(root, std::nullopt);
}

inline size_t prune_prefix_cache_files(const std::filesystem::path& root,
                                       std::chrono::seconds ttl) {
    return remove_prefix_cache_files(root, ttl);
}

// One saved sequence state: the token history it absorbed and the file
// holding it. Like a KvStore snapshot it is cached work, never a source of
// truth — a missing or refused file costs a prefill, nothing else.
struct PrefixCacheEntry {
    std::vector<int32_t> tokens; // history[0, n) the state absorbed
    std::string file;            // name inside the session's cache directory
    uint64_t bytes = 0;          // size llama_state_seq_save_file reported
};

class PrefixCacheIndex {
  public:
    // FNV-1a 64 of the context fingerprint and the token ids. The file name
    // only has to be unique within one session's directory: a collision
    // replaces the older entry (add), and a restore checks the token history
    // the file returns, so it can cost a prefill but never a wrong state.
    static std::string file_name(const std::string& fingerprint, const int32_t* tokens, size_t n) {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&h](const void* data, size_t len) {
            const auto* p = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < len; ++i) {
                h ^= p[i];
                h *= 1099511628211ull;
            }
        };
        mix(fingerprint.data(), fingerprint.size());
        mix(tokens, n * sizeof(int32_t));
        char buf[24];
        std::snprintf(buf, sizeof(buf), "%016llx.kv", static_cast<unsigned long long>(h));
        return buf;
    }

    // The longest entry whose history is a prefix of prompt[0, max_len) and
    // at least min_len long; -1 when none.
    int find(const int32_t* prompt, size_t max_len, size_t min_len) const {
        int best = -1;
        for (size_t i = 0; i < m_entries.size(); ++i) {
            const std::vector<int32_t>& t = m_entries[i].tokens;
            if (t.size() > max_len || t.size() < min_len ||
                (best >= 0 && t.size() <= m_entries[static_cast<size_t>(best)].tokens.size()))
                continue;
            if (std::equal(t.begin(), t.end(), prompt))
                best = static_cast<int>(i);
        }
        return best;
    }

    // The entry holding exactly history[0, n); -1 when none.
    int find_exact(const int32_t* history, size_t n) const {
        for (size_t i = 0; i < m_entries.size(); ++i) {
            const std::vector<int32_t>& t = m_entries[i].tokens;
            if (t.size() == n && std::equal(t.begin(), t.end(), history))
                return static_cast<int>(i);
        }
        return -1;
    }

    // Entry i becomes the most recently used (it moves to the back).
    void touch(size_t i) {
        std::rotate(m_entries.begin() + static_cast<std::ptrdiff_t>(i),
                    m_entries.begin() + static_cast<std::ptrdiff_t>(i) + 1, m_entries.end());
    }

    // Adds e as the most recently used entry — replacing one with the same
    // file name, whose file e's save just overwrote — and evicts the least
    // recently used until the total is within budget. Returns the files the
    // caller must delete; e's own is among them when it alone exceeds budget.
    std::vector<std::string> add(PrefixCacheEntry e, uint64_t budget) {
        for (size_t i = 0; i < m_entries.size(); ++i) {
            if (m_entries[i].file == e.file) {
                m_total -= m_entries[i].bytes;
                m_entries.erase(m_entries.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
        m_total += e.bytes;
        m_entries.push_back(std::move(e));
        std::vector<std::string> evicted;
        while (!m_entries.empty() && m_total > budget) {
            m_total -= m_entries.front().bytes;
            evicted.push_back(std::move(m_entries.front().file));
            m_entries.erase(m_entries.begin());
        }
        return evicted;
    }

    // Drops entry i (a refused restore) and returns its file name.
    std::string remove(size_t i) {
        std::string file = std::move(m_entries[i].file);
        m_total -= m_entries[i].bytes;
        m_entries.erase(m_entries.begin() + static_cast<std::ptrdiff_t>(i));
        return file;
    }

    // Drop stale and missing entries before choosing a prefix or reusing a save.
    void prune(const std::filesystem::path& dir, std::chrono::seconds ttl) {
        for (size_t i = 0; i < m_entries.size();) {
            const auto path = dir / m_entries[i].file;
            if (prefix_cache_file_expired(path, ttl)) {
                std::error_code ec;
                std::filesystem::remove(path, ec);
                remove(i);
            } else {
                ++i;
            }
        }
    }

    void clear() {
        m_entries.clear();
        m_total = 0;
    }

    const std::vector<PrefixCacheEntry>& entries() const {
        return m_entries;
    }
    uint64_t total_bytes() const {
        return m_total;
    }

  private:
    std::vector<PrefixCacheEntry> m_entries; // least recently used first
    uint64_t m_total = 0;
};

} // namespace xllama
