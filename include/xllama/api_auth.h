// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

inline std::string api_key_from_text(std::string_view text) {
    const size_t first = text.find_first_not_of(" \t\r\n\f\v");
    if (first == std::string_view::npos)
        return {};
    text = text.substr(first, text.find_last_not_of(" \t\r\n\f\v") - first + 1);
    return text.size() >= 32 ? std::string(text) : std::string();
}

inline bool bearer_matches(std::string_view authorization_header, std::string_view key) {
    if (key.empty() || authorization_header.size() < 7 ||
        authorization_header.size() - 7 != key.size())
        return false;
    constexpr std::string_view scheme = "bearer ";
    for (size_t i = 0; i < scheme.size(); ++i) {
        char c = authorization_header[i];
        if (c >= 'A' && c <= 'Z')
            c += 'a' - 'A';
        if (c != scheme[i])
            return false;
    }
    // Do not reveal the first differing key byte through an early return.
    volatile unsigned char difference = 0;
    for (size_t i = 0; i < key.size(); ++i)
        difference |= static_cast<unsigned char>(authorization_header[7 + i]) ^
                      static_cast<unsigned char>(key[i]);
    return difference == 0;
}

// Two rollout phases, picked at run time (no rebuild) by LocalState/api-auth.txt next to api.key:
// its first word "enforce" (any case) is phase 2; a missing file or anything else is phase 1.
//   Observe  (1, default) the Bearer is verified; a failure is counted per client and still served.
//   Enforce  (2)          a missing or wrong Bearer is 401 on every route but /health, and so is a
//                         missing or short api.key (an empty key matches nothing: fail closed).
enum class ApiAuthPhase { Observe, Enforce };

inline ApiAuthPhase api_auth_phase_from_text(std::string_view text) {
    const size_t first = text.find_first_not_of(" \t\r\n\f\v");
    if (first == std::string_view::npos)
        return ApiAuthPhase::Observe;
    text = text.substr(first);
    const size_t end = text.find_first_of(" \t\r\n\f\v");
    if (end != std::string_view::npos)
        text = text.substr(0, end);
    constexpr std::string_view word = "enforce";
    if (text.size() != word.size())
        return ApiAuthPhase::Observe;
    for (size_t i = 0; i < word.size(); ++i) {
        char c = text[i];
        if (c >= 'A' && c <= 'Z')
            c += 'a' - 'A';
        if (c != word[i])
            return ApiAuthPhase::Observe;
    }
    return ApiAuthPhase::Enforce;
}

struct ApiAuthDecision {
    bool allow = true;   // serve the request
    bool failed = false; // the Bearer did not match: count it
};

// The verdict for one request on a route that is not /health. `key` is the provisioned key ("" =
// missing or short), so it never matches.
inline ApiAuthDecision decide_api_auth(ApiAuthPhase phase, std::string_view key,
                                       std::string_view authorization) {
    ApiAuthDecision d;
    if (bearer_matches(authorization, key))
        return d;
    d.failed = true;
    d.allow = phase == ApiAuthPhase::Observe;
    return d;
}

// Counts of Bearer failures, never the key or the header. Per client address, bounded: past
// kMaxClients distinct addresses a new one is counted under `other`.
class ApiAuthStats {
  public:
    static constexpr size_t kMaxClients = 64;

    struct Snapshot {
        uint64_t failures = 0; // served + denied
        uint64_t served = 0;   // phase 1: failed and still answered
        uint64_t denied = 0;   // phase 2: answered 401
        uint64_t other = 0;    // failures of clients past kMaxClients
        std::vector<std::pair<std::string, uint64_t>> clients;
    };

    void record_failure(std::string_view client, bool served) {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++(served ? m_served : m_denied);
        auto it = m_clients.find(std::string(client));
        if (it != m_clients.end())
            ++it->second;
        else if (m_clients.size() < kMaxClients)
            m_clients.emplace(std::string(client), 1);
        else
            ++m_other;
    }

    Snapshot snapshot() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        Snapshot s;
        s.served = m_served;
        s.denied = m_denied;
        s.failures = m_served + m_denied;
        s.other = m_other;
        s.clients.assign(m_clients.begin(), m_clients.end());
        return s;
    }

  private:
    mutable std::mutex m_mutex;
    uint64_t m_served = 0;
    uint64_t m_denied = 0;
    uint64_t m_other = 0;
    std::map<std::string, uint64_t> m_clients;
};

inline std::string api_auth_stats_json(ApiAuthPhase phase, const ApiAuthStats::Snapshot& s) {
    std::string clients;
    for (const auto& [client, n] : s.clients) {
        if (!clients.empty())
            clients += ',';
        clients += '"';
        for (const char c : client) {
            const unsigned char u = static_cast<unsigned char>(c);
            if (c == '"' || c == '\\' || u < 0x20 || u >= 0x7f)
                clients += '?'; // addresses are ASCII; never emit a raw control byte
            else
                clients += c;
        }
        clients += "\":" + std::to_string(n);
    }
    return std::string("{\"phase\":\"") + (phase == ApiAuthPhase::Enforce ? "enforce" : "observe") +
           "\",\"failures\":" + std::to_string(s.failures) +
           ",\"served\":" + std::to_string(s.served) + ",\"denied\":" + std::to_string(s.denied) +
           ",\"other\":" + std::to_string(s.other) + ",\"clients\":{" + clients + "}}";
}

} // namespace xllama
