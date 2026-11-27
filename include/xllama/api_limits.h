// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Request-size and rate limits of the LAN API that sit in front of the compute slot, WinRT-free so
// tests/test_api_hardening.cpp pins them on the host. uwp/api-server.cpp calls them.
#pragma once

#include "xllama/compute_admission.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>

namespace xllama {

// Text a chat request may ask the server to tokenize. The framing allows an 8 MiB body
// (kHttpMaxBody); BPE over that, with the compute permit held, is minutes of the one slot. 256 KiB
// is far past any n_ctx the catalogue serves (a 32K-token context is ~128 KiB of text).
inline constexpr size_t kApiMaxPromptTextBytes = 256 * 1024;

enum class TextAdmitStatus { Admitted, TooLarge, Busy };

struct TextAdmission {
    TextAdmitStatus status = TextAdmitStatus::Busy;
    ComputeJobAdmission::Handle permit;
};

// The text cap comes first: an oversized prompt is 413 without touching the permit
// (`acquire_permit` is not even called), so it can neither queue behind nor take the slot.
template <typename AcquirePermit>
TextAdmission admit_text_then_permit(size_t text_bytes, AcquirePermit&& acquire_permit) {
    TextAdmission a;
    if (text_bytes > kApiMaxPromptTextBytes) {
        a.status = TextAdmitStatus::TooLarge;
        return a;
    }
    a.permit = acquire_permit();
    a.status = a.permit ? TextAdmitStatus::Admitted : TextAdmitStatus::Busy;
    return a;
}

// POST /v1/preferences appends to training/samples.jsonl and takes no compute permit (it must not
// block inference), so it has its own bounds: a sample's body size, samples per window across all
// clients, and the file's total size.
inline constexpr size_t kPreferenceMaxBody = 64 * 1024;
inline constexpr int kPreferenceMaxPerWindow = 30;
inline constexpr std::chrono::seconds kPreferenceWindow{60};
inline constexpr uint64_t kPreferenceMaxStoreBytes = 32ull * 1024 * 1024;

class PreferenceQuota {
  public:
    using Clock = std::chrono::steady_clock;
    enum class Verdict { Ok, TooLarge, RateLimited, StoreFull };

    // `store_bytes`: the samples file's current size. Only an Ok sample is counted in the window.
    Verdict admit(size_t body_bytes, uint64_t store_bytes, Clock::time_point now) {
        if (body_bytes > kPreferenceMaxBody)
            return Verdict::TooLarge;
        if (store_bytes + body_bytes > kPreferenceMaxStoreBytes)
            return Verdict::StoreFull;
        std::lock_guard<std::mutex> lock(m_mutex);
        while (!m_stamps.empty() && now - m_stamps.front() >= kPreferenceWindow)
            m_stamps.pop_front();
        if (m_stamps.size() >= static_cast<size_t>(kPreferenceMaxPerWindow))
            return Verdict::RateLimited;
        m_stamps.push_back(now);
        return Verdict::Ok;
    }

  private:
    std::mutex m_mutex;
    std::deque<Clock::time_point> m_stamps; // never more than kPreferenceMaxPerWindow
};

} // namespace xllama
