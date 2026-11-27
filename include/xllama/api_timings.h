// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include "xllama/inference_params.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

namespace xllama {

/// Where a request was, in ms since its connection was accepted: the compute permit and the
/// session granted, the first generated token, the last one. A stamp never set stays -1 and is
/// left out of the reply. Plain data plus explicit time points, so it is host-testable; the server
/// keeps one per request on the thread that serves it.
struct RequestStamps {
    using Clock = std::chrono::steady_clock;

    Clock::time_point accepted{};
    double permit_ms = -1.0;
    double first_ms = -1.0;
    double last_ms = -1.0;

    void begin(Clock::time_point now) {
        *this = RequestStamps{};
        accepted = now;
    }
    double since_accept(Clock::time_point now) const {
        return std::chrono::duration<double, std::milli>(now - accepted).count();
    }
    void permit(Clock::time_point now) {
        permit_ms = since_accept(now);
    }
    /// Only the first call counts.
    void first(Clock::time_point now) {
        if (first_ms < 0.0)
            first_ms = since_accept(now);
    }
    void last(Clock::time_point now) {
        last_ms = since_accept(now);
    }
    /// generate() returned at `now`. A call without a token callback never stamped the first token:
    /// derive it from the result's t_first_token_ms, which counts from the start of the prefill
    /// (a little after `gen_start`, so the derived stamp is the earlier-side estimate).
    void generated(Clock::time_point gen_start, Clock::time_point now, double first_token_ms) {
        if (first_ms < 0.0 && first_token_ms > 0.0 && std::isfinite(first_token_ms))
            first_ms = since_accept(gen_start) + first_token_ms;
        last(now);
    }
};

inline std::string api_timings_json(const InferenceResult& result,
                                    const RequestStamps* stamps = nullptr) {
    // inf/nan print as words JSON cannot parse: the caller's JsonObject::Parse would throw and drop
    // the reply
    const auto ms = [](double v) { return std::isfinite(v) ? v : 0.0; };
    std::ostringstream out;
    // JSON numbers always use a decimal point, whatever the console locale.
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(1) << "{\"prefill_ms\":" << ms(result.t_p_eval_ms)
        << ",\"decode_ms\":" << ms(result.t_eval_ms) << ",\"prompt_n\":" << result.n_p_eval
        << ",\"predicted_n\":" << result.n_eval;
    if (stamps && stamps->permit_ms >= 0.0) {
        out << ",\"stamps\":{\"permit_ms\":" << ms(stamps->permit_ms);
        if (stamps->first_ms >= 0.0)
            out << ",\"first_ms\":" << ms(stamps->first_ms);
        if (stamps->last_ms >= 0.0)
            out << ",\"last_ms\":" << ms(stamps->last_ms);
        out << "}";
    }
    // Only a model with an expert cache has the object: a client that does not know it ignores it,
    // one that does can tell "no cache" from "a cache that did nothing".
    if (result.expert.present) {
        const ExpertStats& e = result.expert;
        out << ",\"expert\":{\"hits\":" << e.hits << ",\"misses\":" << e.misses
            << ",\"swaps\":" << e.swaps << ",\"em_swaps\":" << e.em_swaps
            << ",\"bytes\":" << e.bytes << ",\"read_us\":" << e.read_us
            << ",\"em_reads\":" << e.em_reads << ",\"em_bytes\":" << e.em_bytes << "}";
    }
    out << "}";
    return out.str();
}

} // namespace xllama
