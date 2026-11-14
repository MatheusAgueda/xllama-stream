// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include "xllama/inference_params.h"

#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

namespace xllama {

inline std::string api_timings_json(const InferenceResult& result) {
    // inf/nan print as words JSON cannot parse: the caller's JsonObject::Parse would throw and drop the reply
    const auto ms = [](double v) { return std::isfinite(v) ? v : 0.0; };
    std::ostringstream out;
    // JSON numbers always use a decimal point, whatever the console locale.
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(1) << "{\"prefill_ms\":" << ms(result.t_p_eval_ms)
        << ",\"decode_ms\":" << ms(result.t_eval_ms) << ",\"prompt_n\":" << result.n_p_eval
        << ",\"predicted_n\":" << result.n_eval << "}";
    return out.str();
}

} // namespace xllama
