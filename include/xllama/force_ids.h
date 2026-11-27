// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The "xllama_force_ids" request field of POST /v1/chat/completions (research API): token ids the
// reply is forced to be, scored under the load's own knobs. The WinRT JSON layer in
// uwp/api-server.cpp only turns the array into ForceIdElement values; the checks live here so
// tests/test_api_policy.cpp pins them on the host.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

enum class ForceIdsError {
    None,
    NotArray, // the field is present but is not a JSON array
    BadId,    // an element is not a whole number in [0, INT32_MAX]
    TooMany,  // more ids than max_tokens leaves room for
};

// One array element as the JSON layer saw it; `number` is read only when `is_number`.
struct ForceIdElement {
    bool is_number = false;
    double number = 0.0;
};

struct ForceIdsResult {
    ForceIdsError error = ForceIdsError::None;
    std::vector<int32_t> ids; // every id, also under TooMany (its message counts them)
};

// A token id: a JSON number that is a whole non-negative value up to INT32_MAX. NaN, fractions,
// negatives and anything past int32 are refused before any cast.
inline bool force_id_from_number(double v, int32_t& out) {
    if (!(v >= 0.0 && v <= 2147483647.0) || v != static_cast<double>(static_cast<int64_t>(v)))
        return false;
    out = static_cast<int32_t>(v);
    return true;
}

// `is_array`: the field's JSON type is an array (`elems` is then its elements). Checked in the
// order the handler always did: type, then each element (the first bad one stops), then the count
// against `n_predict` (the validated max_tokens).
inline ForceIdsResult parse_force_ids(bool is_array, const std::vector<ForceIdElement>& elems,
                                      int n_predict) {
    ForceIdsResult r;
    if (!is_array) {
        r.error = ForceIdsError::NotArray;
        return r;
    }
    for (const ForceIdElement& e : elems) {
        int32_t id = 0;
        if (!e.is_number || !force_id_from_number(e.number, id)) {
            r.error = ForceIdsError::BadId;
            r.ids.clear();
            return r;
        }
        r.ids.push_back(id);
    }
    if (static_cast<int>(r.ids.size()) > n_predict)
        r.error = ForceIdsError::TooMany;
    return r;
}

// The 400 body message of an error; `n_ids` is ForceIdsResult::ids.size().
inline std::string force_ids_message(ForceIdsError error, size_t n_ids) {
    switch (error) {
    case ForceIdsError::NotArray:
        return "'xllama_force_ids' must be an array of token ids";
    case ForceIdsError::BadId:
        return "'xllama_force_ids' must hold whole non-negative numbers";
    case ForceIdsError::TooMany:
        return "max_tokens must cover the " + std::to_string(n_ids) + " forced ids";
    case ForceIdsError::None:
        break;
    }
    return std::string();
}

} // namespace xllama
