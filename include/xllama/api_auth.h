// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <string_view>

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

} // namespace xllama
