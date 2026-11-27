// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Which "model" spellings a request may name. Every distinct spelling is its own resident Session
// id with its own manifest entry and env: the catalogue declares aliases on purpose (the same
// weights under a bare id and under a D:\ path, with different env). So a spelling is matched
// byte for byte, never case-folded or merged, and one the server does not know is 404: before, any
// path-looking spelling (and, on NTFS, any case variant of a model directory) registered a new id.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace xllama {

inline bool model_id_is_path(std::string_view model) {
    return model.find_first_of("\\/:") != std::string_view::npos;
}

// The first entry whose name is exactly `name` (std::wstring, as ManifestEntry::name); the entry's
// own env applies to its spelling, nobody else's. Same rule as FindManifestEntry.
template <typename Entry>
const Entry* find_model_spelling(const std::vector<Entry>& entries, const std::wstring& name) {
    for (const Entry& e : entries)
        if (e.name == name)
            return &e;
    return nullptr;
}

// A path spelling is servable when it is the resident model, the operator's model.txt hint, or
// the exact name of a manifest entry (`in_manifest`).
inline bool path_spelling_known(std::string_view model, bool in_manifest, std::string_view resident,
                                std::string_view hint) {
    return !model.empty() && (in_manifest || model == resident || model == hint);
}

} // namespace xllama
