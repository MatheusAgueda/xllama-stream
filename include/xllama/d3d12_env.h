// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The environment-variable parses of the D3D12 bridge and of llama.cpp patch 0026 that gate the
// GPU attention path, as pure functions of the variable's text (nullptr = unset) so
// tests/test_d3d12_env.cpp pins their exact edge semantics on the host. src/bridge/ggml_d3d12.cpp
// calls them; nothing here reads the environment itself.
#pragma once

#include <cstdlib>
#include <cstring>

namespace xllama {

// An on/off knob the bridge reads as `s && *s && *s != '0'`: only the FIRST character decides, so
// "" and "0" and "01" are off while "1", "10" and "yes" are on. XLLAMA_ATTN_GPU (attn_gpu_on) is
// read through this.
inline bool env_flag_on(const char* s) {
    return s && *s && *s != '0';
}

// XLLAMA_FA_DECODE_KV: the smallest KV-cache length (positions) from which a decode token's
// attention runs on the GPU. 0 = off, which is also what unset, empty, non-numeric, below 1 and
// above 0x7fffffff give; it is honoured only when the flash-attention kernels were built
// (`fap_ok`). Parsed like strtoll base 10: leading blanks and a '+' are skipped and trailing text
// after the digits is ignored ("1280abc" = 1280), as production always did.
inline long long fa_decode_kv_from_env(const char* s, bool fap_ok) {
    if (!fap_ok || !s || !*s)
        return 0;
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    return end != s && v >= 1 && v <= 0x7fffffffLL ? v : 0;
}

// LLAMA_KV_ON_DEVICE as llama.cpp patch 0026 reads it (patches/0026-kv-cache-on-device.patch,
// llama_kv_cache's constructor): `kv_dev && *kv_dev && *kv_dev != '0'`, the same first-character
// rule as env_flag_on (XLLAMA_ATTN_GPU). It used to be `strcmp(kv_dev, "0") != 0`, which turned ""
// and "00" ON; production sets only "1", "0" or nothing, which read the same under both. This
// function is a mirror (the patch cannot include xllama headers); test_d3d12_env.cpp reads the
// patch file and fails if its expression drifts from this one.
inline bool kv_on_device_from_env(const char* s) {
    return env_flag_on(s);
}

} // namespace xllama
