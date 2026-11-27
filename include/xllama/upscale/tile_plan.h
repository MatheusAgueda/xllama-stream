// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Pure helpers of the console upscaler (uwp/upscale.cpp): the static-tile plan, the RGBA8 <->
// planar fp16 copies and base64 decoding. Header-only so tests/test_upscale.cpp checks them on the
// host.
//
// TILE PLAN. The ONNX graph has one static input [1,3,T,T] (DirectML fuses static shapes). Each
// axis is cut into cores of T-2*pad pixels that cover [0, n) exactly once; a core is computed
// inside a window of T pixels that starts pad pixels before it and, when the image is at least T
// long, slides back inside [0, n). At the image border the network then sees its own zero padding,
// as in an untiled run, instead of replicated pixels. Measured on the host 25/11
// (upscale/seam_study.py, Real-ESRGAN x4plus, a site photo shrunk 4x, 128x96, T=64): against the
// untiled run, replicated windows 33.5-33.8 dB PSNR (max error 98-116/255 at the border), sliding
// windows 40.3 / 41.9 / 43.5 dB at pad 10 / 16 / 24 (max 37-49/255). Only the core's 4x pixels are
// written: no blending, a pixel has exactly one source tile.
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "xllama/diffusion/half.h"

namespace xllama::upscale {

constexpr int kScale = 4;

struct AxisTile {
    int window =
        0; // first source pixel the T-pixel window reads (may be < 0 or run past n only when n < T)
    int core_lo = 0; // the core [core_lo, core_hi) this tile writes, in source pixels
    int core_hi = 0;
};

// The plan along one axis of n pixels. Empty when n <= 0, tile <= 0 or the core (tile - 2*pad) is
// not positive.
inline std::vector<AxisTile> plan_axis(int n, int tile, int pad) {
    std::vector<AxisTile> out;
    const int core = tile - 2 * pad;
    if (n <= 0 || tile <= 0 || pad < 0 || core <= 0)
        return out;
    for (int lo = 0; lo < n; lo += core) {
        AxisTile t;
        t.core_lo = lo;
        t.core_hi = std::min(n, lo + core);
        t.window = lo - pad;
        if (n >= tile)
            t.window = std::clamp(t.window, 0, n - tile);
        else
            t.window =
                0; // a short axis: the image starts at the window's edge, the rest is replicated
        out.push_back(t);
    }
    return out;
}

// Copy the T x T window starting at (wx, wy) of an RGBA8 image (row stride in bytes) into planar
// fp16 [3,T,T] in [0,1]; coordinates outside the image are clamped to its edge (only reachable when
// an axis is shorter than T).
inline void window_to_planar_f16(const uint8_t* rgba, int width, int height, size_t stride, int wx,
                                 int wy, int T, uint16_t* dst) {
    static const auto lut = [] {
        std::vector<uint16_t> v(256);
        for (int i = 0; i < 256; ++i)
            v[i] = diffusion::float_to_half(static_cast<float>(i) / 255.0f);
        return v;
    }();
    const size_t plane = static_cast<size_t>(T) * T;
    for (int y = 0; y < T; ++y) {
        const int sy = std::clamp(wy + y, 0, height - 1);
        const uint8_t* row = rgba + static_cast<size_t>(sy) * stride;
        for (int x = 0; x < T; ++x) {
            const int sx = std::clamp(wx + x, 0, width - 1);
            const uint8_t* p = row + static_cast<size_t>(sx) * 4;
            const size_t i = static_cast<size_t>(y) * T + x;
            dst[i] = lut[p[0]];
            dst[plane + i] = lut[p[1]];
            dst[2 * plane + i] = lut[p[2]];
        }
    }
}

inline uint8_t unit_to_u8(float v) {
    if (!(v > 0.0f)) // also NaN
        return 0;
    if (v >= 1.0f)
        return 255;
    return static_cast<uint8_t>(v * 255.0f + 0.5f);
}

// Write a tile's output (planar [3, 4T, 4T], fp16 or fp32 through `at`) for its core into the RGBA8
// output image (out_w = 4 * source width). `at(c, y, x)` returns channel c at output-tile pixel (y,
// x) as float.
template <typename At>
void core_to_rgba(At&& at, const AxisTile& ty, const AxisTile& tx, uint8_t* out,
                  size_t out_stride) {
    const int oy = kScale * (ty.core_lo - ty.window), ox = kScale * (tx.core_lo - tx.window);
    const int rows = kScale * (ty.core_hi - ty.core_lo), cols = kScale * (tx.core_hi - tx.core_lo);
    for (int y = 0; y < rows; ++y) {
        uint8_t* row = out + static_cast<size_t>(kScale * ty.core_lo + y) * out_stride +
                       static_cast<size_t>(kScale * tx.core_lo) * 4;
        for (int x = 0; x < cols; ++x) {
            row[4 * x + 0] = unit_to_u8(at(0, oy + y, ox + x));
            row[4 * x + 1] = unit_to_u8(at(1, oy + y, ox + x));
            row[4 * x + 2] = unit_to_u8(at(2, oy + y, ox + x));
            row[4 * x + 3] = 255;
        }
    }
}

// Standard base64 (RFC 4648, '+' '/'), padding optional, ASCII whitespace ignored. nullopt on any
// other byte or a dangling single symbol.
inline std::optional<std::string> base64_decode(std::string_view in) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z')
            return c - 'A';
        if (c >= 'a' && c <= 'z')
            return c - 'a' + 26;
        if (c >= '0' && c <= '9')
            return c - '0' + 52;
        if (c == '+')
            return 62;
        if (c == '/')
            return 63;
        return -1;
    };
    std::string out;
    out.reserve(in.size() / 4 * 3);
    unsigned acc = 0;
    int bits = 0, pad = 0;
    for (char c : in) {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t')
            continue;
        if (c == '=') {
            ++pad;
            continue;
        }
        const int v = val(c);
        if (v < 0 || pad)
            return std::nullopt; // a symbol after '=' is malformed
        acc = (acc << 6) | static_cast<unsigned>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((acc >> bits) & 0xFF));
        }
    }
    if (bits >= 6 || pad > 2)
        return std::nullopt; // 6 leftover bits = one dangling symbol
    return out;
}

// BUDGET GATE. The upscale session shares the process's one D3D12 device with the chat's GGUF
// layers. On 25/11 (xllama-net 1.7.12.106, the 35B loaded, dxgi 4012/4147 MB, app 4608/5120 MB)
// the DirectML init removed that device (887A0005 in DmlGraphFusionHelper, then 887A0007) and the
// chat's next decode stalled until the watchdog. So a session is only built when both pools have
// `need` free; otherwise the request is refused before anything touches the device.
// "" = go ahead, else the refusal with the figures. A need of 0 or a pool the OS does not report
// (known == false) is not judged.
struct PoolFigures {
    std::size_t used_mb = 0, cap_mb = 0;
    bool known = false;
};

inline std::string budget_refusal(const char* pool, PoolFigures f, std::size_t need_mb) {
    if (!f.known || need_mb == 0 || f.cap_mb == 0)
        return {};
    const std::size_t free_mb = f.used_mb < f.cap_mb ? f.cap_mb - f.used_mb : 0;
    if (free_mb >= need_mb)
        return {};
    return std::string(pool) + ": " + std::to_string(free_mb) + " MB free of " +
           std::to_string(f.cap_mb) + " MB, need " + std::to_string(need_mb) + " MB";
}

inline std::string budget_refusal(PoolFigures gpu, std::size_t gpu_need_mb, PoolFigures app,
                                  std::size_t app_need_mb) {
    std::string g = budget_refusal("gpu-budget", gpu, gpu_need_mb);
    std::string a = budget_refusal("app-memory", app, app_need_mb);
    if (g.empty() || a.empty())
        return g.empty() ? a : g;
    return g + "; " + a;
}

// An MB knob from the environment: unset, empty, junk or negative = the default.
inline std::size_t need_mb_from(const char* text, std::size_t fallback) {
    if (!text || !*text)
        return fallback;
    std::size_t v = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9' || v > 1'000'000)
            return fallback;
        v = v * 10 + static_cast<std::size_t>(*p - '0');
    }
    return v;
}

// Admission of POST /v1/images/upscale (uwp/api-server.cpp): one upscale at a time and never while
// a model preload runs (both want the GPU pool the memory gate just measured). `got_lock` = the
// upscale mutex was free (try_lock won); `model_preloading` = session_hub().preloading. Anything
// else is the bounded 503 "busy" (kApiBusyStatus) before a ticket or a body parse.
inline bool upscale_admit(bool got_lock, bool model_preloading) {
    return got_lock && !model_preloading;
}

} // namespace xllama::upscale
