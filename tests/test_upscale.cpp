// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The console upscaler's pure helpers (include/xllama/upscale/tile_plan.h).

#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "xllama/diffusion/half.h"
#include "xllama/upscale/tile_plan.h"

using namespace xllama::upscale;

namespace {

// Every core inside its window, cores partitioning [0, n), and the window inside [0, n) when n >=
// T.
void check_axis(int n, int T, int pad) {
    const auto plan = plan_axis(n, T, pad);
    REQUIRE_FALSE(plan.empty());
    int next = 0;
    for (const auto& t : plan) {
        CHECK(t.core_lo == next);
        CHECK(t.core_hi > t.core_lo);
        CHECK(t.core_lo >= t.window);
        CHECK(t.core_hi <= t.window + T);
        if (n >= T) {
            CHECK(t.window >= 0);
            CHECK(t.window + T <= n);
            // interior sides keep at least pad pixels of context
            if (t.core_lo > 0)
                CHECK(t.core_lo - t.window >= std::min(pad, t.core_lo));
            if (t.core_hi < n)
                CHECK(t.window + T - t.core_hi >= std::min(pad, n - t.core_hi));
        } else {
            CHECK(t.window == 0);
        }
        next = t.core_hi;
    }
    CHECK(next == n);
}

} // namespace

TEST_CASE("upscale: the tile plan partitions every axis length") {
    for (int T : {64, 128, 192})
        for (int pad : {0, 10, 16, 24})
            for (int n : {1, 7, T - 1, T, T + 1, 2 * T - 2 * pad, 1000, 1224, 1500})
                check_axis(n, T, pad);
}

TEST_CASE("upscale: degenerate plans are empty") {
    CHECK(plan_axis(0, 128, 10).empty());
    CHECK(plan_axis(100, 20, 10).empty()); // core 0
    CHECK(plan_axis(100, 128, -1).empty());
}

TEST_CASE("upscale: a 1024 px axis at T=192 pad 16 takes 7 windows, the last slid back inside") {
    const auto p = plan_axis(1024, 192, 16);
    REQUIRE(p.size() == 7);
    CHECK(p.front().window == 0);
    CHECK(p.back().core_hi == 1024);
    CHECK(p.back().window == 1024 - 192);
}

TEST_CASE("upscale: the window copy is planar fp16 with clamped edges") {
    // 3x2 RGBA image, pixel (x,y) = (10x+y, 100+x, 200+y)
    const int W = 3, H = 2;
    std::vector<uint8_t> img(W * H * 4);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint8_t* p = &img[(y * W + x) * 4];
            p[0] = static_cast<uint8_t>(10 * x + y);
            p[1] = static_cast<uint8_t>(100 + x);
            p[2] = static_cast<uint8_t>(200 + y);
            p[3] = 7;
        }
    const int T = 4;
    std::vector<uint16_t> dst(3 * T * T);
    window_to_planar_f16(img.data(), W, H, W * 4, 0, 0, T, dst.data());
    auto at = [&](int c, int y, int x) {
        return xllama::diffusion::half_to_float(dst[c * T * T + y * T + x]);
    };
    CHECK(at(0, 1, 2) == doctest::Approx(21 / 255.0).epsilon(1e-3));
    CHECK(at(1, 0, 3) == doctest::Approx(102 / 255.0).epsilon(1e-3)); // x clamped to 2
    CHECK(at(2, 3, 0) == doctest::Approx(201 / 255.0).epsilon(1e-3)); // y clamped to 1
}

TEST_CASE("upscale: a core lands at 4x its source position") {
    const int T = 4, pad = 1; // core 2
    const auto ax = plan_axis(5, T, pad);
    REQUIRE(ax.size() == 3);
    const int OW = 4 * 5;
    std::vector<uint8_t> out(OW * OW * 4, 0);
    // the tile's output pixel value encodes its output-tile coordinates
    auto at = [](int c, int y, int x) { return c == 0 ? y / 255.0f : c == 1 ? x / 255.0f : 1.0f; };
    core_to_rgba(at, ax[1], ax[1], out.data(), OW * 4);
    // core [2,4) on both axes, window start 1 → output pixel (8,8) is tile pixel (4,4)
    const uint8_t* p = &out[(8 * OW + 8) * 4];
    CHECK(p[0] == 4);
    CHECK(p[1] == 4);
    CHECK(p[2] == 255);
    CHECK(p[3] == 255);
    CHECK(out[(7 * OW + 8) * 4 + 3] == 0); // outside the core: untouched
    CHECK(out[(15 * OW + 15) * 4 + 3] == 255);
    CHECK(out[(16 * OW + 16) * 4 + 3] == 0);
}

TEST_CASE("upscale: unit_to_u8 clamps and rounds") {
    CHECK(unit_to_u8(-0.5f) == 0);
    CHECK(unit_to_u8(0.0f / 0.0f) == 0);
    CHECK(unit_to_u8(1.7f) == 255);
    CHECK(unit_to_u8(0.5f) == 128);
}

TEST_CASE("upscale: base64 decodes RFC 4648 vectors and refuses junk") {
    CHECK(*base64_decode("") == "");
    CHECK(*base64_decode("Zg==") == "f");
    CHECK(*base64_decode("Zm8=") == "fo");
    CHECK(*base64_decode("Zm9v") == "foo");
    CHECK(*base64_decode("Zm9vYg") == "foob"); // padding optional
    CHECK(*base64_decode("Zm9v\nYmFy") == "foobar");
    CHECK(*base64_decode("/+8=") == std::string("\xff\xef", 2));
    CHECK_FALSE(base64_decode("Zm9v!").has_value());
    CHECK_FALSE(base64_decode("Zg==Zg").has_value());
    CHECK_FALSE(base64_decode("Z").has_value());
}
