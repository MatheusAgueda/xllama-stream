// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/gpubw.h"

#include <string>
#include <vector>

using namespace xllama;

TEST_CASE("gpubw: pattern is deterministic and checksum is pure XOR") {
    constexpr std::size_t n = 1024;
    std::vector<std::uint32_t> a(n), b(n);
    gpubw_fill_pattern(a.data(), n);
    gpubw_fill_pattern(b.data(), n);
    for (std::size_t i = 0; i < n; ++i)
        CHECK(a[i] == b[i]);
    CHECK(a[0] == gpubw_pattern_word(0));
    CHECK(a[1] == gpubw_pattern_word(1));

    const std::uint32_t c = gpubw_checksum_words(a.data(), n);
    std::uint32_t expect = 0;
    for (std::size_t i = 0; i < n; ++i)
        expect ^= a[i];
    CHECK(c == expect);

    // Associativity: split reduction matches whole-buffer XOR.
    const std::uint32_t left = gpubw_checksum_words(a.data(), n / 2);
    const std::uint32_t right = gpubw_checksum_words(a.data() + n / 2, n - n / 2);
    CHECK((left ^ right) == c);
}

TEST_CASE("gpubw: kill gate helper requires real D3D12 success") {
    GpubwResult fail{};
    fail.read_gbs = 200.0;
    fail.checksum_ok = true;
    fail.d3d12_ran = false;
    CHECK_FALSE(gpubw_passes_kill_gate(fail));

    GpubwResult low{};
    low.read_gbs = 50.0;
    low.checksum_ok = true;
    low.d3d12_ran = true;
    CHECK_FALSE(gpubw_passes_kill_gate(low));
    CHECK(kGpubwKillReadGbs == 100.0);

    GpubwResult pass{};
    pass.read_gbs = 100.0;
    pass.checksum_ok = true;
    pass.d3d12_ran = true;
    CHECK(gpubw_passes_kill_gate(pass));
}

TEST_CASE("gpubw: CSV row matches header column count") {
    GpubwResult r;
    r.buffer_bytes = 64 * 1024 * 1024;
    r.iterations = 3;
    r.read_gbs = 12.5;
    r.checksum_ok = true;
    r.d3d12_ran = false;
    r.checksum = 1;
    r.expected_checksum = 1;
    r.error_msg = "d3d12 unavailable on this platform";
    std::string header = gpubw_csv_header();
    std::string row = format_gpubw_row(r, "unittest");
    auto commas = [](const std::string& s) {
        int n = 0;
        for (char c : s)
            if (c == ',')
                ++n;
        return n;
    };
    CHECK(commas(header) == commas(row));
    CHECK(row.find("unittest") != std::string::npos);
    CHECK(row.back() == '\n');
}

TEST_CASE("gpubw: measure_gpubw on non-D3D12 host reports unavailable") {
    // On Linux CI this exercises the shipped entry point without claiming a
    // GPU bandwidth number (hardware-dependent; console is the gate).
    GpubwResult r = measure_gpubw(/*bytes=*/1u << 20, /*iterations=*/1);
    CHECK(r.buffer_bytes >= sizeof(std::uint32_t));
    CHECK(r.iterations == 1);
#if !defined(_WIN32)
    CHECK_FALSE(r.d3d12_ran);
    CHECK_FALSE(r.error_msg.empty());
    CHECK_FALSE(gpubw_passes_kill_gate(r));
#endif
}

TEST_CASE("gpubw: plan_dispatch keeps every dim ≤ 65535 for default 1 GiB") {
    // Skeptic: 1 GiB / 4 bytes / 256 threads = 1_048_576 groups > 65535 on X.
    const std::size_t n_words = kGpubwDefaultBufferBytes / sizeof(std::uint32_t);
    CHECK(n_words == (static_cast<std::size_t>(1) << 28));

    const GpubwDispatch d = gpubw_plan_dispatch(n_words);
    CHECK(gpubw_dispatch_dims_legal(d));
    CHECK(d.groups_x <= kGpubwMaxGroupsPerDim);
    CHECK(d.groups_y <= kGpubwMaxGroupsPerDim);
    CHECK(d.groups_z <= kGpubwMaxGroupsPerDim);
    CHECK(d.groups_x >= 1);
    CHECK(d.groups_y >= 1);
    CHECK(d.groups_z >= 1);

    const std::uint64_t need =
        (static_cast<std::uint64_t>(n_words) + kGpubwThreadsPerGroup - 1) / kGpubwThreadsPerGroup;
    const std::uint64_t launched = static_cast<std::uint64_t>(d.groups_x) * d.groups_y * d.groups_z;
    CHECK(launched >= need);
    CHECK(d.n_groups == launched);
    // Prefer a balanced 2D grid for the default working set (1024×1024).
    CHECK(d.groups_x == 1024u);
    CHECK(d.groups_y == 1024u);
    CHECK(d.groups_z == 1u);
}

TEST_CASE("gpubw: plan_dispatch is 1D when groups fit in one axis") {
    // 256 words → 1 group; 65535*256 words → max-X 1D.
    {
        const GpubwDispatch d = gpubw_plan_dispatch(256);
        CHECK(gpubw_dispatch_dims_legal(d));
        CHECK(d.groups_x == 1u);
        CHECK(d.groups_y == 1u);
        CHECK(d.groups_z == 1u);
        CHECK(d.n_groups == 1u);
    }
    {
        const std::size_t n_words =
            static_cast<std::size_t>(kGpubwMaxGroupsPerDim) * kGpubwThreadsPerGroup;
        const GpubwDispatch d = gpubw_plan_dispatch(n_words);
        CHECK(gpubw_dispatch_dims_legal(d));
        CHECK(d.groups_x == kGpubwMaxGroupsPerDim);
        CHECK(d.groups_y == 1u);
        CHECK(d.groups_z == 1u);
        CHECK(d.n_groups == kGpubwMaxGroupsPerDim);
    }
    {
        // Just over the 1D limit → 2D.
        const std::size_t n_words =
            static_cast<std::size_t>(kGpubwMaxGroupsPerDim) * kGpubwThreadsPerGroup + 1;
        const GpubwDispatch d = gpubw_plan_dispatch(n_words);
        CHECK(gpubw_dispatch_dims_legal(d));
        CHECK(d.groups_y >= 2u);
        const std::uint64_t need =
            (static_cast<std::uint64_t>(n_words) + kGpubwThreadsPerGroup - 1) /
            kGpubwThreadsPerGroup;
        CHECK(static_cast<std::uint64_t>(d.n_groups) >= need);
    }
}

TEST_CASE("gpubw: sweep plan fits the buffer with exact whole-group slices") {
    const auto plan = gpubw_sweep_plan(kGpubwDefaultBufferBytes);
    REQUIRE(plan.size() == 13u); // round trip + 6 sizes x {barriers, none}
    CHECK(plan[0].dispatches == 1u);
    CHECK_FALSE(plan[0].barriers);
    for (const GpubwSweepCase& c : plan) {
        CHECK(c.dispatches >= 1u);
        CHECK(c.dispatches <= kGpubwSweepMaxDispatches);
        CHECK(c.reps >= 5);
        CHECK(static_cast<std::size_t>(c.dispatches) * c.dispatch_bytes <=
              kGpubwDefaultBufferBytes);
        const std::size_t wpd = c.dispatch_bytes / sizeof(std::uint32_t);
        CHECK(wpd % kGpubwThreadsPerGroup == 0u);
        const GpubwDispatch d = gpubw_plan_dispatch(wpd);
        CHECK(gpubw_dispatch_dims_legal(d));
        CHECK(static_cast<std::size_t>(d.n_groups) * kGpubwThreadsPerGroup == wpd);
    }
    // The 1 GiB input covers 2048 slices of 16 KiB, 16 of 64 MiB.
    CHECK(plan[1].dispatch_bytes == 16u * 1024u);
    CHECK(plan[1].dispatches == kGpubwSweepMaxDispatches);
    CHECK(plan.back().dispatch_bytes == 64u * 1024u * 1024u);
    CHECK(plan.back().dispatches == 16u);
    CHECK(gpubw_sweep_plan(8 * 1024).empty());
}

TEST_CASE("gpubw: sweep CSV row matches header and derives us/dispatch and GB/s") {
    GpubwSweepRow row;
    row.c = {1024 * 1024, 1000u, true, 5};
    row.best_ms = 10.0; // 1000 x 1 MiB in 10 ms
    row.median_ms = 11.0;
    row.checksum_ok = true;
    const std::string line = format_gpubw_sweep_row(row);
    const std::string header = gpubw_sweep_csv_header();
    auto commas = [](const std::string& s) {
        int n = 0;
        for (char c : s)
            if (c == ',')
                ++n;
        return n;
    };
    CHECK(commas(header) == commas(line));
    CHECK(line == "1024,1000,1,5,10.0000,11.0000,10.000,104.86,1,0,0\n");

    GpubwSweepRow pr;
    pr.c = gpubw_pred_plan(kGpubwDefaultBufferBytes).at(1);
    pr.best_ms = 12.0;
    pr.median_ms = 12.5;
    const std::string pline = format_gpubw_sweep_row(pr);
    CHECK(commas(header) == commas(pline));
    CHECK(pline.substr(pline.size() - 6) == ",39,1\n");
}

TEST_CASE("gpubw: predication plan is ABBA, skip control last, disjoint regions in bounds") {
    const auto plan = gpubw_pred_plan(kGpubwDefaultBufferBytes);
    REQUIRE(plan.size() == 7u);
    const std::uint32_t modes[] = {0u, 1u, 2u, 2u, 1u, 0u, 3u};
    const std::size_t groups =
        kGpubwDefaultBufferBytes / sizeof(std::uint32_t) / kGpubwThreadsPerGroup;
    int reps[4] = {0, 0, 0, 0};
    for (std::size_t j = 0; j < plan.size(); ++j) {
        const GpubwSweepCase& c = plan[j];
        CHECK(c.pred_mode == modes[j]);
        CHECK(c.pred_ranges == (c.pred_mode ? kGpubwPredRanges : 0u));
        CHECK(c.dispatches == kGpubwPredDispatches);
        CHECK(c.dispatch_bytes == 16u * 1024u);
        CHECK(c.barriers);
        reps[c.pred_mode] += c.reps;
        const std::size_t wpd = c.dispatch_bytes / sizeof(std::uint32_t);
        const GpubwDispatch d = gpubw_plan_dispatch(wpd);
        const std::size_t span = static_cast<std::size_t>(c.dispatches) * d.n_groups;
        // Past the 16 KiB sweep case (2048 slices) so stale data never matches.
        CHECK(c.out_offset >= kGpubwSweepMaxDispatches * d.n_groups);
        CHECK(c.out_offset + span <= groups);
        if (j > 0)
            CHECK(plan[j - 1].out_offset + span <= c.out_offset);
    }
    // >= 200 timed lists per arm.
    CHECK(reps[0] >= 200);
    CHECK(reps[1] >= 200);
    CHECK(reps[2] >= 200);
    CHECK(reps[3] >= 1);
    CHECK(gpubw_pred_plan(1u << 20).empty());
}
