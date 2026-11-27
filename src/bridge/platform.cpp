// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/platform.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#ifdef XLLAMA_UWP
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
// clang-format off
    #include <windows.h>
    #define PSAPI_VERSION 2
    #include <psapi.h>
    #include <unknwn.h>
    #include <winrt/Windows.Foundation.h>
    #include <winrt/Windows.Foundation.Metadata.h>
    #include <winrt/Windows.Storage.h>
    #include <winrt/Windows.System.h>
    #include <dxgi1_4.h>
// clang-format on
#endif

namespace xllama {

int detect_threads() noexcept {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    return n > 0 ? n : 4;
}

int detect_threads_llama() noexcept {
#ifdef XLLAMA_UWP
    return std::min(detect_threads(), 6); // ggml livelock at t7/t8 (see platform.h)
#else
    return detect_threads();
#endif
}

void log_output(const char* msg) noexcept {
#ifdef XLLAMA_UWP
    OutputDebugStringA(msg);
    // Also mirror to LocalFolder/xllama.log for post-mortem (bridge errors invisible otherwise).
    static std::mutex s_mtx;
    static FILE* s_fp = []() -> FILE* {
        try {
            auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
            std::wstring path = std::wstring(folder.Path().c_str()) + L"\\xllama.log";
            FILE* fp = _wfopen(path.c_str(), L"a");
            if (fp)
                setvbuf(fp, nullptr, _IONBF, 0); // unbuffered: writes survive hard kill
            return fp;
        } catch (...) {
            return nullptr;
        }
    }();
    if (s_fp) {
        std::lock_guard<std::mutex> g(s_mtx);
        std::fputs(msg, s_fp); // _IONBF: no CRT buffer, write reaches kernel directly
    }
#else
    std::fputs(msg, stderr);
#endif
}

void log_output(const std::string& msg) noexcept {
    log_output(msg.c_str());
}

std::size_t peak_working_set_mb() noexcept {
#ifdef XLLAMA_UWP
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.PeakWorkingSetSize / (1024 * 1024);
    return 0;
#else
    // Linux host: peak RSS from /proc (VmHWM is in kB). Was a hard 0 before —
    // host coding/campaign benches then reported peak=0MB and looked broken.
    FILE* fp = std::fopen("/proc/self/status", "r");
    if (!fp)
        return 0;
    char line[256];
    std::size_t kb = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "VmHWM:", 6) == 0) {
            kb = static_cast<std::size_t>(std::strtoul(line + 6, nullptr, 10));
            break;
        }
    }
    std::fclose(fp);
    return kb / 1024;
#endif
}

std::size_t working_set_mb() noexcept {
#ifdef XLLAMA_UWP
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / (1024 * 1024);
    return 0;
#else
    FILE* fp = std::fopen("/proc/self/status", "r");
    if (!fp)
        return 0;
    char line[256];
    std::size_t kb = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "VmRSS:", 6) == 0) {
            kb = static_cast<std::size_t>(std::strtoul(line + 6, nullptr, 10));
            break;
        }
    }
    std::fclose(fp);
    return kb / 1024;
#endif
}

std::size_t avail_phys_mb() noexcept {
#ifdef XLLAMA_UWP
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms))
        return static_cast<std::size_t>(ms.ullAvailPhys / (1024 * 1024));
    return 0;
#else
    // MemAvailable, not MemFree: the kernel's own estimate of what a new
    // allocation can claim without swapping (includes reclaimable page cache).
    FILE* fp = std::fopen("/proc/meminfo", "r");
    if (!fp)
        return 0;
    char line[256];
    std::size_t kb = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            kb = static_cast<std::size_t>(std::strtoul(line + 13, nullptr, 10));
            break;
        }
    }
    std::fclose(fp);
    return kb / 1024;
#endif
}

std::size_t avail_commit_mb() noexcept {
#ifdef XLLAMA_UWP
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms))
        return static_cast<std::size_t>(ms.ullAvailPageFile / (1024 * 1024));
    return 0;
#else
    return avail_phys_mb();
#endif
}

std::size_t private_mb() noexcept {
#ifdef XLLAMA_UWP
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(),
                                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)))
        return pmc.PrivateUsage / (1024 * 1024);
    return 0;
#else
    return 0;
#endif
}

namespace {
std::atomic<bool> g_gpu_probe_open{false};
std::atomic<GpuReclaimHook> g_gpu_reclaim{nullptr};
} // namespace

bool gpu_mem_probe_open() noexcept {
    return g_gpu_probe_open.load(std::memory_order_acquire);
}

void set_gpu_reclaim_hook(GpuReclaimHook hook) noexcept {
    g_gpu_reclaim.store(hook, std::memory_order_release);
}

void run_gpu_reclaim_hook() noexcept {
    if (GpuReclaimHook h = g_gpu_reclaim.load(std::memory_order_acquire))
        h();
}

GpuMemInfo gpu_mem_info() noexcept {
#ifdef XLLAMA_UWP
    // Adapter cached for process lifetime: gpu_mem_info() is called per phase
    // (pre-load / post-load / post-decode) and must stay cheap.
    static IDXGIAdapter3* s_adapter = []() -> IDXGIAdapter3* {
        IDXGIFactory1* factory = nullptr;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
            return nullptr;
        IDXGIAdapter1* adapter1 = nullptr;
        IDXGIAdapter3* adapter3 = nullptr;
        if (SUCCEEDED(factory->EnumAdapters1(0, &adapter1))) {
            adapter1->QueryInterface(IID_PPV_ARGS(&adapter3));
            adapter1->Release();
        }
        factory->Release();
        g_gpu_probe_open.store(true, std::memory_order_release);
        return adapter3;
    }();

    GpuMemInfo info;
    if (!s_adapter)
        return info;
    DXGI_QUERY_VIDEO_MEMORY_INFO vmi{};
    if (FAILED(s_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vmi)))
        return info;
    info.current_mb = static_cast<std::size_t>(vmi.CurrentUsage / (1024 * 1024));
    info.budget_mb = static_cast<std::size_t>(vmi.Budget / (1024 * 1024));
    info.available = true;
    return info;
#else
    return {};
#endif
}

AppMemInfo app_mem_info() noexcept {
    AppMemInfo info;
#ifdef XLLAMA_UWP
    try {
        using winrt::Windows::System::MemoryManager;
        constexpr unsigned long long kMiB = 1024ull * 1024ull;
        info.usage_mb = static_cast<std::size_t>(MemoryManager::AppMemoryUsage() / kMiB);
        info.limit_mb = static_cast<std::size_t>(MemoryManager::AppMemoryUsageLimit() / kMiB);
        info.level = static_cast<int>(MemoryManager::AppMemoryUsageLevel());
        static const bool s_expected =
            winrt::Windows::Foundation::Metadata::ApiInformation::IsPropertyPresent(
                L"Windows.System.MemoryManager", L"ExpectedAppMemoryUsageLimit");
        if (s_expected)
            info.expected_mb =
                static_cast<std::size_t>(MemoryManager::ExpectedAppMemoryUsageLimit() / kMiB);
        info.available = true;
    } catch (...) {
    }
#endif
    return info;
}

void watch_app_memory() noexcept {
#ifdef XLLAMA_UWP
    static std::once_flag s_once;
    try {
        std::call_once(s_once, [] {
            using winrt::Windows::System::MemoryManager;
            (void)MemoryManager::AppMemoryUsageLimitChanging(
                [](winrt::Windows::Foundation::IInspectable const&,
                   winrt::Windows::System::AppMemoryUsageLimitChangingEventArgs const& a) {
                    try {
                        constexpr unsigned long long kMiB = 1024ull * 1024ull;
                        char b[192];
                        std::snprintf(
                            b, sizeof(b),
                            "[xllama] app-mem limit changing: %llu -> %llu MB (usage %llu MB)\n",
                            static_cast<unsigned long long>(a.OldLimit()) / kMiB,
                            static_cast<unsigned long long>(a.NewLimit()) / kMiB,
                            static_cast<unsigned long long>(MemoryManager::AppMemoryUsage()) /
                                kMiB);
                        log_output(b);
                    } catch (...) {
                    }
                });
            (void)MemoryManager::AppMemoryUsageIncreased(
                [](winrt::Windows::Foundation::IInspectable const&,
                   winrt::Windows::Foundation::IInspectable const&) {
                    const AppMemInfo m = app_mem_info();
                    char b[192];
                    std::snprintf(
                        b, sizeof(b),
                        "[xllama] app-mem usage increased: %zu/%zu MB expected %zu MB level %d\n",
                        m.usage_mb, m.limit_mb, m.expected_mb, m.level);
                    log_output(b);
                });
            const AppMemInfo m = app_mem_info();
            char b[192];
            std::snprintf(b, sizeof(b),
                          "[xllama] app-mem watch: %zu/%zu MB expected %zu MB level %d\n",
                          m.usage_mb, m.limit_mb, m.expected_mb, m.level);
            log_output(b);
        });
    } catch (...) {
    }
#endif
}

void set_cwd_to_local_folder() noexcept {
#ifdef XLLAMA_UWP
    try {
        auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
        SetCurrentDirectoryW(folder.Path().c_str());
    } catch (...) {
    }
#endif
}

} // namespace xllama
