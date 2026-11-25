// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// 4x photo upscale on the console GPU: Real-ESRGAN x4plus (xinntao, BSD-3-Clause) as a static-tile
// fp16 ONNX graph on plain ONNX Runtime DirectML, beside the chat. The chat's decode is bound by
// the expert reads (CPU + SSD) and leaves the GPU idle between its kernels, so this runs on its own
// lock, not the compute permit or the hub.
//
// MODEL CONTRACT: LocalState\models\realesrgan-x4plus\model.onnx, one input [1,3,T,T] and one
// output [1,3,4T,4T], RGB in [0,1], fp16 or fp32 (read from the session). Exported and checked on
// the host by upscale/ export_realesrgan.py in the xbox-xllama workspace (ORT f32 vs torch max
// |err| 2.6e-6, f16 vs f32 72 dB).
//
// Tiling: include/xllama/upscale/tile_plan.h (sliding windows, a pixel has one source tile). Image
// decode and encode: Windows.Graphics.Imaging (EXIF orientation applied on decode; JPEG at quality
// 0.95 or PNG; the optional max_side shrink is the encoder's Fant transform).
//
// The session stays loaded between requests (a queue of catalogue photos pays the DirectML graph
// build once) and is dropped by release_upscale(), a failed run or the next GGUF load on the GPU
// (the reclaim hook, platform.h). The session shares the process's one D3D12 device with the
// chat, so it is only built inside the memory gate (tile_plan.h budget_refusal): a DirectML init
// that did not fit removed the device and stalled the chat (xllama-net 1.7.12.106, 25/11).

#include "inference-bridge.h"

#ifdef XLLAMA_UWP

// clang-format off
    #include "pch.h"
    #include <windows.h>
    #include <eh.h>
    #include <onnxruntime_cxx_api.h>
    #include <dml_provider_factory.h>
    #include <winrt/Windows.Foundation.h>
    #include <winrt/Windows.Foundation.Collections.h>
    #include <winrt/Windows.Graphics.Imaging.h>
    #include <winrt/Windows.Storage.Streams.h>
// clang-format on

    #include "xllama/diffusion/half.h"
    #include "xllama/path_utils.h"
    #include "xllama/platform.h"
    #include "xllama/upscale/tile_plan.h"
    #include "xllama/utf8_utils.h"

    #include <algorithm>
    #include <chrono>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <memory>
    #include <mutex>
    #include <stdexcept>
    #include <string>
    #include <vector>

namespace xllama::bridge {

namespace {

using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Storage::Streams;

// ORT's own warnings and errors into xllama.log: a failed DirectML init says where it failed
// (.107 answered a bare "bad allocation", 25/11). The first OrtEnv of the process keeps its logger.
void ORT_API_CALL ort_log(void*, OrtLoggingLevel, const char* category, const char*,
                          const char* where, const char* message) {
    log_output(std::string("[ort] ") + (category ? category : "") + " " + (where ? where : "") +
               ": " + (message ? message : "") + "\n");
}

struct Engine {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "upscale", &ort_log, nullptr};
    std::unique_ptr<Ort::Session> session;
    int tile = 0;
    bool f16 = true;
};

std::mutex g_engine_mtx; // guards g_engine; the API route serialises requests before this
std::unique_ptr<Engine> g_engine;

// Registered once a session is up: a GGUF load on the GPU waits for a running upscale, then sizes
// itself against a device the upscale no longer holds.
void reclaim_for_gguf() noexcept {
    try {
        std::lock_guard<std::mutex> lk(g_engine_mtx);
        if (g_engine) {
            g_engine.reset();
            log_output("[xllama] upscale: session released for a GGUF load\n");
        }
    } catch (...) {
    }
}

std::size_t need_mb(const char* env, std::size_t fallback) {
    return upscale::need_mb_from(std::getenv(env), fallback);
}

// A DML session as in diffuse.cpp: sequential, no memory pattern, EXTENDED optimisation. `phase`
// names the step a failure came from (the error text and the log carry it).
Engine& engine(const char*& phase) {
    if (g_engine && g_engine->session)
        return *g_engine;
    phase = "session-env";
    auto e = std::make_unique<Engine>();
    const std::string path = resolve_local_path("models\\realesrgan-x4plus\\model.onnx");
    if (GetFileAttributesW(utf8_to_wstring(path).c_str()) == INVALID_FILE_ATTRIBUTES)
        throw std::runtime_error("models\\realesrgan-x4plus\\model.onnx missing");
    Ort::SessionOptions so;
    so.SetExecutionMode(ORT_SEQUENTIAL);
    so.DisableMemPattern();
    so.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    phase = "session-dml-ep"; // creates (or joins) the process's D3D12 device and a DML device
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, /*device_id=*/0));
    phase = "session-load"; // parse, partition, DML graph fusion and compile, weights upload
    e->session = std::make_unique<Ort::Session>(e->env, utf8_to_wstring(path).c_str(), so);
    phase = "session-shape";
    auto info = e->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != shape[3] ||
        shape[2] < 32 || shape[2] > 512)
        throw std::runtime_error("model input is not a static [1,3,T,T] with 32 <= T <= 512");
    e->tile = static_cast<int>(shape[2]);
    e->f16 = info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    log_output("[xllama] upscale: session up, tile " + std::to_string(e->tile) +
               (e->f16 ? " fp16\n" : " fp32\n"));
    g_engine = std::move(e);
    return *g_engine;
}

IBuffer to_buffer(const std::string& bytes) {
    Buffer buf(static_cast<uint32_t>(bytes.size()));
    memcpy(buf.data(), bytes.data(), bytes.size());
    buf.Length(static_cast<uint32_t>(bytes.size()));
    return buf;
}

std::string read_stream(InMemoryRandomAccessStream const& s) {
    const uint64_t n = s.Size();
    s.Seek(0);
    Buffer buf(static_cast<uint32_t>(n));
    auto got = s.ReadAsync(buf, static_cast<uint32_t>(n), InputStreamOptions::None).get();
    return std::string(reinterpret_cast<const char*>(got.data()), got.Length());
}

} // namespace

void release_upscale() {
    std::lock_guard<std::mutex> lk(g_engine_mtx);
    g_engine.reset();
}

UpscaleResult run_upscale(const std::string& image, const std::string& format, int max_side,
                          int pad) {
    _set_se_translator([](unsigned int code, EXCEPTION_POINTERS*) {
        char b[48];
        snprintf(b, sizeof(b), "SEH 0x%08X", code);
        throw std::runtime_error(b);
    });
    UpscaleResult r;
    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(g_engine_mtx);
    const char* phase = "decode";
    try {
        // decode, upright, RGBA8
        InMemoryRandomAccessStream in;
        in.WriteAsync(to_buffer(image)).get();
        in.Seek(0);
        BitmapDecoder dec = BitmapDecoder::CreateAsync(in).get();
        const int w = static_cast<int>(dec.OrientedPixelWidth()),
                  h = static_cast<int>(dec.OrientedPixelHeight());
        r.in_w = w;
        r.in_h = h;
        if (w <= 0 || h <= 0 || static_cast<int64_t>(w) * h > kUpscaleMaxInPixels) {
            r.status = 413;
            r.error = "input " + std::to_string(w) + "x" + std::to_string(h) + " is over " +
                      std::to_string(kUpscaleMaxInPixels) + " pixels";
            return r;
        }
        PixelDataProvider px =
            dec.GetPixelDataAsync(BitmapPixelFormat::Rgba8, BitmapAlphaMode::Ignore,
                                  BitmapTransform(), ExifOrientationMode::RespectExifOrientation,
                                  ColorManagementMode::DoNotColorManage)
                .get();
        const winrt::com_array<uint8_t> src = px.DetachPixelData();
        if (src.size() != static_cast<size_t>(w) * h * 4)
            throw std::runtime_error("decoder returned " + std::to_string(src.size()) + " bytes");

        // the gate: the DXGI figures only through an adapter a GGUF load already opened (no
        // adapter = no GGUF on the GPU, nothing to protect); the GPU judged before a build, the
        // app cap (up to ~350 MB of buffers for a 1.5 MP input) on every request
        phase = "gate";
        const bool build = !(g_engine && g_engine->session);
        const GpuMemInfo g0 = gpu_mem_probe_open() ? gpu_mem_info() : GpuMemInfo{};
        if (g0.available) {
            r.dxgi_before_mb = g0.current_mb;
            r.dxgi_budget_mb = g0.budget_mb;
        }
        const AppMemInfo a0 = app_mem_info();
        const std::string refusal = upscale::budget_refusal(
            {g0.current_mb, g0.budget_mb, g0.available},
            build ? need_mb("XLLAMA_UPSCALE_GPU_NEED_MB", 512) : 0,
            {a0.usage_mb, a0.limit_mb, a0.available}, need_mb("XLLAMA_UPSCALE_APP_NEED_MB", 384));
        if (!refusal.empty()) {
            r.status = 503;
            r.error = refusal;
            log_output("[xllama] upscale: 503 " + refusal + "\n");
            return r;
        }
        if (build)
            log_output(std::string("[xllama] upscale: building the session, dxgi ") +
                       (g0.available ? std::to_string(g0.current_mb) + "/" +
                                           std::to_string(g0.budget_mb) + " MB"
                                     : "unread (no GGUF on the GPU)") +
                       ", app " + std::to_string(a0.usage_mb) + "/" + std::to_string(a0.limit_mb) +
                       " MB\n");
        Engine& e = engine(phase);
        r.session_built = build;
        if (build)
            set_gpu_reclaim_hook(&reclaim_for_gguf);
        phase = "run";
        const int T = e.tile;
        pad = std::clamp(pad, 0, T / 4);
        const auto py = upscale::plan_axis(h, T, pad), pxs = upscale::plan_axis(w, T, pad);
        const int ow = upscale::kScale * w, oh = upscale::kScale * h;
        std::vector<uint8_t> out(static_cast<size_t>(ow) * oh * 4);

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::AllocatorWithDefaultOptions alloc;
        const auto in_name = e.session->GetInputNameAllocated(0, alloc);
        const auto out_name = e.session->GetOutputNameAllocated(0, alloc);
        const char* in_names[] = {in_name.get()};
        const char* out_names[] = {out_name.get()};
        const int64_t shape[] = {1, 3, T, T};
        std::vector<uint16_t> lr16(static_cast<size_t>(3) * T * T);
        std::vector<float> lr32;
        const size_t oplane = static_cast<size_t>(upscale::kScale * T) * (upscale::kScale * T);
        const int OT = upscale::kScale * T;
        for (const auto& ty : py) {
            for (const auto& tx : pxs) {
                upscale::window_to_planar_f16(src.data(), w, h, static_cast<size_t>(w) * 4,
                                              tx.window, ty.window, T, lr16.data());
                Ort::Value v{nullptr};
                if (e.f16) {
                    v = Ort::Value::CreateTensor(mem, lr16.data(), lr16.size() * sizeof(uint16_t),
                                                 shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
                } else {
                    lr32.resize(lr16.size());
                    for (size_t i = 0; i < lr16.size(); ++i)
                        lr32[i] = diffusion::half_to_float(lr16[i]);
                    v = Ort::Value::CreateTensor<float>(mem, lr32.data(), lr32.size(), shape, 4);
                }
                auto res = e.session->Run(Ort::RunOptions{nullptr}, in_names, &v, 1, out_names, 1);
                auto oinfo = res[0].GetTensorTypeAndShapeInfo();
                if (oinfo.GetElementCount() != 3 * oplane)
                    throw std::runtime_error("model output is not [1,3,4T,4T]");
                if (oinfo.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                    const auto* o =
                        reinterpret_cast<const uint16_t*>(res[0].GetTensorData<Ort::Float16_t>());
                    upscale::core_to_rgba(
                        [&](int c, int y, int x) {
                            return diffusion::half_to_float(
                                o[c * oplane + static_cast<size_t>(y) * OT + x]);
                        },
                        ty, tx, out.data(), static_cast<size_t>(ow) * 4);
                } else {
                    const float* o = res[0].GetTensorData<float>();
                    upscale::core_to_rgba(
                        [&](int c, int y, int x) {
                            return o[c * oplane + static_cast<size_t>(y) * OT + x];
                        },
                        ty, tx, out.data(), static_cast<size_t>(ow) * 4);
                }
                ++r.tiles;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        if (g0.available) {
            r.dxgi_after_mb = gpu_mem_info().current_mb;
            if (build)
                log_output("[xllama] upscale: session built, dxgi " +
                           std::to_string(r.dxgi_before_mb) + " -> " +
                           std::to_string(r.dxgi_after_mb) + " / " +
                           std::to_string(r.dxgi_budget_mb) + " MB\n");
        }

        // encode
        phase = "encode";
        InMemoryRandomAccessStream os;
        const bool png = format == "png";
        BitmapEncoder enc{nullptr};
        if (png) {
            enc = BitmapEncoder::CreateAsync(BitmapEncoder::PngEncoderId(), os).get();
        } else {
            BitmapPropertySet props;
            props.Insert(L"ImageQuality",
                         BitmapTypedValue(winrt::box_value(0.95f),
                                          winrt::Windows::Foundation::PropertyType::Single));
            enc = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), os, props).get();
        }
        enc.SetPixelData(BitmapPixelFormat::Rgba8, BitmapAlphaMode::Ignore,
                         static_cast<uint32_t>(ow), static_cast<uint32_t>(oh), 96.0, 96.0, out);
        std::vector<uint8_t>().swap(
            out); // the encoder holds its own copy; up to 96 MB back to the heap
        r.out_w = ow;
        r.out_h = oh;
        if (max_side > 0 && std::max(ow, oh) > max_side) {
            const double k = static_cast<double>(max_side) / std::max(ow, oh);
            r.out_w = std::max(1, static_cast<int>(ow * k + 0.5));
            r.out_h = std::max(1, static_cast<int>(oh * k + 0.5));
            enc.BitmapTransform().InterpolationMode(BitmapInterpolationMode::Fant);
            enc.BitmapTransform().ScaledWidth(static_cast<uint32_t>(r.out_w));
            enc.BitmapTransform().ScaledHeight(static_cast<uint32_t>(r.out_h));
        }
        enc.FlushAsync().get();
        r.bytes = read_stream(os);
        r.tile = T;
        r.pad = pad;
        r.gpu_seconds = std::chrono::duration<double>(t1 - t0).count();
        r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        r.status = 200;
    } catch (winrt::hresult_error const& ex) {
        r.status = 500;
        char b[64];
        snprintf(b, sizeof(b), "hresult 0x%08X", static_cast<unsigned>(ex.code().value));
        r.error = std::string(phase) + ": " + b + " " + winrt::to_string(ex.message());
        g_engine.reset();
    } catch (const Ort::Exception& ex) {
        r.status = 500;
        r.error = std::string(phase) + ": ort code " +
                  std::to_string(static_cast<int>(ex.GetOrtErrorCode())) + " " + ex.what();
        g_engine.reset();
    } catch (const std::bad_alloc& ex) {
        // thrown outside ORT's C API (that one arrives as an Ort::Exception above): ours or a
        // C++/WinRT call answering E_OUTOFMEMORY
        r.status = 500;
        r.error = std::string(phase) + ": std::bad_alloc " + ex.what();
        g_engine.reset();
    } catch (const std::exception& ex) {
        r.status = 500;
        r.error = std::string(phase) + ": " + ex.what();
        g_engine.reset(); // a DML device removal or OOM must not stick to the next request
    }
    if (r.status != 200 && r.status != 503) { // the gate logged its own 503
        const AppMemInfo a = app_mem_info();
        const GpuMemInfo g = gpu_mem_probe_open() ? gpu_mem_info() : GpuMemInfo{};
        log_output("[xllama] upscale: " + std::to_string(r.status) + " " + r.error + " (dxgi " +
                   (g.available ? std::to_string(g.current_mb) + "/" + std::to_string(g.budget_mb)
                                : std::string("unread")) +
                   " MB, app " + std::to_string(a.usage_mb) + "/" + std::to_string(a.limit_mb) +
                   " MB)\n");
    }
    return r;
}

} // namespace xllama::bridge

#else

namespace xllama::bridge {
void release_upscale() {}
UpscaleResult run_upscale(const std::string&, const std::string&, int, int) {
    UpscaleResult r;
    r.status = 501;
    r.error = "upscale needs the UWP build";
    return r;
}
} // namespace xllama::bridge

#endif
