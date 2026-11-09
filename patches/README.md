# patches/

UWP/AppContainer patches against the pinned `llama.cpp` submodule, applied to
both build variants that compile ggml (`uwp/ggml-uwp.vcxproj`): the shipping
`unified` variant (`XLLAMA_USE_ORT=1` + `XLLAMA_USE_LLAMA=1`) and the bench-only
`llamacpp` variant. CI runs `scripts/apply-uwp-patches.sh` for both, which
applies every `patches/0*-*.patch` in order and is idempotent per patch.

## Active patches

| File                                        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0001-uwp-appcontainer-guards.patch`        | Guard Win32 desktop-only APIs across **5 files** via **`WINAPI_FAMILY_PARTITION(DESKTOP)` only** (VS + uwp-crossbuild both set `WINAPI_FAMILY=APP` for UWP). Top-level `#if` lines in `common.cpp` and `llama-mmap.cpp` test `LLAMA_WIN32_APP_PARTITION`, defined only under `_WIN32 && WINAPI_FAMILY`: the partition macro does not exist off Windows, and GCC rejects it in a portable `#if`. Stubs registry CPU-name (`ggml-cpu.cpp`), `SetThreadAffinityMask` / power throttling (`ggml-cpu.c`), packaged `LoadPackagedLibrary` (`ggml-backend-dl.cpp`), desktop mmap/mlock/prefetch (`llama-mmap.cpp`, including `PrefetchVirtualMemory` / `WIN32_MEMORY_RANGE_ENTRY`), and `GetActiveProcessorCount` in `common/common.cpp` (falls back to `hardware_concurrency`; `llama-common-uwp.vcxproj`, ADR 0004). No dual `XLLAMA_UWP` workaround — the family partition is the contract. Always run `./scripts/apply-uwp-patches.sh` before a Linux UWP build; gate the PE with uwp-crossbuild’s `pe-import-audit` (on PATH after install; use uwp-crossbuild ≥ 0.5.1 — the launch floor). |
| `onnxruntime-genai-2280-dml-fallback.patch` | ORT GenAI `CreateDmlObjects`: fall back to system D3D12 when Agility `CreateDevice` fails with `887A0036` (XAML + DML). **Upstream MERGED** [microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280) (2026-07-13 on `main`); **not** in NuGet 0.14.1. Shipping CI installs the pinned DLL from `vendor-dlls-v1` (hash in `vendor/onnxruntime-genai-patched/SHA256SUMS`); rebuild via `scripts/vendor-genai-dml-patch.ps1 -Build` / `build-uwp-patched.yml`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `onnxruntime-extdata-appcontainer.patch`    | ORT core, two AppContainer external-data fixes (`docs/fp16-extdata-runbook.md`, console-validated 2026-07-15, **shipping since 1.1.8.0**): **(1)** `tensorprotoutils.cc` `ValidateExternalDataPath` — guard `weakly_canonical()` (related upstream on ORT `main`: [#28509](https://github.com/microsoft/onnxruntime/pull/28509), **not** in NuGet 1.24.4); **(2)** `env.cc` `ReadFileIntoBuffer` — 1 GB→16 MB chunk (`errcode 1450`, **still open on ORT main**). Shipping CI installs the pinned DLL from `vendor-dlls-v1` (hash in `vendor/onnxruntime-patched/SHA256SUMS`); rebuild via `scripts/vendor-ort-extdata-patch.ps1 -Build` / `build-uwp-ort-patched.yml`.                                                                                                                                                                                                                                                                                                                                                                                                                   |

Rebased against `d81235049` on 2026-10-12; the `common.cpp` guard was added on 2026-10-14. Patch 0002 was retired because
upstream `ggml/src/ggml-cpu/ops.h` now uses fixed cache-line sizes and no longer
references `std::hardware_destructive_interference_size`; there is no matching
code left to patch. The CPU backend remains statically linked on UWP.

`0002-uwp-appcontainer-mmap.patch` (applies after 0001): an AppContainer
`llama_mmap` built on `CreateFileMappingFromApp` / `MapViewOfFileFromApp`, never
prefetched, and `llama_mmap::SUPPORTED` back to true there. It exists for the
catalogue's `"stream": true` GGUF entries (`apply_gguf_residency`): the weights
stay file-backed with no repack copy, so a MoE larger than the title's RAM budget
pages its experts in from storage. Every other model is still read into the heap
— on UWP `load_mode` is set to `NONE` explicitly, since `AUTO` would now map it.

`0003-loader-chunked-upload.patch` (applies after 0002): `load_all_data`'s
synchronous upload path (any non-host weight buffer whose device does not take
the async pinned-buffer route, e.g. `D3D12_Weights`) allocated a staging vector
the size of the whole tensor. A 248K-vocab Q6_K `output.weight` is 796 MiB, which
on top of the weights already placed overflows the title's single ~5 GB pool
(heap and D3D12 share it; measured 2026-11-09, `bad allocation` with
`gguf_gpu_layers.txt=99`). It now reads and uploads 64 MiB at a time through
`ggml_backend_tensor_set(..., offset, len)`. `--check-tensors` keeps the
whole-tensor path, since `ggml_validate_row_data` needs the full buffer.

`0004-expert-cache.patch` (applies after 0003): a routed-expert cache for MoE
models larger than the title's memory, `llama_model_params::n_expert_slots`
(`LLAMA_HAS_EXPERT_SLOTS`; catalogue `"expert_slots": N`,
`apply_gguf_residency`). The `"stream": true` mapping from 0002 does not get
there: the AppContainer caps the mapped total per process at 1.0–1.5 GiB
(measured 2026-11-09), below the ~1.6 GB of non-expert weights of a 35B-A3B.
With the cache, each routed `*_exps.weight` tensor is created with N experts
instead of the GGUF's count (a CPU buffer type, `EXPERT_CACHE`), the loader
records where the tensor lives in the file instead of reading it, and
`build_moe_ffn` adds two custom ops per layer before the expert matmuls: a
one-thread plan that maps the router's ids to slots (LRU per layer) and lists
the misses, and a load that reads the missed experts with positional reads
(`ReadFile` + `OVERLAPPED` on Windows, `pread` elsewhere) spread over the
compute threads. The `mul_mat_id` ops take the slot ids; router weights,
biases and per-expert scales keep the real ids. Every other weight is read into
the heap as before (`load_mode` `NONE`). The expert matmuls run on the CPU (the
buffer type is a host one no GPU backend claims), so `n_gpu_layers` may still
offload the rest of each layer. `llama_model_params::expert_swap_ratio`
(catalogue `"expert_swap"`, 0 = off) adds cache-aware routing, a third op that
replaces the top-k: a resident expert outside a token's top-k takes the place
of a missing one inside it when its router score is at least that fraction of
the missing one's (weakest missing first, strongest resident first); the
weights are still the router's for the ids chosen. Fewer reads, output no longer
exact; not for logit-weighted gating (`SOFTMAX_WEIGHT`). Limits: one context
per model at a time, `n_ubatch` capped at `N / n_expert_used`, not for
`LLM_ARCH_GRANITE_SWITCH` (it calls `ggml_mul_mat_id` outside
`build_moe_ffn`). Hits, misses, swaps, MiB read and the time the reads took are
logged every 64 ubatches and when the model is freed.
`tests/test_expert_cache.cpp` decodes a small MoE with 2 and 3 of its 4 experts
resident and checks the tokens and logits against the same model fully
resident, and that a swap ratio of 0.5 reads less
(`.github/workflows/expert-cache-host.yml`); its opt-in "expert-cache bench"
measures speed, hit rate, top-1 agreement and KL per swap ratio on the 35B and
keeps the routing trace (`.github/workflows/expert-cache-bench.yml`).

`0005-graph-order-shared-input-projections.patch` (applies after 0004): the D3D12 backend runs only the matmuls, so every
switch between it and the CPU is one submit plus a fence wait (~90 µs measured on
Series X). The graph is built depth first from the outputs, which puts the CPU
ops of the q projection between it and the k projection, and the same for the
four linear-attention projections of Qwen3.5 (qkv, z, beta, alpha). Measured on
2026-11-09 with Qwen3.5 9B fully offloaded: 207 graph_compute calls for each token,
~19 ms of a 55 ms token. The patch calls `ggml_build_forward_expand` on each
projection right after it is built (generic separate-QKV path in
`llm_graph_context::build_qkv`, plus `qwen35` and `qwen35moe`). Those matmuls then
sit next to each other in the graph and land in one split, which also quantizes
their shared input once. Only the order changes; the graph and its results stay
the same.

## Applying

```bash
./scripts/apply-uwp-patches.sh   # idempotent; used by the llamacpp CI variant
```

## Rebasing after a submodule bump

The patch is a plain `git diff`. If it no longer applies, redo the three guards
by hand (they are one-liners plus a small `#if` block — see the patch body) and
regenerate with `git -C llama.cpp diff > patches/0001-uwp-appcontainer-guards.patch`.
