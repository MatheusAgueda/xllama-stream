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
offload the rest of each layer. With `n_gpu_layers > 0` the cache offers itself
to each GPU device of the model with the slots in that device's host buft
(`llama_expert_cache_use_host_of`): the device's `supports_buft` sees the type
with its own `device` and the name `EXPERT_CACHE`, and claiming it means its
kernels read the slots in place (unified memory); the buffers keep the host
buft's iface and context. No device claims it today, and an allocation the host
buft refuses falls back to CPU memory. The load line says where the slots are. `llama_model_params::expert_swap_ratio`
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
Pre-gating prefetch (env `LLAMA_EXPERT_PREFETCH` = k, 0 = off, the default, max
64; `qwen35moe` and `llama` graphs): from layer 1 on, the layer's
own router is applied to its input residual through the pre-FFN RMS norm right
after the attention norm (`build_moe_pregate`, on the GPU split with the
projections), and a fourth op after the attention projections
(`build_moe_prefetch`) pins the predicted experts already resident and queues
the missing ones, best first, to four background I/O threads, so their reads
overlap the attention. With split reads on (`LLAMA_EXPERT_SPLIT`, below) a job
is cut into two 4 KiB-aligned parts per tensor, queued job-major: the threads
read the most likely expert together before starting the next one, instead of
one thread reading its three tensors one after the other; with split off each
job is one thread's whole reads. Every queue entry carries its job's
generation, so an entry left from a cancelled round never reads into the
job's next use. The plan settles the round first: unstarted reads are
cancelled (the slot gets its old expert back, its data was never overwritten),
running ones are waited for. Routing is untouched, so the output stays exact.
With a swap ratio the prediction is the swapped routing: the op ranks each
token by softmax, runs the same swap as the routing op against the current
residency (`ec_swap_missing`), pins the residents that would stand in and
queues only the experts still missing among the first k ranks. Before that it
queued every missing expert of the top k and evicted the stand-ins the routing
op would have used.
Offline on the 35B's routing trace (32 slots): 63.0% hits without, 90.4% with
k = 8. Measured on the bench runner (run 42643773195, 32 slots, 2 threads,
reads the bottleneck) k = 8 lifted the hits to 89.8% but read 25% more and
decoded slower (1.54 → 1.16 tok/s exact, 2.62 → 1.35 with swap 0.7), hence off
by default; worth an A/B where compute dominates the reads. The stats line adds
prefetch reads, how many the plan used, dropped reads and the time the plan
waited; `llama_model` stops the I/O threads before freeing the buffers.
Expert-major ubatches (env `LLAMA_EXPERT_MAJOR` = the largest one, 0 = off, the
default; `n_ubatch` is then capped at the larger of the two): a ubatch with
more tokens than `N / n_expert_used` (a prompt) does not go through the plan
and load. Each cached expert tensor's `mul_mat_id` becomes two custom ops
(`llama_expert_cache_build_mm_id`): the activations quantized to the expert
type's CPU `vec_dot_type` (as ggml's `mul_mat_id` does) with the pairs grouped
by expert, then the threads take the experts one at a time, busiest first, and
compute each for all its tokens from its slot if resident, else from one read:
one read per expert, layer and tensor however many tokens use it, so a long
prompt reads at most the model's experts once per ubatch instead of once per
4-token ubatch. The first expert tensor of the layer gives the missing experts
with the most tokens the least recently used slots (the ubatch's resident ones
pinned): their reads land there and they stay resident for the decode that
follows; the rest go to a per-thread staging buffer. Exact up to the summation
order; no prefetch or swap on those ubatches; needs the CPU backend linked in
(not `GGML_BACKEND_DL`) and a CPU dot product for every expert type, else off
with a warning. A graph stopped by its abort callback between the plan and the
load, or in the middle of an expert-major round, leaves no slot pointing at
data that never arrived: the next op of the layer empties those slots. The
stats line adds expert-major ubatches, reads, MiB, resident hits, experts kept
in slots and the time the matmuls took.
Reads are unbuffered (env `LLAMA_EXPERT_DIRECT`, default 1; `CreateFile2` with
`FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS`, `O_DIRECT` on Linux): no
copy through the OS file cache, no read-ahead. A read whose file span,
size and destination are 4 KiB-aligned lands in the slot itself; any other
read takes the aligned span around it into a per-thread staging buffer and
copies it out. If a read into the slot fails, that thread switches to staging;
if the open or a staged read fails (macOS, tmpfs), it switches to buffered
reads (`"rbR"` on Windows). A decode ubatch misses about one expert per layer,
which is 3 reads for 6 threads, so the load op splits each read into 4 KiB-aligned
parts of at least 128 KiB until every thread has one in flight
(`LLAMA_EXPERT_SPLIT`, default 1). On the console: 1.34 GB/s with buffered
whole reads, vs 2.13 GB/s for unbuffered sequential reads in `diskbw`. The
stats line ends with the direct, staged and buffered read counts, plus
`split` when splitting is on.
`tests/test_expert_cache.cpp` decodes a small MoE with 2 and 3 of its 4 experts
resident and checks the tokens and logits against the same model fully
resident, with prefetch on (it must read ahead somewhere) and off, that
buffered whole reads give the same tokens and logits bit for bit as the
unbuffered split ones (which must happen on Linux), that prefetch with whole
jobs gives the same tokens as with parts and logits within 1e-4 of the fully
resident model, that a
swap ratio of 0.5 reads less, and that expert-major prompts (with and without
experts resident from earlier tokens, and with every slot free so the decode
starts from the experts the prompt kept) give the same tokens and logits
(`.github/workflows/expert-cache-host.yml`); its opt-in "expert-cache bench"
measures speed, hit rate, top-1 agreement and KL per `ratio[:k[:em]]` config
on the 35B and keeps the routing trace (`.github/workflows/expert-cache-bench.yml`).

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

`0006-qwen35-mtp-tensors-optional.patch`: Qwen3.5 GGUFs declare
`nextn_predict_layers = 1`, so `block_count` counts the multi-token-prediction
block (33 for the 9B) and the loader asks for `blk.32.*`. Re-quants that drop
that block fail with `check_tensor_dims: tensor 'blk.32.attn_norm.weight' not
found`. Measured 2026-11-09 on mradermacher's
DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic i1-IQ3_M, which has 427 tensors for
blk.0–31 only. Without `load_mtp` the block is skipped anyway (`TENSOR_SKIP`), so
the patch also marks it `TENSOR_NOT_REQUIRED` in `qwen35` and `qwen35moe`. A
missing MTP tensor then loads as null, the same result a skipped one gives. With
MTP on, the tensors are still required.

`0007-qwen35moe-shared-expert-first.patch` (applies after 0004): `qwen35moe`
built the shared expert after the routed experts, so its three matmuls sat after
the routed experts' CPU tail (weights, sums) and took two D3D12 submissions of
their own per layer. Measured on 2026-11-09 with the 35B: 80 of ~203 submissions
per decoded token were the shared expert (`ffn_gate .. ffn_up` and `ffn_shexp`),
~8 ms of a 108 ms token. The patch builds the shared FFN first and expands it, so
its matmuls follow the output projection and the post-attention norm in node
order and join that submission. Only the order changes; the results stay the
same.

## Applying

```bash
./scripts/apply-uwp-patches.sh   # idempotent; used by the llamacpp CI variant
```

## Rebasing after a submodule bump

The patch is a plain `git diff`. If it no longer applies, redo the three guards
by hand (they are one-liners plus a small `#if` block — see the patch body) and
regenerate with `git -C llama.cpp diff > patches/0001-uwp-appcontainer-guards.patch`.
