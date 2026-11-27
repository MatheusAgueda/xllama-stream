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
On the console (35B, D3D12, 256-token decode) k = 2 lost too: 94.0 against
85.6 ms/token, with 2468 of 2853 prefetch jobs dropped before a thread reached
them and about 78 more dispatches per token. The window from the op to the plan
is the attention alone, and the op, a CPU custom op between GPU nodes, costs
each layer an extra GPU→CPU→GPU switch. Prefetch ahead (env
`LLAMA_EXPERT_PREFETCH_AHEAD=1`, with k > 0) moves both: the router of layer
il+1, through il+1's pre-FFN norm, runs on layer il's post-attention residual,
and il+1's fourth op takes il's load op as an
extra source it does not read, expanded right after it in `build_moe_ffn`. It
lands in the CPU split that already holds il's plan and load, after them: no
switch of its own, no contention with il's reads, and its reads overlap il's
expert matmuls and il+1's attention. The prediction is one layer older, so less
exact; the stats line ends its prefetch part with `, ahead`.
The console's next A/B (build 1.7.12.26, 41 slots, swap 0.7) put the loss in
the waits: k = 1 ahead read 185 experts, 182 of them used, and its plans waited
1535 ms for them, about 8 ms each where one expert takes about 1 ms at the
disk's speed, while 1079 of the 1264 jobs had not started when their plan came.
The graph's six threads spin between ops (`poll` = 50) on the six cores the app
gets, so an I/O thread whose read completes waits for a core, likely up to a
scheduler quantum. Two changes, both on by default when k > 0: the I/O threads
run at `THREAD_PRIORITY_HIGHEST` on Windows (env `LLAMA_EXPERT_PREFETCH_PRIO`:
0 normal, 1 highest, 2 time critical; they sleep in their reads, so the
priority costs no compute), and a plan waiting for a read in flight takes the
job's parts no I/O thread has taken out of the queue and reads them itself on
its compute thread (env `LLAMA_EXPERT_PREFETCH_HELP=0`: wait only). The stats
line adds the reads waited for, the parts the plans read, and per read the time
from queued to its first part taken and from there to done, then `, help` and
`, prio N` when they apply.
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
A resident expert of an expert-major ubatch reaches the stage from its slot or
from the file (env `LLAMA_EXPERT_EM_RESIDENT`: `slot`, `file`, `auto` - the
default; also 0, 1, 2). Write-combined slots (`XLLAMA_D3D12_WC_MIB`) are read
by the CPU at about 1 GB/s on the console, slower than the disk: with WC40 a
1038-token prompt took 51.4 s against 40.4 s with write-back slots. `auto`
times every copy from a slot and every read from the file per thread and,
once both have moved 4 MiB, reads a resident expert from the file whenever the
file filled the stage faster; the file's totals are halved every ubatch so the
choice follows the disk's load (the misses share it), the slots' stay the
memory's own. Write-back slots copy far faster than the disk reads, so `auto`
keeps them. Either way the same bytes reach the stage: same results. The stats
line adds the resident experts read from the file, the mode, and the stage
fill rate per thread from slots and from the file (MB/s).
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
A thread's unbuffered read keeps one request in flight, so the disk sees as
many as there are threads reading (6; 1.68 GB/s in a 35B decode on the
console). On Windows, env `LLAMA_EXPERT_IO_QD` (1..64, default 1) opens the
handles with `FILE_FLAG_OVERLAPPED` and issues each read as up to that many
4 KiB-aligned requests of at least 32 KiB at once, then waits for all of them
(`GetOverlappedResult`); a request that fails cancels the others still in
flight, every one is waited for before the read returns (the buffer is the
caller's) and the thread falls back as above. Elsewhere the knob does nothing.
The stats line adds `, qd N` when it is on (Windows).
`tests/test_expert_cache.cpp` decodes a small MoE with 2 and 3 of its 4 experts
resident and checks the tokens and logits against the same model fully
resident, with prefetch on (it must read ahead somewhere) and off, that
buffered whole reads give the same tokens and logits bit for bit as the
unbuffered split ones (which must happen on Linux), and so do reads with 8
requests in flight (`LLAMA_EXPERT_IO_QD=8`, unbuffered and reported on Windows), that prefetch with whole
jobs gives the same tokens as with parts and logits within 1e-4 of the fully
resident model, that prefetch ahead does too (and must read ahead somewhere),
that plans which only wait for the I/O threads give the same tokens and read no
part themselves (with help on, never more parts than the reads they waited
for hold), that a swap ratio of 0.5 reads less, and that expert-major prompts (with and without
experts resident from earlier tokens, and with every slot free so the decode
starts from the experts the prompt kept) give the same tokens and logits,
bit for bit whether their resident experts come from the slots or the file
(`.github/workflows/expert-cache-host.yml`); its opt-in "expert-cache bench"
measures speed, hit rate, top-1 agreement and KL per `ratio[:k[a][:em]]` config
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

`0008-qwen35moe-shared-gate-with-router.patch` (applies after 0004–0007): in
`qwen35moe` the shared expert's gate (`ffn_gate_inp_shexp`, an F32 matmul to one
value per token, then a sigmoid) was built after the routed experts, so on the
D3D12 backend, which takes no F32 weights, each MoE layer had a CPU split between
the routed experts and the residual adds. The patch builds the gate first and
calls `ggml_build_forward_expand` on it before `build_moe_ffn`, so it runs in the
router's CPU split (the router is also F32). With the MoE combine and tail
kernels (`shaders/ggml_d3d12_moe.hlsl`) the routed experts, their combine and
the residual adds then form one GPU split (the shared FFN runs earlier, 0007).
Only the order changes; the graph and its results stay the same.

`0009-qwen35moe-router-logits-with-shared-gate.patch` (applies after 0008): the
D3D12 backend now runs F32 weights (`shaders/ggml_d3d12_mmv_f32.hlsl`), so the
router (`ffn_gate_inp`, 2048 x 256) and the shared expert's gate move to the GPU.
Built as before, the gate's sigmoid sat between the two matmuls and would have
split the GPU submission. The patch builds the router's logits right after the
gate's matmul and hands them to `build_moe_ffn` through `probs_in`, then the
sigmoid: the two matmuls join the shared expert's GPU split and the sigmoid joins
the router's CPU split (softmax, top-k). Only the order changes; the results stay
the same.

`0010-recurrent-state-on-device.patch` (applies after 0009): with
`LLAMA_RS_ON_DEVICE` set (and not `0`), `llama_memory_recurrent` puts each
layer's conv and SSM states in that layer's device buffer type even when
`offload_kqv` is off. xllama sets it whenever layers run on D3D12
(`apply_gguf_gpu_context`): the states then sit in D3D12_Host, which the
delta-net kernels (`shaders/ggml_d3d12_ssm.hlsl`) read and write in place and
the CPU reads as host memory, so a linear-attention layer's chain runs in one
GPU split instead of one per state access. Unset, the placement is upstream's.

`0011-sched-split-profile.patch` (independent of the others): with
`GGML_SCHED_PROFILE` set (and not `0`), `ggml_backend_sched_compute_splits`
times each split: its input copies, then its compute until the backend is idle
(`ggml_backend_synchronize`). The totals are keyed by backend and the split's
first and last node names, with the layer suffix dropped, so the same split of
every layer adds up to one line. `ggml_backend_sched_free` logs the 40 most
expensive kinds (`sched split: <count> x <ms> (copy <ms>) <backend> <first> ..
<last>`). It is for finding where the CPU time of a decoded token goes. The
synchronization costs async backends their overlap, so leave it unset for
timing runs. Unset, the scheduler is unchanged.

`0012-mtp-sidecar.patch` (applies after 0004, which adds `file_paths`): with
`XLLAMA_MTP_SIDECAR=<gguf>` set and `load_mtp` requested, `llama_model_loader`
opens that GGUF as one more file and adds its `blk.*` tensors the model lacks.
It is for re-quantized targets (heretic/abliterated IQ3_S) that dropped the
model's multi-token-prediction block: the sidecar is the original block from
`convert_hf_to_gguf.py --mtp`, quantized on its own, while the embeddings and
the LM head stay the target's. The sidecar's `general.architecture` must match
the model's; its `block_count` and `nextn_predict_layers` become kv overrides
unless the caller set them. Its routed experts go through the expert cache
like the trunk's (its path is in `file_paths`). Unset, the loader is unchanged.

`0013-sched-split-budget.patch`: `ggml_backend_sched_new` sizes its context
buffer for one split per graph node, 60 split-input tensors each. For the
26k-node budget of Qwen3.6-35B-A3B that is ~0.5 GiB per scheduler, committed
and almost never touched (the real graphs split 68-104 ways). On the Series X
the app's commit limit is ~5 GiB, so the second scheduler of an MTP context
failed its `malloc` and the NULL buffer reached `ggml_init` as
`GGML_ASSERT(ctx->mem_buffer != NULL)` (crash dump, 1.7.12.52).
`GGML_SCHED_MAX_SPLITS=<n>` bounds the split count that buffer and the four
backend-id arrays are sized for; a graph that needs more aborts with a message
that names the variable instead of overrunning the arrays. A failed allocation
now aborts in `sched_new` with the size. Separately, an MTP context of
Qwen3.5/3.6 gets a `graph_max_nodes` of `max(4096, 8*n_tokens)` instead of the
whole model's, because it builds only the nextn block. Unset, the scheduler
is unchanged; the MTP budget applies whenever an MTP context is created.

`0014-qwen35moe-shared-expert-after-router.patch` (applies after 0009): with
`XLLAMA_D3D12_SHEXP_ASYNC` set when the graph is built, Qwen3.5/3.6 MoE adds a
layer's shared expert after the router logits instead of before the shared
gate, so the GPU split ends with it. The D3D12 backend (same variable) then
waits only for the ops up to the logits and leaves the shared expert running
while the CPU split routes and reads experts. The CPU and D3D12 splits share
one compute buffer, so for one token the shared expert's intermediates are
marked graph outputs: the allocator never reuses their memory for the CPU
split that runs meanwhile (3 x n_ff_shexp floats per layer). Unset, the graph
is unchanged; set, only the node order (and, for one token, the allocation)
changes, not any value.

`0015-expert-cache-gpu-route.patch` (applies after 0004): two C entry points
for a backend that routes a decode token's experts on the GPU.
`llama_ec_route_query(node, &info)` says whether a graph node is the cache's
select, plan, load or prefetch op (by the op's function), with its layer,
`n_expert`, `n_slots`, swap ratio and tail, a pointer to the layer's
expert-to-slot map, and `plain` (no trace, deferred reads or prefetch).
`llama_ec_route_prepare(node)` settles the layer as the plan would before the
backend reads that map. The D3D12 backend (`XLLAMA_D3D12_GPUROUTE=1`) then
calls the real select, plan and load functions itself, in layer order, on the
scores its kernel computed, so the LRU/LRFU state, the misses and the reads
stay the cache's. Nothing calls them otherwise: the cache is unchanged.

`0016-lazy-tensors-by-name-span-mapping.patch` (applies after 0015): the
tensors named in `LLAMA_LAZY_TENSORS` (comma list) are lazy whatever the lazy
mode and size: they stay on the CPU as views of the file mapping instead of
heap copies. Without mmap (`load_mode` NONE) the mapping is used for nothing
else, so on UWP it maps only the span from the first to the last lazy range,
aligned down to the allocation granularity, and `addr()` stays the address of
file offset 0. A title cannot map a whole 52.9 GB GGUF (its mapped views stop
between 1.0 and 1.5 GiB per process); token_embd + output of
Qwen3.5-122B-A10B sit at the start of the file, 920 MiB. Unset, nothing
changes.

`0017-expert-cache-padded-direct.patch` (applies after 0016): `LLAMA_EXPERT_PAD=1`
lets experts at unaligned file offsets read in place. A GGUF aligns tensor data
to 32 bytes; every expert tensor of the Qwen3.5-122B-A10B GGUFs sits at an offset
that is not a multiple of 4 KiB, so each unbuffered read went through the
thread's staging buffer and a copy into the slot (console, 2026-11-12: 45018
reads staged, 0 direct; ~0.9-1.0 GB/s against ~2 GB/s for the 35B's in-place
decode reads). With the knob, a tensor whose expert size is a multiple of 4 KiB
gets slots 4 KiB apart beyond the expert's bytes (`nb[2]` = expert bytes + 4 KiB,
the stride the MUL_MAT_ID kernels already take), 12 KiB more in its allocation,
and its data moved when bound so that every slot sits at the file offset's
remainder: the aligned span around an expert, or around any aligned part of it,
lands in place and spills only into the gap (a part's spill is the same file
bytes its neighbouring parts write there). The expert-major path reads into a
stage placed the same way. Unset, nothing changes.

`0018-expert-cache-em-gpu.patch` (applies after 0017):
`LLAMA_EXPERT_EM_GPU=<experts>[:<pairs>[:<MiB>]]` runs the expert-major matmuls (prompt
ubatches larger than the slots take) as `MUL_MAT_ID` dispatches on the GPU of
the slots instead of the CPU's dot products (console, 2026-11-12: the 122B's
prefill was CPU-bound, ~69% of thread time in vec_dot, the GPU idle). The
context hands its backends to the cache (`llama_expert_cache_set_backend`); the
one of the slots' device is kept. Per expert tensor and ubatch, threads 1.. read
the round's kept experts into their slots and the missing ones into two staging
halves of `<experts>` experts each (typed EXPERT_CACHE, laid out as the slots,
padded ones too), while thread 0 dispatches: the resident experts in place,
then each staging chunk as soon as it is read; a reader reuses a half once the
dispatch of its previous chunk returned. Each dispatch gathers up to `<pairs>`
(default 2048) token-expert columns of the F32 activations, which the matmul
op then keeps as a source, and scatters the rows back; a dispatch the device
refuses or fails is computed on the CPU from the quantized activations. A
tensor type the device has no kernel for stays on the CPU. The staging and
column buffers are only allocated while the device still reports `<MiB>`
(default 128, max 2048) free beyond them; otherwise, and after a dispatch whose
backend throws (caught, not the app's end), expert-major stays on the CPU for
that load (console, 2026-11-12: at 14 slots the 2327-token prefill ended the
app with E_OUTOFMEMORY next to the KV checkpoints). Unset, nothing changes.

`0019-expert-cache-host-tier.patch` (applies after 0018):
`LLAMA_EXPERT_HOST=<MiB>[:<keep MB>]` adds a host tier of plain heap memory under
the slots of each layer, for the case where the device budget stops the slot count
before the process commit does. A layer's share of `<MiB>` holds up to
`n_expert - slots` whole experts, sized at the layer's first plan. It is
allocated only while `min(avail_phys, avail_page_file)` stays above `<keep>`
(default 512); the entry count is halved when an allocation fails. A slot's evicted
expert is saved into the tier, which takes a free entry, or else the least
recently used one that the ubatch does not use. A miss found in the tier is
memcpy'd into its slot instead of read from the file; when the victim also needs
saving, the two swap places. Slot plans, miss counts and the slot bytes stay
those of a run without the tier (host harness: 6 configurations, plans dropped
before their load included, byte-exact slots and entries; `test_expert_cache`
checks tokens, logits and misses against the plain run). Only bytes read from
the file shrink. Copies run on the load threads, and the disk reads start after a
barrier. The `expert cache (final)` line gains `; host tier X MiB (N entries), F
misses from it (P%, S swaps), V saved, B MiB copied in T ms`. Off with a
prefetch or deferred plans. Unset, nothing changes.

`0020-expert-cache-em-run.patch` (applies after 0019):
`LLAMA_EXPERT_EM_RUN=<n>` (2 or more; needs `LLAMA_EXPERT_EM_GPU`) reads the
expert-major GPU path's staged experts in spans of up to `<n>` consecutive ones. The
reason: on the 122B, `log_c98a` showed the prefill waiting on reads, 108.7 s of
145 s on thread 0, from one QD1 unbuffered read of ~1 MiB per expert at ~232 MB/s
per reader thread, while the same disk streams 2.13 GB/s in 8 MiB requests
(`diskbw`). The staged experts of each tensor whose expert size is a multiple of
4 KiB are sorted by id and packed at their file stride in the staging halves
(`ec_em_gpu_pos`), so the consecutive ones of a chunk are one span of the file.
`ec_em_runs` cuts spans where the ids stop being consecutive, at a chunk's end and
after `<n>` experts. `ec_read_run` reads the span's aligned middle in place and its
partial end blocks through the thread's staging buffer, so it writes no byte
outside the span's experts. The round's experts still go one by one into their
slots. The dispatch is unchanged: same experts, same pairs, only the order of
experts in a chunk differs. The `gpu …` part of the stats line gains `, S spans
of M experts`. Host harness `tests/harness/expert_cache_em_run.cpp` drives the real
queue, span builder and reader threads against a simulated thread 0 and checks
every staged expert and slot byte for byte, plus every other byte of the halves
and slots: 12096 cases with ASan and UBSan, OK buffered (macOS) and with O_DIRECT
(Linux aarch64, 64055 reads in place). A spilling read and a wrong stride both
fail it. Unset, nothing changes.

`0021-cpu-compute-plain.patch` (applies after 0020):
`LLAMA_CPU_COMPUTE_HOST=0` gives the CPU backend's compute buffer its own buffer
type instead of the first device's host buffer type. The reason: the D3D12
device hands out `D3D12_Host` as its host buffer type, so the CPU backend's
intermediates share the device's compute allocator and count against the DXGI
budget. On the 122B that buffer grows 0.266 MiB per ubatch token (1024: 272 MiB,
1536: 408 MiB, `log_c98a`), and at 1536 tokens the expert-major GPU buffers no
longer fit (139.1 MiB free against the 217.4 they need) and the prefill falls back
to the CPU (1151 tokens in 183 s against 87 s). With the variable at 0 the CPU's
part moves to process memory: commit is unchanged, DXGI is relieved. The cost is
the scheduler's copy of every device input that comes from a CPU tensor
(`host_cpy`, a memcpy into the mapped `D3D12_Host` buffer) and node placement:
without a shared buffer type, CPU nodes are no longer upgraded to the device.
The expert-major GPU path copies its activations into its own buffers and is not
affected. `sched_reserve` then prints a `CPU` compute buffer line beside the
`D3D12_Host` one. Unset, nothing changes.

`0022-expert-cache-em-swap.patch` (applies after 0021):
`LLAMA_EXPERT_EM_SWAP=<ratio>` (in (0, 1]; 0 or unset = off) gives expert-major
ubatches the cache-aware routing the slot path has. Until now `build_select`
made no select op for them, so a long prompt routed exactly and an expert-major
ubatch read every non-resident expert its tokens touched: on the 35B (53 slots,
EM on D3D12) 16 ubatches read 183,624 expert tensors (78.9 GB) and the prefill
spent 42.7 s waiting for them against 27.8 s of GPU work (`c106/log_e52g`). With
the variable, `ec_select` runs on those ubatches too, at `<ratio>` instead of the
swap ratio, without `LLAMA_EXPERT_SWAP_TAIL`, and always with the batch rule of
`LLAMA_EXPERT_SWAP_BATCH` (one read of an expert serves every token of an
expert-major ubatch, so an expert some token keeps counts as resident for the
others). The first pass that collects those reads goes token by token and already
counts the reads of the earlier tokens as resident, so a token can give a missing
expert up for one an earlier token reads. Each token then swaps at least what it
would against the slots alone (its missing experts are a subset, its candidates
a superset, and the weakest-first pairing only gets stronger partners), so the
reads never exceed those of a first pass against the slots alone. The slot path
keeps its own ratio, tail and batch setting. Its swaps count apart: the
`expert-major …` part of the stats line gains `, N swaps`, and the slot path's
`swaps` total stays the slot path's. A swap only brings an expert in that is
resident or read anyway, so the reads are a subset of the exact routing's. Host
harness `tests/harness/expert_cache_em_swap.cpp` runs the real `ec_select`
against an independent reference on random scores (ties and zeros included),
residency, slot counts and ubatch sizes on both sides of `max_ubatch`. It checks
every routed expert, both subset properties and which counter the swaps land in.
Unset, nothing changes.

`0023-expert-cache-pool.patch` (applies after 0022):
`LLAMA_EXPERT_POOL=1` (0 or unset = off) makes every layer's expert slot tensors
share ONE allocation instead of one set per layer. Decode walks the layers in
order and a layer reuses nothing from the one before it, so only the current
layer's experts must be resident; the rest occupy the same bytes in turn. In
`llama_expert_cache_shrink` each slot tensor is keyed on its role (the name after
`blk.N.`), type and layout; the first of each key is canonical and really
allocated, every later layer's same-role tensor becomes a `view_src` alias of it,
so ggml reserves one set of slots, not `n_layer` of them. On the 235B (`qwen3moe`,
94 layers, 8 experts used per token, 7.73 MiB per expert slice) the naive 8 slots
× 94 layers ≈ 5.8 GiB of slots alone is over the Series X Dev Mode commit ceiling,
so the model cannot load; with the pool it is one layer's worth, ≈ 62 MiB, and the
model fits with ~0.9 GiB to spare at `n_ctx` 2048. `ec_pool_claim(L)`, called at
the first residency-touching op of each layer (`ec_select`, `ec_plan`,
`ec_em_prep`, `llama_ec_route_prepare`), hands the shared slots to L and first
evicts the previous owner — settles its reads, drops its unloaded entries, ends
its expert-major pass and clears its slot map — so one layer is resident at a time
(0 cache hits by design: there is no cross-layer reuse anyway, 752 misses per
token). The router scores that first op needs depend on the previous layer's
output, so that layer's expert matmuls are finished and its slots are free before
the claim. `bind` clears the alias's `view_src` and re-points `t->data` into the
shared base with the layer's own `LLAMA_EXPERT_PAD` remainder, so the D3D12 graph
walker (`base_of`) treats each layer's slots as their own root and the padded
windows stay disjoint. The pool is incompatible with prefetch, deferred admission
and the host victim tier (they keep per-layer residents across tokens), so
enabling it forces `pf_k`, `pf_ahead`, `defer_b` and the host tier off regardless
of their own variables. Host harness `tests/harness/expert_cache_pool.cpp` drives
the real `ec_pool_claim` on constructed layers: the first claim takes ownership
and evicts nothing, claiming the current owner is a no-op, claiming a new layer
fully evicts the previous one and leaves the new one's residents untouched, and
with the pool off a claim does nothing. Unset, nothing changes.

`0024-cpu-fa-gqa.patch` (applies after 0023, touches only `ggml/src/ggml-cpu/ops.cpp`):
decode flash attention on the CPU (`offload_kqv=false`, f16 K/V, one query row,
`n_kv >= 512`) takes the split-KV path: each thread owns a KV chunk and walked it
once per QUERY head, so with grouped-query attention every K and V row was loaded
`n_head / n_head_kv` times (8 on Qwen3.5-35B-A3B, 16:2). At 10K context a thread's
chunk is ~1.7 MB per K/V head, past a Zen 2 core's 512 KB L2, so the re-reads went
to L3/DRAM and the per-token cost grew faster than linear in context.
`ggml_compute_forward_flash_attn_ext_f16_gqa_chunk` walks the chunk once per K/V
head with all its query heads together: each K/V row is loaded once and used by
the G heads while it sits in L1. Every head keeps its own `Q_q`, fp16 `VKQ`, `M`
and `S` and runs the f16 branch of `_one_chunk` in the same order (same mask skip,
scale, softcap, ALiBi slope, sinks on the first chunk), so the partials and the
output are the same bits; the reduce is untouched. Scratch is `thread_local` (G ×
(Q row + DV fp16), 8 KiB here), so the planned work size is unchanged. Taken when
K and V are f16, `1 < G <= 32`, `n_head == G * n_head_kv`; anything else (q8_0
KV, MHA, f32) keeps the old loop. `GGML_CPU_FA_GQA=0` restores the old path; it is
read on every call, so a manifest `env` honours it per load.

`GGML_CPU_FA_GQA=2` (opt-in; unset or 1 stays on the bit-exact chunk) takes
`ggml_compute_forward_flash_attn_ext_f16_gqa_fused_chunk` instead, built only with
AVX2+FMA+F16C (MSVC gets the last two from `ggml-cpu-impl.h`): blocks of 8 query
heads per K/V head, each K and V row converted from f16 once for the 8 heads, the
8 scores reduced by one `hadd` tree and exponentiated by one `ggml_v_expf`, and Q
and the V accumulators kept in f32. Same maths, not the same bits: the per-head loop
rounds Q to f16 and accumulates V in f16, so its error grows with the context.
Taken when DK and DV are multiples of 8, there is no logit softcap and the mask is
broadcast over heads (`ne[2] == 1`, so a row is masked for all heads or none);
otherwise mode 2 falls back to the bit-exact chunk.

Host harness `tests/harness/cpu_fa_gqa.cpp` (Linux gcc `-O3`, `GGML_AVX2/FMA/F16C`,
no AVX-512, Ryzen Z1 Extreme = Zen 4, 6 layers with their own KV, 2026-11-14): the
bit-exact chunk equals the per-head loop byte for byte in 16 of 16 cases (n_kv
512–16384, 1/2/3/6/30 threads, masked holes, ALiBi 8, softcap 30, sinks, an idle
thread). Error against an f64 reference, per-head loop → fused: max 4–5e-4 → 5–7e-8,
rms 1.4e-3 at 512 rising to 7.6e-3 at 16K → ≤1e-6. Time per 6 layers at 6 threads,
per-head / bit-exact / fused: 10K 52.6 / 25.7 / 17.2 ms, 16K 130.4 / 36.6 / 25.9 ms,
4K 5.2 / 4.9 / 3.4 ms. The handheld's clocks follow its power cap, so its thread
scaling is not the console's (1 → 2 threads was ×5–7 for every path).
Single-thread cost of one (head, row) pair of the per-head loop on L2-resident rows:
f16 dot 9.3 ns, f16 mad 9.7 ns, expf 2.6 ns.

`0025-expert-cache-em-wsum.patch` (applies after 0024; `src/llama-expert-cache.{cpp,h}`,
`src/llama-graph.cpp`): `LLAMA_EXPERT_EM_WSUM=1` (needs `LLAMA_EXPERT_MAJOR`)
fuses an expert-major ubatch's down projection with the routing weights and the
sum over each token's experts. Unfused, `build_moe_ffn` builds the products
`[n_embd, n_used, U]`, their weighted copy of the same size and `n_used - 1` adds;
on the 35B (n_embd 2048, 8 used) that is 2 × 32 MiB per 512 tokens in the CPU
compute buffer, which the app is charged for and which sets how many expert slots
fit at a given `n_ubatch`. `llama_expert_cache_build_mm_id_wsum` builds the
expert-major pair with `weights` as a third source and an F32 `[n_embd, U]`
output: each expert's rows for its tokens (GPU dispatch or CPU `mm_rows` into a
per-thread scratch) are multiplied by the token's weight and added into a double
accumulator `[n_embd, U]` under one of 16 mutex stripes by token; the last thread
out rounds it into `dst`. Double accumulation makes the sum independent of the
order the experts finish in, so the output is deterministic but not the bits of
the float adds (a different rounding of the same sum). Taken only when the graph
would weight after the FFN with no down scale/bias, no LoRA, no slot ids, not
Mistral 4, and the layer's `n_expert_used` is the graph's; otherwise the unfused
ops are built. The stats line's `expert-major …` part gains `, N down x weights
fused`. Host test: `expert cache: expert-major prompts…` runs every case again
with the variable and checks the same reads and kept experts, the same tokens and
logits within 1e-4 of the resident model. Unset, nothing changes.

`0026-kv-cache-on-device.patch` (independent of the others; `src/llama-kv-cache.cpp`):
with `LLAMA_KV_ON_DEVICE` on (its first character set and not `0`, like `XLLAMA_ATTN_GPU`: `1`, `10`, `yes` are on; unset, empty, `0`, `00` are off), `llama_kv_cache` puts each layer's K
and V cache in that layer's device buffer type even when `offload_kqv` is off,
as 0010 does for the recurrent states. xllama keeps `offload_kqv` off on D3D12
(`apply_gguf_gpu_context`), so without it the cache is CPU memory and a
FLASH_ATTN_EXT the d3d12 backend takes (`XLLAMA_FA_PREFILL`,
`XLLAMA_FA_DECODE_KV`) gets the K and V views copied into D3D12_Host by the
scheduler before every ubatch: 2 KiB a position per layer for Qwen3.6-35B-A3B
(D 256, 2 kv heads, f16), 26 MiB per layer per decoded token at 13K. With it the
cache is D3D12_Host, CPU-visible: the kernel reads it in place and the CPU's
attention, where the GPU does not take the node, reads it as host memory. Keep
`XLLAMA_D3D12_WC_MIB` above the cache's size (10 layers x n_ctx x 2 KiB: 80 MiB
at 4K, 320 MiB at 16K), or the buffer is write-combined and CPU reads of it
crawl. The memory moves from the CPU heap to D3D12_Host. Unset, the placement is
upstream's.

`0027-expert-cache-stats.patch` (applies after 0026; `include/llama.h`, `src/llama-model.cpp`,
`src/llama-expert-cache.{h,cpp}`; instruments only, no op changes): `llama_model_expert_stats(model,
&stats)` (`LLAMA_HAS_EXPERT_STATS`) copies the expert cache's cumulative counters - hits, misses, swaps,
expert-major swaps, bytes read, read microseconds (the last load op's time included: the stats line
adds it the same way), expert-major reads and bytes - into a caller struct whose first field is its own
`size` (a smaller struct from an older header is filled up to its size; fields are only appended).
`false` when the model has no cache. The counters are the existing atomics; nothing new is counted. The
bridge (`llama_gpu.h` `expert_stats_snapshot`, `Session::generate` and `run_inference_llama`) takes one
snapshot when a request starts and one when it ends, and the API reply's `timings` carries the
difference as `"expert":{"hits":..,"misses":..,"swaps":..,"em_swaps":..,"bytes":..,"read_us":..,
"em_reads":..,"em_bytes":..}`, only for a model with a cache (`include/xllama/api_timings.h`). The same
`timings` gains `"stamps":{"permit_ms":..,"first_ms":..,"last_ms":..}`, ms since the socket was
accepted: the compute permit and the session granted, the first generated token (measured in the
stream callback when streaming, else derived from `t_first_token_ms`), and generate() returning. The
snapshots are taken while no graph runs (one inference at a time, one context per model), so the
difference is exact; a counter read costs one relaxed atomic load each, twice per request. Host test:
`expert cache: decodes like the resident model` checks the snapshot against the final stats line.

`0028-expert-cache-route-hash.patch` (applies after 0027; `include/llama.h`, `src/llama-model.cpp`,
`src/llama-expert-cache.{h,cpp}`; instruments only): a 64-bit FNV-1a hash of the routing the cache
actually served. Per layer, a chain over the FINAL expert ids (after cache-aware swaps; int16 LE, in
token order) of every token the layer saw, taken where the plan pins them (so the GPU route's replay
and the CPU path hash the same ids) and in the expert-major prep for prompt ubatches (hashed once, at
`pos == 0`, so a prompt split into other ubatches hashes the same). The request's value folds the layers
in layer-id order (4 bytes layer id LE, then the layer's 8 bytes LE). `llama_expert_stats` gains
`route_hash` and `route_n` (the (token, layer) pairs covered; appended, so an older caller's struct is
unaffected) and `llama_model_expert_route_reset(model)` restarts the chains (`LLAMA_HAS_EXPERT_ROUTE_HASH`).
The bridge resets as a request starts, and the reply's `expert` object gains `"route":"<16 hex>"` and
`"route_n":N` (a hex string: JSON numbers are doubles in a WinRT client). A turn that reuses the KV
cache hashes only the tokens it computed. Equal prompt, sampling and cache state give equal `route`
(the same calls, whatever their ubatch size or path; a prompt cut into other decode calls is another
request: the last layer computes only each call's output token); a different route with the same
tokens is a routing change. Cost: one multiply and xor per id byte, 2
bytes per id, `n_used` ids per token per layer.

## Applying

```bash
./scripts/apply-uwp-patches.sh   # idempotent; used by the llamacpp CI variant
```

## Rebasing after a submodule bump

The patch is a plain `git diff`. If it no longer applies, redo the three guards
by hand (they are one-liners plus a small `#if` block — see the patch body) and
regenerate with `git -C llama.cpp diff > patches/0001-uwp-appcontainer-guards.patch`.
