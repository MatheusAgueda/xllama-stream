// ggml backend d3d12: MUL_MAT_ID with IQ3_S experts x q8_K activations — the
// expert matmuls of a MoE layer whose routed experts are IQ3_S (Qwen3.6-35B-A3B
// i1-IQ3_S: 1.289 MiB per expert against IQ4_XS's 1.594, so ~23% more expert
// cache slots in the same D3D12_Host budget). Indexing, constants and grid are
// ggml_d3d12_mmid_iq4_xs.hlsl's (src/bridge/ggml_d3d12_mmid_emu.h plans both);
// the block math and thread layout are ggml_d3d12_mmv_iq3_s.hlsl's, with the
// weight base moved to the slot Ids[t][i] picks and the activation column to
// t * x_stride + i * x_col_stride. d3d12_mmid_emulate mirrors this kernel.
// The group shape comes from pad0 (ggml_d3d12_geom.hlsli): ffn_down_exps (K =
// 512, 2 chunks) can put the 64-thread blob's 2 idle chunk lanes on more rows;
// the block math is ggml_d3d12_iq3s_block.hlsli.

#define PARAMS_EXTRA \
    uint expert_stride; \
    uint n_ids; \
    uint ids_stride; \
    uint x_col_stride; \
    uint y_expert_stride;

#include "ggml_d3d12_common.hlsli"
#include "ggml_d3d12_geom.hlsli"

ByteAddressBuffer Ids : register(t1);

static const uint kIq3sGrid[512] = {
#include "ggml_d3d12_iq3s_grid.inc"
};

groupshared uint grid[512];

#include "ggml_d3d12_iq3s_block.hlsli"

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    for (uint gi = tid; gi < 512u; gi += NUM_THREADS)
        grid[gi] = kIq3sGrid[gi];
    GroupMemoryBarrierWithGroupSync();

    const uint token = gid.y / n_ids;
    const uint slot_i = gid.y - token * n_ids;
    const uint wb = Ids.Load(token * ids_stride + 4u * slot_i) * expert_stride;
    const uint xc = token * x_stride + slot_i * x_col_stride;

    const uint ybase = token * y_stride + slot_i * y_expert_stride;
    const Geom gm = geom_of(tid);
    for (uint t = 0; t < gm.rt; ++t) {
        const uint row0 = geom_row0(gm, gid.x, t);
        float acc[NUM_ROWS];
        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r)
            acc[r] = 0.0;
        iq3s_tile(acc, gm, tid, wb, row0, xc);
        reduce_store_geom(acc, tid, gm, row0, ybase);
    }
}
