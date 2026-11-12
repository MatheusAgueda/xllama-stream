// ggml backend d3d12: MUL_MAT_ID with IQ2_XXS experts x q8_K activations — the
// routed experts of Qwen3.5-122B-A10B i1-IQ2_XXS (gate/up/down 0.773 MiB each, so
// 2.32 MiB per expert per layer against IQ2_M's 2.90: ~20% fewer bytes per cache
// miss and more expert slots in the same D3D12_Host budget). Indexing and
// constants are ggml_d3d12_mmid_iq2_s.hlsl's (src/bridge/ggml_d3d12_mmid_emu.h
// plans both), the block math ggml_d3d12_iq2xxs_block.hlsli's; d3d12_mmid_emulate
// mirrors this kernel.

#define PARAMS_EXTRA \
    uint expert_stride; \
    uint n_ids; \
    uint ids_stride; \
    uint x_col_stride; \
    uint y_expert_stride;

#include "ggml_d3d12_common.hlsli"
#include "ggml_d3d12_geom.hlsli"

ByteAddressBuffer Ids : register(t1);

static const uint kIq2xxsGrid[128] = {
#include "ggml_d3d12_iq2xxs_grid.inc"
};

groupshared uint grid[128];

#include "ggml_d3d12_iq2xxs_block.hlsli"

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    for (uint gi = tid; gi < 128u; gi += NUM_THREADS)
        grid[gi] = kIq2xxsGrid[gi];
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
        iq2xxs_tile(acc, gm, tid, wb, row0, xc);
        reduce_store_geom(acc, tid, gm, row0, ybase);
    }
}
