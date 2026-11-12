// ggml backend d3d12: IQ2_S x q8_K matmul (decode and per-column prefill), the
// non-expert IQ2_S tensors of Qwen3.5-122B-A10B i1-IQ2_M (attn_qkv / attn_gate /
// attn_q / attn_k / ssm_out / ssm_alpha / ssm_beta / the shared expert). Layout,
// grid codes and block math: ggml_d3d12_iq2s_block.hlsli; the group shape comes
// from pad0 (ggml_d3d12_geom.hlsli), as for IQ3_S.

#include "ggml_d3d12_common.hlsli"
#include "ggml_d3d12_geom.hlsli"

static const uint kIq2sGrid[512] = {
#include "ggml_d3d12_iq2s_grid.inc"
};

groupshared uint grid[512];

#include "ggml_d3d12_iq2s_block.hlsli"

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    for (uint gi = tid; gi < 512u; gi += NUM_THREADS)
        grid[gi] = kIq2sGrid[gi];
    GroupMemoryBarrierWithGroupSync();

    const uint col = gid.y;
    const Geom gm = geom_of(tid);
    for (uint t = 0; t < gm.rt; ++t) {
        const uint row0 = geom_row0(gm, gid.x, t);
        float acc[NUM_ROWS];
        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r)
            acc[r] = 0.0;
        iq2s_tile(acc, gm, tid, 0u, row0, col * x_stride);
        reduce_store_geom(acc, tid, gm, row0, col * y_stride);
    }
}
