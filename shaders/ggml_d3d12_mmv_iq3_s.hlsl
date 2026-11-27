// ggml backend d3d12: IQ3_S x q8_K matmul (decode and per-column prefill).
// block_iq3_s = { half d; uint8 qs[64]; uint8 qh[8]; uint8 signs[32]; uint8 scales[4]; }
// = 110 B, so blocks and rows sit on 2-byte boundaries (ld32 / ld16 handle it).
// ggml_vec_dot_iq3_s_q8_K, per 32-element group ib32 (qs += 8, signs += 4 each):
//   ls = 2 * ((scales[ib32/2] >> 4*(ib32&1)) & 0xF) + 1
//   for l in 0..3: grid1 = iq3s_grid[qs[2l]   | ((qh[ib32] >> 2l)   & 1) << 8]
//                  grid2 = iq3s_grid[qs[2l+1] | ((qh[ib32] >> 2l+1) & 1) << 8]
//     sum += grid1[j] * q8[8l+j]   * (signs[l] bit j     ? -1 : 1)
//          + grid2[j] * q8[8l+4+j] * (signs[l] bit j + 4 ? -1 : 1)   (j = 0..3)
//   result = d * d_y * sum(ls * sum)
// Thread itid: ib32 = itid/2, h = itid%2 owns l = 2h and 2h+1, i.e. the 16
// elements 32*ib32 + 16h .. +15: one qs dword (4 grid indices in element
// order), qh bits 4h..4h+3 (one per index), one signs half-word (one nibble
// per index), four q8 dwords. Grid bytes are 1..15, never 0, so a byte-wise
// two's-complement negate under the sign mask has no carry between lanes, and
// the signed magnitudes go straight into dot4add_i8packed (the q8 side is never
// negated: it can be -128). The group shape comes from pad0
// (ggml_d3d12_geom.hlsli), the block math from ggml_d3d12_iq3s_block.hlsli.

#include "ggml_d3d12_common.hlsli"
#include "ggml_d3d12_geom.hlsli"

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

    const uint col = gid.y;
    const Geom gm = geom_of(tid);
    for (uint t = 0; t < gm.rt; ++t) {
        const uint row0 = geom_row0(gm, gid.x, t);
        float acc[NUM_ROWS];
        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r)
            acc[r] = 0.0;
        iq3s_tile(acc, gm, tid, 0u, row0, col * x_stride);
        reduce_store_geom(acc, tid, gm, row0, col * y_stride);
    }
}
