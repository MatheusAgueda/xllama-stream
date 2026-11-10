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
// negated: it can be -128).

#include "ggml_d3d12_common.hlsli"

static const uint kIq3sGrid[512] = {
#include "ggml_d3d12_iq3s_grid.inc"
};

groupshared uint grid[512];

uint ld8(uint a) {
    return (ld16(a & ~1u) >> (8u * (a & 1u))) & 0xffu;
}

// Negate the bytes of g whose bit is set in the 4-bit mask s4.
uint signed4(uint g, uint s4) {
    const uint m = ((s4 * 0x00204081u) & 0x01010101u) * 0xffu;
    return (g ^ m) + (m & 0x01010101u);
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    for (uint gi = tid; gi < 512u; gi += NUM_THREADS)
        grid[gi] = kIq3sGrid[gi];
    GroupMemoryBarrierWithGroupSync();

    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint ib32 = itid >> 1;
    const uint h = itid & 1u;
    const uint e = 32u * ib32 + 16u * h;

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        const uint xb = col * x_stride + blk * 292u;
        const float dy = asfloat(X.Load(xb));
        const uint4 xv = X.Load4(xb + 4u + e);

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 110u;
            const float d = f16tof32(ld16(bb));
            const uint qs = ld32(bb + 2u + 8u * ib32 + 4u * h);
            const uint hb = (ld8(bb + 66u + ib32) >> (4u * h)) & 0xFu;
            const uint sg = ld16(bb + 74u + 4u * ib32 + 2u * h);
            const int ls = 2 * (int)((ld8(bb + 106u + (ib32 >> 1)) >> (4u * (ib32 & 1u))) & 0xFu) + 1;
            const uint g0 = grid[(qs & 0xffu) | ((hb & 1u) << 8)];
            const uint g1 = grid[((qs >> 8) & 0xffu) | (((hb >> 1) & 1u) << 8)];
            const uint g2 = grid[((qs >> 16) & 0xffu) | (((hb >> 2) & 1u) << 8)];
            const uint g3 = grid[(qs >> 24) | (((hb >> 3) & 1u) << 8)];
            const int sumi = dot4(signed4(g0, sg & 0xFu), xv.x) +
                             dot4(signed4(g1, (sg >> 4) & 0xFu), xv.y) +
                             dot4(signed4(g2, (sg >> 8) & 0xFu), xv.z) +
                             dot4(signed4(g3, (sg >> 12) & 0xFu), xv.w);
            acc[r] += (d * dy) * (float)(ls * sumi);
        }
    }
    reduce_store(acc, tid, row0, col);
}
