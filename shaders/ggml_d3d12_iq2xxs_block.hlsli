// IQ2_XXS block math of ggml_d3d12_mmv_iq2_xxs.hlsl and ggml_d3d12_mmid_iq2_xxs.hlsl:
// one row tile (NUM_ROWS rows from row0) of the weights at byte wb against the
// q8_K activation column at byte xc, this thread's 16 elements of each 256-chunk
// its chunk lane walks (ggml_d3d12_geom.hlsli). Include after the geometry
// header, with grid[128] filled from ggml_d3d12_iq2xxs_grid.inc.
//
// block_iq2_xxs = { half d; uint16 qs[32]; } = 66 B, so blocks sit on 2-byte
// boundaries (ld16 handles it). ggml_vec_dot_iq2_xxs_q8_K, per 32-element group
// ib32 (8 bytes at 2 + 8 * ib32): aux0 = four grid indices (a byte per l),
// aux1 = four 7-bit sign indices (bits 7l..7l+6) and the scale (bits 28..31):
//   ls = 2 * (aux1 >> 28) + 1
//   signs = ksigns_iq2xs[(aux1 >> 7l) & 127] = the 7 bits plus their parity as bit 7
//   sum += grid[aux0 byte l][j] * q8[8l + j] * (signs bit j ? -1 : 1)   (j = 0..7)
//   result = 0.125 * sum over blocks of d * d_y * ls * sum
// Thread itid: ib32 = itid/2, h = itid%2 owns l = 2h and 2h+1, the 16 elements
// 32*ib32 + 16h .. +15: one qs half-word of aux0 (two grid indices), aux1's sign
// indices 2h and 2h+1 (bits 14h..14h+13), four q8 dwords. The grid has IQ2_S's
// magnitudes (8, 25, 43) and packing, so iq2s_sumi decodes it unchanged; ksigns is
// ALU (countbits), no table. The 0.125 is folded into d, as for IQ2_S.

#include "ggml_d3d12_iq2s_block.hlsli"

// ksigns_iq2xs[v]: v with bit 7 = the parity of its 7 bits (an even count of negatives).
uint iq2xxs_signs(uint v) {
    return v | ((countbits(v) & 1u) << 7);
}

void iq2xxs_tile(inout float acc[NUM_ROWS], Geom gm, uint tid, uint wb, uint row0, uint xc) {
    const uint itid = tid & 15u;
    const uint ib32 = itid >> 1;
    const uint h = itid & 1u;
    const uint e = 32u * ib32 + 16u * h;
    for (uint blk = gm.ix; blk < nchunk; blk += gm.cl) {
        const uint xb = xc + blk * 292u;
        const float dy = asfloat(X.Load(xb));
        const uint4 xv = X.Load4(xb + 4u + e);
        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = wb + row * w_row_bytes + blk * 66u;
            const float d = f16tof32(ld16(bb)) * 0.125;
            const uint g8 = bb + 2u + 8u * ib32;
            const uint qs = ld16(g8 + 2u * h);
            const uint a1 = ld16(g8 + 4u) | (ld16(g8 + 6u) << 16);
            const int ls = 2 * (int)(a1 >> 28) + 1;
            const uint sv = a1 >> (14u * h);
            const uint sg = iq2xxs_signs(sv & 127u) | (iq2xxs_signs((sv >> 7) & 127u) << 8);
            acc[r] += (d * dy) * (float)(ls * iq2s_sumi(qs & 0xffu, qs >> 8, sg, xv));
        }
    }
}
