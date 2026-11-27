// IQ2_S block math of ggml_d3d12_mmv_iq2_s.hlsl and ggml_d3d12_mmid_iq2_s.hlsl:
// one row tile (NUM_ROWS rows from row0) of the weights at byte wb against the
// q8_K activation column at byte xc, this thread's 16 elements of each 256-chunk
// its chunk lane walks (ggml_d3d12_geom.hlsli). Include after the geometry
// header, with grid[512] filled from ggml_d3d12_iq2s_grid.inc.
//
// block_iq2_s = { half d; uint8 qs[32]; uint8 signs[32]; uint8 qh[8]; uint8 scales[8]; }
// = 82 B, so blocks sit on 2-byte boundaries (ld16 handles it).
// ggml_vec_dot_iq2_s_q8_K, per 32-element group ib32 (qs += 4, signs += 4 each):
//   ls1 = 1 + 2 * (scales[ib32] & 0xF) for l = 0, 1; ls2 = 1 + 2 * (scales[ib32] >> 4) for l = 2, 3
//   grid = iq2s_grid[qs[l] | ((qh[ib32] >> 2l) & 3) << 8]
//   sum_l += grid[j] * q8[8l + j] * (signs[l] bit j ? -1 : 1)   (j = 0..7)
//   result = 0.125 * sum over blocks of d * d_y * (ls1 * (sum_0 + sum_1) + ls2 * (sum_2 + sum_3))
// Thread itid: ib32 = itid/2, h = itid%2 owns l = 2h and 2h+1 (one scale nibble),
// the 16 elements 32*ib32 + 16h .. +15: one qs half-word (two grid indices), qh
// bits 4h..4h+3 (two per index), one signs half-word (a byte per index, a bit per
// element), four q8 dwords. Grid bytes are 8, 25 or 43, never 0, so signed4's
// byte-wise negate has no carry between lanes. The 0.125 is folded into d: a
// power of two, so (d / 8) * dy rounds to exactly (d * dy) / 8.

uint ld8(uint a) {
    return (ld16(a & ~1u) >> (8u * (a & 1u))) & 0xffu;
}

// Negate the bytes of g whose bit is set in the 4-bit mask s4.
uint signed4(uint g, uint s4) {
    const uint m = ((s4 * 0x00204081u) & 0x01010101u) * 0xffu;
    return (g ^ m) + (m & 0x01010101u);
}

// Four 2-bit grid codes (c0 in bits 0..1) -> four magnitude bytes 8 + 17c + (c >> 1).
uint iq2s_mag4(uint c) {
    const uint b = (c | (c << 6) | (c << 12) | (c << 18)) & 0x03030303u;
    return 0x08080808u + 17u * b + ((b >> 1) & 0x01010101u);
}

// Grid entry i's 16 codes (bytes 0..7 at bits 2j..2j+1).
uint iq2s_codes(uint i) {
    return (grid[i >> 1] >> (16u * (i & 1u))) & 0xffffu;
}

int iq2s_sumi(uint i0, uint i1, uint sg, uint4 xv) {
    const uint c0 = iq2s_codes(i0);
    const uint c1 = iq2s_codes(i1);
    const uint s0 = sg & 0xffu;
    const uint s1 = sg >> 8;
    return dot4(signed4(iq2s_mag4(c0 & 0xffu), s0 & 0xFu), xv.x) +
           dot4(signed4(iq2s_mag4(c0 >> 8), s0 >> 4), xv.y) +
           dot4(signed4(iq2s_mag4(c1 & 0xffu), s1 & 0xFu), xv.z) +
           dot4(signed4(iq2s_mag4(c1 >> 8), s1 >> 4), xv.w);
}

void iq2s_tile(inout float acc[NUM_ROWS], Geom gm, uint tid, uint wb, uint row0, uint xc) {
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
            const uint bb = wb + row * w_row_bytes + blk * 82u;
            const float d = f16tof32(ld16(bb)) * 0.125;
            const uint qs = ld16(bb + 2u + 4u * ib32 + 2u * h);
            const uint sg = ld16(bb + 34u + 4u * ib32 + 2u * h);
            const uint qh = ld8(bb + 66u + ib32) >> (4u * h);
            const int ls = 2 * (int)((ld8(bb + 74u + ib32) >> (4u * h)) & 0xFu) + 1;
            const uint i0 = (qs & 0xffu) | ((qh & 3u) << 8);
            const uint i1 = (qs >> 8) | (((qh >> 2) & 3u) << 8);
            acc[r] += (d * dy) * (float)(ls * iq2s_sumi(i0, i1, sg, xv));
        }
    }
}
