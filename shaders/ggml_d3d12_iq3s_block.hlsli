// IQ3_S block math of ggml_d3d12_mmv_iq3_s.hlsl and ggml_d3d12_mmid_iq3_s.hlsl:
// one row tile (NUM_ROWS rows from row0) of the weights at byte wb against the
// q8_K activation column at byte xc, this thread's 16 elements of each 256-chunk
// its chunk lane walks (ggml_d3d12_geom.hlsli). Include after the geometry
// header, with grid[512] filled.
//
// IQ3S_COOP 0 (built in): every thread loads the fields of its 16 elements from
// each of the NUM_ROWS rows itself, ~6 dword loads per row and block.
// IQ3S_COOP 1 (XLLAMA_D3D12_SHADER_DIR variant): the chunk lane's 16 threads
// first copy the NUM_ROWS 110-byte blocks to groupshared memory as 28 aligned
// dwords each (7 coalesced loads per thread instead of ~24), then read the
// fields from there. The same bytes reach the same arithmetic in the same
// order, so both give the same sums.

#ifndef IQ3S_COOP
#define IQ3S_COOP 0
#endif

uint ld8(uint a) {
    return (ld16(a & ~1u) >> (8u * (a & 1u))) & 0xffu;
}

// Negate the bytes of g whose bit is set in the 4-bit mask s4 (grid bytes are
// 1..15, so the byte-wise two's-complement has no carry between lanes).
uint signed4(uint g, uint s4) {
    const uint m = ((s4 * 0x00204081u) & 0x01010101u) * 0xffu;
    return (g ^ m) + (m & 0x01010101u);
}

int iq3s_sumi(uint qs, uint hb, uint sg, uint4 xv) {
    const uint g0 = grid[(qs & 0xffu) | ((hb & 1u) << 8)];
    const uint g1 = grid[((qs >> 8) & 0xffu) | (((hb >> 1) & 1u) << 8)];
    const uint g2 = grid[((qs >> 16) & 0xffu) | (((hb >> 2) & 1u) << 8)];
    const uint g3 = grid[(qs >> 24) | (((hb >> 3) & 1u) << 8)];
    return dot4(signed4(g0, sg & 0xFu), xv.x) + dot4(signed4(g1, (sg >> 4) & 0xFu), xv.y) +
           dot4(signed4(g2, (sg >> 8) & 0xFu), xv.z) + dot4(signed4(g3, (sg >> 12) & 0xFu), xv.w);
}

#if IQ3S_COOP
// A block seen from its dword-aligned start: 110 bytes from a 2-byte boundary
// end within 112 bytes.
#define IQ3S_SLAB 28u
groupshared uint slab[IN_FLIGHT * NUM_ROWS * IQ3S_SLAB];

uint sb8(uint s, uint o) {
    return (slab[s + (o >> 2)] >> (8u * (o & 3u))) & 0xffu;
}
uint sb16(uint s, uint o) { // o even
    return (slab[s + (o >> 2)] >> (8u * (o & 2u))) & 0xffffu;
}
uint sb32(uint s, uint o) { // o even
    const uint lo = slab[s + (o >> 2)];
    if ((o & 2u) == 0u)
        return lo;
    return (lo >> 16) | (slab[s + (o >> 2) + 1u] << 16);
}
#endif

void iq3s_tile(inout float acc[NUM_ROWS], Geom gm, uint tid, uint wb, uint row0, uint xc) {
    const uint itid = tid & 15u;
    const uint ib32 = itid >> 1;
    const uint h = itid & 1u;
    const uint e = 32u * ib32 + 16u * h;
#if IQ3S_COOP
    const uint sbase = (tid >> 4) * (NUM_ROWS * IQ3S_SLAB);
    // Every lane runs the same trip count (a lane past nchunk idles), so the
    // barriers are uniform.
    for (uint b0 = 0; b0 < nchunk; b0 += gm.cl) {
        const uint blk = b0 + gm.ix;
        const bool live = blk < nchunk;
        if (live) {
            [unroll]
            for (uint m = 0; m < (NUM_ROWS * IQ3S_SLAB) / 16u; ++m) {
                const uint dd = itid + 16u * m;
                const uint r = dd / IQ3S_SLAB;
                const uint row = min(row0 + r, n - 1u);
                const uint bb = wb + row * w_row_bytes + blk * 110u;
                slab[sbase + dd] = W.Load((bb & ~3u) + 4u * (dd - r * IQ3S_SLAB));
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (live) {
            const uint xb = xc + blk * 292u;
            const float dy = asfloat(X.Load(xb));
            const uint4 xv = X.Load4(xb + 4u + e);
            [unroll]
            for (uint r = 0; r < NUM_ROWS; ++r) {
                const uint row = min(row0 + r, n - 1u);
                const uint a = (wb + row * w_row_bytes + blk * 110u) & 3u;
                const uint s = sbase + r * IQ3S_SLAB;
                const float d = f16tof32(sb16(s, a));
                const uint qs = sb32(s, a + 2u + 8u * ib32 + 4u * h);
                const uint hb = (sb8(s, a + 66u + ib32) >> (4u * h)) & 0xFu;
                const uint sg = sb16(s, a + 74u + 4u * ib32 + 2u * h);
                const int ls = 2 * (int)((sb8(s, a + 106u + (ib32 >> 1)) >> (4u * (ib32 & 1u))) & 0xFu) + 1;
                acc[r] += (d * dy) * (float)(ls * iq3s_sumi(qs, hb, sg, xv));
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
#else
    for (uint blk = gm.ix; blk < nchunk; blk += gm.cl) {
        const uint xb = xc + blk * 292u;
        const float dy = asfloat(X.Load(xb));
        const uint4 xv = X.Load4(xb + 4u + e);
        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = wb + row * w_row_bytes + blk * 110u;
            const float d = f16tof32(ld16(bb));
            const uint qs = ld32(bb + 2u + 8u * ib32 + 4u * h);
            const uint hb = (ld8(bb + 66u + ib32) >> (4u * h)) & 0xFu;
            const uint sg = ld16(bb + 74u + 4u * ib32 + 2u * h);
            const int ls = 2 * (int)((ld8(bb + 106u + (ib32 >> 1)) >> (4u * (ib32 & 1u))) & 0xFu) + 1;
            acc[r] += (d * dy) * (float)(ls * iq3s_sumi(qs, hb, sg, xv));
        }
    }
#endif
}
