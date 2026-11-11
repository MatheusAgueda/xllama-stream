// ggml backend d3d12: MUL_MAT_ID with IQ3_S experts x q8_K activations — the
// expert matmuls of a MoE layer whose routed experts are IQ3_S (Qwen3.6-35B-A3B
// i1-IQ3_S: 1.289 MiB per expert against IQ4_XS's 1.594, so ~23% more expert
// cache slots in the same D3D12_Host budget). Indexing, constants and grid are
// ggml_d3d12_mmid_iq4_xs.hlsl's (src/bridge/ggml_d3d12_mmid_emu.h plans both);
// the block math and thread layout are ggml_d3d12_mmv_iq3_s.hlsl's, with the
// weight base moved to the slot Ids[t][i] picks and the activation column to
// t * x_stride + i * x_col_stride. d3d12_mmid_emulate mirrors this kernel.
// The group shape comes from pad0 (ggml_d3d12_geom.hlsli): ffn_down_exps (K =
// 512, 2 chunks) can put the 64-thread blob's 2 idle chunk lanes on more rows.

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

uint ld8(uint a) {
    return (ld16(a & ~1u) >> (8u * (a & 1u))) & 0xffu;
}

// Negate the bytes of g whose bit is set in the 4-bit mask s4 (grid bytes are
// 1..15, so the byte-wise two's-complement has no carry between lanes).
uint signed4(uint g, uint s4) {
    const uint m = ((s4 * 0x00204081u) & 0x01010101u) * 0xffu;
    return (g ^ m) + (m & 0x01010101u);
}

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
    const uint itid = tid & 15u;
    const uint ib32 = itid >> 1;
    const uint h = itid & 1u;
    const uint e = 32u * ib32 + 16u * h;

    for (uint t = 0; t < gm.rt; ++t) {
        const uint row0 = geom_row0(gm, gid.x, t);
        float acc[NUM_ROWS];
        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r)
            acc[r] = 0.0;

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
        reduce_store_geom(acc, tid, gm, row0, ybase);
    }
}
