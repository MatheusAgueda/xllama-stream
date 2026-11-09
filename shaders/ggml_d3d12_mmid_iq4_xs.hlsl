// ggml backend d3d12: MUL_MAT_ID with IQ4_XS experts x q8_K activations, the
// expert matmuls of a MoE layer (ggml_mul_mat_id). W holds every expert,
// expert_stride bytes apart; with the expert cache (llama.cpp patch 0004) those
// are the resident slots and Ids holds slot ids. Ids is the I32 ids tensor:
// n_ids per token, ids_stride bytes between tokens (a view, so it can exceed
// 4 * n_ids). Group (gid.x, gid.y) computes rows 4 * gid.x .. +3 of expert
// Ids[t][i], token t = gid.y / n_ids and i = gid.y % n_ids, against the q8_K
// activation column at t * x_stride + i * x_col_stride (x_col_stride 0: gate
// and up share the token's column; down has one column per id), into
// Y[t * y_stride + i * y_expert_stride + row]. The block math, thread layout
// and reduction are ggml_d3d12_mmv_iq4_xs.hlsl's;
// src/bridge/ggml_d3d12_mmid_emu.h emulates this kernel and derives the 13
// constants from the ggml node.

// b0 after the 8 common constants:
//   expert_stride    bytes between experts in W (nb02)
//   n_ids            experts per token (ids ne0)
//   ids_stride       bytes between tokens in Ids (ids nb1)
//   x_col_stride     bytes between a token's activation columns, 0 = one column
//   y_expert_stride  floats between a token's expert outputs (dst nb1 / 4)
// x_stride is the bytes between tokens in X and y_stride the floats between
// tokens in Y (dst nb2 / 4).
#define PARAMS_EXTRA \
    uint expert_stride; \
    uint n_ids; \
    uint ids_stride; \
    uint x_col_stride; \
    uint y_expert_stride;

#include "ggml_d3d12_common.hlsli"

ByteAddressBuffer Ids : register(t1);

// kvalues_iq4nl as two's-complement bytes.
static const uint kIq4nl[16] = {0x81u, 0x98u, 0xADu, 0xBFu, 0xCFu, 0xDDu, 0xEAu, 0xF6u,
                                0x01u, 0x0Du, 0x19u, 0x26u, 0x35u, 0x45u, 0x59u, 0x71u};

groupshared uint lut[16];

uint lut4(uint q, uint sh) {
    return lut[(q >> sh) & 0xFu] | (lut[(q >> (sh + 8u)) & 0xFu] << 8) |
           (lut[(q >> (sh + 16u)) & 0xFu] << 16) | (lut[(q >> (sh + 24u)) & 0xFu] << 24);
}

// reduce_store with the output column at ybase floats.
void reduce_store_at(float acc[NUM_ROWS], uint tid, uint row0, uint ybase) {
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        red[r][tid] = acc[r];
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = NUM_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            [unroll]
            for (uint r = 0; r < NUM_ROWS; ++r)
                red[r][tid] += red[r][tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid < NUM_ROWS && row0 + tid < n)
        Y[ybase + row0 + tid] = red[tid][0];
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    if (tid < 16u)
        lut[tid] = kIq4nl[tid];
    GroupMemoryBarrierWithGroupSync();

    const uint token = gid.y / n_ids;
    const uint slot_i = gid.y - token * n_ids;
    const uint wb = Ids.Load(token * ids_stride + 4u * slot_i) * expert_stride;
    const uint xc = token * x_stride + slot_i * x_col_stride;

    const uint row0 = gid.x * NUM_ROWS;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint ib = itid >> 1;
    const uint h = itid & 1u;
    const uint e_lo = 32u * ib + 8u * h;

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        const uint xb = xc + blk * 292u;
        const float dy = asfloat(X.Load(xb));
        const uint2 xl = X.Load2(xb + 4u + e_lo);
        const uint2 xh = X.Load2(xb + 20u + e_lo);

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = wb + row * w_row_bytes + blk * 136u;
            const uint2 hdr = W.Load2(bb);
            const uint2 q = W.Load2(bb + 8u + 16u * ib + 8u * h);
            const float d = f16tof32(hdr.x & 0xffffu);
            const int ls =
                (int)(((hdr.y >> (4u * ib)) & 0xFu) | (((hdr.x >> (16u + 2u * ib)) & 3u) << 4)) - 32;
            const int sumi = dot4(lut4(q.x, 0u), xl.x) + dot4(lut4(q.y, 0u), xl.y) +
                             dot4(lut4(q.x, 4u), xh.x) + dot4(lut4(q.y, 4u), xh.y);
            acc[r] += (d * dy) * (float)(ls * sumi);
        }
    }
    reduce_store_at(acc, tid, row0, token * y_stride + slot_i * y_expert_stride);
}
