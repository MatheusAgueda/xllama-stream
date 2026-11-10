// ggml backend d3d12: IQ4_XS x q8_K matmul (decode and per-column prefill).
// block_iq4_xs = { half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]; }
// = 136 B, so blocks and rows are 4-byte aligned. dequantize_row_iq4_xs, per
// 32-element group ib (qs += 16 each):
//   ls = ((scales_l[ib/2] >> 4*(ib&1)) & 0xF) | (((scales_h >> 2*ib) & 3) << 4)
//   y[j] = d * (ls - 32) * kvalues_iq4nl[qs[j] & 0xF]
//   y[j + 16] = d * (ls - 32) * kvalues_iq4nl[qs[j] >> 4]           (j = 0..15)
// Thread itid: ib = itid/2, h = itid%2 owns qs bytes 16*ib + 8h .. +7, whose
// low nibbles are elements 32*ib + 8h .. +7 and high nibbles 32*ib + 16 + 8h
// .. +7. The codebook values (-127..113) are valid int8, so four lookups pack
// straight into dot4add_i8packed against the q8 activations.

#include "ggml_d3d12_common.hlsli"

// kvalues_iq4nl as two's-complement bytes.
static const uint kIq4nl[16] = {0x81u, 0x98u, 0xADu, 0xBFu, 0xCFu, 0xDDu, 0xEAu, 0xF6u,
                                0x01u, 0x0Du, 0x19u, 0x26u, 0x35u, 0x45u, 0x59u, 0x71u};

groupshared uint lut[16];

// The nibbles at bits sh, sh + 8, sh + 16, sh + 24 of q through the codebook,
// packed as four int8 lanes in element order.
uint lut4(uint q, uint sh) {
    return lut[(q >> sh) & 0xFu] | (lut[(q >> (sh + 8u)) & 0xFu] << 8) |
           (lut[(q >> (sh + 16u)) & 0xFu] << 16) | (lut[(q >> (sh + 24u)) & 0xFu] << 24);
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    if (tid < 16u)
        lut[tid] = kIq4nl[tid];
    GroupMemoryBarrierWithGroupSync();

    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
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
        const uint xb = col * x_stride + blk * 292u;
        const float dy = asfloat(X.Load(xb));
        const uint2 xl = X.Load2(xb + 4u + e_lo);
        const uint2 xh = X.Load2(xb + 20u + e_lo);

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 136u;
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
    reduce_store(acc, tid, row0, col);
}
