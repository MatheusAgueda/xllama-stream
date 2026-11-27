// ggml backend d3d12: Q5_K x q8_K matmul (decode and per-column prefill).
// The Q4_K kernel's lane mapping plus the fifth bit. block_q5_K = { half d;
// half dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]; } = 176 B, 4-byte
// aligned. dequantize_row_q5_K, per 64-element group il (qs += 32 each):
//   y[64il + l]      = d * sc[2il]   * ((qs[l] & 0xF) | ((qh[l] >> 2il)     & 1) << 4) - dmin * m[2il]
//   y[64il + 32 + l] = d * sc[2il+1] * ((qs[l] >> 4)  | ((qh[l] >> 2il + 1) & 1) << 4) - dmin * m[2il+1]
// Thread itid: il = itid/4, ir = itid%4 owns l = 8ir .. 8ir + 7, i.e. qs bytes
// 32il + 8ir .. +7 and qh bytes 8ir .. +7. Values 0..31 are valid int8 lanes.
// Per super-block, like ggml_vec_dot_q5_K_q8_K:
// (d * d_y) * sum(sc * q5 * q8) - (dmin * d_y) * sum(m * q8).

#include "ggml_d3d12_common.hlsli"

uint scale_byte(uint i, uint4 hdr) {
    uint w = (i < 4u) ? hdr.y : ((i < 8u) ? hdr.z : hdr.w);
    return (w >> ((i & 3u) * 8u)) & 0xffu;
}

void get_scale_min_k4(uint j, uint4 hdr, out uint d, out uint m) {
    if (j < 4u) {
        d = scale_byte(j, hdr) & 63u;
        m = scale_byte(j + 4u, hdr) & 63u;
    } else {
        uint a = scale_byte(j + 4u, hdr);
        uint b = scale_byte(j - 4u, hdr);
        uint c = scale_byte(j, hdr);
        d = (a & 0xFu) | ((b >> 6) << 4);
        m = (a >> 4) | ((c >> 6) << 4);
    }
}

// Four 5-bit values: nibbles at bits sh, sh + 8, ... of q, fifth bit at bit hs
// of each byte of qh.
uint q5x4(uint q, uint sh, uint qh, uint hs) {
    return ((q >> sh) & 0x0F0F0F0Fu) | (((qh >> hs) & 0x01010101u) << 4);
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint il = itid >> 2;
    const uint ir = itid & 3u;
    const uint qs_byte = 48u + il * 32u + ir * 8u;
    const uint qh_byte = 16u + ir * 8u;
    const uint e_lo = il * 64u + ir * 8u;
    const uint e_hi = e_lo + 32u;

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        const uint xb = col * x_stride + blk * 292u;
        const float dy = asfloat(X.Load(xb));
        const uint2 xl = X.Load2(xb + 4u + e_lo);
        const uint2 xh = X.Load2(xb + 4u + e_hi);
        const int sl = sum4(xl.x) + sum4(xl.y);
        const int sh = sum4(xh.x) + sum4(xh.y);

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 176u;
            const uint4 hdr = W.Load4(bb);
            const uint2 q = W.Load2(bb + qs_byte);
            const uint2 qh = W.Load2(bb + qh_byte);

            const float d = f16tof32(hdr.x & 0xffffu);
            const float dmin = f16tof32(hdr.x >> 16);
            uint sc1, m1, sc2, m2;
            get_scale_min_k4(2u * il, hdr, sc1, m1);
            get_scale_min_k4(2u * il + 1u, hdr, sc2, m2);

            const int lo = dot4(q5x4(q.x, 0u, qh.x, 2u * il), xl.x) +
                           dot4(q5x4(q.y, 0u, qh.y, 2u * il), xl.y);
            const int hi = dot4(q5x4(q.x, 4u, qh.x, 2u * il + 1u), xh.x) +
                           dot4(q5x4(q.y, 4u, qh.y, 2u * il + 1u), xh.y);
            acc[r] += (d * dy) * (float)((int)sc1 * lo + (int)sc2 * hi) -
                      (dmin * dy) * (float)((int)m1 * sl + (int)m2 * sh);
        }
    }
    reduce_store(acc, tid, row0, col);
}
