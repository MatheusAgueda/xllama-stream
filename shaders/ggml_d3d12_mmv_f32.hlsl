// ggml backend d3d12: F32 x f32 matmul — the MoE router (ffn_gate_inp) and the
// shared-expert gate, which llama-quantize keeps in f32. X is the f32
// activation itself, not a quantized copy: u1 points at the column data,
// x_stride bytes apart. Thread itid of chunk blk takes the 4 floats at
// 64 * j + 4 * itid (j = 0..3), so each Load4 of the 16 threads reads 256
// contiguous bytes; the products are summed in that order, as
// thread_chunk(GGML_TYPE_F32) does on the host.

#include "ggml_d3d12_common.hlsli"

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        [unroll]
        for (uint j = 0; j < 4u; ++j) {
            const uint e = (blk * 256u + 64u * j + 4u * itid) * 4u;
            const float4 xv = asfloat(X.Load4(col * x_stride + e));
            [unroll]
            for (uint r = 0; r < NUM_ROWS; ++r) {
                const uint row = min(row0 + r, n - 1u);
                const float4 wv = asfloat(W.Load4(row * w_row_bytes + e));
                precise float s = wv.x * xv.x;
                s += wv.y * xv.y;
                s += wv.z * xv.z;
                s += wv.w * xv.w;
                acc[r] += s;
            }
        }
    }
    reduce_store(acc, tid, row0, col);
}
