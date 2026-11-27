// Read-bandwidth roofline probe (shaders/variants.txt tags roof/roofx/roofxo,
// never built in): stands in for an mmv slot under XLLAMA_D3D12_SHADER_DIR and
// streams the bytes that slot's selftest cases read, NUM_ROWS rows per group as
// the host's grid has them, with coalesced loads and no math. The case's
// packed_gbs is then the read ceiling of that shape on the device, the number
// the real kernel's packed_gbs is measured against; its rel_err means nothing
// (ok = 0). ROOF_X picks what a group reads:
//   0 its weight rows (W, a DEFAULT heap buffer);
//   1 its weight rows plus the whole q8_K activation row (X, the CUSTOM L0
//     write-back scratch every real mmv group reads once);
//   2 the activation row only, so 1 - 0 and 2 price the X reads.

#include "ggml_d3d12_common.hlsli"

#ifndef ROOF_X
#define ROOF_X 0
#endif

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint rows = min((uint)NUM_ROWS, n - row0);
    uint acc = 0u;
#if ROOF_X != 2
    const uint base = row0 * w_row_bytes;
    const uint end = base + rows * w_row_bytes;
    for (uint o = base + 16u * tid; o + 16u <= end; o += 16u * NUM_THREADS) {
        const uint4 v = W.Load4(o);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
#endif
#if ROOF_X != 0
    // block_q8_K is 292 B = 73 dwords; a column is nchunk blocks.
    const uint xb = gid.y * x_stride;
    for (uint i = tid; i < nchunk * 73u; i += NUM_THREADS)
        acc ^= X.Load(xb + 4u * i);
#endif
    // Every load feeds a store, so dxc keeps them all.
    acc = WaveActiveBitXor(acc);
    if (WaveIsFirstLane())
        Y[gid.y * y_stride + row0 + (tid / WaveGetLaneCount()) % rows] = asfloat(acc);
}
