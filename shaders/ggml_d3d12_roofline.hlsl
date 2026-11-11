// Read-bandwidth roofline probe (shaders/variants.txt tag roof, never built in):
// stands in for an mmv slot under XLLAMA_D3D12_SHADER_DIR and streams the weight
// bytes that slot's selftest cases read, NUM_ROWS rows per group as the host's
// grid has them, with coalesced 16-byte loads and no math. The case's packed_gbs
// is then the read ceiling of that shape on the device, the number the real
// kernel's packed_gbs is measured against; its rel_err means nothing (ok = 0).

#include "ggml_d3d12_common.hlsli"

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint rows = min((uint)NUM_ROWS, n - row0);
    const uint base = row0 * w_row_bytes;
    const uint end = base + rows * w_row_bytes;
    uint acc = 0u;
    for (uint o = base + 16u * tid; o + 16u <= end; o += 16u * NUM_THREADS) {
        const uint4 v = W.Load4(o);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    // Every load feeds a store, so dxc keeps them all.
    acc = WaveActiveBitXor(acc);
    if (WaveIsFirstLane())
        Y[gid.y * y_stride + row0 + (tid / WaveGetLaneCount()) % rows] = asfloat(acc);
}
