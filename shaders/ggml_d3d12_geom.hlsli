// Group geometry of the IQ3_S mmv and mmid kernels (include/xllama/ggml_d3d12.h,
// d3d12_geom_for): pad0 = rsl | rt << 8, 0 read as 1. The IN_FLIGHT chunk lanes
// (16 threads each) split into rsl row sets of cl = IN_FLIGHT / rsl lanes; row
// set rs of tile t covers NUM_ROWS rows from ((gid.x * rt + t) * rsl + rs) *
// NUM_ROWS, its lanes striding K by cl chunks. rsl = rt = 1 is the plain layout,
// with the same blocks per thread and the same reduction tree, so the same sums.
// Include after ggml_d3d12_common.hlsli.

struct Geom {
    uint rsl; // row sets side by side (power of two, <= IN_FLIGHT)
    uint rt;  // row tiles walked in turn
    uint cl;  // chunk lanes per row set
    uint rs;  // this thread's row set
    uint ix;  // this thread's chunk lane in its row set
    uint seg; // threads per row set (cl * 16)
};

Geom geom_of(uint tid) {
    Geom g;
    g.rsl = clamp(pad0 & 0xffu, 1u, (uint)IN_FLIGHT);
    g.rt = max((pad0 >> 8) & 0xffu, 1u);
    g.cl = IN_FLIGHT / g.rsl;
    const uint u = tid >> 4;
    g.rs = u / g.cl;
    g.ix = u - g.rs * g.cl;
    g.seg = g.cl * 16u;
    return g;
}

uint geom_row0(Geom g, uint gx, uint t) {
    return ((gx * g.rt + t) * g.rsl + g.rs) * NUM_ROWS;
}

// reduce_store per row set: each segment of seg threads sums its own rows'
// partials with the plain tree (stride seg/2 .. 1) and its first NUM_ROWS
// threads store them at ybase. Ends with a barrier when another tile reuses red.
void reduce_store_geom(float acc[NUM_ROWS], uint tid, Geom g, uint row0, uint ybase) {
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        red[r][tid] = acc[r];
    GroupMemoryBarrierWithGroupSync();
    const uint lane = tid & (g.seg - 1u);
    for (uint stride = g.seg / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            [unroll]
            for (uint r = 0; r < NUM_ROWS; ++r)
                red[r][tid] += red[r][tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane < NUM_ROWS && row0 + lane < n)
        Y[ybase + row0 + lane] = red[lane][tid - lane];
    if (g.rt > 1u)
        GroupMemoryBarrierWithGroupSync();
}
