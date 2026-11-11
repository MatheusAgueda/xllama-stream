// ggml backend d3d12: the f32 ops between the matmuls of a decoder layer, so a
// layer's output projection, residual adds, norms, FFN and the next layer's
// input projections run as one submission instead of one per matmul group
// (docs in patches/README.md, 0005; src/bridge/ggml_d3d12.cpp graph_compute).
// One source, four kernels picked with -D KERNEL=n:
//   1 binary:   dst[r][i] = a[r][i] (+ | * | swiglu) b[r * b_stride][i]
//   2 rms_norm: dst[r] = a[r] / sqrt(mean(a[r]^2) + eps), optionally followed
//               by the norm weight: dst2[r][i] = dst[r][i] * w[i]
//   3 quant:    f32 columns -> block_q8_K, ggml's quantize_row_q8_K_ref
//   4 norm_quant: 2 with the weight, then 3 on dst2's rows (into u4), one dispatch
// Root signature: 8 constants at b0, root UAVs u0 (dst), u1 (a), u2 (b / w),
// u3 (dst2). Every buffer is D3D12_Host (UNORDERED_ACCESS), so all are raw UAVs.

#ifndef KERNEL
#define KERNEL 1
#endif

cbuffer Params : register(b0) {
    uint c0;
    uint c1;
    uint c2;
    uint c3;
    uint c4;
    uint c5;
    uint c6;
    uint c7;
};

RWByteAddressBuffer D : register(u0);
RWByteAddressBuffer A : register(u1);
RWByteAddressBuffer B : register(u2);
RWByteAddressBuffer D2 : register(u3);

float lda(uint i) {
    return asfloat(A.Load(4u * i));
}

#if KERNEL == 1
// c0 = ne0, c1 = rows, c2 = a row stride, c3 = b row stride (0: one row for
// all), c4 = dst row stride (floats), c5 = op (0 add, 1 mul, 2 swiglu: silu(a) * b).
[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint i = gid.x * 256u + tid;
    if (i >= c0)
        return;
    const uint r = gid.y;
    const float a = lda(r * c2 + i);
    const float b = asfloat(B.Load(4u * (r * c3 + i)));
    float y;
    if (c5 == 0u)
        y = a + b;
    else if (c5 == 1u)
        y = a * b;
    else
        y = a / (1.0f + exp(-a)) * b; // ggml_vec_swiglu_f32
    D.Store(4u * (r * c4 + i), asuint(y));
}
#elif KERNEL == 2
// One group per row. c0 = ne0, c1 = a row stride, c2 = dst row stride,
// c3 = eps (f32 bits), c4 = 1: also write dst2 = dst * w, c5 = dst2 row stride.
groupshared float red[256];

[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint r = gid.x;
    float s = 0.0f;
    for (uint i = tid; i < c0; i += 256u) {
        const float x = lda(r * c1 + i);
        s += x * x;
    }
    red[tid] = s;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride)
            red[tid] += red[tid + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    const float mean = red[0] / (float)c0;
    const float scale = 1.0f / sqrt(mean + asfloat(c3));
    for (uint j = tid; j < c0; j += 256u) {
        const float y = lda(r * c1 + j) * scale;
        D.Store(4u * (r * c2 + j), asuint(y));
        if (c4 != 0u) // dst2 may alias dst (an in-place MUL): written last
            D2.Store(4u * (r * c5 + j), asuint(y * asfloat(B.Load(4u * j))));
    }
}
#elif KERNEL == 3 || KERNEL == 4
// One group per (256-element block, column). c0 = blocks per column, c1 = a
// column stride (floats), c2 = output column stride (bytes). block_q8_K =
// { float d; int8 qs[256]; int16 bsums[16]; } = 292 B. Thread t owns elements
// 4t..4t+3. The absmax ties go to the lowest index, as the reference loop.

// RN(n / |b|) for an integer 1 <= n < 2^24 and a normal b, by integer long
// division: the correctly rounded quotient the CPU's IEEE division gives
// (fdiv here only has to be within 2.5 ULP). d3d12_div_rn in ggml_d3d12.cpp
// is the same code; the host test checks it against the CPU's 127/x and 1/x.
float div_rn(uint n, float b) {
    const uint bb = asuint(b) & 0x7fffffffu;
    const uint eb = bb >> 23;
    if (eb == 0u)
        return asfloat(0x7f800000u);
    const uint mb = (bb & 0x7fffffu) | 0x800000u; // |b| = mb * 2^(eb - 150)
    const uint s = 23u - firstbithigh(n);         // n = (n << s) * 2^-s
    uint r = n << s;
    uint q = 0u;
    [unroll]
    for (uint i = 0u; i < 26u; ++i) { // q = floor((n << s) / mb * 2^25)
        q <<= 1;
        if (r >= mb) {
            r -= mb;
            q |= 1u;
        }
        r <<= 1;
    }
    int e = 150 - int(s) - int(eb);
    uint m, guard;
    bool sticky = r != 0u;
    if (q >= (1u << 25)) {
        m = q >> 2;
        guard = (q >> 1) & 1u;
        sticky = sticky || (q & 1u) != 0u;
    } else {
        m = q >> 1;
        guard = q & 1u;
        e -= 1;
    }
    if (guard != 0u && (sticky || (m & 1u) != 0u)) {
        m += 1u;
        if (m == (1u << 24)) {
            m >>= 1;
            e += 1;
        }
    }
    if (e > 127)
        return asfloat(0x7f800000u);
    if (e < -126)
        return 0.0f;
    return asfloat((uint(e + 127) << 23) | (m & 0x7fffffu));
}

#if KERNEL == 3
groupshared float s_ax[64];
groupshared float s_v[64];
groupshared uint s_i[64];
groupshared int s_sum[64];

[numthreads(64, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint base = gid.y * c1 + gid.x * 256u + 4u * tid;
    float x[4];
    float ax = 0.0f, v = 0.0f;
    uint idx = 4u * tid;
    [unroll]
    for (uint j = 0; j < 4u; ++j) {
        x[j] = lda(base + j);
        if (abs(x[j]) > ax) {
            ax = abs(x[j]);
            v = x[j];
            idx = 4u * tid + j;
        }
    }
    s_ax[tid] = ax;
    s_v[tid] = v;
    s_i[tid] = idx;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float o = s_ax[tid + stride];
            if (o > s_ax[tid] || (o == s_ax[tid] && s_i[tid + stride] < s_i[tid])) {
                s_ax[tid] = o;
                s_v[tid] = s_v[tid + stride];
                s_i[tid] = s_i[tid + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float amax = s_ax[0];
    const uint ob = gid.y * c2 + gid.x * 292u;
    int q[4] = {0, 0, 0, 0};
    float d = 0.0f;
    if (amax != 0.0f) {
        const float v = s_v[0];
        const float ais = div_rn(127u, v); // |-127 / v|
        const float iscale = v > 0.0f ? -ais : ais;
        const float dm = div_rn(1u, ais); // |1 / iscale|
        d = v > 0.0f ? -dm : dm;
        [unroll]
        for (uint j = 0; j < 4u; ++j)
            q[j] = min(127, (int)round(iscale * x[j])); // round = nearest even, as nearest_int
    }
    D.Store(ob + 4u + 4u * tid, (uint(q[0]) & 0xffu) | ((uint(q[1]) & 0xffu) << 8) |
                                ((uint(q[2]) & 0xffu) << 16) | (uint(q[3]) << 24));
    s_sum[tid] = q[0] + q[1] + q[2] + q[3];
    GroupMemoryBarrierWithGroupSync();
    if ((tid & 7u) == 0u) {
        const int lo = s_sum[tid] + s_sum[tid + 1u] + s_sum[tid + 2u] + s_sum[tid + 3u];
        const int hi = s_sum[tid + 4u] + s_sum[tid + 5u] + s_sum[tid + 6u] + s_sum[tid + 7u];
        D.Store(ob + 260u + (tid >> 1), (uint(lo) & 0xffffu) | (uint(hi) << 16));
    }
    if (tid == 0u)
        D.Store(ob, asuint(d));
}
#else // KERNEL 4
// rms_norm, the norm weight and block_q8_K of the product in one dispatch, one
// group per row (ne0 = 256 * nb, nb <= 16). Kernel 2's arithmetic for dst and
// dst2 (same per-thread sums, same tree), then kernel 3's per block on those
// floats from groupshared (thread 4t..4t+3 of the block, ties to the lowest
// index, div_rn), four blocks per pass: the bytes both kernels write.
// c0 = ne0, c1 = a row stride, c2 = dst row stride, c3 = eps (f32 bits),
// c4 = dst2 row stride (floats), c5 = q8_K row stride (bytes); u4 = q8_K out.
RWByteAddressBuffer Q : register(u4);

groupshared float red[256];
groupshared float zs[4096];
groupshared float s_ax[256];
groupshared float s_v[256];
groupshared uint s_i[256];
groupshared int s_sum[256];

[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint r = gid.x;
    float s = 0.0f;
    for (uint i = tid; i < c0; i += 256u) {
        const float x = lda(r * c1 + i);
        s += x * x;
    }
    red[tid] = s;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride)
            red[tid] += red[tid + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    const float mean = red[0] / (float)c0;
    const float scale = 1.0f / sqrt(mean + asfloat(c3));
    for (uint j = tid; j < c0; j += 256u) {
        const float y = lda(r * c1 + j) * scale;
        const float z = y * asfloat(B.Load(4u * j));
        D.Store(4u * (r * c2 + j), asuint(y));
        D2.Store(4u * (r * c4 + j), asuint(z)); // dst2 may alias dst: written last
        zs[j] = z;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint lb = tid >> 6; // block of this pass
    const uint lt = tid & 63u;
    for (uint b0 = 0u; b0 < c0 / 256u; b0 += 4u) {
        const uint blk = b0 + lb;
        const bool live = blk < c0 / 256u;
        float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float ax = 0.0f, v = 0.0f;
        uint idx = 4u * lt;
        if (live) {
            [unroll]
            for (uint j = 0; j < 4u; ++j) {
                x[j] = zs[blk * 256u + 4u * lt + j];
                if (abs(x[j]) > ax) {
                    ax = abs(x[j]);
                    v = x[j];
                    idx = 4u * lt + j;
                }
            }
        }
        s_ax[tid] = ax;
        s_v[tid] = v;
        s_i[tid] = idx;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint stride = 32u; stride > 0u; stride >>= 1u) {
            if (lt < stride) {
                const float o = s_ax[tid + stride];
                if (o > s_ax[tid] || (o == s_ax[tid] && s_i[tid + stride] < s_i[tid])) {
                    s_ax[tid] = o;
                    s_v[tid] = s_v[tid + stride];
                    s_i[tid] = s_i[tid + stride];
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        const float amax = s_ax[lb * 64u];
        const uint ob = r * c5 + blk * 292u;
        int q[4] = {0, 0, 0, 0};
        float d = 0.0f;
        if (amax != 0.0f) {
            const float vm = s_v[lb * 64u];
            const float ais = div_rn(127u, vm); // |-127 / v|
            const float iscale = vm > 0.0f ? -ais : ais;
            const float dm = div_rn(1u, ais); // |1 / iscale|
            d = vm > 0.0f ? -dm : dm;
            [unroll]
            for (uint j = 0; j < 4u; ++j)
                q[j] = min(127, (int)round(iscale * x[j]));
        }
        if (live)
            Q.Store(ob + 4u + 4u * lt, (uint(q[0]) & 0xffu) | ((uint(q[1]) & 0xffu) << 8) |
                                       ((uint(q[2]) & 0xffu) << 16) | (uint(q[3]) << 24));
        s_sum[tid] = q[0] + q[1] + q[2] + q[3];
        GroupMemoryBarrierWithGroupSync();
        if (live && (lt & 7u) == 0u) {
            const int lo = s_sum[tid] + s_sum[tid + 1u] + s_sum[tid + 2u] + s_sum[tid + 3u];
            const int hi = s_sum[tid + 4u] + s_sum[tid + 5u] + s_sum[tid + 6u] + s_sum[tid + 7u];
            Q.Store(ob + 260u + (lt >> 1), (uint(lo) & 0xffffu) | (uint(hi) << 16));
        }
        if (live && lt == 0u)
            Q.Store(ob, asuint(d));
        GroupMemoryBarrierWithGroupSync(); // s_* are rewritten by the next pass
    }
}
#endif // KERNEL 3 / 4
#endif
