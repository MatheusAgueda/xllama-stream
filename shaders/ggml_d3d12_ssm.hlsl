// ggml backend d3d12: the linear-attention (gated delta net) layers of
// qwen35moe, so the chain between the input projections and ssm_out runs on
// the GPU instead of a CPU split per layer
// (src/bridge/ggml_d3d12_ssm_emu.h matches the nodes and emulates each kernel).
// One source, five kernels picked with -D KERNEL=n:
//   1 unary:    dst = x * s (SCALE, no bias), sigmoid(x), softplus(x), silu(x)
//   2 copy:     4-d strided f32 copy (CPY, CONT, DUP, each half of a CONCAT)
//   3 get_rows: dst[r] = src[ids[r]], f32 rows, int32 ids (build_rs)
//   4 conv:     SSM_CONV, optionally followed by silu (the SILU node it feeds)
//   5 gdn:      GATED_DELTA_NET, S_v = 128, scalar or per-channel (KDA) gate,
//               any token count, K >= 1 state snapshots
// The copies are bit exact. The others round each operation once, in the
// order the emulation uses (`precise` keeps dxc from forming mads); exp and
// log are the GPU's, so they match ggml's CPU ops within a tolerance only.
// Root signature: 13 constants at b0, root SRVs t0 and t1, root UAVs u0 (dst)
// and u1..u5: ggml_d3d12.cpp make_root_sig. Strides and offsets are in floats
// and 32-bit, as the plans check.

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
    uint c8;
    uint c9;
    uint c10;
    uint c11;
    uint c12;
};

ByteAddressBuffer T0 : register(t0);
ByteAddressBuffer T1 : register(t1);
RWByteAddressBuffer D : register(u0);
RWByteAddressBuffer U1 : register(u1);
RWByteAddressBuffer U2 : register(u2);
RWByteAddressBuffer U3 : register(u3);
RWByteAddressBuffer U4 : register(u4);

#if KERNEL == 1
// c0 = ne0, c1 = src row stride, c2 = dst row stride, c3 = op (0 scale,
// 1 sigmoid, 2 softplus, 3 silu), c4 = the scale's bits (op 0). u1 = src
// (may be dst: in place). Grid (ceil(ne0 / 256), rows).
[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint c = gid.x * 256u + tid;
    if (c >= c0)
        return;
    const float x = asfloat(U1.Load(4u * (gid.y * c1 + c)));
    precise float y;
    if (c3 == 0u) {
        y = x * asfloat(c4);
    } else if (c3 == 1u) {
        precise float e = exp(-x);
        precise float den = 1.0f + e;
        y = 1.0f / den;
    } else if (c3 == 2u) {
        precise float e = exp(x);
        precise float arg = 1.0f + e;
        y = x > 20.0f ? x : log(arg);
    } else {
        precise float e = exp(-x);
        precise float den = 1.0f + e;
        y = x / den;
    }
    D.Store(4u * (gid.y * c2 + c), asuint(y));
}
#elif KERNEL == 2
// c0..c3 = ne0..ne3 of the copied block, c4..c7 = its src strides, c8..c11
// = its dst strides (dims 0..3); c12 = 0. u1 = src. Grid (ceil(n / 256)),
// n = ne0 * ne1 * ne2 * ne3.
[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    uint i = gid.x * 256u + tid;
    if (i >= c0 * c1 * c2 * c3)
        return;
    const uint i0 = i % c0;
    i /= c0;
    const uint i1 = i % c1;
    i /= c1;
    const uint i2 = i % c2;
    const uint i3 = i / c2;
    const uint s = i0 * c4 + i1 * c5 + i2 * c6 + i3 * c7;
    const uint d = i0 * c8 + i1 * c9 + i2 * c10 + i3 * c11;
    D.Store(4u * d, U1.Load(4u * s));
}
#elif KERNEL == 3
// c0 = ne0, c1 = src row stride, c2 = dst row stride, c3 = ids stride (in
// int32), c4..c12 = 0. u1 = src, u2 = ids. Grid (ceil(ne0 / 256), rows).
[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint c = gid.x * 256u + tid;
    if (c >= c0)
        return;
    const uint row = U2.Load(4u * (gid.y * c3));
    D.Store(4u * (gid.y * c2 + c), U1.Load(4u * (row * c1 + c)));
}
#elif KERNEL == 4
// c0 = channels (d_inner), c1 = d_conv, c2 = x channel stride, c3 = x
// sequence stride, c4 = w channel stride, c5 = dst token stride, c6 = dst
// sequence stride, c7 = 1 to apply silu; c8..c12 = 0. t0 = w (conv1d
// weight, [d_conv, channels]), u1 = x (conv input, [d_conv - 1 + T,
// channels, seqs]). Grid (ceil(channels / 256), T, seqs).
[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint ch = gid.x * 256u + tid;
    if (ch >= c0)
        return;
    const uint xb = gid.z * c3 + ch * c2 + gid.y;
    const uint wb = ch * c4;
    precise float acc = 0.0f;
    for (uint i = 0u; i < c1; ++i) {
        precise float p = asfloat(U1.Load(4u * (xb + i))) * asfloat(T0.Load(4u * (wb + i)));
        acc = acc + p;
    }
    if (c7 != 0u) {
        precise float e = exp(-acc);
        precise float den = 1.0f + e;
        acc = acc / den;
    }
    D.Store(4u * (gid.z * c6 + gid.y * c5 + ch), asuint(acc));
}
#elif KERNEL == 5
// S_v = 128. A group owns 32 rows j of one (head, sequence) state, kept in
// registers for all tokens: 4 lanes per row, lane l holding the float4
// chunks l, l + 4, ..., l + 28 of row j (elements 4 * (l + 4 * e) + 0..3).
// The state is stored transposed as in ggml: M[j][i] = S[i][j].
// c0 = H (v heads), c1 = T, c2 = B (sequences), c3 = K | kda << 16 |
// fused << 17, c4 = q heads | k heads << 16, c5 = B / q seqs | B / k seqs
// << 16, c6 / c7 = q token / sequence stride, c8 / c9 = k's, c10 / c11 =
// v's (heads are S_v apart in each), c12 = the bits of 1 / sqrt(S_v).
// u1 = q, u2 = k, u3 = v, u4 = g ([1 or S_v, H, T, B]), t0 = beta
// ([1, H, T, B]), t1 = state ([S_v, S_v, H, B]), u0 = dst: the T * B * H
// output rows of S_v, then K snapshots of every state (slot 0 the last).
// Fused (K = 1, B = 1; ggml_d3d12_ssm_emu.h d3d12_gdn_fuse_plan): t1 is the
// recurrent cache row of the state and the new state goes to u5, the cache
// row it is copied to, instead of after the output rows, so build_rs's
// GET_ROWS and the CPY back do not run. A lane loads its 32 floats of the
// state before it stores them and no lane loads another's, so t1 and u5 may
// be the same row.
// Grid (S_v / 32, H, B).
// GDN_LANES (4 built in; 8 or 16 as XLLAMA_D3D12_SHADER_DIR variants) lanes per
// row, each holding GDN_E = 32 / GDN_LANES float4 chunks: the same grid with
// 2x or 4x the threads in flight for the same state bytes. A row's partials
// add as a pairwise tree, so GDN_LANES 4 is the built-in kernel's sum.
#define S_V 128u
#ifndef GDN_LANES
#define GDN_LANES 4
#endif
#define GDN_E (32u / GDN_LANES)
#define GDN_THREADS (32 * GDN_LANES)
#define GDN_SUM4(a, q) ((a[q] + a[q + 1u]) + (a[q + 2u] + a[q + 3u]))
#if GDN_LANES == 4
#define GDN_SUM(a, q) GDN_SUM4(a, q)
#elif GDN_LANES == 8
#define GDN_SUM(a, q) (GDN_SUM4(a, q) + GDN_SUM4(a, q + 4u))
#elif GDN_LANES == 16
#define GDN_SUM(a, q) ((GDN_SUM4(a, q) + GDN_SUM4(a, q + 4u)) + (GDN_SUM4(a, q + 8u) + GDN_SUM4(a, q + 12u)))
#else
#error GDN_LANES must be 4, 8 or 16
#endif
RWByteAddressBuffer U5 : register(u5);
groupshared float red_k[GDN_THREADS];
groupshared float red_y[GDN_THREADS];

[numthreads(GDN_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint H = c0, T = c1, B = c2;
    const uint K = c3 & 0xffffu;
    const bool kda = ((c3 >> 16) & 1u) != 0u;
    const bool fused = ((c3 >> 17) & 1u) != 0u;
    const uint h = gid.y, s = gid.z;
    const uint l = tid & (GDN_LANES - 1u), q4 = tid & ~(GDN_LANES - 1u);
    const uint j = gid.x * 32u + tid / GDN_LANES;
    const uint iq1 = h % (c4 & 0xffffu), ik1 = h % (c4 >> 16);
    const uint iq3 = s / (c5 & 0xffffu), ik3 = s / (c5 >> 16);
    const uint snap = S_V * S_V;
    const uint own = (s * H + h) * snap + j * S_V; // row j of this state in a snapshot
    const uint attn = S_V * H * T * B;
    const uint per_snap = snap * H * B;
    precise float4 m[GDN_E];
    [unroll] for (uint e = 0u; e < GDN_E; ++e)
        m[e] = asfloat(T1.Load4(4u * (own + 4u * (l + GDN_LANES * e))));
    for (uint t = 0u; t < T; ++t) {
        const uint qb = iq3 * c7 + t * c6 + iq1 * S_V;
        const uint kb = ik3 * c9 + t * c8 + ik1 * S_V;
        const uint gb = (s * T + t) * H + h;
        precise float pk = 0.0f;
        if (kda) {
            [unroll] for (uint e = 0u; e < GDN_E; ++e) {
                const uint i = 4u * (l + GDN_LANES * e);
                const float4 ge = exp(asfloat(U4.Load4(4u * (gb * S_V + i))));
                const float4 kx = asfloat(U2.Load4(4u * (kb + i)));
                m[e] = m[e] * ge;
                pk = pk + m[e].x * kx.x;
                pk = pk + m[e].y * kx.y;
                pk = pk + m[e].z * kx.z;
                pk = pk + m[e].w * kx.w;
            }
        } else {
            const float ge = exp(asfloat(U4.Load(4u * gb)));
            [unroll] for (uint e = 0u; e < GDN_E; ++e) {
                const float4 kx = asfloat(U2.Load4(4u * (kb + 4u * (l + GDN_LANES * e))));
                m[e] = m[e] * ge;
                pk = pk + m[e].x * kx.x;
                pk = pk + m[e].y * kx.y;
                pk = pk + m[e].z * kx.z;
                pk = pk + m[e].w * kx.w;
            }
        }
        red_k[tid] = pk;
        GroupMemoryBarrierWithGroupSync();
        precise float sk = GDN_SUM(red_k, q4);
        precise float dv = asfloat(U3.Load(4u * (s * c11 + t * c10 + h * S_V + j))) - sk;
        precise float d = dv * asfloat(T0.Load(4u * gb));
        precise float py = 0.0f;
        [unroll] for (uint e = 0u; e < GDN_E; ++e) {
            const uint i = 4u * (l + GDN_LANES * e);
            const float4 kx = asfloat(U2.Load4(4u * (kb + i)));
            const float4 qx = asfloat(U1.Load4(4u * (qb + i)));
            precise float4 kd = kx * d;
            m[e] = m[e] + kd;
            py = py + m[e].x * qx.x;
            py = py + m[e].y * qx.y;
            py = py + m[e].z * qx.z;
            py = py + m[e].w * qx.w;
        }
        red_y[tid] = py;
        GroupMemoryBarrierWithGroupSync();
        precise float y = GDN_SUM(red_y, q4);
        if (l == 0u) {
            precise float ys = y * asfloat(c12);
            D.Store(4u * (gb * S_V + j), asuint(ys));
        }
        const uint slot = T - 1u - t;
        if (K > 1u && slot < K) {
            const uint ob = attn + slot * per_snap + own;
            [unroll] for (uint e = 0u; e < GDN_E; ++e)
                D.Store4(4u * (ob + 4u * (l + GDN_LANES * e)), asuint(m[e]));
        }
    }
    if (fused) {
        [unroll] for (uint e = 0u; e < GDN_E; ++e)
            U5.Store4(4u * (own + 4u * (l + GDN_LANES * e)), asuint(m[e]));
    } else if (K == 1u) {
        [unroll] for (uint e = 0u; e < GDN_E; ++e)
            D.Store4(4u * (attn + own + 4u * (l + GDN_LANES * e)), asuint(m[e]));
    }
}
#endif
