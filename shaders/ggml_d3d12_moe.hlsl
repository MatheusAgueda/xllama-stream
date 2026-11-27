// ggml backend d3d12: the tail of a MoE layer, so the routed experts, the
// shared expert and the next layer's input projections run as one submission
// instead of three GPU splits with two CPU splits between them
// (src/bridge/ggml_d3d12_moe_emu.h matches the nodes and emulates both kernels).
// One source, two kernels picked with -D KERNEL=n:
//   1 combine: dst[t][c] = E[t][0][c] * w[t][0] + ... + E[t][n-1][c] * w[t][n-1]
//              left to right: llama-graph.cpp build_moe_ffn's MUL of the
//              MUL_MAT_ID output by the expert weights and the n - 1 ADDs of
//              its views.
//   2 tail:    mode 0: dst[r][c] = s[r][c] * g[r]
//              mode 1: dst[r][c] = m[r][c] + s[r][c] * g[r]
//              mode 2: dst[r][c] = (m[r][c] + s[r][c] * g[r]) + q[r][c]
//              qwen35moe.cpp build_layer_ffn's shared expert by its sigmoid
//              gate (a MUL broadcast over ne0), the ADD to the routed experts
//              and the residual ADD.
// Each float operation is one of the CPU's, rounded once, in the CPU's order;
// `precise` keeps dxc from fusing a multiply and an add into a mad, so the
// results equal ggml's CPU ops bit for bit (denormals aside).
// Root signature: 8 constants at b0, root UAVs u0 (dst) and u1..u4, all
// D3D12_Host raw buffers: ggml_d3d12_elt.hlsl's, plus u4.

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
RWByteAddressBuffer U1 : register(u1);
RWByteAddressBuffer U2 : register(u2);
RWByteAddressBuffer U3 : register(u3);
RWByteAddressBuffer U4 : register(u4);

#if KERNEL == 1
// c0 = ne0, c1 = n (experts per token, >= 1), c2 = E expert stride, c3 = E
// token stride, c4 = w expert stride, c5 = w token stride, c6 = dst token
// stride, all in floats; c7 = 0. u1 = E, u2 = w. Grid (ceil(ne0 / 256), T).
[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint c = gid.x * 256u + tid;
    if (c >= c0)
        return;
    const uint e = gid.y * c3 + c;
    const uint w = gid.y * c5;
    precise float acc = asfloat(U1.Load(4u * e)) * asfloat(U2.Load(4u * w));
    for (uint i = 1u; i < c1; ++i) {
        precise float p = asfloat(U1.Load(4u * (e + i * c2))) * asfloat(U2.Load(4u * (w + i * c4)));
        acc = acc + p;
    }
    D.Store(4u * (gid.y * c6 + c), asuint(acc));
}
#elif KERNEL == 2
// c0 = ne0, c1 = s row stride, c2 = g row stride, c3 = m row stride, c4 = q
// row stride, c5 = dst row stride, all in floats; c6 = mode (0, 1, 2); c7 = 0.
// u1 = s, u2 = g (one float per row), u3 = m, u4 = q; the ones a mode does
// not read are never loaded (bind dst there). Grid (ceil(ne0 / 256), rows).
[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint c = gid.x * 256u + tid;
    if (c >= c0)
        return;
    const uint r = gid.y;
    precise float y = asfloat(U1.Load(4u * (r * c1 + c))) * asfloat(U2.Load(4u * (r * c2)));
    if (c6 >= 1u)
        y = asfloat(U3.Load(4u * (r * c3 + c))) + y;
    if (c6 >= 2u)
        y = y + asfloat(U4.Load(4u * (r * c4 + c)));
    D.Store(4u * (r * c5 + c), asuint(y));
}
#endif
