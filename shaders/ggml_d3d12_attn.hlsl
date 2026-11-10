// ggml backend d3d12: the full-attention layers of qwen35moe, so rope, the
// kv cache writes and flash attention run on the GPU instead of a CPU split
// per layer (src/bridge/ggml_d3d12_attn_emu.h matches the nodes and emulates
// each kernel). One source, four kernels picked with -D KERNEL=n:
//   1 rope:      ROPE, NORMAL / NEOX / MROPE / IMROPE, n_offs, freq factors,
//                YaRN; f32
//   2 set_rows:  SET_ROWS of f32 rows into an f16 or f32 cache, i32 or i64 rows
//   3 fa:        FLASH_ATTN_EXT, f32 q, f16 k and v, f16 mask, head size up to
//                256, 1..16 q heads per kv head; one chunk of the kv range per
//                group, written out (one chunk) or kept as a partial
//   4 fa_reduce: merges the partials of KERNEL 3 into the output
// Rope and set_rows round each operation once, in the emulation's order
// (`precise` keeps dxc from forming mads; sin and cos are computed here, not
// by the GPU's approximations), so they match the emulation bit for bit except
// for a division (freq factors, YaRN ramp). Flash attention leaves dxc free
// in its loops and uses the GPU's exp: it matches within a tolerance.
// Root signature: 13 constants at b0, root SRVs t0 and t1, root UAVs u0 (dst)
// and u1..u4: ggml_d3d12.cpp make_root_sig. Offsets are 32-bit, as the plans
// check.

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

#define NEG_INF_BITS 0xff800000u

// -inf compared by its bits, safe from any fast-math assumption.
bool is_neg_inf(float x) {
    return asuint(x) == NEG_INF_BITS;
}

#if KERNEL == 1
// sin and cos of th: a Cody-Waite reduction by pi/4 (pi/4 in four 6-bit
// chunks and a tail), then the Cephes polynomials. Within 8e-8 of the true
// values for |th| up to 411000 (positions up to 4e5 at the first pair).
void attn_sincos(float th, out float sn, out float cs) {
    precise float x = abs(th);
    precise float fx = x * asfloat(0x3fa2f983u); // 4 / pi
    uint j = (uint)fx;
    precise float y = (float)j;
    if ((j & 1u) != 0u) {
        j += 1u;
        y = y + 1.0f;
    }
    j &= 7u;
    precise float r = x;
    precise float p0 = y * asfloat(0x3f480000u);
    r = r - p0;
    precise float p1 = y * asfloat(0x3b840000u);
    r = r - p1;
    precise float p2 = y * asfloat(0x38f80000u);
    r = r - p2;
    precise float p3 = y * asfloat(0x35d40000u);
    r = r - p3;
    precise float p4 = y * asfloat(0x320885a3u);
    r = r - p4;
    precise float z = r * r;
    precise float t = asfloat(0xb94ca1f9u) * z;
    t = t + asfloat(0x3c08839eu);
    t = t * z;
    t = t + asfloat(0xbe2aaaa3u);
    t = t * z;
    t = t * r;
    precise float ps = t + r;
    precise float u = asfloat(0x37ccf5ceu) * z;
    u = u + asfloat(0xbab6061au);
    u = u * z;
    u = u + asfloat(0x3d2aaaa5u);
    u = u * z;
    u = u * z;
    precise float h = 0.5f * z;
    precise float uh = u - h;
    precise float pc = uh + 1.0f;
    if (j == 0u) {
        sn = ps;
        cs = pc;
    } else if (j == 2u) {
        sn = pc;
        cs = -ps;
    } else if (j == 4u) {
        sn = -ps;
        cs = -pc;
    } else {
        sn = -pc;
        cs = ps;
    }
    if (th < 0.0f)
        sn = -sn;
}

// One thread per pair of a row: pairs 0 .. n_dims / 2 - 1 rotate, the others
// copy the channels outside [n_offs, n_offs + n_dims).
// c0 = ne0 | n_dims << 16, c1 = n_offs | mode << 16 | freq factors << 24
// (mode 0 normal, 1 neox, 2 mrope, 3 imrope), c2 = ne1 | ne2 << 16, c3 = the
// four mrope sections, a byte each, c4 / c5 = src row / plane stride, c6 / c7
// = dst's (floats), c8 = theta_scale, c9 = freq_scale, c10 = mscale (attn
// factor, with YaRN's magnitude scale when ext != 0), c11 = ext_factor (all
// float bits), c12 = YaRN corr dims: low | high << 16, high an int16 with
// -32768 for -inf. u1 = src (may be dst), u2 = positions (i32; 4 per token for
// mrope: t, h, w, e), t0 = freq factors (unread without them). Grid
// (ceil(ne0 / 2 * ne1 * ne2 / 256)).
[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint ne0 = c0 & 0xffffu, n_dims = c0 >> 16;
    const uint n_offs = c1 & 0xffffu, mode = (c1 >> 16) & 0xffu;
    const uint ne1 = c2 & 0xffffu, ne2 = c2 >> 16;
    const uint half_ne0 = ne0 >> 1, half_dims = n_dims >> 1;
    uint i = gid.x * 256u + tid;
    if (i >= half_ne0 * ne1 * ne2)
        return;
    const uint r = i % half_ne0;
    i /= half_ne0;
    const uint i1 = i % ne1;
    const uint i2 = i / ne1;
    const uint sb = i2 * c5 + i1 * c4;
    const uint db = i2 * c7 + i1 * c6;
    if (r >= half_dims) {
        const uint e = 2u * (r - half_dims);
        const uint i0 = e < n_offs ? e : e + n_dims;
        D.Store2(4u * (db + i0), U1.Load2(4u * (sb + i0)));
        return;
    }
    uint pi = i2;
    if (mode >= 2u) {
        const uint s0 = c3 & 0xffu, s1 = (c3 >> 8) & 0xffu, s2 = (c3 >> 16) & 0xffu;
        const uint sector = r % (s0 + s1 + s2 + (c3 >> 24));
        uint k = 0u; // 0 t, 1 h, 2 w, 3 e
        if (mode == 3u) {
            if (sector % 3u == 1u && sector < 3u * s1)
                k = 1u;
            else if (sector % 3u == 2u && sector < 3u * s2)
                k = 2u;
            else if (sector % 3u == 0u && sector < 3u * s0)
                k = 0u;
            else
                k = 3u;
        } else {
            const uint sec_w = s0 + s1;
            if (sector >= s0 && sector < sec_w)
                k = 1u;
            else if (sector >= sec_w && sector < sec_w + s2)
                k = 2u;
            else if (sector >= sec_w + s2)
                k = 3u;
        }
        pi = i2 + k * ne2;
    }
    precise float theta = (float)asint(U2.Load(4u * pi));
    const float ts = asfloat(c8);
    for (uint m = 0u; m < r; ++m)
        theta = theta * ts;
    if ((c1 >> 24) != 0u)
        theta = theta / asfloat(T0.Load(4u * r));
    precise float interp = asfloat(c9) * theta;
    precise float th = interp;
    const float ext = asfloat(c11);
    if (ext != 0.0f) {
        const float low = (float)(c12 & 0xffffu);
        const int hi = asint(c12) >> 16;
        const float high = hi == -32768 ? asfloat(NEG_INF_BITS) : (float)hi;
        precise float span = high - low;
        precise float den = max(0.001f, span);
        precise float num = (float)r - low;
        precise float y = num / den;
        precise float ramp = 1.0f - min(1.0f, max(0.0f, y));
        precise float mix = ramp * ext;
        precise float keep = 1.0f - mix;
        precise float a = interp * keep;
        precise float b = theta * mix;
        th = a + b;
    }
    float sn, cs;
    attn_sincos(th, sn, cs);
    precise float c = cs * asfloat(c10);
    precise float s = sn * asfloat(c10);
    uint x0i, x1i;
    if (mode == 0u) {
        x0i = n_offs + 2u * r;
        x1i = x0i + 1u;
    } else {
        x0i = n_offs + r;
        x1i = x0i + half_dims;
    }
    const float x0 = asfloat(U1.Load(4u * (sb + x0i)));
    const float x1 = asfloat(U1.Load(4u * (sb + x1i)));
    precise float o0 = x0 * c - x1 * s;
    precise float o1 = x0 * s + x1 * c;
    D.Store(4u * (db + x0i), asuint(o0));
    D.Store(4u * (db + x1i), asuint(o1));
}
#elif KERNEL == 2
// f32 to f16, round to nearest even, as ggml's CPU conversion (F16C).
uint f32_to_f16(float v) {
    uint x = asuint(v);
    const uint sign = x & 0x80000000u;
    x ^= sign;
    uint o;
    if (x >= 0x47800000u) {
        o = x > 0x7f800000u ? 0x7e00u : 0x7c00u;
    } else if (x < 0x38800000u) {
        precise float f = asfloat(x) + 0.5f; // the f16 subnormal in the low bits
        o = asuint(f) - 0x3f000000u;
    } else {
        const uint odd = (x >> 13) & 1u;
        x += 0xc8000fffu;
        x += odd;
        o = x >> 13;
    }
    return o | (sign >> 16);
}

// dst[idx[r]] = src[r]. c0 = ne0, c1 = src row stride (floats), c2 = dst row
// stride (elements), c3 = idx stride (32-bit words), c4 = flags (1 f16 dst,
// 2 i64 idx), c5 = dst rows (an index at or past it is skipped, as is an i64
// one with a high word), c6 = rows. u1 = src, u2 = idx. f16: two elements per
// thread, ne0 and the dst stride even. Grid (ceil(ne0 / (256 * per)), rows).
[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const bool f16 = (c4 & 1u) != 0u;
    const uint ib = gid.y * c3;
    const uint di = U2.Load(4u * ib);
    if (di >= c5)
        return;
    if ((c4 & 2u) != 0u && U2.Load(4u * (ib + 1u)) != 0u)
        return;
    const uint sb = gid.y * c1, db = di * c2;
    if (f16) {
        const uint c = 2u * (gid.x * 256u + tid);
        if (c >= c0)
            return;
        const uint2 x = U1.Load2(4u * (sb + c));
        D.Store(2u * (db + c), f32_to_f16(asfloat(x.x)) | (f32_to_f16(asfloat(x.y)) << 16));
    } else {
        const uint c = gid.x * 256u + tid;
        if (c >= c0)
            return;
        D.Store(4u * (db + c), U1.Load(4u * (sb + c)));
    }
}
#elif KERNEL == 3
// A group owns one token, one kv head and its G q heads, and one chunk of
// the kv positions, walked in tiles of P = 256 / G positions. Thread
// t = p * G + g scores position p of the tile for head g (K row by Load4, 8
// halves at a time, element e summed into a[e % 4]); a softmax step per head
// (running max M, sum S); then thread d < head size adds the tile's weighted
// V column d into its accumulator for each head.
// c0 = head size | G << 16, c1 = n_kv, c2 = N | n_chunks << 16, c3 = chunk
// length (positions, a multiple of P), c4 / c5 = q token / head stride
// (floats), c6 / c7 = k cell / head stride (32-bit words), c8 / c9 = v's,
// c10 = mask row stride (halves), c11 = scale bits, c12 = n_head | 1 << 16
// (mask) | 1 << 17 (direct). u1 = q, t0 = k, t1 = v, u2 = mask, u3 =
// partials ((token * n_head + head) * n_chunks + chunk) * (head size + 4):
// M, S, 0, 0, then the unnormalized output; u0 = dst [head size, n_head, N]
// (direct). Grid (n_chunks, kv heads, N).
#define FA_MAX_G 16u
#define FA_MAX_D 256u
groupshared float qs[FA_MAX_G * (FA_MAX_D + 4u)];
groupshared float sc[256u + FA_MAX_G];
groupshared float wt[256u + FA_MAX_G];
groupshared float msg[FA_MAX_G];
groupshared float s_sum[FA_MAX_G];

float half_at(uint word, uint i) {
    return f16tof32((i & 1u) != 0u ? word >> 16 : word & 0xffffu);
}

[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint dk = c0 & 0xffffu, G = c0 >> 16;
    const uint P = 256u / G;
    const uint QS = dk + 4u; // qs row: 4 floats of padding against bank conflicts
    const uint SP = P + 1u;  // sc / wt row
    const uint n_chunks = c2 >> 16;
    const uint n_head = c12 & 0xffffu;
    const bool has_mask = (c12 & 0x10000u) != 0u;
    const uint chunk = gid.x, hkv = gid.y, iq1 = gid.z;
    const uint g = tid % G, p = tid / G;
    const uint begin = chunk * c3;
    const uint end = min(begin + c3, c1);
    const float scale = asfloat(c11);

    for (uint qi = tid; qi < G * dk; qi += 256u) {
        const uint qg = qi / dk, e = qi % dk;
        qs[qg * QS + e] = asfloat(U1.Load(4u * (iq1 * c4 + (hkv * G + qg) * c5 + e)));
    }
    GroupMemoryBarrierWithGroupSync();

    float M = asfloat(NEG_INF_BITS); // of head g
    float S = 0.0f;                  // of head tid, threads tid < G
    float acc[FA_MAX_G];
    [unroll] for (uint ai = 0u; ai < FA_MAX_G; ++ai)
        acc[ai] = 0.0f;

    for (uint tile = begin; tile < end; tile += P) {
        const uint np = min(P, end - tile);
        float s = asfloat(NEG_INF_BITS);
        if (p < np) {
            const uint pos = tile + p;
            float m = 0.0f;
            if (has_mask) {
                const uint mi = iq1 * c10 + pos;
                m = half_at(U2.Load(4u * (mi >> 1)), mi);
            }
            if (!is_neg_inf(m)) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                const uint kb = 4u * (pos * c6 + hkv * c7);
                const uint qb = g * QS;
                for (uint e = 0u; e < dk; e += 8u) {
                    const uint4 kw = T0.Load4(kb + 2u * e);
                    a0 += f16tof32(kw.x & 0xffffu) * qs[qb + e];
                    a1 += f16tof32(kw.x >> 16) * qs[qb + e + 1u];
                    a2 += f16tof32(kw.y & 0xffffu) * qs[qb + e + 2u];
                    a3 += f16tof32(kw.y >> 16) * qs[qb + e + 3u];
                    a0 += f16tof32(kw.z & 0xffffu) * qs[qb + e + 4u];
                    a1 += f16tof32(kw.z >> 16) * qs[qb + e + 5u];
                    a2 += f16tof32(kw.w & 0xffffu) * qs[qb + e + 6u];
                    a3 += f16tof32(kw.w >> 16) * qs[qb + e + 7u];
                }
                s = ((a0 + a1) + (a2 + a3)) * scale;
                s = s + m;
            }
        }
        sc[g * SP + p] = s;
        GroupMemoryBarrierWithGroupSync();

        float mx = asfloat(NEG_INF_BITS);
        for (uint i = 0u; i < np; ++i)
            mx = max(mx, sc[g * SP + i]);
        const float m_new = max(M, mx);
        float ms = 1.0f;
        if (asuint(m_new) != asuint(M))
            ms = is_neg_inf(M) ? 0.0f : exp(M - m_new);
        wt[g * SP + p] = is_neg_inf(s) ? 0.0f : exp(s - m_new);
        if (p == 0u)
            msg[g] = ms;
        M = m_new;
        GroupMemoryBarrierWithGroupSync();

        if (tid < dk) {
            [unroll] for (uint a = 0u; a < FA_MAX_G; ++a)
                if (a < G)
                    acc[a] *= msg[a];
            for (uint i = 0u; i < np; ++i) {
                const uint vi = (tile + i) * c8 + hkv * c9 + (tid >> 1);
                const float v = half_at(T1.Load(4u * vi), tid);
                [unroll] for (uint a = 0u; a < FA_MAX_G; ++a)
                    if (a < G) {
                        const float w = wt[a * SP + i];
                        if (w != 0.0f)
                            acc[a] += w * v;
                    }
            }
        }
        if (tid < G) {
            S *= msg[tid];
            for (uint i = 0u; i < np; ++i) {
                const float w = wt[tid * SP + i];
                if (w != 0.0f)
                    S += w;
            }
        }
    }

    if (tid < G)
        s_sum[tid] = S;
    GroupMemoryBarrierWithGroupSync();
    if ((c12 & 0x20000u) != 0u) {
        if (tid < dk) {
            [unroll] for (uint a = 0u; a < FA_MAX_G; ++a)
                if (a < G) {
                    const float sa = s_sum[a];
                    const float o = sa == 0.0f ? 0.0f : acc[a] * (1.0f / sa);
                    D.Store(4u * ((iq1 * n_head + hkv * G + a) * dk + tid), asuint(o));
                }
        }
    } else {
        if (tid < dk) {
            [unroll] for (uint a = 0u; a < FA_MAX_G; ++a)
                if (a < G) {
                    const uint pb = ((iq1 * n_head + hkv * G + a) * n_chunks + chunk) * (dk + 4u);
                    U3.Store(4u * (pb + 4u + tid), asuint(acc[a]));
                }
        }
        if (tid < G) {
            const uint pb = ((iq1 * n_head + hkv * G + tid) * n_chunks + chunk) * (dk + 4u);
            U3.Store4(4u * pb, uint4(asuint(M), asuint(S), 0u, 0u));
        }
    }
}
#elif KERNEL == 4
// The chunks of one (token, head) merged in chunk order; a chunk with S = 0
// (every position masked) is skipped. c0 = head size, c1 = n_chunks, c2 =
// n_head. u1 = partials (KERNEL 3's u3). Grid (n_head, N).
[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint dk = c0;
    if (tid >= dk)
        return;
    const uint row = gid.y * c2 + gid.x;
    float mf = asfloat(NEG_INF_BITS);
    float S = 0.0f, acc = 0.0f;
    for (uint c = 0u; c < c1; ++c) {
        const uint pb = (row * c1 + c) * (dk + 4u);
        const float sc_c = asfloat(U1.Load(4u * (pb + 1u)));
        if (sc_c == 0.0f)
            continue;
        const float mc = asfloat(U1.Load(4u * pb));
        const float mn = max(mf, mc);
        const float so = is_neg_inf(mf) ? 0.0f : exp(mf - mn);
        const float sn = exp(mc - mn);
        acc = acc * so + asfloat(U1.Load(4u * (pb + 4u + tid))) * sn;
        S = S * so + sc_c * sn;
        mf = mn;
    }
    const float o = S == 0.0f ? 0.0f : acc * (1.0f / S);
    D.Store(4u * (row * dk + tid), asuint(o));
}
#endif
