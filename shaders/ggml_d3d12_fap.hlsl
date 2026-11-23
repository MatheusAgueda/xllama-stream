// ggml backend d3d12: FLASH_ATTN_EXT (src/bridge/ggml_d3d12_fap_emu.h matches the node and
// emulates the kernel). A prefill ubatch reads the KV cache once per group of 16 q rows
// instead of once per token. A decode token (or a few) has a group per kv head only, so
// the positions split into chunks (split-K, grid z, c11 & 2): each (group, chunk) writes
// its rows' M, S and unnormalised O to u3, and a second dispatch (c11 & 4) combines them.
//
// A group owns 16 q rows: 16 / G tokens of one kv head, each with that head's G q heads
// (row r = token * G + head). It walks the kv range in tiles of 16 positions:
//   A  thread (r, l) reads the mask of row r at position l of the tile; a tile every
//      row sees as -inf is skipped by the whole group (a flag in groupshared, three of
//      them in rotation so a skip needs one barrier only)
//   B  the K tile, f16, into groupshared (rows padded to D / 2 + 1 words)
//   C  thread (r, l) scores row r against position l: q . k * scale + mask
//   D  per row: running max M, the tile's weights p = exp(s - M), the rescale of the
//      earlier tiles; the V tile overwrites the K tile
//   E  per row: running sum S, and O[r][dims] = O * rescale + sum_l p[r][l] * V[l][dims],
//      thread (r, l) owning dim pairs l, l + 16, l + 32, ... of row r
// then dst = O / S (0 where a row saw nothing). Accumulation is f32 throughout, as the
// CPU's tiled kernel; exp is the GPU's, so it matches the emulation within a tolerance.
//
// c0 = D | G << 16, c1 = n_kv, c2 = N | n_head << 16, c3 / c4 = q token / head stride
// (floats), c5 / c6 = k cell / head stride (32-bit words), c7 / c8 = v's, c9 = mask row
// stride (halves), c10 = scale bits, c11 = 1 with a mask | 2 split | 4 reduce, c12 = the
// chunk (positions, a multiple of 16; split and reduce). u1 = q (f32), t0 = k, t1 = v
// (f16), u2 = mask (f16), u0 = dst [D, n_head, N] (f32), u3 = partials: record
// ((token * n_head + head) * chunks + chunk) of D + 4 floats, M, S, 2 pad, O[D].
// Grid (ceil(N * G / 16), kv heads[, chunks]); reduce (n_head, N).
// D a multiple of 4 up to 256, G in 1, 2, 4, 8, 16. Root signature: ggml_d3d12.cpp
// make_root_sig (13 constants, root SRVs t0 t1, root UAVs u0..u5).

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

#define NEG_INF_BITS 0xff800000u
#define FAP_ROWS 16u
#define FAP_TILE 16u
#define FAP_MAX_D 256u
#define FAP_KW (FAP_MAX_D / 2u + 1u) // words per K / V row in groupshared, padded
#define FAP_QS (FAP_MAX_D + 4u)       // floats per q row: the 4 rows of a wave on 4 banks apart

groupshared float qs[FAP_ROWS * FAP_QS];
groupshared uint kv[FAP_TILE * FAP_KW];
groupshared float sc[FAP_ROWS * (FAP_TILE + 1u)];
groupshared float pw[FAP_ROWS * (FAP_TILE + 1u)];
groupshared uint any_seen[3];

bool is_neg_inf(float x) {
    return asuint(x) == NEG_INF_BITS;
}

float half_lo(uint w) {
    return f16tof32(w & 0xffffu);
}

float half_hi(uint w) {
    return f16tof32(w >> 16);
}

[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint dk = c0 & 0xffffu, G = c0 >> 16;
    const uint n_kv = c1;
    const uint N = c2 & 0xffffu, n_head = c2 >> 16;
    const uint rec = dk + 4u;

    if ((c11 & 4u) != 0u) {
        // reduce: group (head x, token y), thread e < D owns dim e; the chunks' max, then
        // in chunk order w = exp(M_z - max) (0 for a chunk the row never saw), S and O
        const uint chunks = (n_kv + c12 - 1u) / c12;
        const uint r0 = (gid.y * n_head + gid.x) * chunks * rec;
        float mx = asfloat(NEG_INF_BITS);
        for (uint z = 0u; z < chunks; ++z)
            mx = max(mx, asfloat(U3.Load(4u * (r0 + z * rec))));
        float sum = 0.0f, o = 0.0f;
        for (uint z = 0u; z < chunks; ++z) {
            const uint rb = r0 + z * rec;
            const float mz = asfloat(U3.Load(4u * rb));
            if (is_neg_inf(mz))
                continue;
            const float w = exp(mz - mx);
            sum += w * asfloat(U3.Load(4u * (rb + 1u)));
            if (tid < dk)
                o += w * asfloat(U3.Load(4u * (rb + 4u + tid)));
        }
        if (tid < dk) {
            const float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
            D.Store(4u * ((gid.y * n_head + gid.x) * dk + tid), asuint(o * inv));
        }
        return; // uniform (a constant): no barrier below is skipped by part of the group
    }

    const bool split = (c11 & 2u) != 0u;
    const uint pb = split ? gid.z * c12 : 0u;
    const uint pe = split ? min(n_kv, pb + c12) : n_kv;
    const uint half_d = dk >> 1;
    const uint BT = FAP_ROWS / G; // tokens per group
    const uint r = tid >> 4, l = tid & 15u;
    const uint tok = gid.x * BT + r / G;
    const uint head = gid.y * G + (r % G);
    const bool live = tok < N;
    const bool has_mask = (c11 & 1u) != 0u;
    const float scale = asfloat(c10);

    // q rows into groupshared (zeros for rows past the last token)
    for (uint i = tid; i < FAP_ROWS * dk; i += 256u) {
        const uint qr = i / dk, e = i % dk;
        const uint qt = gid.x * BT + qr / G;
        const uint qh = gid.y * G + (qr % G);
        qs[qr * FAP_QS + e] = qt < N ? asfloat(U1.Load(4u * (qt * c3 + qh * c4 + e))) : 0.0f;
    }
    if (tid < 3u)
        any_seen[tid] = 0u;
    GroupMemoryBarrierWithGroupSync();

    float M = asfloat(NEG_INF_BITS);
    float S = 0.0f;
    float acc[FAP_MAX_D / 32u][2];
    [unroll] for (uint j = 0u; j < FAP_MAX_D / 32u; ++j) {
        acc[j][0] = 0.0f;
        acc[j][1] = 0.0f;
    }

    uint it = 0u;
    for (uint pos0 = pb; pos0 < pe; pos0 += FAP_TILE, ++it) {
        const uint np = min(FAP_TILE, pe - pos0);
        const uint f = it % 3u;
        // A: this thread's mask entry; the flag two tiles ahead was last read before the
        // previous barrier, so clearing it here races with nobody
        float m = asfloat(NEG_INF_BITS);
        if (live && l < np) {
            m = 0.0f;
            if (has_mask) {
                const uint mi = tok * c9 + pos0 + l;
                const uint w = U2.Load(4u * (mi >> 1));
                m = (mi & 1u) != 0u ? half_hi(w) : half_lo(w);
            }
        }
        if (!is_neg_inf(m))
            any_seen[f] = 1u;
        if (tid == 0u)
            any_seen[(it + 1u) % 3u] = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (any_seen[f] == 0u)
            continue; // uniform: every thread read the same flag after the barrier

        // B: K tile (positions past np as zeros)
        for (uint w = tid; w < FAP_TILE * half_d; w += 256u) {
            const uint p = w / half_d, e = w % half_d;
            kv[p * FAP_KW + e] = p < np ? T0.Load(4u * ((pos0 + p) * c5 + gid.y * c6 + e)) : 0u;
        }
        GroupMemoryBarrierWithGroupSync();

        // C: score of (r, l)
        float s = asfloat(NEG_INF_BITS);
        if (!is_neg_inf(m)) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            const uint qb = r * FAP_QS, kb = l * FAP_KW;
            for (uint e = 0u; e < half_d; e += 2u) {
                const uint k0 = kv[kb + e], k1 = kv[kb + e + 1u];
                a0 += qs[qb + 2u * e] * half_lo(k0);
                a1 += qs[qb + 2u * e + 1u] * half_hi(k0);
                a2 += qs[qb + 2u * e + 2u] * half_lo(k1);
                a3 += qs[qb + 2u * e + 3u] * half_hi(k1);
            }
            s = ((a0 + a1) + (a2 + a3)) * scale + m;
        }
        sc[r * (FAP_TILE + 1u) + l] = s;
        GroupMemoryBarrierWithGroupSync();

        // D: row max, weights, rescale; V tile over the K tile
        float mx = asfloat(NEG_INF_BITS);
        [unroll] for (uint i = 0u; i < FAP_TILE; ++i)
            mx = max(mx, sc[r * (FAP_TILE + 1u) + i]);
        const float m_new = max(M, mx);
        float alpha = 1.0f;
        if (asuint(m_new) != asuint(M))
            alpha = is_neg_inf(M) ? 0.0f : exp(M - m_new);
        pw[r * (FAP_TILE + 1u) + l] = is_neg_inf(s) ? 0.0f : exp(s - m_new);
        M = m_new;
        for (uint w = tid; w < FAP_TILE * half_d; w += 256u) {
            const uint p = w / half_d, e = w % half_d;
            kv[p * FAP_KW + e] = p < np ? T1.Load(4u * ((pos0 + p) * c7 + gid.y * c8 + e)) : 0u;
        }
        GroupMemoryBarrierWithGroupSync();

        // E: running sum and the weighted V rows
        float ps = 0.0f;
        [unroll] for (uint i = 0u; i < FAP_TILE; ++i)
            ps += pw[r * (FAP_TILE + 1u) + i];
        S = S * alpha + ps;
        [unroll] for (uint j = 0u; j < FAP_MAX_D / 32u; ++j) {
            acc[j][0] *= alpha;
            acc[j][1] *= alpha;
        }
        for (uint i = 0u; i < np; ++i) {
            const float wgt = pw[r * (FAP_TILE + 1u) + i];
            if (wgt != 0.0f) {
                [unroll] for (uint j = 0u; j < FAP_MAX_D / 32u; ++j) {
                    const uint e = l + 16u * j;
                    if (e < half_d) {
                        const uint vw = kv[i * FAP_KW + e];
                        acc[j][0] += wgt * half_lo(vw);
                        acc[j][1] += wgt * half_hi(vw);
                    }
                }
            }
        }
        // the next tile's barrier (A) orders these reads before its K tile overwrites kv
    }

    if (live && split) {
        const uint rb = ((tok * n_head + head) * ((n_kv + c12 - 1u) / c12) + gid.z) * rec;
        if (l == 0u)
            U3.Store2(4u * rb, uint2(asuint(M), asuint(S)));
        [unroll] for (uint j = 0u; j < FAP_MAX_D / 32u; ++j) {
            const uint e = l + 16u * j;
            if (e < half_d) {
                U3.Store2(4u * (rb + 4u + 2u * e), uint2(asuint(acc[j][0]), asuint(acc[j][1])));
            }
        }
    } else if (live) {
        const float inv = S == 0.0f ? 0.0f : 1.0f / S;
        const uint ob = (tok * n_head + head) * dk;
        [unroll] for (uint j = 0u; j < FAP_MAX_D / 32u; ++j) {
            const uint e = l + 16u * j;
            if (e < half_d) {
                D.Store2(4u * (ob + 2u * e), uint2(asuint(acc[j][0] * inv), asuint(acc[j][1] * inv)));
            }
        }
    }
}
