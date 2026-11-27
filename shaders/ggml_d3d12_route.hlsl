// ggml backend d3d12: a MoE layer's routing for one token inside the GPU
// submission, so the router softmax, the cache-aware top-k (llama-expert-
// cache.cpp ec_select + ec_swap_missing), the weight normalisation and, when
// every chosen expert is resident, the slot plan (ec_plan's out = slot_of[id])
// no longer need a CPU split between two GPU splits. A layer whose top-k has a
// missing expert writes the predicate, so every later dispatch of the command
// list is skipped (D3D12 predication, NOT_EQUAL_ZERO) and the host reads the
// expert in and resubmits from that layer's emit (src/bridge/ggml_d3d12.cpp
// submit_routed).
// One source, three kernels picked with -D KERNEL=n:
//   1 route: one group per layer, at the router softmax's place.
//   2 emit:  the record's weights and slots into the tensors the experts'
//            MUL_MAT_ID and combine read, at the expert load's place.
//   3 hs:    selftest probe only (XLLAMA_D3D12_HSPROBE): a word handed GPU ->
//            CPU and back while the submission still runs.
// Bit for bit ggml-cpu's AVX2 + FMA path, the xllama build's ISA
// (CMakeLists.txt GGML_AVX2/GGML_FMA): ops.cpp soft_max_f32 (vec.cpp
// ggml_vec_soft_max_f32 with vec.h ggml_v_expf, the sum in double eight lanes
// at a time), get_rows, sum_rows (ggml_vec_sum_f32, double), clamp
// (MAX(MIN(x, max), min)), binary-ops.cpp div and scale. Every float result is
// computed in double and rounded to float by hand (d2f): exact for + - * /
// (53 >= 2 * 24 + 2 bits) and, rounded to odd first, for the fused
// multiply-adds; nothing depends on the GPU's fp32 rounding or denormal mode.
// Floats travel as their bits. Needs double and int64 shader ops (the host
// checks both) and dxc -Gis.
// Root signature: 11 of the 13 constants at b0, root UAVs u0 (record / dst)
// and u1 to u5, D3D12_Host raw buffers.

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
};

#if KERNEL == 3
globallycoherent RWByteAddressBuffer D : register(u0);
#else
RWByteAddressBuffer D : register(u0);
#endif
RWByteAddressBuffer U1 : register(u1);
RWByteAddressBuffer U2 : register(u2);

// A layer's record (kRoute* in ggml_d3d12.cpp): probs [256] floats, the host's
// copy of the layer's slot_of [256] (>= 0 resident), ids / slots / weights
// [16] each, then the state (0 not run, 1 every expert resident, 2 missing,
// 3 a non-finite logit), the swaps made and the gate's logit and sigmoid (the
// host checks them against its own expf).
#define REC_PROBS 0u
#define REC_MIRROR 1024u
#define REC_IDS 2048u
#define REC_SLOTS 2112u
#define REC_W 2176u
#define REC_STATE 2240u
#define REC_SWAPS 2244u
#define REC_GATE 2248u
#define REC_SIG 2252u
#define REC_LOGITS 3072u // the logits as the kernel read them [256]: the host's probs check

// float bits -> double, exact (denormals included).
double f2d(uint f) {
    const uint s = f & 0x80000000u;
    const uint e = (f >> 23) & 0xFFu;
    const uint m = f & 0x7FFFFFu;
    if (e == 0xFFu)
        return asdouble(m << 29, s | 0x7FF00000u | (m >> 3));
    if (e == 0u) {
        if (m == 0u)
            return asdouble(0u, s);
        precise double v = (double)m * asdouble(0u, (1023u - 149u) << 20); // m * 2^-149
        return s != 0u ? -v : v;
    }
    return asdouble(m << 29, s | ((e + 896u) << 20) | (m >> 3));
}

// double -> float bits, rounded to nearest even (float denormals included).
uint d2f(double d) {
    uint lo, hi;
    asuint(d, lo, hi);
    const uint s = hi & 0x80000000u;
    const uint e = (hi >> 20) & 0x7FFu;
    const uint64_t m = ((uint64_t)(hi & 0xFFFFFu) << 32) | (uint64_t)lo;
    if (e == 0x7FFu)
        return s | 0x7F800000u | (m != 0 ? 0x400000u : 0u);
    if (e == 0u)
        return s; // zero or a double denormal: below every float
    const int fe = (int)e - 1023 + 127;
    if (fe >= 255)
        return s | 0x7F800000u;
    if (fe <= -24)
        return s; // below half the smallest denormal
    const uint64_t sig = m | ((uint64_t)1 << 52);
    const uint shift = fe >= 1 ? 29u : (uint)(30 - fe);
    const uint base = fe >= 1 ? (uint)(fe - 1) << 23 : 0u;
    const uint64_t rem = sig & (((uint64_t)1 << shift) - 1);
    const uint64_t half = (uint64_t)1 << (shift - 1u);
    uint r = (uint)(sig >> shift);
    if (rem > half || (rem == half && (r & 1u) != 0u))
        r += 1u; // a carry out of the mantissa steps the exponent (or reaches inf)
    return s | (base + r);
}

uint add32(uint a, uint b) {
    precise double r = f2d(a) + f2d(b);
    return d2f(r);
}
uint sub32(uint a, uint b) {
    precise double r = f2d(a) - f2d(b);
    return d2f(r);
}
uint mul32(uint a, uint b) {
    precise double r = f2d(a) * f2d(b); // exact: 24 + 24 bits
    return d2f(r);
}

// a * b + c rounded once (the CPU's vfmadd): the product is exact in double,
// TwoSum gives the sum's rounding error, and a sum rounded to odd at 53 bits
// rounds to 24 bits exactly as the infinitely precise one would.
uint fma32(uint a, uint b, uint c) {
    precise double p = f2d(a) * f2d(b);
    precise double z = f2d(c);
    precise double s = p + z;
    precise double bb = s - p;
    precise double t = (p - (s - bb)) + (z - bb);
    if (t != 0.0) {
        uint lo, hi;
        asuint(s, lo, hi);
        if ((t < 0.0) != (s < 0.0)) { // s was rounded away from zero: one ulp back
            if (lo == 0u)
                hi -= 1u;
            lo -= 1u;
        }
        lo |= 1u;
        s = asdouble(lo, hi);
    }
    return d2f(s);
}

// The nearest double to 1 / s (s > 0, normal): a quotient within 2 ulps fixed
// by the exact residuals 1 - q * s of its neighbours (no tie exists).
double recip_rn(double s) {
    precise double q = 1.0 / s;
    uint lo, hi;
    asuint(q, lo, hi);
    const uint64_t qb = ((uint64_t)hi << 32) | (uint64_t)lo;
    double best = q;
    double br = abs(fma(-q, s, 1.0));
    for (int k = -2; k <= 2; ++k) {
        if (k == 0)
            continue;
        const uint64_t cb = qb + (uint64_t)(int64_t)k;
        const double c = asdouble((uint)cb, (uint)(cb >> 32));
        const double r = abs(fma(-c, s, 1.0));
        if (r < br) {
            br = r;
            best = c;
        }
    }
    return best;
}

// a / b rounded to nearest (a >= 0, b > 0 finite): the double quotient
// rounded, then whichever float neighbour leaves the least exact residual.
uint div32(uint a, uint b) {
    precise double qd = f2d(a) / f2d(b);
    const uint f = d2f(qd);
    if (f == 0u || f >= 0x7F800000u)
        return f;
    const double da = f2d(a), db = f2d(b);
    uint best = f;
    double br = abs(fma(-f2d(f), db, da));
    for (int k = -1; k <= 1; k += 2) {
        const uint c = f + (uint)k;
        if (c == 0u || c >= 0x7F800000u)
            continue;
        const double r = abs(fma(-f2d(c), db, da));
        if (r < br) {
            br = r;
            best = c;
        }
    }
    return best;
}

#if KERNEL == 1
// ggml_v_expf (vec.h, __m256): the lanes of the slow path (|n| > 126) give
// fma(k, j, k) for the others, the fast path's fma(j, k, k).
uint expf_avx2(uint x) {
    const uint r = 0x4B400000u; // 0x1.8p23
    const uint z = fma32(x, 0x3FB8AA3Bu, r); // 0x1.715476p+0
    const uint n = sub32(z, r);
    const uint nn = n ^ 0x80000000u;
    const uint b = fma32(nn, 0x35BFBE8Eu, fma32(nn, 0x3F317200u, x)); // 0x1.7f7d1cp-20, 0x1.62e4p-1
    const uint e = z << 23;
    const uint k = e + 0x3F800000u;
    const double an = f2d(n & 0x7FFFFFFFu);
    const bool c = an > 126.0;
    const uint u = mul32(b, b);
    const uint j = fma32(fma32(fma32(0x3C072010u, b, 0x3D2B9F17u), u, fma32(0x3E2AAF33u, b, 0x3EFFFEDBu)),
                         u, mul32(0x3F7FFFF6u, b));
    if (!c)
        return fma32(j, k, k);
    const uint g = f2d(n) <= 0.0 ? 0x82000000u : 0u;
    const uint s1 = g + 0x7F000000u;
    const uint s2 = e - g;
    if (an > 192.0)
        return mul32(s1, s1);
    return mul32(fma32(s2, j, s2), s1);
}

// expf as the CPU's ucrt runs it on x64 (AMD libm expf_fma3, whose result
// sits 1 ulp off the correctly rounded one next to a float midpoint: console,
// 11/11): n = x * 64/ln 2 to the nearest integer, r = x - n ln 2/64 in one fma,
// e^r - 1 as r + r^2 (1/2 + r/6), times the table's 2^(j/64), 2^(n >> 6) added
// to the exponent bits, the double rounded to float.
static const uint kTwoJ64[128] = {
    0x00000000u, 0x3FF00000u, 0x3E778061u, 0x3FF02C9Au, 0xD3158574u, 0x3FF059B0u, 0x18759BC8u, 0x3FF08745u,
    0x6CF9890Fu, 0x3FF0B558u, 0x32D3D1A2u, 0x3FF0E3ECu, 0xD0125B51u, 0x3FF11301u, 0xAEA92DE0u, 0x3FF1429Au,
    0x3C7D517Bu, 0x3FF172B8u, 0xEB6FCB75u, 0x3FF1A35Bu, 0x3168B9AAu, 0x3FF1D487u, 0x88628CD6u, 0x3FF2063Bu,
    0x6E756238u, 0x3FF2387Au, 0x65E27CDDu, 0x3FF26B45u, 0xF51FDEE1u, 0x3FF29E9Du, 0xA6E4030Bu, 0x3FF2D285u,
    0x0A31B715u, 0x3FF306FEu, 0xB26416FFu, 0x3FF33C08u, 0x373AA9CBu, 0x3FF371A7u, 0x34E59FF7u, 0x3FF3A7DBu,
    0x4C123422u, 0x3FF3DEA6u, 0x21F72E2Au, 0x3FF4160Au, 0x6061892Du, 0x3FF44E08u, 0xB5C13CD0u, 0x3FF486A2u,
    0xD5362A27u, 0x3FF4BFDAu, 0x769D2CA7u, 0x3FF4F9B2u, 0x569D4F82u, 0x3FF5342Bu, 0x36B527DAu, 0x3FF56F47u,
    0xDD485429u, 0x3FF5AB07u, 0x15AD2148u, 0x3FF5E76Fu, 0xB03A5585u, 0x3FF6247Eu, 0x82552225u, 0x3FF66238u,
    0x667F3BCDu, 0x3FF6A09Eu, 0x3C651A2Fu, 0x3FF6DFB2u, 0xE8EC5F74u, 0x3FF71F75u, 0x564267C9u, 0x3FF75FEBu,
    0x73EB0187u, 0x3FF7A114u, 0x36CF4E62u, 0x3FF7E2F3u, 0x994CCE13u, 0x3FF82589u, 0x9B4492EDu, 0x3FF868D9u,
    0x422AA0DBu, 0x3FF8ACE5u, 0x99157736u, 0x3FF8F1AEu, 0xB0CDC5E5u, 0x3FF93737u, 0x9FDE4E50u, 0x3FF97D82u,
    0x82A3F090u, 0x3FF9C491u, 0x7B5DE565u, 0x3FFA0C66u, 0xB23E255Du, 0x3FFA5503u, 0x5579FDBFu, 0x3FFA9E6Bu,
    0x995AD3ADu, 0x3FFAE89Fu, 0xB84F15FBu, 0x3FFB33A2u, 0xF2FB5E47u, 0x3FFB7F76u, 0x904BC1D2u, 0x3FFBCC1Eu,
    0xDD85529Cu, 0x3FFC199Bu, 0x2E57D14Bu, 0x3FFC67F1u, 0xDCEF9069u, 0x3FFCB720u, 0x4A07897Cu, 0x3FFD072Du,
    0xDCFBA487u, 0x3FFD5818u, 0x03DB3285u, 0x3FFDA9E6u, 0x337B9B5Fu, 0x3FFDFC97u, 0xE78B3FF6u, 0x3FFE502Eu,
    0xA2A490DAu, 0x3FFEA4AFu, 0xEE615A27u, 0x3FFEFA1Bu, 0x5B6E4540u, 0x3FFF5076u, 0x819E90D8u, 0x3FFFA7C1u,
}; // 2^(j/64), j = 0 .. 63, (lo, hi) pairs
uint expf_ucrt(uint x) {
    const double xd = f2d(x);
    if (xd >= f2d(0x42B17218u))
        return 0x7F800000u; // from 88.7228394 (and +inf)
    if (xd < f2d(0xC2CE8ED0u))
        return 0u; // below -103.278931 (and -inf)
    precise double z = xd * asdouble(0x652B82FEu, 0x40571547u); // 64/ln 2
    precise double zr = z + asdouble(0u, 0x43380000u); // + 0x1.8p52: z to the nearest integer, ties to even
    uint nlo, nhi;
    asuint(zr, nlo, nhi);
    const int n = (int)nlo;
    precise double r = fma((double)n, asdouble(0xFEFA39EFu, 0xBF862E42u), xd); // -ln 2/64
    precise double p = fma(r, asdouble(0x55555555u, 0x3FC55555u), 0.5); // 1/6
    precise double rr = r * r;
    precise double q = fma(rr, p, r);
    const uint j = (uint)n & 63u;
    const double t = asdouble(kTwoJ64[2u * j], kTwoJ64[2u * j + 1u]);
    precise double y = fma(q, t, t);
    uint lo, hi;
    asuint(y, lo, hi);
    hi += ((uint)(n >> 6) & 0xFFFu) << 20; // the 64-bit integer add of (n >> 6) << 52
    return d2f(asdouble(lo, hi));
}

// ggml-cpu unary-ops.cpp op_sigmoid, 1.f / (1.f + expf(-x)), one value
uint sigmoid32(uint x) {
    if ((x & 0x7FFFFFFFu) > 0x7F800000u)
        return (x ^ 0x80000000u) | 0x400000u; // NaN: the quieted -x the SSE ops carry through
    const uint e = expf_ucrt(x ^ 0x80000000u);
    return div32(0x3F800000u, add32(0x3F800000u, e));
}

groupshared double gs_d[256];
groupshared uint gs_p[256];
groupshared uint gs_order[256];
groupshared int gs_res[256];
groupshared uint gs_bad;
groupshared uint gs_v;

// c0 = n_expert (a multiple of 8, <= 256), c1 = n_used (<= 16), c2 = swap
// ratio (float bits, >= 0), c3 = swap tail ranks, c4 = clamp minimum (float
// bits), c5 = scale (float bits), c6 = predicate value (layer + 1, != 0), c7 =
// flags: bit 0 normalise (sum_rows, clamp, div), bit 1 scale, bit 2 the
// shared expert's gate sigmoid (u3 one logit in, u4 out). u0 = record,
// u1 = router logits (n_expert floats), u2 = predicate (uint64, the host
// zeroes it; written only when an expert is missing). Grid (1, 1, 1).
// bit 3 alone is the selftest's sigmoid over c0 values (u3 in, u4 out, a
// group per 256).
// bit 4 fuses the emit (XLLAMA_D3D12_GPUROUTE_FUSE): on a hit the kernel
// also writes the weights at u5 + c9 and the slots, clamped into [0, c8 =
// n_slots), at u5 + c10, which the emit kernel would have copied from the
// record after the gate; on a miss neither is touched (the host writes them).
RWByteAddressBuffer U3 : register(u3);
RWByteAddressBuffer U4 : register(u4);
RWByteAddressBuffer U5 : register(u5);

[numthreads(256, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint n = c0;
    if ((c7 & 8u) != 0u) {
        const uint idx = gid.x * 256u + tid;
        if (idx < n)
            U4.Store(4u * idx, sigmoid32(U3.Load(4u * idx)));
        return;
    }
    if (tid == 0u) {
        gs_bad = 0u;
        if ((c7 & 4u) != 0u) {
            const uint g = U3.Load(0u);
            const uint sg = sigmoid32(g);
            U4.Store(0u, sg);
            D.Store(REC_GATE, g);
            D.Store(REC_SIG, sg);
        }
    }
    uint x = 0xFF800000u; // -inf
    if (tid < n) {
        x = U1.Load(4u * tid);
        D.Store(REC_LOGITS + 4u * tid, x);
        gs_res[tid] = (int)D.Load(REC_MIRROR + 4u * tid);
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid < n && (x & 0x7F800000u) == 0x7F800000u)
        InterlockedOr(gs_bad, 1u); // assert(!isnan) in ops.cpp; an inf logit ends in NaN probs
    gs_d[tid] = f2d(x);
    GroupMemoryBarrierWithGroupSync();
    // gs_bad is uniform past the barrier: a non-finite logit leaves the layer
    // to the host before any ranking (NaN probs tie, gs_order would have holes)
    if (gs_bad != 0u) {
        if (tid == 0u) {
            D.Store(REC_SWAPS, 0u);
            D.Store(REC_STATE, 3u);
            U2.Store(0u, c6);
        }
        return;
    }
    for (uint w = 128u; w > 0u; w >>= 1) {
        if (tid < w) {
            const double a = gs_d[tid], b = gs_d[tid + w];
            gs_d[tid] = b > a ? b : a;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const uint mx = d2f(gs_d[0]); // a logit: exact
    GroupMemoryBarrierWithGroupSync();

    uint val = 0u;
    if (tid < n)
        val = expf_avx2(sub32(x, mx));
    gs_p[tid] = val;
    GroupMemoryBarrierWithGroupSync();
    // eight lanes at a time: ((v0 + v4) + (v2 + v6)) + ((v1 + v5) + (v3 + v7))
    if (tid < n / 8u) {
        const uint o = tid * 8u;
        const uint a0 = add32(gs_p[o + 0u], gs_p[o + 4u]);
        const uint a1 = add32(gs_p[o + 1u], gs_p[o + 5u]);
        const uint a2 = add32(gs_p[o + 2u], gs_p[o + 6u]);
        const uint a3 = add32(gs_p[o + 3u], gs_p[o + 7u]);
        gs_d[tid] = f2d(add32(add32(a0, a2), add32(a1, a3)));
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u) {
        precise double sum = 0.0;
        for (uint g = 0u; g < n / 8u; ++g)
            sum = sum + gs_d[g];
        gs_v = sum > 0.0 ? d2f(recip_rn(sum)) : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint p = 0u;
    if (tid < n) {
        p = mul32(val, gs_v);
        D.Store(REC_PROBS + 4u * tid, p);
    }
    gs_p[tid] = p;
    GroupMemoryBarrierWithGroupSync();
    // ec_rank: by score, best first, ties by lower id (probs are >= +0, so
    // their bits order as the floats do)
    if (tid < n) {
        uint rank = 0u;
        for (uint i = 0u; i < n; ++i) {
            const uint q = gs_p[i];
            rank += (q > p || (q == p && i < tid)) ? 1u : 0u;
        }
        gs_order[rank] = tid;
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid != 0u)
        return;

    const uint n_used = c1;
    // ec_swap_missing: residents outside the top n_used stand in for missing
    // experts inside it, strongest resident for weakest missing
    uint swaps = 0u;
    uint i = n_used;
    for (int jj = (int)n_used - 1; jj >= 0; --jj) {
        const uint m = gs_order[jj];
        if (gs_res[m] >= 0)
            continue;
        while (i < n && gs_res[gs_order[i]] < 0)
            i++;
        if (i == n)
            break;
        const uint r = gs_order[i];
        const uint rj = n_used - (uint)jj <= c3 ? 0u : c2;
        const uint sr = gs_p[r];
        if (sr == 0u || sr < mul32(rj, gs_p[m]))
            break;
        gs_order[jj] = r;
        gs_order[i] = m;
        i++;
        swaps++;
    }

    uint wv[16];
    bool hit = true;
    precise double ws = 0.0;
    for (uint k = 0u; k < n_used; ++k) {
        const uint e = gs_order[k];
        hit = hit && gs_res[e] >= 0;
        wv[k] = gs_p[e];
        ws = ws + f2d(wv[k]);
        D.Store(REC_IDS + 4u * k, e);
        D.Store(REC_SLOTS + 4u * k, (uint)gs_res[e]);
    }
    uint den = 0u;
    if ((c7 & 1u) != 0u) {
        den = d2f(ws);
        const double dd = f2d(den);
        den = dd < asdouble(0u, 0x7FF00000u) ? den : 0x7F800000u; // MIN(x, INFINITY)
        den = f2d(den) > f2d(c4) ? den : c4;      // MAX(., 6.103515625e-5)
    }
    const bool fuse = hit && (c7 & 16u) != 0u;
    for (uint k = 0u; k < n_used; ++k) {
        uint w = wv[k];
        if ((c7 & 1u) != 0u)
            w = div32(w, den);
        if ((c7 & 2u) != 0u)
            w = mul32(w, c5);
        D.Store(REC_W + 4u * k, w);
        if (fuse) {
            U5.Store(c9 + 4u * k, w);
            U5.Store(c10 + 4u * k, (uint)clamp(gs_res[gs_order[k]], 0, (int)c8 - 1));
        }
    }
    D.Store(REC_SWAPS, swaps);
    const uint state = gs_bad != 0u ? 3u : hit ? 1u : 2u;
    D.Store(REC_STATE, state);
    if (state != 1u)
        U2.Store(0u, c6);
}
#elif KERNEL == 2
// c0 = n_used, c1 = n_slots: slots are clamped into [0, n_slots), so no record
// can send MUL_MAT_ID outside the expert cache. u0 = the weights tensor
// (n_used floats), u1 = record, u2 = the expert load's slots (n_used int32).
// Grid (1, 1, 1).
[numthreads(16, 1, 1)]
void CSMain(uint tid : SV_GroupIndex) {
    if (tid >= c0)
        return;
    D.Store(4u * tid, U1.Load(REC_W + 4u * tid));
    const int s = (int)U1.Load(REC_SLOTS + 4u * tid);
    U2.Store(4u * tid, (uint)clamp(s, 0, (int)c1 - 1));
}
#elif KERNEL == 3
// u0 (globallycoherent, write-back CUSTOM L0 memory): [0] the token for the
// CPU, [64] the CPU's answer, [128] spins done, [132] the answer seen, [136]
// ~token once the wait is over. c0 bit 0: read the answer with an atomic
// (InterlockedOr 0) instead of a load; bit 1: write the token with an atomic
// exchange instead of a store. c1 = spin limit (the host sizes it from a
// calibration, so a lost answer still ends the dispatch), c2 = token.
// Grid (1, 1, 1).
[numthreads(1, 1, 1)]
void CSMain() {
    if ((c0 & 2u) != 0u) {
        uint old;
        D.InterlockedExchange(0u, c2, old);
    } else {
        D.Store(0u, c2);
    }
    DeviceMemoryBarrier();
    uint i = 0u;
    uint seen = 0u;
    [loop]
    for (; i < c1; ++i) {
        uint v;
        if ((c0 & 1u) != 0u)
            D.InterlockedOr(64u, 0u, v);
        else
            v = D.Load(64u);
        if (v == c2) {
            seen = v;
            break;
        }
    }
    D.Store(128u, i);
    D.Store(132u, seen);
    DeviceMemoryBarrier();
    D.Store(136u, ~c2);
}
#endif
