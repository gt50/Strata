// src/kernels/cuda/kda.cu - see include/strata/kernels/kda.hpp.
#include "strata/kernels/kda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int D = kKdaHeadDim;  // head width: state rows = cols = 128
constexpr int DC = kKdaConvKernel;
constexpr int RG = 8;           // row groups of the scan's block
constexpr int RPG = D / RG;     // 16 state rows per thread
constexpr int kScanCols = 32;   // value columns a scan block takes (D / kScanCols blocks a head)
static_assert(D == 128 && DC == 4, "the reductions below are written for 4 warps of 32 over a 128-wide head");

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
/// ggml_compute_softplus_f32: log1p(exp(x)), with the large-x branch that avoids overflow.
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

/// One block per (128-channel head of q, k or v, token).  Channel c of part p reads its taps from the history for
/// the first d_conv - 1 tokens and from x after that: with ext = [history | x], tap i of token t is ext[t + i].
/// With one token the block can also slide the history in place, as no other block reads this channel's history;
/// with more, kda_conv_hist_kernel does it after every token has read the old one.
__global__ void __launch_bounds__(D) kda_conv_l2_kernel(float* __restrict__ hist, const float* __restrict__ x,
                                                        const float* __restrict__ w_q, const float* __restrict__ w_k,
                                                        const float* __restrict__ w_v, float* __restrict__ out,
                                                        int heads, int n_tokens, float eps) {
    __shared__ float part_sum[D / 32];
    const int part = blockIdx.x / heads;            // 0 q, 1 k, 2 v: uniform over the block
    const int t = blockIdx.y;
    const int d_inner = heads * D;
    const int c = (blockIdx.x % heads) * D + threadIdx.x;   // channel within the part
    const size_t gc = (size_t) part * d_inner + c;          // channel within [q | k | v]
    const size_t row = (size_t) 3 * d_inner;
    const float* w = (part == 0 ? w_q : part == 1 ? w_k : w_v) + (size_t) c * DC;
    float* h = hist + gc * (DC - 1);

    float tap[DC];
#pragma unroll
    for (int i = 0; i < DC; ++i) {
        const int e = t + i;
        tap[i] = e < DC - 1 ? h[e] : x[(size_t) (e - (DC - 1)) * row + gc];
    }
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < DC; ++i) sum = fmaf(tap[i], w[i], sum);   // kernel[0] reads the OLDEST input
    float y = sum / (1.0f + __expf(-sum));                         // SiLU
    if (n_tokens == 1) {
#pragma unroll
        for (int i = 0; i < DC - 1; ++i) h[i] = tap[i + 1];
    }
    if (part < 2) {                                                // L2 of the q and k heads
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part_sum[threadIdx.x >> 5] = sq;
        __syncthreads();
        y *= rsqrtf(part_sum[0] + part_sum[1] + part_sum[2] + part_sum[3] + eps);
    }
    out[(size_t) t * row + gc] = y;
}

/// The history after n_tokens > 1 tokens: the last d_conv - 1 entries of ext = [history | x].  Entry j comes from
/// ext[n_tokens + j], at or after the entry it replaces, so walking j upward reads every old value before it is
/// overwritten.
__global__ void kda_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ x, int channels,
                                     int n_tokens) {
    const int gc = blockIdx.x * blockDim.x + threadIdx.x;
    if (gc >= channels) return;
    float* h = hist + (size_t) gc * (DC - 1);
    for (int j = 0; j < DC - 1; ++j) {
        const int e = n_tokens + j;
        h[j] = e < DC - 1 ? h[e] : x[(size_t) (e - (DC - 1)) * channels + gc];
    }
}

__global__ void kda_gate_kernel(const float* raw, const float* __restrict__ dt,
                                const float* __restrict__ ssm_a, float* gate, int d_inner, size_t n, int has_lb,
                                float lb) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int c = (int) (idx % (size_t) d_inner);
    const float a = ssm_a[c / D];
    const float u = raw[idx] + dt[c];
    // sigmoid(-a * u) = 1 / (1 + exp(a * u)); a < 0, so a large positive u gives exp -> 0 and the gate -> lb
    gate[idx] = has_lb ? lb / (1.0f + expf(a * u)) : softplus_f(u) * a;
}

/// The scan: a block per (head, CW of its 128 value columns), threads (column j, row group); each thread holds 32 rows
/// of its column of the state in registers for the whole run of tokens.  Per token, row i is decayed by exp(gate[i])
/// BEFORE it is contracted against k (llama.cpp decays first, then forms S^T k from the decayed state).  A column's
/// update reads only that column (kv_j = sum_i S[i][j] k[i], o_j likewise), so a head's columns split across blocks;
/// only the output norm spans the head - o goes to y, and kda_out_norm_kernel finishes it.
///
/// On the 8060S, a decode token's 64 heads: one block a head of 4 x 32 rows, the state (D, H, D) - a head's rows 32 KB
/// apart - 59 us (8 MB moved at ~140 GB/s); split 4 ways by column, 56 us; the same on the state (H, D, D), a head's
/// 64 KB in one piece, 50 us; and 8 row groups of 16 rows (64K threads: more than are resident, so one wave's writes
/// overlap the next one's reads) 44-45 us.  16 groups of 8: 44 us; 8 groups on (D, H, D): 59 us.
template <int CW>
__global__ void __launch_bounds__(CW * RG) kda_scan_kernel(float* __restrict__ state, const float* __restrict__ qkv,
                                                           const float* __restrict__ gate,
                                                           const float* __restrict__ beta, float* __restrict__ o_out,
                                                           int heads, int n_tokens) {
    __shared__ float sk[D], sq[D], sdec[D];
    __shared__ float red_kv[RG][CW];
    __shared__ float red_o[RG][CW];
    __shared__ float sbeta;
    const int head = blockIdx.x / (D / CW);
    const int cl = threadIdx.x;           // the column within the block's
    const int col = (blockIdx.x % (D / CW)) * CW + cl;   // 0..127, the value column j
    const int rg = threadIdx.y;           // 0..RG - 1
    const int tid = rg * CW + cl;
    const int d_inner = heads * D;
    const size_t row3 = (size_t) 3 * d_inner;

    float s[RPG];
    float* base = state + ((size_t) head * D + rg * RPG) * D + col;   // (H, D, D): row i of head h at (h D + i) D
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * D];

    for (int t = 0; t < n_tokens; ++t) {
        const float* qt = qkv + (size_t) t * row3 + (size_t) head * D;
        const size_t ht = (size_t) t * d_inner + (size_t) head * D;
        __syncthreads();                  // the previous token is done with the shared arrays
        for (int i = tid; i < D; i += CW * RG) {
            sq[i] = qt[i];
            sk[i] = qt[d_inner + i];
            sdec[i] = expf(gate[ht + i]);
        }
        if (tid == 0) sbeta = beta[(size_t) t * heads + head];
        __syncthreads();
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] *= sdec[rg * RPG + r];
            kv = fmaf(s[r], sk[rg * RPG + r], kv);
        }
        red_kv[rg][cl] = kv;
        __syncthreads();
        float kv_col = red_kv[0][cl];
#pragma unroll
        for (int g2 = 1; g2 < RG; ++g2) kv_col += red_kv[g2][cl];
        const float delta = (qt[2 * d_inner + col] - kv_col) * sbeta;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(sk[rg * RPG + r], delta, s[r]);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
        __syncthreads();
        if (rg == 0) {
            float oo = red_o[0][cl];
#pragma unroll
            for (int g2 = 1; g2 < RG; ++g2) oo += red_o[g2][cl];
            o_out[ht + col] = oo * rsqrtf((float) D);
        }
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * D] = s[r];
}

// ---------------------------------------------------------------- the chunked scan (kda_scan_norm_chunked)
constexpr int GCH = kKdaChunk;       // tokens per chunk: the prep's solve is a lane per token
constexpr int GSB = kKdaSuperBlock;  // tokens per prep + scan launch pair
constexpr int PCH = D / 2;           // channels the prep stages at a time
constexpr int PRS = PCH + 4;         // the prep's padded k / q / gate rows (float4-aligned; tile rows 4 apart land on
                                     // other banks)
constexpr int GTP = GCH + 1;         // padded A / T rows
static_assert(GCH == 32, "kda_chunk_prep_kernel: a lane per token of the chunk, 8 x 8 tiles of 4 x 4 tokens");
static_assert(GSB % GCH == 0, "a super-block is whole chunks");
static_assert(GCH * GTP <= GCH * PRS, "A fits where k was");

/// Four channels of one (t, i) pair: kk += k_t.k_i and qk += q_t.k_i, each weighted by exp(Gam_t - Gam_i).  For
/// i > t (masked by the caller) the exponent is positive; the clamp keeps it from overflowing on the way.
__device__ __forceinline__ void kda_pair4(const float4& kt, const float4& qt, const float4& gt, const float4& ki,
                                          const float4& gi, float& kk, float& qk) {
    const float e0 = __expf(fminf(gt.x - gi.x, 0.0f)) * ki.x, e1 = __expf(fminf(gt.y - gi.y, 0.0f)) * ki.y;
    const float e2 = __expf(fminf(gt.z - gi.z, 0.0f)) * ki.z, e3 = __expf(fminf(gt.w - gi.w, 0.0f)) * ki.w;
    kk = fmaf(kt.x, e0, kk); kk = fmaf(kt.y, e1, kk); kk = fmaf(kt.z, e2, kk); kk = fmaf(kt.w, e3, kk);
    qk = fmaf(qt.x, e0, qk); qk = fmaf(qt.y, e1, qk); qk = fmaf(qt.z, e2, qk); qk = fmaf(qt.w, e3, qk);
}

/// A block per (chunk, head), 256 threads, in two passes of 64 key channels (26 KB of shared memory, so two blocks
/// share a CU; staging all 128 at once took 51 KB and one block a CU, too few loads in flight once a prompt no longer
/// fits in cache).  Per pass: the cumulative gate Gam, a thread per channel with its 32 tokens in registers; then the
/// products k_t.k_i and q_t.k_i weighted by exp(Gam_t - Gam_i) per channel, a thread per (4 x 4 tile of (t, i),
/// quarter of the channels) - the 28 tiles above the diagonal are zero and skipped.  Then the quarters are summed by
/// shuffles and T = (I + A)^-1 is solved on one warp.  Past the prompt's end a chunk is padded with k = q = 0, gate 0
/// and beta 0, which change neither the state nor any output.
__global__ void __launch_bounds__(256) kda_chunk_prep_kernel(const float* __restrict__ qkv, const float* __restrict__ gate,
                                                             const float* __restrict__ beta, float* __restrict__ tm,
                                                             float* __restrict__ pm, float* __restrict__ gm, int heads,
                                                             int64_t t0, int64_t T) {
    __shared__ __align__(16) float sk[GCH * PRS];
    __shared__ __align__(16) float sq[GCH * PRS];
    __shared__ __align__(16) float sg[GCH * PRS];   // the gate's sum from the chunk's start
    float* sa = sk;                                 // [GCH][GTP]  A (over k, once it is read)
    const int ch = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int d_inner = heads * D;
    const size_t row3 = (size_t) 3 * d_inner;
    const int64_t tb0 = t0 + (int64_t) ch * GCH;
    const int n = (int) min((int64_t) GCH, T - tb0);
    float* G = gm + ((size_t) ch * heads + h) * GCH * D;
    const int cg = tid & 3, tile = tid >> 2, tb = tile >> 3, ib = tile & 7;   // a warp: one tb, ib 0..7, cg 0..3
    float kk[4][4] = {}, qk[4][4] = {};
    for (int c0 = 0; c0 < D; c0 += PCH) {
        if (c0 > 0) __syncthreads();   // the last pass's reads done
#pragma unroll
        for (int p = tid; p < GCH * (PCH / 4); p += 256) {
            const int t = p / (PCH / 4), r = (p % (PCH / 4)) * 4;
            float4 qv = make_float4(0.0f, 0.0f, 0.0f, 0.0f), kv = qv;
            if (t < n) {
                const float* row = qkv + (size_t) (tb0 + t) * row3 + (size_t) h * D + c0 + r;
                qv = *reinterpret_cast<const float4*>(row);
                kv = *reinterpret_cast<const float4*>(row + d_inner);
            }
            *reinterpret_cast<float4*>(&sq[t * PRS + r]) = qv;
            *reinterpret_cast<float4*>(&sk[t * PRS + r]) = kv;
        }
        if (tid < PCH) {   // a sum of terms <= 0: never increasing in t, so Gam_t - Gam_i <= 0 exactly for i <= t
            const float* g = gate + (size_t) tb0 * d_inner + (size_t) h * D + c0 + tid;
            float acc = 0.0f;
#pragma unroll
            for (int t = 0; t < GCH; ++t) {
                acc += t < n ? g[(size_t) t * d_inner] : 0.0f;
                sg[t * PRS + tid] = acc;
                G[t * D + c0 + tid] = acc;
            }
        }
        __syncthreads();
        if (ib <= tb) {
#pragma unroll 1
            for (int m = 0; m < PCH / 16; ++m) {
                const int c = 4 * (cg + 4 * m);
                float4 kt[4], qt[4], gt[4], ki[4], gi[4];
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    kt[x] = *reinterpret_cast<const float4*>(&sk[(4 * tb + x) * PRS + c]);
                    qt[x] = *reinterpret_cast<const float4*>(&sq[(4 * tb + x) * PRS + c]);
                    gt[x] = *reinterpret_cast<const float4*>(&sg[(4 * tb + x) * PRS + c]);
                    ki[x] = *reinterpret_cast<const float4*>(&sk[(4 * ib + x) * PRS + c]);
                    gi[x] = *reinterpret_cast<const float4*>(&sg[(4 * ib + x) * PRS + c]);
                }
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int y = 0; y < 4; ++y) kda_pair4(kt[x], qt[x], gt[x], ki[y], gi[y], kk[x][y], qk[x][y]);
            }
        }
    }
#pragma unroll
    for (int x = 0; x < 4; ++x)
#pragma unroll
        for (int y = 0; y < 4; ++y)
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {   // the four channel quarters are neighbouring lanes
                kk[x][y] += __shfl_xor_sync(0xffffffffu, kk[x][y], o);
                qk[x][y] += __shfl_xor_sync(0xffffffffu, qk[x][y], o);
            }
    __syncthreads();   // every read of k, q and the gate done: A goes over k
    if (cg == 0) {
        float* P_ = pm + ((size_t) ch * heads + h) * GCH * GCH;
#pragma unroll
        for (int x = 0; x < 4; ++x) {
            const int t = 4 * tb + x;
            const float bt = t < n ? beta[(size_t) (tb0 + t) * heads + h] : 0.0f;
#pragma unroll
            for (int y = 0; y < 4; ++y) {
                const int i = 4 * ib + y;
                P_[t * GCH + i] = i <= t ? qk[x][y] : 0.0f;
                sa[t * GTP + i] = i < t ? bt * kk[x][y] : (i == t ? 1.0f : 0.0f);
            }
        }
    }   // a tile above the diagonal has i > t throughout: its cg == 0 thread writes its zeros
    __syncthreads();
    if (tid >= 32) return;
    const int i = tid;
    // T = (I + A)^-1 by forward substitution, T[t][i] = -sum_{k < t} A[t][k] T[k][i] (t > i), lane i holding column
    // i of T in registers: the rows of A are broadcast reads and nothing waits on a write, so the unrolled loop's
    // loads all go ahead.  T[k][i] = 0 for k < i, so the sum may run from k = 0.
    float col[GCH];
#pragma unroll
    for (int t = 0; t < GCH; ++t) {
        float v = 0.0f;
#pragma unroll
        for (int k = 0; k < t; ++k) v = fmaf(-sa[t * GTP + k], col[k], v);
        col[t] = t > i ? v : (t == i ? 1.0f : 0.0f);
    }
    float* T_ = tm + ((size_t) ch * heads + h) * GCH * GCH;
#pragma unroll
    for (int t = 0; t < GCH; ++t) T_[t * GCH + i] = col[t];
}

/// A block per head and a PAIR of neighbouring lanes per value column j, each lane holding half the column's state
/// rows (64) in registers for the whole super-block: a contraction over the rows is the lane's own sum plus one
/// shuffle.  (One lane per column held 128 rows, and on gfx1151 spilled 280 registers.)  Two 16 KB buffers, so 4
/// blocks fit a 128 KB AMD WGP, and on AMD at most 192 registers (8 waves a SIMD), so 2 fit a CU: all 64 heads are
/// resident on a 40-CU Radeon 8060S.  Each block walks every chunk, so a second round of blocks would double the time
/// (left to itself the compiler took 255 registers, one block a CU).  A chunk, with the buffers' roles in brackets:
///   [A] = beta_t exp(Gam_t) k_t, [B] = beta_t v_t;   [B] -= S0^T [A]_t        ([B] = Y)
///   [A] = exp(Gam_t) q_t;   the output = S0^T [A]_t
///   [A] = D = T [B];   [B] = P D;   the output = (the output + [B]) / sqrt(128)
///   [B] = exp(Gam_last - Gam_t) k_t;   S = diag(exp(Gam_last)) S0 + sum_t [B]_t [A]_t^T
/// No loop that runs token by token waits on a global load: on the 8060S one costs about a microsecond, and with v,
/// beta and the output read inside those loops a chunk took 110 us.  In a staged row the second half's float4s are
/// swapped in pairs, so the two lanes of a column, reading rows i and 64 + i, hit different banks.  For the two
/// triangular products a lane is (column, row set) with the row set the same across a warp, so the rows of T and P
/// are uniform loads.  The barriers order the output's writes between lanes as they do shared memory's.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define KDA_SCAN_OCCUPANCY __attribute__((amdgpu_waves_per_eu(8)))
#else
#define KDA_SCAN_OCCUPANCY
#endif
__global__ void __launch_bounds__(2 * D) KDA_SCAN_OCCUPANCY
kda_chunk_scan_kernel(float* __restrict__ state, const float* __restrict__ qkv, const float* __restrict__ beta,
                      const float* __restrict__ tm, const float* __restrict__ pm, const float* __restrict__ gm,
                      float* __restrict__ o_out, int heads, int64_t t0, int nch, int64_t T, int stride_rows) {
    constexpr int HD = D / 2;                     // state rows a lane holds
    constexpr int RPL = GCH / 2;                  // rows of D and of P D a lane computes
    __shared__ __align__(16) float sa[GCH * D];
    __shared__ __align__(16) float sb[GCH * D];
    const int h = blockIdx.x, tid = threadIdx.x, j = tid >> 1, half = tid & 1;
    const int i0 = half * HD, sw = half * 4;      // this lane's rows; its float4 swizzle
    const int jc = tid & (D - 1), rset = tid >> 7;   // the triangular products' lane: column, rows rset + 2m
    const int d_inner = heads * D;
    const size_t row3 = (size_t) 3 * d_inner;
    // the state's row stride, D in (H, D, D), as stride_rows (1) x D: a runtime multiple of D - with the constant D,
    // or a stride argument of D, the scan took 192 VGPRs and spilled 114; with this, 182 and none (as with the old
    // layout's heads x D)
    const size_t rs = (size_t) stride_rows * D;
    const size_t hb = (size_t) h * D * D + j;   // the lane's column of its head
    float s[HD];
#pragma unroll
    for (int i = 0; i < HD; ++i) s[i] = state[(size_t) (i0 + i) * rs + hb];
    const float out_scale = rsqrtf((float) D);
    // this lane's dot of its half column with staged row t
    auto dot_half = [&](const float* buf, int t) {
        float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
#pragma unroll
        for (int i = 0; i < HD; i += 4) {
            const float4 k4 = *reinterpret_cast<const float4*>(&buf[t * D + ((i0 + i) ^ sw)]);
            x0 = fmaf(s[i], k4.x, x0); x1 = fmaf(s[i + 1], k4.y, x1);
            x2 = fmaf(s[i + 2], k4.z, x2); x3 = fmaf(s[i + 3], k4.w, x3);
        }
        const float x = (x0 + x1) + (x2 + x3);
        return x + __shfl_xor_sync(0xffffffffu, x, 1);
    };
    // out = M [in] for column jc, rows rset + 2m, M lower triangular [GCH][GCH]: its zeros above the diagonal are
    // multiplied in rather than skipped (no branches, and whole rows of M as uniform loads)
    auto tri_mul = [&](const float* M, const float* in, float* out) {
        float acc[RPL];
#pragma unroll
        for (int m = 0; m < RPL; ++m) acc[m] = 0.0f;
#pragma unroll 2
        for (int i = 0; i < GCH; i += 4) {
            const float y0 = in[i * D + jc], y1 = in[(i + 1) * D + jc], y2 = in[(i + 2) * D + jc],
                        y3 = in[(i + 3) * D + jc];
#pragma unroll
            for (int m = 0; m < RPL; ++m) {
                const float4 m4 = *reinterpret_cast<const float4*>(M + (rset + 2 * m) * GCH + i);
                acc[m] = fmaf(m4.x, y0, acc[m]); acc[m] = fmaf(m4.y, y1, acc[m]);
                acc[m] = fmaf(m4.z, y2, acc[m]); acc[m] = fmaf(m4.w, y3, acc[m]);
            }
        }
#pragma unroll
        for (int m = 0; m < RPL; ++m) out[(rset + 2 * m) * D + jc] = acc[m];
    };
    for (int ch = 0; ch < nch; ++ch) {
        const int64_t c_t0 = t0 + (int64_t) ch * GCH;
        const int n = (int) min((int64_t) GCH, T - c_t0);
        const float* G = gm + ((size_t) ch * heads + h) * GCH * D;
        const float* GL = G + (GCH - 1) * D;
        const float* T_ = tm + ((size_t) ch * heads + h) * GCH * GCH;
        const float* P_ = pm + ((size_t) ch * heads + h) * GCH * GCH;
        // what: 0 beta_t exp(Gam_t) k_t into buf and beta_t v_t into vbuf, 1 exp(Gam_t) q_t, 2 exp(Gam_last - Gam_t) k_t
        auto stage = [&](float* buf, float* vbuf, int what) {
#pragma unroll
            for (int p = tid; p < GCH * (D / 4); p += 2 * D) {
                const int t = p / (D / 4), r = (p % (D / 4)) * 4;
                float4 out = make_float4(0.0f, 0.0f, 0.0f, 0.0f), vout = out;
                if (t < n) {
                    const float* row = qkv + (size_t) (c_t0 + t) * row3 + (size_t) h * D + r;
                    const float4 x4 = *reinterpret_cast<const float4*>(what == 1 ? row : row + d_inner);
                    float4 g4 = *reinterpret_cast<const float4*>(G + t * D + r);
                    float b = 1.0f;
                    if (what == 0) {
                        b = beta[(size_t) (c_t0 + t) * heads + h];
                        const float4 v4 = *reinterpret_cast<const float4*>(row + 2 * d_inner);
                        vout = make_float4(b * v4.x, b * v4.y, b * v4.z, b * v4.w);
                    }
                    if (what == 2) {
                        const float4 l4 = *reinterpret_cast<const float4*>(GL + r);
                        g4 = make_float4(l4.x - g4.x, l4.y - g4.y, l4.z - g4.z, l4.w - g4.w);
                    }
                    out = make_float4(b * x4.x * __expf(g4.x), b * x4.y * __expf(g4.y), b * x4.z * __expf(g4.z),
                                      b * x4.w * __expf(g4.w));
                }
                *reinterpret_cast<float4*>(&buf[t * D + (r ^ (r >= HD ? 4 : 0))]) = out;
                if (what == 0) *reinterpret_cast<float4*>(&vbuf[t * D + r]) = vout;   // [t][column], unswizzled
            }
        };
        stage(sa, sb, 0);
        __syncthreads();
        // Y_t = beta_t v_t - S0^T (beta_t exp(Gam_t) k_t), column j
#pragma unroll 1
        for (int t = 0; t < GCH; ++t) {
            const float x = dot_half(sa, t);
            if (half == 0) sb[t * D + j] -= x;
        }
        __syncthreads();   // Y complete; every read of the staged k done
        stage(sa, nullptr, 1);
        __syncthreads();
        // the output's first term, S0^T exp(Gam_t) q_t (written, not read: nothing here waits on memory)
#pragma unroll 1
        for (int t = 0; t < n; ++t) {
            const float x = dot_half(sa, t);
            if (half == 0) o_out[(size_t) (c_t0 + t) * d_inner + (size_t) h * D + j] = x;
        }
        __syncthreads();   // every read of the staged q done
        tri_mul(T_, sb, sa);   // D = T Y
        __syncthreads();
        tri_mul(P_, sa, sb);   // P D
        __syncthreads();
        // o_t = S0^T exp(Gam_t) q_t + (P D)_t: sixteen rows a lane, so their loads overlap
#pragma unroll
        for (int m = 0; m < RPL; ++m) {
            const int t = rset + 2 * m;
            if (t < n) {
                float* out = o_out + (size_t) (c_t0 + t) * d_inner + (size_t) h * D + jc;
                *out = (*out + sb[t * D + jc]) * out_scale;
            }
        }
        __syncthreads();   // every read of P D done
        stage(sb, nullptr, 2);
        __syncthreads();
        // S = diag(exp(Gam_last)) S0 + sum_t (exp(Gam_last - Gam_t) k_t) D_t^T
#pragma unroll
        for (int i = 0; i < HD; ++i) s[i] *= __expf(GL[i0 + i]);
#pragma unroll 1
        for (int t = 0; t < GCH; ++t) {
            const float d = sa[t * D + j];
#pragma unroll
            for (int i = 0; i < HD; i += 4) {
                const float4 k4 = *reinterpret_cast<const float4*>(&sb[t * D + ((i0 + i) ^ sw)]);
                s[i] = fmaf(k4.x, d, s[i]); s[i + 1] = fmaf(k4.y, d, s[i + 1]);
                s[i + 2] = fmaf(k4.z, d, s[i + 2]); s[i + 3] = fmaf(k4.w, d, s[i + 3]);
            }
        }
        __syncthreads();   // every read of both buffers done before the next chunk's
    }
    // the same addresses as the loads, from a stride the compiler cannot prove equal to `rs`: left to itself it kept
    // the loads' 64 addresses (128 registers) live through the whole scan, and spilled
    const size_t rs2 = (size_t) __shfl_sync(0xffffffffu, (int) rs, 0);
#pragma unroll
    for (int i = 0; i < HD; ++i) state[(size_t) (i0 + i) * rs2 + hb] = s[i];
}

/// y = rms_norm(o) * o_norm * sigmoid(z) in place, a block per (token, head): both scans' lanes hold parts of a
/// column, so the norm, which couples the head's columns, runs after them.
__global__ void __launch_bounds__(D) kda_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                         float eps, float* __restrict__ y) {
    __shared__ float wsum[D / 32];
    const size_t at = (size_t) blockIdx.x * D + threadIdx.x;
    const float oc = y[at];
    float sq = oc * oc;
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o2);
    if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float scale = rsqrtf((wsum[0] + wsum[1] + wsum[2] + wsum[3]) / (float) D + eps);
    y[at] = oc * scale * gamma[threadIdx.x] * sigmoid_f(z[at]);
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void kda_conv_l2(float* history, const float* x, const float* w_q, const float* w_k, const float* w_v, float* out,
                 int heads, int n_tokens, float eps, void* stream) {
    if (!history || !x || !w_q || !w_k || !w_v || !out || heads <= 0 || n_tokens <= 0 || n_tokens > 65535) {
        std::fprintf(stderr, "kda_conv_l2: invalid arguments\n");
        std::exit(1);
    }
    const dim3 grid((unsigned) (3 * heads), (unsigned) n_tokens);
    kda_conv_l2_kernel<<<grid, D, 0, (cudaStream_t) stream>>>(history, x, w_q, w_k, w_v, out, heads, n_tokens, eps);
    check_launch("kda_conv_l2");
    if (n_tokens > 1) {
        const int channels = 3 * heads * D;
        kda_conv_hist_kernel<<<(unsigned) ((channels + 255) / 256), 256, 0, (cudaStream_t) stream>>>(history, x, channels,
                                                                                                    n_tokens);
        check_launch("kda_conv_l2 (history)");
    }
}

void kda_gate(const float* raw, const float* dt_bias, const float* ssm_a, float* gate, int heads, int n_tokens,
              bool has_lower_bound, float lower_bound, void* stream) {
    if (!raw || !dt_bias || !ssm_a || !gate || heads <= 0 || n_tokens <= 0) {
        std::fprintf(stderr, "kda_gate: invalid arguments\n");
        std::exit(1);
    }
    const int d_inner = heads * D;
    const size_t n = (size_t) n_tokens * (size_t) d_inner;
    kda_gate_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(
        raw, dt_bias, ssm_a, gate, d_inner, n, has_lower_bound ? 1 : 0, lower_bound);
    check_launch("kda_gate");
}

void kda_scan_norm(float* state, const float* qkv, const float* gate, const float* beta, const float* z,
                   const float* o_norm, float eps, float* y, int heads, int n_tokens, void* stream) {
    if (!state || !qkv || !gate || !beta || !z || !o_norm || !y || heads <= 0 || n_tokens <= 0 ||
        (int64_t) n_tokens * heads > 0x7fffffff) {
        std::fprintf(stderr, "kda_scan_norm: invalid arguments\n");
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    kda_scan_kernel<kScanCols><<<(unsigned) (heads * (D / kScanCols)), dim3(kScanCols, RG), 0, s>>>(state, qkv, gate, beta,
                                                                                                   y, heads, n_tokens);
    check_launch("kda_scan_norm (scan)");
    kda_out_norm_kernel<<<(unsigned) ((int64_t) n_tokens * heads), D, 0, s>>>(z, o_norm, eps, y);
    check_launch("kda_scan_norm (norm)");
}

size_t kda_chunked_scratch_floats(int heads, int n_tokens) {
    if (heads <= 0 || n_tokens <= 0) return 0;
    const size_t nch = (size_t) ((n_tokens < GSB ? n_tokens : GSB) + GCH - 1) / GCH;
    return nch * (size_t) heads * (2 * GCH * GCH + GCH * D);
}

void kda_scan_norm_chunked(float* state, const float* qkv, const float* gate, const float* beta, const float* z,
                           const float* o_norm, float eps, float* y, int heads, int n_tokens, float* scratch,
                           void* stream) {
    if (!state || !qkv || !gate || !beta || !z || !o_norm || !y || !scratch || heads <= 0 || n_tokens <= 0 ||
        heads > 65535) {
        std::fprintf(stderr, "kda_scan_norm_chunked: invalid arguments\n");
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const int nb = (int) ((std::min<int64_t>(GSB, n_tokens) + GCH - 1) / GCH);
    const size_t tsz = (size_t) nb * heads * GCH * GCH;
    float *tm = scratch, *pm = tm + tsz, *gm = pm + tsz;
    for (int64_t t0 = 0; t0 < n_tokens; t0 += GSB) {
        const int nch = (int) ((std::min<int64_t>(GSB, n_tokens - t0) + GCH - 1) / GCH);
        kda_chunk_prep_kernel<<<dim3((unsigned) nch, (unsigned) heads), 256, 0, s>>>(qkv, gate, beta, tm, pm, gm, heads,
                                                                                    t0, n_tokens);
        check_launch("kda_scan_norm_chunked (prep)");
        kda_chunk_scan_kernel<<<(unsigned) heads, 2 * D, 0, s>>>(state, qkv, beta, tm, pm, gm, y, heads, t0, nch,
                                                                n_tokens, 1);
        check_launch("kda_scan_norm_chunked (scan)");
    }
    kda_out_norm_kernel<<<(unsigned) ((int64_t) n_tokens * heads), D, 0, s>>>(z, o_norm, eps, y);
    check_launch("kda_scan_norm_chunked (norm)");
}

}  // namespace strata::kernels
