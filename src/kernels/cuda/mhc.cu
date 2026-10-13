// src/kernels/cuda/mhc.cu - see include/strata/kernels/mhc.hpp.
#include "strata/kernels/mhc.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int HC = kMhcStreams;
constexpr int MIX = kMhcMix;
constexpr int MT = 64;      // GEMV threads a block: 2 warps, so a decode token's GEMV is 128 blocks, not 32
constexpr int KS = 2 * MT;  // GEMV columns a block takes: 2 a thread
constexpr int TG = 4;       // tokens a block takes, so a weight chunk is read once for 4
constexpr int NV = MIX + 1; // per token: the 24 dots and the sum of squares
constexpr int RG = 4;       // hc_fn rows a block takes (blockIdx.z picks which 4 of the 24): 6 x the blocks, so a decode
                            // token's GEMV is 768 of them - 128 left the 1.5 MB at ~100 GB/s (16 us), 768 take 8-12 us
static_assert(HC == 4 && MIX == 24, "the finalize kernel's Sinkhorn is written for 4 streams");

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

/// A block per (128 columns of the flattened 4 n_embd, 4 tokens, 4 of the 24 rows of hc_fn): each thread 2 columns,
/// its 4 rows (and in the first row group the sum of squares), summed over the block; the partial sums
/// [split][token][25] are added in split order by the finalize kernel (no atomics: the same bits every run - and the
/// same bits as one block taking all 24 rows, every dot summed in the same order).
__global__ void __launch_bounds__(MT) mhc_mix_partial_kernel(const float* __restrict__ x, const float* __restrict__ hc_fn,
                                                             int n_tok, int K, float* __restrict__ part) {
    constexpr int NR = RG + 1;   // the group's rows, then the sum of squares (first group only)
    __shared__ float red[MT / 32][NR * TG];
    const int split = blockIdx.x, t0 = blockIdx.y * TG, r0 = blockIdx.z * RG, tid = threadIdx.x, lane = tid & 31,
              warp = tid >> 5;
    const bool squares = blockIdx.z == 0;
    const int k = split * KS + 2 * tid;
    float2 xv[TG];
#pragma unroll
    for (int g = 0; g < TG; ++g)
        xv[g] = t0 + g < n_tok ? *reinterpret_cast<const float2*>(x + (size_t) (t0 + g) * K + k) : make_float2(0.0f, 0.0f);
    float v[NR][TG];
#pragma unroll
    for (int r = 0; r < RG; ++r) {
        const float2 w = *reinterpret_cast<const float2*>(hc_fn + (size_t) (r0 + r) * K + k);
#pragma unroll
        for (int g = 0; g < TG; ++g) v[r][g] = fmaf(w.y, xv[g].y, w.x * xv[g].x);
    }
#pragma unroll
    for (int g = 0; g < TG; ++g) v[RG][g] = fmaf(xv[g].y, xv[g].y, xv[g].x * xv[g].x);
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int g = 0; g < TG; ++g) {
            float s = v[r][g];
            for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
            if (lane == 0) red[warp][r * TG + g] = s;
        }
    __syncthreads();
    for (int i = tid; i < NR * TG; i += MT) {
        const int r = i / TG, g = i % TG;
        if (t0 + g < n_tok && (r < RG || squares)) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < MT / 32; ++w) s += red[w][i];
            part[((size_t) split * n_tok + t0 + g) * NV + (r < RG ? r0 + r : MIX)] = s;
        }
    }
}

/// A block per token, 256 threads.
///   the splits' sums: a warp per value (24 dots and the sum of squares), its lanes striding over the splits, then a
///                     shuffle sum (one thread walking the splits waited on memory at every one: 45-57 us a decode
///                     token)
///   the gating:       on warp 0, a lane per entry of comb (lanes 16..31 repeat lanes 0..15 and write nothing): the
///                     softmax over dst and the row sums are shuffles across lanes 4s .. 4s + 3, the column sums
///                     across lanes d, d + 4, d + 8, d + 12
///   h:                sum_i pre_i X[i] across the block
///   the norm:         with norm_w, h = (h norm_w) / rms(h): the squares summed as h is written, a block reduction,
///                     and each thread scales the values it wrote
__global__ void __launch_bounds__(256) mhc_finalize_kernel(const float* __restrict__ x, const float* __restrict__ part,
                                                           int n_splits, int n_tok, int n_embd,
                                                           const float* __restrict__ hc_scale,
                                                           const float* __restrict__ hc_base, float rms_eps, float hc_eps,
                                                           int iters, float* __restrict__ h, float* __restrict__ post,
                                                           float* __restrict__ comb, float* __restrict__ pre,
                                                           const float* __restrict__ norm_w, float norm_eps) {
    __shared__ float mix[NV];
    __shared__ float pre_s[HC];
    __shared__ float sq[8];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int v = warp; v < NV; v += 8) {
        float s = 0.0f;
        for (int sp = lane; sp < n_splits; sp += 32) s += part[((size_t) sp * n_tok + t) * NV + v];
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) mix[v] = s;
    }
    __syncthreads();
    if (warp == 0) {
        const int K = HC * n_embd;
        const float inv = rsqrtf(mix[MIX] / (float) K + rms_eps);
        const int e = lane & 15;   // comb[e / 4][e % 4], lane e (and e + 16)
        if (lane < HC) {
            const float p = sigmoid_f(mix[lane] * inv * hc_scale[0] + hc_base[lane]) + hc_eps;
            pre_s[lane] = p;
            if (pre) pre[(size_t) t * HC + lane] = p;
            post[(size_t) t * HC + lane] = 2.0f * sigmoid_f(mix[HC + lane] * inv * hc_scale[1] + hc_base[HC + lane]);
        }
        float c = mix[2 * HC + e] * inv * hc_scale[2] + hc_base[2 * HC + e];
        // softmax over dst (the 4 lanes of a row), then + eps
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        c = expf(c - mx);
        float sum = c + __shfl_xor_sync(0xffffffffu, c, 1);
        sum = sum + __shfl_xor_sync(0xffffffffu, sum, 2);
        c = c * (1.0f / sum) + hc_eps;
        auto row_sum = [&](float v) {
            v = v + __shfl_xor_sync(0xffffffffu, v, 1);
            return v + __shfl_xor_sync(0xffffffffu, v, 2);
        };
        auto col_sum = [&](float v) {
            v = v + __shfl_xor_sync(0xffffffffu, v, 4);
            return v + __shfl_xor_sync(0xffffffffu, v, 8);
        };
        c = c / (col_sum(c) + hc_eps);
        for (int it = 1; it < iters; ++it) {
            c = c / (row_sum(c) + hc_eps);
            c = c / (col_sum(c) + hc_eps);
        }
        if (lane < HC * HC) comb[(size_t) t * HC * HC + lane] = c;
    }
    __syncthreads();
    const float* xt = x + (size_t) t * HC * n_embd;
    float* ht = h + (size_t) t * n_embd;
    float ss = 0.0f;
    for (int d = tid; d < n_embd; d += 256) {
        float acc = xt[d] * pre_s[0];
#pragma unroll
        for (int i = 1; i < HC; ++i) acc = acc + xt[(size_t) i * n_embd + d] * pre_s[i];
        ht[d] = acc;
        ss = fmaf(acc, acc, ss);
    }
    if (!norm_w) return;
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    if (lane == 0) sq[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) tot += sq[w];
    const float inv = rsqrtf(tot / (float) n_embd + norm_eps);
    for (int d = tid; d < n_embd; d += 256) ht[d] = ht[d] * norm_w[d] * inv;   // this thread's own values
}

/// A thread per (token, column): the 4 streams' values read, then the 4 new ones written over them.
__global__ void mhc_post_kernel(const float* __restrict__ f, const float* __restrict__ post,
                                const float* __restrict__ comb, float* __restrict__ x, int n_embd, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t t = i / n_embd;
    const int d = (int) (i % n_embd);
    float* xt = x + (size_t) t * HC * n_embd + d;
    float r[HC];
#pragma unroll
    for (int s = 0; s < HC; ++s) r[s] = xt[(size_t) s * n_embd];
    const float fv = f[i];
    const float* pt = post + t * HC;
    const float* ct = comb + t * HC * HC;
    float out[HC];
#pragma unroll
    for (int dst = 0; dst < HC; ++dst) {
        float acc = fv * pt[dst];
#pragma unroll
        for (int s = 0; s < HC; ++s) acc = acc + r[s] * ct[s * HC + dst];
        out[dst] = acc;
    }
#pragma unroll
    for (int dst = 0; dst < HC; ++dst) xt[(size_t) dst * n_embd] = out[dst];
}

__global__ void mhc_init_kernel(const float* __restrict__ emb, float* __restrict__ x, int n_embd, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t t = i / n_embd;
    const int d = (int) (i % n_embd);
    const float v = emb[i];
#pragma unroll
    for (int s = 0; s < HC; ++s) x[((size_t) t * HC + s) * n_embd + d] = v;
}

__global__ void mhc_mean_kernel(const float* __restrict__ x, float* __restrict__ y, int n_embd, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t t = i / n_embd;
    const int d = (int) (i % n_embd);
    const float* xt = x + (size_t) t * HC * n_embd + d;
    float acc = xt[0];
#pragma unroll
    for (int s = 1; s < HC; ++s) acc = acc + xt[(size_t) s * n_embd];
    y[i] = acc * (1.0f / HC);
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

unsigned blocks_for(int64_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

uint64_t mhc_pre_scratch_floats(int n_tok, int n_embd) {
    if (n_tok <= 0 || n_embd <= 0) return 0;
    return (uint64_t) (HC * n_embd / KS) * (uint64_t) n_tok * NV;
}

void mhc_pre(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base, int n_tok, int n_embd,
             float rms_eps, float hc_eps, int sinkhorn_iters, float* scratch, float* h, float* post, float* comb,
             float* pre, void* stream, const float* norm_w, float norm_eps) {
    if (!x || !hc_fn || !hc_scale || !hc_base || !scratch || !h || !post || !comb || n_tok <= 0 || n_embd <= 0 ||
        (HC * n_embd) % KS != 0 || sinkhorn_iters < 1 || (n_tok + TG - 1) / TG > 65535) {
        std::fprintf(stderr, "mhc_pre: invalid arguments\n");
        std::exit(1);
    }
    const int K = HC * n_embd, n_splits = K / KS;
    const cudaStream_t s = (cudaStream_t) stream;
    mhc_mix_partial_kernel<<<dim3((unsigned) n_splits, (unsigned) ((n_tok + TG - 1) / TG), MIX / RG), MT, 0, s>>>(
        x, hc_fn, n_tok, K, scratch);
    check_launch("mhc_pre (mixes)");
    mhc_finalize_kernel<<<(unsigned) n_tok, 256, 0, s>>>(x, scratch, n_splits, n_tok, n_embd, hc_scale, hc_base, rms_eps,
                                                         hc_eps, sinkhorn_iters, h, post, comb, pre, norm_w, norm_eps);
    check_launch("mhc_pre (finalize)");
}

void mhc_post(const float* f, const float* post, const float* comb, float* x, int n_tok, int n_embd, void* stream) {
    if (!f || !post || !comb || !x || n_tok <= 0 || n_embd <= 0) {
        std::fprintf(stderr, "mhc_post: invalid arguments\n");
        std::exit(1);
    }
    const int64_t n = (int64_t) n_tok * n_embd;
    mhc_post_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(f, post, comb, x, n_embd, n);
    check_launch("mhc_post");
}

void mhc_init(const float* emb, float* x, int n_tok, int n_embd, void* stream) {
    if (!emb || !x || n_tok <= 0 || n_embd <= 0) {
        std::fprintf(stderr, "mhc_init: invalid arguments\n");
        std::exit(1);
    }
    const int64_t n = (int64_t) n_tok * n_embd;
    mhc_init_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(emb, x, n_embd, n);
    check_launch("mhc_init");
}

void mhc_mean(const float* x, float* y, int n_tok, int n_embd, void* stream) {
    if (!x || !y || n_tok <= 0 || n_embd <= 0) {
        std::fprintf(stderr, "mhc_mean: invalid arguments\n");
        std::exit(1);
    }
    const int64_t n = (int64_t) n_tok * n_embd;
    mhc_mean_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(x, y, n_embd, n);
    check_launch("mhc_mean");
}

}  // namespace strata::kernels
