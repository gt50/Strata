// src/kernels/cuda/kpool.cu - see include/strata/kernels/kpool.hpp.
#include "strata/kernels/kpool.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int D = kKpoolDim;
constexpr int NH = kKpoolHeads;
constexpr int KP = kKpool;
constexpr int BPB = 64;        // pools scored per block of kpool_scores (8 warps x 8)
constexpr int IQS = D + 4;     // the staged query rows' stride (float4-aligned; 8 lanes of a quarter-wave on 8 banks)
static_assert(D == 128 && NH == 32 && KP == 4, "kpool kernels: a thread per channel, a lane per indexer head");

__device__ __forceinline__ float block_sum_128(float v, float* part) {   // 128 threads
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    const float s = (part[0] + part[1]) + (part[2] + part[3]);
    __syncthreads();   // part reused by the next call
    return s;
}

/// A block per token, a thread per channel: ggml_norm (mean, then the biased variance of the centred row), the norm's
/// weight and bias, and the gate copied beside it.
__global__ void __launch_bounds__(D) kpool_key_store_kernel(const float* __restrict__ k_raw, const float* __restrict__ gate,
                                                            const float* __restrict__ ln_w,
                                                            const float* __restrict__ ln_b, float eps,
                                                            float* __restrict__ cache, int64_t row0,
                                                            const int32_t* __restrict__ step) {
    __shared__ float part[D / 32];
    const int t = blockIdx.x, d = threadIdx.x;
    const float x = k_raw[(size_t) t * D + d];
    const float mean = block_sum_128(x, part) / (float) D;
    const float xc = x - mean;
    const float var = block_sum_128(xc * xc, part) / (float) D;
    const int64_t row = (step ? (int64_t) step[kStepPos] : row0) + t;
    float* out = cache + (size_t) row * (2 * D);
    out[d] = xc * rsqrtf(var + eps) * ln_w[d] + ln_b[d];
    out[D + d] = gate[(size_t) t * D + d];
}

/// A block per pool, a thread per channel: the softmax over the pool's 4 members of gate + ape, then the weighted
/// sum of their keys (ggml_soft_max: max subtracted, exp, times 1 / sum).
__global__ void __launch_bounds__(D) kpool_pool_kernel(const float* __restrict__ cache, const float* __restrict__ ape,
                                                       float* __restrict__ pooled, int64_t b0,
                                                       const int32_t* __restrict__ step) {
    int64_t b = b0 + blockIdx.x;
    if (step) {
        const int64_t n_kv = step[kStepNKv];
        if (n_kv % KP != 0) return;   // this token completes no pool
        b = n_kv / KP - 1;
    }
    const int d = threadIdx.x;
    const float* cell = cache + (size_t) b * KP * (2 * D);
    float lg[KP], mx = -INFINITY;
#pragma unroll
    for (int m = 0; m < KP; ++m) {
        lg[m] = cell[(size_t) m * (2 * D) + D + d] + ape[m * D + d];
        mx = fmaxf(mx, lg[m]);
    }
    float e[KP], sum = 0.0f;
#pragma unroll
    for (int m = 0; m < KP; ++m) {
        e[m] = expf(lg[m] - mx);
        sum += e[m];
    }
    const float inv = 1.0f / sum;
    float acc = 0.0f;
#pragma unroll
    for (int m = 0; m < KP; ++m) acc = fmaf(e[m] * inv, cell[(size_t) m * (2 * D) + d], acc);
    pooled[(size_t) b * D + d] = acc;
}

/// A block per (BPB pools, query), a warp per pool at a time and a lane per indexer head: lane h dots its staged
/// query row with the pool's key (the same 512 bytes for the whole warp, a broadcast), relu, times its weight, and
/// the warp sums the heads.
__global__ void __launch_bounds__(256) kpool_scores_kernel(const float* __restrict__ pooled, const float* __restrict__ iq,
                                                           const float* __restrict__ w, float w_scale,
                                                           const int32_t* __restrict__ steps, int64_t max_blocks,
                                                           int select_tail, float* __restrict__ scores) {
    __shared__ __align__(16) float qs[NH * IQS];
    __shared__ float ws[NH];
    const int64_t q = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int32_t* st = steps + q * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t bstart = (int64_t) blockIdx.x * BPB;
    if (bstart > n_bid || bstart >= max_blocks) return;   // uniform over the block
    for (int i = tid; i < NH * D; i += 256) qs[(i / D) * IQS + i % D] = iq[q * NH * D + i];
    if (tid < NH) ws[tid] = w[q * NH + tid] * w_scale;
    __syncthreads();
    float* out = scores + q * max_blocks;
    for (int k = warp; k < BPB; k += 8) {
        const int64_t b = bstart + k;
        if (b > n_bid || b >= max_blocks) break;
        if (b == n_bid) {   // the tail: forced in, or out
            if (lane == 0) out[b] = (select_tail && n_kv % KP != 0) ? 1e9f : -3e38f;
            continue;
        }
        const float4* key = reinterpret_cast<const float4*>(pooled + (size_t) b * D);
        const float4* qrow = reinterpret_cast<const float4*>(qs + lane * IQS);
        float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
#pragma unroll 8
        for (int i = 0; i < D / 4; ++i) {
            const float4 kv = key[i], qv = qrow[i];
            x0 = fmaf(qv.x, kv.x, x0); x1 = fmaf(qv.y, kv.y, x1);
            x2 = fmaf(qv.z, kv.z, x2); x3 = fmaf(qv.w, kv.w, x3);
        }
        float v = fmaxf((x0 + x1) + (x2 + x3), 0.0f) * ws[lane];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0) out[b] = v;
    }
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void kpool_key_store(const float* k_raw, const float* gate, const float* ln_w, const float* ln_b, float eps,
                     float* cache, int64_t row0, int n_tok, const int32_t* step, void* stream) {
    if (!k_raw || !gate || !ln_w || !ln_b || !cache || n_tok <= 0 || (step && n_tok != 1) || (!step && row0 < 0)) {
        std::fprintf(stderr, "kpool_key_store: invalid arguments\n");
        std::exit(1);
    }
    kpool_key_store_kernel<<<(unsigned) n_tok, D, 0, (cudaStream_t) stream>>>(k_raw, gate, ln_w, ln_b, eps, cache,
                                                                             row0, step);
    check_launch("kpool_key_store");
}

void kpool_pool(const float* cache, const float* ape, float* pooled, int64_t b0, int64_t n_pools, const int32_t* step,
                void* stream) {
    if (!cache || !ape || !pooled || (!step && (b0 < 0 || n_pools < 0))) {
        std::fprintf(stderr, "kpool_pool: invalid arguments\n");
        std::exit(1);
    }
    const int64_t n = step ? 1 : n_pools;
    if (n == 0) return;
    kpool_pool_kernel<<<(unsigned) n, D, 0, (cudaStream_t) stream>>>(cache, ape, pooled, b0, step);
    check_launch("kpool_pool");
}

void kpool_scores(const float* pooled, const float* iq, const float* w, float w_scale, const int32_t* steps,
                  int64_t nq, int64_t max_blocks, bool select_tail, float* scores, void* stream) {
    if (!pooled || !iq || !w || !steps || !scores || nq <= 0 || nq > 65535 || max_blocks <= 0) {
        std::fprintf(stderr, "kpool_scores: invalid arguments\n");
        std::exit(1);
    }
    const dim3 grid((unsigned) ((max_blocks + BPB - 1) / BPB), (unsigned) nq);
    kpool_scores_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(pooled, iq, w, w_scale, steps, max_blocks,
                                                                 select_tail ? 1 : 0, scores);
    check_launch("kpool_scores");
}

}  // namespace strata::kernels
