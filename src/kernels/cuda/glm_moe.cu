// src/kernels/cuda/glm_moe.cu - see include/strata/kernels/glm_moe.hpp.
#include "strata/kernels/glm_moe.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int RT = 256;        // router threads: up to 4 experts each
constexpr int MAX_E = 1024;
constexpr int MAX_K = 32;

/// (value, index) of the larger, ties to the lower index; -inf entries lose to any finite one.
__device__ __forceinline__ void arg_better(float& v, int& i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) { v = v2; i = i2; }
}

__device__ __forceinline__ float block_max(float v, float* red) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float m = red[0];
    for (int w = 1; w < RT / 32; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    return m;
}
__device__ __forceinline__ float block_sum(float v, float* red) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.0f;
    for (int w = 0; w < RT / 32; ++w) s += red[w];
    __syncthreads();
    return s;
}

/// A block per token: the gating into shared memory, then k rounds of a block-wide argmax of probs + bias, each
/// round's winner struck out; one thread forms the weights.
__global__ void __launch_bounds__(RT) glm_router_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                        int n_expert, int k, int gating, int norm, float scale,
                                                        int32_t* __restrict__ ids, float* __restrict__ weights) {
    __shared__ float probs[MAX_E], sel[MAX_E];
    __shared__ float red[RT / 32];
    __shared__ float rv[RT / 32];
    __shared__ int ri[RT / 32];
    __shared__ int chosen[MAX_K];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float* lg = logits + (size_t) t * n_expert;
    float mx = -INFINITY;
    for (int e = tid; e < n_expert; e += RT) mx = fmaxf(mx, lg[e]);
    if (gating == kGatingSoftmax) mx = block_max(mx, red);
    float sum = 0.0f;
    for (int e = tid; e < n_expert; e += RT) {
        const float x = lg[e];
        float p;
        if (gating == kGatingSoftmax) {
            p = expf(x - mx);
            sum += p;
        } else if (gating == kGatingSqrtSoftplus) {
            p = sqrtf(x > 20.0f ? x : log1pf(expf(x)));
        } else {
            p = 1.0f / (1.0f + expf(-x));
        }
        probs[e] = p;
    }
    if (gating == kGatingSoftmax) {
        const float tot = block_sum(sum, red);
        for (int e = tid; e < n_expert; e += RT) probs[e] *= 1.0f / tot;   // ggml_soft_max: times 1 / sum
    }
    __syncthreads();
    for (int e = tid; e < n_expert; e += RT) sel[e] = probs[e] + (bias ? bias[e] : 0.0f);
    __syncthreads();
    for (int r = 0; r < k; ++r) {
        float v = -INFINITY;
        int i = 0x7fffffff;
        for (int e = tid; e < n_expert; e += RT) arg_better(v, i, sel[e], e);
        for (int off = 16; off > 0; off >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, v, off);
            const int i2 = __shfl_xor_sync(0xffffffffu, i, off);
            arg_better(v, i, v2, i2);
        }
        if ((tid & 31) == 0) { rv[tid >> 5] = v; ri[tid >> 5] = i; }
        __syncthreads();
        if (tid == 0) {
            float bv = rv[0];
            int bi = ri[0];
            for (int w = 1; w < RT / 32; ++w) arg_better(bv, bi, rv[w], ri[w]);
            chosen[r] = bi;
            sel[bi] = -INFINITY;   // struck out
        }
        __syncthreads();
    }
    if (tid == 0) {
        float wsum = 0.0f;
        for (int r = 0; r < k; ++r) wsum += probs[chosen[r]];
        const float div = fmaxf(wsum, 6.103515625e-5f);   // ggml_clamp(weights_sum, 2^-14, inf)
        for (int r = 0; r < k; ++r) {
            float w = probs[chosen[r]];
            if (norm) w = w / div;
            ids[(size_t) t * k + r] = chosen[r];
            weights[(size_t) t * k + r] = w * scale;
        }
    }
}

__global__ void swiglu_clamp_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                                    int64_t n, float lim) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i], u = up[i];
    if (lim > 0.0f) {
        g = fminf(g, lim);
        u = fminf(fmaxf(u, -lim), lim);
    }
    h[i] = g / (1.0f + expf(-g)) * u;
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void glm_router(const float* logits, const float* bias, int n_tok, int n_expert, int k, int gating, bool norm,
                float scale, int32_t* ids, float* weights, void* stream) {
    if (!logits || !ids || !weights || n_tok <= 0 || n_expert <= 0 || n_expert > MAX_E || k <= 0 || k > MAX_K ||
        k > n_expert || (gating != kGatingSoftmax && gating != kGatingSigmoid && gating != kGatingSqrtSoftplus)) {
        std::fprintf(stderr, "glm_router: invalid arguments\n");
        std::exit(1);
    }
    glm_router_kernel<<<(unsigned) n_tok, RT, 0, (cudaStream_t) stream>>>(logits, bias, n_expert, k, gating, norm ? 1 : 0,
                                                                          scale, ids, weights);
    check_launch("glm_router");
}

void swiglu_clamp(const float* gate, const float* up, float* h, int64_t n, float lim, void* stream) {
    if (!gate || !up || !h || n < 0) {
        std::fprintf(stderr, "swiglu_clamp: invalid arguments\n");
        std::exit(1);
    }
    if (n == 0) return;
    swiglu_clamp_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(gate, up, h, n, lim);
    check_launch("swiglu_clamp");
}

}  // namespace strata::kernels
