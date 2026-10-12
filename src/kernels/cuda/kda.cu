// src/kernels/cuda/kda.cu - see include/strata/kernels/kda.hpp.
#include "strata/kernels/kda.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int D = kKdaHeadDim;  // head width: state rows = cols = 128
constexpr int DC = kKdaConvKernel;
constexpr int RG = 4;           // row groups of the scan's block
constexpr int RPG = D / RG;     // 32 state rows per thread
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

/// One block per head, threads (column j, row group): each thread holds 32 rows of one column of the state in
/// registers for the whole run of tokens.  Per token, row i is decayed by exp(gate[i]) BEFORE it is contracted
/// against k (llama.cpp decays first, then forms S^T k from the decayed state).
__global__ void __launch_bounds__(D * RG) kda_scan_norm_kernel(float* __restrict__ state, const float* __restrict__ qkv,
                                                               const float* __restrict__ gate,
                                                               const float* __restrict__ beta,
                                                               const float* __restrict__ z,
                                                               const float* __restrict__ gamma, float eps,
                                                               float* __restrict__ y, int heads, int n_tokens) {
    __shared__ float sk[D], sq[D], sdec[D];
    __shared__ float red_kv[RG][D];
    __shared__ float red_o[RG][D];
    __shared__ float wsum[D * RG / 32];
    __shared__ float sbeta;
    const int head = blockIdx.x;
    const int col = threadIdx.x;          // 0..127, the value column j
    const int rg = threadIdx.y;           // 0..3
    const int tid = rg * D + col;
    const int d_inner = heads * D;
    const size_t row3 = (size_t) 3 * d_inner;

    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * heads + head) * D + col;
    const size_t row_stride = (size_t) heads * D;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];

    for (int t = 0; t < n_tokens; ++t) {
        const float* qt = qkv + (size_t) t * row3 + (size_t) head * D;
        const size_t ht = (size_t) t * d_inner + (size_t) head * D;
        __syncthreads();                  // the previous token is done with the shared arrays
        if (tid < D) {
            sq[tid] = qt[tid];
            sk[tid] = qt[d_inner + tid];
            sdec[tid] = expf(gate[ht + tid]);
        } else if (tid == D) {
            sbeta = beta[(size_t) t * heads + head];
        }
        __syncthreads();
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] *= sdec[rg * RPG + r];
            kv = fmaf(s[r], sk[rg * RPG + r], kv);
        }
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (qt[2 * d_inner + col] - kv_col) * sbeta;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(sk[rg * RPG + r], delta, s[r]);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) D);
            sq_part = oc * oc;
        }
        // RMS over the head's 128 outputs: the warps of row group 0 are threads 0..127.
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) D + eps);
            y[ht + col] = oc * scale * gamma[col] * sigmoid_f(z[ht + col]);
        }
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
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
    if (!state || !qkv || !gate || !beta || !z || !o_norm || !y || heads <= 0 || n_tokens <= 0) {
        std::fprintf(stderr, "kda_scan_norm: invalid arguments\n");
        std::exit(1);
    }
    kda_scan_norm_kernel<<<(unsigned) heads, dim3(D, RG), 0, (cudaStream_t) stream>>>(state, qkv, gate, beta, z, o_norm,
                                                                                    eps, y, heads, n_tokens);
    check_launch("kda_scan_norm");
}

}  // namespace strata::kernels
