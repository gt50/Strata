// src/kernels/cuda/glm_linear.cu - see include/strata/kernels/glm_linear.hpp.
#include "strata/kernels/glm_linear.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int kNativeCols = 8;   // native_mmvq's columns a call

/// The sum of v over a block of 256, every thread getting it (red: 8 floats of shared memory, free again after).
__device__ __forceinline__ float block_sum_256(float v, float* red) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[w];
    __syncthreads();
    return s;
}

/// A block of 256 per output row, 4 values a thread a step (float4 when the row allows: n_in % 4 == 0, which a GGUF
/// F32 row of a multiple-of-32 width is); the row's sums for every token, the row read once per token from cache.  A
/// warp a row left the router's and the indexer's few rows at 20 us a call.
__global__ void __launch_bounds__(256) f32_gemv_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                       float* __restrict__ y, int64_t n_in, int64_t n_out, int n_tok) {
    __shared__ float red[8];
    const int64_t row = blockIdx.x;
    const float* wr = w + row * n_in;
    const bool vec = (n_in & 3) == 0;
    for (int t = 0; t < n_tok; ++t) {
        const float* xt = x + (int64_t) t * n_in;
        float acc = 0.0f;
        if (vec) {
            for (int64_t i = 4 * (int64_t) threadIdx.x; i < n_in; i += 4 * 256) {
                const float4 a = *reinterpret_cast<const float4*>(wr + i), b = *reinterpret_cast<const float4*>(xt + i);
                acc = fmaf(a.x, b.x, acc);
                acc = fmaf(a.y, b.y, acc);
                acc = fmaf(a.z, b.z, acc);
                acc = fmaf(a.w, b.w, acc);
            }
        } else {
            for (int64_t i = threadIdx.x; i < n_in; i += 256) acc = fmaf(wr[i], xt[i], acc);
        }
        acc = block_sum_256(acc, red);
        if (threadIdx.x == 0) y[(int64_t) t * n_out + row] = acc;
    }
}

/// v from lane (lane ^ M), M < 16: within a row of 16 lanes - on AMD a DPP row_xmask move, elsewhere a shuffle.
template <int M> __device__ __forceinline__ float lane_xor16(float v) {
    static_assert(M > 0 && M < 16, "within a row of 16 lanes");
#if defined(__HIP_PLATFORM_AMD__) && defined(__HIP_DEVICE_COMPILE__)
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x160 + M, 0xf, 0xf,
                                                                 false));
#else
    return __shfl_xor_sync(0xffffffffu, v, M);
#endif
}

__device__ __forceinline__ float bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

/// One token through BF16 rows of a few hundred values (KDA's f_b and g_b: 8192 rows of 128): 16 lanes a row, 8
/// values (16 bytes) a lane a step, 16 rows a block of 256.  bf16_gemv_fp32_mmvf's block of 64 a row ran 8192 blocks
/// of 256 bytes each: 14 us for the 2 MB on the 8060S.
constexpr int kBf16RowLanes = 16;
__global__ void __launch_bounds__(256) bf16_short_rows_kernel(const uint16_t* __restrict__ w,
                                                              const float* __restrict__ x, float* __restrict__ y,
                                                              int n_in, int64_t n_out, int n_tok) {
    const int sub = threadIdx.x & (kBf16RowLanes - 1);
    const int64_t row = (int64_t) blockIdx.x * (256 / kBf16RowLanes) + (threadIdx.x / kBf16RowLanes);
    const int64_t r = row < n_out ? row : n_out - 1;   // every lane takes part in the exchanges
    const uint16_t* wr = w + r * n_in;
    for (int t = 0; t < n_tok; ++t) {   // each token's sums as one token's call makes them (the row from cache after)
        const float* xt = x + (int64_t) t * n_in;
        float acc = 0.0f;
        for (int c = sub * 8; c < n_in; c += kBf16RowLanes * 8) {
            const uint4 wv = *reinterpret_cast<const uint4*>(wr + c);
            const float4 x0 = *reinterpret_cast<const float4*>(xt + c);
            const float4 x1 = *reinterpret_cast<const float4*>(xt + c + 4);
            acc = fmaf(bf16_lo(wv.x), x0.x, acc);
            acc = fmaf(bf16_hi(wv.x), x0.y, acc);
            acc = fmaf(bf16_lo(wv.y), x0.z, acc);
            acc = fmaf(bf16_hi(wv.y), x0.w, acc);
            acc = fmaf(bf16_lo(wv.z), x1.x, acc);
            acc = fmaf(bf16_hi(wv.z), x1.y, acc);
            acc = fmaf(bf16_lo(wv.w), x1.z, acc);
            acc = fmaf(bf16_hi(wv.w), x1.w, acc);
        }
        acc += lane_xor16<8>(acc);
        acc += lane_xor16<4>(acc);
        acc += lane_xor16<2>(acc);
        acc += lane_xor16<1>(acc);
        if (sub == 0 && row < n_out) y[(int64_t) t * n_out + row] = acc;
    }
}

/// A block of 256 per row: the squares summed, then x = x w / rms in place (each thread its own values).
__global__ void __launch_bounds__(256) glm_rms_norm_kernel(float* __restrict__ x, const float* __restrict__ w,
                                                           int64_t cols, float eps) {
    __shared__ float red[8];
    float* r = x + (int64_t) blockIdx.x * cols;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += 256) ss = fmaf(r[c], r[c], ss);
    const float inv = rsqrtf(block_sum_256(ss, red) / (float) cols + eps);
    for (int64_t c = threadIdx.x; c < cols; c += 256) r[c] = (w ? r[c] * w[c] : r[c]) * inv;
}

/// The larger of two (value, index) picks; equal values: the lower index.
__device__ __forceinline__ void pick_max(float& v, int& i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) {
        v = v2;
        i = i2;
    }
}

__global__ void __launch_bounds__(1024) glm_argmax_kernel(const float* __restrict__ logits, int n,
                                                          int32_t* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* x = logits + (int64_t) blockIdx.x * n;
    float v = -INFINITY;
    int i = 0x7fffffff;
    for (int c = threadIdx.x; c < n; c += 1024) {
        const float a = x[c];
        if (a > v) {   // ascending c: the first of equal values stays; NaN is never > anything
            v = a;
            i = c;
        }
    }
    for (int off = 16; off > 0; off >>= 1) pick_max(v, i, __shfl_xor_sync(0xffffffffu, v, off),
                                                    __shfl_xor_sync(0xffffffffu, i, off));
    if ((threadIdx.x & 31) == 0) {
        sv[threadIdx.x >> 5] = v;
        si[threadIdx.x >> 5] = i;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        v = sv[threadIdx.x];
        i = si[threadIdx.x];
        for (int off = 16; off > 0; off >>= 1) pick_max(v, i, __shfl_xor_sync(0xffffffffu, v, off),
                                                        __shfl_xor_sync(0xffffffffu, i, off));
        if (threadIdx.x == 0) out[blockIdx.x] = i == 0x7fffffff ? 0 : i;
    }
}

/// One block of E threads: each entry's expert's first entry (shared), then its group, its slot (the entries of the
/// groups before, and of its own before it) - every count O(E), E <= 1024.
__global__ void __launch_bounds__(1024) expert_groups_window_kernel(const int32_t* __restrict__ ids,
                                                                    const unsigned long long* __restrict__ expert_ptrs,
                                                                    int k, int E, unsigned long long* __restrict__ grp_ptr,
                                                                    int32_t* __restrict__ grp_start,
                                                                    int32_t* __restrict__ n_groups,
                                                                    int32_t* __restrict__ ent_dst,
                                                                    int32_t* __restrict__ ent_tok) {
    __shared__ int first[1024];
    const int i = threadIdx.x;
    int e = -1, f = i;
    if (i < E) {
        e = ids[i];
        for (int j = 0; j < i; ++j)
            if (ids[j] == e) {
                f = j;
                break;
            }
        first[i] = f;
    }
    __syncthreads();
    if (i < E) {
        int g = 0, start = 0, p = 0;
        for (int j = 0; j < E; ++j) {
            const int fj = first[j];
            if (fj == j && j < f) ++g;      // a group before this one
            if (fj < f) ++start;            // an entry of a group before this one
            if (fj == f && j < i) ++p;      // an entry of this group before this one
        }
        ent_tok[start + p] = i / k;
        ent_dst[start + p] = i;
        if (f == i) {
            grp_ptr[g] = expert_ptrs[e];
            grp_start[g] = start;
        }
    }
    if (i == 0) {
        int ng = 0;
        for (int j = 0; j < E; ++j) ng += first[j] == j ? 1 : 0;
        grp_start[ng] = E;
        *n_groups = ng;
    }
}

__global__ void expert_groups_decode_kernel(const int32_t* __restrict__ ids,
                                            const unsigned long long* __restrict__ expert_ptrs, int k,
                                            unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
                                            int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
                                            int32_t* __restrict__ ent_tok, int n_tok) {
    const int i = threadIdx.x, e = n_tok * k;
    if (i < e) {
        grp_ptr[i] = expert_ptrs[ids[i]];
        grp_start[i] = i;
        ent_dst[i] = i;
        ent_tok[i] = i / k;
    }
    if (i == 0) {
        grp_start[e] = e;
        *n_groups = e;
    }
}

__global__ void expert_group_ptrs_kernel(const unsigned long long* __restrict__ table,
                                         const int32_t* __restrict__ experts, int n,
                                         unsigned long long* __restrict__ grp_ptr) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) grp_ptr[i] = table[experts[i]];
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

GlmLinearKind glm_linear_kind(int type, int64_t n_in) noexcept {
    if (n_in <= 0) return kLinearNone;
    if (type == 0) return kLinearF32;
    if (type == 30) return kLinearBf16;
    if (n_in % 32 != 0) return kLinearNone;
    if (native_mmvq_supported(type)) {
        // the 256-value block types need whole blocks; the 32-value ones (Q4_0, Q5_0, Q8_0, IQ4_NL) and Q2_0's 64 less
        const bool small = type == 2 || type == 6 || type == 8 || type == 20;
        if (small || (type == 42 ? n_in % 64 == 0 : n_in % 256 == 0)) return kLinearNative;
        return kLinearNone;
    }
    if (iq_mmvq_supported(type) && iq_row_bytes(type, n_in) > 0) return kLinearIq;
    return kLinearNone;
}

size_t glm_linear_scratch_bytes(int64_t max_n_in, int n_tok) {
    return (size_t) n_tok * (size_t) (max_n_in / 32) * 36 + 256;   // block_q8_1: fp16 scale and sum, 32 int8
}

void glm_linear(const GlmLinear& w, const float* x, float* y, int n_tok, void* scratch, void* stream, bool quantized) {
    const GlmLinearKind kind = glm_linear_kind(w.type, w.n_in);
    if (!w.data || !x || !y || n_tok <= 0 || w.n_out <= 0 || kind == kLinearNone ||
        ((kind == kLinearNative || kind == kLinearIq) && !scratch)) {
        std::fprintf(stderr, "glm_linear: type %d, %lld x %lld: invalid arguments or an unsupported type\n", w.type,
                     (long long) w.n_out, (long long) w.n_in);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    switch (kind) {
        case kLinearF32:
            f32_gemv_kernel<<<(unsigned) w.n_out, 256, 0, s>>>((const float*) w.data, x, y, w.n_in, w.n_out, n_tok);
            check_launch("glm_linear (F32)");
            break;
        case kLinearBf16:
            if (n_tok <= kNativeCols && w.n_in % (kBf16RowLanes * 8) == 0 && w.n_in <= 2048 &&
                ((uintptr_t) w.data & 15) == 0 && ((uintptr_t) x & 15) == 0) {   // short rows, 16-byte loads
                bf16_short_rows_kernel<<<(unsigned) ((w.n_out + 15) / 16), 256, 0, s>>>(
                    (const uint16_t*) w.data, x, y, (int) w.n_in, w.n_out, n_tok);
                check_launch("glm_linear (BF16, short rows)");
                break;
            }
            for (int t0 = 0; t0 < n_tok; t0 += kNativeCols) {
                const int nc = std::min(kNativeCols, n_tok - t0);
                bf16_gemv_fp32_mmvf_cols(x + (int64_t) t0 * w.n_in, (const uint16_t*) w.data, y + (int64_t) t0 * w.n_out,
                                         w.n_in, w.n_out, nc, stream);
            }
            break;
        case kLinearNative:
            for (int t0 = 0; t0 < n_tok; t0 += kNativeCols) {
                const int nc = std::min(kNativeCols, n_tok - t0);
                if (!quantized || n_tok > kNativeCols)
                    native_quantize_q8_1(x + (int64_t) t0 * w.n_in, scratch, (int) w.n_in, nc, stream);
                native_mmvq(w.type, w.data, scratch, y + (int64_t) t0 * w.n_out, (int) w.n_in, (int) w.n_out, nc, stream);
            }
            break;
        case kLinearIq:
            if (!quantized) quantize_q8_1_rows(x, n_tok, w.n_in, scratch, stream);
            iq_mmvq(w.type, w.data, scratch, y, (int) w.n_in, (int) w.n_out, n_tok, stream);
            break;
        default:
            break;
    }
}

void glm_rms_norm(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    if (!x || rows <= 0 || cols <= 0 || rows > 0x7fffffff) {
        std::fprintf(stderr, "glm_rms_norm: invalid arguments\n");
        std::exit(1);
    }
    glm_rms_norm_kernel<<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, w, cols, eps);
    check_launch("glm_rms_norm");
}

void glm_expert_groups_decode(const int32_t* ids, const unsigned long long* expert_ptrs, int k,
                              unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                              int32_t* ent_tok, void* stream, int n_tok) {
    const int e = n_tok * k;
    if (!ids || !expert_ptrs || !grp_ptr || !grp_start || !n_groups || !ent_dst || !ent_tok || k <= 0 || n_tok <= 0 ||
        e > 1024) {
        std::fprintf(stderr, "glm_expert_groups_decode: invalid arguments\n");
        std::exit(1);
    }
    expert_groups_decode_kernel<<<1, e < 32 ? 32 : e, 0, (cudaStream_t) stream>>>(ids, expert_ptrs, k, grp_ptr,
                                                                                 grp_start, n_groups, ent_dst,
                                                                                 ent_tok, n_tok);
    check_launch("glm_expert_groups_decode");
}

void glm_expert_groups_window(const int32_t* ids, const unsigned long long* expert_ptrs, int k, int n_tok,
                              unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                              int32_t* ent_tok, void* stream) {
    const int e = n_tok * k;
    if (!ids || !expert_ptrs || !grp_ptr || !grp_start || !n_groups || !ent_dst || !ent_tok || k <= 0 || n_tok <= 0 ||
        e > 1024) {
        std::fprintf(stderr, "glm_expert_groups_window: invalid arguments\n");
        std::exit(1);
    }
    expert_groups_window_kernel<<<1, e < 32 ? 32 : (e + 31) / 32 * 32, 0, (cudaStream_t) stream>>>(
        ids, expert_ptrs, k, e, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
    check_launch("glm_expert_groups_window");
}

void glm_argmax_rows(const float* logits, int rows, int n, int32_t* out, void* stream) {
    if (!logits || !out || rows <= 0 || n <= 0) {
        std::fprintf(stderr, "glm_argmax_rows: invalid arguments\n");
        std::exit(1);
    }
    glm_argmax_kernel<<<(unsigned) rows, 1024, 0, (cudaStream_t) stream>>>(logits, n, out);
    check_launch("glm_argmax_rows");
}

void glm_expert_group_ptrs(const unsigned long long* table, const int32_t* experts, int n, unsigned long long* grp_ptr,
                           void* stream) {
    if (n <= 0) return;
    if (!table || !experts || !grp_ptr) {
        std::fprintf(stderr, "glm_expert_group_ptrs: invalid arguments\n");
        std::exit(1);
    }
    expert_group_ptrs_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(table, experts, n, grp_ptr);
    check_launch("glm_expert_group_ptrs");
}

}  // namespace strata::kernels
