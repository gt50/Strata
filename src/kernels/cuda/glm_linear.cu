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

/// A warp per output row, its lanes striding the row; the row's sums for every token, the row read once per token
/// from cache.
__global__ void __launch_bounds__(256) f32_gemv_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                       float* __restrict__ y, int64_t n_in, int64_t n_out, int n_tok) {
    const int64_t row = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n_out) return;
    const float* wr = w + row * n_in;
    for (int t = 0; t < n_tok; ++t) {
        const float* xt = x + (int64_t) t * n_in;
        float acc = 0.0f;
        for (int64_t i = lane; i < n_in; i += 32) acc = fmaf(wr[i], xt[i], acc);
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) y[(int64_t) t * n_out + row] = acc;
    }
}

__global__ void expert_groups_decode_kernel(const int32_t* __restrict__ ids,
                                            const unsigned long long* __restrict__ expert_ptrs, int k,
                                            unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
                                            int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
                                            int32_t* __restrict__ ent_tok) {
    const int i = threadIdx.x;
    if (i < k) {
        grp_ptr[i] = expert_ptrs[ids[i]];
        grp_start[i] = i;
        ent_dst[i] = i;
        ent_tok[i] = 0;
    }
    if (i == 0) {
        grp_start[k] = k;
        *n_groups = k;
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
    if (iq_supported(type) && iq_row_bytes(type, n_in) > 0) return kLinearIq;
    return kLinearNone;
}

size_t glm_linear_scratch_bytes(int64_t max_n_in, int n_tok) {
    return (size_t) n_tok * (size_t) (max_n_in / 32) * 36 + 256;   // block_q8_1: fp16 scale and sum, 32 int8
}

void glm_linear(const GlmLinear& w, const float* x, float* y, int n_tok, void* scratch, void* stream) {
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
            f32_gemv_kernel<<<(unsigned) ((w.n_out + 7) / 8), 256, 0, s>>>((const float*) w.data, x, y, w.n_in, w.n_out,
                                                                          n_tok);
            check_launch("glm_linear (F32)");
            break;
        case kLinearBf16:
            for (int t0 = 0; t0 < n_tok; t0 += kNativeCols) {
                const int nc = std::min(kNativeCols, n_tok - t0);
                bf16_gemv_fp32_mmvf_cols(x + (int64_t) t0 * w.n_in, (const uint16_t*) w.data, y + (int64_t) t0 * w.n_out,
                                         w.n_in, w.n_out, nc, stream);
            }
            break;
        case kLinearNative:
            for (int t0 = 0; t0 < n_tok; t0 += kNativeCols) {
                const int nc = std::min(kNativeCols, n_tok - t0);
                native_quantize_q8_1(x + (int64_t) t0 * w.n_in, scratch, (int) w.n_in, nc, stream);
                native_mmvq(w.type, w.data, scratch, y + (int64_t) t0 * w.n_out, (int) w.n_in, (int) w.n_out, nc, stream);
            }
            break;
        case kLinearIq:
            quantize_q8_1_rows(x, n_tok, w.n_in, scratch, stream);
            iq_mmvq(w.type, w.data, scratch, y, (int) w.n_in, (int) w.n_out, n_tok, stream);
            break;
        default:
            break;
    }
}

void glm_expert_groups_decode(const int32_t* ids, const unsigned long long* expert_ptrs, int k,
                              unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                              int32_t* ent_tok, void* stream) {
    if (!ids || !expert_ptrs || !grp_ptr || !grp_start || !n_groups || !ent_dst || !ent_tok || k <= 0 || k > 1024) {
        std::fprintf(stderr, "glm_expert_groups_decode: invalid arguments\n");
        std::exit(1);
    }
    expert_groups_decode_kernel<<<1, 1024 < k ? 1024 : (k < 32 ? 32 : k), 0, (cudaStream_t) stream>>>(
        ids, expert_ptrs, k, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
    check_launch("glm_expert_groups_decode");
}

}  // namespace strata::kernels
