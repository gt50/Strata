// include/strata/kernels/glm_linear.hpp - one GLM-5.3-Flash projection on its GGUF weight as it is, whatever the
// type, and the small pieces of glue the GLM layer needs between kernels.
//
// The type picks the kernel: F32 a warp per output row (glm_linear's own); BF16 bf16_gemv_fp32_mmvf_cols (FP32
// activations); the types native_mmvq takes (Q4_0, Q5_0, Q8_0, Q2_0, Q3_K .. Q6_K, IQ4_NL, IQ4_XS) on q8_1
// activations, llama.cpp's CUDA contract; the other i-quants through iq_mmvq, also on q8_1.  Which one a type takes is
// glm_linear_kind, so a loader can refuse a file before allocating anything.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// A projection y = W x: W is n_out rows of n_in, ggml type `type`, rows contiguous (GGUF order, ne0 = n_in).
struct GlmLinear {
    int type = -1;
    const void* data = nullptr;
    int64_t n_in = 0, n_out = 0;
};

enum GlmLinearKind : int { kLinearNone = 0, kLinearF32 = 1, kLinearBf16 = 2, kLinearNative = 3, kLinearIq = 4 };
/// The path `type` takes at this width, or kLinearNone.
GlmLinearKind glm_linear_kind(int type, int64_t n_in) noexcept;

/// Bytes of q8_1 scratch glm_linear needs for n_tok tokens of up to max_n_in inputs.
size_t glm_linear_scratch_bytes(int64_t max_n_in, int n_tok);

/// y [n_tok][n_out] = W x [n_tok][n_in].  scratch: glm_linear_scratch_bytes (unused by F32 / BF16).
void glm_linear(const GlmLinear& w, const float* x, float* y, int n_tok, void* scratch, void* stream);

/// The grouped-expert call for one token's k chosen experts (distinct, as a top-k is): group i is expert ids[i], its
/// blob at expert_ptrs[ids[i]], with the one entry i (token 0, output row i).  Read on the device, so a decode step can
/// be captured.  grp_ptr [k], grp_start [k + 1], n_groups [1], ent_dst / ent_tok [k].
void glm_expert_groups_decode(const int32_t* ids, const unsigned long long* expert_ptrs, int k,
                              unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                              int32_t* ent_tok, void* stream);

/// grp_ptr[g] = table[experts[g]] for g < n: a prompt round's blob addresses from a provider's table (both device).
void glm_expert_group_ptrs(const unsigned long long* table, const int32_t* experts, int n, unsigned long long* grp_ptr,
                           void* stream);

}  // namespace strata::kernels
