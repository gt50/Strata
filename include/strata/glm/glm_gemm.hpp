// include/strata/glm/glm_gemm.hpp - GLM-5.3-Flash's prompt projections and experts as matrix products, the way
// llama.cpp's batched CUDA / HIP path computes them:
//   - a quantized weight through llama.cpp's MMQ (vendored ggml-cuda mmq.cuh, Strata's prefill::mmq): the weights
//     stay quantized, the activations are rounded to MMQ's q8_1 (q = round(x * (127 / amax)); the scale kept in FP32 for
//     the D4 types - Q8_0, Q5_0, Q3_K, Q6_K, the i-quants - and in FP16 with the sum for the DS4 ones - Q4_0, Q4_1,
//     Q5_1, Q4_K, Q5_K), int8 dot products
//   - a BF16 weight through Strata's prefill::Gemm (hipBLAS / cuBLAS), the activations rounded to BF16
//   - an F32 weight, or a type MMQ does not cover in this build, through glm_linear's GEMV, as at decode
// The experts of a prompt chunk run as two MMQ products over their blobs ([gate | up] then down), the rows grouped
// by expert, when the blobs lie a fixed stride apart (resident: GlmLayerWeights::expert_base / expert_stride; a
// cache: its slots) and MMQ covers their types.
//
// glm_prompt_path / glm_prompt_experts_mmq are the choice itself, so a reference can round as the device did.
//
// Measured on the Radeon 8060S (gfx1151, Windows; glm_prompt_parity --bench T), at GLM-5.3-Flash's sizes, against the
// GEMV path these replace (glm_linear, 8 tokens a call; the grouped expert kernel):
//                                     64 tokens          256 tokens          1024 tokens
//   4096 -> 8192 IQ4_XS, MMQ          0.36 ms (2.9x)     0.86 ms (4.6x)      3.13 ms (5.2x)
//   4096 -> 8192 Q8_0, MMQ            0.38 ms (3.5x)     1.0 ms (4.9-5.3x)   3.44 ms (6.4x)
//   4096 -> 8192 BF16, BF16 GEMM      1.90 ms (3.4x)     4.82 ms (5.4x)      19.95 ms (5.2x)
//   one MoE layer, 288 IQ4_XS experts 19.9 ms (1.0x)     22.7 ms (1.8-1.9x)  39.0 ms (3.3x)
// The experts are bound by reading them (3.3-3.85 GB a layer) until a chunk is long enough to amortize the read:
// a prompt wants large chunks (GlmModel's max_chunk).  BF16 GEMM's accuracy against double on BF16 activations:
// 2.0-2.5e-7 for 2 to 64 rows (glm_prompt_parity --gemm); one row goes to the GEMV (rocBLAS has no kernel for it here).
#pragma once

#include "strata/kernels/glm_linear.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>
#include <string>

namespace strata::glm {

using kernels::GlmLinear;

enum class GlmProjPath { Gemv, Mmq, Bf16Gemm };

/// The prompt path's product for a weight of `type`, [n_out][n_in].
GlmProjPath glm_prompt_path(int type, int64_t n_in, int64_t n_out);
/// MMQ's activation scale for a weight of `type`: true = FP16 (DS4), false = FP32 (D4).
bool glm_mmq_fp16_scale(int type);
/// Whether a prompt's experts of this layout, blobs `stride` bytes apart, run through MMQ.
bool glm_prompt_experts_mmq(const kernels::NativeExpertLayout& L, uint64_t stride);

/// The buffers and handles of the prompt products, for chunks of up to max_tokens tokens.
class GlmPromptGemm {
public:
    GlmPromptGemm() = default;
    ~GlmPromptGemm();
    GlmPromptGemm(const GlmPromptGemm&) = delete;
    GlmPromptGemm& operator=(const GlmPromptGemm&) = delete;

    /// max_in: the widest projection input; n_embd / n_ff / k: the experts' (k 0: no MoE).
    bool init(int64_t max_tokens, int64_t max_in, int64_t n_embd, int64_t n_ff, int64_t k, std::string& err);

    /// y [n][n_out] = x [n][n_in] . W^T by glm_prompt_path's product (`q8`: glm_linear's scratch for the GEMV).
    void linear(const GlmLinear& w, const float* x, float* y, int n, void* q8, void* stream);

    /// The experts of `rows` entries in one pass: blob s at base + s * stride for s < n_slots; the rows grouped by
    /// slot (bounds [n_slots + 1], device), row r reading token tok[r]'s h and writing parts row ent[r] (tok and ent on
    /// the device); max_rows the most rows one slot has.  h [tokens][n_embd]; parts [entries][n_embd].
    void experts(const kernels::NativeExpertLayout& L, const uint8_t* base, uint64_t stride, int n_slots,
                 const int32_t* bounds, const int32_t* tok, const int32_t* ent, int rows, int max_rows, const float* h,
                 float* parts, float swiglu_limit, void* stream);

private:
    void* mmq_ = nullptr;      ///< prefill::mmq::Context
    void* gemm_ = nullptr;     ///< prefill::Gemm, bound to gemm_stream_
    void* gemm_stream_ = nullptr;
    void* xq_ = nullptr;
    void* xq2_ = nullptr;
    int32_t* ident_ = nullptr;
    int32_t* bounds1_ = nullptr;
    uint16_t* xb_ = nullptr;
    float *gu_ = nullptr, *hh_ = nullptr;
    int64_t max_tokens_ = 0, max_in_ = 0, n_embd_ = 0, n_ff_ = 0, k_ = 0;
    void release();
};

}  // namespace strata::glm
