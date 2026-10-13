// include/strata/kernels/glm_moe.hpp - GLM-5.3-Flash's MoE: the router and the clamped SwiGLU, transcribed from
// llama.cpp's build_moe_ffn and build_ffn for glm5-next (src/llama-graph.cpp, upstream).
//
// Per token, with 288 routed experts of width 2048, 8 of them used, and one shared expert of the same width:
//
//     logits  = x @ ffn_gate_inp                                    (288)   a GEMV, not here
//     probs   = sigmoid(logits)                                             expert_gating_func 2 (llama.cpp's
//     choice  = the top 8 of probs + exp_probs_b                            default for glm5-next)
//     w       = probs[choice] / max(sum, 2^-14) * 2.5                       expert_weights_norm, _scale
//     y       = sum_i w_i expert_i(x) + shared(x)                           native_moe_combine
//     expert(x) = down @ (silu(min(gate @ x, L)) * clamp(up @ x, -L, L))    L = swiglu_clamp_exp = 10
//
// THE BIAS CHOOSES, IT DOES NOT WEIGH: exp_probs_b is added to the probabilities only for the top-k (DeepSeek V3's
// e_score_correction_bias); the weights are the unbiased probabilities of the chosen experts.  glm_moe_parity shows
// both rival readings (choosing without it, weighing with it) give other answers.
//
// THE CLAMP is ggml_swiglu_clamp's, which upstream llama.cpp takes for LLM_ARCH_GLM5_NEXT in both build_moe_ffn and
// build_ffn: the gate is clamped BEFORE the SiLU, min(g, L), and up to [-L, L].  The other arches' form clamps the
// SiLU's output instead, min(silu(g), L), and differs once the gate passes L.  build_ffn reads swiglu_clamp_shexp
// for the shared expert and the 3 dense layers alike (Unsloth's file has one value, 10, for every layer).
//
// The routed experts run in native_expert_grouped_clamped (iq_kernels.hpp), on the GGUF blocks as they are.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// llama_expert_gating_func_type: 1 softmax, 2 sigmoid, 4 sqrt(softplus).
enum GlmGating : int { kGatingSoftmax = 1, kGatingSigmoid = 2, kGatingSqrtSoftplus = 4 };

/// For n_tok tokens: probs = gating(logits[t]), the k experts with the largest probs + bias (bias may be null; ties
/// to the lower expert), in descending order of that sum, into ids [n_tok][k]; their unbiased probs, normalized to
/// sum 1 (the sum clamped below at 2^-14, as llama.cpp) when `norm`, times `scale`, into weights [n_tok][k].
/// n_expert <= 1024, k <= 32.
void glm_router(const float* logits, const float* bias, int n_tok, int n_expert, int k, int gating, bool norm,
                float scale, int32_t* ids, float* weights, void* stream);

/// h = silu(min(gate, lim)) * clamp(up, -lim, lim) over n values; lim <= 0: silu(gate) * up.  For the shared expert
/// and the dense layers, whose projections are single GEMVs.
void swiglu_clamp(const float* gate, const float* up, float* h, int64_t n, float lim, void* stream);

}  // namespace strata::kernels
