// include/strata/kernels/kda.hpp - GLM-5.3-Flash's KDA (Kimi delta attention) linear layer: everything between its
// projections, for one token (decode) or a run of tokens of one sequence (prefill).
//
// Transcribed from llama.cpp's `build_kda_layer` and `glm5_conv1d` (src/models/glm5-next.cpp) and the KDA branch of
// `ggml_compute_forward_gated_delta_net` (ggml-cpu/ops.cpp).  The projections (wq, wk, wv, f_a, f_b, beta, g_a, g_b,
// wo) are GEMVs/GEMMs and do not live here.  Per token, with H heads of D = 128 and d_inner = H * D:
//
//     q, k, v  = silu(causal_conv4(x @ w{q,k,v}))         three separate convs, then q and k L2-normalised per head
//     gate     = lb * sigmoid(-ssm_a[h] * (f_b(f_a(x)) + dt_bias))     (kda.gate_lower_bound = lb, GLM: -5)
//              =      ssm_a[h] * softplus(f_b(f_a(x)) + dt_bias)       (no lower bound in the file)
//     beta     = sigmoid(x @ w_beta)                       (H,): gdn_beta_gate, unchanged
//     S        = diag(exp(gate[h])) S                      per KEY channel i - the one change from GDN
//     S       += k (beta (v - S^T k))^T
//     o        = S^T q / sqrt(D)
//     y        = rms_norm(o) * o_norm * sigmoid(g_b(g_a(x)))   per head, o_norm is (D,) shared by every head
//
// `ssm_a` holds -exp(A_log) (the converter writes it so), so `-ssm_a * u = exp(A_log) * u` and the gate lies in
// (lb, 0): the decay exp(gate) is in (e^lb, 1), never a hard reset (in float the sigmoid saturates to the ends).
//
// WHY THE GDN DESIGN STILL APPLIES.  GDN decays the whole (D, D) state of a head by one scalar; KDA decays row i
// (the KEY channel, the axis contracted against k and q) by exp(gate[h][i]).  Every line of the step still touches
// only one value column j of one head, so a thread can own a column end to end, exactly as in gdn.hpp - it just
// multiplies each row by that row's own decay.  Decaying COLUMNS instead (the value axis j) is the plausible wrong
// reading; it keeps every shape, and kda_parity asserts the two are told apart.
//
// STATE LAYOUT: Strata's GDN layout, (D rows i, H, D cols j), `state[(i * H + h) * D + j]`.  ggml stores a head's
// state transposed (`s[j * D + i] = S[i][j]`); the numbers are the same, the layout is not, and a mix-up is silent.
//
// Q/K/V/Z ROW LAYOUT: token t's q, k and v are one row of a [T][3 * d_inner] buffer, q then k then v, each
// (H, D) head-major - the order llama.cpp's conv state uses.  gate, z (= g_b(g_a(x))) and y are [T][d_inner];
// beta is [T][H].
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The KDA head width every kernel here is compiled for (GLM-5.3-Flash's kda.head_dim).
inline constexpr int kKdaHeadDim = 128;
/// The conv width the conv kernel is compiled for (ssm.conv_kernel).
inline constexpr int kKdaConvKernel = 4;

/// The causal conv + SiLU over q, k and v, then the L2 norm of every q and k head (x / sqrt(sum(x^2) + eps), FLA's
/// and `build_gdn_l2_norm`'s reading; GLM uses eps = 1e-6), for `n_tokens` consecutive tokens of one sequence.
///
///   history (3 * d_inner, d_conv - 1) per channel, OLDEST first, channels in [q | k | v] order; updated in place
///                                      to the last d_conv - 1 inputs
///   x       [T][3 * d_inner]          the projections, q | k | v
///   w_q/k/v (d_inner, d_conv)         the GGUF tensors as stored: `w[c * d_conv + i]` is tap i of channel c
///   out     [T][3 * d_inner]          may not alias x or history
void kda_conv_l2(float* history, const float* x, const float* w_q, const float* w_k, const float* w_v, float* out,
                 int heads, int n_tokens, float eps, void* stream);

/// The per-channel log decay, elementwise over [T][d_inner]: `raw` is f_b(f_a(x)) before the bias.
///
///   has_lower_bound:  gate = lower_bound * sigmoid(-ssm_a[h] * (raw + dt_bias))
///   otherwise:        gate = ssm_a[h] * softplus(raw + dt_bias)
///
/// `dt_bias` is (d_inner,), `ssm_a` is (H,) and holds -exp(A_log).  `gate` may alias `raw`.
void kda_gate(const float* raw, const float* dt_bias, const float* ssm_a, float* gate, int heads, int n_tokens,
              bool has_lower_bound, float lower_bound, void* stream);

/// The recurrence and the output norm for `n_tokens` consecutive tokens of one sequence, the state carried in
/// registers from one token to the next: one block per head, state read once and written once per launch.
///
///   state  (D, H, D) f32, updated in place
///   qkv    [T][3 * d_inner]  q and k L2-normalised, q NOT scaled (the 1/sqrt(D) is applied to the readout)
///   gate   [T][d_inner]      kda_gate's output (log decay, <= 0)
///   beta   [T][H]            already sigmoided
///   z      [T][d_inner]      the output gate's pre-activation, g_b(g_a(x))
///   o_norm (D,)              ssm_norm, the same for every head
///   y      [T][d_inner]      rms_norm(o, eps) * o_norm * sigmoid(z); may not alias any input
void kda_scan_norm(float* state, const float* qkv, const float* gate, const float* beta, const float* z,
                   const float* o_norm, float eps, float* y, int heads, int n_tokens, void* stream);

}  // namespace strata::kernels
