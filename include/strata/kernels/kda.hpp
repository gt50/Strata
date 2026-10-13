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
///   update_history false: the history is read, not changed - a verify window, whose tokens may not all stay
///                     (kda_conv_advance then moves it over the ones that do)
void kda_conv_l2(float* history, const float* x, const float* w_q, const float* w_k, const float* w_v, float* out,
                 int heads, int n_tokens, float eps, void* stream, bool update_history = true);

/// The history after n_tokens more inputs x [T][3 * d_inner] (the raw projections kda_conv_l2 read): what
/// kda_conv_l2 with update_history leaves, without the conv.
void kda_conv_advance(float* history, const float* x, int heads, int n_tokens, void* stream);

/// The per-channel log decay, elementwise over [T][d_inner]: `raw` is f_b(f_a(x)) before the bias.
///
///   has_lower_bound:  gate = lower_bound * sigmoid(-ssm_a[h] * (raw + dt_bias))
///   otherwise:        gate = ssm_a[h] * softplus(raw + dt_bias)
///
/// `dt_bias` is (d_inner,), `ssm_a` is (H,) and holds -exp(A_log).  `gate` may alias `raw`.
void kda_gate(const float* raw, const float* dt_bias, const float* ssm_a, float* gate, int heads, int n_tokens,
              bool has_lower_bound, float lower_bound, void* stream);

/// The recurrence and the output norm for `n_tokens` consecutive tokens of one sequence, the state carried in
/// registers from one token to the next: a block per (head, 32 value columns), 8 row groups of 16 rows, the state read
/// once and written once per launch; then the norm.  On a Radeon 8060S, 64 heads: 44.7 us a decode token (one block a
/// head on the (D, H, D) layout: 59 us), and 3.7-3.9 us a token over a prompt of 128 to 8192.
///
///   state  (H, D, D) f32, updated in place: row i (key channel), column j (value channel) of head h at
///          (h D + i) D + j - each head's 64 KB in one piece
///   qkv    [T][3 * d_inner]  q and k L2-normalised, q NOT scaled (the 1/sqrt(D) is applied to the readout)
///   gate   [T][d_inner]      kda_gate's output (log decay, <= 0)
///   beta   [T][H]            already sigmoided
///   z      [T][d_inner]      the output gate's pre-activation, g_b(g_a(x))
///   o_norm (D,)              ssm_norm, the same for every head
///   y      [T][d_inner]      rms_norm(o, eps) * o_norm * sigmoid(z); may not alias any input
///
/// write_state false: the state is read, not written back - y as with it (a verify window's; the tokens that stay are
/// scanned again with it, which leaves the state bitwise as one-token calls would: the per-token arithmetic is the same
/// for any n_tokens)
void kda_scan_norm(float* state, const float* qkv, const float* gate, const float* beta, const float* z,
                   const float* o_norm, float eps, float* y, int heads, int n_tokens, void* stream,
                   bool write_state = true);

/// Tokens per chunk of kda_scan_norm_chunked, and per super-block (one prep and one scan launch each).
inline constexpr int kKdaChunk = 32;
inline constexpr int kKdaSuperBlock = 512;

/// The scratch kda_scan_norm_chunked needs, in floats: per chunk of the first super-block and per head, the chunk's
/// T and P (kKdaChunk^2 each) and its cumulative gate (kKdaChunk x 128).  It stops growing at kKdaSuperBlock tokens
/// (64 heads: 25.2 MB).
size_t kda_chunked_scratch_floats(int heads, int n_tokens);

/// kda_scan_norm in chunks of kKdaChunk tokens: the WY form of the delta rule as flash-linear-attention computes it,
/// with KDA's per-channel decay inside every product, all in FP32.  Per head, with Gam_t the gate's sum per key
/// channel from the chunk's start to t (inclusive), (x) the elementwise product, and S0 the state before the chunk:
///
///     A[t][i] = beta_t sum_c k_t[c] k_i[c] exp(Gam_t[c] - Gam_i[c])   (i < t),   T = (I + A)^-1
///     P[t][i] =        sum_c q_t[c] k_i[c] exp(Gam_t[c] - Gam_i[c])   (i <= t)
///     Y_t     = beta_t (v_t - S0^T (exp(Gam_t) (x) k_t)),   D = T Y   (the delta rule's correction per token)
///     o_t     = S0^T (exp(Gam_t) (x) q_t) + sum_{i <= t} P[t][i] D_i
///     S       = diag(exp(Gam_last)) S0 + sum_t (exp(Gam_last - Gam_t) (x) k_t) D_t^T
///
/// Every exponent is <= 0 (gates are <= 0), so nothing overflows whatever the lower bound.  The same state and
/// outputs as kda_scan_norm up to FP32 rounding (the sums run in another order; kda_parity: largest difference over
/// largest value 3.5e-6 at 2048 tokens).  Same arguments, plus `scratch` (kda_chunked_scratch_floats(heads,
/// n_tokens) floats, the caller's); y is written twice (o, then the norm).
///
/// Which to use is the caller's choice, by measurement.  On a Radeon 8060S (gfx1151, 40 CUs; kda_parity --bench, 64
/// heads, the two launches alternating, host clock): kda_scan_norm takes 0.85x this one's time at 32 tokens, 0.82x at
/// 128, 0.55x at 512, 0.62x at 2048, 0.64x at 8192 (this one 4.6-6.7 us a token: a prep of about 2.5 once a prompt
/// outgrows the cache, and a scan of about 3.5) - so GLM's layer runs kda_scan_norm for prompts too.  (Against the
/// one-block-a-head kda_scan_norm before it, this one was 1.02-1.27x faster, timed by events.)  Not yet run on a CUDA
/// card.
void kda_scan_norm_chunked(float* state, const float* qkv, const float* gate, const float* beta, const float* z,
                           const float* o_norm, float eps, float* y, int heads, int n_tokens, float* scratch,
                           void* stream);

}  // namespace strata::kernels
