// include/strata/kernels/mhc.hpp - GLM-5.3-Flash's mHC: four residual streams per token, mixed around every attention
// and FFN sublayer.  Transcribed from llama.cpp's build_hc_pre, build_hc_sinkhorn, build_hc_post and build_hc_mean
// (upstream src/models/glm5-next.cpp, the unfused graph; DeepSeek V4's hyper-connections).  Per token, with the
// streams X[0..3] of n_embd each (X is [n_tok][4][n_embd], stream-major within a token):
//
//     mixes  = hc_fn . rms_norm(X as one row of 4 n_embd)          (24)    hc_fn F32 [24][4 n_embd], no norm weight
//     pre_i  = sigmoid(mixes_i scale[0] + base_i) + eps                    i < 4
//     post_i = 2 sigmoid(mixes_{4+i} scale[1] + base_{4+i})                i < 4
//     comb   = sinkhorn(mixes_{8..23} scale[2] + base_{8..23})     (4 x 4) comb[src][dst], element 8 + 4 src + dst
//     h      = sum_i pre_i X[i]                                             the sublayer's input (then its own norm)
//     X'[dst] = post_dst f + sum_src comb[src][dst] X[src]                  f = the sublayer's output
//
// SINKHORN, in the reference's order: a softmax over dst for each src row, + eps, a column normalization (each dst
// column divided by its sum over src, + eps), then iters - 1 rounds of a row normalization and a column one, every
// sum + eps.  The streams start as copies of the token embedding (mhc_init) and end as their mean (mhc_mean).
//
// THE NORM IS ONE SCALAR A TOKEN, so it is applied after the GEMV: mixes = (hc_fn . x) / rms(x).  hc_fn is 1.5 MB of
// F32 a sublayer (the converter keeps it exact), 3 MB a layer.
#pragma once

#include <cstdint>

namespace strata::kernels {

inline constexpr int kMhcStreams = 4;                         ///< hyper_connection.count
inline constexpr int kMhcMix = (2 + kMhcStreams) * kMhcStreams;   ///< 24: pre, post, and the 4 x 4 comb

/// Scratch floats mhc_pre needs for n_tok tokens (the GEMV's partial sums).
uint64_t mhc_pre_scratch_floats(int n_tok, int n_embd);

/// The pre half for n_tok tokens: h [n_tok][n_embd] = sum_i pre_i X[i], and the post weights [n_tok][4] and comb
/// [n_tok][16] (comb[src * 4 + dst]) that mhc_post needs; pre [n_tok][4] too when non-null.  hc_scale (3,),
/// hc_base (24,).  n_embd a multiple of 32.
void mhc_pre(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base, int n_tok, int n_embd,
             float rms_eps, float hc_eps, int sinkhorn_iters, float* scratch, float* h, float* post, float* comb,
             float* pre, void* stream);

/// The post half, in place: X[t][dst] = post[t][dst] f[t] + sum_src comb[t][src][dst] X[t][src], f [n_tok][n_embd].
void mhc_post(const float* f, const float* post, const float* comb, float* x, int n_tok, int n_embd, void* stream);

/// X[t][i] = emb[t] for the 4 streams.
void mhc_init(const float* emb, float* x, int n_tok, int n_embd, void* stream);

/// y[t] = (X[t][0] + X[t][1] + X[t][2] + X[t][3]) / 4.
void mhc_mean(const float* x, float* y, int n_tok, int n_embd, void* stream);

}  // namespace strata::kernels
