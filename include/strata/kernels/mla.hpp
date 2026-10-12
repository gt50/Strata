// include/strata/kernels/mla.hpp - GLM-5.3-Flash's MLA attention (the 11 non-KDA layers and the MTP block): the
// parts between its projections, absorbed so the cache holds one latent row per token.
//
// Transcribed from llama.cpp's `build_dsa_layer` (src/models/glm5-next.cpp) and `build_attn_mha` with its `v_mla`.
// GLM's MLA has no rotary part (rope.dimension_count = 0, the loader asserts it), so a key is its latent alone.  Per
// token, with H = 64 heads, DK = DV = 256 (key_length_mla, value_length_mla) and L = 512 (kv_lora_rank):
//
//     qr      = rms_norm(x @ wq_a) * q_a_norm                     (1536)   projections and norm: not here
//     q       = qr @ wq_b                                         (H, DK)
//     latent  = rms_norm(x @ wkv_a_mqa) * kv_a_norm               (L)      mla_kv_store: the norm, into the cache
//     q_abs_h = Wk_b[h] q_h                                       (H, L)   mla_head_gemv with attn_k_b
//     s_h,j   = q_abs_h . latent_j / sqrt(DK)                              mla_decode_attn, over the SELECTED
//     o_h     = sum_j softmax(s_h)_j latent_j                     (H, L)   positions j (the k-pool indexer's)
//     out_h   = Wv_b[h] o_h                                       (H, DV)  mla_head_gemv with attn_v_b
//     y       = out @ wo                                                    not here
//
// The scale is 1 / sqrt(DK = 256), the head's width before absorption, not 1 / sqrt(L): llama.cpp's
// `kq_scale = 1 / sqrt(n_embd_head_k_mla)`.  Absorbing is exact algebra - q.(W k) = (W^T q).k - and mla_parity
// checks it against the unabsorbed form, explicit keys and values per head.
//
// WEIGHT LAYOUT: attn_k_b is ne {DK, L, H} and attn_v_b ne {L, DV, H} (conversion/glm.py splits kv_b_proj and
// transposes k_b), so both are [H][n_out][n_in] with n_in contiguous - a GEMV per head.  The kernels take them in
// BF16 (8.4 M values each a layer; Strata converts small weights once at load, as for the GDN's alpha / beta).
//
// THE CACHE: FP16 latent rows [row][L], as llama.cpp's default cache type; K and V are the same row.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The latent width every kernel here is compiled for (GLM-5.3-Flash's kv_lora_rank).
inline constexpr int kMlaLatent = 512;
/// Positions per block of mla_decode_attn's split, and heads per block.
inline constexpr int kMlaChunk = 32;
inline constexpr int kMlaHeadGroup = 8;

/// y[t][h][o] = sum_i w[h][o][i] * x[t][h][i] for n_tok tokens: q_abs = Wk_b q (n_in 256, n_out 512) and
/// out = Wv_b o (n_in 512, n_out 256).  w is BF16 bits [heads][n_out][n_in]; x and y FP32 [n_tok][heads][n].  A warp
/// per (head, output row), the row read once for all the tokens.  n_in must be a multiple of 256.
void mla_head_gemv(const float* x, const uint16_t* w, float* y, int heads, int n_in, int n_out, int n_tok,
                   void* stream);

/// latent = rms_norm(kv) * gamma for n_tok tokens, stored as FP16 bits into cache rows row0 .. row0 + n_tok - 1.
/// kv [n_tok][L] FP32 (x @ wkv_a_mqa), gamma (L,) = attn_kv_a_norm.  The norm is FP32 like build_norm's; the cache
/// keeps the rounded row, which is what the attention then reads, the current token's included.
void mla_kv_store(const float* kv, const float* gamma, float eps, uint16_t* cache, int64_t row0, int n_tok,
                  void* stream);

/// Scratch floats mla_decode_attn needs for up to `max_sel` selected positions.
uint64_t mla_decode_attn_scratch_floats(int heads, int64_t max_sel);

/// One query token: o[h] = sum_j softmax_j(q_abs[h] . c_j * scale) c_j over the n_sel selected cache rows,
/// c_j = cache[ids[j]] (ids null: rows 0 .. n_sel - 1).  The selection must already be causal and include the current
/// token's row.  q_abs [heads][L] FP32, cache FP16 [rows][L], o [heads][L] FP32.  Split-K: a block per (chunk of
/// kMlaChunk positions, kMlaHeadGroup heads) leaves a partial (max, sum, accumulator), and a second launch merges
/// them.  heads must be a multiple of kMlaHeadGroup; n_sel >= 1.
void mla_decode_attn(const float* q_abs, const uint16_t* cache, const int32_t* ids, int64_t n_sel, float scale,
                     float* scratch, float* o, int heads, void* stream);

}  // namespace strata::kernels
