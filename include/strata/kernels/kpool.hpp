// include/strata/kernels/kpool.hpp - GLM-5.3-Flash's k-pool indexer: which cached cells an MLA layer attends to.
//
// Transcribed from llama.cpp's `build_kpool_select` (src/models/glm5-next.cpp).  The projections (indexer.attn_k,
// indexer.kpool_gate, indexer.attn_q_b, indexer.proj) are GEMVs and not here.  With kpool = 4, D = 128
// (indexer.key_length) and 32 indexer heads:
//
//     key_c   = LayerNorm(x @ attn_k) * k_norm + k_norm.bias   (D)   kpool_key_store: with the gate, into the
//     gate_c  = x @ kpool_gate                                 (D)   indexer cache, a row per cell
//     pooled_b[d] = sum_m softmax_m(gate_{4b+m}[d] + ape[m][d]) key_{4b+m}[d]   kpool_pool, once a pool's 4 cells
//                                                                               are all cached
//     score_b = sum_h w_h relu(iq_h . pooled_b),  w = (x @ proj) / sqrt(D * 32)  kpool_scores
//     selected cells = the 4 cells of each of the top min(n_pool, top_k / 4) pools, plus the incomplete tail's cells
//
// The softmax runs over a pool's 4 MEMBERS, per channel d (llama.cpp permutes the gate to [kpool, D] and soft-maxes
// rows of kpool); `ape` is indexer.kpool_ape, ne {D, kpool}: ape[m * D + d].
//
// THE SELECTION IS qsa_block_topk (qsa_select.hpp), unchanged.  QSA's rule is the `width` highest-scoring cells,
// each cell scoring its block's score, ties to the lower index, the tail block forced in with +1e9; at kpool = 4
// that is GLM's rule exactly when the width is GLM's live count, 4 min(n_pool, top_k / 4) + tail cells - whole
// pools, never part of one (QSA's own width, min(n_kv, top_k + 3), would take up to 3 cells of a 513th block).  The
// ids come back ascending, where llama.cpp orders the pools by score; attention is a softmax over the set, so the
// order changes only the summation order.
//
// THE STEP: as QSA's (qsa.hpp), four int32 in device memory per query - pos, n_kv = pos + 1, n_bid = n_kv / 4 (the
// completed pools), width - so a decode step can be a captured graph; kpool_step fills them.
//
// INDEXER CACHE: FP32 (key | gate), 256 floats a cell; pooled keys FP32, 128 a pool.  FP32 for QSA's reason: a
// selection boundary is where a rounded score flips a pool in or out.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

inline constexpr int kKpoolDim = 128;    ///< indexer.key_length
inline constexpr int kKpoolHeads = 32;   ///< indexer.head_count
inline constexpr int kKpool = 4;         ///< indexer.kpool (the same 4 as QSA's idx_block, which the top-k needs)

/// The step for the query at `pos`: GLM's width, 4 min(n_bid, top_k / 4) + (select_tail ? n_kv - 4 n_bid : 0).
/// Below top_k + 4 cells (with the tail) the selection is every cell.
inline void kpool_step(int64_t pos, int64_t top_k, bool select_tail, int32_t* step) {
    const int64_t n_kv = pos + 1, n_bid = n_kv / kKpool;
    const int64_t pools = n_bid < top_k / kKpool ? n_bid : top_k / kKpool;
    step[kStepPos] = (int32_t) pos;
    step[kStepNKv] = (int32_t) n_kv;
    step[kStepNBid] = (int32_t) n_bid;
    step[kStepWidth] = (int32_t) (kKpool * pools + (select_tail ? n_kv - kKpool * n_bid : 0));
}

/// The QsaShapes qsa_block_topk reads for GLM's selection (idx_block 4, idx_top_k).
inline QsaShapes kpool_topk_shapes(int64_t top_k) {
    QsaShapes s;
    s.idx_n_head = kKpoolHeads;
    s.idx_dim = kKpoolDim;
    s.idx_block = kKpool;
    s.idx_top_k = top_k;
    s.page_size = 1;
    return s;
}

/// For n_tok tokens: cache[row][0, D) = LayerNorm(k_raw) * ln_w + ln_b (mean and biased variance, FP32), and
/// cache[row][D, 2D) = gate.  Rows row0 + t; with `step` (device, one token) the row is step[kStepPos] instead, so a
/// captured decode step writes the right one.  k_raw, gate [n_tok][D]; cache [rows][2D].
void kpool_key_store(const float* k_raw, const float* gate, const float* ln_w, const float* ln_b, float eps,
                     float* cache, int64_t row0, int n_tok, const int32_t* step, void* stream);

/// pooled[b] for the pools b0 .. b0 + n_pools - 1, from their 4 cached cells each.  With `step` (device; b0 and
/// n_pools ignored) the pool the step's token completes, n_bid - 1, and nothing when n_kv % 4 != 0.
/// ape [kpool][D]; pooled [pools][D].
void kpool_pool(const float* cache, const float* ape, float* pooled, int64_t b0, int64_t n_pools, const int32_t* step,
                void* stream);

/// scores[q][b] = w_scale sum_h w[q][h] relu(iq[q][h] . pooled[b]) for each query's completed pools b < n_bid, and
/// the tail block n_bid +1e9 (select_tail and cells in it) or -3e38 (not selected).  iq [nq][32][D], w [nq][32] (the
/// raw x @ indexer.proj; w_scale = 1 / sqrt(D * 32)), steps [nq][kStepCount], scores [nq][max_blocks].
void kpool_scores(const float* pooled, const float* iq, const float* w, float w_scale, const int32_t* steps,
                  int64_t nq, int64_t max_blocks, bool select_tail, float* scores, void* stream);

}  // namespace strata::kernels
