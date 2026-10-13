// include/strata/glm/glm_layer.hpp - one GLM-5.3-Flash layer at decode: the kernels of kda.hpp, mla.hpp, kpool.hpp,
// glm_moe.hpp and mhc.hpp in the order llama.cpp's glm5-next graph runs them (upstream src/models/glm5-next.cpp):
//
//     h = mhc_pre(X, hc_attn);  h = rms_norm(h) * attn_norm
//     f = KDA(h)   or   MLA(h) over the k-pool indexer's selection
//     X = mhc_post(f, X)
//     h = mhc_pre(X, hc_ffn);   h = rms_norm(h) * ffn_norm
//     f = dense SwiGLU(h)   or   sum_i w_i expert_i(h) + shared(h)        (both with the SwiGLU clamp)
//     X = mhc_post(f, X)
//
// The MTP block (after the main layers) is the same two sublayers around a plain residual instead of the mHC
// (glm_mtp_layer_decode); glm_model.hpp runs the whole model around these.
//
// What it is not yet: one token (decode), one layer a call, the weights already on the device in their GGUF form
// (GlmLinear) with the small exceptions listed on GlmLayerWeights.  No loader, no prompt path, no capture: the
// per-token values are in the step (GlmContext::set_pos), so the kernels' argument lists are constant across tokens
// except mla_kv_store's row, but nothing here captures yet.
#pragma once

#include "strata/core/geometry_glm5_next.hpp"
#include "strata/kernels/glm_linear.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>
#include <string>

namespace strata::glm {

using kernels::GlmLinear;

/// One block's weights, device pointers.  Projections are GlmLinear (any type glm_linear takes); the rest F32 as
/// the converter writes them, except attn_k_b / attn_v_b, which mla_head_gemv takes in BF16 (a loader converts).
struct GlmLayerWeights {
    // mHC, both halves: fn [24][4 n_embd], scale (3,), base (24,)
    const float *hc_attn_fn = nullptr, *hc_attn_scale = nullptr, *hc_attn_base = nullptr;
    const float *hc_ffn_fn = nullptr, *hc_ffn_scale = nullptr, *hc_ffn_base = nullptr;
    const float *attn_norm = nullptr, *ffn_norm = nullptr;
    GlmLinear wo;   ///< attn_output, of either attention

    // KDA
    GlmLinear wq, wk, wv, f_a, f_b, beta, g_a, g_b;
    const float *conv_q = nullptr, *conv_k = nullptr, *conv_v = nullptr;   ///< [d_inner][d_conv]
    const float *ssm_a = nullptr;      ///< (n_head,): -exp(A_log)
    const float *dt_bias = nullptr;    ///< (d_inner,)
    const float *o_norm = nullptr;     ///< (kda_head_dim,)

    // MLA
    GlmLinear wq_a, wq_b, wkv_a;
    const float *q_a_norm = nullptr, *kv_a_norm = nullptr;
    const uint16_t *k_b = nullptr, *v_b = nullptr;   ///< BF16 [H][kv_lora][DK], [H][DV][kv_lora]

    // the indexer (a full-indexer layer)
    GlmLinear idx_q_b, idx_k, idx_gate, idx_proj;
    const float *idx_k_norm = nullptr, *idx_k_norm_b = nullptr, *idx_ape = nullptr;

    // FFN: dense
    GlmLinear ffn_gate, ffn_up, ffn_down;
    // FFN: MoE
    GlmLinear gate_inp;
    const float* exp_probs_b = nullptr;                 ///< (n_expert,)
    kernels::NativeExpertLayout expert_layout;          ///< [gate rows | up rows | down rows] a blob
    const unsigned long long* expert_ptrs = nullptr;    ///< device: the n_expert blobs' device addresses
    GlmLinear sh_gate, sh_up, sh_down;
};

/// One block's recurrent and cache state, device memory.
struct GlmLayerState {
    float* conv_hist = nullptr;   ///< KDA: [3 d_inner][d_conv - 1]
    float* kda_state = nullptr;   ///< KDA: [D][H][D], kda.hpp's layout
    uint16_t* latent = nullptr;   ///< MLA: [max_ctx][kv_lora] FP16
    float* idx_cache = nullptr;   ///< MLA + indexer: [max_ctx][2 x 128] (key | gate)
    float* pooled = nullptr;      ///< MLA + indexer: [max_ctx / 4][128]
};

/// Where a MoE layer's experts are when they are not all resident (GlmLayerWeights::expert_ptrs null): asked after
/// the router, before the experts run.
class GlmExpertProvider {
public:
    virtual ~GlmExpertProvider() = default;
    /// The device array of layer il's n_expert blob addresses, in which the k experts the router chose (`ids`,
    /// device, written on `stream`) are resident until the next call; the others may hold anything.  May synchronize
    /// the stream.  Null on failure, with `err` set.
    virtual const unsigned long long* experts(int64_t il, const int32_t* ids, int k, void* stream,
                                              std::string& err) = 0;
};

/// The per-token values and the scratch every layer shares: one per model (and stream).
class GlmContext {
public:
    GlmContext() = default;
    ~GlmContext();
    GlmContext(const GlmContext&) = delete;
    GlmContext& operator=(const GlmContext&) = delete;
    /// Allocates for the geometry and a context of max_ctx cells.
    bool init(const core::Glm5NextGeometry& g, int64_t max_ctx, std::string& err);
    /// The step for the token at `pos` (kpool_step's, which MLA and the indexer read), uploaded on `stream`.
    void set_pos(int64_t pos, void* stream);
    int64_t max_ctx() const { return max_ctx_; }

    /// the experts of a layer whose expert_ptrs are null (a cache), else unused
    GlmExpertProvider* expert_provider = nullptr;
    // the step, and the selection a full-indexer layer leaves for the shared ones after it
    int32_t* step = nullptr;
    int32_t* sel_ids = nullptr;
    bool sel_valid = false;
    // scratch
    float *h = nullptr, *f = nullptr, *post = nullptr, *comb = nullptr, *mhc_scr = nullptr;
    float *qkv = nullptr, *qkv_c = nullptr, *raw = nullptr, *gate = nullptr, *beta = nullptr, *z = nullptr, *y = nullptr,
          *lo = nullptr;
    float *qr = nullptr, *q = nullptr, *kv = nullptr, *q_abs = nullptr, *o_lat = nullptr, *o_mla = nullptr,
          *attn_scr = nullptr;
    float *iq = nullptr, *ik = nullptr, *ig = nullptr, *iw = nullptr, *scores = nullptr;
    float *logits = nullptr, *ew = nullptr, *parts = nullptr, *shared = nullptr, *fg = nullptr, *fu = nullptr,
          *fh = nullptr;
    int32_t* eids = nullptr;
    unsigned long long* grp_ptr = nullptr;
    int32_t *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr;
    void *q8 = nullptr, *xq = nullptr, *exp_scr = nullptr;

private:
    int64_t max_ctx_ = 0, cap_ = 0, max_blocks_ = 0;
    int64_t pos_ = -1;   ///< set_pos's, for mla_kv_store's row
    int64_t top_k_ = 0;
    bool select_tail_ = true;
    void* arena_ = nullptr;
    friend struct GlmLayerImpl;
};

/// Whether block `il`'s weights are all there, in the shapes the geometry gives and types the kernels take: what a
/// loader checks before it allocates anything.  The MTP block (il >= n_layers) has no mHC weights and a full indexer.
/// `provided`: the experts come from a GlmExpertProvider, so expert_ptrs may be null.
bool glm_layer_check(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w, std::string& err,
                     bool provided = false);

/// Bytes of state one block needs (KDA: conv history and state; MLA: latent, indexer cache, pooled keys).
size_t glm_layer_state_bytes(const core::Glm5NextGeometry& g, int64_t il, int64_t max_ctx);
/// Carves `base` (device, glm_layer_state_bytes of it) into `st`.
void glm_layer_state_carve(const core::Glm5NextGeometry& g, int64_t il, int64_t max_ctx, void* base, GlmLayerState& st);

/// The token at ctx's position through block `il`: X [4][n_embd] (the four streams) updated in place, the block's
/// state too.  A shared-indexer MLA layer takes the selection the last full-indexer layer left in ctx.
bool glm_layer_decode(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w, const GlmLayerState& st,
                      GlmContext& ctx, float* x, void* stream, std::string& err);

/// The MTP block (il >= n_layers) at ctx's position, llama.cpp's graph_mtp between eh_proj and the head norm: a plain
/// residual, no mHC - x += MLA(rms_norm(x) attn_norm) over its own (full) indexer, then x += FFN(rms_norm(x) ffn_norm).
/// x [n_embd] in place.
bool glm_mtp_layer_decode(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w,
                          const GlmLayerState& st, GlmContext& ctx, float* x, void* stream, std::string& err);

}  // namespace strata::glm
