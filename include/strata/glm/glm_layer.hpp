// include/strata/glm/glm_layer.hpp - one GLM-5.3-Flash layer at decode: the kernels of kda.hpp, mla.hpp, kpool.hpp,
// glm_moe.hpp and mhc.hpp in the order llama.cpp's glm5-next graph runs them (upstream src/models/glm5-next.cpp):
//
//     h = mhc_pre(X, hc_attn);  h = rms_norm(h) * attn_norm           (one call: mhc_pre takes the norm)
//     f = KDA(h)   or   MLA(h) over the k-pool indexer's selection
//     X = mhc_post(f, X)
//     h = mhc_pre(X, hc_ffn);   h = rms_norm(h) * ffn_norm
//     f = dense SwiGLU(h)   or   sum_i w_i expert_i(h) + shared(h)        (both with the SwiGLU clamp)
//     X = mhc_post(f, X)
//
// The MTP block (after the main layers) is the same two sublayers around a plain residual instead of the mHC
// (glm_mtp_layer_decode); glm_model.hpp runs the whole model around these.
//
// The weights are on the device in their GGUF form (GlmLinear) with the small exceptions listed on GlmLayerWeights
// (glm_loader.hpp puts them there).  A decode step's per-token values are in the device step (GlmContext::set_pos /
// stage_pos), so its kernels' argument lists are the same at every token and it can be captured
// (GlmModel::set_decode_graph).
#pragma once

#include "strata/core/geometry_glm5_next.hpp"
#include "strata/kernels/glm_linear.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>
#include <string>
#include <vector>

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
    /// when the blobs lie a fixed stride apart (blob e at expert_base + e * expert_stride): a prompt's experts then
    /// run as matrix products over them (glm_gemm.hpp); else 0
    const uint8_t* expert_base = nullptr;
    uint64_t expert_stride = 0;
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
    /// A prompt's experts, a round at a time: the same table with `experts` (host ids, distinct, at most
    /// capacity()) resident until the next call.  Synchronizes the stream first (the round before may still read).
    virtual const unsigned long long* resident(int64_t il, const int32_t* experts, int n, void* stream,
                                               std::string& err) = 0;
    /// The experts one call can hold.
    virtual int64_t capacity() const = 0;
    /// The slots' layout when they lie a fixed stride apart (slot s at slot_base() + s * slot_stride(), device
    /// addresses), so a prompt round runs as matrix products over them; null / 0 otherwise.
    virtual const uint8_t* slot_base() const { return nullptr; }
    virtual uint64_t slot_stride() const { return 0; }
    /// After resident(): the slot expert e of block il is in.
    virtual int32_t slot_of(int64_t il, int32_t e) const { (void) il; (void) e; return -1; }
};

class GlmPromptGemm;

/// The per-token values and the scratch every layer shares: one per model (and stream).
class GlmContext {
public:
    GlmContext() = default;
    ~GlmContext();
    GlmContext(const GlmContext&) = delete;
    GlmContext& operator=(const GlmContext&) = delete;
    /// Allocates for the geometry, a context of max_ctx cells, and up to max_tokens tokens a call (a prompt's chunk).
    bool init(const core::Glm5NextGeometry& g, int64_t max_ctx, std::string& err, int64_t max_tokens = 1);
    /// The steps for the n tokens at pos0 .. pos0 + n - 1 (kpool_step's, which MLA and the indexer read), uploaded on
    /// `stream`: the tokens the next layer calls run.  n <= max_tokens; set_pos is the one-token case.
    void set_positions(int64_t pos0, int64_t n, void* stream);
    void set_pos(int64_t pos, void* stream) { set_positions(pos, 1, stream); }
    /// The one-token step for `pos` written to `pinned` (page-locked host memory, kStepCount values) and copied from
    /// there on `stream` without waiting: the caller keeps `pinned` until the copy has run (a captured decode).
    void stage_pos(int64_t pos, int32_t* pinned, void* stream);
    int64_t max_ctx() const { return max_ctx_; }
    int64_t max_tokens() const { return max_tokens_; }
    int64_t tokens() const { return n_; }
    /// y [n][n_out] = x [n][n_in] . W^T: the GEMV for one token, the prompt products (glm_gemm.hpp) for more.
    void linear(const GlmLinear& w, const float* x, float* y, int n, void* stream);

    /// the experts of a layer whose expert_ptrs are null (a cache), else unused
    GlmExpertProvider* expert_provider = nullptr;
    // the steps [tokens][kStepCount], and the selections [tokens][cap] a full-indexer layer leaves for the shared ones
    // after it
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
    int32_t* grp_expert = nullptr;
    void *q8 = nullptr, *xq = nullptr, *exp_scr = nullptr;
    float* kda_scr = nullptr;   ///< kda_scan_norm_chunked's, for more than one token

private:
    int64_t max_ctx_ = 0, cap_ = 0, max_blocks_ = 0, max_tokens_ = 1;
    int64_t pos_ = -1;   ///< set_positions' first position, for mla_kv_store's row
    int64_t n_ = 0;      ///< and its token count
    int32_t* eids_host_ = nullptr;   ///< pinned: a prompt's router choices, grouped on the host
    GlmPromptGemm* gemm_ = nullptr;  ///< the prompt products (max_tokens > 1)
    int32_t* mmq_bounds_ = nullptr;  ///< device: a round's per-slot bounds for the expert products
    int64_t mmq_bounds_cap_ = 0;
    std::vector<int32_t> h_expert_, h_start_, h_tok_, h_dst_;
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

/// The tokens of ctx's positions (set_positions: one at decode, a chunk of a prompt) through block `il`: X
/// [tokens][4][n_embd] (each token's four streams) updated in place, the block's state too.  A shared-indexer MLA layer
/// takes the selections the last full-indexer layer left in ctx.  One token runs the decode kernels (kda_scan_norm,
/// mla_decode_attn_step, the experts grouped on the device); more run the prompt ones (kda_scan_norm_chunked,
/// mla_prompt_attn, the experts grouped on the host, in rounds of what an expert provider holds).
bool glm_layer_decode(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w, const GlmLayerState& st,
                      GlmContext& ctx, float* x, void* stream, std::string& err);

/// The MTP block (il >= n_layers) at ctx's position, llama.cpp's graph_mtp between eh_proj and the head norm: a plain
/// residual, no mHC - x += MLA(rms_norm(x) attn_norm) over its own (full) indexer, then x += FFN(rms_norm(x) ffn_norm).
/// x [n_embd] in place.
bool glm_mtp_layer_decode(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w,
                          const GlmLayerState& st, GlmContext& ctx, float* x, void* stream, std::string& err);

}  // namespace strata::glm
