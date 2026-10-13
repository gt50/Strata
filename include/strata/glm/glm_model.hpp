// include/strata/glm/glm_model.hpp - GLM-5.3-Flash's whole forward pass for one token at decode, around
// glm_layer_decode: llama.cpp's glm5-next graph and graph_mtp (upstream src/models/glm5-next.cpp).
//
//     e = tok_embd[token];  X = four copies of e                            (mhc_init)
//     X = block_il(X)  for il < n_layers                                    (glm_layer_decode)
//     h = rms_norm(mean_i X[i]) * output_norm                               (h_nextn: what the MTP block reads)
//     logits = output . h
//
// and the MTP block (the one after the main layers), which drafts the token after `token` from the h of the token
// before it - the main model's h_nextn, or the MTP block's own when drafting a chain:
//
//     x = eh_proj . [rms_norm(embed[token]) * enorm ; rms_norm(h) * hnorm]
//     x = mtp_block(x)                                                      (glm_mtp_layer_decode: no mHC)
//     h' = rms_norm(x) * head_norm;  logits = head . h'
//
// embed, head and head_norm are the MTP block's own (nextn.embed_tokens, shared_head_head, shared_head_norm) where
// the file has them, else the model's.
//
// Speculative decoding (glm_spec.hpp runs the loop): verify() runs a window - the last token kept and the drafts
// after it - through the main layers by the decode kernels, so each row's greedy pick is bitwise decode()'s, without
// committing the KDA blocks' history and state; commit() keeps the accepted prefix and runs the MTP block over it.
#pragma once

#include "strata/core/geometry_glm5_next.hpp"
#include "strata/glm/glm_layer.hpp"
#include "strata/kernels/sampler.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::glm {

/// A GGUF token-embedding table on the device: n_vocab rows of n_embd values, `row_bytes` a row, any type
/// iq_embed_rows reads (kernels::embed_type_supported).
struct GlmEmbedding {
    int type = -1;
    const void* data = nullptr;
    size_t row_bytes = 0;
};

struct GlmModelWeights {
    GlmEmbedding tok_embd;
    const float* output_norm = nullptr;
    GlmLinear output;                        ///< n_embd -> n_vocab
    std::vector<GlmLayerWeights> blocks;     ///< n_blocks(): the main layers, then the MTP block's

    // the MTP block's own: none of it is needed unless mtp_decode is called (eh_proj.data null = no MTP)
    GlmLinear eh_proj;                       ///< nextn.eh_proj: 2 n_embd -> n_embd
    const float *enorm = nullptr, *hnorm = nullptr;
    GlmEmbedding mtp_embd;                   ///< nextn.embed_tokens; data null = tok_embd
    GlmLinear mtp_head;                      ///< nextn.shared_head_head; data null = output
    const float* mtp_head_norm = nullptr;    ///< nextn.shared_head_norm; null = output_norm
};

/// A test's view of the last decode / prompt / mtp_decode call: the device buffers it fills when they are set.  A
/// prompt of several chunks leaves its last chunk's.
struct GlmTaps {
    float* x = nullptr;        ///< [n_layers + 1][tokens][4][n_embd]: the streams after mhc_init and after every main block
    int32_t* sel = nullptr;    ///< [tokens][top_k + kpool - 1]: the main layers' last selections (before the MTP block's)
    int32_t* eids = nullptr;   ///< [tokens][n_expert_used]: the main layers' last router choices, likewise
    float* mtp = nullptr;      ///< [3][rows][n_embd]: an MTP pass's h rows, eh_proj's output and the block's output
    int64_t mtp_pos0 = -1, mtp_rows = 0;   ///< (host) the last MTP pass's positions
};

/// The model's state (every block's caches and recurrent state) and buffers for a context of max_ctx cells.
class GlmModel {
public:
    GlmModel() = default;
    ~GlmModel();
    GlmModel(const GlmModel&) = delete;
    GlmModel& operator=(const GlmModel&) = delete;

    /// Checks every weight against the geometry (glm_layer_check) and allocates; the weights must outlive the model.
    /// `experts`: where the MoE layers' experts are when their expert_ptrs are null (a cache; it must outlive the
    /// model too).
    /// `max_chunk`: the most tokens prompt() runs a call (its scratch; 1 = decode only).
    bool init(const core::Glm5NextGeometry& g, const GlmModelWeights& w, int64_t max_ctx, std::string& err,
              GlmExpertProvider* experts = nullptr, int64_t max_chunk = 1);
    /// A new sequence: the KDA conv histories and recurrent states zeroed (the caches are written before read).
    void reset(void* stream);

    /// The token at `pos` through the model: logits() and h_nextn() hold the result when the stream gets there.
    bool decode(int32_t token, int64_t pos, void* stream, std::string& err);
    /// Decode as one captured graph: the token and its step are staged through pinned memory and copied without a
    /// host wait, and the step itself - the embedding, every main block, the head - replays as one launch.  The
    /// first decode after this runs as it always does (any first-use setup outside the capture); the second is
    /// captured; the rest replay.  The same results bit for bit.  Not with an expert provider (a cache answers on the
    /// host in the middle of the step: those decodes run as before); a stream of the caller's own (not the default
    /// one).  Changing the taps drops the graph.
    ///
    /// Measured on the Radeon 8060S (glm_graph_parity --bench 45: GLM-5.3-Flash's shapes, 45 layers, 8 resident
    /// IQ4_XS experts a layer): 51.1-53.4 ms a token either way (before the small-kernel fixes 59.4-59.9) - the step
    /// reads 9.75 GB of weights (~190 GB/s), and its 1623 kernels run on the GPU no faster replayed than launched one by
    /// one (glm_graph_parity --kernels: the step is the GPU's time, not the launches').  The graph is for the steps the
    /// host must not wait on (a verify window, a faster card), not for this one.
    void set_decode_graph(bool on);
    bool decode_graph_ready() const { return graph_exec_ != nullptr; }
    /// The captured step (a cudaGraph_t, null before the capture), for a profiler to read its nodes; the model owns it.
    void* decode_graph() const { return graph_; }
    /// The n tokens at pos0 .. pos0 + n - 1 (a prompt, or its next part) through the main layers, max_chunk tokens a
    /// call: every block's state and caches advance as n decode() calls would advance them, by the prompt kernels;
    /// logits() and h_nextn() hold the last token's.
    ///
    /// With an MTP block, each chunk is then run through it too (set_mtp_over_prompt), so that drafting can start
    /// where the prompt ends: position p with token p + 1 and the main model's h at p - every position whose next
    /// token the call has.  The last one waits for its next token: the next prompt() call's first (when it continues
    /// at the next position), or the caller's mtp_decode(next, last, h_nextn()), which drafts as it fills it.
    bool prompt(const int32_t* tokens, int64_t n, int64_t pos0, void* stream, std::string& err);
    /// Whether prompt() runs the MTP block over its tokens (on by default when the model has one).
    void set_mtp_over_prompt(bool on) { mtp_prompt_ = on; }
    bool has_mtp() const { return has_mtp_; }
    /// The MTP block for `token` at `pos`, reading h [n_embd] (device; h_nextn() of the token before, or mtp_h() to
    /// draft on): mtp_logits() and mtp_h() hold the result.  The block keeps its own caches by `pos`.
    bool mtp_decode(int32_t token, int64_t pos, const float* h, void* stream, std::string& err);

    /// Buffers for verify windows of up to max_t tokens (max_t + 1 <= the max_chunk init was given).
    bool init_verify(int max_t, std::string& err);
    int max_verify() const { return max_t_; }
    /// A verify window: tokens[0..T) at pos0 .. pos0 + T - 1 - the last token kept and T - 1 drafts - through the
    /// main layers by the decode kernels, every token's arithmetic a decode step's (glm_layer.hpp: GlmContext::exact),
    /// so picks[t] (host; pick() with the window as its tokens and tail[0..n_tail) the ones before it) is the token
    /// decoding a token at a time would pick after token t - bitwise its logits, and the same draw.  The positions'
    /// caches are written (a later token at a position rewrites them before anything reads them); the KDA conv
    /// histories and states are left as they were until commit().  Synchronizes the stream.
    bool verify(const int32_t* tokens, int T, int64_t pos0, int32_t* picks, void* stream, std::string& err,
                const int32_t* tail = nullptr, int64_t n_tail = 0);
    /// Row t of the last window's logits (device [n_vocab]).
    const float* window_logits(int t) const { return wlogits_ + (size_t) t * (size_t) g_.n_vocab; }
    /// Keeps the last window's first n_keep tokens (1..T): the KDA histories and states advance over them (bitwise
    /// as n_keep decode() calls would leave them, from the inputs the window kept), logits() and h_nextn() become
    /// token n_keep - 1's, and with an MTP block it runs over the kept positions with `next`, the token after them
    /// (and over the prompt's pending last position, when the window follows it) - mtp_logits() / mtp_h() then draft
    /// the token after `next`.
    bool commit(int n_keep, int32_t next, void* stream, std::string& err);
    /// The greedy picks of `rows` device logit rows of n_vocab (glm_argmax_rows: ties to the lowest index) into host
    /// `out`; synchronizes the stream.
    bool argmax(const float* logits, int rows, int32_t* out, void* stream, std::string& err);
    /// How pick() and verify() choose: greedy (the default: glm_argmax_rows) or sample_tokens' chain (penalties,
    /// top_k, top_p, min_p, temperature).  The token after position q draws Philox(seed, q) - its draw depends on its
    /// position alone, so a verify window's row picks what decoding a token at a time picks, and a draft is kept when
    /// it equals that.  p.counter is set per call; p.logit_bias (device) stays the caller's.  Penalty windows up to
    /// kPickHistory tokens.
    static constexpr int kPickHistory = 1024;
    void set_sampling(const kernels::SamplerParams& p) { sp_ = p; }
    const kernels::SamplerParams& sampling() const { return sp_; }
    /// The tokens after window[r] (at position pos + r) for the rows r < rows of `logits` (device, n_vocab a row),
    /// into host `out`: row r draws Philox(seed, pos + r), and the penalties count tail[0..n_tail) then window[0..r]
    /// (the tokens before the window, then its own).  Synchronizes the stream.
    bool pick(const float* logits, int rows, int64_t pos, const int32_t* tail, int64_t n_tail, const int32_t* window,
              int32_t* out, void* stream, std::string& err);

    const float* logits() const { return logits_; }          ///< device [n_vocab]
    const float* h_nextn() const { return h_; }               ///< device [n_embd]
    const float* mtp_logits() const { return mtp_logits_; }  ///< device [n_vocab]
    const float* mtp_h() const { return mtp_h_; }             ///< device [n_embd]

    int64_t max_ctx() const { return max_ctx_; }
    const GlmLayerState& state(int64_t il) const { return states_[(size_t) il]; }
    const GlmContext& context() const { return ctx_; }
    /// A test's view of the calls (GlmTaps); null stops it.
    void set_taps(GlmTaps* taps) {
        taps_ = taps;
        drop_graph();
    }

private:
    core::Glm5NextGeometry g_;
    GlmModelWeights w_;
    GlmContext ctx_;
    std::vector<GlmLayerState> states_;
    bool has_mtp_ = false;
    int64_t max_ctx_ = 0, max_chunk_ = 1;
    void* state_arena_ = nullptr;
    void* buf_arena_ = nullptr;
    int32_t* tok_ = nullptr;
    float *e_ = nullptr, *x_ = nullptr, *h_ = nullptr, *logits_ = nullptr;
    float *xm_ = nullptr, *mtp_h_ = nullptr, *mtp_logits_ = nullptr;
    GlmTaps* taps_ = nullptr;
    // the captured decode step
    bool graph_on_ = false, graph_warm_ = false;
    void* graph_ = nullptr;        ///< cudaGraph_t
    void* graph_exec_ = nullptr;   ///< cudaGraphExec_t
    void* graph_stream_ = nullptr;
    int32_t* pinned_ = nullptr;    ///< the token and the step, staged
    void* staged_ = nullptr;       ///< cudaEvent_t: the staging copies have run
    void drop_graph();
    bool decode_body(void* stream, std::string& err);   ///< the step once tok_ and the step are on the device
    bool mtp_prompt_ = true;
    // verify windows
    int max_t_ = 0;
    void* win_arena_ = nullptr;
    float *wlogits_ = nullptr, *whall_ = nullptr;
    int32_t *wpick_ = nullptr, *pick_host_ = nullptr;
    int32_t *hist_dev_ = nullptr, *hist_host_ = nullptr;   ///< pick()'s penalty rows [max_t][kPickHistory]
    kernels::SamplerParams sp_ = [] {
        kernels::SamplerParams p;
        p.greedy = true;
        return p;
    }();
    std::vector<float*> win_kda_;           ///< per block: its window storage (KDA blocks), else null
    std::vector<int32_t> win_tokens_;       ///< the last window's tokens, until commit
    int64_t win_pos0_ = -1;
    int64_t pending_pos_ = -1;   ///< a prompt's last position, its MTP row waiting for the next token
    float *hall_ = nullptr, *eb_ = nullptr, *h_pending_ = nullptr;
    int32_t* mtp_tok_ = nullptr;

    bool start(int32_t token, int64_t pos, const char* who, void* stream, std::string& err);
    void head(int64_t row, void* stream);   ///< logits() and h_nextn() from token `row` of x_
    /// The MTP block over `rows` positions from pos0: the next tokens in mtp_tok_, the h rows at h_rows (copied
    /// first: it may be mtp_h_); leaves the block's output in xm_.
    bool mtp_pass(int64_t pos0, int64_t rows, const float* h_rows, void* stream, std::string& err);
    void tap_main(int64_t n, void* stream);   ///< the main layers' selections and router choices into the taps
};

}  // namespace strata::glm
