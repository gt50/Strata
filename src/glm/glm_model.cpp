// src/glm/glm_model.cpp - see include/strata/glm/glm_model.hpp.
#include "strata/glm/glm_model.hpp"

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/kda.hpp"
#include "strata/kernels/kpool.hpp"
#include "strata/kernels/mhc.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace strata::glm {
namespace k = strata::kernels;
namespace {

size_t align256(size_t n) { return (n + 255) & ~(size_t) 255; }

bool embedding_ok(const GlmEmbedding& e, int64_t n_embd, const char* name, std::string& err) {
    if (!e.data || !k::embed_type_supported(e.type) || n_embd % 256 != 0 ||
        e.row_bytes != k::iq_row_bytes(e.type, n_embd) || e.row_bytes == 0) {
        err = std::string("GlmModel: ") + name + ": missing, or ggml type " + std::to_string(e.type) +
              " / a row of " + std::to_string(e.row_bytes) + " bytes the embedding kernel does not read";
        return false;
    }
    return true;
}

bool head_ok(const GlmLinear& m, int64_t n_in, int64_t n_out, const char* name, std::string& err) {
    if (!m.data || m.n_in != n_in || m.n_out != n_out || k::glm_linear_kind(m.type, m.n_in) == k::kLinearNone) {
        err = std::string("GlmModel: ") + name + ": missing, not " + std::to_string(n_out) + " x " +
              std::to_string(n_in) + ", or ggml type " + std::to_string(m.type) + " has no GEMV here";
        return false;
    }
    return true;
}

bool launched(const char* who, std::string& err) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string(who) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

}  // namespace

void GlmModel::drop_graph() {
    if (graph_exec_) cudaGraphExecDestroy((cudaGraphExec_t) graph_exec_);
    if (graph_) cudaGraphDestroy((cudaGraph_t) graph_);
    graph_exec_ = graph_ = nullptr;
    graph_warm_ = false;
}

void GlmModel::set_decode_graph(bool on) {
    graph_on_ = on;
    if (!on) drop_graph();
}

GlmModel::~GlmModel() {
    drop_graph();
    if (pinned_) cudaFreeHost(pinned_);
    if (staged_) cudaEventDestroy((cudaEvent_t) staged_);
    if (state_arena_) cudaFree(state_arena_);
    if (buf_arena_) cudaFree(buf_arena_);
    if (gx_) cudaFree(gx_);
    if (win_arena_) cudaFree(win_arena_);
    if (pick_host_) cudaFreeHost(pick_host_);
    if (hist_host_) cudaFreeHost(hist_host_);
}

namespace {
/// A KDA block's window storage, floats a token: raw projections and conv output (3 d_inner each), gate (d_inner),
/// beta (H).
int64_t kda_window_floats(const core::Glm5NextGeometry& g) {
    const int64_t di = g.n_head * g.kda_head_dim;
    return 7 * di + g.n_head;
}
}  // namespace

bool GlmModel::init_verify(int max_t, std::string& err) {
    if (!buf_arena_ || win_arena_) {
        err = "GlmModel::init_verify: the model is not initialized, or windows already are";
        return false;
    }
    if (max_t < 1 || max_t + 1 > max_chunk_) {
        err = "GlmModel::init_verify: windows of " + std::to_string(max_t) + " tokens need a max_chunk of at least " +
              std::to_string(max_t + 1) + " (it is " + std::to_string(max_chunk_) + ")";
        return false;
    }
    const size_t T = (size_t) max_t, V = (size_t) g_.n_vocab, D = (size_t) g_.n_embd;
    const size_t per_kda = (size_t) kda_window_floats(g_) * T * sizeof(float);
    int64_t n_kda = 0;
    for (int64_t il = 0; il < g_.n_layers; ++il) n_kda += g_.is_mla[(size_t) il] ? 0 : 1;
    const size_t sizes[] = {T * V * sizeof(float), T * D * sizeof(float), T * sizeof(int32_t), per_kda * (size_t) n_kda,
                            T * kPickHistory * sizeof(int32_t)};
    size_t total = 0;
    for (size_t sz : sizes) total += align256(sz);
    if (cudaMalloc(&win_arena_, total) != cudaSuccess ||
        cudaHostAlloc((void**) &pick_host_, T * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &hist_host_, T * kPickHistory * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        err = "GlmModel::init_verify: " + std::to_string(total) + " bytes for the windows";
        return false;
    }
    cudaMemset(win_arena_, 0, total);
    uint8_t* p = (uint8_t*) win_arena_;
    auto take = [&](size_t i) { uint8_t* q = p; p += align256(sizes[i]); return q; };
    wlogits_ = (float*) take(0);
    whall_ = (float*) take(1);
    wpick_ = (int32_t*) take(2);
    float* kda = (float*) take(3);
    hist_dev_ = (int32_t*) take(4);
    win_kda_.assign((size_t) g_.n_layers, nullptr);
    for (int64_t il = 0; il < g_.n_layers; ++il)
        if (!g_.is_mla[(size_t) il]) {
            win_kda_[(size_t) il] = kda;
            kda += (size_t) kda_window_floats(g_) * T;
        }
    max_t_ = max_t;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        err = std::string("GlmModel::init_verify: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool GlmModel::argmax(const float* logits, int rows, int32_t* out, void* stream, std::string& err) {
    if (!wpick_ || rows < 1 || rows > max_t_ || !logits || !out) {
        err = "GlmModel::argmax: no verify buffers (init_verify), or " + std::to_string(rows) + " rows";
        return false;
    }
    const cudaStream_t s = (cudaStream_t) stream;
    k::glm_argmax_rows(logits, rows, (int) g_.n_vocab, wpick_, stream);
    cudaMemcpyAsync(pick_host_, wpick_, (size_t) rows * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) return launched("GlmModel::argmax", err);
    std::copy(pick_host_, pick_host_ + rows, out);
    return true;
}

bool GlmModel::pick(const float* logits, int rows, int64_t pos, const int32_t* tail, int64_t n_tail,
                    const int32_t* window, int32_t* out, void* stream, std::string& err) {
    if (!wpick_ || rows < 1 || rows > max_t_ || !logits || !window || !out || pos < 0 || n_tail < 0 ||
        (n_tail > 0 && !tail)) {
        err = "GlmModel::pick: no verify buffers (init_verify), " + std::to_string(rows) + " rows, or no tokens";
        return false;
    }
    const bool greedy = sp_.greedy || sp_.temperature <= 0.0f;
    const bool pen = sp_.penalty_last_n > 0 &&
                     (sp_.penalty_repeat != 1.0f || sp_.penalty_freq != 0.0f || sp_.penalty_present != 0.0f);
    if (greedy && !pen && !sp_.logit_bias) return argmax(logits, rows, out, stream, err);
    if (pen && sp_.penalty_last_n > kPickHistory) {
        err = "GlmModel::pick: a penalty window of " + std::to_string(sp_.penalty_last_n) + " tokens (at most " +
              std::to_string(kPickHistory) + ")";
        return false;
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const int h = pen ? sp_.penalty_last_n : 0;
    if (pen) {   // row r's history: tail, then window[0..r] (pinned: the copy runs before the sync below)
        k::penalty_rows(tail, n_tail, window, rows, h, hist_host_);
        cudaMemcpyAsync(hist_dev_, hist_host_, (size_t) rows * h * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    }
    k::SamplerParams p = sp_;
    p.counter = (uint64_t) pos;   // row r: Philox(seed, pos + r)
    k::sample_tokens(logits, rows, (int) g_.n_vocab, pen ? hist_dev_ : nullptr, h, p, wpick_, stream);
    cudaMemcpyAsync(pick_host_, wpick_, (size_t) rows * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) return launched("GlmModel::pick", err);
    std::copy(pick_host_, pick_host_ + rows, out);
    return true;
}

bool GlmModel::verify(const int32_t* tokens, int T, int64_t pos0, int32_t* picks, void* stream, std::string& err,
                      const int32_t* tail, int64_t n_tail) {
    if (!win_arena_ || !tokens || !picks || T < 1 || T > max_t_ || pos0 < 0 || pos0 + T > max_ctx_) {
        err = "GlmModel::verify: no window buffers (init_verify), or " + std::to_string(T) + " tokens at " +
              std::to_string(pos0) + " outside the window or the context";
        return false;
    }
    for (int t = 0; t < T; ++t)
        if (tokens[t] < 0 || tokens[t] >= g_.n_vocab) {
            err = "GlmModel::verify: token " + std::to_string(tokens[t]) + " outside the vocabulary";
            return false;
        }
    const int D = (int) g_.n_embd;
    const cudaStream_t s = (cudaStream_t) stream;
    win_pos0_ = -1;
    // pageable source: staged before the call returns; set_positions synchronizes the stream after its own copy
    cudaMemcpyAsync(tok_, tokens, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    ctx_.set_positions(pos0, T, stream);
    ctx_.sel_valid = false;
    k::iq_embed_rows(w_.tok_embd.type, w_.tok_embd.data, w_.tok_embd.row_bytes, tok_, T, D, e_, stream);
    k::mhc_init(e_, x_, T, D, stream);
    const int64_t di = g_.n_head * g_.kda_head_dim;
    ctx_.exact = true;
    bool ok = true;
    for (int64_t il = 0; il < g_.n_layers && ok; ++il) {
        float* wk = win_kda_[(size_t) il];
        if (wk) {   // raw [T][3 di] | conv output [T][3 di] | gate [T][di] | beta [T][H]
            ctx_.win_raw = wk;
            ctx_.win_qkv = wk + (size_t) T * 3 * di;
            ctx_.win_gate = wk + (size_t) T * 6 * di;
            ctx_.win_beta = wk + (size_t) T * 7 * di;
        }
        ok = glm_layer_decode(g_, il, w_.blocks[(size_t) il], states_[(size_t) il], ctx_, x_, stream, err);
        ctx_.win_raw = ctx_.win_qkv = ctx_.win_gate = ctx_.win_beta = nullptr;
    }
    ctx_.exact = false;
    if (!ok) return false;
    // every row's head, as head() computes one
    k::mhc_mean(x_, whall_, T, D, stream);
    k::glm_rms_norm(whall_, w_.output_norm, T, D, (float) g_.rms_eps, stream);
    k::glm_linear(w_.output, whall_, wlogits_, T, ctx_.q8, stream);
    if (!pick(wlogits_, T, pos0, tail, n_tail, tokens, picks, stream, err)) return false;
    win_tokens_.assign(tokens, tokens + T);
    win_pos0_ = pos0;
    return true;
}

bool GlmModel::commit(int n_keep, int32_t next, void* stream, std::string& err) {
    const int T = (int) win_tokens_.size();
    if (win_pos0_ < 0 || n_keep < 1 || n_keep > T || next < 0 || next >= g_.n_vocab) {
        err = "GlmModel::commit: no window to commit, " + std::to_string(n_keep) + " of its " + std::to_string(T) +
              " tokens, or a next token outside the vocabulary";
        return false;
    }
    const int D = (int) g_.n_embd, H = (int) g_.n_head;
    const int64_t di = g_.n_head * g_.kda_head_dim;
    const float eps = (float) g_.rms_eps;
    const cudaStream_t s = (cudaStream_t) stream;
    const int64_t pos0 = win_pos0_;
    win_pos0_ = -1;
    // the KDA blocks over the kept tokens: what decode leaves (the output, into scratch, unused)
    for (int64_t il = 0; il < g_.n_layers; ++il) {
        float* wk = win_kda_[(size_t) il];
        if (!wk) continue;
        const GlmLayerState& st = states_[(size_t) il];
        k::kda_conv_advance(st.conv_hist, wk, H, n_keep, stream);
        float* gate = wk + (size_t) T * 6 * di;
        k::kda_scan_norm(st.kda_state, wk + (size_t) T * 3 * di, gate, wk + (size_t) T * 7 * di, gate,
                         w_.blocks[(size_t) il].o_norm, eps, ctx_.y, H, n_keep, stream);
    }
    cudaMemcpyAsync(h_, whall_ + (size_t) (n_keep - 1) * D, (size_t) D * sizeof(float), cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(logits_, window_logits(n_keep - 1), (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToDevice,
                    s);
    if (has_mtp_) {   // the MTP block over the kept positions (and the prompt's pending one), the main model's h each
        const int first = pending_pos_ >= 0 && pending_pos_ == pos0 - 1 ? 1 : 0;
        pending_pos_ = -1;
        std::vector<int32_t> nt;
        if (first) nt.push_back(win_tokens_[0]);
        for (int t = 1; t < n_keep; ++t) nt.push_back(win_tokens_[(size_t) t]);
        nt.push_back(next);
        const int R = first + n_keep;
        // pageable: staged before the call returns
        cudaMemcpyAsync(mtp_tok_, nt.data(), nt.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s);
        if (first) cudaMemcpyAsync(hall_, h_pending_, (size_t) D * sizeof(float), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(hall_ + (size_t) first * D, whall_, (size_t) n_keep * D * sizeof(float), cudaMemcpyDeviceToDevice,
                        s);
        if (!mtp_pass(pos0 - first, R, hall_, stream, err)) return false;
        cudaMemcpyAsync(mtp_h_, xm_ + (size_t) (R - 1) * D, (size_t) D * sizeof(float), cudaMemcpyDeviceToDevice, s);
        k::glm_rms_norm(mtp_h_, w_.mtp_head_norm ? w_.mtp_head_norm : w_.output_norm, 1, D, eps, stream);
        k::glm_linear(w_.mtp_head.data ? w_.mtp_head : w_.output, mtp_h_, mtp_logits_, 1, ctx_.q8, stream);
    }
    win_tokens_.clear();
    return launched("GlmModel::commit", err);
}

bool GlmModel::init(const core::Glm5NextGeometry& g, const GlmModelWeights& w, int64_t max_ctx, std::string& err,
                    GlmExpertProvider* experts, int64_t max_chunk) {
    if (buf_arena_) { err = "GlmModel::init: already initialized"; return false; }
    const int64_t D = g.n_embd;
    if ((int64_t) w.blocks.size() != g.n_blocks() || (int64_t) g.is_mla.size() != g.n_blocks()) {
        err = "GlmModel::init: " + std::to_string(w.blocks.size()) + " blocks of weights for a geometry of " +
              std::to_string(g.n_blocks());
        return false;
    }
    if (!embedding_ok(w.tok_embd, D, "token_embd", err) || !head_ok(w.output, D, g.n_vocab, "output", err))
        return false;
    if (!w.output_norm) { err = "GlmModel: output_norm is missing"; return false; }
    for (int64_t il = 0; il < g.n_layers; ++il)
        if (!glm_layer_check(g, il, w.blocks[(size_t) il], err, experts != nullptr)) return false;
    has_mtp_ = g.n_nextn > 0 && w.eh_proj.data != nullptr;
    if (has_mtp_) {
        if (!glm_layer_check(g, g.n_layers, w.blocks[(size_t) g.n_layers], err, experts != nullptr) ||
            !head_ok(w.eh_proj, 2 * D, D, "nextn.eh_proj", err) ||
            (w.mtp_embd.data && !embedding_ok(w.mtp_embd, D, "nextn.embed_tokens", err)) ||
            (w.mtp_head.data && !head_ok(w.mtp_head, D, g.n_vocab, "nextn.shared_head_head", err)))
            return false;
        if (!w.enorm || !w.hnorm) { err = "GlmModel: nextn.enorm / nextn.hnorm missing"; return false; }
    }
    if (max_chunk < 1) { err = "GlmModel::init: max_chunk must be at least 1"; return false; }
    if (!ctx_.init(g, max_ctx, err, max_chunk)) return false;
    ctx_.expert_provider = experts;
    g_ = g;
    w_ = w;
    max_ctx_ = max_ctx;
    max_chunk_ = max_chunk;

    // the blocks' state, one allocation
    const int64_t n_state = g.n_layers + (has_mtp_ ? 1 : 0);
    std::vector<size_t> off((size_t) n_state + 1, 0);
    for (int64_t il = 0; il < n_state; ++il)
        off[(size_t) il + 1] = off[(size_t) il] + align256(glm_layer_state_bytes(g, il, max_ctx));
    if (cudaMalloc(&state_arena_, off.back()) != cudaSuccess) {
        cudaGetLastError();
        state_arena_ = nullptr;
        err = "GlmModel::init: " + std::to_string(off.back()) + " bytes of state";
        return false;
    }
    states_.assign((size_t) g.n_blocks(), GlmLayerState{});
    for (int64_t il = 0; il < n_state; ++il)
        glm_layer_state_carve(g, il, max_ctx, (uint8_t*) state_arena_ + off[(size_t) il], states_[(size_t) il]);

    // the model's own buffers
    const size_t V = (size_t) g.n_vocab;
    const size_t C = (size_t) max_chunk, C1 = C + 1;   // an MTP pass has a row more: the last call's last position
    const size_t sizes[] = {C * sizeof(int32_t),   C1 * 2 * D * sizeof(float), C * 4 * D * sizeof(float),
                            D * sizeof(float),     V * sizeof(float),          C1 * D * sizeof(float),
                            D * sizeof(float),     V * sizeof(float),          C1 * D * sizeof(float),
                            C1 * D * sizeof(float), D * sizeof(float),         C1 * sizeof(int32_t)};
    size_t total = 0;
    for (size_t s : sizes) total += align256(s);
    if (cudaMalloc(&buf_arena_, total) != cudaSuccess) {
        cudaGetLastError();
        buf_arena_ = nullptr;
        err = "GlmModel::init: " + std::to_string(total) + " bytes of buffers";
        return false;
    }
    uint8_t* p = (uint8_t*) buf_arena_;
    auto take = [&](size_t i) { uint8_t* q = p; p += align256(sizes[i]); return q; };
    tok_ = (int32_t*) take(0);
    e_ = (float*) take(1);
    x_ = (float*) take(2);
    h_ = (float*) take(3);
    logits_ = (float*) take(4);
    xm_ = (float*) take(5);
    mtp_h_ = (float*) take(6);
    mtp_logits_ = (float*) take(7);
    hall_ = (float*) take(8);
    eb_ = (float*) take(9);
    h_pending_ = (float*) take(10);
    mtp_tok_ = (int32_t*) take(11);
    cudaMemset(buf_arena_, 0, total);
    reset(nullptr);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        err = std::string("GlmModel::init: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

uint64_t GlmModel::prompt_group_bytes(const core::Glm5NextGeometry& g, int64_t tokens) {
    if (tokens <= 0) return 0;   // (a group no larger than the chunk allocates nothing either: an upper bound)
    const uint64_t per = (uint64_t) (4 * g.n_embd) * sizeof(float) +
                         (uint64_t) (k::kStepCount + g.idx_top_k + g.idx_kpool - 1) * sizeof(int32_t);
    return (uint64_t) tokens * per + 512;
}

bool GlmModel::init_prompt_group(int64_t tokens, std::string& err) {
    if (!buf_arena_ || gx_) { err = "GlmModel::init_prompt_group: before init, or twice"; return false; }
    if (tokens <= max_chunk_) return true;   // chunk by chunk
    tokens = std::min((tokens + max_chunk_ - 1) / max_chunk_ * max_chunk_, max_ctx_);   // a call has max_ctx at most
    if (cudaMalloc((void**) &gx_, (size_t) (tokens * 4 * g_.n_embd) * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        gx_ = nullptr;
        err = "GlmModel::init_prompt_group: the four streams of " + std::to_string(tokens) + " tokens";
        return false;
    }
    if (!ctx_.init_group(tokens, err)) return false;
    group_tokens_ = tokens;
    return true;
}

void GlmModel::reset(void* stream) {
    pending_pos_ = -1;
    const cudaStream_t s = (cudaStream_t) stream;
    const int64_t di = g_.n_head * g_.kda_head_dim;
    for (int64_t il = 0; il < g_.n_layers; ++il) {
        if (g_.is_mla[(size_t) il]) continue;
        const GlmLayerState& st = states_[(size_t) il];
        cudaMemsetAsync(st.conv_hist, 0, (size_t) (3 * di * (g_.ssm_d_conv - 1)) * sizeof(float), s);
        cudaMemsetAsync(st.kda_state, 0, (size_t) (g_.kda_head_dim * di) * sizeof(float), s);
    }
}

bool GlmModel::start(int32_t token, int64_t pos, const char* who, void* stream, std::string& err) {
    if (!buf_arena_) { err = std::string(who) + ": not initialized"; return false; }
    if (pos < 0 || pos >= max_ctx_ || token < 0 || token >= g_.n_vocab) {
        err = std::string(who) + ": token " + std::to_string(token) + " at " + std::to_string(pos) +
              ", outside the vocabulary or the context of " + std::to_string(max_ctx_);
        return false;
    }
    // pageable source: staged before the call returns, and set_pos synchronizes the stream after its own copy
    cudaMemcpyAsync(tok_, &token, sizeof token, cudaMemcpyHostToDevice, (cudaStream_t) stream);
    ctx_.set_pos(pos, stream);
    ctx_.sel_valid = false;
    return true;
}

void GlmModel::tap_main(int64_t n, void* stream) {
    if (!taps_) return;
    const cudaStream_t s = (cudaStream_t) stream;
    const size_t cap = (size_t) (g_.idx_top_k + g_.idx_kpool - 1);
    if (taps_->sel) cudaMemcpyAsync(taps_->sel, ctx_.sel_ids, (size_t) n * cap * 4, cudaMemcpyDeviceToDevice, s);
    if (taps_->eids && g_.n_expert_used > 0)
        cudaMemcpyAsync(taps_->eids, ctx_.eids, (size_t) (n * g_.n_expert_used) * 4, cudaMemcpyDeviceToDevice, s);
}

bool GlmModel::decode(int32_t token, int64_t pos, void* stream, std::string& err) {
    pending_pos_ = -1;   // the caller drafts on from here (mtp_decode fills the MTP block's rows)
    const bool graph = graph_on_ && graph_warm_ && !ctx_.expert_provider && stream;
    if (!graph) {
        if (!start(token, pos, "GlmModel::decode", stream, err)) return false;
        if (graph_on_) graph_warm_ = true;   // the next decode may capture
        return decode_body(stream, err);
    }
    if (pos < 0 || pos >= max_ctx_ || token < 0 || token >= g_.n_vocab) {
        err = "GlmModel::decode: token " + std::to_string(token) + " at " + std::to_string(pos) +
              ", outside the vocabulary or the context of " + std::to_string(max_ctx_);
        return false;
    }
    const cudaStream_t s = (cudaStream_t) stream;
    if (!pinned_) {
        if (cudaHostAlloc((void**) &pinned_, 64 * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess ||
            cudaEventCreateWithFlags((cudaEvent_t*) &staged_, cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            err = "GlmModel::decode: the graph's pinned staging";
            return false;
        }
    } else {
        cudaEventSynchronize((cudaEvent_t) staged_);   // the last token's copies have read the staging
    }
    // the token and its step: copied on the stream ahead of the graph, no host wait
    pinned_[0] = token;
    cudaMemcpyAsync(tok_, pinned_, sizeof(int32_t), cudaMemcpyHostToDevice, s);
    ctx_.stage_pos(pos, pinned_ + 1, stream);
    cudaEventRecord((cudaEvent_t) staged_, s);
    ctx_.sel_valid = false;
    if (graph_exec_ && graph_stream_ != stream) drop_graph();
    if (!graph_exec_) {
        cudaGraph_t gr = nullptr;
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            cudaGetLastError();
            err = "GlmModel::decode: the capture would not start";
            return false;
        }
        const bool ok = decode_body(stream, err);
        const cudaError_t e = cudaStreamEndCapture(s, &gr);
        if (!ok || e != cudaSuccess) {
            if (gr) cudaGraphDestroy(gr);
            cudaGetLastError();
            if (ok) err = std::string("GlmModel::decode: the capture: ") + cudaGetErrorString(e);
            return false;
        }
        cudaGraphExec_t ex = nullptr;
        if (cudaGraphInstantiate(&ex, gr, 0) != cudaSuccess) {
            cudaGraphDestroy(gr);
            cudaGetLastError();
            err = "GlmModel::decode: the graph would not instantiate";
            return false;
        }
        graph_ = gr;
        graph_exec_ = ex;
        graph_stream_ = stream;
    }
    if (cudaGraphLaunch((cudaGraphExec_t) graph_exec_, s) != cudaSuccess) return launched("GlmModel::decode", err);
    return true;
}

bool GlmModel::decode_body(void* stream, std::string& err) {
    const int D = (int) g_.n_embd;
    const size_t xb = (size_t) 4 * D * sizeof(float);
    const cudaStream_t s = (cudaStream_t) stream;
    k::iq_embed_rows(w_.tok_embd.type, w_.tok_embd.data, w_.tok_embd.row_bytes, tok_, 1, D, e_, stream);
    k::mhc_init(e_, x_, 1, D, stream);
    float* tx = taps_ ? taps_->x : nullptr;
    if (tx) cudaMemcpyAsync(tx, x_, xb, cudaMemcpyDeviceToDevice, s);
    static const int64_t ahead = [] {   // how many blocks on the expert prefetch guesses
        const char* v = std::getenv("STRATA_GLM_PREFETCH_AHEAD");
        return v && std::atoi(v) > 0 ? (int64_t) std::atoi(v) : (int64_t) 1;
    }();
    for (int64_t il = 0; il < g_.n_layers; ++il) {
        ctx_.next_block_il = il + ahead;
        ctx_.next_block = il + ahead < g_.n_layers ? &w_.blocks[(size_t) (il + ahead)] : nullptr;
        const bool ok = glm_layer_decode(g_, il, w_.blocks[(size_t) il], states_[(size_t) il], ctx_, x_, stream, err);
        ctx_.next_block = nullptr;
        ctx_.next_block_il = -1;
        if (!ok) return false;
        if (tx) cudaMemcpyAsync((uint8_t*) tx + (size_t) (il + 1) * xb, x_, xb, cudaMemcpyDeviceToDevice, s);
    }
    tap_main(1, stream);
    head(0, stream);
    return launched("GlmModel::decode", err);
}

void GlmModel::head(int64_t row, void* stream) {
    const int D = (int) g_.n_embd;
    k::mhc_mean(x_ + (size_t) row * 4 * D, h_, 1, D, stream);
    k::glm_rms_norm(h_, w_.output_norm, 1, D, (float) g_.rms_eps, stream);
    k::glm_linear(w_.output, h_, logits_, 1, ctx_.q8, stream);
}

bool GlmModel::prompt(const int32_t* tokens, int64_t n, int64_t pos0, void* stream, std::string& err) {
    if (!buf_arena_) { err = "GlmModel::prompt: not initialized"; return false; }
    if (!tokens || n < 1 || pos0 < 0 || pos0 + n > max_ctx_) {
        err = "GlmModel::prompt: " + std::to_string(n) + " tokens at " + std::to_string(pos0) +
              " do not fit the context of " + std::to_string(max_ctx_);
        return false;
    }
    for (int64_t i = 0; i < n; ++i)
        if (tokens[i] < 0 || tokens[i] >= g_.n_vocab) {
            err = "GlmModel::prompt: token " + std::to_string(tokens[i]) + " outside the vocabulary";
            return false;
        }
    const int D = (int) g_.n_embd;
    const cudaStream_t s = (cudaStream_t) stream;
    const bool mtp = has_mtp_ && mtp_prompt_;
    const bool pend = mtp && pending_pos_ >= 0 && pending_pos_ == pos0 - 1;   // the last call's last position
    pending_pos_ = -1;
    float* tx = taps_ ? taps_->x : nullptr;
    // the MTP block over the positions of the call's chunk c0 .. c0 + nc - 1 whose next token the call has (from the
    // last call's last), xc the chunk's four streams after the main layers
    auto mtp_chunk = [&](int64_t c0, int64_t nc, const float* xc) {
        const int64_t first = c0 == 0 && pend ? 1 : 0;
        const int64_t m = c0 + nc < n ? nc : nc - 1;
        if (first + m == 0) return true;
        std::vector<int32_t> next;
        if (first) next.push_back(tokens[0]);
        for (int64_t i = 0; i < m; ++i) next.push_back(tokens[c0 + 1 + i]);
        // pageable: staged before the call returns
        cudaMemcpyAsync(mtp_tok_, next.data(), next.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s);
        if (first) cudaMemcpyAsync(hall_, h_pending_, (size_t) D * sizeof(float), cudaMemcpyDeviceToDevice, s);
        if (m > 0) {   // h_nextn of the chunk's tokens: as head() computes it, a row each
            k::mhc_mean(xc, hall_ + first * D, (int) m, D, stream);
            k::glm_rms_norm(hall_ + first * D, w_.output_norm, m, D, (float) g_.rms_eps, stream);
        }
        return mtp_pass(pos0 + c0 - first, first + m, hall_, stream, err);
    };
    const int64_t group = std::max(group_tokens_, max_chunk_);
    int64_t ng = 0, head_row = 0;
    for (int64_t g0 = 0; g0 < n; g0 += ng) {
        ng = std::min(group, n - g0);
        if (ng <= max_chunk_) {   // one chunk
            const int64_t c0 = g0, nc = ng;
            const size_t xb = (size_t) nc * 4 * D * sizeof(float);
            // pageable source: staged before the call returns; set_positions synchronizes the stream after its copy
            cudaMemcpyAsync(tok_, tokens + c0, (size_t) nc * sizeof(int32_t), cudaMemcpyHostToDevice, s);
            ctx_.set_positions(pos0 + c0, nc, stream);
            ctx_.sel_valid = false;
            k::iq_embed_rows(w_.tok_embd.type, w_.tok_embd.data, w_.tok_embd.row_bytes, tok_, nc, D, e_, stream);
            k::mhc_init(e_, x_, (int) nc, D, stream);
            if (tx) cudaMemcpyAsync(tx, x_, xb, cudaMemcpyDeviceToDevice, s);
            for (int64_t il = 0; il < g_.n_layers; ++il) {
                if (!glm_layer_decode(g_, il, w_.blocks[(size_t) il], states_[(size_t) il], ctx_, x_, stream, err))
                    return false;
                if (tx) cudaMemcpyAsync((uint8_t*) tx + (size_t) (il + 1) * xb, x_, xb, cudaMemcpyDeviceToDevice, s);
            }
            tap_main(nc, stream);
            if (mtp && !mtp_chunk(c0, nc, x_)) return false;
            head_row = nc - 1;
            continue;
        }
        // layer by layer: every chunk of the group through a block before the next block, the chunks' four streams
        // in gx_ and their selections in the context's group rows
        ctx_.set_group(pos0 + g0, ng, stream);
        ctx_.sel_valid = false;
        for (int64_t c = 0; c < ng; c += max_chunk_) {
            const int64_t nc = std::min(max_chunk_, ng - c);
            cudaMemcpyAsync(tok_, tokens + g0 + c, (size_t) nc * sizeof(int32_t), cudaMemcpyHostToDevice, s);
            k::iq_embed_rows(w_.tok_embd.type, w_.tok_embd.data, w_.tok_embd.row_bytes, tok_, nc, D, e_, stream);
            k::mhc_init(e_, gx_ + (size_t) c * 4 * D, (int) nc, D, stream);
        }
        const int64_t c_last = (ng - 1) / max_chunk_ * max_chunk_, n_last = ng - c_last;   // the taps': as chunk by
        const size_t xb = (size_t) n_last * 4 * D * sizeof(float);                       // chunk, the last chunk's
        if (tx) cudaMemcpyAsync(tx, gx_ + (size_t) c_last * 4 * D, xb, cudaMemcpyDeviceToDevice, s);
        bool ok = true;
        for (int64_t il = 0; il < g_.n_layers && ok; ++il) {
            for (int64_t c = 0; c < ng && ok; c += max_chunk_) {
                ctx_.use_chunk(c, std::min(max_chunk_, ng - c));
                ok = glm_layer_decode(g_, il, w_.blocks[(size_t) il], states_[(size_t) il], ctx_,
                                      gx_ + (size_t) c * 4 * D, stream, err);
            }
            if (ok && tx)
                cudaMemcpyAsync((uint8_t*) tx + (size_t) (il + 1) * xb, gx_ + (size_t) c_last * 4 * D, xb,
                                cudaMemcpyDeviceToDevice, s);
        }
        if (ok) tap_main(n_last, stream);   // the context's selections are the last chunk's rows still
        ctx_.end_group();
        if (!ok) return false;
        for (int64_t c = 0; mtp && c < ng; c += max_chunk_)
            if (!mtp_chunk(g0 + c, std::min(max_chunk_, ng - c), gx_ + (size_t) c * 4 * D)) return false;
        cudaMemcpyAsync(x_, gx_ + (size_t) (ng - 1) * 4 * D, (size_t) 4 * D * sizeof(float), cudaMemcpyDeviceToDevice,
                        s);
        head_row = 0;
    }
    head(head_row, stream);
    if (mtp) {
        cudaMemcpyAsync(h_pending_, h_, (size_t) D * sizeof(float), cudaMemcpyDeviceToDevice, s);
        pending_pos_ = pos0 + n - 1;
    }
    return launched("GlmModel::prompt", err);
}

bool GlmModel::mtp_pass(int64_t pos0, int64_t rows, const float* h_rows, void* stream, std::string& err) {
    const int D = (int) g_.n_embd, R = (int) rows;
    const float eps = (float) g_.rms_eps;
    const cudaStream_t s = (cudaStream_t) stream;
    const size_t rb = (size_t) R * D * sizeof(float);
    const GlmEmbedding& emb = w_.mtp_embd.data ? w_.mtp_embd : w_.tok_embd;
    float* tm = taps_ ? taps_->mtp : nullptr;
    // [rms_norm(e) enorm ; rms_norm(h) hnorm] a row, llama.cpp's ggml_concat(e_norm, h_norm, 0)
    if (h_rows != hall_) cudaMemcpyAsync(hall_, h_rows, rb, cudaMemcpyDeviceToDevice, s);
    if (tm) cudaMemcpyAsync(tm, hall_, rb, cudaMemcpyDeviceToDevice, s);
    k::iq_embed_rows(emb.type, emb.data, emb.row_bytes, mtp_tok_, R, D, eb_, stream);
    k::glm_rms_norm(eb_, w_.enorm, R, D, eps, stream);
    k::glm_rms_norm(hall_, w_.hnorm, R, D, eps, stream);
    const size_t row = (size_t) D * sizeof(float);
    cudaMemcpy2DAsync(e_, 2 * row, eb_, row, row, (size_t) R, cudaMemcpyDeviceToDevice, s);
    cudaMemcpy2DAsync(e_ + D, 2 * row, hall_, row, row, (size_t) R, cudaMemcpyDeviceToDevice, s);
    ctx_.set_positions(pos0, R, stream);
    ctx_.sel_valid = false;
    ctx_.linear(w_.eh_proj, e_, xm_, R, stream);
    if (tm) cudaMemcpyAsync(tm + (size_t) R * D, xm_, rb, cudaMemcpyDeviceToDevice, s);
    const int64_t il = g_.n_layers;
    if (!glm_mtp_layer_decode(g_, il, w_.blocks[(size_t) il], states_[(size_t) il], ctx_, xm_, stream, err))
        return false;
    if (tm) {
        cudaMemcpyAsync(tm + (size_t) 2 * R * D, xm_, rb, cudaMemcpyDeviceToDevice, s);
        taps_->mtp_pos0 = pos0;
        taps_->mtp_rows = rows;
    }
    return true;
}

bool GlmModel::mtp_decode(int32_t token, int64_t pos, const float* h, void* stream, std::string& err) {
    if (!has_mtp_ || !h) {
        err = "GlmModel::mtp_decode: no MTP block loaded, or no h";
        return false;
    }
    if (!buf_arena_ || pos < 0 || pos >= max_ctx_ || token < 0 || token >= g_.n_vocab) {
        err = "GlmModel::mtp_decode: token " + std::to_string(token) + " at " + std::to_string(pos) +
              ", outside the vocabulary or the context, or not initialized";
        return false;
    }
    if (pos == pending_pos_) pending_pos_ = -1;   // the prompt's last row, filled here
    const int D = (int) g_.n_embd;
    const cudaStream_t s = (cudaStream_t) stream;
    cudaMemcpyAsync(mtp_tok_, &token, sizeof token, cudaMemcpyHostToDevice, s);   // staged before it returns
    if (!mtp_pass(pos, 1, h, stream, err)) return false;
    cudaMemcpyAsync(mtp_h_, xm_, (size_t) D * sizeof(float), cudaMemcpyDeviceToDevice, s);
    k::glm_rms_norm(mtp_h_, w_.mtp_head_norm ? w_.mtp_head_norm : w_.output_norm, 1, D, (float) g_.rms_eps,
                         stream);
    k::glm_linear(w_.mtp_head.data ? w_.mtp_head : w_.output, mtp_h_, mtp_logits_, 1, ctx_.q8, stream);
    return launched("GlmModel::mtp_decode", err);
}

}  // namespace strata::glm
