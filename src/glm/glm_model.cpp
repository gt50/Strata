// src/glm/glm_model.cpp - see include/strata/glm/glm_model.hpp.
#include "strata/glm/glm_model.hpp"

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/mhc.hpp"

#include <cuda_runtime.h>

#include <algorithm>

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

GlmModel::~GlmModel() {
    if (state_arena_) cudaFree(state_arena_);
    if (buf_arena_) cudaFree(buf_arena_);
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
    const size_t C = (size_t) max_chunk;
    const size_t sizes[] = {C * sizeof(int32_t), std::max<size_t>(2, C) * D * sizeof(float), C * 4 * D * sizeof(float),
                            D * sizeof(float),   V * sizeof(float), D * sizeof(float), D * sizeof(float),
                            V * sizeof(float)};
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
    cudaMemset(buf_arena_, 0, total);
    reset(nullptr);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        err = std::string("GlmModel::init: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

void GlmModel::reset(void* stream) {
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

bool GlmModel::decode(int32_t token, int64_t pos, void* stream, std::string& err) {
    if (!start(token, pos, "GlmModel::decode", stream, err)) return false;
    const int D = (int) g_.n_embd;
    const size_t xb = (size_t) 4 * D * sizeof(float);
    const cudaStream_t s = (cudaStream_t) stream;
    k::iq_embed_rows(w_.tok_embd.type, w_.tok_embd.data, w_.tok_embd.row_bytes, tok_, 1, D, e_, stream);
    k::mhc_init(e_, x_, 1, D, stream);
    if (taps_) cudaMemcpyAsync(taps_, x_, xb, cudaMemcpyDeviceToDevice, s);
    for (int64_t il = 0; il < g_.n_layers; ++il) {
        if (!glm_layer_decode(g_, il, w_.blocks[(size_t) il], states_[(size_t) il], ctx_, x_, stream, err))
            return false;
        if (taps_) cudaMemcpyAsync((uint8_t*) taps_ + (size_t) (il + 1) * xb, x_, xb, cudaMemcpyDeviceToDevice, s);
    }
    head(0, stream);
    return launched("GlmModel::decode", err);
}

void GlmModel::head(int64_t row, void* stream) {
    const int D = (int) g_.n_embd;
    k::mhc_mean(x_ + (size_t) row * 4 * D, h_, 1, D, stream);
    k::rms_norm_weighted(h_, w_.output_norm, 1, D, (float) g_.rms_eps, stream);
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
    int64_t nc = 0;
    for (int64_t c0 = 0; c0 < n; c0 += nc) {
        nc = std::min(max_chunk_, n - c0);
        const size_t xb = (size_t) nc * 4 * D * sizeof(float);
        // pageable source: staged before the call returns; set_positions synchronizes the stream after its own copy
        cudaMemcpyAsync(tok_, tokens + c0, (size_t) nc * sizeof(int32_t), cudaMemcpyHostToDevice, s);
        ctx_.set_positions(pos0 + c0, nc, stream);
        ctx_.sel_valid = false;
        k::iq_embed_rows(w_.tok_embd.type, w_.tok_embd.data, w_.tok_embd.row_bytes, tok_, nc, D, e_, stream);
        k::mhc_init(e_, x_, (int) nc, D, stream);
        if (taps_) cudaMemcpyAsync(taps_, x_, xb, cudaMemcpyDeviceToDevice, s);
        for (int64_t il = 0; il < g_.n_layers; ++il) {
            if (!glm_layer_decode(g_, il, w_.blocks[(size_t) il], states_[(size_t) il], ctx_, x_, stream, err))
                return false;
            if (taps_) cudaMemcpyAsync((uint8_t*) taps_ + (size_t) (il + 1) * xb, x_, xb, cudaMemcpyDeviceToDevice, s);
        }
    }
    head(nc - 1, stream);
    return launched("GlmModel::prompt", err);
}

bool GlmModel::mtp_decode(int32_t token, int64_t pos, const float* h, void* stream, std::string& err) {
    if (!has_mtp_ || !h) {
        err = "GlmModel::mtp_decode: no MTP block loaded, or no h";
        return false;
    }
    if (!start(token, pos, "GlmModel::mtp_decode", stream, err)) return false;
    const int D = (int) g_.n_embd;
    const float eps = (float) g_.rms_eps;
    const cudaStream_t s = (cudaStream_t) stream;
    const GlmEmbedding& emb = w_.mtp_embd.data ? w_.mtp_embd : w_.tok_embd;
    // [rms_norm(e) enorm ; rms_norm(h) hnorm], llama.cpp's ggml_concat(e_norm, h_norm, 0); h is copied first, so
    // it may be mtp_h() itself
    k::iq_embed_rows(emb.type, emb.data, emb.row_bytes, tok_, 1, D, e_, stream);
    cudaMemcpyAsync(e_ + D, h, (size_t) D * sizeof(float), cudaMemcpyDeviceToDevice, s);
    k::rms_norm_weighted(e_, w_.enorm, 1, D, eps, stream);
    k::rms_norm_weighted(e_ + D, w_.hnorm, 1, D, eps, stream);
    k::glm_linear(w_.eh_proj, e_, xm_, 1, ctx_.q8, stream);
    const int64_t il = g_.n_layers;
    if (!glm_mtp_layer_decode(g_, il, w_.blocks[(size_t) il], states_[(size_t) il], ctx_, xm_, stream, err))
        return false;
    cudaMemcpyAsync(mtp_h_, xm_, (size_t) D * sizeof(float), cudaMemcpyDeviceToDevice, s);
    k::rms_norm_weighted(mtp_h_, w_.mtp_head_norm ? w_.mtp_head_norm : w_.output_norm, 1, D, eps, stream);
    k::glm_linear(w_.mtp_head.data ? w_.mtp_head : w_.output, mtp_h_, mtp_logits_, 1, ctx_.q8, stream);
    return launched("GlmModel::mtp_decode", err);
}

}  // namespace strata::glm
