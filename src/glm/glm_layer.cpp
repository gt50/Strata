// src/glm/glm_layer.cpp - see include/strata/glm/glm_layer.hpp.
#include "strata/glm/glm_layer.hpp"

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/glm_moe.hpp"
#include "strata/kernels/kda.hpp"
#include "strata/kernels/kpool.hpp"
#include "strata/kernels/mhc.hpp"
#include "strata/kernels/mla.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace strata::glm {
namespace k = strata::kernels;
namespace {

size_t align256(size_t n) { return (n + 255) & ~(size_t) 255; }

/// Offsets into one allocation, 256-byte aligned: a first pass sums the sizes, a second hands out the pointers.
struct Carver {
    uint8_t* base = nullptr;
    size_t off = 0;
    template <class T> T* take(size_t count) {
        T* p = base ? reinterpret_cast<T*>(base + off) : nullptr;
        off += align256(count * sizeof(T));
        return p;
    }
};

bool full_indexer(const core::Glm5NextGeometry& g, int64_t il) {
    return il >= g.n_layers || g.idx_full.empty() || g.idx_full[(size_t) il] != 0;
}

}  // namespace

GlmContext::~GlmContext() {
    if (arena_) cudaFree(arena_);
}

bool GlmContext::init(const core::Glm5NextGeometry& g, int64_t max_ctx, std::string& err) {
    if (arena_) { err = "GlmContext::init: already initialized"; return false; }
    if (max_ctx <= 0 || g.idx_kpool != k::kKpool || g.idx_key_dim != k::kKpoolDim || g.idx_heads != k::kKpoolHeads ||
        g.kv_lora_rank != k::kMlaLatent || g.kda_head_dim != k::kKdaHeadDim || g.hc != k::kMhcStreams) {
        err = "GlmContext::init: a geometry the kernels are not compiled for, or no context";
        return false;
    }
    max_ctx_ = max_ctx;
    top_k_ = g.idx_top_k;
    select_tail_ = g.idx_kpool_select_tail;
    cap_ = g.idx_top_k + g.idx_kpool - 1;
    max_blocks_ = max_ctx / g.idx_kpool + 1;
    const int64_t D = g.n_embd, H = g.n_head, di = g.n_head * g.kda_head_dim, L = g.kv_lora_rank;
    const int64_t kU = g.n_expert_used, ff = std::max(g.n_ff_dense, g.n_ff_shexp);
    const int64_t max_in = std::max({D, di, g.kda_head_dim, g.q_lora_rank, H * g.head_dim_v, ff});
    const int64_t mix_scr = (int64_t) k::mhc_pre_scratch_floats(1, (int) D);
    const int64_t attn_floats = (int64_t) k::mla_decode_attn_scratch_floats((int) H, cap_);
    for (int pass = 0; pass < 2; ++pass) {
        Carver c;
        c.base = (uint8_t*) arena_;
        step = c.take<int32_t>(k::kStepCount);
        sel_ids = c.take<int32_t>((size_t) cap_);
        h = c.take<float>(D); f = c.take<float>(D); post = c.take<float>(4); comb = c.take<float>(16);
        mhc_scr = c.take<float>(mix_scr);
        qkv = c.take<float>(3 * di); qkv_c = c.take<float>(3 * di); raw = c.take<float>(di); gate = c.take<float>(di);
        beta = c.take<float>(H); z = c.take<float>(di); y = c.take<float>(di); lo = c.take<float>(g.kda_head_dim);
        qr = c.take<float>(g.q_lora_rank); q = c.take<float>(H * g.head_dim_k); kv = c.take<float>(L);
        q_abs = c.take<float>(H * L); o_lat = c.take<float>(H * L); o_mla = c.take<float>(H * g.head_dim_v);
        attn_scr = c.take<float>(attn_floats);
        iq = c.take<float>(g.idx_heads * g.idx_key_dim); ik = c.take<float>(g.idx_key_dim);
        ig = c.take<float>(g.idx_key_dim); iw = c.take<float>(g.idx_heads); scores = c.take<float>(max_blocks_);
        logits = c.take<float>(std::max<int64_t>(g.n_expert, 1)); ew = c.take<float>(std::max<int64_t>(kU, 1));
        parts = c.take<float>(std::max<int64_t>(kU, 1) * D); shared = c.take<float>(D);
        fg = c.take<float>(std::max<int64_t>(ff, 1)); fu = c.take<float>(std::max<int64_t>(ff, 1));
        fh = c.take<float>(std::max<int64_t>(ff, 1));
        eids = c.take<int32_t>(std::max<int64_t>(kU, 1));
        grp_ptr = c.take<unsigned long long>(std::max<int64_t>(kU, 1));
        grp_start = c.take<int32_t>(kU + 1); n_groups = c.take<int32_t>(1);
        ent_dst = c.take<int32_t>(std::max<int64_t>(kU, 1)); ent_tok = c.take<int32_t>(std::max<int64_t>(kU, 1));
        q8 = c.take<uint8_t>(k::glm_linear_scratch_bytes(max_in, 1));
        xq = c.take<uint8_t>(k::glm_linear_scratch_bytes(D, 1));
        exp_scr = c.take<uint8_t>(g.n_expert_used > 0 ? k::native_expert_scratch_bytes(kU, g.n_ff_exp) : 0);
        if (pass == 0) {
            if (cudaMalloc(&arena_, c.off) != cudaSuccess) {
                cudaGetLastError();
                arena_ = nullptr;
                err = "GlmContext::init: " + std::to_string(c.off) + " bytes of scratch";
                return false;
            }
            cudaMemset(arena_, 0, c.off);
        }
    }
    return true;
}

void GlmContext::set_pos(int64_t pos, void* stream) {
    int32_t st[k::kStepCount];
    k::kpool_step(pos, top_k_, select_tail_, st);
    // from the stack: the copy is complete when the call returns (pageable memory), so `st` may go out of scope
    cudaMemcpyAsync(step, st, sizeof st, cudaMemcpyHostToDevice, (cudaStream_t) stream);
    cudaStreamSynchronize((cudaStream_t) stream);
    pos_ = pos;
}

size_t glm_layer_state_bytes(const core::Glm5NextGeometry& g, int64_t il, int64_t max_ctx) {
    Carver c;
    if (!g.is_mla[(size_t) il]) {
        const int64_t di = g.n_head * g.kda_head_dim;
        c.take<float>(3 * di * (g.ssm_d_conv - 1));
        c.take<float>(g.kda_head_dim * g.n_head * g.kda_head_dim);
    } else {
        c.take<uint16_t>(max_ctx * g.kv_lora_rank);
        if (full_indexer(g, il)) {
            c.take<float>(max_ctx * 2 * g.idx_key_dim);
            c.take<float>((max_ctx / g.idx_kpool + 1) * g.idx_key_dim);
        }
    }
    return c.off;
}

void glm_layer_state_carve(const core::Glm5NextGeometry& g, int64_t il, int64_t max_ctx, void* base, GlmLayerState& st) {
    Carver c;
    c.base = (uint8_t*) base;
    st = GlmLayerState{};
    if (!g.is_mla[(size_t) il]) {
        const int64_t di = g.n_head * g.kda_head_dim;
        st.conv_hist = c.take<float>(3 * di * (g.ssm_d_conv - 1));
        st.kda_state = c.take<float>(g.kda_head_dim * g.n_head * g.kda_head_dim);
    } else {
        st.latent = c.take<uint16_t>(max_ctx * g.kv_lora_rank);
        if (full_indexer(g, il)) {
            st.idx_cache = c.take<float>(max_ctx * 2 * g.idx_key_dim);
            st.pooled = c.take<float>((max_ctx / g.idx_kpool + 1) * g.idx_key_dim);
        }
    }
}

bool glm_layer_decode(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w, const GlmLayerState& st,
                      GlmContext& ctx, float* x, void* stream, std::string& err) {
    if (!ctx.arena_ || il < 0 || il >= g.n_blocks() || !x) {
        err = "glm_layer_decode: no context, or block " + std::to_string(il) + " out of range";
        return false;
    }
    if (ctx.pos_ < 0 || ctx.pos_ >= ctx.max_ctx_) {
        err = "glm_layer_decode: position " + std::to_string(ctx.pos_) + " outside the context";
        return false;
    }
    const int D = (int) g.n_embd, H = (int) g.n_head, Dh = (int) g.kda_head_dim, di = H * Dh;
    const int L = (int) g.kv_lora_rank, DK = (int) g.head_dim_k, DV = (int) g.head_dim_v;
    const float rms_eps = (float) g.rms_eps, hc_eps = (float) g.hc_eps;
    const int iters = (int) g.hc_sinkhorn_iters;
    void* q8 = ctx.q8;
    auto lin = [&](const GlmLinear& m, const float* in, float* out) { k::glm_linear(m, in, out, 1, q8, stream); };

    // ---- attention
    k::mhc_pre(x, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, 1, D, rms_eps, hc_eps, iters, ctx.mhc_scr, ctx.h,
               ctx.post, ctx.comb, nullptr, stream);
    k::rms_norm_weighted(ctx.h, w.attn_norm, 1, D, rms_eps, stream);
    if (!g.is_mla[(size_t) il]) {
        lin(w.wq, ctx.h, ctx.qkv);
        lin(w.wk, ctx.h, ctx.qkv + di);
        lin(w.wv, ctx.h, ctx.qkv + 2 * di);
        k::kda_conv_l2(st.conv_hist, ctx.qkv, w.conv_q, w.conv_k, w.conv_v, ctx.qkv_c, H, 1, 1e-6f, stream);
        lin(w.f_a, ctx.h, ctx.lo);
        lin(w.f_b, ctx.lo, ctx.raw);
        k::kda_gate(ctx.raw, w.dt_bias, w.ssm_a, ctx.gate, H, 1, g.has_kda_gate_lower_bound,
                    (float) g.kda_gate_lower_bound, stream);
        lin(w.beta, ctx.h, ctx.beta);
        k::gdn_beta_gate(ctx.beta, H, stream);
        lin(w.g_a, ctx.h, ctx.lo);
        lin(w.g_b, ctx.lo, ctx.z);
        k::kda_scan_norm(st.kda_state, ctx.qkv_c, ctx.gate, ctx.beta, ctx.z, w.o_norm, rms_eps, ctx.y, H, 1, stream);
        lin(w.wo, ctx.y, ctx.f);
    } else {
        lin(w.wq_a, ctx.h, ctx.qr);
        k::rms_norm_weighted(ctx.qr, w.q_a_norm, 1, g.q_lora_rank, rms_eps, stream);
        lin(w.wq_b, ctx.qr, ctx.q);
        lin(w.wkv_a, ctx.h, ctx.kv);
        // (the one argument that still comes from the host: mla_kv_store's row)
        k::mla_kv_store(ctx.kv, w.kv_a_norm, rms_eps, st.latent, ctx.pos_, 1, stream);
        k::mla_head_gemv(ctx.q, w.k_b, ctx.q_abs, H, DK, L, 1, stream);
        if (full_indexer(g, il)) {
            lin(w.idx_q_b, ctx.qr, ctx.iq);
            lin(w.idx_k, ctx.h, ctx.ik);
            lin(w.idx_gate, ctx.h, ctx.ig);
            k::kpool_key_store(ctx.ik, ctx.ig, w.idx_k_norm, w.idx_k_norm_b, (float) g.norm_eps, st.idx_cache, 0, 1,
                               ctx.step, stream);
            k::kpool_pool(st.idx_cache, w.idx_ape, st.pooled, 0, 0, ctx.step, stream);
            lin(w.idx_proj, ctx.h, ctx.iw);
            k::kpool_scores(st.pooled, ctx.iq, ctx.iw, 1.0f / std::sqrt((float) (g.idx_key_dim * g.idx_heads)),
                            ctx.step, 1, ctx.max_blocks_, ctx.select_tail_, ctx.scores, stream);
            k::qsa_block_topk(ctx.scores, ctx.step, 1, ctx.max_blocks_, ctx.cap_, k::kpool_topk_shapes(ctx.top_k_),
                              ctx.sel_ids, stream);
            ctx.sel_valid = true;
        } else if (!ctx.sel_valid) {
            err = "glm_layer_decode: block " + std::to_string(il) + " shares an indexer no earlier block computed";
            return false;
        }
        k::mla_decode_attn_step(ctx.q_abs, st.latent, ctx.sel_ids, ctx.step, ctx.cap_,
                                1.0f / std::sqrt((float) DK), ctx.attn_scr, ctx.o_lat, H, stream);
        k::mla_head_gemv(ctx.o_lat, w.v_b, ctx.o_mla, H, L, DV, 1, stream);
        lin(w.wo, ctx.o_mla, ctx.f);
    }
    k::mhc_post(ctx.f, ctx.post, ctx.comb, x, 1, D, stream);

    // ---- FFN
    k::mhc_pre(x, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, 1, D, rms_eps, hc_eps, iters, ctx.mhc_scr, ctx.h,
               ctx.post, ctx.comb, nullptr, stream);
    k::rms_norm_weighted(ctx.h, w.ffn_norm, 1, D, rms_eps, stream);
    if (il < g.n_dense_lead) {
        lin(w.ffn_gate, ctx.h, ctx.fg);
        lin(w.ffn_up, ctx.h, ctx.fu);
        k::swiglu_clamp(ctx.fg, ctx.fu, ctx.fh, g.n_ff_dense, (float) g.swiglu_clamp_shexp, stream);
        lin(w.ffn_down, ctx.fh, ctx.f);
    } else {
        const int kU = (int) g.n_expert_used;
        lin(w.gate_inp, ctx.h, ctx.logits);
        k::glm_router(ctx.logits, w.exp_probs_b, 1, (int) g.n_expert, kU, (int) g.expert_gating, g.expert_weights_norm,
                      (float) g.expert_weights_scale, ctx.eids, ctx.ew, stream);
        k::glm_expert_groups_decode(ctx.eids, w.expert_ptrs, kU, ctx.grp_ptr, ctx.grp_start, ctx.n_groups, ctx.ent_dst,
                                    ctx.ent_tok, stream);
        k::quantize_q8_1_rows(ctx.h, 1, D, ctx.xq, stream);
        k::native_expert_grouped_clamped(w.expert_layout, ctx.grp_ptr, ctx.grp_start, ctx.n_groups, ctx.ent_dst,
                                         ctx.ent_tok, kU, kU, ctx.xq, ctx.exp_scr, ctx.parts,
                                         (float) g.swiglu_clamp_exp, stream);
        lin(w.sh_gate, ctx.h, ctx.fg);
        lin(w.sh_up, ctx.h, ctx.fu);
        k::swiglu_clamp(ctx.fg, ctx.fu, ctx.fh, g.n_ff_shexp, (float) g.swiglu_clamp_shexp, stream);
        lin(w.sh_down, ctx.fh, ctx.shared);
        k::native_moe_combine(ctx.parts, ctx.ew, ctx.shared, ctx.f, D, kU, stream);
    }
    k::mhc_post(ctx.f, ctx.post, ctx.comb, x, 1, D, stream);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm_layer_decode: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

}  // namespace strata::glm
