// src/glm/glm_layer.cpp - see include/strata/glm/glm_layer.hpp.
#include "strata/glm/glm_layer.hpp"
#include "strata/glm/glm_gemm.hpp"

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

/// A sublayer's projections: one token through glm_linear, each input quantized once (a projection of the same input,
/// on the same path, as the quantizing call before it reuses that call's q8_1), more through ctx.linear (the prompt
/// products).  So an input must not change between two calls that read it: the sublayers write each buffer before
/// they project it, never after.
struct Linear {
    GlmContext& ctx;
    int n;
    void* stream;
    const float* last = nullptr;
    k::GlmLinearKind kind = k::kLinearNone;
    int64_t n_in = 0;
    void operator()(const GlmLinear& m, const float* in, float* out) {
        if (n != 1) {
            ctx.linear(m, in, out, n, stream);
            return;
        }
        const k::GlmLinearKind kd = k::glm_linear_kind(m.type, m.n_in);
        const bool q = k::glm_linear_quantizes(kd);
        k::glm_linear(m, in, out, 1, ctx.q8, stream, q && in == last && kd == kind && m.n_in == n_in);
        if (q) {
            last = in;
            kind = kd;
            n_in = m.n_in;
        }
    }
};

}  // namespace

GlmContext::~GlmContext() {
    if (arena_) cudaFree(arena_);
    if (eids_host_) cudaFreeHost(eids_host_);
    if (mmq_bounds_) cudaFree(mmq_bounds_);
    delete gemm_;
}

bool GlmContext::init(const core::Glm5NextGeometry& g, int64_t max_ctx, std::string& err, int64_t max_tokens) {
    if (arena_) { err = "GlmContext::init: already initialized"; return false; }
    if (max_ctx <= 0 || max_tokens <= 0 || g.idx_kpool != k::kKpool || g.idx_key_dim != k::kKpoolDim ||
        g.idx_heads != k::kKpoolHeads || g.kv_lora_rank != k::kMlaLatent || g.kda_head_dim != k::kKdaHeadDim ||
        g.hc != k::kMhcStreams) {
        err = "GlmContext::init: a geometry the kernels are not compiled for, or no context";
        return false;
    }
    max_ctx_ = max_ctx;
    max_tokens_ = max_tokens;
    top_k_ = g.idx_top_k;
    select_tail_ = g.idx_kpool_select_tail;
    cap_ = g.idx_top_k + g.idx_kpool - 1;
    max_blocks_ = max_ctx / g.idx_kpool + 1;
    const int64_t T = max_tokens;
    const int64_t D = g.n_embd, H = g.n_head, di = g.n_head * g.kda_head_dim, L = g.kv_lora_rank;
    const int64_t kU = std::max<int64_t>(g.n_expert_used, 1), ff = std::max<int64_t>(std::max(g.n_ff_dense, g.n_ff_shexp), 1);
    const int64_t max_in = std::max({2 * D, di, g.kda_head_dim, g.q_lora_rank, H * g.head_dim_v, ff});
    const int64_t mix_scr = (int64_t) k::mhc_pre_scratch_floats((int) T, (int) D);
    const int64_t attn_floats = (int64_t) k::mla_decode_attn_scratch_floats((int) H, cap_);
    const int64_t groups = std::min<int64_t>(std::max<int64_t>(g.n_expert, 1), T * kU);
    for (int pass = 0; pass < 2; ++pass) {
        Carver c;
        c.base = (uint8_t*) arena_;
        step = c.take<int32_t>(T * k::kStepCount);
        sel_ids = c.take<int32_t>(T * cap_);
        h = c.take<float>(T * D); f = c.take<float>(T * D); post = c.take<float>(T * 4); comb = c.take<float>(T * 16);
        mhc_scr = c.take<float>(mix_scr);
        qkv = c.take<float>(T * 3 * di); qkv_c = c.take<float>(T * 3 * di); raw = c.take<float>(T * di);
        gate = c.take<float>(T * di); beta = c.take<float>(T * H); z = c.take<float>(T * di); y = c.take<float>(T * di);
        lo = c.take<float>(T * g.kda_head_dim);
        qr = c.take<float>(T * g.q_lora_rank); q = c.take<float>(T * H * g.head_dim_k); kv = c.take<float>(T * L);
        q_abs = c.take<float>(T * H * L); o_lat = c.take<float>(T * H * L); o_mla = c.take<float>(T * H * g.head_dim_v);
        attn_scr = c.take<float>(attn_floats);
        iq = c.take<float>(T * g.idx_heads * g.idx_key_dim); ik = c.take<float>(T * g.idx_key_dim);
        ig = c.take<float>(T * g.idx_key_dim); iw = c.take<float>(T * g.idx_heads); scores = c.take<float>(T * max_blocks_);
        logits = c.take<float>(T * std::max<int64_t>(g.n_expert, 1)); ew = c.take<float>(T * kU);
        parts = c.take<float>(T * kU * D); shared = c.take<float>(T * D);
        fg = c.take<float>(T * ff); fu = c.take<float>(T * ff); fh = c.take<float>(T * ff);
        eids = c.take<int32_t>(T * kU);
        grp_ptr = c.take<unsigned long long>(groups);
        grp_start = c.take<int32_t>(groups + 1); n_groups = c.take<int32_t>(1);
        ent_dst = c.take<int32_t>(T * kU); ent_tok = c.take<int32_t>(T * kU);
        grp_expert = c.take<int32_t>(groups);
        q8 = c.take<uint8_t>(k::glm_linear_scratch_bytes(max_in, (int) T));
        xq = c.take<uint8_t>(k::glm_linear_scratch_bytes(D, (int) T));
        exp_scr = c.take<uint8_t>(g.n_expert_used > 0 ? k::native_expert_scratch_bytes(T * kU, g.n_ff_exp) : 0);
        kda_scr = c.take<float>(T > 1 ? k::kda_chunked_scratch_floats((int) H, (int) T) : 0);
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
    if (cudaHostAlloc((void**) &eids_host_, (size_t) (T * kU) * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        eids_host_ = nullptr;
        err = "GlmContext::init: the router's pinned host buffer";
        return false;
    }
    if (T > 1) {
        gemm_ = new GlmPromptGemm();
        if (!gemm_->init(T, max_in, D, std::max<int64_t>(g.n_ff_exp, 1), g.n_expert_used, err)) return false;
    }
    return true;
}

void GlmContext::stage_pos(int64_t pos, int32_t* pinned, void* stream) {
    k::kpool_step(pos, top_k_, select_tail_, pinned);
    cudaMemcpyAsync(step, pinned, k::kStepCount * sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) stream);
    pos_ = pos;
    n_ = 1;
}

void GlmContext::set_positions(int64_t pos0, int64_t n, void* stream) {
    std::vector<int32_t> st((size_t) (n * k::kStepCount));
    for (int64_t t = 0; t < n; ++t) k::kpool_step(pos0 + t, top_k_, select_tail_, &st[(size_t) (t * k::kStepCount)]);
    // pageable: the copy is staged when the call returns, so `st` may go out of scope
    cudaMemcpyAsync(step, st.data(), st.size() * sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) stream);
    cudaStreamSynchronize((cudaStream_t) stream);
    pos_ = pos0;
    n_ = n;
}

void GlmContext::linear(const GlmLinear& w, const float* x, float* y, int n, void* stream) {
    if (n == 1 || !gemm_) k::glm_linear(w, x, y, n, q8, stream);
    else gemm_->linear(w, x, y, n, q8, stream);
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

/// The two sublayers between the mHC (or the MTP block's residual) and the block: ctx.h (normed) in, ctx.f out, for
/// ctx's n tokens.
struct GlmLayerImpl {
    static bool ready(const core::Glm5NextGeometry& g, int64_t il, const GlmContext& ctx, const float* x,
                      const char* who, std::string& err);
    static bool attention(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w,
                          const GlmLayerState& st, GlmContext& ctx, void* stream, std::string& err);
    static bool ffn(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w, GlmContext& ctx,
                    void* stream, std::string& err);
    static bool experts_prompt(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w, GlmContext& ctx,
                               void* stream, std::string& err);
};

bool GlmLayerImpl::ready(const core::Glm5NextGeometry& g, int64_t il, const GlmContext& ctx, const float* x,
                         const char* who, std::string& err) {
    if (!ctx.arena_ || il < 0 || il >= g.n_blocks() || !x) {
        err = std::string(who) + ": no context, or block " + std::to_string(il) + " out of range";
        return false;
    }
    if (ctx.n_ < 1 || ctx.n_ > ctx.max_tokens_ || ctx.pos_ < 0 || ctx.pos_ + ctx.n_ > ctx.max_ctx_) {
        err = std::string(who) + ": positions " + std::to_string(ctx.pos_) + " + " + std::to_string(ctx.n_) +
              " outside the context, or more tokens than the context was made for";
        return false;
    }
    return true;
}

bool GlmLayerImpl::attention(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w,
                             const GlmLayerState& st, GlmContext& ctx, void* stream, std::string& err) {
    const int H = (int) g.n_head, L = (int) g.kv_lora_rank, DK = (int) g.head_dim_k, DV = (int) g.head_dim_v;
    const int di = H * (int) g.kda_head_dim;
    const int n = (int) ctx.n_;
    const float rms_eps = (float) g.rms_eps;
    Linear lin{ctx, n, stream};
    if (!g.is_mla[(size_t) il]) {
        if (n == 1) {
            lin(w.wq, ctx.h, ctx.qkv);
            lin(w.wk, ctx.h, ctx.qkv + di);
            lin(w.wv, ctx.h, ctx.qkv + 2 * di);
        } else {   // [n][di] each, then interleaved into [n][q | k | v] (y, z and gate are free until later)
            lin(w.wq, ctx.h, ctx.y);
            lin(w.wk, ctx.h, ctx.z);
            lin(w.wv, ctx.h, ctx.gate);
            const cudaStream_t s = (cudaStream_t) stream;
            const size_t row = (size_t) di * sizeof(float);
            cudaMemcpy2DAsync(ctx.qkv, 3 * row, ctx.y, row, row, (size_t) n, cudaMemcpyDeviceToDevice, s);
            cudaMemcpy2DAsync(ctx.qkv + di, 3 * row, ctx.z, row, row, (size_t) n, cudaMemcpyDeviceToDevice, s);
            cudaMemcpy2DAsync(ctx.qkv + 2 * di, 3 * row, ctx.gate, row, row, (size_t) n, cudaMemcpyDeviceToDevice, s);
        }
        k::kda_conv_l2(st.conv_hist, ctx.qkv, w.conv_q, w.conv_k, w.conv_v, ctx.qkv_c, H, n, 1e-6f, stream);
        lin(w.f_a, ctx.h, ctx.lo);
        lin(w.f_b, ctx.lo, ctx.raw);
        k::kda_gate(ctx.raw, w.dt_bias, w.ssm_a, ctx.gate, H, n, g.has_kda_gate_lower_bound,
                    (float) g.kda_gate_lower_bound, stream);
        lin(w.beta, ctx.h, ctx.beta);
        k::gdn_beta_gate(ctx.beta, (int64_t) n * H, stream);
        lin(w.g_a, ctx.h, ctx.lo);
        lin(w.g_b, ctx.lo, ctx.z);
        if (n == 1)
            k::kda_scan_norm(st.kda_state, ctx.qkv_c, ctx.gate, ctx.beta, ctx.z, w.o_norm, rms_eps, ctx.y, H, 1, stream);
        else
            k::kda_scan_norm_chunked(st.kda_state, ctx.qkv_c, ctx.gate, ctx.beta, ctx.z, w.o_norm, rms_eps, ctx.y, H,
                                     n, ctx.kda_scr, stream);
        lin(w.wo, ctx.y, ctx.f);
        return true;
    }
    const bool full = full_indexer(g, il);
    // h's projections, then qr's: at decode each input is quantized once
    lin(w.wq_a, ctx.h, ctx.qr);
    lin(w.wkv_a, ctx.h, ctx.kv);
    if (full) lin(w.idx_k, ctx.h, ctx.ik);
    k::glm_rms_norm(ctx.qr, w.q_a_norm, n, g.q_lora_rank, rms_eps, stream);
    lin(w.wq_b, ctx.qr, ctx.q);
    if (full) lin(w.idx_q_b, ctx.qr, ctx.iq);
    // one token: the row from the device step (a captured decode step writes the right one at every replay)
    k::mla_kv_store(ctx.kv, w.kv_a_norm, rms_eps, st.latent, ctx.pos_, n, stream, n == 1 ? ctx.step : nullptr);
    k::mla_head_gemv(ctx.q, w.k_b, ctx.q_abs, H, DK, L, n, stream);
    if (full) {
        lin(w.idx_gate, ctx.h, ctx.ig);
        if (n == 1) {   // the decode step's row and pool from the device step (capturable)
            k::kpool_key_store(ctx.ik, ctx.ig, w.idx_k_norm, w.idx_k_norm_b, (float) g.norm_eps, st.idx_cache, 0, 1,
                               ctx.step, stream);
            k::kpool_pool(st.idx_cache, w.idx_ape, st.pooled, 0, 0, ctx.step, stream);
        } else {   // the chunk's rows, then every pool completed in it (each query reads only the ones before it)
            k::kpool_key_store(ctx.ik, ctx.ig, w.idx_k_norm, w.idx_k_norm_b, (float) g.norm_eps, st.idx_cache,
                               ctx.pos_, n, nullptr, stream);
            const int64_t b0 = ctx.pos_ / g.idx_kpool, b1 = (ctx.pos_ + n) / g.idx_kpool;
            if (b1 > b0) k::kpool_pool(st.idx_cache, w.idx_ape, st.pooled, b0, b1 - b0, nullptr, stream);
        }
        lin(w.idx_proj, ctx.h, ctx.iw);
        k::kpool_scores(st.pooled, ctx.iq, ctx.iw, 1.0f / std::sqrt((float) (g.idx_key_dim * g.idx_heads)),
                        ctx.step, n, ctx.max_blocks_, ctx.select_tail_, ctx.scores, stream);
        k::qsa_block_topk(ctx.scores, ctx.step, n, ctx.max_blocks_, ctx.cap_, k::kpool_topk_shapes(ctx.top_k_),
                          ctx.sel_ids, stream);
        ctx.sel_valid = true;
    } else if (!ctx.sel_valid) {
        err = "glm_layer_decode: block " + std::to_string(il) + " shares an indexer no earlier block computed";
        return false;
    }
    const float scale = 1.0f / std::sqrt((float) DK);
    if (n == 1)
        k::mla_decode_attn_step(ctx.q_abs, st.latent, ctx.sel_ids, ctx.step, ctx.cap_, scale, ctx.attn_scr, ctx.o_lat,
                                H, stream);
    else
        k::mla_prompt_attn(ctx.q_abs, st.latent, ctx.sel_ids, ctx.step, ctx.cap_, n, scale, ctx.o_lat, H, stream);
    k::mla_head_gemv(ctx.o_lat, w.v_b, ctx.o_mla, H, L, DV, n, stream);
    lin(w.wo, ctx.o_mla, ctx.f);
    return true;
}

/// A prompt chunk's routed experts: the router's choices to the host, grouped by expert (entries t * k + j in order),
/// and run in rounds of what the provider holds (one round when they are all resident).
bool GlmLayerImpl::experts_prompt(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w,
                                  GlmContext& ctx, void* stream, std::string& err) {
    const int n = (int) ctx.n_, kU = (int) g.n_expert_used, NE = (int) g.n_expert;
    const int entries = n * kU;
    const cudaStream_t s = (cudaStream_t) stream;
    if (cudaMemcpyAsync(ctx.eids_host_, ctx.eids, (size_t) entries * sizeof(int32_t), cudaMemcpyDeviceToHost, s) !=
            cudaSuccess ||
        cudaStreamSynchronize(s) != cudaSuccess) {
        err = "block " + std::to_string(il) + ": the router's choices";
        return false;
    }
    std::vector<int32_t> count((size_t) NE, 0);
    for (int i = 0; i < entries; ++i) {
        const int32_t e = ctx.eids_host_[i];
        if (e < 0 || e >= NE) {
            err = "block " + std::to_string(il) + ": the router chose expert " + std::to_string(e);
            return false;
        }
        ++count[(size_t) e];
    }
    std::vector<int32_t> order;   // the experts chosen, ascending
    for (int e = 0; e < NE; ++e)
        if (count[(size_t) e]) order.push_back(e);
    std::vector<int32_t> first((size_t) NE + 1, 0);   // each expert's first entry slot, in expert order
    for (int e = 0; e < NE; ++e) first[(size_t) e + 1] = first[(size_t) e] + count[(size_t) e];
    std::vector<int32_t> by_expert((size_t) entries), fill(first.begin(), first.end() - 1);
    for (int i = 0; i < entries; ++i) by_expert[(size_t) fill[(size_t) ctx.eids_host_[i]]++] = i;
    const int64_t cap = w.expert_ptrs ? (int64_t) order.size() : ctx.expert_provider->capacity();
    if (cap < 1) {
        err = "block " + std::to_string(il) + ": the expert provider holds no expert";
        return false;
    }
    // matrix products over the blobs when they lie a fixed stride apart (the resident arena, or the cache's slots)
    const uint8_t* base = w.expert_ptrs ? w.expert_base : ctx.expert_provider->slot_base();
    const uint64_t stride = w.expert_ptrs ? w.expert_stride : ctx.expert_provider->slot_stride();
    const bool mmq = base && glm_prompt_experts_mmq(w.expert_layout, stride);
    for (size_t r0 = 0; r0 < order.size(); r0 += (size_t) cap) {
        const size_t r1 = std::min(order.size(), r0 + (size_t) cap);
        const int ng = (int) (r1 - r0);
        const unsigned long long* table = w.expert_ptrs;
        if (!table) {
            table = ctx.expert_provider->resident(il, order.data() + r0, ng, stream, err);
            if (!table) return false;
        }
        if (mmq) {   // the round's rows grouped by slot; bounds over the slots [s0, s1]
            std::vector<std::pair<int32_t, int32_t>> by_slot;   // (slot, expert)
            for (size_t i = r0; i < r1; ++i) {
                const int32_t e = order[i];
                by_slot.emplace_back(w.expert_ptrs ? e : ctx.expert_provider->slot_of(il, e), e);
            }
            std::sort(by_slot.begin(), by_slot.end());
            const int32_t s0 = by_slot.front().first, s1 = by_slot.back().first;
            if (s0 < 0) {
                err = "block " + std::to_string(il) + ": an expert the provider did not place";
                return false;
            }
            const int64_t n_slots = s1 - s0 + 1;
            if (n_slots + 1 > ctx.mmq_bounds_cap_) {
                if (ctx.mmq_bounds_) cudaFree(ctx.mmq_bounds_);   // (synchronizes: no product still reads it)
                ctx.mmq_bounds_cap_ = std::max<int64_t>(n_slots + 1, 1024);
                if (cudaMalloc((void**) &ctx.mmq_bounds_, (size_t) ctx.mmq_bounds_cap_ * 4) != cudaSuccess) {
                    cudaGetLastError();
                    ctx.mmq_bounds_ = nullptr;
                    ctx.mmq_bounds_cap_ = 0;
                    err = "block " + std::to_string(il) + ": the expert products' bounds";
                    return false;
                }
            }
            ctx.h_start_.assign((size_t) n_slots + 1, 0);
            ctx.h_tok_.clear();
            ctx.h_dst_.clear();
            int max_rows = 0;
            size_t q = 0;
            for (int64_t sl = s0; sl <= s1; ++sl) {
                int rows = 0;
                if (q < by_slot.size() && by_slot[q].first == sl) {
                    const int32_t e = by_slot[q++].second;
                    for (int32_t j = first[(size_t) e]; j < first[(size_t) e + 1]; ++j, ++rows) {
                        const int32_t entry = by_expert[(size_t) j];
                        ctx.h_tok_.push_back(entry / kU);
                        ctx.h_dst_.push_back(entry);
                    }
                }
                max_rows = std::max(max_rows, rows);
                ctx.h_start_[(size_t) (sl - s0 + 1)] = (int32_t) ctx.h_tok_.size();
            }
            const int rows = (int) ctx.h_tok_.size();
            cudaMemcpyAsync(ctx.mmq_bounds_, ctx.h_start_.data(), (size_t) (n_slots + 1) * 4, cudaMemcpyHostToDevice, s);
            cudaMemcpyAsync(ctx.ent_tok, ctx.h_tok_.data(), (size_t) rows * 4, cudaMemcpyHostToDevice, s);
            cudaMemcpyAsync(ctx.ent_dst, ctx.h_dst_.data(), (size_t) rows * 4, cudaMemcpyHostToDevice, s);
            ctx.gemm_->experts(w.expert_layout, base + (uint64_t) s0 * stride, stride, (int) n_slots,
                               ctx.mmq_bounds_, ctx.ent_tok, ctx.ent_dst, rows, max_rows, ctx.h, ctx.parts,
                               (float) g.swiglu_clamp_exp, stream);
            continue;
        }
        ctx.h_expert_.assign(order.begin() + (ptrdiff_t) r0, order.begin() + (ptrdiff_t) r1);
        ctx.h_start_.assign(1, 0);
        ctx.h_tok_.clear();
        ctx.h_dst_.clear();
        for (int32_t e : ctx.h_expert_) {
            for (int32_t j = first[(size_t) e]; j < first[(size_t) e + 1]; ++j) {
                const int32_t entry = by_expert[(size_t) j];
                ctx.h_tok_.push_back(entry / kU);
                ctx.h_dst_.push_back(entry);
            }
            ctx.h_start_.push_back((int32_t) ctx.h_tok_.size());
        }
        // pageable copies: staged before each call returns, so the vectors may change for the next round
        cudaMemcpyAsync(ctx.grp_expert, ctx.h_expert_.data(), (size_t) ng * 4, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(ctx.grp_start, ctx.h_start_.data(), (size_t) (ng + 1) * 4, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(ctx.n_groups, &ng, 4, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(ctx.ent_tok, ctx.h_tok_.data(), ctx.h_tok_.size() * 4, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(ctx.ent_dst, ctx.h_dst_.data(), ctx.h_dst_.size() * 4, cudaMemcpyHostToDevice, s);
        k::glm_expert_group_ptrs(table, ctx.grp_expert, ng, ctx.grp_ptr, stream);
        k::native_expert_grouped_clamped(w.expert_layout, ctx.grp_ptr, ctx.grp_start, ctx.n_groups, ctx.ent_dst,
                                         ctx.ent_tok, ng, entries, ctx.xq, ctx.exp_scr, ctx.parts,
                                         (float) g.swiglu_clamp_exp, stream);
    }
    return true;
}

bool GlmLayerImpl::ffn(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w, GlmContext& ctx,
                       void* stream, std::string& err) {
    const int D = (int) g.n_embd;
    const int n = (int) ctx.n_;
    Linear lin{ctx, n, stream};
    if (il < g.n_dense_lead) {
        lin(w.ffn_gate, ctx.h, ctx.fg);
        lin(w.ffn_up, ctx.h, ctx.fu);
        k::swiglu_clamp(ctx.fg, ctx.fu, ctx.fh, (int64_t) n * g.n_ff_dense, (float) g.swiglu_clamp_shexp, stream);
        lin(w.ffn_down, ctx.fh, ctx.f);
        return true;
    }
    const int kU = (int) g.n_expert_used;
    lin(w.gate_inp, ctx.h, ctx.logits);
    k::glm_router(ctx.logits, w.exp_probs_b, n, (int) g.n_expert, kU, (int) g.expert_gating, g.expert_weights_norm,
                  (float) g.expert_weights_scale, ctx.eids, ctx.ew, stream);
    if (!w.expert_ptrs && !ctx.expert_provider) {
        err = "block " + std::to_string(il) + ": no expert addresses and no provider";
        return false;
    }
    k::quantize_q8_1_rows(ctx.h, n, D, ctx.xq, stream);
    if (n == 1) {
        const unsigned long long* ptrs = w.expert_ptrs;
        if (!ptrs) {
            ptrs = ctx.expert_provider->experts(il, ctx.eids, kU, stream, err);
            if (!ptrs) return false;
        }
        k::glm_expert_groups_decode(ctx.eids, ptrs, kU, ctx.grp_ptr, ctx.grp_start, ctx.n_groups, ctx.ent_dst,
                                    ctx.ent_tok, stream);
        k::native_expert_grouped_clamped(w.expert_layout, ctx.grp_ptr, ctx.grp_start, ctx.n_groups, ctx.ent_dst,
                                         ctx.ent_tok, kU, kU, ctx.xq, ctx.exp_scr, ctx.parts,
                                         (float) g.swiglu_clamp_exp, stream);
    } else if (!experts_prompt(g, il, w, ctx, stream, err)) {
        return false;
    }
    lin(w.sh_gate, ctx.h, ctx.fg);
    lin(w.sh_up, ctx.h, ctx.fu);
    k::swiglu_clamp(ctx.fg, ctx.fu, ctx.fh, (int64_t) n * g.n_ff_shexp, (float) g.swiglu_clamp_shexp, stream);
    lin(w.sh_down, ctx.fh, ctx.shared);
    if (n == 1) k::native_moe_combine(ctx.parts, ctx.ew, ctx.shared, ctx.f, D, kU, stream);
    else k::native_moe_combine_multi(ctx.parts, ctx.ew, ctx.shared, ctx.f, D, kU, n, stream);
    return true;
}

namespace {
bool launched(const char* who, std::string& err) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string(who) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}
}  // namespace

bool glm_layer_check(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w, std::string& err,
                     bool provided) {
    const int64_t D = g.n_embd, H = g.n_head, Dh = g.kda_head_dim, di = H * Dh;
    const std::string at = "block " + std::to_string(il) + ": ";
    auto lin = [&](const char* name, const GlmLinear& m, int64_t n_in, int64_t n_out) {
        if (!m.data) {
            err = at + name + " is missing";
            return false;
        }
        if (m.n_in != n_in || m.n_out != n_out) {
            err = at + name + " is " + std::to_string(m.n_out) + " x " + std::to_string(m.n_in) + ", not " +
                  std::to_string(n_out) + " x " + std::to_string(n_in);
            return false;
        }
        if (k::glm_linear_kind(m.type, m.n_in) == k::kLinearNone) {
            err = at + name + ": ggml type " + std::to_string(m.type) + " has no GEMV here";
            return false;
        }
        return true;
    };
    auto have = [&](const char* name, const void* p) {
        if (!p) err = at + name + " is missing";
        return p != nullptr;
    };
    if (il < 0 || il >= g.n_blocks()) {
        err = at + "out of range";
        return false;
    }
    const bool mtp = il >= g.n_layers, mla = g.is_mla[(size_t) il] != 0;
    if (mtp && !mla) {
        err = at + "an MTP block of KDA attention";
        return false;
    }
    if (!mtp && !(have("hc_attn_fn", w.hc_attn_fn) && have("hc_attn_scale", w.hc_attn_scale) &&
                  have("hc_attn_base", w.hc_attn_base) && have("hc_ffn_fn", w.hc_ffn_fn) &&
                  have("hc_ffn_scale", w.hc_ffn_scale) && have("hc_ffn_base", w.hc_ffn_base)))
        return false;
    if (!have("attn_norm", w.attn_norm) || !have("ffn_norm", w.ffn_norm)) return false;
    if (!mla) {
        if (!(lin("attn_q", w.wq, D, di) && lin("attn_k", w.wk, D, di) && lin("attn_v", w.wv, D, di) &&
              lin("ssm_f_a", w.f_a, D, Dh) && lin("ssm_f_b", w.f_b, Dh, di) && lin("ssm_beta", w.beta, D, H) &&
              lin("ssm_g_a", w.g_a, D, Dh) && lin("ssm_g_b", w.g_b, Dh, di) && lin("attn_output", w.wo, di, D) &&
              have("ssm_conv1d_q", w.conv_q) && have("ssm_conv1d_k", w.conv_k) && have("ssm_conv1d_v", w.conv_v) &&
              have("ssm_a", w.ssm_a) && have("ssm_dt.bias", w.dt_bias) && have("ssm_norm", w.o_norm)))
            return false;
    } else {
        if (!(lin("attn_q_a", w.wq_a, D, g.q_lora_rank) && lin("attn_q_b", w.wq_b, g.q_lora_rank, H * g.head_dim_k) &&
              lin("attn_kv_a_mqa", w.wkv_a, D, g.kv_lora_rank) && lin("attn_output", w.wo, H * g.head_dim_v, D) &&
              have("attn_q_a_norm", w.q_a_norm) && have("attn_kv_a_norm", w.kv_a_norm) &&
              have("attn_k_b (BF16)", w.k_b) && have("attn_v_b (BF16)", w.v_b)))
            return false;
        const int64_t ih = g.idx_heads, ik = g.idx_key_dim;
        if (full_indexer(g, il) &&
            !(lin("indexer.attn_q_b", w.idx_q_b, g.q_lora_rank, ih * ik) && lin("indexer.attn_k", w.idx_k, D, ik) &&
              lin("indexer.kpool_gate", w.idx_gate, D, ik) && lin("indexer.proj", w.idx_proj, D, ih) &&
              have("indexer.k_norm", w.idx_k_norm) && have("indexer.k_norm.bias", w.idx_k_norm_b) &&
              have("indexer.kpool_ape", w.idx_ape)))
            return false;
    }
    if (il < g.n_dense_lead)
        return lin("ffn_gate", w.ffn_gate, D, g.n_ff_dense) && lin("ffn_up", w.ffn_up, D, g.n_ff_dense) &&
               lin("ffn_down", w.ffn_down, g.n_ff_dense, D);
    if (!(lin("ffn_gate_inp", w.gate_inp, D, g.n_expert) && have("exp_probs_b", w.exp_probs_b) &&
          (provided || have("the experts' device addresses", w.expert_ptrs)) && lin("ffn_gate_shexp", w.sh_gate, D, g.n_ff_shexp) &&
          lin("ffn_up_shexp", w.sh_up, D, g.n_ff_shexp) && lin("ffn_down_shexp", w.sh_down, g.n_ff_shexp, D)))
        return false;
    const k::NativeExpertLayout& el = w.expert_layout;
    if (el.n_embd != D || el.n_ff != g.n_ff_exp ||
        !k::native_expert_supported(el.gu_type, el.d_type, el.n_embd, el.n_ff)) {
        err = at + "experts of types " + std::to_string(el.gu_type) + " / " + std::to_string(el.d_type) + ", " +
              std::to_string(el.n_ff) + " x " + std::to_string(el.n_embd) + ": not the geometry's, or no kernels";
        return false;
    }
    return true;
}

bool glm_layer_decode(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w, const GlmLayerState& st,
                      GlmContext& ctx, float* x, void* stream, std::string& err) {
    if (!GlmLayerImpl::ready(g, il, ctx, x, "glm_layer_decode", err)) return false;
    if (il >= g.n_layers) {
        err = "glm_layer_decode: block " + std::to_string(il) + " is the MTP block (glm_mtp_layer_decode)";
        return false;
    }
    const int D = (int) g.n_embd, n = (int) ctx.tokens();
    const float rms_eps = (float) g.rms_eps, hc_eps = (float) g.hc_eps;
    const int iters = (int) g.hc_sinkhorn_iters;

    k::mhc_pre(x, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, n, D, rms_eps, hc_eps, iters, ctx.mhc_scr, ctx.h,
               ctx.post, ctx.comb, nullptr, stream, w.attn_norm, rms_eps);   // and the attention's norm
    if (!GlmLayerImpl::attention(g, il, w, st, ctx, stream, err)) return false;
    k::mhc_post(ctx.f, ctx.post, ctx.comb, x, n, D, stream);

    k::mhc_pre(x, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, n, D, rms_eps, hc_eps, iters, ctx.mhc_scr, ctx.h,
               ctx.post, ctx.comb, nullptr, stream, w.ffn_norm, rms_eps);   // and the FFN's norm
    if (!GlmLayerImpl::ffn(g, il, w, ctx, stream, err)) return false;
    k::mhc_post(ctx.f, ctx.post, ctx.comb, x, n, D, stream);
    return launched("glm_layer_decode", err);
}

bool glm_mtp_layer_decode(const core::Glm5NextGeometry& g, int64_t il, const GlmLayerWeights& w,
                          const GlmLayerState& st, GlmContext& ctx, float* x, void* stream, std::string& err) {
    if (!GlmLayerImpl::ready(g, il, ctx, x, "glm_mtp_layer_decode", err)) return false;
    if (il < g.n_layers || !g.is_mla[(size_t) il]) {
        err = "glm_mtp_layer_decode: block " + std::to_string(il) + " is not an MLA MTP block";
        return false;
    }
    const int D = (int) g.n_embd, n = (int) ctx.tokens();
    const float rms_eps = (float) g.rms_eps;
    const cudaStream_t s = (cudaStream_t) stream;
    cudaMemcpyAsync(ctx.h, x, (size_t) n * D * sizeof(float), cudaMemcpyDeviceToDevice, s);
    k::glm_rms_norm(ctx.h, w.attn_norm, n, D, rms_eps, stream);
    if (!GlmLayerImpl::attention(g, il, w, st, ctx, stream, err)) return false;
    k::add_inplace(x, ctx.f, (int64_t) n * D, stream);
    cudaMemcpyAsync(ctx.h, x, (size_t) n * D * sizeof(float), cudaMemcpyDeviceToDevice, s);
    k::glm_rms_norm(ctx.h, w.ffn_norm, n, D, rms_eps, stream);
    if (!GlmLayerImpl::ffn(g, il, w, ctx, stream, err)) return false;
    k::add_inplace(x, ctx.f, (int64_t) n * D, stream);
    return launched("glm_mtp_layer_decode", err);
}

}  // namespace strata::glm
