// src/glm/glm_loader.cpp - see include/strata/glm/glm_loader.hpp.
#include "strata/glm/glm_loader.hpp"

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/glm_linear.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <map>

namespace strata::glm {
namespace k = strata::kernels;
namespace {

uint64_t align256(uint64_t n) { return (n + 255) & ~(uint64_t) 255; }

enum class Kind { Linear, Vec, Bf16, Embed, ExpGate, ExpUp, ExpDown };

/// One tensor the model reads: its name and shape (GGUF order, dim 0 fastest), what it becomes, and where it goes.
struct Need {
    std::string name;
    std::vector<uint64_t> shape;
    Kind kind = Kind::Linear;
    bool optional = false;
    GlmLinear* lin = nullptr;
    const float** vec = nullptr;
    const uint16_t** bf16 = nullptr;
    GlmEmbedding* emb = nullptr;
    int64_t il = -1;
    // resolved by plan
    const TensorInfo* t = nullptr;
    size_t shard = 0;
};

std::vector<uint64_t> trimmed(std::vector<uint64_t> s) {   // ggml's trailing dimensions of 1 are not stored, or are
    while (s.size() > 1 && s.back() == 1) s.pop_back();
    return s;
}
std::string shape_str(const std::vector<uint64_t>& s) {
    std::string r = "{";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "}";
}
const char* role(Kind kd) {
    switch (kd) {
        case Kind::Linear: return "projection";
        case Kind::Vec: return "vector";
        case Kind::Bf16: return "attn_k_b / attn_v_b";
        case Kind::Embed: return "embedding";
        case Kind::ExpGate: case Kind::ExpUp: return "experts gate / up";
        case Kind::ExpDown: return "experts down";
    }
    return "?";
}
bool small_float(uint32_t type) { return type == 0 || type == 1 || type == 30; }

/// The tensors of llama.cpp's load_arch_tensors for glm5-next, with their destinations in `w` (whose blocks are
/// sized here, so the pointers stay valid).
std::vector<Need> needs(const core::Glm5NextGeometry& g, bool mtp, GlmModelWeights& w) {
    const uint64_t D = (uint64_t) g.n_embd, V = (uint64_t) g.n_vocab, H = (uint64_t) g.n_head;
    const uint64_t Dh = (uint64_t) g.kda_head_dim, di = H * Dh, L = (uint64_t) g.kv_lora_rank;
    const uint64_t QL = (uint64_t) g.q_lora_rank, DK = (uint64_t) g.head_dim_k, DV = (uint64_t) g.head_dim_v;
    const uint64_t IH = (uint64_t) g.idx_heads, IK = (uint64_t) g.idx_key_dim, HC = (uint64_t) g.hc;
    const uint64_t MIX = (2 + HC) * HC, NE = (uint64_t) g.n_expert, F = (uint64_t) g.n_ff_exp;
    const uint64_t FS = (uint64_t) g.n_ff_shexp, FD = (uint64_t) g.n_ff_dense, NR = (uint64_t) g.n_rot;
    w = GlmModelWeights{};
    w.blocks.assign((size_t) g.n_blocks(), GlmLayerWeights{});
    std::vector<Need> r;
    auto lin = [&](std::string n, std::vector<uint64_t> s, GlmLinear* d, bool opt = false) {
        Need x; x.name = std::move(n); x.shape = std::move(s); x.kind = Kind::Linear; x.lin = d; x.optional = opt;
        r.push_back(std::move(x));
    };
    auto vec = [&](std::string n, std::vector<uint64_t> s, const float** d, bool opt = false) {
        Need x; x.name = std::move(n); x.shape = std::move(s); x.kind = Kind::Vec; x.vec = d; x.optional = opt;
        r.push_back(std::move(x));
    };
    auto bf16 = [&](std::string n, std::vector<uint64_t> s, const uint16_t** d) {
        Need x; x.name = std::move(n); x.shape = std::move(s); x.kind = Kind::Bf16; x.bf16 = d;
        r.push_back(std::move(x));
    };
    auto emb = [&](std::string n, GlmEmbedding* d, bool opt = false) {
        Need x; x.name = std::move(n); x.shape = {D, V}; x.kind = Kind::Embed; x.emb = d; x.optional = opt;
        r.push_back(std::move(x));
    };
    auto experts = [&](std::string n, std::vector<uint64_t> s, Kind kd, int64_t il) {
        Need x; x.name = std::move(n); x.shape = std::move(s); x.kind = kd; x.il = il;
        r.push_back(std::move(x));
    };

    emb("token_embd.weight", &w.tok_embd);
    const int64_t n_load = g.n_layers + (mtp ? 1 : 0);
    for (int64_t il = 0; il < n_load; ++il) {
        GlmLayerWeights& b = w.blocks[(size_t) il];
        const std::string p = "blk." + std::to_string(il) + ".";
        vec(p + "attn_norm.weight", {D}, &b.attn_norm);
        vec(p + "ffn_norm.weight", {D}, &b.ffn_norm);
        if (il < g.n_layers) {
            vec(p + "hc_attn_fn.weight", {HC * D, MIX}, &b.hc_attn_fn);
            vec(p + "hc_attn_base.weight", {MIX}, &b.hc_attn_base);
            vec(p + "hc_attn_scale.weight", {3}, &b.hc_attn_scale);
            vec(p + "hc_ffn_fn.weight", {HC * D, MIX}, &b.hc_ffn_fn);
            vec(p + "hc_ffn_base.weight", {MIX}, &b.hc_ffn_base);
            vec(p + "hc_ffn_scale.weight", {3}, &b.hc_ffn_scale);
        }
        if (!g.is_mla[(size_t) il]) {
            const uint64_t dc = (uint64_t) g.ssm_d_conv;
            lin(p + "attn_q.weight", {D, di}, &b.wq);
            lin(p + "attn_k.weight", {D, di}, &b.wk);
            lin(p + "attn_v.weight", {D, di}, &b.wv);
            vec(p + "ssm_conv1d_q.weight", {dc, 1, di, 1}, &b.conv_q);
            vec(p + "ssm_conv1d_k.weight", {dc, 1, di, 1}, &b.conv_k);
            vec(p + "ssm_conv1d_v.weight", {dc, 1, di, 1}, &b.conv_v);
            lin(p + "ssm_f_a.weight", {D, Dh}, &b.f_a);
            lin(p + "ssm_f_b.weight", {Dh, di}, &b.f_b);
            lin(p + "ssm_beta.weight", {D, H}, &b.beta);
            vec(p + "ssm_a", {H}, &b.ssm_a);
            vec(p + "ssm_dt.bias", {di}, &b.dt_bias);
            lin(p + "ssm_g_a.weight", {D, Dh}, &b.g_a);
            lin(p + "ssm_g_b.weight", {Dh, di}, &b.g_b);
            vec(p + "ssm_norm.weight", {Dh}, &b.o_norm);
            lin(p + "attn_output.weight", {di, D}, &b.wo);
        } else {
            vec(p + "attn_q_a_norm.weight", {QL}, &b.q_a_norm);
            vec(p + "attn_kv_a_norm.weight", {L}, &b.kv_a_norm);
            lin(p + "attn_q_a.weight", {D, QL}, &b.wq_a);
            lin(p + "attn_q_b.weight", {QL, H * DK}, &b.wq_b);
            lin(p + "attn_kv_a_mqa.weight", {D, L + NR}, &b.wkv_a);
            bf16(p + "attn_k_b.weight", {DK - NR, L, H}, &b.k_b);
            bf16(p + "attn_v_b.weight", {L, DV, H}, &b.v_b);
            lin(p + "attn_output.weight", {H * DV, D}, &b.wo);
            if (il >= g.n_layers || g.idx_full.empty() || g.idx_full[(size_t) il]) {
                vec(p + "indexer.k_norm.weight", {IK}, &b.idx_k_norm);
                vec(p + "indexer.k_norm.bias", {IK}, &b.idx_k_norm_b);
                lin(p + "indexer.proj.weight", {D, IH}, &b.idx_proj);
                lin(p + "indexer.attn_k.weight", {D, IK}, &b.idx_k);
                lin(p + "indexer.attn_q_b.weight", {QL, IH * IK}, &b.idx_q_b);
                lin(p + "indexer_compressor_gate.weight", {D, IK}, &b.idx_gate);
                vec(p + "indexer_compressor_ape.weight", {IK, (uint64_t) g.idx_kpool}, &b.idx_ape);
            }
        }
        if (il < g.n_dense_lead) {
            lin(p + "ffn_gate.weight", {D, FD}, &b.ffn_gate);
            lin(p + "ffn_up.weight", {D, FD}, &b.ffn_up);
            lin(p + "ffn_down.weight", {FD, D}, &b.ffn_down);
        } else {
            lin(p + "ffn_gate_inp.weight", {D, NE}, &b.gate_inp);
            vec(p + "exp_probs_b.bias", {NE}, &b.exp_probs_b);
            experts(p + "ffn_gate_exps.weight", {D, F, NE}, Kind::ExpGate, il);
            experts(p + "ffn_up_exps.weight", {D, F, NE}, Kind::ExpUp, il);
            experts(p + "ffn_down_exps.weight", {F, D, NE}, Kind::ExpDown, il);
            lin(p + "ffn_gate_shexp.weight", {D, FS}, &b.sh_gate);
            lin(p + "ffn_up_shexp.weight", {D, FS}, &b.sh_up);
            lin(p + "ffn_down_shexp.weight", {FS, D}, &b.sh_down);
        }
        if (il >= g.n_layers) {
            lin(p + "nextn.eh_proj.weight", {2 * D, D}, &w.eh_proj);
            vec(p + "nextn.enorm.weight", {D}, &w.enorm);
            vec(p + "nextn.hnorm.weight", {D}, &w.hnorm);
            emb(p + "nextn.embed_tokens.weight", &w.mtp_embd, true);
            lin(p + "nextn.shared_head_head.weight", {D, V}, &w.mtp_head, true);
            vec(p + "nextn.shared_head_norm.weight", {D}, &w.mtp_head_norm, true);
        }
    }
    vec("output_norm.weight", {D}, &w.output_norm);
    lin("output.weight", {D, V}, &w.output);
    return r;
}

/// Resolves every need against the file and checks it; fills `plan`.  "" or the first problem.
std::string resolve(const GgufModel& m, const core::Glm5NextGeometry& g, std::vector<Need>& ns, GlmPlan& plan) {
    plan = GlmPlan{};
    std::map<std::string, int> types;
    std::map<int64_t, const Need*> gate, up, down;
    for (Need& n : ns) {
        n.t = m.find(n.name, &n.shard);
        if (!n.t) {
            if (n.optional) continue;
            return n.name + ": missing from the file";
        }
        const TensorInfo& t = *n.t;
        if (trimmed(t.shape) != trimmed(n.shape))
            return n.name + ": shape " + shape_str(t.shape) + ", the geometry gives " + shape_str(n.shape);
        if (!m.in_bounds(t, n.shard))
            return n.name + ": " + t.type_name() + " payload unknown or past the end of " + m.shard(n.shard).path();
        const uint64_t bytes = tensor_payload_bytes(t);
        bool ok = true;
        switch (n.kind) {
            case Kind::Linear:
                ok = k::glm_linear_kind((int) t.type, (int64_t) t.shape[0]) != k::kLinearNone;
                plan.device_bytes += align256(bytes);
                break;
            case Kind::Vec:
                ok = small_float(t.type);
                plan.device_bytes += align256(t.elements() * 4);
                break;
            case Kind::Bf16:
                ok = small_float(t.type) || (k::embed_type_supported((int) t.type) && t.elements() % 256 == 0);
                plan.device_bytes += align256(t.elements() * 2);
                break;
            case Kind::Embed:
                ok = k::embed_type_supported((int) t.type) && k::iq_row_bytes((int) t.type, g.n_embd) > 0;
                plan.device_bytes += align256(bytes);
                break;
            case Kind::ExpGate: gate[n.il] = &n; break;
            case Kind::ExpUp: up[n.il] = &n; break;
            case Kind::ExpDown: down[n.il] = &n; break;
        }
        if (!ok) return n.name + ": " + t.type_name() + " is not a type the " + role(n.kind) + " kernels take";
        plan.file_bytes += bytes;
        ++plan.tensors;
        ++types[std::string(role(n.kind)) + ": " + t.type_name()];
    }
    for (const auto& [il, gn] : gate) {
        const Need* un = up[il];
        const Need* dn = down[il];
        if (gn->t->type != un->t->type)
            return gn->name + ": " + gn->t->type_name() + " and its up_exps " + un->t->type_name() +
                   ": one blob holds both, of one type";
        if (!k::native_expert_supported((int) gn->t->type, (int) dn->t->type, g.n_embd, g.n_ff_exp))
            return gn->name + ": experts of " + gn->t->type_name() + " / " + dn->t->type_name() + " have no kernels";
        const k::NativeExpertLayout el =
            k::native_expert_layout((int) gn->t->type, (int) dn->t->type, g.n_embd, g.n_ff_exp);
        plan.expert_bytes += align256(el.bytes) * (uint64_t) g.n_expert;
        plan.device_bytes += align256((uint64_t) g.n_expert * 8);   // the blobs' addresses
    }
    for (const auto& [name, count] : types) plan.types.emplace_back(name, count);
    return "";
}

bool cuda_ok(cudaError_t e, const std::string& what, std::string& err) {
    if (e == cudaSuccess) return true;
    cudaGetLastError();
    err = "GlmLoadedWeights: " + what + ": " + cudaGetErrorString(e);
    return false;
}

uint16_t bf16_round(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float bf16_value(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
/// F32, F16 or BF16 values to float.
std::vector<float> floats(const uint8_t* p, uint32_t type, uint64_t n) {
    std::vector<float> v((size_t) n);
    for (uint64_t i = 0; i < n; ++i) {
        if (type == 0) std::memcpy(&v[(size_t) i], p + i * 4, 4);
        else {
            uint16_t h;
            std::memcpy(&h, p + i * 2, 2);
            v[(size_t) i] = type == 1 ? k::f32_from_f16(h) : bf16_value(h);
        }
    }
    return v;
}

}  // namespace

std::string glm_plan(const GgufModel& m, const core::Glm5NextGeometry& g, const GlmLoadOptions& opt, GlmPlan& plan) {
    GlmModelWeights w;
    const bool mtp = opt.mtp && g.n_nextn > 0;
    std::vector<Need> ns = needs(g, mtp, w);
    const std::string e = resolve(m, g, ns, plan);
    plan.mtp = mtp;
    return e;
}

GlmLoadedWeights::~GlmLoadedWeights() { release(); }

void GlmLoadedWeights::release() {
    for (void* p : device_) cudaFree(p);
    for (void* p : host_) cudaFreeHost(p);
    cache_.reset();
    device_.clear();
    host_.clear();
    w_ = GlmModelWeights{};
}

bool GlmLoadedWeights::load(const GgufModel& m, const core::Glm5NextGeometry& g, const GlmLoadOptions& opt,
                            std::string& err) {
    release();
    const bool mtp = opt.mtp && g.n_nextn > 0;
    std::vector<Need> ns = needs(g, mtp, w_);
    if (const std::string e = resolve(m, g, ns, plan_); !e.empty()) {
        err = "GlmLoadedWeights: " + e;
        release();
        return false;
    }
    plan_.mtp = mtp;
    auto fail = [&]() { release(); return false; };
    // before a byte is copied: does it fit (a 157 GB file should fail here, not after an hour of copying)
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        uint64_t experts = plan_.expert_bytes;
        if (opt.expert_cache_slots > 0) {   // the cache's slots, of the largest blob
            uint64_t slot = 0;
            for (const Need& n : ns)
                if (n.kind == Kind::ExpGate && n.t) {
                    const Need* dn = nullptr;
                    for (const Need& d : ns)
                        if (d.kind == Kind::ExpDown && d.il == n.il) dn = &d;
                    slot = std::max<uint64_t>(slot, align256(k::native_expert_layout((int) n.t->type, (int) dn->t->type,
                                                                                     g.n_embd, g.n_ff_exp).bytes));
                }
            experts = slot * (uint64_t) opt.expert_cache_slots;
        }
        const uint64_t need = plan_.device_bytes + (opt.experts == GlmExpertHome::Device ? experts : 0);
        if (need > free_b) {
            err = "GlmLoadedWeights: the weights need " + std::to_string(need >> 20) + " MiB of device memory" +
                  (opt.experts == GlmExpertHome::Device ? " (the experts included)" : "") + ", " +
                  std::to_string(free_b >> 20) + " MiB are free";
            return fail();
        }
    }

    // everything but the experts' blobs: one device arena
    uint8_t* arena = nullptr;
    if (!cuda_ok(cudaMalloc((void**) &arena, plan_.device_bytes ? plan_.device_bytes : 256),
                 std::to_string(plan_.device_bytes) + " bytes of weights", err))
        return fail();
    device_.push_back(arena);
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { uint8_t* p = arena + at; at += align256(bytes); return p; };
    std::map<int64_t, const Need*> gate, up, down;
    for (const Need& n : ns) {
        if (!n.t) continue;
        const TensorInfo& t = *n.t;
        const uint8_t* src = m.shard(n.shard).tensor_data(t);
        const uint64_t bytes = tensor_payload_bytes(t);
        switch (n.kind) {
            case Kind::Linear: {
                uint8_t* d = take(bytes);
                if (!cuda_ok(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice), n.name, err)) return fail();
                *n.lin = GlmLinear{(int) t.type, d, (int64_t) t.shape[0], (int64_t) t.shape[1]};
                break;
            }
            case Kind::Embed: {
                uint8_t* d = take(bytes);
                if (!cuda_ok(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice), n.name, err)) return fail();
                *n.emb = GlmEmbedding{(int) t.type, d, k::iq_row_bytes((int) t.type, g.n_embd)};
                break;
            }
            case Kind::Vec: {
                const std::vector<float> v = floats(src, t.type, t.elements());
                uint8_t* d = take(v.size() * 4);
                if (!cuda_ok(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice), n.name, err)) return fail();
                *n.vec = (const float*) d;
                break;
            }
            case Kind::Bf16: {
                const uint64_t ne = t.elements();
                uint16_t* d = (uint16_t*) take(ne * 2);
                if (small_float(t.type)) {
                    std::vector<uint16_t> v((size_t) ne);
                    if (t.type == 30) std::memcpy(v.data(), src, ne * 2);
                    else {
                        const std::vector<float> f = floats(src, t.type, ne);
                        for (uint64_t i = 0; i < ne; ++i) v[(size_t) i] = bf16_round(f[(size_t) i]);
                    }
                    if (!cuda_ok(cudaMemcpy(d, v.data(), ne * 2, cudaMemcpyHostToDevice), n.name, err)) return fail();
                } else {   // quantized: dequantize on the device, then round to BF16
                    void* raw = nullptr;
                    float* f = nullptr;
                    bool ok = cuda_ok(cudaMalloc(&raw, bytes), n.name, err) &&
                              cuda_ok(cudaMalloc((void**) &f, ne * 4), n.name, err) &&
                              cuda_ok(cudaMemcpy(raw, src, bytes, cudaMemcpyHostToDevice), n.name, err);
                    if (ok) {
                        k::iq_dequant_f32((int) t.type, raw, (int64_t) ne, f, nullptr);
                        k::f32_to_bf16_bulk(f, d, (int64_t) ne, nullptr);
                        ok = cuda_ok(cudaDeviceSynchronize(), n.name, err);
                    }
                    if (raw) cudaFree(raw);
                    if (f) cudaFree(f);
                    if (!ok) return fail();
                }
                *n.bf16 = d;
                break;
            }
            case Kind::ExpGate: gate[n.il] = &n; break;
            case Kind::ExpUp: up[n.il] = &n; break;
            case Kind::ExpDown: down[n.il] = &n; break;
        }
    }

    const int64_t NE = g.n_expert;
    // streamed: the experts stay in the file, the cache reads them
    if (opt.expert_cache_slots > 0) {
        std::vector<GlmExpertSource> src((size_t) g.n_blocks());
        for (const auto& [il, gn] : gate) {
            const Need* un = up[il];
            const Need* dn = down[il];
            GlmExpertSource& s = src[(size_t) il];
            s.gate = m.shard(gn->shard).tensor_data(*gn->t);
            s.up = m.shard(un->shard).tensor_data(*un->t);
            s.down = m.shard(dn->shard).tensor_data(*dn->t);
            s.layout = k::native_expert_layout((int) gn->t->type, (int) dn->t->type, g.n_embd, g.n_ff_exp);
            w_.blocks[(size_t) il].expert_layout = s.layout;
            w_.blocks[(size_t) il].expert_ptrs = nullptr;
        }
        cache_ = std::make_unique<GlmExpertCache>();
        if (!cache_->init(std::move(src), NE, opt.expert_cache_slots, opt.experts == GlmExpertHome::MappedHost,
                          opt.cache_threads, err) ||
            (opt.cache_fill && !cache_->fill_all(err)))
            return fail();
        return true;
    }
    // the experts: per layer, one allocation of n_expert blobs [gate rows | up rows | down rows]
    for (const auto& [il, gn] : gate) {
        const Need* un = up[il];
        const Need* dn = down[il];
        const k::NativeExpertLayout el =
            k::native_expert_layout((int) gn->t->type, (int) dn->t->type, g.n_embd, g.n_ff_exp);
        const uint64_t slot = align256(el.bytes), total = slot * (uint64_t) NE;
        const uint8_t* gs = m.shard(gn->shard).tensor_data(*gn->t);
        const uint8_t* us = m.shard(un->shard).tensor_data(*un->t);
        const uint8_t* ds = m.shard(dn->shard).tensor_data(*dn->t);
        const uint64_t gu_slab = el.up_off, d_slab = el.bytes - el.down_off;
        uint8_t* blobs = nullptr;    // the address the GPU reads
        const std::string what = "blk." + std::to_string(il) + " experts, " + std::to_string(total) + " bytes";
        if (opt.experts == GlmExpertHome::Device) {
            if (!cuda_ok(cudaMalloc((void**) &blobs, total), what, err)) return fail();
            device_.push_back(blobs);
            for (int64_t e = 0; e < NE; ++e) {
                uint8_t* b = blobs + slot * (uint64_t) e;
                if (!cuda_ok(cudaMemcpy(b, gs + gu_slab * (uint64_t) e, gu_slab, cudaMemcpyHostToDevice), what, err) ||
                    !cuda_ok(cudaMemcpy(b + el.up_off, us + gu_slab * (uint64_t) e, gu_slab, cudaMemcpyHostToDevice),
                             what, err) ||
                    !cuda_ok(cudaMemcpy(b + el.down_off, ds + d_slab * (uint64_t) e, d_slab, cudaMemcpyHostToDevice),
                             what, err))
                    return fail();
            }
        } else {
            uint8_t* host = nullptr;
            if (!cuda_ok(cudaHostAlloc((void**) &host, total, cudaHostAllocMapped), what, err)) return fail();
            host_.push_back(host);
            for (int64_t e = 0; e < NE; ++e) {
                uint8_t* b = host + slot * (uint64_t) e;
                std::memcpy(b, gs + gu_slab * (uint64_t) e, gu_slab);
                std::memcpy(b + el.up_off, us + gu_slab * (uint64_t) e, gu_slab);
                std::memcpy(b + el.down_off, ds + d_slab * (uint64_t) e, d_slab);
            }
            if (!cuda_ok(cudaHostGetDevicePointer((void**) &blobs, host, 0), what, err)) return fail();
        }
        std::vector<unsigned long long> ptrs((size_t) NE);
        for (int64_t e = 0; e < NE; ++e) ptrs[(size_t) e] = (unsigned long long) (uintptr_t) (blobs + slot * (uint64_t) e);
        uint8_t* pd = take(ptrs.size() * 8);
        if (!cuda_ok(cudaMemcpy(pd, ptrs.data(), ptrs.size() * 8, cudaMemcpyHostToDevice), what, err)) return fail();
        GlmLayerWeights& b = w_.blocks[(size_t) il];
        b.expert_layout = el;
        b.expert_ptrs = (const unsigned long long*) pd;
        b.expert_base = blobs;
        b.expert_stride = slot;
    }
    return true;
}

}  // namespace strata::glm
