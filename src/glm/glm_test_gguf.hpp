// src/glm/glm_test_gguf.hpp - the GLM tests' synthetic model (glm_ref.hpp's build_test_model) written as a glm5-next
// GGUF: the tensors under llama.cpp's names, in its converter's shapes and layouts, with the metadata read_geometry
// reads.  Test-only (one translation unit each: an unnamed namespace).
#pragma once

#include "glm_ref.hpp"

#include "../../tests/core/gguf_fixture.hpp"

#include <chrono>
#include <filesystem>

namespace {

namespace fs = std::filesystem;

/// n bytes of device memory, to the host.
vector<uint8_t> raw(const void* dev, size_t n) {
    vector<uint8_t> v(n);
    check(cudaMemcpy(v.data(), dev, n, cudaMemcpyDeviceToHost), "download");
    return v;
}

uint64_t payload(uint32_t type, std::vector<uint64_t> shape) {
    strata::TensorInfo t;
    t.shape = std::move(shape);
    t.type = type;
    return strata::tensor_payload_bytes(t);
}
fixture::Tensor tensor(const std::string& name, std::vector<uint64_t> shape, uint32_t type, vector<uint8_t> data) {
    fixture::Tensor t;
    t.name = name;
    t.shape = std::move(shape);
    t.type = type;
    t.data = std::move(data);
    return t;
}
fixture::Tensor lin_t(const std::string& name, const Lin& L) {
    const std::vector<uint64_t> s = {(uint64_t) L.dev.n_in, (uint64_t) L.dev.n_out};
    return tensor(name, s, (uint32_t) L.dev.type, raw(L.dev.data, (size_t) payload((uint32_t) L.dev.type, s)));
}
fixture::Tensor f32_t(const std::string& name, std::vector<uint64_t> shape, const vector<float>& v) {
    vector<uint8_t> b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return tensor(name, std::move(shape), 0, std::move(b));
}
fixture::Tensor bf16_t(const std::string& name, std::vector<uint64_t> shape, const vector<float>& v) {
    vector<uint8_t> b(v.size() * 2);
    for (size_t i = 0; i < v.size(); ++i) {
        const uint16_t h = bf16_bits(v[i]);
        std::memcpy(&b[i * 2], &h, 2);
    }
    return tensor(name, std::move(shape), 30, std::move(b));
}

/// The file's tensors for the model, as llama.cpp's converter writes them.
std::vector<fixture::Tensor> model_tensors(const TestModel& tm) {
    const Glm5NextGeometry& g = tm.g;
    const uint64_t D = (uint64_t) g.n_embd, H = (uint64_t) g.n_head, di = H * 128, L = 512;
    std::vector<fixture::Tensor> ts;
    ts.push_back(lin_t("token_embd.weight", tm.tok));
    for (int64_t il = 0; il < g.n_blocks(); ++il) {
        const Block& b = tm.blocks[(size_t) il];
        const std::string p = "blk." + std::to_string(il) + ".";
        ts.push_back(f32_t(p + "attn_norm.weight", {D}, b.attn_norm));
        ts.push_back(f32_t(p + "ffn_norm.weight", {D}, b.ffn_norm));
        if (il < g.n_layers) {
            ts.push_back(f32_t(p + "hc_attn_fn.weight", {4 * D, 24}, b.hc_attn_fn));
            ts.push_back(f32_t(p + "hc_attn_base.weight", {24}, b.hc_attn_base));
            ts.push_back(f32_t(p + "hc_attn_scale.weight", {3}, b.hc_attn_scale));
            ts.push_back(f32_t(p + "hc_ffn_fn.weight", {4 * D, 24}, b.hc_ffn_fn));
            ts.push_back(f32_t(p + "hc_ffn_base.weight", {24}, b.hc_ffn_base));
            ts.push_back(f32_t(p + "hc_ffn_scale.weight", {3}, b.hc_ffn_scale));
        }
        if (!g.is_mla[(size_t) il]) {
            ts.push_back(lin_t(p + "attn_q.weight", b.wq));
            ts.push_back(lin_t(p + "attn_k.weight", b.wk));
            ts.push_back(lin_t(p + "attn_v.weight", b.wv));
            ts.push_back(f32_t(p + "ssm_conv1d_q.weight", {4, 1, di, 1}, b.conv_q));
            ts.push_back(f32_t(p + "ssm_conv1d_k.weight", {4, 1, di, 1}, b.conv_k));
            ts.push_back(f32_t(p + "ssm_conv1d_v.weight", {4, 1, di, 1}, b.conv_v));
            ts.push_back(lin_t(p + "ssm_f_a.weight", b.f_a));
            ts.push_back(lin_t(p + "ssm_f_b.weight", b.f_b));
            ts.push_back(lin_t(p + "ssm_beta.weight", b.beta));
            ts.push_back(f32_t(p + "ssm_a", {H}, b.ssm_a));
            ts.push_back(f32_t(p + "ssm_dt.bias", {di}, b.dt_bias));
            ts.push_back(lin_t(p + "ssm_g_a.weight", b.g_a));
            ts.push_back(lin_t(p + "ssm_g_b.weight", b.g_b));
            ts.push_back(f32_t(p + "ssm_norm.weight", {128}, b.o_norm));
            ts.push_back(lin_t(p + "attn_output.weight", b.wo));
        } else {
            ts.push_back(f32_t(p + "attn_q_a_norm.weight", {(uint64_t) g.q_lora_rank}, b.q_a_norm));
            ts.push_back(f32_t(p + "attn_kv_a_norm.weight", {L}, b.kv_a_norm));
            ts.push_back(lin_t(p + "attn_q_a.weight", b.wq_a));
            ts.push_back(lin_t(p + "attn_q_b.weight", b.wq_b));
            ts.push_back(lin_t(p + "attn_kv_a_mqa.weight", b.wkv_a));
            ts.push_back(bf16_t(p + "attn_k_b.weight", {256, L, H}, b.k_b));
            ts.push_back(bf16_t(p + "attn_v_b.weight", {L, 256, H}, b.v_b));
            ts.push_back(lin_t(p + "attn_output.weight", b.wo));
            if (full_idx(g, il)) {
                ts.push_back(f32_t(p + "indexer.k_norm.weight", {128}, b.idx_k_norm));
                ts.push_back(f32_t(p + "indexer.k_norm.bias", {128}, b.idx_k_norm_b));
                ts.push_back(lin_t(p + "indexer.proj.weight", b.idx_proj));
                ts.push_back(lin_t(p + "indexer.attn_k.weight", b.idx_k));
                ts.push_back(lin_t(p + "indexer.attn_q_b.weight", b.idx_q_b));
                ts.push_back(lin_t(p + "indexer_compressor_gate.weight", b.idx_gate));
                ts.push_back(f32_t(p + "indexer_compressor_ape.weight", {128, 4}, b.idx_ape));
            }
        }
        if (il < g.n_dense_lead) {
            ts.push_back(lin_t(p + "ffn_gate.weight", b.ffn_gate));
            ts.push_back(lin_t(p + "ffn_up.weight", b.ffn_up));
            ts.push_back(lin_t(p + "ffn_down.weight", b.ffn_down));
        } else {
            ts.push_back(lin_t(p + "ffn_gate_inp.weight", b.gate_inp));
            ts.push_back(f32_t(p + "exp_probs_b.bias", {(uint64_t) g.n_expert}, b.exp_probs_b));
            // the blobs [gate | up | down] back into three stacked tensors
            const k::NativeExpertLayout& el = b.w.expert_layout;
            const vector<unsigned long long> ptrs = download(b.w.expert_ptrs, (size_t) g.n_expert);
            vector<uint8_t> gs, us, ds;
            for (unsigned long long a : ptrs) {
                const vector<uint8_t> blob = raw((const void*) (uintptr_t) a, el.bytes);
                gs.insert(gs.end(), blob.begin(), blob.begin() + (ptrdiff_t) el.up_off);
                us.insert(us.end(), blob.begin() + (ptrdiff_t) el.up_off, blob.begin() + (ptrdiff_t) el.down_off);
                ds.insert(ds.end(), blob.begin() + (ptrdiff_t) el.down_off, blob.end());
            }
            const uint64_t F = (uint64_t) g.n_ff_exp, NE = (uint64_t) g.n_expert;
            ts.push_back(tensor(p + "ffn_gate_exps.weight", {D, F, NE}, (uint32_t) el.gu_type, gs));
            ts.push_back(tensor(p + "ffn_up_exps.weight", {D, F, NE}, (uint32_t) el.gu_type, us));
            ts.push_back(tensor(p + "ffn_down_exps.weight", {F, D, NE}, (uint32_t) el.d_type, ds));
            ts.push_back(lin_t(p + "ffn_gate_shexp.weight", b.sh_gate));
            ts.push_back(lin_t(p + "ffn_up_shexp.weight", b.sh_up));
            ts.push_back(lin_t(p + "ffn_down_shexp.weight", b.sh_down));
        }
        if (il >= g.n_layers) {
            ts.push_back(lin_t(p + "nextn.eh_proj.weight", tm.eh));
            ts.push_back(f32_t(p + "nextn.enorm.weight", {D}, tm.enorm));
            ts.push_back(f32_t(p + "nextn.hnorm.weight", {D}, tm.hnorm));
            ts.push_back(f32_t(p + "nextn.shared_head_norm.weight", {D}, tm.head_norm));
        }
    }
    ts.push_back(f32_t("output_norm.weight", {D}, tm.out_norm));
    ts.push_back(lin_t("output.weight", tm.out));
    return ts;
}

/// The metadata llama.cpp's converter writes, for the model's geometry.
std::vector<fixture::Kv> model_keys(const Glm5NextGeometry& g) {
    using namespace fixture;
    const std::string P = "glm5-next.";
    std::vector<uint64_t> kv, types;
    for (uint8_t m : g.is_mla) kv.push_back(m);
    for (uint8_t t : g.idx_full) types.push_back(t);
    const size_t B = (size_t) g.n_blocks();
    return {str("general.architecture", "glm5-next"),
            u32(P + "block_count", (uint64_t) g.n_blocks()),
            u32(P + "context_length", (uint64_t) g.n_ctx_train),
            u32(P + "embedding_length", (uint64_t) g.n_embd),
            u32(P + "feed_forward_length", (uint64_t) g.n_ff_dense),
            u32(P + "attention.head_count", (uint64_t) g.n_head),
            u32s(P + "attention.head_count_kv", kv),
            u32(P + "vocab_size", (uint64_t) g.n_vocab),
            f32(P + "attention.layer_norm_rms_epsilon", (float) g.rms_eps),
            f32(P + "attention.layer_norm_epsilon", (float) g.norm_eps),
            u32(P + "nextn_predict_layers", (uint64_t) g.n_nextn),
            u32(P + "ssm.conv_kernel", (uint64_t) g.ssm_d_conv),
            u32(P + "kda.head_dim", (uint64_t) g.kda_head_dim),
            f32(P + "kda.gate_lower_bound", (float) g.kda_gate_lower_bound),
            u32(P + "attention.q_lora_rank", (uint64_t) g.q_lora_rank),
            u32(P + "attention.kv_lora_rank", (uint64_t) g.kv_lora_rank),
            u32(P + "rope.dimension_count", 0),
            u32(P + "attention.key_length", (uint64_t) g.kv_lora_rank),
            u32(P + "attention.value_length", (uint64_t) g.kv_lora_rank),
            u32(P + "attention.key_length_mla", (uint64_t) g.head_dim_k),
            u32(P + "attention.value_length_mla", (uint64_t) g.head_dim_v),
            u32(P + "attention.indexer.head_count", (uint64_t) g.idx_heads),
            u32(P + "attention.indexer.key_length", (uint64_t) g.idx_key_dim),
            u32(P + "attention.indexer.top_k", (uint64_t) g.idx_top_k),
            u32(P + "attention.indexer.kpool", (uint64_t) g.idx_kpool),
            boolean(P + "attention.indexer.kpool_select_tail", g.idx_kpool_select_tail),
            bools(P + "attention.indexer.types", types),
            u32(P + "hyper_connection.count", (uint64_t) g.hc),
            u32(P + "hyper_connection.sinkhorn_iterations", (uint64_t) g.hc_sinkhorn_iters),
            f32(P + "hyper_connection.epsilon", (float) g.hc_eps),
            u32(P + "leading_dense_block_count", (uint64_t) g.n_dense_lead),
            u32(P + "expert_feed_forward_length", (uint64_t) g.n_ff_exp),
            u32(P + "expert_shared_feed_forward_length", (uint64_t) g.n_ff_shexp),
            u32(P + "expert_shared_count", (uint64_t) g.n_expert_shared),
            f32(P + "expert_weights_scale", (float) g.expert_weights_scale),
            boolean(P + "expert_weights_norm", g.expert_weights_norm),
            f32s(P + "swiglu_clamp_exp", vector<float>(B, (float) g.swiglu_clamp_exp)),
            f32s(P + "swiglu_clamp_shexp", vector<float>(B, (float) g.swiglu_clamp_shexp)),
            u32(P + "expert_count", (uint64_t) g.n_expert),
            u32(P + "expert_used_count", (uint64_t) g.n_expert_used),
            u32(P + "expert_gating_func", (uint64_t) g.expert_gating)};
}

/// A directory under the system's temporary one, removed with its files at the end.
struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-glm-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

/// The model's file at `path`: model_tensors and model_keys.
inline void write_model_gguf(const fs::path& path, const TestModel& tm) {
    fixture::write(path, model_keys(tm.g), model_tensors(tm));
}

}  // namespace
