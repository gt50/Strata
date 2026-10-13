// src/glm/glm_loader_parity.cpp - the GGUF loader (glm_loader.hpp) on glm_model_parity's synthetic model written as a
// glm5-next GGUF: the tensors under llama.cpp's names, in its converter's shapes and layouts (the experts stacked
// three tensors a layer, attn_k_b / attn_v_b split per head, the conv1d weights {d_conv, 1, d_inner, 1}), with the
// metadata read_geometry reads.
//
//   1. the geometry read back is the one the model was built with
//   2. loaded with the experts in device memory, and again in mapped host memory, GlmModel gives bitwise the logits,
//      h_nextn and MTP logits of the model built directly from the same weights, for every token
//   3. the conversions, on a second file: an F16 and a BF16 vector to F32, a Q8_0 attn_k_b and an F32 attn_v_b to
//      BF16, a conv1d of three dimensions - each loaded value the one expected, bitwise
//   4. what glm_plan refuses, naming the tensor: one missing, a shape the geometry does not give, a type no kernel
//      takes, experts whose gate and up types differ, a missing MTP block (and the same file loads without it)
//
// `glm_loader_parity --plan SHARD`: the plan of a real model (any shard of it) - the types per role and the bytes.
#include "glm_ref.hpp"

#include "../../tests/core/gguf_fixture.hpp"
#include "strata/glm/glm_loader.hpp"

#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

constexpr int64_t MAX_CTX = 64;
constexpr int STEPS = 6;
int g_bad = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "*** FAIL ***");
    if (!ok) ++g_bad;
}

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

fixture::Tensor* find_t(std::vector<fixture::Tensor>& ts, const std::string& name) {
    for (auto& t : ts)
        if (t.name == name) return &t;
    std::fprintf(stderr, "no tensor %s\n", name.c_str());
    std::exit(1);
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-glm-loader-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

/// Writes `ts` and runs glm_plan on it: "" or its error.
std::string plan_of(const fs::path& path, const Glm5NextGeometry& g, const std::vector<fixture::Tensor>& ts,
                    const glm::GlmLoadOptions& opt, glm::GlmPlan* out = nullptr) {
    fixture::write(path, model_keys(g), ts);
    const strata::GgufModel m({path.string()});
    glm::GlmPlan plan;
    const std::string e = glm::glm_plan(m, g, opt, plan);
    if (out) *out = plan;
    return e;
}

void print_plan(const glm::GlmPlan& p) {
    std::printf("  %d tensors, %.1f MB read; %.1f MB of weights and %.1f MB of experts as loaded%s\n", p.tensors,
                (double) p.file_bytes / 1e6, (double) p.device_bytes / 1e6, (double) p.expert_bytes / 1e6,
                p.mtp ? ", the MTP block included" : "");
    for (const auto& [what, n] : p.types) std::printf("    %-40s %d\n", what.c_str(), n);
}

int plan_real(const char* shard) {
    try {
        const strata::GgufModel m = strata::GgufModel::open(shard);
        Glm5NextGeometry g;
        if (const std::string e = strata::read_geometry(m.meta(), g); !e.empty()) {
            std::fprintf(stderr, "geometry: %s\n", e.c_str());
            return 1;
        }
        glm::GlmPlan plan;
        const std::string e = glm::glm_plan(m, g, glm::GlmLoadOptions{}, plan);
        std::printf("%s: %zu shards, %d layers + %d MTP\n", shard, m.size(), (int) g.n_layers, (int) g.n_nextn);
        print_plan(plan);
        if (!e.empty()) {
            std::printf("refused: %s\n", e.c_str());
            return 1;
        }
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}

/// Runs `a` and `b` over the same tokens, comparing logits, h_nextn and the MTP logits bitwise.
void same_outputs(glm::GlmModel& a, glm::GlmModel& b, const Glm5NextGeometry& g, const char* what) {
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    a.reset(s);
    b.reset(s);
    std::string err;
    bool same = true;
    const int32_t tokens[STEPS + 1] = {1513, 265, 1973, 1060, 658, 670, 1695};
    for (int t = 0; t < STEPS; ++t) {
        for (glm::GlmModel* m : {&a, &b}) {
            if (!m->decode(tokens[t], t, s, err) || !m->mtp_decode(tokens[t + 1], t, m->h_nextn(), s, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                std::exit(1);
            }
        }
        check(cudaStreamSynchronize(s), "decode");
        const size_t V = (size_t) g.n_vocab, D = (size_t) g.n_embd;
        same &= raw(a.logits(), V * 4) == raw(b.logits(), V * 4);
        same &= raw(a.h_nextn(), D * 4) == raw(b.h_nextn(), D * 4);
        same &= raw(a.mtp_logits(), V * 4) == raw(b.mtp_logits(), V * 4);
    }
    cudaStreamDestroy(s);
    expect(same, std::string(what) + ": logits, h_nextn and MTP logits bitwise the direct model's, " +
                     std::to_string(STEPS) + " tokens");
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--plan" && i + 1 < argc) return plan_real(argv[++i]);
        else { std::fprintf(stderr, "usage: glm_loader_parity [--selftest] | --plan SHARD\n"); return 2; }
    }
    TestModel tm = build_test_model(MAX_CTX);
    const Glm5NextGeometry& g0 = tm.g;
    TempDir dir;
    const fs::path path = dir.path / "glm-test.gguf";
    std::vector<fixture::Tensor> ts = model_tensors(tm);
    fixture::write(path, model_keys(g0), ts);
    std::printf("glm_loader: the synthetic model (5 layers + MTP) as a glm5-next GGUF of %.1f MB, %zu tensors\n",
                (double) fs::file_size(path) / 1e6, ts.size());

    // 1. the geometry
    Glm5NextGeometry g;
    {
        const strata::GgufModel m({path.string()});
        const std::string e = strata::read_geometry(m.meta(), g);
        expect(e.empty() && g.n_layers == g0.n_layers && g.n_nextn == g0.n_nextn && g.is_mla == g0.is_mla &&
                   g.idx_full == g0.idx_full && g.n_embd == g0.n_embd && g.n_vocab == g0.n_vocab &&
                   g.n_expert == g0.n_expert && g.n_ff_exp == g0.n_ff_exp && g.n_dense_lead == g0.n_dense_lead &&
                   (float) g.rms_eps == (float) g0.rms_eps && g.n_ff_shexp == g0.n_ff_shexp,
               "the geometry read back is the model's" + (e.empty() ? std::string() : ": " + e));
    }

    // 2. loaded, both expert homes, against the model built directly
    glm::GlmModel direct;
    std::string err;
    if (!direct.init(g0, tm.w, MAX_CTX, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    for (glm::GlmExpertHome home : {glm::GlmExpertHome::Device, glm::GlmExpertHome::MappedHost}) {
        const bool dev = home == glm::GlmExpertHome::Device;
        const strata::GgufModel m({path.string()});
        glm::GlmLoadedWeights lw;
        glm::GlmLoadOptions opt;
        opt.experts = home;
        if (!lw.load(m, g, opt, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        if (dev) print_plan(lw.plan());
        glm::GlmModel loaded;
        if (!loaded.init(g, lw.weights(), MAX_CTX, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        same_outputs(direct, loaded, g, dev ? "experts in device memory" : "experts in mapped host memory");
    }

    // 3. the conversions
    {
        std::vector<fixture::Tensor> cv = ts;
        const Block& b0 = tm.blocks[0];
        const Block& b1 = tm.blocks[1];
        const Block& b3 = tm.blocks[3];
        // blk.0.ssm_norm F16, blk.1.attn_norm BF16
        vector<uint16_t> f16v(b0.o_norm.size()), bf16v(b1.attn_norm.size());
        for (size_t i = 0; i < f16v.size(); ++i) f16v[i] = k::f16_from_f32(b0.o_norm[i]);
        for (size_t i = 0; i < bf16v.size(); ++i) bf16v[i] = bf16_bits(b1.attn_norm[i]);
        auto bytes16 = [](const vector<uint16_t>& v) {
            vector<uint8_t> r(v.size() * 2);
            std::memcpy(r.data(), v.data(), r.size());
            return r;
        };
        *find_t(cv, "blk.0.ssm_norm.weight") = tensor("blk.0.ssm_norm.weight", {128}, 1, bytes16(f16v));
        *find_t(cv, "blk.1.attn_norm.weight") = tensor("blk.1.attn_norm.weight", {1024}, 30, bytes16(bf16v));
        // blk.1.attn_k_b Q8_0: 32 values a block, an FP16 scale amax / 127, the values rounded
        const size_t nk = b1.k_b.size();
        vector<uint8_t> q8(nk / 32 * 34);
        vector<uint16_t> kb_want(nk);
        for (size_t blk = 0; blk < nk / 32; ++blk) {
            float amax = 0;
            for (size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(b1.k_b[blk * 32 + i]));
            const uint16_t dh = k::f16_from_f32(amax / 127.0f);
            const float d = k::f32_from_f16(dh), id = d != 0 ? 1.0f / d : 0.0f;
            std::memcpy(&q8[blk * 34], &dh, 2);
            for (size_t i = 0; i < 32; ++i) {
                const int8_t q = (int8_t) std::lround(b1.k_b[blk * 32 + i] * id);
                q8[blk * 34 + 2 + i] = (uint8_t) q;
                kb_want[blk * 32 + i] = bf16_bits((float) q * d);
            }
        }
        *find_t(cv, "blk.1.attn_k_b.weight") = tensor("blk.1.attn_k_b.weight", {256, 512, 8}, 8, q8);
        // blk.3.attn_v_b F32 (BF16 values, so the rounding gives them back), blk.0.ssm_conv1d_q {4, 1, d_inner}
        *find_t(cv, "blk.3.attn_v_b.weight") = f32_t("blk.3.attn_v_b.weight", {512, 256, 8}, b3.v_b);
        *find_t(cv, "blk.0.ssm_conv1d_q.weight") = f32_t("blk.0.ssm_conv1d_q.weight", {4, 1, 1024}, b0.conv_q);
        const fs::path cpath = dir.path / "glm-conv.gguf";
        fixture::write(cpath, model_keys(g0), cv);
        const strata::GgufModel m({cpath.string()});
        glm::GlmLoadedWeights lw;
        if (!lw.load(m, g, glm::GlmLoadOptions{}, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const glm::GlmModelWeights& w = lw.weights();
        bool ok = true;
        const vector<float> on = download(w.blocks[0].o_norm, f16v.size());
        for (size_t i = 0; i < f16v.size(); ++i) ok &= on[i] == k::f32_from_f16(f16v[i]);
        expect(ok, "an F16 vector (blk.0.ssm_norm) loaded as its F32 values");
        ok = true;
        const vector<float> an = download(w.blocks[1].attn_norm, bf16v.size());
        for (size_t i = 0; i < bf16v.size(); ++i) ok &= an[i] == bf16_val(bf16v[i]);
        expect(ok, "a BF16 vector (blk.1.attn_norm) loaded as its F32 values");
        expect(download(w.blocks[1].k_b, nk) == kb_want,
               "a Q8_0 attn_k_b loaded as the BF16 of its dequantized values");
        vector<uint16_t> vb_want(b3.v_b.size());
        for (size_t i = 0; i < vb_want.size(); ++i) vb_want[i] = bf16_bits(b3.v_b[i]);
        expect(download(w.blocks[3].v_b, vb_want.size()) == vb_want, "an F32 attn_v_b loaded as its BF16 values");
        expect(download(w.blocks[0].conv_q, b0.conv_q.size()) == b0.conv_q,
               "a conv1d of three dimensions {4, 1, d_inner} loaded as [d_inner][4]");
    }

    // 4. what the plan refuses
    {
        const fs::path epath = dir.path / "glm-bad.gguf";
        auto refused = [&](std::vector<fixture::Tensor> bad, const std::string& want, const char* what,
                           const glm::GlmLoadOptions& opt = {}) {
            const std::string e = plan_of(epath, g0, bad, opt);
            expect(e.find(want) != std::string::npos, std::string(what) + ": \"" + e + "\"");
        };
        std::vector<fixture::Tensor> bad = ts;
        bad.erase(bad.begin() + (find_t(bad, "blk.3.exp_probs_b.bias") - bad.data()));
        refused(bad, "blk.3.exp_probs_b.bias: missing", "a tensor missing");
        bad = ts;
        *find_t(bad, "blk.1.attn_kv_a_mqa.weight") = tensor("blk.1.attn_kv_a_mqa.weight", {1024, 256}, 23, {});
        refused(bad, "blk.1.attn_kv_a_mqa.weight: shape {1024, 256}", "a shape the geometry does not give");
        bad = ts;
        *find_t(bad, "blk.0.ssm_norm.weight") = tensor("blk.0.ssm_norm.weight", {128}, 8, {});
        refused(bad, "blk.0.ssm_norm.weight: Q8_0 is not a type", "a type no kernel takes");
        bad = ts;
        *find_t(bad, "blk.2.ffn_up_exps.weight") = tensor("blk.2.ffn_up_exps.weight", {1024, 256, 16}, 8, {});
        refused(bad, "blk.2.ffn_gate_exps.weight: IQ4_XS and its up_exps Q8_0", "experts' gate and up of two types");
        bad = ts;
        for (const char* n : {"nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
                              "nextn.shared_head_norm.weight"})
            bad.erase(bad.begin() + (find_t(bad, std::string("blk.5.") + n) - bad.data()));
        refused(bad, "blk.5.nextn.eh_proj.weight: missing", "the MTP block missing");
        glm::GlmLoadOptions no_mtp;
        no_mtp.mtp = false;
        glm::GlmPlan p;
        const std::string e = plan_of(epath, g0, bad, no_mtp, &p);
        expect(e.empty() && !p.mtp, "the same file planned without the MTP block" + (e.empty() ? "" : ": " + e));
    }

    std::printf("\nglm_loader: %d failures\n", g_bad);
    for (void* p : g_free) cudaFree(p);
    if (g_bad) return 1;
    if (selftest) std::printf("glm_loader_parity OK\n");
    return 0;
}
