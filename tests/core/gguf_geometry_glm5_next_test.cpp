// tests/core/gguf_geometry_glm5_next_test.cpp - GLM-5.3-Flash's geometry from its GGUF metadata
// (strata::read_geometry into core::Glm5NextGeometry), on synthetic GGUFs: no model, no GPU.
//
// The full key set is what llama.cpp's converter (conversion/glm.py, Glm5NextModel) writes for zai-org/GLM-5.3-Flash's
// config.json, with the value types it writes them in.  Then: the keys llama.cpp's loader requires, the defaults of
// the optional ones, the keys that must agree, and that the engine itself still refuses the architecture.
//
// `gguf_geometry_glm5_next_test --real SHARD1`: the geometry a real glm5-next metadata shard declares.
#include "gguf_fixture.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::Glm5NextGeometry;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("strata-glm5-next-geometry-test-" +
                                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

const std::string P = "glm5-next.";
constexpr int64_t LAYERS = 45, BLOCKS = 46;   // 45 main layers and one MTP layer

/// head_count_kv as the converter writes it: 0 on a linear_attention layer, 1 on the sparse ones (3, 7, ... 43), and
/// the MTP blocks padded with 1.
std::vector<uint64_t> kv_heads(int64_t layers = LAYERS, int64_t blocks = BLOCKS) {
    std::vector<uint64_t> r((size_t) blocks, 1);
    for (int64_t l = 0; l < layers; ++l) r[(size_t) l] = l % 4 == 3 ? 1 : 0;
    return r;
}

std::vector<fixture::Kv> glm_keys() {
    using namespace fixture;
    return {str("general.architecture", "glm5-next"),
            u32(P + "block_count", BLOCKS),
            u32(P + "context_length", 1048576),
            u32(P + "embedding_length", 4096),
            u32(P + "feed_forward_length", 12288),
            u32(P + "attention.head_count", 64),
            u32s(P + "attention.head_count_kv", kv_heads()),
            u32(P + "vocab_size", 154880),
            f32(P + "attention.layer_norm_rms_epsilon", 1e-5f),
            f32(P + "attention.layer_norm_epsilon", 1e-6f),
            u32(P + "nextn_predict_layers", 1),
            u32(P + "ssm.conv_kernel", 4),
            u32(P + "kda.head_dim", 128),
            f32(P + "kda.gate_lower_bound", -5.0f),
            u32(P + "attention.q_lora_rank", 1536),
            u32(P + "attention.kv_lora_rank", 512),
            u32(P + "rope.dimension_count", 0),
            u32(P + "attention.key_length", 512),
            u32(P + "attention.value_length", 512),
            u32(P + "attention.key_length_mla", 256),
            u32(P + "attention.value_length_mla", 256),
            u32(P + "attention.indexer.head_count", 32),
            u32(P + "attention.indexer.key_length", 128),
            u32(P + "attention.indexer.top_k", 2048),
            u32(P + "attention.indexer.kpool", 4),
            boolean(P + "attention.indexer.kpool_select_tail", true),
            bools(P + "attention.indexer.types", std::vector<uint64_t>((size_t) LAYERS, 1)),
            u32(P + "hyper_connection.count", 4),
            u32(P + "hyper_connection.sinkhorn_iterations", 20),
            f32(P + "hyper_connection.epsilon", 1e-6f),
            u32(P + "leading_dense_block_count", 3),
            u32(P + "expert_feed_forward_length", 2048),
            u32(P + "expert_shared_feed_forward_length", 2048),   // Unsloth's file has it; the converter does not write it
            u32(P + "expert_shared_count", 1),
            f32(P + "expert_weights_scale", 2.5f),
            boolean(P + "expert_weights_norm", true),
            f32s(P + "swiglu_clamp_exp", std::vector<float>((size_t) BLOCKS, 10.0f)),
            f32s(P + "swiglu_clamp_shexp", std::vector<float>((size_t) BLOCKS, 10.0f)),
            u32(P + "expert_count", 288),
            u32(P + "expert_used_count", 8),
            u32(P + "expert_group_count", 1),
            u32(P + "expert_group_used_count", 1),
            u32(P + "expert_gating_func", 2)};
}

/// `kv` with `key` replaced by `v` (or removed, when `v` is null).
std::vector<fixture::Kv> with(std::vector<fixture::Kv> kv, const std::string& key, const fixture::Kv* v) {
    for (auto it = kv.begin(); it != kv.end(); ++it)
        if (it->key == key) {
            if (v) *it = *v;
            else kv.erase(it);
            return kv;
        }
    if (v) kv.push_back(*v);
    return kv;
}
std::vector<fixture::Kv> with(std::vector<fixture::Kv> kv, const fixture::Kv& v) { return with(std::move(kv), v.key, &v); }
std::vector<fixture::Kv> without(std::vector<fixture::Kv> kv, const std::string& key) { return with(std::move(kv), key, nullptr); }

struct Result {
    std::string err, guard, qwen_err;
    Glm5NextGeometry g;
};
Result run(const fs::path& dir, const std::vector<fixture::Kv>& kv) {
    static int n = 0;
    const fs::path p = dir / ("m" + std::to_string(n++) + ".gguf");
    fixture::write(p, kv, {});
    Result r;
    const strata::GgufFile f(p.string());
    r.err = strata::read_geometry(f, r.g);
    r.guard = strata::check_architecture(f);
    strata::core::ModelGeometry q;
    r.qwen_err = strata::read_geometry(f, q);
    return r;
}
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

std::string layer_map(const Glm5NextGeometry& g) {
    std::string m;
    for (int64_t b = 0; b < g.n_blocks(); ++b) {
        if (b == g.n_layers) m += '|';
        m += g.is_mla[(size_t) b] ? 'M' : 'k';
    }
    return m;
}

int real_model(const std::string& shard) {
    try {
        const strata::GgufFile f(shard);
        Glm5NextGeometry g;
        const std::string err = strata::read_geometry(f, g);
        if (!err.empty()) {
            std::printf("  refused: %s\n", err.c_str());
            return 1;
        }
        std::printf("  %lld layers + %lld MTP, n_embd %lld, vocab %lld, context %lld, rms_eps %g\n", (long long) g.n_layers,
                    (long long) g.n_nextn, (long long) g.n_embd, (long long) g.n_vocab, (long long) g.n_ctx_train, g.rms_eps);
        std::printf("  layers (M = MLA + indexer, k = KDA, | = MTP): %s  -> %lld MLA, %lld KDA\n", layer_map(g).c_str(),
                    (long long) g.n_mla_layers(), (long long) g.n_kda_layers());
        std::printf("  KDA: %lld heads x %lld, conv %lld, gate lower bound %s%g\n", (long long) g.n_head,
                    (long long) g.kda_head_dim, (long long) g.ssm_d_conv, g.has_kda_gate_lower_bound ? "" : "(none) ",
                    g.kda_gate_lower_bound);
        std::printf("  MLA: q_lora %lld, kv_lora %lld, rope %lld, head k %lld / v %lld\n", (long long) g.q_lora_rank,
                    (long long) g.kv_lora_rank, (long long) g.n_rot, (long long) g.head_dim_k, (long long) g.head_dim_v);
        int64_t full = 0;
        for (uint8_t x : g.idx_full) full += x;
        std::printf("  indexer: %lld heads x %lld, top-k %lld, kpool %lld (select tail %s), %lld of %lld layers full\n",
                    (long long) g.idx_heads, (long long) g.idx_key_dim, (long long) g.idx_top_k, (long long) g.idx_kpool,
                    g.idx_kpool_select_tail ? "yes" : "no", (long long) full, (long long) g.n_layers);
        std::printf("  mHC: %lld streams, %lld Sinkhorn iterations, eps %g\n", (long long) g.hc,
                    (long long) g.hc_sinkhorn_iters, g.hc_eps);
        std::printf("  MoE: %lld dense layers (ff %lld), then %lld experts x %lld (%lld per token, %lld shared of %lld), "
                    "gating %s, scale %g, norm %s, clamp %g / %g\n",
                    (long long) g.n_dense_lead, (long long) g.n_ff_dense, (long long) g.n_expert, (long long) g.n_ff_exp,
                    (long long) g.n_expert_used, (long long) g.n_expert_shared, (long long) g.n_ff_shexp,
                    g.expert_gating == 2 ? "sigmoid" : g.expert_gating == 1 ? "softmax" : "sqrtsoftplus",
                    g.expert_weights_scale, g.expert_weights_norm ? "yes" : "no", g.swiglu_clamp_exp, g.swiglu_clamp_shexp);
        std::printf("  the engine: %s\n", strata::check_architecture(f).c_str());
        return 0;
    } catch (const std::exception& e) {
        std::printf("  refused: %s\n", e.what());
        return 1;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--real") return real_model(argv[2]);
    std::printf("gguf_geometry_glm5_next_test\n");
    TempDir d;
    using namespace fixture;
    {
        const Result r = run(d.path, glm_keys());
        const Glm5NextGeometry& g = r.g;
        check(r.err.empty(), "GLM-5.3-Flash's key set reads" + (r.err.empty() ? "" : ": " + r.err));
        check(g.n_layers == 45 && g.n_nextn == 1 && g.n_blocks() == 46, "  45 main layers and 1 MTP layer out of 46 blocks");
        check(g.n_mla_layers() == 11 && g.n_kda_layers() == 34, "  11 MLA layers and 34 KDA layers");
        bool pattern = true;
        for (int64_t l = 0; l < g.n_layers; ++l) pattern = pattern && (g.is_mla[(size_t) l] != 0) == (l % 4 == 3);
        check(pattern && g.is_mla[45] == 1, "  MLA on layers 3, 7, ... 43, and on the MTP block");
        check(g.n_embd == 4096 && g.n_vocab == 154880 && g.n_ctx_train == 1048576 && g.n_head == 64,
              "  n_embd 4096, vocab 154880, context 1048576, 64 heads");
        check(g.rms_eps > 0.99e-5 && g.rms_eps < 1.01e-5, "  the RMS epsilon (1e-5), not the plain layer-norm one (1e-6)");
        check(g.kda_head_dim == 128 && g.ssm_d_conv == 4 && g.has_kda_gate_lower_bound && g.kda_gate_lower_bound == -5.0,
              "  KDA: head 128, conv 4, gate lower bound -5");
        check(g.q_lora_rank == 1536 && g.kv_lora_rank == 512 && g.n_rot == 0 && g.head_dim_k == 256 && g.head_dim_v == 256,
              "  MLA: q_lora 1536, kv_lora 512, no RoPE, head 256 / 256");
        check(g.idx_heads == 32 && g.idx_key_dim == 128 && g.idx_top_k == 2048 && g.idx_kpool == 4 &&
                  g.idx_kpool_select_tail && g.idx_full == std::vector<uint8_t>(45, 1),
              "  indexer: 32 x 128, top-k 2048, kpool 4, tail selected, every layer full");
        check(g.hc == 4 && g.hc_sinkhorn_iters == 20 && g.hc_eps > 0.99e-6 && g.hc_eps < 1.01e-6,
              "  mHC: 4 streams, 20 Sinkhorn iterations, eps 1e-6");
        check(g.n_dense_lead == 3 && g.n_ff_dense == 12288 && g.n_expert == 288 && g.n_expert_used == 8 &&
                  g.n_expert_shared == 1 && g.n_ff_exp == 2048 && g.n_ff_shexp == 2048,
              "  MoE: 3 dense layers (12288), 288 experts x 2048, 8 per token, 1 shared of 2048");
        check(g.expert_gating == 2 && g.expert_weights_scale == 2.5 && g.expert_weights_norm &&
                  g.swiglu_clamp_exp == 10.0 && g.swiglu_clamp_shexp == 10.0,
              "  sigmoid routing, x2.5, normalised, SwiGLU clamped at 10");
        check(r.guard == "architecture is 'glm5-next', this engine requires 'qwen4exp'" && r.qwen_err == r.guard,
              "  and the engine still refuses it by name (no forward pass yet)");
    }
    {
        // the keys llama.cpp's loader requires (get_key without `false`)
        for (const char* key : {"block_count", "embedding_length", "attention.head_count", "attention.head_count_kv",
                                "attention.layer_norm_rms_epsilon", "kda.head_dim", "ssm.conv_kernel",
                                "attention.q_lora_rank", "attention.kv_lora_rank", "attention.key_length_mla",
                                "attention.value_length_mla", "attention.indexer.head_count",
                                "attention.indexer.key_length", "attention.indexer.top_k", "attention.indexer.kpool",
                                "hyper_connection.count", "hyper_connection.sinkhorn_iterations",
                                "hyper_connection.epsilon", "expert_count", "expert_used_count", "expert_shared_count",
                                "expert_feed_forward_length", "feed_forward_length"}) {
            const Result r = run(d.path, without(glm_keys(), P + key));
            check(r.err == "missing " + P + key, std::string("without ") + key + ": refused, naming it");
        }
    }
    {
        // the optional ones, absent: llama.cpp's defaults
        std::vector<std::string> tokens(1000, "t");
        auto kv = without(without(without(without(without(without(glm_keys(), P + "vocab_size"), P + "kda.gate_lower_bound"),
                                                  P + "attention.indexer.types"), P + "expert_gating_func"),
                                  P + "swiglu_clamp_shexp"), P + "attention.indexer.kpool_select_tail");
        kv.push_back(strs("tokenizer.ggml.tokens", tokens));
        const Result r = run(d.path, kv);
        check(r.err.empty() && r.g.n_vocab == 1000, "no vocab_size: the vocabulary is tokenizer.ggml.tokens' length");
        check(!r.g.has_kda_gate_lower_bound, "no kda.gate_lower_bound: none");
        check(r.g.idx_full == std::vector<uint8_t>(45, 1) && r.g.idx_kpool_select_tail,
              "no indexer types or kpool_select_tail: every layer full, the tail selected");
        check(r.g.expert_gating == 2, "no expert_gating_func: sigmoid, as llama.cpp's loader defaults");
        check(r.g.swiglu_clamp_shexp == 10.0, "no swiglu_clamp_shexp: the routed experts' clamp");
        const Result s = run(d.path, without(without(glm_keys(), P + "swiglu_clamp_exp"), P + "swiglu_clamp_shexp"));
        check(s.err.empty() && s.g.swiglu_clamp_exp == 0 && s.g.swiglu_clamp_shexp == 0, "no clamps at all: 0, unclamped");
        const Result dense0 = run(d.path, without(with(glm_keys(), u32(P + "leading_dense_block_count", 0)),
                                                  P + "feed_forward_length"));
        check(dense0.err.empty() && dense0.g.n_ff_dense == 0, "no dense layers: feed_forward_length is not needed");
        const Result sh = run(d.path, with(without(glm_keys(), P + "expert_shared_feed_forward_length"),
                                           u32(P + "expert_shared_count", 2)));
        check(sh.err.empty() && sh.g.n_ff_shexp == 4096,
              "no expert_shared_feed_forward_length: 2 shared experts of 2048 make a 4096-wide one");
        const Result scalar = run(d.path, with(glm_keys(), u32(P + "attention.head_count_kv", 1)));
        check(scalar.err.empty() && scalar.g.n_kda_layers() == 0, "a scalar head_count_kv of 1: every layer MLA");
    }
    {
        // what must agree, and what must add up
        auto e = [&](const fixture::Kv& kv, const char* part, const std::string& what) {
            const Result r = run(d.path, with(glm_keys(), kv));
            check(has(r.err, part), what + ": " + r.err);
        };
        e(u32(P + "attention.key_length", 576), "key_length = 576", "a key_length other than kv_lora_rank + rope");
        e(u32(P + "attention.value_length", 256), "value_length = 256", "a value_length other than kv_lora_rank");
        e(u32(P + "rope.dimension_count", 512), "wider than key_length_mla", "a rope slice wider than the MLA head");
        e(u32(P + "attention.indexer.top_k", 2050), "whole number of pools", "a top_k that is not a whole number of pools");
        e(u32(P + "attention.indexer.kpool", 1), "at least 2 cells", "a pool of one cell");
        e(u32s(P + "attention.head_count_kv", kv_heads(45, 45)), "each of the 46 blocks", "head_count_kv for 45 of 46 blocks");
        e(bools(P + "attention.indexer.types", std::vector<uint64_t>(46, 1)), "each of the 45 main layers",
          "indexer types for 46 of 45 layers");
        std::vector<float> clamps(46, 10.0f);
        clamps[20] = 7.0f;
        e(f32s(P + "swiglu_clamp_exp", clamps), "differs between layers", "a clamp that changes at layer 20");
        e(u32(P + "leading_dense_block_count", 45), "leaves no MoE layer", "45 dense layers of 45");
        e(u32(P + "nextn_predict_layers", 46), "leaves no main layer", "46 MTP layers of 46 blocks");
        e(u32(P + "expert_used_count", 300), "more than expert_count", "300 experts per token of 288");
        e(u32(P + "expert_gating_func", 3), "not 1 (softmax), 2 (sigmoid) or 4", "an unknown gating function");
        e(u32(P + "hyper_connection.count", 1), "nothing to mix", "one mHC stream");
        e(f32(P + "block_count", 46.0f), "is not an integer", "a float block count");
    }
    {
        // the readers do not mistake one architecture for the other
        const std::vector<fixture::Kv> qwen = {
            str("general.architecture", "qwen4exp"), u32("qwen4exp.block_count", 48), u32("qwen4exp.embedding_length", 2560),
            u32("qwen4exp.expert_count", 512), u32("qwen4exp.expert_used_count", 10),
            u32("qwen4exp.attention.head_count", 24), u32("qwen4exp.attention.head_count_kv", 2)};
        const Result r = run(d.path, qwen);
        check(r.err == "architecture is 'qwen4exp', not 'glm5-next'" && r.guard.empty(),
              "a qwen4exp file: the GLM reader names it, the engine's guard takes it");
        const Result unhyphenated = run(d.path, with(glm_keys(), str("general.architecture", "glm5next")));
        check(unhyphenated.err == "architecture is 'glm5next', not 'glm5-next'",
              "'glm5next' without the hyphen is another architecture (llama.cpp's name has one)");
    }
    std::printf(g_fail ? "gguf_geometry_glm5_next_test: %d FAILED\n" : "gguf_geometry_glm5_next_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
