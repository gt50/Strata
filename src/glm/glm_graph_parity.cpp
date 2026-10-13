// src/glm/glm_graph_parity.cpp - GlmModel's captured decode step (set_decode_graph) against the same model decoding as
// it always did, on glm_model_parity's synthetic model: the logits and h_nextn bitwise the same at every token.
//
//   1. 10 tokens from a new sequence: the first decode runs as before, the second is captured, the rest replay
//   2. a 16-token prompt (the prompt path, not captured), then 6 decodes replaying the graph from there
//   3. reset, and another sequence on the same graph
//
// `glm_graph_parity --bench [layers]`: the time of a decode token, as before and replayed, on a model of GLM-5.3-Flash's
// shapes (n_embd 4096, 64 heads, the KDA / MLA pattern, mHC, the dense leading layers, the shared expert, a vocabulary
// of 154880) with random weights - but 8 experts a layer (all used: a token reads what 8 resident experts cost).
#include "glm_ref.hpp"

#include "strata/glm/glm_model.hpp"

#include <chrono>

namespace {

constexpr int64_t MAX_CTX = 64;
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

// ---------------------------------------------------------------- --bench: a model of GLM-5.3-Flash's shapes
uint64_t g_seed = 0x9e3779b97f4a7c15ull;
uint64_t next_rand() {   // xorshift64*
    g_seed ^= g_seed >> 12;
    g_seed ^= g_seed << 25;
    g_seed ^= g_seed >> 27;
    return g_seed * 0x2545f4914f6cdd1dull;
}
/// A device tensor of random bytes; the FP16 scale of every block set (Q8_0: 34 bytes, IQ4_XS: 136) so the values
/// are of the size a weight is.
uint64_t g_bytes = 0;   // the weights made, for the bytes a token reads
void* rand_tensor(int type, int64_t n_in, int64_t n_out) {
    const size_t row = type == 0 ? (size_t) n_in * 4 : type == 30 ? (size_t) n_in * 2 : k::iq_row_bytes(type, n_in);
    vector<uint8_t> b(row * (size_t) n_out);
    g_bytes += b.size();
    for (size_t i = 0; i + 8 <= b.size(); i += 8) {
        const uint64_t r = next_rand();
        std::memcpy(&b[i], &r, 8);
    }
    const float target = 1.0f / std::sqrt((float) n_in);
    if (type == 0 || type == 30) {
        for (size_t i = 0; i < b.size() / (type == 0 ? 4 : 2); ++i) {
            const float v = target * ((float) (next_rand() >> 40) / (float) (1u << 24) * 2.0f - 1.0f) * 1.7f;
            if (type == 0) std::memcpy(&b[i * 4], &v, 4);
            else {
                const uint16_t h = bf16_bits(v);
                std::memcpy(&b[i * 2], &h, 2);
            }
        }
    } else {
        const size_t bb = type == 8 ? 34 : 136;
        const float per = type == 8 ? 74.0f : 1221.0f;
        const uint16_t d = k::f16_from_f32(target / per);
        for (size_t o = 0; o < b.size(); o += bb) std::memcpy(&b[o], &d, 2);
    }
    void* p = nullptr;
    check(cudaMalloc(&p, b.size() + 256), "tensor");
    check(cudaMemcpy(p, b.data(), b.size(), cudaMemcpyHostToDevice), "tensor");
    g_free.push_back(p);
    return p;
}
k::GlmLinear rlin(int type, int64_t n_in, int64_t n_out) { return {type, rand_tensor(type, n_in, n_out), n_in, n_out}; }
const float* rvec(int64_t n, float mean, float s) {
    vector<float> v((size_t) n);
    for (auto& x : v) x = mean + s * ((float) (next_rand() >> 40) / (float) (1u << 24) * 2.0f - 1.0f);
    return up_f(v);
}

int bench(int64_t layers) {
    Glm5NextGeometry g;
    g.n_layers = layers; g.n_nextn = 0; g.n_embd = 4096; g.n_vocab = 154880; g.n_ctx_train = 4096;
    g.rms_eps = 1e-5; g.norm_eps = 1e-6;
    for (int64_t l = 0; l < layers; ++l) g.is_mla.push_back(l % 4 == 3 ? 1 : 0);
    g.n_head = 64; g.kda_head_dim = 128; g.ssm_d_conv = 4; g.has_kda_gate_lower_bound = true; g.kda_gate_lower_bound = -5;
    g.q_lora_rank = 1536; g.kv_lora_rank = 512; g.n_rot = 0; g.head_dim_k = 256; g.head_dim_v = 256;
    g.idx_heads = 32; g.idx_key_dim = 128; g.idx_top_k = 2048; g.idx_kpool = 4; g.idx_kpool_select_tail = true;
    g.hc = 4; g.hc_sinkhorn_iters = 20; g.hc_eps = 1e-6;
    g.n_dense_lead = std::min<int64_t>(3, layers - 1); g.n_ff_dense = 12288; g.n_expert = 8; g.n_expert_used = 8;
    g.n_expert_shared = 1; g.n_ff_exp = 2048; g.n_ff_shexp = 2048; g.expert_gating = 2; g.expert_weights_scale = 2.5;
    g.expert_weights_norm = true; g.swiglu_clamp_exp = 10; g.swiglu_clamp_shexp = 10;
    const int64_t D = 4096, H = 64, Dh = 128, di = H * Dh;
    std::printf("decode on a model of GLM-5.3-Flash's shapes, %lld layers, 8 resident IQ4_XS experts a layer (building)\n",
                (long long) layers);
    glm::GlmModelWeights w;
    w.tok_embd = {8, rand_tensor(8, D, g.n_vocab), k::iq_row_bytes(8, D)};
    g_bytes = 0;   // a token reads one row of the embedding: not counted
    w.output_norm = rvec(D, 1.0f, 0.1f);
    w.output = rlin(8, D, g.n_vocab);
    w.blocks.resize((size_t) layers);
    for (int64_t il = 0; il < layers; ++il) {
        glm::GlmLayerWeights& b = w.blocks[(size_t) il];
        b.hc_attn_fn = rvec(24 * 4 * D, 0.0f, 0.02f); b.hc_attn_scale = rvec(3, 1.0f, 0.2f); b.hc_attn_base = rvec(24, 0.0f, 0.5f);
        b.hc_ffn_fn = rvec(24 * 4 * D, 0.0f, 0.02f); b.hc_ffn_scale = rvec(3, 1.0f, 0.2f); b.hc_ffn_base = rvec(24, 0.0f, 0.5f);
        b.attn_norm = rvec(D, 1.0f, 0.1f); b.ffn_norm = rvec(D, 1.0f, 0.1f);
        if (!g.is_mla[(size_t) il]) {
            b.wq = rlin(23, D, di); b.wk = rlin(23, D, di); b.wv = rlin(23, D, di);
            b.f_a = rlin(23, D, Dh); b.f_b = rlin(30, Dh, di); b.beta = rlin(23, D, H);
            b.g_a = rlin(23, D, Dh); b.g_b = rlin(30, Dh, di); b.wo = rlin(23, di, D);
            b.conv_q = rvec(di * 4, 0.0f, 0.5f); b.conv_k = rvec(di * 4, 0.0f, 0.5f); b.conv_v = rvec(di * 4, 0.0f, 0.5f);
            b.ssm_a = rvec(H, -4.0f, 3.0f); b.dt_bias = rvec(di, 0.0f, 0.5f); b.o_norm = rvec(Dh, 1.0f, 0.1f);
        } else {
            b.wq_a = rlin(23, D, g.q_lora_rank); b.wq_b = rlin(23, g.q_lora_rank, H * 256);
            b.wkv_a = rlin(23, D, 512); b.wo = rlin(23, H * 256, D);
            b.q_a_norm = rvec(g.q_lora_rank, 1.0f, 0.1f); b.kv_a_norm = rvec(512, 1.0f, 0.1f);
            b.k_b = (const uint16_t*) rand_tensor(30, 256, H * 512); b.v_b = (const uint16_t*) rand_tensor(30, 512, H * 256);
            b.idx_q_b = rlin(23, g.q_lora_rank, 32 * 128); b.idx_k = rlin(23, D, 128); b.idx_gate = rlin(0, D, 128);
            b.idx_proj = rlin(0, D, 32); b.idx_k_norm = rvec(128, 1.0f, 0.1f); b.idx_k_norm_b = rvec(128, 0.0f, 0.1f);
            b.idx_ape = rvec(4 * 128, 0.0f, 1.0f);
        }
        if (il < g.n_dense_lead) {
            b.ffn_gate = rlin(23, D, g.n_ff_dense); b.ffn_up = rlin(23, D, g.n_ff_dense); b.ffn_down = rlin(23, g.n_ff_dense, D);
        } else {
            b.gate_inp = rlin(0, D, g.n_expert); b.exp_probs_b = rvec(g.n_expert, 0.0f, 0.1f);
            b.expert_layout = k::native_expert_layout(23, 23, D, g.n_ff_exp);
            const uint64_t slot = (b.expert_layout.bytes + 255) & ~(uint64_t) 255;
            uint8_t* blobs = (uint8_t*) rand_tensor(23, (int64_t) slot / 136 * 256, g.n_expert);   // n_expert slots
            vector<unsigned long long> ptrs;
            for (int64_t e = 0; e < g.n_expert; ++e) ptrs.push_back((unsigned long long) (uintptr_t) (blobs + slot * (uint64_t) e));
            b.expert_ptrs = upload(ptrs);
            g_free.push_back((void*) b.expert_ptrs);
            b.expert_base = blobs;
            b.expert_stride = slot;
            b.sh_gate = rlin(23, D, g.n_ff_shexp); b.sh_up = rlin(23, D, g.n_ff_shexp); b.sh_down = rlin(23, g.n_ff_shexp, D);
        }
    }
    glm::GlmModel model;
    std::string err;
    if (!model.init(g, w, 4096, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    auto run = [&](bool graph, int64_t from, int n) {
        model.set_decode_graph(graph);
        model.decode(1, from, s, err);   // a warm-up (and with the graph: the capture comes next)
        model.decode(2, from + 1, s, err);
        check(cudaStreamSynchronize(s), "warm");
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) {
            if (!model.decode((int32_t) (i % 1000), from + 2 + i, s, err)) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(1); }
            check(cudaStreamSynchronize(s), "token");   // a token at a time, as sampling reads each
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n;
    };
    model.reset(s);
    const double eager = run(false, 0, 30);
    const double graph = run(true, 100, 30);
    const double eager2 = run(false, 200, 30);
    const double graph2 = run(true, 300, 30);
    std::printf("  a token as before: %.2f / %.2f ms; replayed: %.2f / %.2f ms (%.2f ms saved)\n", eager, eager2, graph,
                graph2, (eager + eager2 - graph - graph2) / 2);
    const double ms = std::min({eager, eager2, graph, graph2});
    std::printf("  the weights a token reads: %.2f GB (every tensor but the embedding; all 8 experts used): %.0f GB/s\n",
                (double) g_bytes / 1e9, (double) g_bytes / 1e6 / ms);
    cudaStreamDestroy(s);
    for (void* p : g_free) cudaFree(p);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--bench") return bench(i + 1 < argc ? std::atoll(argv[i + 1]) : 45);
        else { std::fprintf(stderr, "usage: glm_graph_parity [--selftest] | --bench [layers]\n"); return 2; }
    }
    TestModel tm = build_test_model(MAX_CTX);
    const Glm5NextGeometry& g = tm.g;
    glm::GlmModel eager, graph;
    std::string err;
    if (!eager.init(g, tm.w, MAX_CTX, err, nullptr, 16) || !graph.init(g, tm.w, MAX_CTX, err, nullptr, 16)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    graph.set_decode_graph(true);
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    vector<int32_t> tokens(40);
    for (auto& t : tokens) t = (int32_t) (unit() * (float) g.n_vocab) % (int32_t) g.n_vocab;
    const size_t V = (size_t) g.n_vocab, D = (size_t) g.n_embd;
    auto same_step = [&]() {
        return raw(eager.logits(), V * 4) == raw(graph.logits(), V * 4) &&
               raw(eager.h_nextn(), D * 4) == raw(graph.h_nextn(), D * 4);
    };
    auto decode_both = [&](int32_t tok, int64_t pos) {
        if (!eager.decode(tok, pos, s, err) || !graph.decode(tok, pos, s, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            std::exit(1);
        }
        check(cudaStreamSynchronize(s), "decode");
        return same_step();
    };
    std::printf("glm_graph: the captured decode step against decoding as before (5 layers + MTP, n_embd 1024)\n");

    eager.reset(s);
    graph.reset(s);
    bool same = true, ready_after_2 = false;
    for (int64_t p = 0; p < 10; ++p) {
        same &= decode_both(tokens[(size_t) p], p);
        if (p == 1) ready_after_2 = graph.decode_graph_ready();
    }
    expect(ready_after_2, "the second decode captured the step");
    expect(same, "10 tokens from a new sequence: logits and h_nextn bitwise, every token");

    eager.reset(s);
    graph.reset(s);
    if (!eager.prompt(tokens.data(), 16, 0, s, err) || !graph.prompt(tokens.data(), 16, 0, s, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    check(cudaStreamSynchronize(s), "prompt");
    same = same_step();
    for (int64_t p = 16; p < 22; ++p) same &= decode_both(tokens[(size_t) p], p);
    expect(same && graph.decode_graph_ready(), "a 16-token prompt, then 6 decodes replaying the graph: bitwise, every token");

    eager.reset(s);
    graph.reset(s);
    same = true;
    for (int64_t p = 0; p < 8; ++p) same &= decode_both(tokens[(size_t) (39 - p)], p);
    expect(same, "reset, and another sequence on the same graph: bitwise, every token");

    std::printf("\nglm_graph: %d failures\n", g_bad);
    cudaStreamDestroy(s);
    for (void* p : g_free) cudaFree(p);
    if (g_bad) return 1;
    if (selftest) std::printf("glm_graph_parity OK\n");
    return 0;
}
