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
//
// `glm_graph_parity --kernels [layers]` (`--kernels-dump`: each kind of half layer's kernels listed too): that model's
// captured step taken apart - its kernel nodes grouped by kind of half layer and by position within one, each group
// run as a graph of its own across all layers and timed on the host clock.  Kernel names need the ROCm SDK's
// amdhip64_7.dll beside a copy of the program (the driver's in System32 has no hipKernelNameRefByPtr); without it the
// groups are listed by address.
//
// Measured on the Radeon 8060S, 45 layers (ms a step; 9.75 GB of weights):
//
//     GlmModel::decode 60.5, replayed 58.6, the captured graph alone 58.5, its kernels rebuilt as a chain 58.3 -
//     the launches cost next to nothing: the step is the GPU's time
//
//     the streaming kernels near the memory's speed: the IQ4_XS GEMVs 216-226 GB/s (KDA q/k/v/o 81 us each, 11.1 ms
//     a step), the experts' gate+up 230 GB/s (310 us, 13.0 ms) and down 139 us (5.8 ms; its 36 MB - here the same 8
//     experts at every call, so partly from the 32 MB cache), the head's Q8_0 GEMV 233 GB/s (2.9 ms), MLA's BF16
//     k_b / v_b 215 GB/s - together ~44 ms of the 58
//
//     the rest, ~14 ms, in small kernels that move little: rms_norm_weighted (one block) 28-30 us x 102 = 2.9 ms,
//     kda_scan_norm 59 us x 34 = 2.0 ms, mhc_mix_partial 16 us x 90 = 1.4 ms, the F32 GEMVs (router, indexer gate and
//     proj; one block per 8 rows) 19-21 us x 64 = 1.3 ms, mhc_finalize 7.8 us x 90 = 0.7 ms, q8_1 quantizations
//     1.8 us x 440 = 0.8 ms (a KDA layer quantizes the same h six times), ~1.5-2 us for every tiny kernel
//
//     then fixed: the attention / FFN norms in mhc_pre's finalize (2.9 ms gone, finalize +0.2), the F32 GEMV a block
//     a row (0.16 ms), mhc_mix_partial over 6 x the blocks (1.0 ms), each decode input quantized once (192 calls,
//     0.33 ms), the other GLM norms a block a row (glm_rms_norm) - 1623 kernels, GlmModel::decode 53.0, rebuilt 50.7-51.5
//
// On HIP under Windows the events around launches close early (the whole step "at" 692-1182 GB/s, over the memory's
// 256): the cross-check table prints them beside the host clock.
#include "glm_ref.hpp"

#include "strata/glm/glm_model.hpp"

#if defined(STRATA_USE_HIP) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <cctype>
#include <chrono>
#include <map>
#include <set>

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
    g_bytes += (uint64_t) n * 4;
    return up_f(v);
}

/// A model of GLM-5.3-Flash's shapes with `layers` main layers and random weights (8 resident experts a layer); g_bytes
/// is then what a decode token reads: g_half_bytes[2 il] layer il's attention half's (its mHC, norm and attention
/// weights), [2 il + 1] its FFN half's, g_head_bytes the output norm's and head's.
vector<uint64_t> g_half_bytes;
uint64_t g_head_bytes = 0;
void build_bench(int64_t layers, Glm5NextGeometry& g, glm::GlmModelWeights& w) {
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
    w.tok_embd = {8, rand_tensor(8, D, g.n_vocab), k::iq_row_bytes(8, D)};
    g_bytes = 0;   // a token reads one row of the embedding: not counted
    w.output_norm = rvec(D, 1.0f, 0.1f);
    w.output = rlin(8, D, g.n_vocab);
    g_head_bytes = g_bytes;
    g_half_bytes.assign((size_t) (2 * layers), 0);
    w.blocks.resize((size_t) layers);
    for (int64_t il = 0; il < layers; ++il) {
        glm::GlmLayerWeights& b = w.blocks[(size_t) il];
        uint64_t mark = g_bytes;
        b.hc_attn_fn = rvec(24 * 4 * D, 0.0f, 0.02f); b.hc_attn_scale = rvec(3, 1.0f, 0.2f); b.hc_attn_base = rvec(24, 0.0f, 0.5f);
        b.attn_norm = rvec(D, 1.0f, 0.1f);
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
        g_half_bytes[(size_t) (2 * il)] = g_bytes - mark;
        mark = g_bytes;
        b.hc_ffn_fn = rvec(24 * 4 * D, 0.0f, 0.02f); b.hc_ffn_scale = rvec(3, 1.0f, 0.2f); b.hc_ffn_base = rvec(24, 0.0f, 0.5f);
        b.ffn_norm = rvec(D, 1.0f, 0.1f);
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
        g_half_bytes[(size_t) (2 * il + 1)] = g_bytes - mark;
    }
}

int bench(int64_t layers) {
    std::printf("decode on a model of GLM-5.3-Flash's shapes, %lld layers, 8 resident IQ4_XS experts a layer (building)\n",
                (long long) layers);
    Glm5NextGeometry g;
    glm::GlmModelWeights w;
    build_bench(layers, g, w);
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

// ---------------------------------------------------------------- --kernels: the captured step, kernel by kernel
#if defined(STRATA_USE_HIP)
using GNode = hipGraphNode_t;
using KParams = hipKernelNodeParams;
bool node_list(cudaGraph_t gr, vector<GNode>& out) {
    size_t n = 0;
    if (hipGraphGetNodes(gr, nullptr, &n) != hipSuccess) return false;
    out.resize(n);
    return hipGraphGetNodes(gr, out.data(), &n) == hipSuccess;
}
bool kernel_params(GNode nd, KParams& p) {
    hipGraphNodeType t;
    return hipGraphNodeGetType(nd, &t) == hipSuccess && t == hipGraphNodeTypeKernel &&
           hipGraphKernelNodeGetParams(nd, &p) == hipSuccess;
}
std::string mangled(const void* f) {
#if defined(_WIN32)
    // looked up, not linked: the driver's amdhip64 in System32 (which Windows loads before PATH) has no
    // hipKernelNameRefByPtr - the ROCm SDK's has (its amdhip64_7.dll beside a copy of this program: the names)
    using NameFn = const char* (*) (const void*, hipStream_t);
    static const NameFn fn = [] {
        HMODULE m = GetModuleHandleA("amdhip64_7.dll");
        if (!m) m = GetModuleHandleA("amdhip64_6.dll");
        return m ? (NameFn) (void*) GetProcAddress(m, "hipKernelNameRefByPtr") : (NameFn) nullptr;
    }();
    const char* n = fn ? fn(f, nullptr) : nullptr;
#else
    const char* n = hipKernelNameRefByPtr(f, nullptr);
#endif
    if (n) return n;
    char b[32];
    std::snprintf(b, sizeof b, "kernel@%p", f);
    return b;
}
cudaError_t graph_new(cudaGraph_t* gr) { return hipGraphCreate(gr, 0); }
cudaError_t launch(const KParams& p, cudaStream_t s) {
    return hipLaunchKernel(p.func, p.gridDim, p.blockDim, p.kernelParams, p.sharedMemBytes, s);
}
cudaError_t add_kernel(cudaGraph_t gr, GNode* nd, const GNode* dep, const KParams& p) {
    return hipGraphAddKernelNode(nd, gr, dep, dep ? 1 : 0, &p);
}
#else
using GNode = cudaGraphNode_t;
using KParams = cudaKernelNodeParams;
bool node_list(cudaGraph_t gr, vector<GNode>& out) {
    size_t n = 0;
    if (cudaGraphGetNodes(gr, nullptr, &n) != cudaSuccess) return false;
    out.resize(n);
    return cudaGraphGetNodes(gr, out.data(), &n) == cudaSuccess;
}
bool kernel_params(GNode nd, KParams& p) {
    cudaGraphNodeType t;
    return cudaGraphNodeGetType(nd, &t) == cudaSuccess && t == cudaGraphNodeTypeKernel &&
           cudaGraphKernelNodeGetParams(nd, &p) == cudaSuccess;
}
std::string mangled(const void* f) {
#if CUDART_VERSION >= 12030   // cudaFuncGetName arrived in CUDA 12.3
    const char* n = nullptr;
    if (cudaFuncGetName(&n, f) == cudaSuccess && n) return n;
#endif
    char b[32];
    std::snprintf(b, sizeof b, "kernel@%p", f);
    return b;
}
cudaError_t graph_new(cudaGraph_t* gr) { return cudaGraphCreate(gr, 0); }
cudaError_t launch(const KParams& p, cudaStream_t s) {
    return cudaLaunchKernel(p.func, p.gridDim, p.blockDim, p.kernelParams, p.sharedMemBytes, s);
}
cudaError_t add_kernel(cudaGraph_t gr, GNode* nd, const GNode* dep, const KParams& p) {
    return cudaGraphAddKernelNode(nd, gr, dep, dep ? 1 : 0, &p);
}
#endif

/// The function's own name out of an Itanium-mangled one (_ZN6strata7kernels12_GLOBAL__N_14nameI...: "name"), with the
/// start of its template arguments, which tells the instances apart.
std::string short_name(const std::string& m) {
    if (m.rfind("_Z", 0) != 0) return m.substr(0, 48);
    size_t i = 2;
    const bool nested = i < m.size() && m[i] == 'N';
    if (nested) ++i;
    std::string last;
    while (i < m.size() && std::isdigit((unsigned char) m[i])) {
        size_t len = 0;
        while (i < m.size() && std::isdigit((unsigned char) m[i])) len = len * 10 + (size_t) (m[i++] - '0');
        last = m.substr(i, len);
        i += len;
        if (!nested) break;
    }
    if (last.empty()) return m.substr(0, 48);
    if (i < m.size() && m[i] == 'I') last += "<" + m.substr(i + 1, std::min<size_t>(10, m.size() - i - 1)) + ">";
    return last;
}

/// How a run is timed.  On HIP under Windows the events around graph launches - and even around plain launches - can
/// close before the work has run (the whole step "at" 470-1200 GB/s, over the memory's 256): every time here is on the
/// host clock, R runs launched back to back and then waited for, except the cross-check's event column.
enum class Timing { Wall, WaitEach, Events };

/// ms one run of graph `ex` takes: R runs back to back (enough for ~100 ms), the best of three batches.
double time_exec(cudaGraphExec_t ex, cudaStream_t s, Timing how) {
    cudaEvent_t a = nullptr, b = nullptr;
    if (how == Timing::Events) {
        check(cudaEventCreate(&a), "event");
        check(cudaEventCreate(&b), "event");
    }
    auto batch = [&](int reps) {
        const auto t0 = std::chrono::steady_clock::now();
        if (a) check(cudaEventRecord(a, s), "event");
        for (int r = 0; r < reps; ++r) {
            check(cudaGraphLaunch(ex, s), "launch");
            if (how == Timing::WaitEach) check(cudaStreamSynchronize(s), "sync");
        }
        if (b) check(cudaEventRecord(b, s), "event");
        check(cudaStreamSynchronize(s), "sync");
        float ms = 0;
        if (a) check(cudaEventElapsedTime(&ms, a, b), "event");
        const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return (a ? (double) ms : wall) / reps;
    };
    batch(1);   // the first run's setup
    const double one = batch(1);
    const int reps = (int) std::clamp(100.0 / std::max(one, 1e-3), 3.0, 2000.0);
    double best = 1e30;
    for (int t = 0; t < 3; ++t) best = std::min(best, batch(reps));
    if (a) cudaEventDestroy(a);
    if (b) cudaEventDestroy(b);
    return best;
}

/// ms one run of `nodes` (kernel parameters, chained in this order as a graph of their own) takes.
double time_chain(const vector<const KParams*>& nodes, cudaStream_t s, Timing how = Timing::Wall) {
    cudaGraph_t gr = nullptr;
    check(graph_new(&gr), "graph");
    GNode prev = nullptr;
    for (const KParams* p : nodes) {
        GNode nd = nullptr;
        check(add_kernel(gr, &nd, prev ? &prev : nullptr, *p), "kernel node");
        prev = nd;
    }
    cudaGraphExec_t ex = nullptr;
    check(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
    const double ms = time_exec(ex, s, how);
    cudaGraphExecDestroy(ex);
    cudaGraphDestroy(gr);
    return ms;
}

/// ms one pass over `nodes` takes, launched one by one (not a graph): R passes back to back, timed by events around
/// them (`events`) or on the host (launch, then wait).
double time_eager(const vector<const KParams*>& nodes, cudaStream_t s, bool events, int reps) {
    auto pass = [&] {
        for (const KParams* p : nodes) check(launch(*p, s), "launch");
    };
    pass();
    check(cudaStreamSynchronize(s), "sync");
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");
    double best = 1e30;
    for (int t = 0; t < 3; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        check(cudaEventRecord(a, s), "event");
        for (int r = 0; r < reps; ++r) pass();
        check(cudaEventRecord(b, s), "event");
        check(cudaStreamSynchronize(s), "sync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, a, b), "event");
        const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        best = std::min(best, (events ? (double) ms : wall) / reps);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return best;
}

/// R launches of a graph of `nodes`, then a wait: ms a launch on the host clock.
double time_graph_wall(const vector<const KParams*>& nodes, cudaStream_t s, int reps) {
    cudaGraph_t gr = nullptr;
    check(graph_new(&gr), "graph");
    GNode prev = nullptr;
    for (const KParams* p : nodes) {
        GNode nd = nullptr;
        check(add_kernel(gr, &nd, prev ? &prev : nullptr, *p), "kernel node");
        prev = nd;
    }
    cudaGraphExec_t ex = nullptr;
    check(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
    check(cudaGraphLaunch(ex, s), "launch");
    check(cudaStreamSynchronize(s), "sync");
    double best = 1e30;
    for (int t = 0; t < 3; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) check(cudaGraphLaunch(ex, s), "launch");
        check(cudaStreamSynchronize(s), "sync");
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps);
    }
    cudaGraphExecDestroy(ex);
    cudaGraphDestroy(gr);
    return best;
}

/// The decode step at GLM-5.3-Flash's shapes (the --bench model), taken apart: the captured step's kernels by kind of
/// half layer (KDA / MLA attention, dense / MoE FFN) and by kernel position within a kind, every group timed as a graph
/// of its own of its kernels from all layers back to back - so a group's weights stream from memory as they do in the
/// step (one layer's tensor alone, repeated, would sit in the 8060S's 32 MB Infinity Cache).  A group's kernels read
/// what the step left in the buffers: a routed-expert group reads the last layer's 8 experts every time (107 MB).

/// `--spec [layers]`: a verify window's time (GlmModel::verify, T tokens through the decode kernels) and commit's,
/// against a decode token's, on the --bench model.  Host clock, a host wait after every call.
int spec(int64_t layers) {
    std::printf("verify windows on a model of GLM-5.3-Flash's shapes, %lld layers, 8 resident IQ4_XS experts a layer"
                " (building)\n", (long long) layers);
    Glm5NextGeometry g;
    glm::GlmModelWeights w;
    build_bench(layers, g, w);
    glm::GlmModel model;
    std::string err;
    if (!model.init(g, w, 4096, err, nullptr, 9) || !model.init_verify(8, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    model.reset(s);
    auto die = [&]() {
        std::fprintf(stderr, "%s\n", err.c_str());
        std::exit(1);
    };
    for (int i = 0; i < 40; ++i)   // 40 cells of context
        if (!model.decode((int32_t) (7 * i + 1), i, s, err)) die();
    check(cudaStreamSynchronize(s), "decode");
    auto ms_of = [&](auto&& f) {
        double best = 1e30;
        for (int t = 0; t < 3; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 5; ++r) {
                f();
                check(cudaStreamSynchronize(s), "sync");
            }
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5);
        }
        return best;
    };
    const double dec = ms_of([&] { if (!model.decode(3, 40, s, err)) die(); });
    std::printf("  a decode token: %.2f ms\n", dec);
    std::printf("  %3s %12s %14s %12s %18s\n", "T", "verify ms", "ms a token", "commit ms", "T decodes would be");
    for (const int T : {1, 2, 4, 8}) {
        int32_t toks[8], picks[8];
        for (int i = 0; i < T; ++i) toks[i] = 11 * i + 5;
        const double v = ms_of([&] { if (!model.verify(toks, T, 40, picks, s, err)) die(); });
        const double c = ms_of([&] {
            if (!model.verify(toks, T, 40, picks, s, err) || !model.commit(T, 1, s, err)) die();
        }) - v;
        std::printf("  %3d %12.2f %14.2f %12.2f %15.2f ms\n", T, v, v / T, c, dec * T);
    }
    cudaStreamDestroy(s);
    for (void* p : g_free) cudaFree(p);
    return 0;
}

int kernels(int64_t layers, bool dump) {
    std::printf("the decode step at GLM-5.3-Flash's shapes, %lld layers, 8 resident IQ4_XS experts a layer, kernel by kernel\n",
                (long long) layers);
    Glm5NextGeometry g;
    glm::GlmModelWeights w;
    build_bench(layers, g, w);
    glm::GlmModel model;
    std::string err;
    if (!model.init(g, w, 4096, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    model.reset(s);
    model.set_decode_graph(true);
    for (int i = 0; i < 40; ++i)   // 40 cells of context for the MLA layers; the step is captured on the 2nd
        if (!model.decode((int32_t) (7 * i + 1), i, s, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    check(cudaStreamSynchronize(s), "decode");
    const cudaGraph_t gr = (cudaGraph_t) model.decode_graph();
    vector<GNode> all;
    if (!gr || !node_list(gr, all)) { std::fprintf(stderr, "no captured step\n"); return 1; }

    struct KNode {
        KParams p;
        std::string name, label;
        int half = -1, pos = 0;
    };
    vector<KNode> ks;
    size_t other = 0;
    for (GNode nd : all) {
        KNode kn;
        if (!kernel_params(nd, kn.p)) { ++other; continue; }
        kn.name = mangled(kn.p.func);
        ks.push_back(kn);
    }
    // the halves: each starts with mhc_pre's first kernel (the first one that runs 2 x layers times) and ends with
    // mhc_post's (the one before the second start); the head follows the last
    std::map<std::string, int64_t> count;
    for (const KNode& kn : ks) ++count[kn.name];
    std::string marker;
    for (const KNode& kn : ks)
        if (count[kn.name] == 2 * layers) { marker = kn.name; break; }
    size_t second = 0;
    for (size_t i = 0, seen = 0; i < ks.size(); ++i)
        if (ks[i].name == marker && ++seen == 2) { second = i; break; }
    if (marker.empty() || second == 0) { std::fprintf(stderr, "the step's halves not found\n"); return 1; }
    const std::string post = ks[second - 1].name;
    size_t last_post = 0;
    for (size_t i = 0; i < ks.size(); ++i)
        if (ks[i].name == post) last_post = i;
    const int n_half = (int) (2 * layers);
    int half = -1, pos = 0;
    for (size_t i = 0; i < ks.size(); ++i) {
        if (i > last_post && half < n_half) { half = n_half; pos = 0; }
        else if (ks[i].name == marker) { ++half; pos = 0; }
        ks[i].half = half;
        ks[i].pos = pos++;
    }
    auto kind = [&](int h) -> std::string {
        if (h < 0) return "embedding";
        if (h >= n_half) return "head";
        const int64_t il = h / 2;
        if (h % 2 == 0) return g.is_mla[(size_t) il] ? "MLA attention" : "KDA attention";
        return il < g.n_dense_lead ? "dense FFN" : "MoE FFN";
    };
    for (KNode& kn : ks) kn.label = kind(kn.half);
    std::printf("  the captured step: %zu nodes, %zu kernels (%zu other nodes); a half layer starts at %s\n", all.size(),
                ks.size(), other, short_name(marker).c_str());
    if (dump) {   // every kind's first half, kernel by kernel
        std::map<std::string, int> first_half;
        for (const KNode& kn : ks)
            if (!first_half.count(kn.label)) first_half[kn.label] = kn.half;
        for (const KNode& kn : ks) {
            if (first_half[kn.label] != kn.half) continue;
            std::printf("  %-14s %3d  %-48s grid %5u x %3u x %2u  block %4u x %2u  lds %6u\n", kn.label.c_str(), kn.pos,
                        short_name(kn.name).c_str(), kn.p.gridDim.x, kn.p.gridDim.y, kn.p.gridDim.z, kn.p.blockDim.x,
                        kn.p.blockDim.y, (unsigned) kn.p.sharedMemBytes);
        }
        std::printf("\n");
    }

    // the whole step's kernels back to back, and the step as captured
    vector<const KParams*> chain;
    for (const KNode& kn : ks) chain.push_back(&kn.p);
    const double step_ms = time_chain(chain, s);
    const double step_wait = time_chain(chain, s, Timing::WaitEach);
    cudaGraphExec_t cap = nullptr;   // the captured graph itself
    check(cudaGraphInstantiate(&cap, gr, 0), "instantiate");
    const double cap_ms = time_exec(cap, s, Timing::Wall), cap_wait = time_exec(cap, s, Timing::WaitEach);
    cudaGraphExecDestroy(cap);
    auto decodes = [&](bool graph, bool wait) {
        model.set_decode_graph(graph);
        for (int i = 0; i < 3; ++i) model.decode(1, 39, s, err);   // (with the graph: captured again)
        check(cudaStreamSynchronize(s), "decode");
        double best = 1e30;
        for (int t = 0; t < 3; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 10; ++i) {
                if (!model.decode(1, 39, s, err)) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(1); }
                if (wait) check(cudaStreamSynchronize(s), "decode");
            }
            check(cudaStreamSynchronize(s), "decode");
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 10);
        }
        return best;
    };
    {   // the timing methods against each other: the whole step, and three groups of known size
        auto group = [&](const std::string& label, int pos) {
            vector<const KParams*> v;
            for (const KNode& kn : ks)
                if (kn.label == label && kn.pos == pos) v.push_back(&kn.p);
            return v;
        };
        struct Probe { const char* what; vector<const KParams*> v; double mb; };
        vector<Probe> probes = {
            {"the whole step", chain, (double) (g_head_bytes + std::accumulate(g_half_bytes.begin(), g_half_bytes.end(), (uint64_t) 0)) / 1e6},
            {"MoE gate+up experts (71.3 MB each)", group("MoE FFN", 7), 71.3 * (double) group("MoE FFN", 7).size()},
            {"KDA attn_q (17.8 MB each)", group("KDA attention", 4), 17.8 * (double) group("KDA attention", 4).size()},
            {"attention's rms_norm", group("KDA attention", 2), 0.0},
            {"KDA's first q8_1 quantize", group("KDA attention", 3), 0.0},
        };
        std::printf("  ms a pass, by four timings (best of 3; R passes back to back):\n");
        std::printf("  %-38s %6s %12s %12s %12s %12s\n", "", "nodes", "graph+events", "graph+wall", "eager+events", "eager+wall");
        for (const Probe& pr : probes) {
            if (pr.v.empty()) continue;
            const int reps = pr.v.size() > 500 ? 5 : 50;
            const double ge = time_chain(pr.v, s, Timing::Events), gw = time_graph_wall(pr.v, s, reps);
            const double ee = time_eager(pr.v, s, true, reps), ew = time_eager(pr.v, s, false, reps);
            std::printf("  %-38s %6zu %12.3f %12.3f %12.3f %12.3f", pr.what, pr.v.size(), ge, gw, ee, ew);
            if (pr.mb > 0)
                std::printf("   GB/s %4.0f %4.0f %4.0f %4.0f", pr.mb / ge, pr.mb / gw, pr.mb / ee, pr.mb / ew);
            std::printf("\n");
        }
        std::printf("\n");
    }

    // by kind of half layer
    std::printf("  %-14s %6s %10s %10s %10s %8s %7s\n", "half layer", "count", "ms each", "ms total", "MB each", "GB/s",
                "share");
    const vector<std::string> kinds = {"embedding", "KDA attention", "MLA attention", "dense FFN", "MoE FFN", "head"};
    double sum_kinds = 0;
    for (const std::string& kd : kinds) {
        vector<const KParams*> c;
        std::set<int> halves;
        uint64_t b = 0;
        for (const KNode& kn : ks)
            if (kn.label == kd) {
                c.push_back(&kn.p);
                if (halves.insert(kn.half).second)
                    b += kn.half < 0 ? 0 : kn.half >= n_half ? g_head_bytes : g_half_bytes[(size_t) kn.half];
            }
        if (c.empty()) continue;
        const double ms = time_chain(c, s);
        sum_kinds += ms;
        const double n = (double) halves.size();
        std::printf("  %-14s %6zu %10.3f %10.2f %10.1f %8.0f %6.1f%%\n", kd.c_str(), halves.size(), ms / n, ms,
                    (double) b / 1e6 / n, b ? (double) b / 1e6 / ms : 0.0, 100.0 * ms / step_ms);
    }
    std::printf("  %-14s %6s %10s %10.2f   (the step's kernels back to back: %.2f ms)\n\n", "sum", "", "", sum_kinds,
                step_ms);

    // by kernel position within a kind
    struct Group {
        std::string label, name;
        int pos = 0;
        bool mixed = false;
        vector<const KParams*> nodes;
        double ms = 0;
    };
    std::map<std::pair<std::string, int>, Group> groups;
    for (const KNode& kn : ks) {
        Group& gp = groups[{kn.label, kn.pos}];
        if (gp.nodes.empty()) {
            gp.label = kn.label;
            gp.pos = kn.pos;
            gp.name = kn.name;
        } else if (gp.name != kn.name) {
            gp.mixed = true;
        }
        gp.nodes.push_back(&kn.p);
    }
    double sum_groups = 0;
    for (auto& [key, gp] : groups) {
        gp.ms = time_chain(gp.nodes, s);
        sum_groups += gp.ms;
    }
    std::printf("  %-14s %3s  %-40s %-17s %5s %9s %8s %7s\n", "half layer", "#", "kernel", "grid x block", "calls",
                "us each", "ms total", "share");
    for (const std::string& kd : kinds)
        for (auto& [key, gp] : groups) {
            if (gp.label != kd) continue;
            const KParams& p = *gp.nodes.front();
            char gb[48];
            std::snprintf(gb, sizeof gb, "%ux%u x %u", p.gridDim.x, p.gridDim.y * p.gridDim.z, p.blockDim.x * p.blockDim.y);
            std::printf("  %-14s %3d  %-40.40s %-17s %5zu %9.1f %8.3f %6.1f%%%s\n", kd.c_str(), gp.pos,
                        short_name(gp.name).c_str(), gb, gp.nodes.size(), 1000.0 * gp.ms / (double) gp.nodes.size(), gp.ms,
                        100.0 * gp.ms / step_ms, gp.mixed ? "  (kernels differ by layer)" : "");
        }
    std::printf("  %-14s %3s  %-40s %-17s %5s %9s %8.2f   (the step's kernels back to back: %.2f ms)\n", "sum", "", "",
                "", "", "", sum_groups, step_ms);

    // by kernel, whatever the position
    std::map<std::string, std::pair<double, size_t>> by_name;
    for (auto& [key, gp] : groups) {
        auto& e = by_name[short_name(gp.name)];
        e.first += gp.ms;
        e.second += gp.nodes.size();
    }
    vector<std::pair<double, std::string>> top;
    for (auto& [n, e] : by_name) top.push_back({e.first, n});
    std::sort(top.rbegin(), top.rend());
    std::printf("\n  by kernel:\n");
    for (auto& [ms, n] : top)
        std::printf("  %-48.48s %5zu calls %8.3f ms %6.1f%%\n", n.c_str(), by_name[n].second, ms, 100.0 * ms / step_ms);

    // last: decoding drops the captured graph (whose kernel parameters the groups above read)
    const double eager_wait = decodes(false, true), eager_run = decodes(false, false);
    const double graph_wait = decodes(true, true), graph_run = decodes(true, false);
    uint64_t bytes = g_head_bytes;
    for (uint64_t b : g_half_bytes) bytes += b;
    std::printf("\n  ms a step              back to back   a host wait after each\n");
    std::printf("  GlmModel::decode         %8.2f         %8.2f\n", eager_run, eager_wait);
    std::printf("  ... with the graph       %8.2f         %8.2f\n", graph_run, graph_wait);
    std::printf("  the captured graph       %8.2f         %8.2f    (cudaGraphLaunch alone)\n", cap_ms, cap_wait);
    std::printf("  its kernels rebuilt      %8.2f         %8.2f    (a graph of the same kernel nodes, chained)\n",
                step_ms, step_wait);
    std::printf("  %.2f GB of weights a step: %.0f GB/s at the rebuilt graph's pace\n\n", (double) bytes / 1e9,
                (double) bytes / 1e6 / step_ms);
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
        else if (a == "--spec") return spec(i + 1 < argc ? std::atoll(argv[i + 1]) : 45);
        else if (a == "--kernels" || a == "--kernels-dump")
            return kernels(i + 1 < argc ? std::atoll(argv[i + 1]) : 45, a == "--kernels-dump");
        else {
            std::fprintf(stderr, "usage: glm_graph_parity [--selftest] | --bench [layers] | --kernels[-dump] [layers] | "
                                 "--spec [layers]\n");
            return 2;
        }
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
