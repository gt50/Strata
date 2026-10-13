// src/glm/glm_layer_parity.cpp - glm_layer_decode (glm_layer.hpp) through three synthetic GLM-5.3-Flash blocks
// against a double-precision transcription of llama.cpp's glm5-next layer, token by token.
//
// The blocks have GLM's structure at a smaller width (n_embd 1024, 8 heads, 16 experts of 256, 4 used) and the
// kernels' compiled sizes (KDA heads of 128, the 512-wide latent, the indexer's 32 x 128 and k-pool 4, 4 mHC
// streams, top_k 2048):
//   block 0: KDA attention and a dense SwiGLU FFN
//   block 1: MLA with its own (full) indexer, and the MoE (router, 4 of 16 IQ4_XS experts, a shared expert)
//   block 2: MLA sharing block 1's selection, and the MoE
// Projections in every type glm_linear routes differently: F32, BF16, Q8_0 (native_mmvq) and IQ4_XS.
//
// The caches start with 2100 cells of history, past the indexer's 2051, so the selection is sparse (512 of 525
// pools and the tail); four decode tokens follow (positions 2100 .. 2103: a tail of 1, 2, 3 cells, then a pool
// completes).  After every block the four streams are compared with the reference, and so are the discrete choices:
// the indexer's cells and the router's experts and their order.
//
// The reference and the synthetic weights are glm_ref.hpp's, shared with glm_model_parity.
//
// Each block's reference starts from the streams the device gave it (and an MLA block's from the device's latent
// history, glm_ref.hpp's latent_sync), and the state a block carries from token to token (the KDA conv history and
// recurrent state; the latent row a token writes; the indexer's cache through its choices) is compared on its own.
// q8_1 roundings are steps: where the reference rounds a code or a block's FP16 scale on a boundary, the device's
// value, 1e-7 away, can step the other way.  A block is held to 1e-4 - against the reference as rounded, or with
// some of its roundings on a boundary taken the other way (glm_ref.hpp's verify; on this fixture one block in
// twelve: a scale one FP16 step apart, 8.5e-4 as rounded).
#include "glm_ref.hpp"

namespace {

constexpr int64_t P0 = 2100;     // cells of history
constexpr int STEPS = 4;
constexpr int64_t MAX_CTX = 2304;

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: glm_layer_parity [--selftest]\n"); return 2; }
    }
    Glm5NextGeometry g;
    g.n_layers = 3; g.n_nextn = 0; g.n_embd = 1024; g.n_vocab = 1024; g.n_ctx_train = MAX_CTX;
    g.rms_eps = 1e-5; g.norm_eps = 1e-6;
    g.is_mla = {0, 1, 1};
    g.n_head = 8;
    g.kda_head_dim = 128; g.ssm_d_conv = 4; g.has_kda_gate_lower_bound = true; g.kda_gate_lower_bound = -5;
    g.q_lora_rank = 512; g.kv_lora_rank = 512; g.n_rot = 0; g.head_dim_k = 256; g.head_dim_v = 256;
    g.idx_heads = 32; g.idx_key_dim = 128; g.idx_top_k = 2048; g.idx_kpool = 4; g.idx_kpool_select_tail = true;
    g.idx_full = {1, 1, 0};
    g.hc = 4; g.hc_sinkhorn_iters = 20; g.hc_eps = 1e-6;
    g.n_dense_lead = 1; g.n_ff_dense = 512; g.n_expert = 16; g.n_expert_used = 4; g.n_expert_shared = 1;
    g.n_ff_exp = 256; g.n_ff_shexp = 256; g.expert_gating = 2; g.expert_weights_scale = 2.5;
    g.expert_weights_norm = true; g.swiglu_clamp_exp = 10; g.swiglu_clamp_shexp = 10;
    const int D = (int) g.n_embd, H = (int) g.n_head, Dh = 128, di = H * Dh, L = 512;

    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "stream");
    vector<Block> blocks(3);
    // ---- block 0: KDA and the dense FFN
    {
        Block& b = blocks[0];
        build_mhc(b, D);
        build_kda_attn(b, g);
        build_dense_ffn(b, g);
        // state: a conv history and a recurrent state as after some tokens
        const vector<float> hist = gvec((size_t) 3 * di * 3, 1.0f), stt = gvec((size_t) Dh * H * Dh, 0.05f);
        b.conv_hist.assign(hist.begin(), hist.end());
        b.kda_state.assign(stt.size(), 0.0);
        vector<float> dev_state(stt.size());
        for (int hh = 0; hh < H; ++hh)   // stt is the reference's [h][j][i]; the device's is [h][i][j]
            for (int j = 0; j < Dh; ++j)
                for (int i = 0; i < Dh; ++i) {
                    const float v = stt[((size_t) hh * Dh + j) * Dh + i];
                    b.kda_state[((size_t) hh * Dh + j) * Dh + i] = v;
                    dev_state[((size_t) hh * Dh + i) * Dh + j] = v;
                }
        void* base = nullptr;
        check(cudaMalloc(&base, glm::glm_layer_state_bytes(g, 0, MAX_CTX)), "state 0");
        g_free.push_back(base);
        glm::glm_layer_state_carve(g, 0, MAX_CTX, base, b.st);
        check(cudaMemcpy(b.st.conv_hist, hist.data(), hist.size() * 4, cudaMemcpyHostToDevice), "hist");
        check(cudaMemcpy(b.st.kda_state, dev_state.data(), dev_state.size() * 4, cudaMemcpyHostToDevice), "kda state");
    }
    // ---- blocks 1, 2: MLA (1 with its own indexer) and the MoE
    for (int il = 1; il <= 2; ++il) {
        Block& b = blocks[(size_t) il];
        const bool full = g.idx_full[(size_t) il] != 0;
        build_mhc(b, D);
        build_mla_attn(b, g, full);
        build_moe(b, g);
        // state: P0 cells of history
        void* base = nullptr;
        check(cudaMalloc(&base, glm::glm_layer_state_bytes(g, il, MAX_CTX)), "state");
        g_free.push_back(base);
        glm::glm_layer_state_carve(g, il, MAX_CTX, base, b.st);
        b.latent.assign((size_t) MAX_CTX * L, 0);
        for (size_t i = 0; i < (size_t) P0 * L; ++i) b.latent[i] = k::f16_from_f32(gauss());
        check(cudaMemcpy(b.st.latent, b.latent.data(), b.latent.size() * 2, cudaMemcpyHostToDevice), "latent");
        if (full) {
            vector<float> cache((size_t) MAX_CTX * 256, 0.0f);
            for (size_t i = 0; i < (size_t) P0 * 256; ++i) cache[i] = gauss();
            b.idx_cache.assign(cache.begin(), cache.end());
            b.pooled.assign((size_t) (MAX_CTX / 4 + 1) * 128, 0.0);
            for (int64_t p = 0; p < P0 / 4; ++p) pool_ref(b, p);
            check(cudaMemcpy(b.st.idx_cache, cache.data(), cache.size() * 4, cudaMemcpyHostToDevice), "idx cache");
            k::kpool_pool(b.st.idx_cache, b.w.idx_ape, b.st.pooled, 0, P0 / 4, nullptr, stream);
            check(cudaStreamSynchronize(stream), "pool history");
        }
    }

    glm::GlmContext ctx;
    std::string err;
    if (!ctx.init(g, MAX_CTX, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    vector<float> x0 = gvec((size_t) 4 * D, 1.0f);
    float* d_x = upload(x0);
    vector<double> X(x0.begin(), x0.end());
    Ref r{g, {}, {}};
    int bad = 0;
    std::printf("glm_layer_decode, 3 blocks (KDA + dense, MLA + own indexer + MoE, MLA + shared indexer + MoE),\n"
                "n_embd %d, %d cells of history, %d tokens:\n", D, (int) P0, STEPS);
    for (int s = 0; s < STEPS; ++s) {
        const int64_t pos = P0 + s;
        ctx.set_pos(pos, stream);
        for (int il = 0; il < 3; ++il) {
            Block& b = blocks[(size_t) il];
            // The reference takes the block's input from the device: the q8_1 roundings make a quantized projection
            // jump where an input sits on a rounding boundary, so a 1e-6 difference carried in from the blocks before
            // would sometimes show as a 1e-3 one here.  Each block is checked on its own arithmetic; the state it
            // carries from token to token is checked separately below.
            const vector<float> xin = download(d_x, X.size());
            if (!glm::glm_layer_decode(g, il, b.w, b.st, ctx, d_x, stream, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            check(cudaStreamSynchronize(stream), "layer");
            const bool mla = g.is_mla[(size_t) il] != 0, full = g.idx_full[(size_t) il] != 0;
            if (mla) latent_sync(b, pos);
            const vector<float> xout = download(d_x, X.size());
            const KdaDev kd = mla ? KdaDev{} : kda_dev(b, g);
            const Snap snap = save(b, r);
            if (!verify("the four streams", pos, il, [&] {
                    X.assign(xin.begin(), xin.end());
                    block_ref(b, g, il, pos, X, r);
                    double rh, rs;
                    return Fit{rel_l1(X, xout), mla ? latent_row_far(b) == 0 : kda_close(b, kd, rh, rs)};
                }, [&] { restore(b, r, snap); }))
                ++bad;
            if (mla && !latent_row_ok(b, pos, il)) ++bad;
            if (mla && full) {
                vector<int32_t> got = download(ctx.sel_ids, r.sel.size());
                std::sort(got.begin(), got.end());
                const bool ok = got == r.sel;
                if (!ok) ++bad;
                std::printf("  pos %lld block %d: the indexer's %zu cells %s\n", (long long) pos, il, r.sel.size(),
                            ok ? "match" : "*** DIFFER ***");
            }
            if (il >= g.n_dense_lead) {
                const vector<int32_t> got = download(ctx.eids, (size_t) g.n_expert_used);
                const bool ok = got == r.eids;
                if (!ok) ++bad;
                std::printf("  pos %lld block %d: experts %d %d %d %d %s\n", (long long) pos, il, r.eids[0], r.eids[1],
                            r.eids[2], r.eids[3], ok ? "match, in order" : "*** DIFFER ***");
            }
            if (!mla && !kda_state_ok(b, kd, pos, il)) ++bad;   // the carried state
        }
    }
    std::printf("\nglm_layer: %d failures\n", bad);
    for (void* p : g_free) cudaFree(p);
    cudaFree(d_x);
    cudaStreamDestroy(stream);
    if (bad) return 1;
    if (selftest) std::printf("glm_layer_parity OK\n");
    return 0;
}
