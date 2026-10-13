// src/glm/glm_prompt_parity.cpp - GlmModel::prompt (many tokens a call: kda_scan_norm_chunked, mla_prompt_attn, the
// experts grouped on the host) on glm_model_parity's synthetic model, against glm_ref.hpp's transcription of llama.cpp's
// glm5-next graph, token by token.
//
//   A  a 40-token prompt from a new sequence in one call (the selection is every cell; pools complete inside it),
//      then 4 decode tokens that continue from the state it left
//   B  the same prompt in three calls of 16, 16 and 8 tokens: the KDA state, the conv history and the pools carried
//      from one chunk to the next
//   C  a 24-token prompt after 2100 cells of history (random caches and KDA state, as glm_layer_parity's), so the
//      indexer's selection is sparse: 512 pools of the 525 and the tail
//   D  scenario A's prompt with the experts streamed through a cache of 6 slots: every MoE layer runs in rounds of
//      what the cache holds, and the outputs are bitwise those of A
//
// Every block is checked for every token of a call from the streams the device gave it (the taps) and its latent
// history, the way glm_model_parity checks a decode token (glm_ref.hpp's verify: 1e-4, or a set of the reference's
// q8_1 roundings on a boundary taken the other way); the KDA blocks' conv history and state at the end of each call;
// the last indexer's cells and the last router's experts for every token; the head on the last token.
#include "glm_ref.hpp"

#include "strata/glm/glm_expert_cache.hpp"
#include "strata/glm/glm_model.hpp"
#include "strata/kernels/kpool.hpp"

namespace {

constexpr int64_t MAX_CTX = 2304;
constexpr int64_t CHUNK = 64;
int g_bad = 0;

size_t argmax(const vector<float>& v) { return (size_t) (std::max_element(v.begin(), v.end()) - v.begin()); }
size_t argmax(const vector<double>& v) { return (size_t) (std::max_element(v.begin(), v.end()) - v.begin()); }

struct Run {
    TestModel& tm;
    glm::GlmModel& model;
    float* taps;
    Ref r;
    int64_t fails = 0;   // this run's
};

/// The n tokens at pos0 that the last prompt() or decode() call ran, block by block from the taps.
void verify_call(Run& run, int64_t pos0, int64_t n) {
    const Glm5NextGeometry& g = run.tm.g;
    const int64_t D = g.n_embd, NL = g.n_layers;
    const vector<float> tp = download(run.taps, (size_t) ((NL + 1) * n * 4 * D));
    const glm::GlmContext& ctx = run.model.context();
    const vector<int32_t> sel_dev = download(ctx.sel_ids, (size_t) (n * (g.idx_top_k + g.idx_kpool - 1)));
    const vector<int32_t> eid_dev = download(ctx.eids, (size_t) (n * g.n_expert_used));
    int64_t last_full = -1, last_moe = -1;
    for (int64_t il = 0; il < NL; ++il) {
        if (g.is_mla[(size_t) il] && full_idx(g, il)) last_full = il;
        if (il >= g.n_dense_lead) last_moe = il;
    }
    vector<KdaDev> kd((size_t) NL);
    for (int64_t il = 0; il < NL; ++il)
        if (!g.is_mla[(size_t) il]) kd[(size_t) il] = kda_dev(run.tm.blocks[(size_t) il], g);
    auto row = [&](int64_t stage, int64_t t) { return tp.begin() + (ptrdiff_t) (((stage * n) + t) * 4 * D); };
    const int64_t before = g_bad;
    for (int64_t t = 0; t < n; ++t) {
        const int64_t pos = pos0 + t;
        for (int64_t il = 0; il < NL; ++il) {
            Block& b = run.tm.blocks[(size_t) il];
            const bool mla = g.is_mla[(size_t) il] != 0, last = t == n - 1;
            if (mla) latent_sync(b, pos);
            const vector<float> got(row(il + 1, t), row(il + 1, t) + 4 * D);
            const Snap snap = save(b, run.r);
            const bool ok = verify("the four streams", pos, il, [&] {
                vector<double> X(row(il, t), row(il, t) + 4 * D);
                block_ref(b, g, il, pos, X, run.r);
                double rh, rs;
                // the KDA state is on the device only as the call left it: compared at its last token
                const bool rest = mla ? latent_row_far(b) == 0 : (!last || kda_close(b, kd[(size_t) il], rh, rs));
                return Fit{rel_l1(X, got), rest};
            }, [&] { restore(b, run.r, snap); });
            if (!ok) ++g_bad;
            if (mla && !latent_row_ok(b, pos, il)) ++g_bad;
            if (!mla && last && !kda_state_ok(b, kd[(size_t) il], pos, il)) ++g_bad;
            if (il == last_full) {
                const size_t cap = (size_t) (g.idx_top_k + g.idx_kpool - 1);
                vector<int32_t> got_sel(sel_dev.begin() + (ptrdiff_t) (t * (int64_t) cap),
                                        sel_dev.begin() + (ptrdiff_t) (t * (int64_t) cap + (int64_t) run.r.sel.size()));
                std::sort(got_sel.begin(), got_sel.end());
                if (got_sel != run.r.sel) {
                    ++g_bad;
                    std::printf("  pos %lld block %lld: the indexer's %zu cells *** DIFFER ***\n", (long long) pos,
                                (long long) il, run.r.sel.size());
                }
            }
            if (il == last_moe) {
                const vector<int32_t> got_e(eid_dev.begin() + (ptrdiff_t) (t * g.n_expert_used),
                                            eid_dev.begin() + (ptrdiff_t) ((t + 1) * g.n_expert_used));
                if (got_e != run.r.eids) {
                    ++g_bad;
                    std::printf("  pos %lld block %lld: experts *** DIFFER ***\n", (long long) pos, (long long) il);
                }
            }
        }
    }
    // the head on the last token
    vector<double> hm((size_t) D, 0.0);
    for (int64_t d = 0; d < D; ++d) {
        for (int i = 0; i < 4; ++i) hm[(size_t) d] += row(NL, n - 1)[(ptrdiff_t) (i * D + d)];
        hm[(size_t) d] /= 4;
    }
    rms_w(hm, run.tm.out_norm, g.rms_eps);
    const vector<float> h = download(run.model.h_nextn(), (size_t) D), lg = download(run.model.logits(), (size_t) g.n_vocab);
    const double rh = rel_l1(hm, h);
    vector<double> lg_ref;
    const bool lg_ok = verify("logits", pos0 + n - 1, -1, [&] {
        lg_ref = lin_ref(run.tm.out, vector<double>(h.begin(), h.end()));
        return Fit{rel_l1(lg_ref, lg)};
    }, [] {});
    const size_t am = argmax(lg), am_ref = argmax(lg_ref);
    const bool am_ok = am == am_ref || std::fabs(lg_ref[am] - lg_ref[am_ref]) < 1e-4 * std::fabs(lg_ref[am_ref]);
    if (rh > 1e-6 || !am_ok || !lg_ok) ++g_bad;
    std::printf("  pos %lld head: h rel %.3e%s, argmax %zu %s\n", (long long) (pos0 + n - 1), rh,
                rh <= 1e-6 ? "" : " *** over 1e-6 ***", am, am_ok ? "matches" : "*** DIFFERS ***");
    run.fails += g_bad - before;
}

/// A fresh sequence on both sides: the device's KDA state zeroed, the reference's state too.
void fresh(Run& run, void* stream) {
    run.model.reset(stream);
    for (int64_t il = 0; il < run.tm.g.n_blocks(); ++il) ref_state_zero(run.tm.blocks[(size_t) il], run.tm.g, il, MAX_CTX);
    run.r.sel.clear();
    run.r.eids.clear();
}

/// p0 cells of random history in every block's caches and a random KDA state, on both sides.
void history(Run& run, int64_t p0, void* stream) {
    fresh(run, stream);
    const Glm5NextGeometry& g = run.tm.g;
    const int H = (int) g.n_head, Dh = 128, di = H * Dh, L = 512;
    for (int64_t il = 0; il < g.n_layers; ++il) {
        Block& b = run.tm.blocks[(size_t) il];
        const glm::GlmLayerState& st = run.model.state(il);
        if (!g.is_mla[(size_t) il]) {
            const vector<float> hist = gvec((size_t) 3 * di * 3, 1.0f), stt = gvec((size_t) Dh * H * Dh, 0.05f);
            b.conv_hist.assign(hist.begin(), hist.end());
            vector<float> dev((size_t) stt.size());
            for (int hh = 0; hh < H; ++hh)   // the reference's [h][j][i]; the device's [i][h][j]
                for (int j = 0; j < Dh; ++j)
                    for (int i = 0; i < Dh; ++i) {
                        const float v = stt[((size_t) hh * Dh + j) * Dh + i];
                        b.kda_state[((size_t) hh * Dh + j) * Dh + i] = v;
                        dev[((size_t) i * H + hh) * Dh + j] = v;
                    }
            check(cudaMemcpy(st.conv_hist, hist.data(), hist.size() * 4, cudaMemcpyHostToDevice), "hist");
            check(cudaMemcpy(st.kda_state, dev.data(), dev.size() * 4, cudaMemcpyHostToDevice), "state");
            continue;
        }
        for (size_t i = 0; i < (size_t) p0 * L; ++i) b.latent[i] = k::f16_from_f32(gauss());
        check(cudaMemcpy(st.latent, b.latent.data(), (size_t) p0 * L * 2, cudaMemcpyHostToDevice), "latent");
        if (full_idx(g, il)) {
            vector<float> cache((size_t) p0 * 256);
            for (auto& v : cache) v = gauss();
            std::copy(cache.begin(), cache.end(), b.idx_cache.begin());
            for (int64_t p = 0; p < p0 / 4; ++p) pool_ref(b, p);
            check(cudaMemcpy(st.idx_cache, cache.data(), cache.size() * 4, cudaMemcpyHostToDevice), "idx cache");
            k::kpool_pool(st.idx_cache, b.w.idx_ape, st.pooled, 0, p0 / 4, nullptr, stream);
        }
    }
    check(cudaStreamSynchronize((cudaStream_t) stream), "history");
}

void prompt(Run& run, const vector<int32_t>& tokens, int64_t pos0, void* stream) {
    std::string err;
    if (!run.model.prompt(tokens.data(), (int64_t) tokens.size(), pos0, stream, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        std::exit(1);
    }
    check(cudaStreamSynchronize((cudaStream_t) stream), "prompt");
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: glm_prompt_parity [--selftest]\n"); return 2; }
    }
    TestModel tm = build_test_model(MAX_CTX);
    const Glm5NextGeometry& g = tm.g;
    const int64_t D = g.n_embd, NL = g.n_layers;
    glm::GlmModel model;
    std::string err;
    if (!model.init(g, tm.w, MAX_CTX, err, nullptr, CHUNK)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    for (int64_t il = 0; il < g.n_blocks(); ++il) tm.blocks[(size_t) il].st = model.state(il);
    float* taps = nullptr;
    check(cudaMalloc(&taps, (size_t) ((NL + 1) * CHUNK * 4 * D) * sizeof(float)), "taps");
    model.set_taps(taps);
    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "stream");
    Run run{tm, model, taps, Ref{g, {}, {}}};
    g_quiet = true;   // a line for each token and block only where it needed more than the rounding as made
    vector<int32_t> tokens(48);
    for (auto& t : tokens) t = (int32_t) (unit() * (float) g.n_vocab) % (int32_t) g.n_vocab;
    const vector<int32_t> prompt40(tokens.begin(), tokens.begin() + 40);

    std::printf("A: a 40-token prompt from a new sequence, one call; then 4 decode tokens\n");
    fresh(run, stream);
    prompt(run, prompt40, 0, stream);
    verify_call(run, 0, 40);
    const vector<float> logits_a = download(model.logits(), (size_t) g.n_vocab);
    for (int64_t p = 40; p < 44; ++p) {
        if (!model.decode(tokens[(size_t) p], p, stream, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        check(cudaStreamSynchronize(stream), "decode");
        verify_call(run, p, 1);
    }
    const int64_t fails_a = run.fails;

    std::printf("\nB: the same prompt in calls of 16, 16 and 8 tokens\n");
    run.fails = 0;
    fresh(run, stream);
    for (int64_t c0 : {0, 16, 32}) {
        const int64_t nc = c0 < 32 ? 16 : 8;
        prompt(run, vector<int32_t>(prompt40.begin() + c0, prompt40.begin() + c0 + nc), c0, stream);
        verify_call(run, c0, nc);
    }
    const int64_t fails_b = run.fails;

    std::printf("\nC: a 24-token prompt after 2100 cells of history (a sparse selection)\n");
    run.fails = 0;
    history(run, 2100, stream);
    prompt(run, vector<int32_t>(tokens.begin(), tokens.begin() + 24), 2100, stream);
    verify_call(run, 2100, 24);
    const int64_t fails_c = run.fails;

    std::printf("\nD: scenario A's prompt with the experts streamed through a cache of 6 slots\n");
    bool same = false;
    uint64_t lookups = 0, misses = 0;
    {
        glm::GlmModelWeights ws = tm.w;
        vector<glm::GlmExpertSource> src((size_t) g.n_blocks());
        vector<vector<uint8_t>> keep;   // the stacked tensors the cache reads
        for (int64_t il = 0; il < g.n_blocks(); ++il) {
            if (il < g.n_dense_lead) continue;
            // the cache copies from host memory: the blobs, as a loader would read them from the file
            const k::NativeExpertLayout& el = tm.blocks[(size_t) il].w.expert_layout;
            const vector<unsigned long long> ptrs = download(tm.blocks[(size_t) il].w.expert_ptrs, (size_t) g.n_expert);
            vector<uint8_t> gu, up, dn;
            for (unsigned long long a : ptrs) {
                vector<uint8_t> blob(el.bytes);
                check(cudaMemcpy(blob.data(), (const void*) (uintptr_t) a, el.bytes, cudaMemcpyDeviceToHost), "blob");
                gu.insert(gu.end(), blob.begin(), blob.begin() + (ptrdiff_t) el.up_off);
                up.insert(up.end(), blob.begin() + (ptrdiff_t) el.up_off, blob.begin() + (ptrdiff_t) el.down_off);
                dn.insert(dn.end(), blob.begin() + (ptrdiff_t) el.down_off, blob.end());
            }
            src[(size_t) il].layout = el;
            src[(size_t) il].gate = gu.data();   // a moved vector keeps its buffer
            src[(size_t) il].up = up.data();
            src[(size_t) il].down = dn.data();
            keep.push_back(std::move(gu));
            keep.push_back(std::move(up));
            keep.push_back(std::move(dn));
            ws.blocks[(size_t) il].expert_ptrs = nullptr;
        }
        glm::GlmExpertCache cache;
        if (!cache.init(src, g.n_expert, 6, true, 4, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        glm::GlmModel streamed;
        if (!streamed.init(g, ws, MAX_CTX, err, &cache, CHUNK)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        streamed.reset(stream);
        if (!streamed.prompt(prompt40.data(), 40, 0, stream, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        check(cudaStreamSynchronize(stream), "streamed prompt");
        same = download(streamed.logits(), (size_t) g.n_vocab) == logits_a;
        lookups = cache.stats().hits + cache.stats().misses;
        misses = cache.stats().misses;
    }
    std::printf("  logits bitwise scenario A's: %s (%llu lookups in rounds of 6 slots, %llu misses)\n",
                same ? "yes" : "*** NO ***", (unsigned long long) lookups, (unsigned long long) misses);
    if (!same) ++g_bad;

    std::printf("\nglm_prompt: A %lld, B %lld, C %lld failures; %d in all\n", (long long) fails_a, (long long) fails_b,
                (long long) fails_c, g_bad);
    for (void* p : g_free) cudaFree(p);
    cudaFree(taps);
    cudaStreamDestroy(stream);
    if (g_bad) return 1;
    if (selftest) std::printf("glm_prompt_parity OK\n");
    return 0;
}
