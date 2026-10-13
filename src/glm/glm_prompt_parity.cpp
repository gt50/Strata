// src/glm/glm_prompt_parity.cpp - GlmModel::prompt (many tokens a call: kda_scan_norm over them, mla_prompt_attn, the
// experts grouped on the host) on glm_model_parity's synthetic model, against glm_ref.hpp's transcription of llama.cpp's
// glm5-next graph, token by token.
//
//   A  a 40-token prompt from a new sequence in one call (the selection is every cell; pools complete inside it),
//      its MTP pass over positions 0 .. 38, the first draft at 39 (mtp_decode on the MTP rows the prompt filled),
//      then 4 decode tokens that continue from the state it left
//   B  the same prompt in three calls of 16, 16 and 8 tokens: the KDA state, the conv history and the pools carried
//      from one chunk to the next, and each call's last MTP row completed by the next call's first token
//   C  a 24-token prompt after 2100 cells of history (random caches and KDA state, as glm_layer_parity's), so the
//      indexer's selection is sparse: 512 pools of the 525 and the tail
//   D  scenario A's prompt with the experts streamed through a cache of 6 slots: every MoE layer runs in rounds of
//      what the cache holds, and the outputs are bitwise those of A
//
// Every block is checked for every token of a call from the streams the device gave it (the taps) and its latent
// history, the way glm_model_parity checks a decode token (glm_ref.hpp's verify: 1e-4, or a set of the reference's
// q8_1 roundings on a boundary taken the other way); the KDA blocks' conv history and state at the end of each call;
// the last indexer's cells and the last router's experts for every token; the head on the last token; and the MTP
// pass: its h rows the main model's at their positions, each row's eh_proj, block (verify), latent row, cells and
// experts.  A prompt
// call's projections and experts are the prompt products (glm_gemm.hpp: MMQ, the BF16 GEMM), so the reference rounds
// their activations as those do (glm_ref.hpp's g_prompt).
//
// `--bench [T]`: the prompt products' times at GLM-5.3-Flash's sizes; `--gemm`: the BF16 GEMM's accuracy by rows.
#include "glm_ref.hpp"

#include "strata/glm/glm_expert_cache.hpp"
#include "strata/glm/glm_model.hpp"
#include "strata/kernels/kpool.hpp"

#include <chrono>

namespace {

constexpr int64_t MAX_CTX = 2304;
constexpr int64_t CHUNK = 64;
int g_bad = 0;

size_t argmax(const vector<float>& v) { return (size_t) (std::max_element(v.begin(), v.end()) - v.begin()); }
size_t argmax(const vector<double>& v) { return (size_t) (std::max_element(v.begin(), v.end()) - v.begin()); }

struct Run {
    TestModel& tm;
    glm::GlmModel& model;
    glm::GlmTaps& taps;
    Ref r;
    int64_t fails = 0;   // this run's
};

/// The n tokens at pos0 that the last prompt() or decode() call ran, block by block from the taps.
void verify_call(Run& run, int64_t pos0, int64_t n) {
    g_prompt = n > 1;   // a prompt call took the prompt products (glm_gemm.hpp), a decode step the GEMVs
    const Glm5NextGeometry& g = run.tm.g;
    const int64_t D = g.n_embd, NL = g.n_layers;
    const vector<float> tp = download(run.taps.x, (size_t) ((NL + 1) * n * 4 * D));
    // the main layers' last selections and choices, as the call left them before its MTP pass
    const vector<int32_t> sel_dev = download(run.taps.sel, (size_t) (n * (g.idx_top_k + g.idx_kpool - 1)));
    const vector<int32_t> eid_dev = download(run.taps.eids, (size_t) (n * g.n_expert_used));
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
    g_prompt = 0;   // the head runs the last token alone, by the GEMV
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

/// The h_nextn rows of the last call's tokens [from, to), from its taps: rms_norm(mean of the streams) * output_norm.
vector<vector<double>> h_rows(Run& run, int64_t n, int64_t from, int64_t to) {
    const Glm5NextGeometry& g = run.tm.g;
    const int64_t D = g.n_embd, NL = g.n_layers;
    const vector<float> tp = download(run.taps.x + (size_t) (NL * n * 4 * D), (size_t) (n * 4 * D));
    vector<vector<double>> r;
    for (int64_t t = from; t < to; ++t) {
        vector<double> h((size_t) D, 0.0);
        for (int64_t d = 0; d < D; ++d) {
            for (int i = 0; i < 4; ++i) h[(size_t) d] += tp[(size_t) (t * 4 * D + i * D + d)];
            h[(size_t) d] /= 4;
        }
        rms_w(h, run.tm.out_norm, g.rms_eps);
        r.push_back(std::move(h));
    }
    return r;
}

/// The last MTP pass (a prompt's, or mtp_decode's one row): its rows' h are `want_h` (the main model's at their
/// positions), next tokens `next`; then for each row eh_proj from the device's h and embedding row, the block from the
/// device's eh_proj output (verify), its latent row, cells and experts.
void verify_mtp(Run& run, const vector<vector<double>>& want_h, const vector<int32_t>& next) {
    const Glm5NextGeometry& g = run.tm.g;
    const int64_t D = g.n_embd, R = run.taps.mtp_rows, pos0 = run.taps.mtp_pos0, MTP = g.n_layers;
    const int64_t before = g_bad;
    if (R != (int64_t) want_h.size() || R != (int64_t) next.size()) {
        ++g_bad;
        std::printf("  the MTP pass ran %lld rows from %lld, not %zu *** ***\n", (long long) R, (long long) pos0,
                    want_h.size());
        run.fails += g_bad - before;
        return;
    }
    g_prompt = R > 1;
    const vector<float> tm = download(run.taps.mtp, (size_t) (3 * R * D));
    const glm::GlmContext& ctx = run.model.context();
    const size_t cap = (size_t) (g.idx_top_k + g.idx_kpool - 1);
    const vector<int32_t> sel_dev = download(ctx.sel_ids, (size_t) R * cap);
    const vector<int32_t> eid_dev = download(ctx.eids, (size_t) (R * g.n_expert_used));
    auto row = [&](int64_t part, int64_t r) { return tm.begin() + (ptrdiff_t) ((part * R + r) * D); };
    Block& b = run.tm.blocks[(size_t) MTP];
    double worst_h = 0;
    for (int64_t r = 0; r < R; ++r) {
        const int64_t pos = pos0 + r;
        const vector<float> hd(row(0, r), row(0, r) + D);
        worst_h = std::max(worst_h, rel_l1(want_h[(size_t) r], hd));
        // eh_proj: [rms(e) enorm ; rms(h) hnorm] from the device's h row
        vector<double> e(run.tm.tok.w.begin() + (ptrdiff_t) (next[(size_t) r] * D),
                         run.tm.tok.w.begin() + (ptrdiff_t) ((next[(size_t) r] + 1) * D));
        vector<double> hh(hd.begin(), hd.end());
        rms_w(e, run.tm.enorm, g.rms_eps);
        rms_w(hh, run.tm.hnorm, g.rms_eps);
        e.insert(e.end(), hh.begin(), hh.end());
        const vector<float> x0(row(1, r), row(1, r) + D), x1(row(2, r), row(2, r) + D);
        if (!verify("MTP eh_proj", pos, MTP, [&] { return Fit{rel_l1(lin_ref(run.tm.eh, e), x0)}; }, [] {})) ++g_bad;
        latent_sync(b, pos);
        const Snap snap = save(b, run.r);
        if (!verify("MTP block", pos, MTP, [&] {
                vector<double> x(x0.begin(), x0.end());
                mtp_block_ref(b, g, MTP, pos, x, run.r);
                return Fit{rel_l1(x, x1), latent_row_far(b) == 0};
            }, [&] { restore(b, run.r, snap); }))
            ++g_bad;
        if (!latent_row_ok(b, pos, MTP)) ++g_bad;
        vector<int32_t> got_sel(sel_dev.begin() + (ptrdiff_t) (r * (int64_t) cap),
                                sel_dev.begin() + (ptrdiff_t) (r * (int64_t) cap + (int64_t) run.r.sel.size()));
        std::sort(got_sel.begin(), got_sel.end());
        const vector<int32_t> got_e(eid_dev.begin() + (ptrdiff_t) (r * g.n_expert_used),
                                    eid_dev.begin() + (ptrdiff_t) ((r + 1) * g.n_expert_used));
        if (got_sel != run.r.sel || got_e != run.r.eids) {
            ++g_bad;
            std::printf("  pos %lld MTP block: the cells or the experts *** DIFFER ***\n", (long long) pos);
        }
    }
    const bool h_ok = worst_h <= 1e-6;
    if (!h_ok) ++g_bad;
    std::printf("  MTP pass, positions %lld .. %lld: the h rows the main model's at them (worst rel %.3e)%s%s\n",
                (long long) pos0, (long long) (pos0 + R - 1), worst_h, h_ok ? "" : " *** ***",
                g_bad == before ? ", every row's eh_proj, block, latent row, cells and experts match" : "");
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
    for (int64_t il = 0; il < g.n_blocks(); ++il) {   // the MTP block's caches too
        Block& b = run.tm.blocks[(size_t) il];
        const glm::GlmLayerState& st = run.model.state(il);
        if (!g.is_mla[(size_t) il]) {
            const vector<float> hist = gvec((size_t) 3 * di * 3, 1.0f), stt = gvec((size_t) Dh * H * Dh, 0.05f);
            b.conv_hist.assign(hist.begin(), hist.end());
            vector<float> dev((size_t) stt.size());
            for (int hh = 0; hh < H; ++hh)   // the reference's [h][j][i]; the device's [h][i][j]
                for (int j = 0; j < Dh; ++j)
                    for (int i = 0; i < Dh; ++i) {
                        const float v = stt[((size_t) hh * Dh + j) * Dh + i];
                        b.kda_state[((size_t) hh * Dh + j) * Dh + i] = v;
                        dev[((size_t) hh * Dh + i) * Dh + j] = v;
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

/// `--gemm`: a BF16 projection (1024 x 1024) through GlmPromptGemm for T rows against double on the same BF16-rounded
/// activations: the largest difference over the largest value.
int gemm_check() {
    const int64_t K = 1024, N = 1024;
    const Lin L = make_lin(30, K, N);
    glm::GlmPromptGemm pg;
    std::string err;
    if (!pg.init(64, K, 1024, 256, 4, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    for (int T : {64, 40, 32, 24, 16, 12, 8, 4, 2}) {
        const vector<float> x = gvec((size_t) (T * K), 1.0f);
        float* dx = upload(x);
        float* dy = nullptr;
        check(cudaMalloc(&dy, (size_t) (T * N) * 4), "y");
        pg.linear(L.dev, dx, dy, T, nullptr, nullptr);
        check(cudaDeviceSynchronize(), "gemm");
        const vector<float> y = download(dy, (size_t) (T * N));
        double worst = 0, top = 0;
        for (int t = 0; t < T; ++t) {
            vector<double> xt(x.begin() + (ptrdiff_t) (t * K), x.begin() + (ptrdiff_t) ((t + 1) * K));
            for (auto& v : xt) v = bf16_val(bf16_bits((float) v));
            for (int64_t o = 0; o < N; ++o) {
                double a = 0;
                for (int64_t i = 0; i < K; ++i) a += (double) L.w[(size_t) (o * K + i)] * xt[(size_t) i];
                worst = std::max(worst, std::fabs(a - y[(size_t) (t * N + o)]));
                top = std::max(top, std::fabs(a));
            }
        }
        std::printf("  BF16 GEMM, %2d rows: largest difference / largest value %.3e\n", T, worst / top);
        cudaFree(dx);
        cudaFree(dy);
    }
    return 0;
}

/// `--bench [T]`: a chunk of T tokens (256) at GLM-5.3-Flash's sizes, the prompt products against the GEMV path they replace.
/// Projections 4096 -> 8192 (KDA's q / k / v) in IQ4_XS, Q8_0 and BF16; one MoE layer, 288 IQ4_XS experts of
/// 4096 x 2048, 8 a token (random choices), through the grouped kernel and through MMQ.  Times back to back (one
/// synchronization around 10 runs).
int bench(int T) {
    const int64_t D = 4096, N = 8192, F = 2048, NE = 288, KU = 8;
    std::string err;
    glm::GlmPromptGemm pg;
    if (!pg.init(T, N, D, F, KU, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    float* x = upload(gvec((size_t) (T * N), 1.0f));
    float* y = nullptr;
    check(cudaMalloc(&y, (size_t) (T * N) * 4), "y");
    void* q8 = nullptr;
    check(cudaMalloc(&q8, k::glm_linear_scratch_bytes(N, T)), "q8");
    auto time = [&](auto fn) {
        fn();
        check(cudaStreamSynchronize(s), "warm");
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) fn();
        check(cudaStreamSynchronize(s), "timed");
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 10;
    };
    std::printf("prompt products at GLM-5.3-Flash's sizes, %d tokens (Radeon 8060S):\n", T);
    for (int type : {23, 8, 30}) {
        const Lin L = make_lin(type, D, N);
        const double gemv = time([&] { k::glm_linear(L.dev, x, y, T, q8, s); });
        const double prod = time([&] { pg.linear(L.dev, x, y, T, q8, s); });
        std::printf("  projection 4096 -> 8192 %-6s: GEMV %7.2f ms, %s %6.2f ms (%5.1fx; %.0f tokens/s through it)\n",
                    type == 23 ? "IQ4_XS" : type == 8 ? "Q8_0" : "BF16", gemv,
                    type == 30 ? "BF16 GEMM" : "MMQ      ", prod, gemv / prod, T / prod * 1000);
    }
    // one MoE layer: 288 experts in one arena, the blob of one random expert repeated
    const k::NativeExpertLayout el = k::native_expert_layout(23, 23, D, F);
    const uint64_t slot = (el.bytes + 255) & ~(uint64_t) 255;
    uint8_t* arena = nullptr;
    check(cudaMalloc(&arena, slot * (uint64_t) NE), "experts");
    {
        const Lin g = make_lin(23, D, F), u = make_lin(23, D, F), dn = make_lin(23, F, D);
        for (int64_t e = 0; e < NE; ++e) {
            uint8_t* b = arena + slot * (uint64_t) e;
            check(cudaMemcpy(b, g.dev.data, el.up_off, cudaMemcpyDeviceToDevice), "gate");
            check(cudaMemcpy(b + el.up_off, u.dev.data, el.up_off, cudaMemcpyDeviceToDevice), "up");
            check(cudaMemcpy(b + el.down_off, dn.dev.data, el.bytes - el.down_off, cudaMemcpyDeviceToDevice), "down");
        }
    }
    // random routing, grouped by expert (entries t * 8 + j)
    vector<vector<int32_t>> by_e((size_t) NE);
    for (int t = 0; t < T; ++t) {
        vector<int32_t> pick;
        while ((int64_t) pick.size() < KU) {
            const int32_t e = (int32_t) (unit() * (float) NE) % (int32_t) NE;
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        for (int j = 0; j < KU; ++j) by_e[(size_t) pick[(size_t) j]].push_back(t * (int32_t) KU + j);
    }
    vector<int32_t> starts{0}, tok, ent, bounds{0};
    vector<unsigned long long> ptrs;
    int max_rows = 0, groups = 0;
    for (int64_t e = 0; e < NE; ++e) {
        for (int32_t en : by_e[(size_t) e]) { tok.push_back(en / (int32_t) KU); ent.push_back(en); }
        bounds.push_back((int32_t) tok.size());
        max_rows = std::max(max_rows, (int) by_e[(size_t) e].size());
        if (!by_e[(size_t) e].empty()) {
            ++groups;
            starts.push_back((int32_t) tok.size());
            ptrs.push_back((unsigned long long) (uintptr_t) (arena + slot * (uint64_t) e));
        }
    }
    int32_t *d_start = upload(starts), *d_tok = upload(tok), *d_ent = upload(ent), *d_bounds = upload(bounds);
    unsigned long long* d_ptrs = upload(ptrs);
    int32_t* d_ng = upload(vector<int32_t>{groups});
    void* xq = nullptr;
    void* scr = nullptr;
    float* parts = nullptr;
    check(cudaMalloc(&xq, k::glm_linear_scratch_bytes(D, T)), "xq");
    check(cudaMalloc(&scr, k::native_expert_scratch_bytes(T * KU, F)), "scratch");
    check(cudaMalloc(&parts, (size_t) (T * KU * D) * 4), "parts");
    const double grouped = time([&] {
        k::quantize_q8_1_rows(x, T, D, xq, s);
        k::native_expert_grouped_clamped(el, d_ptrs, d_start, d_ng, d_ent, d_tok, groups, T * KU, xq, scr, parts, 10.0f,
                                         s);
    });
    const double mmq = time([&] {
        pg.experts(el, arena, slot, (int) NE, d_bounds, d_tok, d_ent, (int) (T * KU), max_rows, x, parts, 10.0f, s);
    });
    const double gb = (double) groups * (double) el.bytes / 1e9;
    std::printf("  one MoE layer, %d of 288 experts used (%.2f GB of blobs): grouped kernel %7.2f ms, MMQ %6.2f ms "
                "(%4.1fx)\n", groups, gb, grouped, mmq, grouped / mmq);
    cudaStreamDestroy(s);
    return 0;
}

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else if (std::string(argv[i]) == "--bench") return bench(i + 1 < argc ? std::atoi(argv[i + 1]) : 256);
        else if (std::string(argv[i]) == "--gemm") return gemm_check();
        else { std::fprintf(stderr, "usage: glm_prompt_parity [--selftest | --gemm]\n"); return 2; }
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
    glm::GlmTaps gtaps;
    gtaps.x = taps;
    check(cudaMalloc(&gtaps.sel, (size_t) (CHUNK * (g.idx_top_k + g.idx_kpool - 1)) * 4), "sel taps");
    check(cudaMalloc(&gtaps.eids, (size_t) (CHUNK * g.n_expert_used) * 4), "eids taps");
    check(cudaMalloc(&gtaps.mtp, (size_t) (3 * (CHUNK + 1) * D) * sizeof(float)), "mtp taps");
    model.set_taps(&gtaps);
    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "stream");
    Run run{tm, model, gtaps, Ref{g, {}, {}}};
    g_quiet = true;   // a line for each token and block only where it needed more than the rounding as made
    vector<int32_t> tokens(48);
    for (auto& t : tokens) t = (int32_t) (unit() * (float) g.n_vocab) % (int32_t) g.n_vocab;
    const vector<int32_t> prompt40(tokens.begin(), tokens.begin() + 40);

    std::printf("A: a 40-token prompt from a new sequence, one call; then 4 decode tokens\n");
    fresh(run, stream);
    prompt(run, prompt40, 0, stream);
    verify_call(run, 0, 40);
    verify_mtp(run, h_rows(run, 40, 0, 39), vector<int32_t>(tokens.begin() + 1, tokens.begin() + 40));
    const vector<float> logits_a = download(model.logits(), (size_t) g.n_vocab);
    {   // the first draft where the prompt ends: position 39 with token 40, on the MTP rows the prompt filled
        const vector<float> h39 = download(model.h_nextn(), (size_t) D);
        if (!model.mtp_decode(tokens[40], 39, model.h_nextn(), stream, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        check(cudaStreamSynchronize(stream), "mtp_decode");
        verify_mtp(run, {vector<double>(h39.begin(), h39.end())}, {tokens[40]});
    }
    for (int64_t p = 40; p < 44; ++p) {
        if (!model.decode(tokens[(size_t) p], p, stream, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        check(cudaStreamSynchronize(stream), "decode");
        verify_call(run, p, 1);
    }
    const int64_t fails_a = run.fails;

    std::printf("\nB: the same prompt in calls of 16, 16 and 8 tokens\n");
    run.fails = 0;
    fresh(run, stream);
    vector<float> h_last;   // the last call's h_nextn: its last position's MTP row waits for the next call
    for (int64_t c0 : {0, 16, 32}) {
        const int64_t nc = c0 < 32 ? 16 : 8;
        prompt(run, vector<int32_t>(prompt40.begin() + c0, prompt40.begin() + c0 + nc), c0, stream);
        verify_call(run, c0, nc);
        vector<vector<double>> want;
        vector<int32_t> next;
        if (c0 > 0) {
            want.emplace_back(h_last.begin(), h_last.end());
            next.push_back(prompt40[(size_t) c0]);
        }
        for (auto& h : h_rows(run, nc, 0, nc - 1)) want.push_back(h);
        for (int64_t t = 1; t < nc; ++t) next.push_back(prompt40[(size_t) (c0 + t)]);
        verify_mtp(run, want, next);
        h_last = download(model.h_nextn(), (size_t) D);
    }
    const int64_t fails_b = run.fails;

    std::printf("\nC: a 24-token prompt after 2100 cells of history (a sparse selection)\n");
    run.fails = 0;
    history(run, 2100, stream);
    prompt(run, vector<int32_t>(tokens.begin(), tokens.begin() + 24), 2100, stream);
    verify_call(run, 2100, 24);
    verify_mtp(run, h_rows(run, 24, 0, 23), vector<int32_t>(tokens.begin() + 1, tokens.begin() + 24));
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
