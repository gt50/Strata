// src/kernels/kpool_parity.cpp - the k-pool indexer (kpool.hpp) against a double-precision transcription of
// llama.cpp's build_kpool_select, and GLM's selection rule through qsa_block_topk.
//
//   1. kpool_key_store: ggml_norm (LayerNorm) with weight and bias, the gate copied; also through a step
//   2. kpool_pool: the softmax over a pool's 4 members per channel, of gate + ape; also through a step, which pools
//      only when the token completes one
//   3. kpool_scores: sum_h w_h relu(iq_h . pooled_b) / sqrt(128 x 32), three queries with their own steps; the tail
//      block forced in or out
//   4. the selection: kpool_scores -> qsa_block_topk with kpool_step's width, against GLM's rule written out - the
//      top min(n_pool, 512) pools by score (ties to the lower pool), their 4 cells each, plus the tail's cells - at
//      context lengths on both sides of every boundary (2048, 2051, 2052, ...), with and without the tail
//
// Rival readings are shown to differ first: RMSNorm for LayerNorm, the pooling softmax over channels instead of
// members, ape read [d][m], and QSA's own width min(n_kv, 2051), which would add cells of a 513th pool.
// `--bench` times scores and selection for one query at long contexts.
#include "strata/kernels/kpool.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int D = strata::kernels::kKpoolDim;
constexpr int NH = strata::kernels::kKpoolHeads;
constexpr int KP = strata::kernels::kKpool;
constexpr int64_t kTopK = 2048;           // indexer.top_k
constexpr float kEps = 1e-6f;             // attention.layer_norm_epsilon
constexpr int64_t kCap = kTopK + KP - 1;  // the largest selection: 512 pools and a 3-cell tail

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

double rel_l1(const std::vector<float>& want, const std::vector<float>& got) {
    double d = 0, m = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        d += std::fabs((double) want[i] - (double) got[i]);
        m += std::fabs((double) want[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

template <class T> T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    check(cudaMalloc(&p, v.size() * sizeof(T)), "cudaMalloc");
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template <class T> std::vector<T> download(const T* p, size_t n) {
    std::vector<T> v(n);
    check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

int g_bad = 0;
void expect(bool ok, const char* what, const std::string& detail = "") {
    std::printf("  %-60s %s%s\n", what, ok ? "yes" : "*** NO ***", detail.c_str());
    if (!ok) ++g_bad;
}
void expect_apart(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    const double rel = rel_l1(a, b);
    char buf[48];
    std::snprintf(buf, sizeof buf, " (%.2f%% apart)", rel * 100);
    expect(rel > 0.05, what, buf);
}
void expect_close(const char* what, const std::vector<float>& want, const std::vector<float>& got, double tol) {
    const double rel = rel_l1(want, got);
    const bool ok = rel <= tol;
    std::printf("  %-60s rel %.3e %s\n", what, rel, ok ? "" : "*** over tolerance ***");
    if (!ok) ++g_bad;
}

/// ggml_norm then weight and bias; `rms` is the rival (no centring).
std::vector<float> ref_norm(const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>& b,
                            int n_tok, bool rms = false) {
    std::vector<float> y(x.size());
    for (int t = 0; t < n_tok; ++t) {
        const float* r = &x[(size_t) t * D];
        double mean = 0;
        if (!rms) {
            for (int d = 0; d < D; ++d) mean += r[d];
            mean /= D;
        }
        double var = 0;
        for (int d = 0; d < D; ++d) var += (r[d] - mean) * (r[d] - mean);
        var /= D;
        for (int d = 0; d < D; ++d)
            y[(size_t) t * D + d] = (float) ((r[d] - mean) / std::sqrt(var + kEps) * w[(size_t) d] + b[(size_t) d]);
    }
    return y;
}

/// pooled[b][d] = sum_m softmax_m(gate[4b+m][d] + ape[m][d]) key[4b+m][d] from cache rows (key | gate).  Rivals:
/// `over_channels` soft-maxes each member's 128 channels instead; `ape_dm` reads ape as [d][m].
std::vector<float> ref_pool(const std::vector<float>& cache, const std::vector<float>& ape, int64_t b0, int64_t n,
                            bool over_channels = false, bool ape_dm = false) {
    std::vector<float> out((size_t) n * D);
    for (int64_t b = b0; b < b0 + n; ++b) {
        double lg[KP][D];
        for (int m = 0; m < KP; ++m)
            for (int d = 0; d < D; ++d)
                lg[m][d] = cache[(size_t) (b * KP + m) * 2 * D + D + d] + (double) ape[ape_dm ? (size_t) d * KP + m : (size_t) m * D + d];
        double p[KP][D];
        if (!over_channels) {
            for (int d = 0; d < D; ++d) {
                double mx = -INFINITY, s = 0;
                for (int m = 0; m < KP; ++m) mx = std::max(mx, lg[m][d]);
                for (int m = 0; m < KP; ++m) s += std::exp(lg[m][d] - mx);
                for (int m = 0; m < KP; ++m) p[m][d] = std::exp(lg[m][d] - mx) / s;
            }
        } else {
            for (int m = 0; m < KP; ++m) {
                double mx = -INFINITY, s = 0;
                for (int d = 0; d < D; ++d) mx = std::max(mx, lg[m][d]);
                for (int d = 0; d < D; ++d) s += std::exp(lg[m][d] - mx);
                for (int d = 0; d < D; ++d) p[m][d] = std::exp(lg[m][d] - mx) / s;
            }
        }
        for (int d = 0; d < D; ++d) {
            double acc = 0;
            for (int m = 0; m < KP; ++m) acc += p[m][d] * cache[(size_t) (b * KP + m) * 2 * D + d];
            out[(size_t) (b - b0) * D + d] = (float) acc;
        }
    }
    return out;
}

/// GLM's selection written out: the top min(n_bid, top_k / 4) pools by score (descending, ties to the lower pool),
/// their cells, plus the tail's cells; ascending.
std::vector<int32_t> ref_select(const std::vector<float>& scores, int64_t n_kv, bool select_tail) {
    const int64_t n_bid = n_kv / KP;
    std::vector<int64_t> order((size_t) n_bid);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return scores[(size_t) a] > scores[(size_t) b]; });
    const int64_t pools = std::min<int64_t>(n_bid, kTopK / KP);
    std::vector<int32_t> cells;
    for (int64_t i = 0; i < pools; ++i)
        for (int m = 0; m < KP; ++m) cells.push_back((int32_t) (order[(size_t) i] * KP + m));
    if (select_tail)
        for (int64_t c = n_bid * KP; c < n_kv; ++c) cells.push_back((int32_t) c);
    std::sort(cells.begin(), cells.end());
    return cells;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else if (std::string(argv[i]) == "--bench") bench = true;
        else { std::fprintf(stderr, "usage: kpool_parity [--selftest] [--bench]\n"); return 2; }
    }
    using namespace strata::kernels;
    std::mt19937 rng(4097);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const float w_scale = 1.0f / std::sqrt((float) (D * NH));

    // ================= 1. keys and gates into the cache =================
    constexpr int T = 24;   // six pools' worth of cells
    std::vector<float> kraw((size_t) T * D), gate((size_t) T * D), lnw((size_t) D), lnb((size_t) D), ape((size_t) KP * D);
    for (auto& x : kraw) x = 2.0f * gauss(rng) + 0.7f;   // an offset, so centring matters
    for (auto& x : gate) x = gauss(rng);
    for (auto& x : lnw) x = 1.0f + 0.2f * gauss(rng);
    for (auto& x : lnb) x = 0.1f * gauss(rng);
    for (auto& x : ape) x = gauss(rng);
    std::printf("kpool_key_store, %d tokens:\n", T);
    const std::vector<float> want_k = ref_norm(kraw, lnw, lnb, T);
    expect_apart("rival: RMSNorm for LayerNorm", want_k, ref_norm(kraw, lnw, lnb, T, true));
    float *d_kraw = upload(kraw), *d_gate = upload(gate), *d_lnw = upload(lnw), *d_lnb = upload(lnb), *d_ape = upload(ape);
    float* d_cache = nullptr;
    check(cudaMalloc(&d_cache, (size_t) T * 2 * D * 4), "cache");
    kpool_key_store(d_kraw, d_gate, d_lnw, d_lnb, kEps, d_cache, 0, T, nullptr, nullptr);
    check(cudaDeviceSynchronize(), "key_store");
    std::vector<float> cache = download(d_cache, (size_t) T * 2 * D);
    {
        std::vector<float> got_k((size_t) T * D), got_g((size_t) T * D);
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D; ++d) {
                got_k[(size_t) t * D + d] = cache[(size_t) t * 2 * D + d];
                got_g[(size_t) t * D + d] = cache[(size_t) t * 2 * D + D + d];
            }
        expect_close("keys: LayerNorm * weight + bias", want_k, got_k, 1e-6);
        expect(got_g == gate, "gates copied unchanged");
        // the same token through a step: row step[pos]
        int32_t st[kStepCount];
        kpool_step(13, kTopK, true, st);
        int32_t* d_st = upload(std::vector<int32_t>(st, st + kStepCount));
        float* d_c2 = nullptr;
        check(cudaMalloc(&d_c2, (size_t) T * 2 * D * 4), "c2");
        check(cudaMemset(d_c2, 0, (size_t) T * 2 * D * 4), "c2 0");
        kpool_key_store(d_kraw + 13 * D, d_gate + 13 * D, d_lnw, d_lnb, kEps, d_c2, 0, 1, d_st, nullptr);
        check(cudaDeviceSynchronize(), "key_store step");
        const std::vector<float> c2 = download(d_c2, (size_t) T * 2 * D);
        bool same = true, others_zero = true;
        for (int i = 0; i < 2 * D; ++i) same = same && c2[(size_t) 13 * 2 * D + i] == cache[(size_t) 13 * 2 * D + i];
        for (size_t i = 0; i < c2.size(); ++i)
            if (i / (2 * D) != 13) others_zero = others_zero && c2[i] == 0.0f;
        expect(same && others_zero, "through a step: row step[pos] only, the same bits");
        cudaFree(d_st); cudaFree(d_c2);
    }

    // ================= 2. pooling =================
    std::printf("\nkpool_pool, pools 0 .. 5:\n");
    {
        const std::vector<float> want = ref_pool(cache, ape, 0, 6);
        expect_apart("rival: softmax over the channels, not the 4 members", want, ref_pool(cache, ape, 0, 6, true));
        expect_apart("rival: ape read [d][m]", want, ref_pool(cache, ape, 0, 6, false, true));
        float* d_pooled = nullptr;
        check(cudaMalloc(&d_pooled, (size_t) 6 * D * 4), "pooled");
        kpool_pool(d_cache, d_ape, d_pooled, 0, 6, nullptr, nullptr);
        check(cudaDeviceSynchronize(), "pool");
        const std::vector<float> got = download(d_pooled, (size_t) 6 * D);
        expect_close("pooled keys", want, got, 1e-6);
        // through steps: the token at pos 19 completes pool 4 (cells 16..19); pos 18 completes none
        std::vector<float> sentinel((size_t) 6 * D, -7.0f);
        float* d_p2 = upload(sentinel);
        int32_t st[kStepCount];
        kpool_step(18, kTopK, true, st);
        int32_t* d_st = upload(std::vector<int32_t>(st, st + kStepCount));
        kpool_pool(d_cache, d_ape, d_p2, 0, 0, d_st, nullptr);
        check(cudaDeviceSynchronize(), "pool step 18");
        const bool untouched = download(d_p2, sentinel.size()) == sentinel;
        kpool_step(19, kTopK, true, st);
        check(cudaMemcpy(d_st, st, sizeof st, cudaMemcpyHostToDevice), "st 19");
        kpool_pool(d_cache, d_ape, d_p2, 0, 0, d_st, nullptr);
        check(cudaDeviceSynchronize(), "pool step 19");
        const std::vector<float> p2 = download(d_p2, sentinel.size());
        bool pool4 = true, rest = true;
        for (int b = 0; b < 6; ++b)
            for (int d = 0; d < D; ++d) {
                const float v = p2[(size_t) b * D + d];
                if (b == 4) pool4 = pool4 && v == got[(size_t) b * D + d];
                else rest = rest && v == -7.0f;
            }
        expect(untouched, "through a step at pos 18 (no pool completes): nothing written");
        expect(pool4 && rest, "through a step at pos 19: pool 4 only, the same bits");
        cudaFree(d_pooled); cudaFree(d_p2); cudaFree(d_st);
    }

    // ================= 3. scores =================
    constexpr int64_t MAXB = 6000;   // pooled-key capacity: 24,000 cells
    std::vector<float> pooled((size_t) MAXB * D);
    for (auto& x : pooled) x = gauss(rng);
    float* d_pooled = upload(pooled);
    std::printf("\nkpool_scores, 3 queries with their own steps:\n");
    {
        const int64_t kvs[3] = {9, 2052, 5001};
        std::vector<int32_t> steps(3 * kStepCount);
        for (int q = 0; q < 3; ++q) kpool_step(kvs[q] - 1, kTopK, true, &steps[(size_t) q * kStepCount]);
        std::vector<float> iq((size_t) 3 * NH * D), w((size_t) 3 * NH);
        for (auto& x : iq) x = gauss(rng);
        for (auto& x : w) x = gauss(rng);
        float *d_iq = upload(iq), *d_w = upload(w), *d_sc = nullptr;
        int32_t* d_steps = upload(steps);
        check(cudaMalloc(&d_sc, (size_t) 3 * MAXB * 4), "scores");
        kpool_scores(d_pooled, d_iq, d_w, w_scale, d_steps, 3, MAXB, true, d_sc, nullptr);
        check(cudaDeviceSynchronize(), "scores");
        const std::vector<float> got = download(d_sc, (size_t) 3 * MAXB);
        std::vector<float> want_v, got_v;
        bool tails = true;
        for (int q = 0; q < 3; ++q) {
            const int64_t n_bid = kvs[q] / KP;
            for (int64_t b = 0; b < n_bid; ++b) {
                double s = 0;
                for (int h = 0; h < NH; ++h) {
                    double dot = 0;
                    for (int d = 0; d < D; ++d) dot += (double) iq[((size_t) q * NH + h) * D + d] * pooled[(size_t) b * D + d];
                    s += (double) w[(size_t) q * NH + h] * w_scale * std::max(dot, 0.0);
                }
                want_v.push_back((float) s);
                got_v.push_back(got[(size_t) q * MAXB + b]);
            }
            const float tail = got[(size_t) q * MAXB + n_bid];
            tails = tails && tail == (kvs[q] % KP != 0 ? 1e9f : -3e38f);
        }
        expect_close("scores of the completed pools (9, 2052 and 5001 cells)", want_v, got_v, 1e-5);
        expect(tails, "the tail block: +1e9 with cells (9, 5001), -3e38 without (2052)");
        cudaFree(d_iq); cudaFree(d_w); cudaFree(d_sc); cudaFree(d_steps);
    }

    // ================= 4. the selection =================
    std::printf("\nselection: kpool_scores -> qsa_block_topk at GLM's width, against GLM's rule:\n");
    {
        const QsaShapes sh = kpool_topk_shapes(kTopK);
        std::vector<float> iq((size_t) NH * D), w((size_t) NH);
        for (auto& x : iq) x = gauss(rng);
        for (auto& x : w) x = gauss(rng);
        float *d_iq = upload(iq), *d_w = upload(w), *d_sc = nullptr;
        int32_t *d_ids = nullptr, *d_st = nullptr;
        check(cudaMalloc(&d_sc, (size_t) MAXB * 4), "scores");
        check(cudaMalloc(&d_ids, (size_t) kCap * 4), "ids");
        check(cudaMalloc(&d_st, kStepCount * 4), "step");
        struct Case { int64_t n_kv; bool tail; };
        for (const Case cs : {Case{7, true}, Case{2048, true}, Case{2051, true}, Case{2052, true}, Case{2053, true},
                              Case{2055, true}, Case{4097, true}, Case{5001, true}, Case{23999, true}, Case{5001, false},
                              Case{2049, false}}) {
            int32_t st[kStepCount];
            kpool_step(cs.n_kv - 1, kTopK, cs.tail, st);
            check(cudaMemcpy(d_st, st, sizeof st, cudaMemcpyHostToDevice), "step");
            kpool_scores(d_pooled, d_iq, d_w, w_scale, d_st, 1, MAXB, cs.tail, d_sc, nullptr);
            qsa_block_topk(d_sc, d_st, 1, MAXB, kCap, sh, d_ids, nullptr);
            check(cudaDeviceSynchronize(), "select");
            const std::vector<float> sc = download(d_sc, (size_t) MAXB);
            std::vector<int32_t> got = download(d_ids, (size_t) st[kStepWidth]);
            std::sort(got.begin(), got.end());
            const std::vector<int32_t> want = ref_select(sc, cs.n_kv, cs.tail);
            char what[96];
            std::snprintf(what, sizeof what, "%6lld cells%s: %d selected", (long long) cs.n_kv,
                          cs.tail ? "" : " (no tail)", (int) st[kStepWidth]);
            expect(got == want, what, cs.n_kv <= kTopK + KP - 1 && cs.tail ? " (every cell)" : "");
        }
        {   // QSA's width min(n_kv, 2051) at 4097 cells (a 1-cell tail): two cells of a 513th pool beyond GLM's set
            int32_t st[kStepCount];
            kpool_step(4096, kTopK, true, st);
            const int glm = st[kStepWidth];
            st[kStepWidth] = (int32_t) qsa_selection_width(4097, sh);
            expect(glm == 2049 && st[kStepWidth] == 2051,
                   "rival: QSA's width at 4097 cells is 2051, GLM selects 2049 (512 pools + 1)");
        }
        cudaFree(d_iq); cudaFree(d_w); cudaFree(d_sc); cudaFree(d_ids); cudaFree(d_st);
    }

    // ================= --bench =================
    if (bench) {
        std::printf("\none query, scores then selection, mean of 50:\n");
        const QsaShapes sh = kpool_topk_shapes(kTopK);
        for (const int64_t n_kv : {32768, 131072, 1048576}) {
            const int64_t maxb = n_kv / KP + 1;
            float *d_p = nullptr, *d_sc = nullptr;
            check(cudaMalloc(&d_p, (size_t) maxb * D * 4), "p");
            check(cudaMemset(d_p, 0, (size_t) maxb * D * 4), "p0");
            check(cudaMalloc(&d_sc, (size_t) maxb * 4), "sc");
            std::vector<float> iq((size_t) NH * D, 0.01f), w((size_t) NH, 1.0f);
            float *d_iq = upload(iq), *d_w = upload(w);
            int32_t st[kStepCount];
            kpool_step(n_kv - 1, kTopK, true, st);
            int32_t* d_st = upload(std::vector<int32_t>(st, st + kStepCount));
            int32_t* d_ids = nullptr;
            check(cudaMalloc(&d_ids, (size_t) kCap * 4), "ids");
            cudaEvent_t e0, e1, e2;
            check(cudaEventCreate(&e0), "ev"); check(cudaEventCreate(&e1), "ev"); check(cudaEventCreate(&e2), "ev");
            float ms_s = 0, ms_k = 0;
            for (int it = -5; it < 50; ++it) {
                check(cudaEventRecord(e0, nullptr), "rec");
                kpool_scores(d_p, d_iq, d_w, w_scale, d_st, 1, maxb, true, d_sc, nullptr);
                check(cudaEventRecord(e1, nullptr), "rec");
                qsa_block_topk(d_sc, d_st, 1, maxb, kCap, sh, d_ids, nullptr);
                check(cudaEventRecord(e2, nullptr), "rec");
                check(cudaEventSynchronize(e2), "sync");
                if (it < 0) continue;
                float a = 0, b = 0;
                check(cudaEventElapsedTime(&a, e0, e1), "el");
                check(cudaEventElapsedTime(&b, e1, e2), "el");
                ms_s += a;
                ms_k += b;
            }
            std::printf("  %8lld cells (%7lld pools)   scores %8.1f us   selection %8.1f us\n", (long long) n_kv,
                        (long long) (n_kv / KP), ms_s * 1000 / 50, ms_k * 1000 / 50);
            cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
            cudaFree(d_p); cudaFree(d_sc); cudaFree(d_iq); cudaFree(d_w); cudaFree(d_st); cudaFree(d_ids);
        }
    }

    cudaFree(d_kraw); cudaFree(d_gate); cudaFree(d_lnw); cudaFree(d_lnb); cudaFree(d_ape); cudaFree(d_cache);
    cudaFree(d_pooled);
    std::printf("\nkpool: %d failures\n", g_bad);
    if (g_bad) return 1;
    if (selftest) std::printf("kpool_parity OK\n");
    return 0;
}
