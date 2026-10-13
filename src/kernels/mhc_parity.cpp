// src/kernels/mhc_parity.cpp - the mHC kernels (mhc.hpp) against a double transcription of llama.cpp's build_hc_pre,
// build_hc_sinkhorn, build_hc_post and build_hc_mean, at GLM-5.3-Flash's geometry (4 streams of 4096, 20 Sinkhorn
// rounds), 5 tokens with their own streams.
//
//   mhc_pre: pre, post and comb (and comb's rows and columns summing to 1), and h = sum_i pre_i X[i]
//   mhc_post: X'[dst] = post_dst f + sum_src comb[src][dst] X[src], in place
//   mhc_init, mhc_mean
//
// Rivals shown to differ first: comb without the Sinkhorn rounds, comb read [dst][src] in the post, and the RMS norm
// per stream instead of over the flattened 4 x 4096.  (The softmax's axis is NOT one: soft-maxing over src or over
// dst differ by a diagonal scaling, and the Sinkhorn rounds converge both to the same doubly stochastic matrix.)  `--bench` times one sublayer's pre and post.
#include "strata/kernels/mhc.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

constexpr int HC = k::kMhcStreams, MIX = k::kMhcMix;
constexpr int D = 4096;                  // embedding_length
constexpr int ITERS = 20;                // hyper_connection.sinkhorn_iterations
constexpr float kRmsEps = 1e-5f, kHcEps = 1e-6f;

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
void expect(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("  %-60s %s%s\n", what.c_str(), ok ? "yes" : "*** NO ***", detail.c_str());
    if (!ok) ++g_bad;
}
void expect_close(const std::string& what, const std::vector<float>& want, const std::vector<float>& got, double tol) {
    const double rel = rel_l1(want, got);
    const bool ok = rel <= tol;
    std::printf("  %-60s rel %.3e %s\n", what.c_str(), rel, ok ? "" : "*** over tolerance ***");
    if (!ok) ++g_bad;
}
void expect_apart(const std::string& what, const std::vector<float>& a, const std::vector<float>& b) {
    const double rel = rel_l1(a, b);
    char buf[48];
    std::snprintf(buf, sizeof buf, " (%.2f%% apart)", rel * 100);
    expect(rel > 0.05, what, buf);
}

double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Pre { std::vector<float> pre, post, comb, h; };

/// build_hc_pre and build_hc_sinkhorn for n tokens.  Rivals: `per_stream` normalizes each stream by its own RMS;
/// `softmax_src` soft-maxes each dst column over src; `no_sinkhorn` stops after the softmax.
Pre ref_pre(const std::vector<float>& x, const std::vector<float>& fn, const std::vector<float>& sc,
            const std::vector<float>& base, int n, bool per_stream = false, bool softmax_src = false,
            bool no_sinkhorn = false) {
    Pre r;
    const int K = HC * D;
    for (int t = 0; t < n; ++t) {
        const float* xt = &x[(size_t) t * K];
        std::vector<double> xn((size_t) K);
        if (!per_stream) {
            double ss = 0;
            for (int i = 0; i < K; ++i) ss += (double) xt[i] * xt[i];
            const double inv = 1.0 / std::sqrt(ss / K + kRmsEps);
            for (int i = 0; i < K; ++i) xn[(size_t) i] = xt[i] * inv;
        } else {
            for (int s = 0; s < HC; ++s) {
                double ss = 0;
                for (int i = 0; i < D; ++i) ss += (double) xt[s * D + i] * xt[s * D + i];
                const double inv = 1.0 / std::sqrt(ss / D + kRmsEps);
                for (int i = 0; i < D; ++i) xn[(size_t) (s * D + i)] = xt[s * D + i] * inv;
            }
        }
        double m[MIX];
        for (int j = 0; j < MIX; ++j) {
            double a = 0;
            for (int i = 0; i < K; ++i) a += (double) fn[(size_t) j * K + i] * xn[(size_t) i];
            m[j] = a;
        }
        double pre[HC];
        for (int i = 0; i < HC; ++i) {
            pre[i] = sig(m[i] * sc[0] + base[(size_t) i]) + kHcEps;
            r.pre.push_back((float) pre[i]);
            r.post.push_back((float) (2.0 * sig(m[HC + i] * sc[1] + base[(size_t) (HC + i)])));
        }
        double c[HC][HC];
        for (int s = 0; s < HC; ++s)
            for (int d = 0; d < HC; ++d) c[s][d] = m[2 * HC + s * HC + d] * sc[2] + base[(size_t) (2 * HC + s * HC + d)];
        if (!softmax_src) {
            for (int s = 0; s < HC; ++s) {
                double mx = -INFINITY, sum = 0;
                for (int d = 0; d < HC; ++d) mx = std::max(mx, c[s][d]);
                for (int d = 0; d < HC; ++d) sum += std::exp(c[s][d] - mx);
                for (int d = 0; d < HC; ++d) c[s][d] = std::exp(c[s][d] - mx) / sum + kHcEps;
            }
        } else {
            for (int d = 0; d < HC; ++d) {
                double mx = -INFINITY, sum = 0;
                for (int s = 0; s < HC; ++s) mx = std::max(mx, c[s][d]);
                for (int s = 0; s < HC; ++s) sum += std::exp(c[s][d] - mx);
                for (int s = 0; s < HC; ++s) c[s][d] = std::exp(c[s][d] - mx) / sum + kHcEps;
            }
        }
        auto cols = [&] {
            for (int d = 0; d < HC; ++d) {
                double sum = kHcEps;
                for (int s = 0; s < HC; ++s) sum += c[s][d];
                for (int s = 0; s < HC; ++s) c[s][d] /= sum;
            }
        };
        auto rows = [&] {
            for (int s = 0; s < HC; ++s) {
                double sum = kHcEps;
                for (int d = 0; d < HC; ++d) sum += c[s][d];
                for (int d = 0; d < HC; ++d) c[s][d] /= sum;
            }
        };
        if (!no_sinkhorn) {
            cols();
            for (int it = 1; it < ITERS; ++it) { rows(); cols(); }
        }
        for (int s = 0; s < HC; ++s)
            for (int d = 0; d < HC; ++d) r.comb.push_back((float) c[s][d]);
        for (int d = 0; d < D; ++d) {
            double a = 0;
            for (int i = 0; i < HC; ++i) a += pre[i] * xt[i * D + d];
            r.h.push_back((float) a);
        }
    }
    return r;
}

/// build_hc_post; `transposed` reads comb as [dst][src].
std::vector<float> ref_post(const std::vector<float>& f, const std::vector<float>& x, const std::vector<float>& post,
                            const std::vector<float>& comb, int n, bool transposed = false) {
    std::vector<float> out(x.size());
    for (int t = 0; t < n; ++t)
        for (int dst = 0; dst < HC; ++dst)
            for (int d = 0; d < D; ++d) {
                double a = (double) f[(size_t) t * D + d] * post[(size_t) t * HC + dst];
                for (int s = 0; s < HC; ++s) {
                    const float c = transposed ? comb[(size_t) t * 16 + dst * HC + s] : comb[(size_t) t * 16 + s * HC + dst];
                    a += (double) x[((size_t) t * HC + s) * D + d] * c;
                }
                out[((size_t) t * HC + dst) * D + d] = (float) a;
            }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else if (std::string(argv[i]) == "--bench") bench = true;
        else { std::fprintf(stderr, "usage: mhc_parity [--selftest] [--bench]\n"); return 2; }
    }
    std::mt19937 rng(4104);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    constexpr int T = 5, K = HC * D;

    // streams of different scales (as after some layers), weights giving mixes of order one
    std::vector<float> x((size_t) T * K), fn((size_t) MIX * K), base(MIX), sc = {0.7f, 1.3f, 2.0f};
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < HC; ++s)
            for (int d = 0; d < D; ++d) x[((size_t) t * HC + s) * D + d] = (0.5f + 0.5f * s) * gauss(rng);
    for (auto& v : fn) v = gauss(rng) / std::sqrt((float) K);
    for (auto& v : base) v = 0.5f * gauss(rng);

    std::printf("mhc_pre, %d tokens, 4 streams of %d, %d Sinkhorn rounds:\n", T, D, ITERS);
    const Pre want = ref_pre(x, fn, sc, base, T);
    expect_apart("rival: the RMS norm per stream", want.h, ref_pre(x, fn, sc, base, T, true).h);
    expect_apart("rival: the softmax alone, no Sinkhorn", want.comb, ref_pre(x, fn, sc, base, T, false, false, true).comb);
    {   // not a rival: the two softmaxes differ by a diagonal scaling, which the Sinkhorn rounds remove
        const double rel = rel_l1(want.comb, ref_pre(x, fn, sc, base, T, false, true).comb);
        std::printf("  %-60s %.1e apart (the same limit)\n", "(the softmax over src instead: after 20 rounds", rel);
    }
    float *d_x = upload(x), *d_fn = upload(fn), *d_sc = upload(sc), *d_base = upload(base);
    float *d_scr = nullptr, *d_h = nullptr, *d_post = nullptr, *d_comb = nullptr, *d_pre = nullptr;
    check(cudaMalloc(&d_scr, k::mhc_pre_scratch_floats(T, D) * 4), "scratch");
    check(cudaMalloc(&d_h, (size_t) T * D * 4), "h");
    check(cudaMalloc(&d_post, (size_t) T * HC * 4), "post");
    check(cudaMalloc(&d_comb, (size_t) T * 16 * 4), "comb");
    check(cudaMalloc(&d_pre, (size_t) T * HC * 4), "pre");
    k::mhc_pre(d_x, d_fn, d_sc, d_base, T, D, kRmsEps, kHcEps, ITERS, d_scr, d_h, d_post, d_comb, d_pre, nullptr);
    check(cudaDeviceSynchronize(), "pre");
    const std::vector<float> got_comb = download(d_comb, (size_t) T * 16), got_post = download(d_post, (size_t) T * HC);
    expect_close("pre", want.pre, download(d_pre, (size_t) T * HC), 1e-6);
    expect_close("post", want.post, got_post, 1e-6);
    expect_close("comb", want.comb, got_comb, 1e-5);
    {
        double worst = 0;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < HC; ++i) {
                double r = 0, c = 0;
                for (int j = 0; j < HC; ++j) { r += got_comb[(size_t) t * 16 + i * HC + j]; c += got_comb[(size_t) t * 16 + j * HC + i]; }
                worst = std::max(worst, std::max(std::fabs(r - 1), std::fabs(c - 1)));
            }
        char buf[48];
        std::snprintf(buf, sizeof buf, " (largest |sum - 1| %.1e)", worst);
        expect(worst < 1e-4, "comb's rows and columns each sum to 1", buf);
    }
    expect_close("h = sum_i pre_i X[i]", want.h, download(d_h, (size_t) T * D), 1e-6);

    std::printf("\nmhc_post, in place:\n");
    {
        std::vector<float> f((size_t) T * D);
        for (auto& v : f) v = gauss(rng);
        const std::vector<float> want_x = ref_post(f, x, got_post, got_comb, T);
        expect_apart("rival: comb read [dst][src]", want_x, ref_post(f, x, got_post, got_comb, T, true));
        float* d_f = upload(f);
        k::mhc_post(d_f, d_post, d_comb, d_x, T, D, nullptr);
        check(cudaDeviceSynchronize(), "post");
        expect_close("X'", want_x, download(d_x, x.size()), 1e-6);
        cudaFree(d_f);
    }

    std::printf("\nmhc_init, mhc_mean:\n");
    {
        std::vector<float> e((size_t) T * D);
        for (auto& v : e) v = gauss(rng);
        float* d_e = upload(e);
        k::mhc_init(d_e, d_x, T, D, nullptr);
        check(cudaDeviceSynchronize(), "init");
        const std::vector<float> xs = download(d_x, x.size());
        bool same = true;
        for (int t = 0; t < T; ++t)
            for (int s = 0; s < HC; ++s)
                for (int d = 0; d < D; ++d) same = same && xs[((size_t) t * HC + s) * D + d] == e[(size_t) t * D + d];
        expect(same, "every stream a copy of the embedding");
        // the mean of four different streams
        k::mhc_post(d_e, d_post, d_comb, d_x, T, D, nullptr);
        std::vector<float> y_want((size_t) T * D);
        const std::vector<float> xm = download(d_x, x.size());
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D; ++d) {
                double a = 0;
                for (int s = 0; s < HC; ++s) a += xm[((size_t) t * HC + s) * D + d];
                y_want[(size_t) t * D + d] = (float) (a / HC);
            }
        k::mhc_mean(d_x, d_e, T, D, nullptr);
        check(cudaDeviceSynchronize(), "mean");
        expect_close("the mean of the streams", y_want, download(d_e, (size_t) T * D), 1e-7);
        cudaFree(d_e);
    }

    if (bench) {
        std::printf("\none sublayer's pre and post, 200 calls back to back, the mean:\n");
        for (const int n : {1, 512}) {
            float *bx = nullptr, *bh = nullptr, *bs = nullptr, *bp = nullptr, *bc = nullptr;
            check(cudaMalloc(&bx, (size_t) n * K * 4), "bx");
            check(cudaMemset(bx, 0, (size_t) n * K * 4), "bx0");
            check(cudaMalloc(&bh, (size_t) n * D * 4), "bh");
            check(cudaMemset(bh, 0, (size_t) n * D * 4), "bh0");
            check(cudaMalloc(&bs, k::mhc_pre_scratch_floats(n, D) * 4), "bs");
            check(cudaMalloc(&bp, (size_t) n * HC * 4), "bp");
            check(cudaMalloc(&bc, (size_t) n * 16 * 4), "bc");
            cudaEvent_t e0, e1, e2;
            check(cudaEventCreate(&e0), "ev"); check(cudaEventCreate(&e1), "ev"); check(cudaEventCreate(&e2), "ev");
            // 200 calls back to back, so the per-launch cost of an event pair on this host does not pose as kernel time
            for (int it = 0; it < 10; ++it) {
                k::mhc_pre(bx, d_fn, d_sc, d_base, n, D, kRmsEps, kHcEps, ITERS, bs, bh, bp, bc, nullptr, nullptr);
                k::mhc_post(bh, bp, bc, bx, n, D, nullptr);
            }
            check(cudaEventRecord(e0, nullptr), "rec");
            for (int it = 0; it < 200; ++it)
                k::mhc_pre(bx, d_fn, d_sc, d_base, n, D, kRmsEps, kHcEps, ITERS, bs, bh, bp, bc, nullptr, nullptr);
            check(cudaEventRecord(e1, nullptr), "rec");
            for (int it = 0; it < 200; ++it) k::mhc_post(bh, bp, bc, bx, n, D, nullptr);
            check(cudaEventRecord(e2, nullptr), "rec");
            check(cudaEventSynchronize(e2), "sync");
            float a = 0, b = 0;
            check(cudaEventElapsedTime(&a, e0, e1), "el");
            check(cudaEventElapsedTime(&b, e1, e2), "el");
            std::printf("  %4d tokens   pre %8.1f us   post %8.1f us   (%.2f us a token in all)\n", n, a * 5, b * 5,
                        (a + b) * 5 / n);
            cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
            cudaFree(bx); cudaFree(bh); cudaFree(bs); cudaFree(bp); cudaFree(bc);
        }
    }

    cudaFree(d_x); cudaFree(d_fn); cudaFree(d_sc); cudaFree(d_base); cudaFree(d_scr); cudaFree(d_h); cudaFree(d_post);
    cudaFree(d_comb); cudaFree(d_pre);
    std::printf("\nmhc: %d failures\n", g_bad);
    if (g_bad) return 1;
    if (selftest) std::printf("mhc_parity OK\n");
    return 0;
}
