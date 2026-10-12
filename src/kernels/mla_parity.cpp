// src/kernels/mla_parity.cpp - the MLA kernels (mla.hpp) against a double-precision transcription of llama.cpp.
//
//   1. mla_head_gemv, both shapes (attn_k_b: 256 -> 512, attn_v_b: 512 -> 256), 64 heads, 3 tokens
//   2. mla_kv_store: build_norm's RMS norm, rounded to FP16 - within one FP16 step of the double reference
//   3. mla_decode_attn at 64 heads against the absorbed reference over the same FP16 rows: dense and selected
//      positions, chunk-boundary counts (1, 31, 32, 33) and 2048 shuffled rows out of 5000
//   4. the chain q -> Wk_b -> attention -> Wv_b at 8 heads against the UNABSORBED form, llama.cpp's algebra before
//      absorption: explicit keys k_hj = Wk_b[h]^T c_j and values v_hj = Wv_b[h] c_j per head and position
//
// Rival readings are shown to differ on the fixture first: the weights read transposed ([h][n_in][n_out]), and the
// score scale 1 / sqrt(512) (the latent width) for 1 / sqrt(256).  `--bench` times the decode-time kernels.
#include "strata/kernels/mla.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int H = 64;      // attention.head_count
constexpr int DK = 256;    // key_length_mla (nope only)
constexpr int DV = 256;    // value_length_mla
constexpr int L = strata::kernels::kMlaLatent;
constexpr float kEps = 1e-5f;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

uint16_t bf16_bits(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
double bf16_val(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
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
void expect_apart(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    const double rel = rel_l1(a, b);
    const bool ok = rel > 0.05;
    std::printf("  %-56s %-10s (%.2f%% apart)\n", what, ok ? "yes" : "*** NO ***", rel * 100);
    if (!ok) ++g_bad;
}
void expect_close(const char* what, const std::vector<float>& want, const std::vector<float>& got, double tol) {
    const double rel = rel_l1(want, got);
    const bool ok = rel <= tol;
    std::printf("  %-56s rel %.3e %s\n", what, rel, ok ? "" : "*** over tolerance ***");
    if (!ok) ++g_bad;
}

/// y[t][h][o] = sum_i w[h][o][i] x[t][h][i]; `transposed` reads w as [h][i][o] (the rival).
std::vector<float> ref_head_gemv(const std::vector<float>& x, const std::vector<uint16_t>& w, int nh, int n_in,
                                 int n_out, int n_tok, bool transposed = false) {
    std::vector<float> y((size_t) n_tok * nh * n_out);
    for (int t = 0; t < n_tok; ++t)
        for (int h = 0; h < nh; ++h)
            for (int o = 0; o < n_out; ++o) {
                double acc = 0;
                for (int i = 0; i < n_in; ++i) {
                    const size_t wi = transposed ? ((size_t) h * n_in + i) * n_out + o : ((size_t) h * n_out + o) * n_in + i;
                    acc += bf16_val(w[wi]) * (double) x[((size_t) t * nh + h) * n_in + i];
                }
                y[((size_t) t * nh + h) * n_out + o] = (float) acc;
            }
    return y;
}

/// The latent row as the cache holds it, in double.
std::vector<double> row_of(const std::vector<uint16_t>& cache, int64_t r) {
    std::vector<double> c((size_t) L);
    for (int i = 0; i < L; ++i) c[(size_t) i] = strata::kernels::f32_from_f16(cache[(size_t) r * L + i]);
    return c;
}

/// Absorbed attention, one query: o[h] = sum_j softmax_j(q_abs[h].c_j * scale) c_j.
std::vector<float> ref_attn(const std::vector<float>& q_abs, const std::vector<uint16_t>& cache,
                            const std::vector<int32_t>& rows, int nh, double scale) {
    std::vector<float> o((size_t) nh * L);
    std::vector<std::vector<double>> c;
    for (int32_t r : rows) c.push_back(row_of(cache, r));
    for (int h = 0; h < nh; ++h) {
        std::vector<double> s(rows.size());
        double m = -INFINITY;
        for (size_t j = 0; j < rows.size(); ++j) {
            double d = 0;
            for (int i = 0; i < L; ++i) d += (double) q_abs[(size_t) h * L + i] * c[j][(size_t) i];
            s[j] = d * scale;
            m = std::max(m, s[j]);
        }
        double l = 0;
        std::vector<double> acc((size_t) L, 0.0);
        for (size_t j = 0; j < rows.size(); ++j) {
            const double e = std::exp(s[j] - m);
            l += e;
            for (int i = 0; i < L; ++i) acc[(size_t) i] += e * c[j][(size_t) i];
        }
        for (int i = 0; i < L; ++i) o[(size_t) h * L + i] = (float) (acc[(size_t) i] / l);
    }
    return o;
}

/// llama.cpp's MLA before absorption, one query, per head h and selected position j:
///   k_hj[d] = sum_c Wk_b[h][c][d] c_j[c],  v_hj[e] = sum_c Wv_b[h][e][c] c_j[c],
///   out_h   = sum_j softmax_j(q_h . k_hj * scale) v_hj
std::vector<float> ref_unabsorbed(const std::vector<float>& q, const std::vector<uint16_t>& wk,
                                  const std::vector<uint16_t>& wv, const std::vector<uint16_t>& cache,
                                  const std::vector<int32_t>& rows, int nh, double scale) {
    std::vector<float> out((size_t) nh * DV);
    std::vector<std::vector<double>> c;
    for (int32_t r : rows) c.push_back(row_of(cache, r));
    for (int h = 0; h < nh; ++h) {
        std::vector<double> s(rows.size());
        std::vector<std::vector<double>> v(rows.size(), std::vector<double>((size_t) DV, 0.0));
        double m = -INFINITY;
        for (size_t j = 0; j < rows.size(); ++j) {
            double d = 0;
            for (int dd = 0; dd < DK; ++dd) {
                double k = 0;
                for (int cc = 0; cc < L; ++cc) k += bf16_val(wk[((size_t) h * L + cc) * DK + dd]) * c[j][(size_t) cc];
                d += (double) q[(size_t) h * DK + dd] * k;
            }
            for (int e = 0; e < DV; ++e) {
                double a = 0;
                for (int cc = 0; cc < L; ++cc) a += bf16_val(wv[((size_t) h * DV + e) * L + cc]) * c[j][(size_t) cc];
                v[j][(size_t) e] = a;
            }
            s[j] = d * scale;
            m = std::max(m, s[j]);
        }
        double l = 0;
        std::vector<double> acc((size_t) DV, 0.0);
        for (size_t j = 0; j < rows.size(); ++j) {
            const double e = std::exp(s[j] - m);
            l += e;
            for (int i = 0; i < DV; ++i) acc[(size_t) i] += e * v[j][(size_t) i];
        }
        for (int i = 0; i < DV; ++i) out[(size_t) h * DV + i] = (float) (acc[(size_t) i] / l);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else if (std::string(argv[i]) == "--bench") bench = true;
        else { std::fprintf(stderr, "usage: mla_parity [--selftest] [--bench]\n"); return 2; }
    }
    using namespace strata::kernels;
    std::mt19937 rng(7349);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const double scale = 1.0 / std::sqrt((double) DK);

    // weights at the real geometry; a scale that keeps q_abs . c of order a few (softmax neither flat nor one-hot)
    std::vector<uint16_t> wk((size_t) H * L * DK), wv((size_t) H * DV * L);
    for (auto& b : wk) b = bf16_bits(0.05f * gauss(rng));
    for (auto& b : wv) b = bf16_bits(0.05f * gauss(rng));
    uint16_t* d_wk = upload(wk);
    uint16_t* d_wv = upload(wv);

    // ================= 1. the per-head GEMVs =================
    std::printf("mla_head_gemv, %d heads, 3 tokens:\n", H);
    {
        constexpr int T = 3;
        std::vector<float> q((size_t) T * H * DK), ol((size_t) T * H * L);
        for (auto& x : q) x = gauss(rng);
        for (auto& x : ol) x = gauss(rng);
        const std::vector<float> want_qa = ref_head_gemv(q, wk, H, DK, L, T);
        const std::vector<float> want_out = ref_head_gemv(ol, wv, H, L, DV, T);
        expect_apart("rival: attn_k_b read [h][n_in][n_out]", want_qa, ref_head_gemv(q, wk, H, DK, L, T, true));
        expect_apart("rival: attn_v_b read [h][n_in][n_out]", want_out, ref_head_gemv(ol, wv, H, L, DV, T, true));
        float *d_q = upload(q), *d_ol = upload(ol), *d_qa = nullptr, *d_out = nullptr;
        check(cudaMalloc(&d_qa, want_qa.size() * 4), "qa");
        check(cudaMalloc(&d_out, want_out.size() * 4), "out");
        mla_head_gemv(d_q, d_wk, d_qa, H, DK, L, T, nullptr);
        mla_head_gemv(d_ol, d_wv, d_out, H, L, DV, T, nullptr);
        check(cudaDeviceSynchronize(), "gemv");
        expect_close("q_abs = Wk_b q (256 -> 512)", want_qa, download(d_qa, want_qa.size()), 1e-6);
        expect_close("out = Wv_b o (512 -> 256)", want_out, download(d_out, want_out.size()), 1e-6);
        cudaFree(d_q); cudaFree(d_ol); cudaFree(d_qa); cudaFree(d_out);
    }

    // ================= 2. the latent into the cache =================
    std::printf("\nmla_kv_store, 5 tokens into rows 7 .. 11:\n");
    {
        constexpr int T = 5, R0 = 7, ROWS = 16;
        std::vector<float> kv((size_t) T * L), gamma((size_t) L);
        for (auto& x : kv) x = 3.0f * gauss(rng);
        for (auto& x : gamma) x = 1.0f + 0.3f * gauss(rng);
        float *d_kv = upload(kv), *d_g = upload(gamma);
        std::vector<uint16_t> blank((size_t) ROWS * L, 0x7e00);   // NaN: a row it should not touch stays so
        uint16_t* d_c = upload(blank);
        mla_kv_store(d_kv, d_g, kEps, d_c, R0, T, nullptr);
        check(cudaDeviceSynchronize(), "kv_store");
        const std::vector<uint16_t> got = download(d_c, blank.size());
        int off_by_one = 0, worse = 0, touched = 0;
        for (int r = 0; r < ROWS; ++r)
            for (int i = 0; i < L; ++i) {
                const uint16_t g = got[(size_t) r * L + i];
                if (r < R0 || r >= R0 + T) { touched += g != 0x7e00; continue; }
                const int t = r - R0;
                double ss = 0;
                for (int k = 0; k < L; ++k) ss += (double) kv[(size_t) t * L + k] * kv[(size_t) t * L + k];
                const double want = kv[(size_t) t * L + i] / std::sqrt(ss / L + kEps) * gamma[(size_t) i];
                const uint16_t w = f16_from_f32((float) want);
                const int d = std::abs((int) (int16_t) w - (int) (int16_t) g);
                off_by_one += d == 1;
                worse += d > 1;
            }
        const bool ok = worse == 0 && touched == 0;
        std::printf("  %-56s %s (%d of %d one FP16 step off, %d further, %d outside the rows)\n",
                    "FP16 rows vs the double norm, rounded", ok ? "yes" : "*** NO ***", off_by_one, T * L, worse,
                    touched);
        if (!ok) ++g_bad;
        cudaFree(d_kv); cudaFree(d_g); cudaFree(d_c);
    }

    // a cache of 5000 latent rows, as mla_kv_store leaves them (unit RMS)
    constexpr int64_t ROWS = 5000;
    std::vector<uint16_t> cache((size_t) ROWS * L);
    for (auto& b : cache) b = f16_from_f32(gauss(rng));
    uint16_t* d_cache = upload(cache);

    // ================= 3. decode attention, 64 heads =================
    std::printf("\nmla_decode_attn, %d heads, against the absorbed reference over the same FP16 rows:\n", H);
    {
        std::vector<float> qa((size_t) H * L);
        for (auto& x : qa) x = 0.15f * gauss(rng);   // scores of order a few after the scale
        float* d_qa = upload(qa);
        float *d_o = nullptr, *d_scr = nullptr;
        check(cudaMalloc(&d_o, qa.size() * 4), "o");
        check(cudaMalloc(&d_scr, mla_decode_attn_scratch_floats(H, 4096) * 4), "scratch");
        std::vector<int32_t> perm((size_t) ROWS);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        struct Case { int64_t n; bool ids; };
        for (const Case cs : {Case{1, true}, Case{31, true}, Case{32, true}, Case{33, true}, Case{100, false},
                              Case{2048, true}}) {
            std::vector<int32_t> rows(perm.begin(), perm.begin() + cs.n);
            if (!cs.ids) std::iota(rows.begin(), rows.end(), 0);
            int32_t* d_ids = cs.ids ? upload(rows) : nullptr;
            mla_decode_attn(d_qa, d_cache, d_ids, cs.n, (float) scale, d_scr, d_o, H, nullptr);
            check(cudaDeviceSynchronize(), "attn");
            char what[96];
            std::snprintf(what, sizeof what, "%4lld positions%s", (long long) cs.n,
                          cs.ids ? ", selected rows (shuffled)" : ", rows 0 .. n - 1 (no ids)");
            expect_close(what, ref_attn(qa, cache, rows, H, scale), download(d_o, qa.size()), 1e-5);
            if (d_ids) cudaFree(d_ids);
        }
        cudaFree(d_qa); cudaFree(d_o); cudaFree(d_scr);
    }

    // ================= 4. the chain against the unabsorbed form =================
    constexpr int HC = 8;   // heads, so the explicit keys and values stay quick on the CPU
    constexpr int64_t NS = 300;
    std::printf("\nq -> Wk_b -> attention -> Wv_b, %d heads x %lld selected rows, against llama.cpp's unabsorbed MLA:\n",
                HC, (long long) NS);
    {
        std::vector<float> q((size_t) HC * DK);
        for (auto& x : q) x = 0.6f * gauss(rng);
        std::vector<int32_t> rows((size_t) NS);
        for (auto& r : rows) r = (int32_t) (rng() % ROWS);
        const std::vector<float> want = ref_unabsorbed(q, wk, wv, cache, rows, HC, scale);
        expect_apart("rival: scale 1/sqrt(512) for 1/sqrt(256)", want,
                     ref_unabsorbed(q, wk, wv, cache, rows, HC, 1.0 / std::sqrt((double) L)));
        float *d_q = upload(q), *d_qa = nullptr, *d_ol = nullptr, *d_out = nullptr, *d_scr = nullptr;
        int32_t* d_ids = upload(rows);
        check(cudaMalloc(&d_qa, (size_t) HC * L * 4), "qa");
        check(cudaMalloc(&d_ol, (size_t) HC * L * 4), "ol");
        check(cudaMalloc(&d_out, (size_t) HC * DV * 4), "out");
        check(cudaMalloc(&d_scr, mla_decode_attn_scratch_floats(HC, NS) * 4), "scratch");
        // the first HC heads' weights are the leading slices of attn_k_b and attn_v_b
        mla_head_gemv(d_q, d_wk, d_qa, HC, DK, L, 1, nullptr);
        mla_decode_attn(d_qa, d_cache, d_ids, NS, (float) scale, d_scr, d_ol, HC, nullptr);
        mla_head_gemv(d_ol, d_wv, d_out, HC, L, DV, 1, nullptr);
        check(cudaDeviceSynchronize(), "chain");
        expect_close("out", want, download(d_out, want.size()), 1e-5);
        cudaFree(d_q); cudaFree(d_qa); cudaFree(d_ol); cudaFree(d_out); cudaFree(d_scr); cudaFree(d_ids);
    }

    // ================= --bench: one layer's decode-time kernels =================
    if (bench) {
        std::printf("\none MLA layer at decode (the parts here), %d heads, mean of 200:\n", H);
        float *d_q = nullptr, *d_qa = nullptr, *d_ol = nullptr, *d_out = nullptr, *d_scr = nullptr;
        check(cudaMalloc(&d_q, (size_t) H * DK * 4), "q");
        check(cudaMalloc(&d_qa, (size_t) H * L * 4), "qa");
        check(cudaMalloc(&d_ol, (size_t) H * L * 4), "ol");
        check(cudaMalloc(&d_out, (size_t) H * DV * 4), "out");
        check(cudaMalloc(&d_scr, mla_decode_attn_scratch_floats(H, 4096) * 4), "scratch");
        check(cudaMemset(d_q, 0, (size_t) H * DK * 4), "q0");
        check(cudaMemset(d_qa, 0, (size_t) H * L * 4), "qa0");
        cudaEvent_t e0, e1;
        check(cudaEventCreate(&e0), "ev");
        check(cudaEventCreate(&e1), "ev");
        auto time = [&](const char* what, auto&& fn) {
            for (int i = 0; i < 10; ++i) fn();
            check(cudaEventRecord(e0, nullptr), "rec");
            for (int i = 0; i < 200; ++i) fn();
            check(cudaEventRecord(e1, nullptr), "rec");
            check(cudaEventSynchronize(e1), "sync");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, e0, e1), "el");
            std::printf("  %-40s %8.1f us\n", what, ms * 1000.0f / 200.0f);
        };
        time("q_abs = Wk_b q (16.8 MB of BF16)", [&] { mla_head_gemv(d_q, d_wk, d_qa, H, DK, L, 1, nullptr); });
        time("out = Wv_b o (16.8 MB of BF16)", [&] { mla_head_gemv(d_ol, d_wv, d_out, H, L, DV, 1, nullptr); });
        for (const int64_t n : {256, 1024, 2048, 4096}) {
            char what[64];
            std::snprintf(what, sizeof what, "attention over %lld rows", (long long) n);
            time(what, [&] { mla_decode_attn(d_qa, d_cache, nullptr, n, (float) scale, d_scr, d_ol, H, nullptr); });
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(d_q); cudaFree(d_qa); cudaFree(d_ol); cudaFree(d_out); cudaFree(d_scr);
    }

    cudaFree(d_wk); cudaFree(d_wv); cudaFree(d_cache);
    std::printf("\nmla: %d failures\n", g_bad);
    if (g_bad) return 1;
    if (selftest) std::printf("mla_parity OK\n");
    return 0;
}
