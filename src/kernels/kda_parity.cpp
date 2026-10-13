// src/kernels/kda_parity.cpp - the KDA kernels (kda.hpp) against a double-precision transcription of llama.cpp.
//
// The reference transcribes, at GLM-5.3-Flash's real geometry (64 heads of 128):
//   - glm5_conv1d: ggml_ssm_conv over [state | x] for each of q, k, v, then SiLU; build_gdn_l2_norm on q and k
//   - build_kda_layer's gate: (raw + dt_bias) * A, then sigmoid(-.) * lower_bound (or softplus, then * A)
//   - ggml_compute_forward_gated_delta_net's KDA branch, with the state in ggml's own layout (s[j*D + i] = S[i][j])
//   - build_kda_layer's tail: rms_norm(o) * ssm_norm * sigmoid(g2)
//
// Before the kernels are judged, each rival reading of the source that keeps every shape is shown to give a
// DIFFERENT answer on this fixture, so passing means the kernel took the right one:
//   1. decay along the KEY channel i (rows of S, contracted against k) vs the value channel j
//   2. decay applied before the rank-1 update vs after it
//   3. sigmoid(-A * u) vs sigmoid(A * u): ssm_a holds -exp(A_log), and dropping that sign flips the gate's range
//   4. the conv's tap order: kernel[0] reads the OLDEST input
//
// Then the kernels run one token, and a run of seven tokens both in one launch and split 1 + 2 + 4, which must be
// bitwise identical to the single launch (the state and the conv history carry across launches exactly).
// The chunked scan (kda_scan_norm_chunked) is checked against the reference at 1, 100 and (8 heads) 1100 tokens, and
// against kda_scan_norm at those and 2048 tokens (largest |difference| over largest |value|, gdn_rec_parity's measure).
// `--bench` adds each kernel's time for one layer at T = 1 (decode) and T = 512 (a prompt run), and the two scans'
// times from 32 to 8192 tokens.
#include "strata/kernels/kda.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int H = 64;                         // GLM-5.3-Flash: attention.head_count
constexpr int D = strata::kernels::kKdaHeadDim;
constexpr int DI = H * D;                     // d_inner
constexpr int DC = strata::kernels::kKdaConvKernel;
constexpr float kL2Eps = 1e-6f;               // build_kda_layer's l2_eps
constexpr float kRmsEps = 1e-5f;              // attention.layer_norm_rms_epsilon
constexpr float kLowerBound = -5.0f;          // kda.gate_lower_bound

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double softplus_d(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }

double rel_l1(const std::vector<float>& want, const std::vector<float>& got) {
    double d = 0, m = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        d += std::fabs((double) want[i] - (double) got[i]);
        m += std::fabs((double) want[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

/// glm5_conv1d for T tokens of all three parts, then the L2 norm of q and k.  `hist` is per channel, oldest first,
/// [3 * DI][DC - 1]; `x` and `out` are [T][3 * DI].  `reverse_taps` is rival 4.
void ref_conv_l2(std::vector<float>& hist, const std::vector<float>& x, const std::vector<float> (&w)[3], int T,
                 std::vector<float>& out, bool reverse_taps = false) {
    const int C = 3 * DI;
    out.assign((size_t) T * C, 0.0f);
    for (int gc = 0; gc < C; ++gc) {
        const int part = gc / DI, c = gc % DI;
        std::vector<double> ext((size_t) (DC - 1 + T));                   // conv_x = concat(state, x^T)
        for (int i = 0; i < DC - 1; ++i) ext[(size_t) i] = hist[(size_t) gc * (DC - 1) + i];
        for (int t = 0; t < T; ++t) ext[(size_t) (DC - 1 + t)] = x[(size_t) t * C + gc];
        for (int t = 0; t < T; ++t) {
            double acc = 0;
            for (int i = 0; i < DC; ++i) {
                const int tap = reverse_taps ? DC - 1 - i : i;
                acc += ext[(size_t) (t + i)] * (double) w[part][(size_t) c * DC + tap];
            }
            out[(size_t) t * C + gc] = (float) (acc * sigmoid_d(acc));
        }
        for (int i = 0; i < DC - 1; ++i) hist[(size_t) gc * (DC - 1) + i] = (float) ext[(size_t) (T + i)];
    }
    for (int t = 0; t < T; ++t)
        for (int hh = 0; hh < 2 * H; ++hh) {                              // q heads then k heads
            float* p = &out[(size_t) t * C + (size_t) hh * D];
            double ss = 0;
            for (int i = 0; i < D; ++i) ss += (double) p[i] * (double) p[i];
            const double inv = 1.0 / std::sqrt(ss + (double) kL2Eps);
            for (int i = 0; i < D; ++i) p[i] = (float) ((double) p[i] * inv);
        }
}

/// build_kda_layer's gate.  `flip_a` is rival 3: treating ssm_a as +exp(A_log).
void ref_gate(const std::vector<float>& raw, const std::vector<float>& dt, const std::vector<float>& ssm_a, int T,
              bool has_lb, std::vector<float>& gate, bool flip_a = false) {
    gate.assign(raw.size(), 0.0f);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < DI; ++c) {
            const double a = (flip_a ? -1.0 : 1.0) * (double) ssm_a[(size_t) (c / D)];
            const double g1 = (double) raw[(size_t) t * DI + c] + (double) dt[(size_t) c];
            gate[(size_t) t * DI + c] =
                (float) (has_lb ? sigmoid_d(-(g1 * a)) * (double) kLowerBound : softplus_d(g1) * a);
        }
}

/// The KDA branch of ggml_compute_forward_gated_delta_net, then build_kda_layer's output norm, for T tokens.
/// `st` is [nh][D * D] in ggml's layout, s[j * D + i] = S[i][j].  `decay_on_value` is rival 1, `decay_after` 2;
/// `nh` heads (the layer's 64 unless a long run is to stay quick on the CPU).
void ref_scan_norm(std::vector<double>& st, const std::vector<float>& qkv, const std::vector<float>& gate,
                   const std::vector<float>& beta, const std::vector<float>& z, const std::vector<float>& gamma, int T,
                   std::vector<float>& y, bool decay_on_value = false, bool decay_after = false, int nh = H) {
    const int di = nh * D;
    y.assign((size_t) T * di, 0.0f);
    const double scale = 1.0 / std::sqrt((double) D);
    std::vector<double> delta((size_t) D), dec((size_t) D), o((size_t) D);
    for (int h = 0; h < nh; ++h) {
        double* M = &st[(size_t) h * D * D];
        for (int t = 0; t < T; ++t) {
            const float* q = &qkv[(size_t) t * 3 * di + (size_t) h * D];
            const float* k = q + di;
            const float* v = q + 2 * di;
            const float* g = &gate[(size_t) t * di + (size_t) h * D];
            const double b = beta[(size_t) t * nh + h];
            for (int i = 0; i < D; ++i) dec[(size_t) i] = std::exp((double) g[i]);
            auto apply_decay = [&] {          // S[i][:] *= exp(g[i])  =>  M[j][i] *= exp(g[i])
                for (int j = 0; j < D; ++j)
                    for (int i = 0; i < D; ++i) M[(size_t) j * D + i] *= dec[(size_t) (decay_on_value ? j : i)];
            };
            if (!decay_after) apply_decay();
            for (int j = 0; j < D; ++j) {
                double sum = 0;
                for (int i = 0; i < D; ++i) sum += M[(size_t) j * D + i] * (double) k[i];
                delta[(size_t) j] = ((double) v[j] - sum) * b;
            }
            for (int j = 0; j < D; ++j)
                for (int i = 0; i < D; ++i) M[(size_t) j * D + i] += (double) k[i] * delta[(size_t) j];
            if (decay_after) apply_decay();
            double ss = 0;
            for (int j = 0; j < D; ++j) {
                double sum = 0;
                for (int i = 0; i < D; ++i) sum += M[(size_t) j * D + i] * (double) q[i];
                o[(size_t) j] = sum * scale;
                ss += o[(size_t) j] * o[(size_t) j];
            }
            const double inv = 1.0 / std::sqrt(ss / D + (double) kRmsEps);
            for (int j = 0; j < D; ++j) {
                const size_t at = (size_t) t * di + (size_t) h * D + j;
                y[at] = (float) (o[(size_t) j] * inv * (double) gamma[(size_t) j] * sigmoid_d((double) z[at]));
            }
        }
    }
}

/// Strata's (H, D, D) state, [(h * D + i) * D + j], from and to ggml's per-head s[j * D + i].
std::vector<float> to_dev(const std::vector<double>& st, int nh = H) {
    std::vector<float> d((size_t) D * nh * D);
    for (int h = 0; h < nh; ++h)
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j) d[((size_t) h * D + i) * D + j] = (float) st[(size_t) h * D * D + (size_t) j * D + i];
    return d;
}
std::vector<float> from_dev(const std::vector<float>& d, int nh = H) {
    std::vector<float> st((size_t) nh * D * D);
    for (int h = 0; h < nh; ++h)
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j) st[(size_t) h * D * D + (size_t) j * D + i] = d[((size_t) h * D + i) * D + j];
    return st;
}
std::vector<float> to_float(const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); }

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
    std::printf("  %-52s %-10s (%.2f%% apart)\n", what, ok ? "yes" : "*** NO ***", rel * 100);
    if (!ok) ++g_bad;
}

void expect_close(const char* what, const std::vector<float>& want, const std::vector<float>& got, double tol) {
    const double rel = rel_l1(want, got);
    const bool ok = rel <= tol;
    std::printf("  %-52s rel %.3e %s\n", what, rel, ok ? "" : "*** over tolerance ***");
    if (!ok) ++g_bad;
}

/// Largest |difference| over largest |value|: gdn_rec_parity's measure for two FP32 kernels that sum in other orders.
void expect_max_rel(const char* what, const std::vector<float>& want, const std::vector<float>& got, double tol) {
    double d = 0, m = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        d = std::fmax(d, std::fabs((double) want[i] - (double) got[i]));
        m = std::fmax(m, std::fabs((double) want[i]));
    }
    const double rel = d / (m > 1e-30 ? m : 1e-30);
    const bool ok = rel <= tol;
    std::printf("  %-52s max %.3e %s\n", what, rel, ok ? "" : "*** over tolerance ***");
    if (!ok) ++g_bad;
}

void expect_bitwise(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    const bool ok = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    std::printf("  %-52s %s\n", what, ok ? "bitwise" : "*** DIFFERENT ***");
    if (!ok) ++g_bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else if (std::string(argv[i]) == "--bench") bench = true;
        else { std::fprintf(stderr, "usage: kda_parity [--selftest] [--bench]\n"); return 2; }
    }

    constexpr int T = 7;
    const int splits[] = {1, 2, 4};          // 1 takes the in-place history path, 2 < d_conv - 1 mixes old and new
    std::mt19937 rng(5309);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    // ---- the layer's inputs, at the real geometry ----
    std::vector<float> w[3];
    for (auto& wp : w) { wp.resize((size_t) DI * DC); for (auto& x : wp) x = gauss(rng) * 0.5f; }
    std::vector<float> hist0((size_t) 3 * DI * (DC - 1)), xproj((size_t) T * 3 * DI);
    for (auto& x : hist0) x = gauss(rng);
    for (auto& x : xproj) x = gauss(rng);
    std::vector<float> raw((size_t) T * DI), dt((size_t) DI), ssm_a((size_t) H);
    for (auto& x : raw) x = gauss(rng);
    for (auto& x : dt) x = gauss(rng) * 0.5f;
    for (auto& x : ssm_a) x = -std::exp(unit(rng) * std::log(16.0f));   // -exp(A_log), A in [1, 16] as Mamba inits it
    std::vector<float> beta((size_t) T * H), z((size_t) T * DI), gamma((size_t) D);
    for (auto& x : beta) x = (float) sigmoid_d(gauss(rng));
    for (auto& x : z) x = gauss(rng);
    for (auto& x : gamma) x = 1.0f + 0.2f * gauss(rng);
    std::vector<double> st0((size_t) H * D * D);
    for (auto& x : st0) x = 0.05 * gauss(rng);

    // the reference chain: conv -> gate -> scan
    std::vector<float> hist_want = hist0, qkv_want, gate_want, y_want;
    ref_conv_l2(hist_want, xproj, w, T, qkv_want);
    ref_gate(raw, dt, ssm_a, T, true, gate_want);
    std::vector<double> st_want = st0;
    ref_scan_norm(st_want, qkv_want, gate_want, beta, z, gamma, T, y_want);

    // ================= the rival readings are observable =================
    std::printf("rival readings this fixture tells apart:\n");
    {
        std::vector<double> s = st0;
        std::vector<float> yy;
        ref_scan_norm(s, qkv_want, gate_want, beta, z, gamma, T, yy, /*decay_on_value=*/true);
        expect_apart("1. decay on the key channel i vs the value channel j", y_want, yy);
    }
    {
        std::vector<double> s = st0;
        std::vector<float> yy;
        ref_scan_norm(s, qkv_want, gate_want, beta, z, gamma, T, yy, false, /*decay_after=*/true);
        expect_apart("2. decay before vs after the rank-1 update", y_want, yy);
    }
    {
        std::vector<float> gg;
        ref_gate(raw, dt, ssm_a, T, true, gg, /*flip_a=*/true);
        expect_apart("3. sigmoid(-A u) vs sigmoid(A u)", gate_want, gg);
    }
    {
        std::vector<float> h = hist0, qq;
        ref_conv_l2(h, xproj, w, T, qq, /*reverse_taps=*/true);
        expect_apart("4. conv tap 0 on the oldest vs the newest input", qkv_want, qq);
    }

    // ================= the kernels, one launch of T tokens =================
    std::printf("\nkernels vs reference, %d heads of %d, %d tokens in one launch:\n", H, D, T);
    float* d_hist = upload(hist0);
    float* d_x = upload(xproj);
    float* d_w[3] = {upload(w[0]), upload(w[1]), upload(w[2])};
    float* d_raw = upload(raw);
    float* d_dt = upload(dt);
    float* d_a = upload(ssm_a);
    float* d_beta = upload(beta);
    float* d_z = upload(z);
    float* d_gamma = upload(gamma);
    float* d_st = upload(to_dev(st0));
    float *d_qkv = nullptr, *d_gate = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_qkv, qkv_want.size() * 4), "qkv");
    check(cudaMalloc(&d_gate, gate_want.size() * 4), "gate");
    check(cudaMalloc(&d_y, y_want.size() * 4), "y");

    strata::kernels::kda_conv_l2(d_hist, d_x, d_w[0], d_w[1], d_w[2], d_qkv, H, T, kL2Eps, nullptr);
    strata::kernels::kda_gate(d_raw, d_dt, d_a, d_gate, H, T, true, kLowerBound, nullptr);
    check(cudaDeviceSynchronize(), "conv/gate");
    const std::vector<float> qkv_got = download(d_qkv, qkv_want.size());
    const std::vector<float> hist_got = download(d_hist, hist0.size());
    const std::vector<float> gate_got = download(d_gate, gate_want.size());
    expect_close("kda_conv_l2: q | k | v", qkv_want, qkv_got, 1e-5);
    expect_bitwise("kda_conv_l2: conv history (copied inputs)", hist_want, hist_got);
    expect_close("kda_gate (lower bound -5)", gate_want, gate_got, 1e-5);
    {
        // the gate lies in (lb, 0), so the decay never resets the state and never grows it; in float the sigmoid
        // saturates, and this fixture (A up to 16) reaches both ends exactly, as llama.cpp's float graph would
        float lo = 0, hi = -10;
        for (float g : gate_got) { lo = std::fmin(lo, g); hi = std::fmax(hi, g); }
        const bool ok = lo >= kLowerBound && hi <= 0.0f;
        std::printf("  %-52s %s [%.3f, %.3f]\n", "kda_gate inside [lower bound, 0]", ok ? "yes" : "*** NO ***", lo, hi);
        if (!ok) ++g_bad;
    }
    {
        std::vector<float> want, got;
        ref_gate(raw, dt, ssm_a, T, false, want);
        float* d_g2 = nullptr;
        check(cudaMalloc(&d_g2, want.size() * 4), "g2");
        strata::kernels::kda_gate(d_raw, d_dt, d_a, d_g2, H, T, false, 0.0f, nullptr);
        check(cudaDeviceSynchronize(), "gate softplus");
        got = download(d_g2, want.size());
        expect_close("kda_gate (no lower bound: softplus)", want, got, 1e-5);
        cudaFree(d_g2);
    }

    // the scan is fed the REFERENCE's conv and gate outputs, so its error is its own
    float* d_qkv_ref = upload(qkv_want);
    float* d_gate_ref = upload(gate_want);
    strata::kernels::kda_scan_norm(d_st, d_qkv_ref, d_gate_ref, d_beta, d_z, d_gamma, kRmsEps, d_y, H, T, nullptr);
    check(cudaDeviceSynchronize(), "scan");
    const std::vector<float> y_got = download(d_y, y_want.size());
    const std::vector<float> st_got = download(d_st, st0.size());
    expect_close("kda_scan_norm: y", y_want, y_got, 2e-5);
    expect_close("kda_scan_norm: state, read through (H, D, D)", to_float(st_want), from_dev(st_got), 2e-5);

    // one token: what decode runs
    {
        std::vector<float> h1 = hist0, q1, g1, y1, x1(xproj.begin(), xproj.begin() + 3 * DI), r1(raw.begin(), raw.begin() + DI);
        ref_conv_l2(h1, x1, w, 1, q1);
        ref_gate(r1, dt, ssm_a, 1, true, g1);
        std::vector<double> s1 = st0;
        ref_scan_norm(s1, q1, g1, beta, z, gamma, 1, y1);
        float* dh = upload(hist0);
        float* ds = upload(to_dev(st0));
        strata::kernels::kda_conv_l2(dh, d_x, d_w[0], d_w[1], d_w[2], d_qkv, H, 1, kL2Eps, nullptr);
        strata::kernels::kda_gate(d_raw, d_dt, d_a, d_gate, H, 1, true, kLowerBound, nullptr);
        strata::kernels::kda_scan_norm(ds, d_qkv, d_gate, d_beta, d_z, d_gamma, kRmsEps, d_y, H, 1, nullptr);
        check(cudaDeviceSynchronize(), "one token");
        expect_close("one token, conv -> gate -> scan chained: y", y1, download(d_y, (size_t) DI), 2e-5);
        expect_bitwise("one token: conv history slid in place", h1, download(dh, hist0.size()));
        expect_close("one token: state", to_float(s1), from_dev(download(ds, st0.size())), 2e-5);
        cudaFree(dh); cudaFree(ds);
    }

    // ================= split launches carry the state exactly =================
    std::printf("\n%d tokens as launches of 1 + 2 + 4, against the single launch:\n", T);
    {
        float* dh = upload(hist0);
        float* ds = upload(to_dev(st0));
        float* dy = nullptr;
        check(cudaMalloc(&dy, y_want.size() * 4), "y split");
        int t0 = 0;
        for (const int n : splits) {
            strata::kernels::kda_conv_l2(dh, d_x + (size_t) t0 * 3 * DI, d_w[0], d_w[1], d_w[2],
                                         d_qkv + (size_t) t0 * 3 * DI, H, n, kL2Eps, nullptr);
            strata::kernels::kda_scan_norm(ds, d_qkv_ref + (size_t) t0 * 3 * DI, d_gate_ref + (size_t) t0 * DI,
                                           d_beta + (size_t) t0 * H, d_z + (size_t) t0 * DI, d_gamma, kRmsEps,
                                           dy + (size_t) t0 * DI, H, n, nullptr);
            t0 += n;
        }
        check(cudaDeviceSynchronize(), "split");
        expect_bitwise("kda_conv_l2: q | k | v", qkv_got, download(d_qkv, qkv_want.size()));
        expect_bitwise("kda_conv_l2: conv history", hist_got, download(dh, hist0.size()));
        expect_bitwise("kda_scan_norm: y", y_got, download(dy, y_want.size()));
        expect_bitwise("kda_scan_norm: state", st_got, download(ds, st0.size()));
        cudaFree(dh); cudaFree(ds); cudaFree(dy);
    }

    // ================= the chunked scan =================
    std::printf("\nkda_scan_norm_chunked (chunks of %d, super-blocks of %d), against the reference and kda_scan_norm:\n",
                strata::kernels::kKdaChunk, strata::kernels::kKdaSuperBlock);
    {
        struct Case { int nh, T; bool ref; };
        // one token; a partial last chunk; three super-blocks, the last partial (8 heads keep the CPU quick); and
        // 2048 tokens of all 64 heads against the token-by-token kernel only
        const Case cases[] = {{H, 1, true}, {H, 100, true}, {8, 1100, true}, {H, 2048, false}};
        for (const Case& cs : cases) {
            const int nh = cs.nh, n = cs.T, di = nh * D;
            std::vector<float> qkv((size_t) n * 3 * di), gt((size_t) n * di), bt((size_t) n * nh), zt((size_t) n * di),
                ah((size_t) nh);
            for (auto& x : qkv) x = gauss(rng);
            for (int t = 0; t < n; ++t)   // q and k L2-normalized per head, as kda_conv_l2 leaves them
                for (int hh = 0; hh < 2 * nh; ++hh) {
                    float* p = &qkv[(size_t) t * 3 * di + (size_t) hh * D];
                    double ss = 0;
                    for (int i = 0; i < D; ++i) ss += (double) p[i] * (double) p[i];
                    for (int i = 0; i < D; ++i) p[i] = (float) ((double) p[i] / std::sqrt(ss + (double) kL2Eps));
                }
            for (auto& a : ah) a = -std::exp(unit(rng) * std::log(16.0f));
            for (int t = 0; t < n; ++t)   // the lower-bound gate of a unit-normal pre-activation
                for (int c = 0; c < di; ++c)
                    gt[(size_t) t * di + c] =
                        (float) ((double) kLowerBound * sigmoid_d(-(double) ah[(size_t) (c / D)] * (double) gauss(rng)));
            for (auto& x : bt) x = (float) sigmoid_d(gauss(rng));
            for (auto& x : zt) x = gauss(rng);
            std::vector<double> s0((size_t) nh * D * D);
            for (auto& x : s0) x = 0.05 * gauss(rng);

            float *dq = upload(qkv), *dg = upload(gt), *db = upload(bt), *dz = upload(zt);
            float *ds_seq = upload(to_dev(s0, nh)), *ds_ch = upload(to_dev(s0, nh));
            float *dy_seq = nullptr, *dy_ch = nullptr, *dscr = nullptr;
            check(cudaMalloc(&dy_seq, (size_t) n * di * 4), "y seq");
            check(cudaMalloc(&dy_ch, (size_t) n * di * 4), "y chunked");
            check(cudaMalloc(&dscr, strata::kernels::kda_chunked_scratch_floats(nh, n) * 4), "scratch");
            strata::kernels::kda_scan_norm(ds_seq, dq, dg, db, dz, d_gamma, kRmsEps, dy_seq, nh, n, nullptr);
            strata::kernels::kda_scan_norm_chunked(ds_ch, dq, dg, db, dz, d_gamma, kRmsEps, dy_ch, nh, n, dscr, nullptr);
            check(cudaDeviceSynchronize(), "chunked");
            const std::vector<float> y_seq = download(dy_seq, (size_t) n * di), y_ch = download(dy_ch, (size_t) n * di);
            const std::vector<float> st_seq = from_dev(download(ds_seq, s0.size()), nh);
            const std::vector<float> st_ch = from_dev(download(ds_ch, s0.size()), nh);
            char what[96];
            if (cs.ref) {
                std::vector<double> s_ref = s0;
                std::vector<float> y_ref;
                ref_scan_norm(s_ref, qkv, gt, bt, zt, gamma, n, y_ref, false, false, nh);
                std::snprintf(what, sizeof what, "%2d heads x %4d tokens: y vs reference", nh, n);
                expect_close(what, y_ref, y_ch, 2e-5);
                std::snprintf(what, sizeof what, "%2d heads x %4d tokens: state vs reference", nh, n);
                expect_close(what, to_float(s_ref), st_ch, 2e-5);
            }
            std::snprintf(what, sizeof what, "%2d heads x %4d tokens: y vs kda_scan_norm", nh, n);
            expect_max_rel(what, y_seq, y_ch, 1e-4);
            std::snprintf(what, sizeof what, "%2d heads x %4d tokens: state vs kda_scan_norm", nh, n);
            expect_max_rel(what, st_seq, st_ch, 1e-4);
            for (float* p : {dq, dg, db, dz, ds_seq, ds_ch, dy_seq, dy_ch, dscr}) cudaFree(p);
        }
    }

    // ================= --bench: the two scans over a prompt =================
    if (bench) {
        std::printf("\nkda_scan_norm vs kda_scan_norm_chunked, all %d heads, best of 5 launches (host clock: launch, wait):\n", H);
        for (const int n : {32, 128, 512, 2048, 8192}) {
            std::vector<float> qkv((size_t) n * 3 * DI), gt((size_t) n * DI), bt((size_t) n * H), zt((size_t) n * DI);
            for (auto& x : qkv) x = 0.088f * gauss(rng);   // about unit rows of 128
            for (auto& x : gt) x = kLowerBound * unit(rng);
            for (auto& x : bt) x = unit(rng);
            for (auto& x : zt) x = gauss(rng);
            float *dq = upload(qkv), *dg = upload(gt), *db = upload(bt), *dz = upload(zt), *ds = upload(to_dev(st0));
            float *dy = nullptr, *dscr = nullptr;
            check(cudaMalloc(&dy, (size_t) n * DI * 4), "y");
            check(cudaMalloc(&dscr, strata::kernels::kda_chunked_scratch_floats(H, n) * 4), "scratch");
            // the host clock: HIP's events under Windows can close before the work has run
            double best[2] = {1e30, 1e30};
            for (int it = 0; it < 6; ++it)
                for (int which = 0; which < 2; ++which) {   // alternate, so a clock step affects both alike
                    check(cudaDeviceSynchronize(), "sync");
                    const auto c0 = std::chrono::steady_clock::now();
                    if (which == 0) strata::kernels::kda_scan_norm(ds, dq, dg, db, dz, d_gamma, kRmsEps, dy, H, n, nullptr);
                    else strata::kernels::kda_scan_norm_chunked(ds, dq, dg, db, dz, d_gamma, kRmsEps, dy, H, n, dscr, nullptr);
                    check(cudaDeviceSynchronize(), "sync");
                    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
                    if (it > 0) best[which] = std::fmin(best[which], ms);   // the first is a warm-up
                }
            std::printf("  T = %-5d  token by token %9.3f ms (%5.2f us/token)   chunked %9.3f ms (%5.2f us/token)   %.2fx\n", n,
                        best[0], 1000.0 * best[0] / n, best[1], 1000.0 * best[1] / n, best[0] / best[1]);
            for (float* p : {dq, dg, db, dz, ds, dy, dscr}) cudaFree(p);
        }
    }

    for (float* p : {d_hist, d_x, d_w[0], d_w[1], d_w[2], d_raw, d_dt, d_a, d_beta, d_z, d_gamma, d_st, d_qkv, d_gate,
                     d_y, d_qkv_ref, d_gate_ref})
        cudaFree(p);

    // ================= --bench: one layer's time per token, decode and a prompt run =================
    if (bench) {
        std::printf("\none KDA layer (the parts here, no projections), mean of 200 launches:\n");
        for (const int n : {1, 512}) {
            std::vector<float> xs((size_t) n * 3 * DI), rs((size_t) n * DI), bs((size_t) n * H), zs((size_t) n * DI);
            for (auto& x : xs) x = gauss(rng);
            for (auto& x : rs) x = gauss(rng);
            for (auto& x : bs) x = (float) sigmoid_d(gauss(rng));
            for (auto& x : zs) x = gauss(rng);
            float *bh = upload(hist0), *bx = upload(xs), *br = upload(rs), *bb = upload(bs), *bz = upload(zs);
            float *bst = upload(to_dev(st0)), *bw[3] = {upload(w[0]), upload(w[1]), upload(w[2])};
            float *bdt = upload(dt), *ba = upload(ssm_a), *bg = upload(gamma);
            float *bq = nullptr, *bgt = nullptr, *by = nullptr;
            check(cudaMalloc(&bq, xs.size() * 4), "bq");
            check(cudaMalloc(&bgt, rs.size() * 4), "bgt");
            check(cudaMalloc(&by, rs.size() * 4), "by");
            cudaEvent_t e0, e1, e2, e3;
            check(cudaEventCreate(&e0), "ev"); check(cudaEventCreate(&e1), "ev");
            check(cudaEventCreate(&e2), "ev"); check(cudaEventCreate(&e3), "ev");
            float ms_conv = 0, ms_gate = 0, ms_scan = 0;
            constexpr int kIters = 200;
            for (int it = -10; it < kIters; ++it) {
                check(cudaEventRecord(e0, nullptr), "rec");
                strata::kernels::kda_conv_l2(bh, bx, bw[0], bw[1], bw[2], bq, H, n, kL2Eps, nullptr);
                check(cudaEventRecord(e1, nullptr), "rec");
                strata::kernels::kda_gate(br, bdt, ba, bgt, H, n, true, kLowerBound, nullptr);
                check(cudaEventRecord(e2, nullptr), "rec");
                strata::kernels::kda_scan_norm(bst, bq, bgt, bb, bz, bg, kRmsEps, by, H, n, nullptr);
                check(cudaEventRecord(e3, nullptr), "rec");
                check(cudaEventSynchronize(e3), "sync");
                if (it < 0) continue;     // warm-up
                float a = 0, b = 0, c = 0;
                check(cudaEventElapsedTime(&a, e0, e1), "el");
                check(cudaEventElapsedTime(&b, e1, e2), "el");
                check(cudaEventElapsedTime(&c, e2, e3), "el");
                ms_conv += a; ms_gate += b; ms_scan += c;
            }
            const double k = 1000.0 / kIters;     // us per launch
            std::printf("  T = %-4d conv+l2 %8.1f us   gate %7.1f us   scan+norm %8.1f us   (%.2f us/token in all)\n",
                        n, ms_conv * k, ms_gate * k, ms_scan * k, (ms_conv + ms_gate + ms_scan) * k / n);
            for (float* p : {bh, bx, br, bb, bz, bst, bw[0], bw[1], bw[2], bdt, ba, bg, bq, bgt, by}) cudaFree(p);
            cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
        }
    }

    std::printf("\nkda: %d failures\n", g_bad);
    if (g_bad) return 1;
    if (selftest) std::printf("kda_parity OK\n");
    return 0;
}
