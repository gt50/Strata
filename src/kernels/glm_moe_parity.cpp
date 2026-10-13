// src/kernels/glm_moe_parity.cpp - GLM-5.3-Flash's MoE kernels: glm_router and swiglu_clamp (glm_moe.hpp) against a
// double transcription of llama.cpp's build_moe_ffn, and native_expert_grouped_clamped (iq_kernels.hpp) at GLM's
// real expert shape against the same experts run one GEMV at a time.
//
//   1. glm_router, 64 tokens of 288 experts, top 8, sigmoid, the selection bias, norm, x2.5: the chosen experts
//      and their order exactly, the weights to FP32 rounding; also softmax and sqrt-softplus gating.  Rivals shown to
//      differ first: choosing without the bias, weighing with it, softmax for sigmoid.
//   2. swiglu_clamp: silu(min(g, 10)) * clamp(u, -10, 10) over values to +-40; the other arches' form,
//      min(silu(g), 10) * clamp(u), shown to differ.
//   3. native_expert_grouped_clamped, IQ4_XS gate / up and down, 4096 x 2048 (n_embd x n_ff), 6 experts and 16
//      tokens: against iq_mmvq for gate and up, the clamped SwiGLU on the host, quantize_q8_1_rows and iq_mmvq for
//      down - in every expert layout the card runs (the default, v1's separate SwiGLU pass, modes 0, 7 and 8).  The
//      block scales put the gate and up values around +-10, so the clamp matters: limit 10 is shown to differ from
//      none, and limit 0 is bitwise native_expert_grouped.
//   4. the other expert types of Unsloth's UD-IQ4_XS file: IQ3_S / IQ4_XS (most layers), IQ4_XS / Q6_K (layer 11),
//      IQ3_S / Q6_K (12, 44) and Q3_K / Q4_K (the MTP block), against glm_linear's GEMVs (native_mmvq: other code)
//      for gate, up and down with the clamped SwiGLU on the host between them.
#include "strata/kernels/glm_linear.hpp"
#include "strata/kernels/glm_moe.hpp"
#include "strata/kernels/iq_kernels.hpp"

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

namespace k = strata::kernels;

namespace {

constexpr int NE = 288, KU = 8;            // expert_count, expert_used_count
constexpr float kScale = 2.5f, kLim = 10.0f;

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
    check(cudaMalloc(&p, v.size() * sizeof(T) + 256), "cudaMalloc");
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
    std::printf("  %-64s %s%s\n", what.c_str(), ok ? "yes" : "*** NO ***", detail.c_str());
    if (!ok) ++g_bad;
}
void expect_close(const std::string& what, const std::vector<float>& want, const std::vector<float>& got, double tol) {
    const double rel = rel_l1(want, got);
    const bool ok = rel <= tol;
    std::printf("  %-64s rel %.3e %s\n", what.c_str(), rel, ok ? "" : "*** over tolerance ***");
    if (!ok) ++g_bad;
}
void expect_apart(const std::string& what, const std::vector<float>& a, const std::vector<float>& b) {
    const double rel = rel_l1(a, b);
    char buf[48];
    std::snprintf(buf, sizeof buf, " (%.2f%% apart)", rel * 100);
    expect(rel > 0.05, what, buf);
}

struct Route { std::vector<int32_t> ids; std::vector<float> w; double margin = 1e30; };

/// build_moe_ffn's routing for one token, in double: `bias_choose` / `bias_weigh` are the readings under test.
Route ref_route(const float* lg, const std::vector<float>& bias, int gating, bool bias_choose, bool bias_weigh) {
    std::vector<double> p((size_t) NE);
    if (gating == k::kGatingSoftmax) {
        double mx = -INFINITY, s = 0;
        for (int e = 0; e < NE; ++e) mx = std::max(mx, (double) lg[e]);
        for (int e = 0; e < NE; ++e) s += std::exp((double) lg[e] - mx);
        for (int e = 0; e < NE; ++e) p[(size_t) e] = std::exp((double) lg[e] - mx) / s;
    } else if (gating == k::kGatingSqrtSoftplus) {
        for (int e = 0; e < NE; ++e) p[(size_t) e] = std::sqrt(std::log1p(std::exp((double) lg[e])));
    } else {
        for (int e = 0; e < NE; ++e) p[(size_t) e] = 1.0 / (1.0 + std::exp(-(double) lg[e]));
    }
    std::vector<double> sel(p);
    if (bias_choose)
        for (int e = 0; e < NE; ++e) sel[(size_t) e] += bias[(size_t) e];
    std::vector<int> order((size_t) NE);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sel[(size_t) a] > sel[(size_t) b]; });
    Route r;
    r.margin = sel[(size_t) order[KU - 1]] - sel[(size_t) order[KU]];
    double sum = 0;
    for (int i = 0; i < KU; ++i) {
        const int e = order[(size_t) i];
        r.ids.push_back(e);
        sum += p[(size_t) e] + (bias_weigh ? bias[(size_t) e] : 0.0);
    }
    for (int i = 0; i < KU; ++i) {
        const int e = order[(size_t) i];
        const double w = p[(size_t) e] + (bias_weigh ? bias[(size_t) e] : 0.0);
        r.w.push_back((float) (w / std::max(sum, 6.103515625e-5) * kScale));
    }
    return r;
}

// IQ4_XS rows (136 bytes per 256 values): random codes, the fp16 block scale d = +-2^(e - 15) (e in [e_lo, e_hi])
std::vector<uint8_t> iq4xs_rows(int64_t rows, int64_t n, int e_lo, int e_hi, std::mt19937& rng) {
    const size_t bs = 136;
    std::vector<uint8_t> w((size_t) rows * (size_t) (n / 256) * bs);
    std::uniform_int_distribution<int> byte(0, 255), ex(e_lo, e_hi), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}

// Rows of a GGUF block type: random bytes, then each block's fp16 scales (at `halves`, bytes into the block) set to
// +-2^(e - 15) with e in [e_lo, e_hi] - random bytes there would give infinities and NaNs.
struct BlockType { int type; size_t bytes; std::vector<size_t> halves; int e_lo, e_hi; const char* name; };
std::vector<uint8_t> gguf_rows(const BlockType& bt, int64_t rows, int64_t n, std::mt19937& rng) {
    std::vector<uint8_t> w((size_t) rows * (size_t) (n / 256) * bt.bytes);
    std::uniform_int_distribution<int> byte(0, 255), ex(bt.e_lo, bt.e_hi), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bt.bytes)
        for (const size_t at : bt.halves) {
            const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
            std::memcpy(&w[o + at], &h, 2);
        }
    return w;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // the lines before a crash are kept
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: glm_moe_parity [--selftest]\n"); return 2; }
    }
    std::mt19937 rng(2880);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // ================= 1. the router =================
    constexpr int T = 64;
    std::printf("glm_router, %d tokens of %d experts, top %d, x%.1f:\n", T, NE, KU, kScale);
    {
        std::vector<float> logits((size_t) T * NE), bias((size_t) NE);
        for (auto& x : logits) x = 1.5f * gauss(rng);
        for (auto& x : bias) x = 0.3f * gauss(rng);   // of the order of the probabilities' spread at the top
        float *d_lg = upload(logits), *d_b = upload(bias), *d_w = nullptr;
        int32_t* d_ids = nullptr;
        check(cudaMalloc(&d_w, (size_t) T * KU * 4), "w");
        check(cudaMalloc(&d_ids, (size_t) T * KU * 4), "ids");
        for (const int gating : {(int) k::kGatingSigmoid, (int) k::kGatingSoftmax, (int) k::kGatingSqrtSoftplus}) {
            k::glm_router(d_lg, d_b, T, NE, KU, gating, true, kScale, d_ids, d_w, nullptr);
            check(cudaDeviceSynchronize(), "router");
            const std::vector<int32_t> ids = download(d_ids, (size_t) T * KU);
            const std::vector<float> w = download(d_w, (size_t) T * KU);
            int same = 0, near_tie = 0;
            std::vector<float> want_w, got_w, rival_w, soft_w;
            int rival_sets = 0;
            for (int t = 0; t < T; ++t) {
                const Route r = ref_route(&logits[(size_t) t * NE], bias, gating, true, false);
                const bool eq = std::equal(r.ids.begin(), r.ids.end(), ids.begin() + (size_t) t * KU);
                same += eq;
                near_tie += !eq && r.margin < 1e-5;   // a float rounding may break a near tie either way
                want_w.insert(want_w.end(), r.w.begin(), r.w.end());
                got_w.insert(got_w.end(), w.begin() + (size_t) t * KU, w.begin() + (size_t) (t + 1) * KU);
                if (gating == k::kGatingSigmoid) {
                    const Route nb = ref_route(&logits[(size_t) t * NE], bias, gating, false, false);
                    std::vector<int32_t> a = nb.ids, b = r.ids;
                    std::sort(a.begin(), a.end());
                    std::sort(b.begin(), b.end());
                    rival_sets += a != b;
                    const Route bw = ref_route(&logits[(size_t) t * NE], bias, gating, true, true);
                    rival_w.insert(rival_w.end(), bw.w.begin(), bw.w.end());
                    const Route sm = ref_route(&logits[(size_t) t * NE], bias, k::kGatingSoftmax, true, false);
                    soft_w.insert(soft_w.end(), sm.w.begin(), sm.w.end());
                }
            }
            const char* gname = gating == k::kGatingSigmoid ? "sigmoid" : gating == k::kGatingSoftmax ? "softmax" : "sqrt(softplus)";
            if (gating == k::kGatingSigmoid) {
                char buf[64];
                std::snprintf(buf, sizeof buf, " (%d of %d tokens choose otherwise)", rival_sets, T);
                expect(rival_sets > T / 4, "rival: choosing by the probabilities without the bias", buf);
                expect_apart("rival: weighing with the bias too", want_w, rival_w);
                expect_apart("rival: softmax for sigmoid", want_w, soft_w);
            }
            char buf[64];
            std::snprintf(buf, sizeof buf, " (%d of %d; %d near ties)", same, T, near_tie);
            expect(same + near_tie == T, std::string(gname) + ": the chosen experts, in order", buf);
            expect_close(std::string(gname) + ": the weights", want_w, got_w, 1e-6);
        }
        cudaFree(d_lg); cudaFree(d_b); cudaFree(d_w); cudaFree(d_ids);
    }

    // ================= 2. the clamped SwiGLU =================
    std::printf("\nswiglu_clamp, limit %.0f:\n", kLim);
    {
        constexpr int N = 4096;
        std::vector<float> g(N), u(N);
        for (auto& x : g) x = 12.0f * gauss(rng);
        for (auto& x : u) x = 12.0f * gauss(rng);
        std::vector<float> want(N), other(N), none(N);
        for (int i = 0; i < N; ++i) {
            const double gc = std::min((double) g[i], (double) kLim), uc = std::clamp((double) u[i], -(double) kLim, (double) kLim);
            want[i] = (float) (gc / (1.0 + std::exp(-gc)) * uc);
            const double s = g[i] / (1.0 + std::exp(-(double) g[i]));
            other[i] = (float) (std::min(s, (double) kLim) * uc);
            none[i] = (float) (s * u[i]);
        }
        // the two forms differ only where g > L, and there by silu(L) against L: 4.5e-5 relative at L = 10
        int differ = 0, past = 0;
        for (int i = 0; i < N; ++i) {
            differ += std::fabs(want[i] - other[i]) > 1e-6f * std::fabs(want[i]);
            past += g[i] > kLim && u[i] != 0.0f;
        }
        expect(differ > 0 && differ == past, "rival: min(silu(g), L) for silu(min(g, L))",
               " (" + std::to_string(differ) + " values differ: every one with g > 10)");
        float *d_g = upload(g), *d_u = upload(u), *d_h = nullptr;
        check(cudaMalloc(&d_h, N * 4), "h");
        k::swiglu_clamp(d_g, d_u, d_h, N, kLim, nullptr);
        check(cudaDeviceSynchronize(), "swiglu");
        expect_close("h, limit 10", want, download(d_h, N), 1e-6);
        k::swiglu_clamp(d_g, d_u, d_h, N, 0.0f, nullptr);
        check(cudaDeviceSynchronize(), "swiglu 0");
        expect_close("h, no limit", none, download(d_h, N), 1e-6);
        cudaFree(d_g); cudaFree(d_u); cudaFree(d_h);
    }

    // ================= 3. the routed experts =================
    constexpr int64_t H = 4096, FF = 2048;   // n_embd, expert_feed_forward_length
    constexpr int G = 6, TT = 16;            // experts, tokens
    constexpr int TY = 23;                   // IQ4_XS
    std::printf("\nnative_expert_grouped_clamped, IQ4_XS, %lld x %lld, %d experts, %d tokens:\n", (long long) H,
                (long long) FF, G, TT);
    {
        const k::NativeExpertLayout L = k::native_expert_layout(TY, TY, H, FF);
        expect(k::native_expert_supported(TY, TY, H, FF), "the grouped kernels take this shape and these types");
        const size_t slot = (L.bytes + 255) & ~(size_t) 255;
        std::vector<std::vector<uint8_t>> blobs(G);
        uint8_t* d_blobs = nullptr;
        check(cudaMalloc(&d_blobs, slot * G), "blobs");
        for (int b = 0; b < G; ++b) {
            // gate / up: d = 2^-13 .. 2^-12, so that over 4096 unit inputs some products pass +-10
            blobs[(size_t) b] = iq4xs_rows(2 * FF, H, 2, 3, rng);
            const auto down = iq4xs_rows(H, FF, 2, 4, rng);
            blobs[(size_t) b].insert(blobs[(size_t) b].end(), down.begin(), down.end());
            if (blobs[(size_t) b].size() != L.bytes) { std::fprintf(stderr, "blob size\n"); return 2; }
            check(cudaMemcpy(d_blobs + slot * b, blobs[(size_t) b].data(), L.bytes, cudaMemcpyHostToDevice), "blob");
        }
        std::vector<float> x((size_t) TT * H);
        for (auto& v : x) v = gauss(rng);
        float* d_x = upload(x);
        uint8_t* d_xq = nullptr;
        check(cudaMalloc(&d_xq, (size_t) TT * (H / 32) * 36), "xq");
        k::quantize_q8_1_rows(d_x, TT, H, d_xq, nullptr);
        // entries: every token to experts (t % G) and ((t + 1) % G) - groups of 5 or 6 entries, 32 in all
        std::vector<int32_t> start, dst, tok;
        std::vector<unsigned long long> ptr;
        int e = 0;
        for (int b = 0; b < G; ++b) {
            start.push_back(e);
            ptr.push_back((unsigned long long) (uintptr_t) (d_blobs + slot * b));
            for (int t = 0; t < TT; ++t)
                if (t % G == b || (t + 1) % G == b) { dst.push_back(e); tok.push_back(t); ++e; }
        }
        start.push_back(e);
        const int NENT = e;
        const int32_t ng = G;
        unsigned long long* d_ptr = upload(ptr);
        int32_t *d_start = upload(start), *d_dst = upload(dst), *d_tok = upload(tok), *d_ng = upload(std::vector<int32_t>{ng});
        void* d_scr = nullptr;
        check(cudaMalloc(&d_scr, k::native_expert_scratch_bytes(NENT, FF)), "scratch");
        float* d_out = nullptr;
        check(cudaMalloc(&d_out, (size_t) NENT * H * 4), "out");

        // the reference: one GEMV at a time with the same q8_1 activations
        std::vector<float> want((size_t) NENT * H), want_none((size_t) NENT * H);
        int clamped = 0, values = 0;
        {
            float *d_g = nullptr, *d_u = nullptr, *d_h = nullptr, *d_y = nullptr;
            uint8_t* d_hq = nullptr;
            check(cudaMalloc(&d_g, FF * 4), "g");
            check(cudaMalloc(&d_u, FF * 4), "u");
            check(cudaMalloc(&d_h, FF * 4), "h");
            check(cudaMalloc(&d_y, H * 4), "y");
            check(cudaMalloc(&d_hq, (FF / 32) * 36), "hq");
            for (int b = 0; b < G; ++b)
                for (int en = start[(size_t) b]; en < start[(size_t) b + 1]; ++en) {
                    const uint8_t* blob = d_blobs + slot * b;
                    const uint8_t* xq = d_xq + (size_t) tok[(size_t) en] * (H / 32) * 36;
                    k::iq_mmvq(TY, blob, xq, d_g, (int) H, (int) FF, 1, nullptr);
                    k::iq_mmvq(TY, blob + L.up_off, xq, d_u, (int) H, (int) FF, 1, nullptr);
                    check(cudaDeviceSynchronize(), "ref gu");
                    const std::vector<float> g = download(d_g, FF), u = download(d_u, FF);
                    for (int lim_i = 0; lim_i < 2; ++lim_i) {
                        std::vector<float> h(FF);
                        for (int i = 0; i < FF; ++i) {
                            float gg = g[(size_t) i], uu = u[(size_t) i];
                            if (lim_i == 0) {
                                clamped += gg > kLim || std::fabs(uu) > kLim;
                                values += 2;
                                gg = std::min(gg, kLim);
                                uu = std::clamp(uu, -kLim, kLim);
                            }
                            h[(size_t) i] = gg / (1.0f + std::exp(-gg)) * uu;
                        }
                        check(cudaMemcpy(d_h, h.data(), FF * 4, cudaMemcpyHostToDevice), "h");
                        k::quantize_q8_1_rows(d_h, 1, FF, d_hq, nullptr);
                        k::iq_mmvq(TY, blob + L.down_off, d_hq, d_y, (int) FF, (int) H, 1, nullptr);
                        check(cudaDeviceSynchronize(), "ref down");
                        const std::vector<float> y = download(d_y, H);
                        std::copy(y.begin(), y.end(), (lim_i == 0 ? want : want_none).begin() + (size_t) en * H);
                    }
                }
            cudaFree(d_g); cudaFree(d_u); cudaFree(d_h); cudaFree(d_y); cudaFree(d_hq);
        }
        char buf[64];
        std::snprintf(buf, sizeof buf, " (%.1f%% of gate / up values past it)", 100.0 * clamped / values);
        expect_apart("the limit matters on this fixture: limit 10 against none", want, want_none);
        std::printf("  %-64s%s\n", "", buf);

        auto run = [&](float lim) {
            k::native_expert_grouped_clamped(L, d_ptr, d_start, d_ng, d_dst, d_tok, G, NENT, d_xq, d_scr, d_out, lim, nullptr);
            check(cudaDeviceSynchronize(), "grouped");
            return download(d_out, (size_t) NENT * H);
        };
        // the summation orders differ from the GEMVs', and a q8_1 rounding of h can follow: a few 1e-4
        struct Variant { const char* name; bool v1; int mode; };
        for (const Variant v : {Variant{"the default layout", false, -1}, Variant{"v1: SwiGLU and q8_1 as two passes", true, -1},
                                Variant{"mode 0", false, 0}, Variant{"mode 7", false, 7}, Variant{"mode 8 (SwiGLU in the gate/up epilogue)", false, 8}}) {
            k::native_grouped_set_v1(v.v1);
            k::native_expert_set_mode(v.mode, 0);
            expect_close(std::string(v.name) + ", limit 10", want, run(kLim), 2e-3);
        }
        k::native_grouped_set_v1(false);
        k::native_expert_set_mode(-1, 0);
        {
            const std::vector<float> zero = run(0.0f);
            k::native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, G, NENT, d_xq, d_scr, d_out, nullptr);
            check(cudaDeviceSynchronize(), "grouped plain");
            expect(zero == download(d_out, (size_t) NENT * H), "limit 0: bitwise native_expert_grouped");
            expect_close("limit 0 against the unclamped reference", want_none, zero, 2e-3);
        }
        cudaFree(d_blobs); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_ptr); cudaFree(d_start); cudaFree(d_dst);
        cudaFree(d_tok); cudaFree(d_ng); cudaFree(d_scr); cudaFree(d_out);
    }

    // ================= 4. the file's other expert types =================
    {
        constexpr int64_t H = 4096, FF = 2048;
        constexpr int G = 4, TT = 8;
        cudaStream_t st = nullptr;   // native_mmvq, under glm_linear, takes no default stream
        check(cudaStreamCreate(&st), "stream");
        // the scales chosen so a gate / up value is of the order of 10 over 4096 unit inputs, as in 3
        const BlockType iq3s{21, 110, {0}, 6, 7, "IQ3_S"}, iq4xs{23, 136, {0}, 2, 3, "IQ4_XS"},
            q6k{14, 210, {208}, 3, 4, "Q6_K"}, q4k{12, 144, {0, 2}, 5, 6, "Q4_K"}, q3k{11, 110, {108}, 7, 8, "Q3_K"};
        struct Pair { BlockType gu, d; };
        for (const Pair& pr : {Pair{iq3s, iq4xs}, Pair{iq4xs, q6k}, Pair{iq3s, q6k}, Pair{q3k, q4k}}) {
            std::printf("\nnative_expert_grouped_clamped, %s gate / up, %s down, %lld x %lld, %d experts, %d tokens:\n",
                        pr.gu.name, pr.d.name, (long long) H, (long long) FF, G, TT);
            const bool ok = k::native_expert_supported(pr.gu.type, pr.d.type, H, FF);
            expect(ok, "the grouped kernels take this shape and these types");
            if (!ok) continue;
            const k::NativeExpertLayout L = k::native_expert_layout(pr.gu.type, pr.d.type, H, FF);
            const size_t slot = (L.bytes + 255) & ~(size_t) 255;
            uint8_t* d_blobs = nullptr;
            check(cudaMalloc(&d_blobs, slot * G), "blobs");
            for (int b = 0; b < G; ++b) {
                std::vector<uint8_t> blob = gguf_rows(pr.gu, 2 * FF, H, rng);
                const auto down = gguf_rows(pr.d, H, FF, rng);
                blob.insert(blob.end(), down.begin(), down.end());
                if (blob.size() != L.bytes) {
                    std::fprintf(stderr, "blob size %zu, layout %zu\n", blob.size(), L.bytes);
                    return 2;
                }
                check(cudaMemcpy(d_blobs + slot * b, blob.data(), L.bytes, cudaMemcpyHostToDevice), "blob");
            }
            std::vector<float> x((size_t) TT * H);
            for (auto& v : x) v = gauss(rng);
            float* d_x = upload(x);
            uint8_t* d_xq = nullptr;
            check(cudaMalloc(&d_xq, (size_t) TT * (H / 32) * 36), "xq");
            k::quantize_q8_1_rows(d_x, TT, H, d_xq, nullptr);
            std::vector<int32_t> start, dst, tok;   // token t to experts t % G and (t + 1) % G
            std::vector<unsigned long long> ptr;
            int e = 0;
            for (int b = 0; b < G; ++b) {
                start.push_back(e);
                ptr.push_back((unsigned long long) (uintptr_t) (d_blobs + slot * b));
                for (int t = 0; t < TT; ++t)
                    if (t % G == b || (t + 1) % G == b) { dst.push_back(e); tok.push_back(t); ++e; }
            }
            start.push_back(e);
            const int NENT = e;
            unsigned long long* d_ptr = upload(ptr);
            int32_t *d_start = upload(start), *d_dst = upload(dst), *d_tok = upload(tok),
                    *d_ng = upload(std::vector<int32_t>{G});
            void* d_scr = nullptr;
            check(cudaMalloc(&d_scr, k::native_expert_scratch_bytes(NENT, FF)), "scratch");
            float* d_out = nullptr;
            check(cudaMalloc(&d_out, (size_t) NENT * H * 4), "out");
            // the reference: glm_linear's GEMVs a role at a time on the same x
            std::vector<float> want((size_t) NENT * H);
            int clamped = 0, values = 0;
            {
                float *d_g = nullptr, *d_u = nullptr, *d_h = nullptr, *d_y = nullptr;
                void* d_lq = nullptr;
                check(cudaMalloc(&d_g, FF * 4), "g");
                check(cudaMalloc(&d_u, FF * 4), "u");
                check(cudaMalloc(&d_h, FF * 4), "h");
                check(cudaMalloc(&d_y, H * 4), "y");
                check(cudaMalloc(&d_lq, k::glm_linear_scratch_bytes(H, 1)), "lq");
                for (int b = 0; b < G; ++b)
                    for (int en = start[(size_t) b]; en < start[(size_t) b + 1]; ++en) {
                        const uint8_t* blob = d_blobs + slot * b;
                        const float* xt = d_x + (size_t) tok[(size_t) en] * H;
                        k::glm_linear({pr.gu.type, blob, H, FF}, xt, d_g, 1, d_lq, st);
                        k::glm_linear({pr.gu.type, blob + L.up_off, H, FF}, xt, d_u, 1, d_lq, st);
                        check(cudaDeviceSynchronize(), "ref gu");
                        const std::vector<float> g = download(d_g, FF), u = download(d_u, FF);
                        std::vector<float> h(FF);
                        for (int i = 0; i < FF; ++i) {
                            clamped += g[(size_t) i] > kLim || std::fabs(u[(size_t) i]) > kLim;
                            values += 2;
                            const float gg = std::min(g[(size_t) i], kLim), uu = std::clamp(u[(size_t) i], -kLim, kLim);
                            h[(size_t) i] = gg / (1.0f + std::exp(-gg)) * uu;
                        }
                        check(cudaMemcpy(d_h, h.data(), FF * 4, cudaMemcpyHostToDevice), "h");
                        k::glm_linear({pr.d.type, blob + L.down_off, FF, H}, d_h, d_y, 1, d_lq, st);
                        check(cudaDeviceSynchronize(), "ref down");
                        const std::vector<float> y = download(d_y, H);
                        std::copy(y.begin(), y.end(), want.begin() + (size_t) en * H);
                    }
                cudaFree(d_g); cudaFree(d_u); cudaFree(d_h); cudaFree(d_y); cudaFree(d_lq);
            }
            char buf[64];
            std::snprintf(buf, sizeof buf, " (%.1f%% of gate / up values past it)", 100.0 * clamped / values);
            std::printf("  %-64s%s\n", "limit 10", buf);
            for (const int mode : {-1, 0}) {
                k::native_expert_set_mode(mode, 0);
                k::native_expert_grouped_clamped(L, d_ptr, d_start, d_ng, d_dst, d_tok, G, NENT, d_xq, d_scr, d_out,
                                                 kLim, nullptr);
                check(cudaDeviceSynchronize(), "grouped");
                expect_close(mode < 0 ? "the default layout, limit 10" : "mode 0, limit 10", want,
                             download(d_out, (size_t) NENT * H), 2e-3);
            }
            k::native_expert_set_mode(-1, 0);
            cudaFree(d_blobs); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_ptr); cudaFree(d_start); cudaFree(d_dst);
            cudaFree(d_tok); cudaFree(d_ng); cudaFree(d_scr); cudaFree(d_out);
        }
        cudaStreamDestroy(st);
    }

    std::printf("\nglm_moe: %d failures\n", g_bad);
    if (g_bad) return 1;
    if (selftest) std::printf("glm_moe_parity OK\n");
    return 0;
}
