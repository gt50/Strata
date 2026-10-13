// src/glm/glm_ref.hpp - the GLM tests' double-precision transcription of llama.cpp's glm5-next layer (upstream
// src/models/glm5-next.cpp), and the synthetic weights both tests build: glm_layer_parity (one block at a time, from
// a long history) and glm_model_parity (the whole model from a new sequence, and the MTP block).  Each test is one
// translation unit; this header is included once per test.
//
// The reference is exact arithmetic on the same numbers: the quantized weights read back through iq_dequant_f32,
// and before a quantized projection the activation rounded to q8_1 as the kernels round it (a scale amax / 127 kept
// in FP16, the values rounded to integers), so what remains is FP32 summation against double - except where a q8_1
// rounding falls on a boundary and the device's value, 1e-7 away, goes the other way.  verify (at the end) searches
// for those: a result passes against the reference as rounded, or with a set of its boundary roundings taken the
// other way that explains the result and the state the block carries together.
#pragma once

#include "strata/glm/glm_gemm.hpp"
#include "strata/glm/glm_layer.hpp"
#include "strata/glm/glm_model.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/kpool.hpp"

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
namespace glm = strata::glm;
using strata::core::Glm5NextGeometry;
using std::vector;

// each test uses its own part of what follows
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <class T> T* upload(const vector<T>& v) {
    T* p = nullptr;
    check(cudaMalloc(&p, v.size() * sizeof(T) + 256), "cudaMalloc");
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template <class T> vector<T> download(const T* p, size_t n) {
    vector<T> v(n);
    check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}
double rel_l1(const vector<double>& want, const vector<float>& got) {
    double d = 0, m = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        d += std::fabs(want[i] - (double) got[i]);
        m += std::fabs(want[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

std::mt19937 g_rng(5353);
float gauss() { static std::normal_distribution<float> n(0.0f, 1.0f); return n(g_rng); }
float unit() { static std::uniform_real_distribution<float> u(0.0f, 1.0f); return u(g_rng); }
vector<float> gvec(size_t n, float s, float mean = 0.0f) {
    vector<float> v(n);
    for (auto& x : v) x = mean + s * gauss();
    return v;
}
uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float bf16_val(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---------------------------------------------------------------- weights: the device form and the reference's
struct Lin {
    k::GlmLinear dev;
    vector<float> w;   // [n_out][n_in], the values the device form holds
    bool quant = false;
};
vector<void*> g_free;

/// Q8_0 / IQ4_XS rows of random codes with block scales giving a std of about s / sqrt(n_in); BF16 / F32 rounded.
Lin make_lin(int type, int64_t n_in, int64_t n_out, float s = 1.0f) {
    Lin L;
    L.dev.type = type;
    L.dev.n_in = n_in;
    L.dev.n_out = n_out;
    const float target = s / std::sqrt((float) n_in);
    const size_t n = (size_t) (n_in * n_out);
    if (type == 0 || type == 30) {
        L.w = gvec(n, target);
        if (type == 30) {
            vector<uint16_t> b(n);
            for (size_t i = 0; i < n; ++i) { b[i] = bf16_bits(L.w[i]); L.w[i] = bf16_val(b[i]); }
            L.dev.data = upload(b);
        } else {
            L.dev.data = upload(L.w);
        }
        g_free.push_back((void*) L.dev.data);
        return L;
    }
    L.quant = true;
    const size_t bv = type == 8 ? 32 : 256, bb = type == 8 ? 34 : 136;   // Q8_0, IQ4_XS
    vector<uint8_t> bytes(n / bv * bb);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : bytes) b = (uint8_t) byte(g_rng);
    const float per = type == 8 ? 74.0f : 1221.0f;   // std of the codes times the sub-scales
    for (size_t o = 0; o < bytes.size(); o += bb) {
        const uint16_t d = k::f16_from_f32(target / per * (0.75f + 0.5f * unit()) * (unit() < 0.5f ? -1.0f : 1.0f));
        std::memcpy(&bytes[o], &d, 2);
    }
    void* dp = upload(bytes);
    g_free.push_back(dp);
    L.dev.data = dp;
    float* dw = nullptr;
    check(cudaMalloc(&dw, n * 4), "dequant");
    k::iq_dequant_f32(type, dp, (int64_t) n, dw, nullptr);
    check(cudaDeviceSynchronize(), "dequant sync");
    L.w = download(dw, n);
    cudaFree(dw);
    return L;
}

/// Roundings the reference makes on a boundary, where the device's FP32 activation, ~1e-7 from the reference's, can
/// go the other way: a code (x / d within 3e-4 of .5 - 1e-4 was enough at decode; the prompt kernels' sums reach
/// further: glm_prompt_parity B, pos 25, a code 1e-4 .. 3e-4 from .5 rounded the other way), or a block's FP16 scale
/// (d within 1e-5 of the midpoint
/// between two FP16 values - then all 32 values of the block move by one FP16 step, 4.9e-4).  Found by
/// glm_layer_parity: at 2103 a block's amax was 0.79767389 in the reference and 0.79767448 on the device, either side
/// of one.  Each such rounding is an event, counted in g_near in the order the reference meets them; the events
/// whose indices are in g_flip are taken the other way (verify's search).
int g_near = 0;
vector<int> g_flip;
bool flip_next() {
    const int i = g_near++;
    return std::find(g_flip.begin(), g_flip.end(), i) != g_flip.end();
}
bool scale_on_boundary(float d) {
    const uint16_t h = k::f16_from_f32(d);
    const float v = k::f32_from_f16(h);
    if (d == v || h == 0) return false;
    const float nb = d > v ? k::f32_from_f16((uint16_t) (h + 1)) : k::f32_from_f16((uint16_t) (h - 1));
    const float mid = 0.5f * (v + nb);
    return std::fabs(d - mid) < 1e-5f * d;
}

/// The quantizations made since the events were last reset: the device quantizes an activation once in the same way
/// for every projection that reads it (wq and wk both read the attention's h), so where it rounds a value the other way
/// it does so for all of them.  A vector quantized again gives the same result and no new events.
struct Memo {
    int kind;   // 0 q8_1 (GEMV), 1 MMQ with an FP32 scale, 2 MMQ with an FP16 scale, 3 BF16
    vector<double> in, out;
};
vector<Memo> g_q8_memo;
const vector<double>* memo_find(int kind, const vector<double>& x) {
    for (const Memo& m : g_q8_memo)
        if (m.kind == kind && m.in == x) return &m.out;
    return nullptr;
}

/// 1: the calls being checked ran as a prompt (more than one token): the projections and the experts took the prompt
/// products (glm_gemm.hpp), which round their activations as llama.cpp's batched path does.
int g_prompt = 0;

/// The q8_1 rounding of an activation the quantized projections and the grouped experts read (the GEMV's: x / d, the
/// scale amax / 127 in FP16).
vector<double> q8_1(const vector<double>& x) {
    if (const vector<double>* r = memo_find(0, x)) return *r;
    vector<double> out(x.size());
    for (size_t b = 0; b < x.size(); b += 32) {
        float amax = 0.0f;
        for (size_t i = b; i < b + 32; ++i) amax = std::max(amax, std::fabs((float) x[i]));
        const float d = amax / 127.0f;
        const uint16_t h16 = k::f16_from_f32(d);
        double dh = k::f32_from_f16(h16);
        if (amax != 0.0f && scale_on_boundary(d) && flip_next())   // the other FP16 neighbour
            dh = k::f32_from_f16((uint16_t) (d > (float) dh ? h16 + 1 : h16 - 1));
        for (size_t i = b; i < b + 32; ++i) {
            float q = amax == 0.0f ? 0.0f : std::round((float) x[i] / d);
            if (amax != 0.0f) {
                const double fl = std::floor((double) x[i] / d), fr = (double) x[i] / d - fl;
                if (std::fabs(fr - 0.5) < 3e-4 && flip_next()) q = (float) (q == (float) fl ? fl + 1 : fl);
            }
            out[i] = q * dh;
        }
    }
    g_q8_memo.push_back({0, x, out});
    return out;
}

/// MMQ's q8_1 rounding (llama.cpp's quantize_mmq_q8_1): q = roundf(x * (127 / amax)) in FP32, the scale
/// 1 / (127 / amax) kept in FP32 (D4) or FP16 (DS4).
vector<double> q8_1_mmq(const vector<double>& x, bool fp16_scale) {
    const int kind = fp16_scale ? 2 : 1;
    if (const vector<double>* r = memo_find(kind, x)) return *r;
    vector<double> out(x.size());
    for (size_t b = 0; b < x.size(); b += 32) {
        float amax = 0.0f;
        for (size_t i = b; i < b + 32; ++i) amax = std::max(amax, std::fabs((float) x[i]));
        const float d_inv = 127.0f / amax, d = 1.0f / d_inv;
        double dh = d;
        if (fp16_scale) {
            const uint16_t h16 = k::f16_from_f32(d);
            dh = k::f32_from_f16(h16);
            if (amax != 0.0f && scale_on_boundary(d) && flip_next())
                dh = k::f32_from_f16((uint16_t) (d > (float) dh ? h16 + 1 : h16 - 1));
        }
        for (size_t i = b; i < b + 32; ++i) {
            const float p = amax == 0.0f ? 0.0f : (float) x[i] * d_inv;
            float q = std::round(p);
            if (amax != 0.0f) {
                const double fl = std::floor((double) p), fr = (double) p - fl;
                if (std::fabs(fr - 0.5) < 3e-4 && flip_next()) q = (float) (q == (float) fl ? fl + 1 : fl);
            }
            out[i] = amax == 0.0f ? 0.0 : (double) q * dh;
        }
    }
    g_q8_memo.push_back({kind, x, out});
    return out;
}

/// The BF16 rounding of the activations a BF16 GEMM reads (round to nearest even), a value within 3e-4 of a BF16 step
/// of the midpoint an event.
vector<double> bf16_act(const vector<double>& x) {
    if (const vector<double>* r = memo_find(3, x)) return *r;
    vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const float f = (float) x[i];
        uint32_t u;
        std::memcpy(&u, &f, 4);
        const uint32_t lo = u & 0xffffu;
        uint16_t b = bf16_bits(f);
        if (std::fabs((double) lo / 65536.0 - 0.5) < 3e-4 && flip_next())
            b = (uint16_t) ((u >> 16) + (b == (uint16_t) (u >> 16) ? 1 : 0));   // the other neighbour
        out[i] = bf16_val(b);
    }
    g_q8_memo.push_back({3, x, out});
    return out;
}

/// What a projection reads: at decode the GEMV's (q8_1 for a quantized weight, the FP32 values otherwise); in a prompt
/// the product glm_prompt_path chose (MMQ's q8_1, BF16 for a BF16 GEMM, or the GEMV's).
vector<double> act_for(const Lin& L, const vector<double>& x) {
    if (g_prompt) {
        switch (glm::glm_prompt_path(L.dev.type, L.dev.n_in, L.dev.n_out)) {
            case glm::GlmProjPath::Mmq: return q8_1_mmq(x, glm::glm_mmq_fp16_scale(L.dev.type));
            case glm::GlmProjPath::Bf16Gemm: return bf16_act(x);
            case glm::GlmProjPath::Gemv: break;
        }
    }
    return L.quant ? q8_1(x) : x;
}
vector<double> lin_ref(const Lin& L, const vector<double>& x) {
    const vector<double> xa = act_for(L, x);
    vector<double> y((size_t) L.dev.n_out);
    for (int64_t o = 0; o < L.dev.n_out; ++o) {
        double a = 0;
        const float* r = &L.w[(size_t) (o * L.dev.n_in)];
        for (int64_t i = 0; i < L.dev.n_in; ++i) a += (double) r[i] * xa[(size_t) i];
        y[(size_t) o] = a;
    }
    return y;
}
double sig(double v) { return 1.0 / (1.0 + std::exp(-v)); }
void rms_w(vector<double>& v, const vector<float>& w, double eps) {
    double ss = 0;
    for (double a : v) ss += a * a;
    const double inv = 1.0 / std::sqrt(ss / (double) v.size() + eps);
    for (size_t i = 0; i < v.size(); ++i) v[i] = v[i] * inv * w[i];
}
vector<double> swiglu_clamp(const vector<double>& g, const vector<double>& u, double lim) {
    vector<double> h(g.size());
    for (size_t i = 0; i < g.size(); ++i) {
        const double gg = std::min(g[i], lim), uu = std::clamp(u[i], -lim, lim);
        h[i] = gg * sig(gg) * uu;
    }
    return h;
}

struct Block {
    glm::GlmLayerWeights w;
    // the reference's copies
    vector<float> hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, attn_norm, ffn_norm;
    Lin wo, wq, wk, wv, f_a, f_b, beta, g_a, g_b, wq_a, wq_b, wkv_a, idx_q_b, idx_k, idx_gate, idx_proj, ffn_gate,
        ffn_up, ffn_down, gate_inp, sh_gate, sh_up, sh_down;
    vector<float> conv_q, conv_k, conv_v, ssm_a, dt_bias, o_norm, q_a_norm, kv_a_norm, k_b, v_b, idx_k_norm,
        idx_k_norm_b, idx_ape, exp_probs_b;
    vector<vector<float>> e_gate, e_up, e_down;   // per expert, dequantized
    // the reference's state
    vector<double> conv_hist, kda_state;          // kda_state in ggml's layout: [h][j][i]
    vector<uint16_t> latent;                      // [ctx][512] FP16
    // latent_sync's: the device's row for this token, which the attention then reads, and the reference's own row,
    // which latent_row_ok compares
    vector<uint16_t> dev_row, ref_row;
    vector<double> idx_cache, pooled;             // [ctx][256], [pools][128]
    glm::GlmLayerState st;
};

const float* up_f(const vector<float>& v) {
    float* p = upload(v);
    g_free.push_back(p);
    return p;
}

/// The mHC weights (hc = false: the MTP block, which has none) and the two norms.
void build_mhc(Block& b, int D, bool hc = true) {
    const int K = 4 * D;
    if (!hc) {
        b.attn_norm = gvec((size_t) D, 0.1f, 1.0f);
        b.ffn_norm = gvec((size_t) D, 0.1f, 1.0f);
        b.w.attn_norm = up_f(b.attn_norm);
        b.w.ffn_norm = up_f(b.ffn_norm);
        return;
    }
    b.hc_attn_fn = gvec((size_t) 24 * K, 1.0f / std::sqrt((float) K));
    b.hc_ffn_fn = gvec((size_t) 24 * K, 1.0f / std::sqrt((float) K));
    b.hc_attn_scale = {0.8f, 1.1f, 1.5f};
    b.hc_ffn_scale = {0.9f, 1.2f, 1.3f};
    b.hc_attn_base = gvec(24, 0.5f);
    b.hc_ffn_base = gvec(24, 0.5f);
    b.attn_norm = gvec((size_t) D, 0.1f, 1.0f);
    b.ffn_norm = gvec((size_t) D, 0.1f, 1.0f);
    b.w.hc_attn_fn = up_f(b.hc_attn_fn); b.w.hc_attn_scale = up_f(b.hc_attn_scale); b.w.hc_attn_base = up_f(b.hc_attn_base);
    b.w.hc_ffn_fn = up_f(b.hc_ffn_fn); b.w.hc_ffn_scale = up_f(b.hc_ffn_scale); b.w.hc_ffn_base = up_f(b.hc_ffn_base);
    b.w.attn_norm = up_f(b.attn_norm);
    b.w.ffn_norm = up_f(b.ffn_norm);
}

// ---------------------------------------------------------------- the reference layer
struct Ref {
    const Glm5NextGeometry& g;
    vector<int32_t> sel;   // the last full indexer's cells
    vector<int32_t> eids;  // the last router's experts
};

void mhc_pre_ref(const vector<double>& X, const vector<float>& fn, const vector<float>& sc, const vector<float>& base,
                 const Glm5NextGeometry& g, vector<double>& h, double post[4], double comb[4][4]) {
    const size_t D = (size_t) g.n_embd, K = 4 * D;
    double ss = 0;
    for (double v : X) ss += v * v;
    const double inv = 1.0 / std::sqrt(ss / (double) K + g.rms_eps);
    double m[24];
    for (int j = 0; j < 24; ++j) {
        double a = 0;
        for (size_t i = 0; i < K; ++i) a += (double) fn[(size_t) j * K + i] * X[i] * inv;
        m[j] = a;
    }
    double pre[4];
    for (int i = 0; i < 4; ++i) {
        pre[i] = sig(m[i] * sc[0] + base[(size_t) i]) + g.hc_eps;
        post[i] = 2.0 * sig(m[4 + i] * sc[1] + base[(size_t) (4 + i)]);
    }
    for (int s = 0; s < 4; ++s) {
        double mx = -INFINITY, sum = 0, c[4];
        for (int d = 0; d < 4; ++d) { c[d] = m[8 + 4 * s + d] * sc[2] + base[(size_t) (8 + 4 * s + d)]; mx = std::max(mx, c[d]); }
        for (int d = 0; d < 4; ++d) sum += std::exp(c[d] - mx);
        for (int d = 0; d < 4; ++d) comb[s][d] = std::exp(c[d] - mx) / sum + g.hc_eps;
    }
    auto cols = [&] { for (int d = 0; d < 4; ++d) { double sum = g.hc_eps; for (int s = 0; s < 4; ++s) sum += comb[s][d]; for (int s = 0; s < 4; ++s) comb[s][d] /= sum; } };
    auto rows = [&] { for (int s = 0; s < 4; ++s) { double sum = g.hc_eps; for (int d = 0; d < 4; ++d) sum += comb[s][d]; for (int d = 0; d < 4; ++d) comb[s][d] /= sum; } };
    cols();
    for (int64_t it = 1; it < g.hc_sinkhorn_iters; ++it) { rows(); cols(); }
    h.assign(D, 0.0);
    for (size_t d = 0; d < D; ++d)
        for (int i = 0; i < 4; ++i) h[d] += pre[i] * X[(size_t) i * D + d];
}
void mhc_post_ref(const vector<double>& f, const double post[4], const double comb[4][4], vector<double>& X) {
    const size_t D = f.size();
    vector<double> out(X.size());
    for (int dst = 0; dst < 4; ++dst)
        for (size_t d = 0; d < D; ++d) {
            double a = f[d] * post[dst];
            for (int s = 0; s < 4; ++s) a += X[(size_t) s * D + d] * comb[s][dst];
            out[(size_t) dst * D + d] = a;
        }
    X = out;
}

vector<double> kda_ref(Block& b, const Glm5NextGeometry& g, const vector<double>& h) {
    const int H = (int) g.n_head, Dh = (int) g.kda_head_dim, di = H * Dh;
    vector<double> x(3 * (size_t) di);
    {
        const vector<double> q = lin_ref(b.wq, h), kk = lin_ref(b.wk, h), v = lin_ref(b.wv, h);
        std::copy(q.begin(), q.end(), x.begin());
        std::copy(kk.begin(), kk.end(), x.begin() + di);
        std::copy(v.begin(), v.end(), x.begin() + 2 * di);
    }
    vector<double> qkv(x.size());
    for (int gc = 0; gc < 3 * di; ++gc) {
        const vector<float>& w = gc < di ? b.conv_q : gc < 2 * di ? b.conv_k : b.conv_v;
        const int c = gc % di;
        double* hs = &b.conv_hist[(size_t) gc * 3];
        double a = hs[0] * w[(size_t) c * 4] + hs[1] * w[(size_t) c * 4 + 1] + hs[2] * w[(size_t) c * 4 + 2] +
                   x[(size_t) gc] * w[(size_t) c * 4 + 3];
        qkv[(size_t) gc] = a * sig(a);
        hs[0] = hs[1]; hs[1] = hs[2]; hs[2] = x[(size_t) gc];
    }
    for (int hh = 0; hh < 2 * H; ++hh) {
        double ss = 0;
        for (int i = 0; i < Dh; ++i) ss += qkv[(size_t) (hh * Dh + i)] * qkv[(size_t) (hh * Dh + i)];
        const double inv = 1.0 / std::sqrt(ss + 1e-6);
        for (int i = 0; i < Dh; ++i) qkv[(size_t) (hh * Dh + i)] *= inv;
    }
    const vector<double> raw = lin_ref(b.f_b, lin_ref(b.f_a, h));
    vector<double> gate((size_t) di);
    for (int c = 0; c < di; ++c)
        gate[(size_t) c] = g.kda_gate_lower_bound * sig(-(double) b.ssm_a[(size_t) (c / Dh)] * (raw[(size_t) c] + b.dt_bias[(size_t) c]));
    vector<double> beta = lin_ref(b.beta, h);
    for (auto& v : beta) v = sig(v);
    const vector<double> z = lin_ref(b.g_b, lin_ref(b.g_a, h));
    vector<double> y((size_t) di);
    for (int hh = 0; hh < H; ++hh) {
        double* M = &b.kda_state[(size_t) hh * Dh * Dh];
        const double* q = &qkv[(size_t) hh * Dh];
        const double* kk = q + di;
        const double* v = q + 2 * di;
        for (int j = 0; j < Dh; ++j)
            for (int i = 0; i < Dh; ++i) M[(size_t) j * Dh + i] *= std::exp(gate[(size_t) (hh * Dh + i)]);
        vector<double> delta((size_t) Dh);
        for (int j = 0; j < Dh; ++j) {
            double s = 0;
            for (int i = 0; i < Dh; ++i) s += M[(size_t) j * Dh + i] * kk[i];
            delta[(size_t) j] = (v[j] - s) * beta[(size_t) hh];
        }
        for (int j = 0; j < Dh; ++j)
            for (int i = 0; i < Dh; ++i) M[(size_t) j * Dh + i] += kk[i] * delta[(size_t) j];
        double ss = 0;
        vector<double> o((size_t) Dh);
        for (int j = 0; j < Dh; ++j) {
            double s = 0;
            for (int i = 0; i < Dh; ++i) s += M[(size_t) j * Dh + i] * q[i];
            o[(size_t) j] = s / std::sqrt((double) Dh);
            ss += o[(size_t) j] * o[(size_t) j];
        }
        const double inv = 1.0 / std::sqrt(ss / Dh + g.rms_eps);
        for (int j = 0; j < Dh; ++j)
            y[(size_t) (hh * Dh + j)] = o[(size_t) j] * inv * b.o_norm[(size_t) j] * sig(z[(size_t) (hh * Dh + j)]);
    }
    return lin_ref(b.wo, y);
}

void pool_ref(Block& b, int64_t p) {
    for (int d = 0; d < 128; ++d) {
        double lg[4], mx = -INFINITY, s = 0;
        for (int m = 0; m < 4; ++m) {
            lg[m] = b.idx_cache[(size_t) (p * 4 + m) * 256 + 128 + d] + b.idx_ape[(size_t) (m * 128 + d)];
            mx = std::max(mx, lg[m]);
        }
        for (int m = 0; m < 4; ++m) s += std::exp(lg[m] - mx);
        double a = 0;
        for (int m = 0; m < 4; ++m) a += std::exp(lg[m] - mx) / s * b.idx_cache[(size_t) (p * 4 + m) * 256 + d];
        b.pooled[(size_t) p * 128 + d] = a;
    }
}

vector<double> mla_ref(Block& b, const Glm5NextGeometry& g, const vector<double>& h, int64_t pos, bool full, Ref& r) {
    const int H = (int) g.n_head, L = 512, DK = (int) g.head_dim_k, DV = (int) g.head_dim_v;
    vector<double> qr = lin_ref(b.wq_a, h);
    rms_w(qr, b.q_a_norm, g.rms_eps);
    const vector<double> q = lin_ref(b.wq_b, qr);
    vector<double> kv = lin_ref(b.wkv_a, h);
    rms_w(kv, b.kv_a_norm, g.rms_eps);
    for (int i = 0; i < L; ++i) b.latent[(size_t) (pos * L + i)] = k::f16_from_f32((float) kv[(size_t) i]);
    if (!b.dev_row.empty()) {   // the attention reads the device's row; the reference's is checked on its own
        b.ref_row.assign(b.latent.begin() + pos * L, b.latent.begin() + (pos + 1) * L);
        std::copy(b.dev_row.begin(), b.dev_row.end(), b.latent.begin() + pos * L);
    }
    vector<double> q_abs((size_t) H * L);
    for (int hh = 0; hh < H; ++hh)
        for (int c = 0; c < L; ++c) {
            double a = 0;
            for (int d = 0; d < DK; ++d) a += (double) b.k_b[((size_t) hh * L + c) * DK + d] * q[(size_t) (hh * DK + d)];
            q_abs[(size_t) (hh * L + c)] = a;
        }
    const int64_t n_kv = pos + 1, n_bid = n_kv / 4;
    if (full) {
        const vector<double> iq = lin_ref(b.idx_q_b, qr), ik = lin_ref(b.idx_k, h), ig = lin_ref(b.idx_gate, h);
        double mean = 0, var = 0;
        for (double v : ik) mean += v;
        mean /= 128;
        for (double v : ik) var += (v - mean) * (v - mean);
        var /= 128;
        for (int d = 0; d < 128; ++d) {
            b.idx_cache[(size_t) (pos * 256 + d)] = (ik[(size_t) d] - mean) / std::sqrt(var + g.norm_eps) * b.idx_k_norm[(size_t) d] + b.idx_k_norm_b[(size_t) d];
            b.idx_cache[(size_t) (pos * 256 + 128 + d)] = ig[(size_t) d];
        }
        if (n_kv % 4 == 0) pool_ref(b, n_bid - 1);
        const vector<double> iw = lin_ref(b.idx_proj, h);
        vector<double> sc((size_t) n_bid);
        for (int64_t p = 0; p < n_bid; ++p) {
            double s = 0;
            for (int hh = 0; hh < 32; ++hh) {
                double dot = 0;
                for (int d = 0; d < 128; ++d) dot += iq[(size_t) (hh * 128 + d)] * b.pooled[(size_t) (p * 128 + d)];
                s += iw[(size_t) hh] / std::sqrt(128.0 * 32.0) * std::max(dot, 0.0);
            }
            sc[(size_t) p] = s;
        }
        vector<int64_t> order((size_t) n_bid);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t c) { return sc[(size_t) a] > sc[(size_t) c]; });
        r.sel.clear();
        for (int64_t i = 0; i < std::min<int64_t>(n_bid, g.idx_top_k / 4); ++i)
            for (int m = 0; m < 4; ++m) r.sel.push_back((int32_t) (order[(size_t) i] * 4 + m));
        for (int64_t c = n_bid * 4; c < n_kv; ++c) r.sel.push_back((int32_t) c);
        std::sort(r.sel.begin(), r.sel.end());
    }
    vector<double> o_lat((size_t) H * L, 0.0);
    for (int hh = 0; hh < H; ++hh) {
        vector<double> s(r.sel.size());
        double mx = -INFINITY;
        for (size_t j = 0; j < r.sel.size(); ++j) {
            double a = 0;
            for (int c = 0; c < L; ++c) a += q_abs[(size_t) (hh * L + c)] * k::f32_from_f16(b.latent[(size_t) r.sel[j] * L + c]);
            s[j] = a / std::sqrt((double) DK);
            mx = std::max(mx, s[j]);
        }
        double l = 0;
        for (size_t j = 0; j < r.sel.size(); ++j) {
            const double e = std::exp(s[j] - mx);
            l += e;
            for (int c = 0; c < L; ++c) o_lat[(size_t) (hh * L + c)] += e * k::f32_from_f16(b.latent[(size_t) r.sel[j] * L + c]);
        }
        for (int c = 0; c < L; ++c) o_lat[(size_t) (hh * L + c)] /= l;
    }
    vector<double> out((size_t) H * DV);
    for (int hh = 0; hh < H; ++hh)
        for (int e = 0; e < DV; ++e) {
            double a = 0;
            for (int c = 0; c < L; ++c) a += (double) b.v_b[((size_t) hh * DV + e) * L + c] * o_lat[(size_t) (hh * L + c)];
            out[(size_t) (hh * DV + e)] = a;
        }
    return lin_ref(b.wo, out);
}

vector<double> ffn_ref(Block& b, const Glm5NextGeometry& g, const vector<double>& h, int64_t il, Ref& r) {
    if (il < g.n_dense_lead)
        return lin_ref(b.ffn_down, swiglu_clamp(lin_ref(b.ffn_gate, h), lin_ref(b.ffn_up, h), g.swiglu_clamp_shexp));
    const vector<double> lg = lin_ref(b.gate_inp, h);
    const int NE = (int) g.n_expert, KU = (int) g.n_expert_used;
    vector<double> p((size_t) NE), sel((size_t) NE);
    for (int e = 0; e < NE; ++e) { p[(size_t) e] = sig(lg[(size_t) e]); sel[(size_t) e] = p[(size_t) e] + b.exp_probs_b[(size_t) e]; }
    vector<int> order((size_t) NE);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int c) { return sel[(size_t) a] > sel[(size_t) c]; });
    double sum = 0;
    r.eids.clear();
    for (int i = 0; i < KU; ++i) { r.eids.push_back(order[(size_t) i]); sum += p[(size_t) order[(size_t) i]]; }
    // the experts' activations: a prompt's MMQ products round as MMQ does, else the grouped kernel's q8_1
    const k::NativeExpertLayout& el = b.w.expert_layout;
    const bool mmq = g_prompt && glm::glm_prompt_experts_mmq(el, b.w.expert_stride);
    auto round_in = [&](const vector<double>& v, int type) {
        return mmq ? q8_1_mmq(v, glm::glm_mmq_fp16_scale(type)) : q8_1(v);
    };
    const vector<double> xq = round_in(h, el.gu_type);
    vector<double> out((size_t) g.n_embd, 0.0);
    for (int i = 0; i < KU; ++i) {
        const int e = order[(size_t) i];
        const double wgt = p[(size_t) e] / std::max(sum, 6.103515625e-5) * g.expert_weights_scale;
        const int64_t D = g.n_embd, F = g.n_ff_exp;
        vector<double> gg((size_t) F), uu((size_t) F);
        for (int64_t f = 0; f < F; ++f) {
            double a = 0, c = 0;
            for (int64_t d = 0; d < D; ++d) {
                a += (double) b.e_gate[(size_t) e][(size_t) (f * D + d)] * xq[(size_t) d];
                c += (double) b.e_up[(size_t) e][(size_t) (f * D + d)] * xq[(size_t) d];
            }
            gg[(size_t) f] = a;
            uu[(size_t) f] = c;
        }
        const vector<double> s = round_in(swiglu_clamp(gg, uu, g.swiglu_clamp_exp), el.d_type);
        for (int64_t d = 0; d < D; ++d) {
            double a = 0;
            for (int64_t f = 0; f < F; ++f) a += (double) b.e_down[(size_t) e][(size_t) (d * F + f)] * s[(size_t) f];
            out[(size_t) d] += wgt * a;
        }
    }
    const vector<double> sh = lin_ref(b.sh_down, swiglu_clamp(lin_ref(b.sh_gate, h), lin_ref(b.sh_up, h), g.swiglu_clamp_shexp));
    for (size_t d = 0; d < out.size(); ++d) out[d] += sh[d];
    return out;
}

// ---------------------------------------------------------------- building blocks (each draws from g_rng in a fixed
// order, so a test's fixture is fixed by the order it builds its blocks)
void build_kda_attn(Block& b, const Glm5NextGeometry& g) {
    const int D = (int) g.n_embd, H = (int) g.n_head, Dh = 128, di = H * Dh;
    b.wq = make_lin(23, D, di); b.wk = make_lin(8, D, di); b.wv = make_lin(30, D, di);
    b.f_a = make_lin(0, D, Dh); b.f_b = make_lin(0, Dh, di); b.beta = make_lin(30, D, H);
    b.g_a = make_lin(0, D, Dh); b.g_b = make_lin(30, Dh, di); b.wo = make_lin(8, di, D);
    b.conv_q = gvec((size_t) di * 4, 0.5f); b.conv_k = gvec((size_t) di * 4, 0.5f); b.conv_v = gvec((size_t) di * 4, 0.5f);
    b.ssm_a.resize((size_t) H);
    for (auto& a : b.ssm_a) a = -std::exp(unit() * std::log(16.0f));
    b.dt_bias = gvec((size_t) di, 0.5f);
    b.o_norm = gvec((size_t) Dh, 0.1f, 1.0f);
    auto& w = b.w;
    w.wq = b.wq.dev; w.wk = b.wk.dev; w.wv = b.wv.dev; w.f_a = b.f_a.dev; w.f_b = b.f_b.dev; w.beta = b.beta.dev;
    w.g_a = b.g_a.dev; w.g_b = b.g_b.dev; w.wo = b.wo.dev;
    w.conv_q = up_f(b.conv_q); w.conv_k = up_f(b.conv_k); w.conv_v = up_f(b.conv_v);
    w.ssm_a = up_f(b.ssm_a); w.dt_bias = up_f(b.dt_bias); w.o_norm = up_f(b.o_norm);
}

void build_dense_ffn(Block& b, const Glm5NextGeometry& g) {
    const int D = (int) g.n_embd;
    b.ffn_gate = make_lin(8, D, g.n_ff_dense); b.ffn_up = make_lin(23, D, g.n_ff_dense);
    b.ffn_down = make_lin(23, g.n_ff_dense, D);
    b.w.ffn_gate = b.ffn_gate.dev; b.w.ffn_up = b.ffn_up.dev; b.w.ffn_down = b.ffn_down.dev;
}

void build_mla_attn(Block& b, const Glm5NextGeometry& g, bool full) {
    const int D = (int) g.n_embd, H = (int) g.n_head, L = 512;
    b.wq_a = make_lin(30, D, g.q_lora_rank); b.wq_b = make_lin(8, g.q_lora_rank, H * 256);
    b.wkv_a = make_lin(23, D, L); b.wo = make_lin(8, H * 256, D);
    b.q_a_norm = gvec((size_t) g.q_lora_rank, 0.1f, 1.0f);
    b.kv_a_norm = gvec((size_t) L, 0.1f, 1.0f);
    vector<uint16_t> kb((size_t) H * L * 256), vb((size_t) H * 256 * L);
    b.k_b.resize(kb.size());
    b.v_b.resize(vb.size());
    for (size_t i = 0; i < kb.size(); ++i) { kb[i] = bf16_bits(gauss() / 16.0f); b.k_b[i] = bf16_val(kb[i]); }
    for (size_t i = 0; i < vb.size(); ++i) { vb[i] = bf16_bits(gauss() / std::sqrt((float) L)); b.v_b[i] = bf16_val(vb[i]); }
    auto& w = b.w;
    w.wq_a = b.wq_a.dev; w.wq_b = b.wq_b.dev; w.wkv_a = b.wkv_a.dev; w.wo = b.wo.dev;
    w.q_a_norm = up_f(b.q_a_norm); w.kv_a_norm = up_f(b.kv_a_norm);
    w.k_b = upload(kb); w.v_b = upload(vb);
    g_free.push_back((void*) w.k_b); g_free.push_back((void*) w.v_b);
    if (full) {
        b.idx_q_b = make_lin(0, g.q_lora_rank, 32 * 128); b.idx_k = make_lin(30, D, 128);
        b.idx_gate = make_lin(0, D, 128); b.idx_proj = make_lin(0, D, 32);
        b.idx_k_norm = gvec(128, 0.1f, 1.0f); b.idx_k_norm_b = gvec(128, 0.1f); b.idx_ape = gvec(4 * 128, 1.0f);
        w.idx_q_b = b.idx_q_b.dev; w.idx_k = b.idx_k.dev; w.idx_gate = b.idx_gate.dev; w.idx_proj = b.idx_proj.dev;
        w.idx_k_norm = up_f(b.idx_k_norm); w.idx_k_norm_b = up_f(b.idx_k_norm_b); w.idx_ape = up_f(b.idx_ape);
    }
}

/// The MoE: router F32, n_expert IQ4_XS experts (blobs [gate | up | down]), the shared expert.
void build_moe(Block& b, const Glm5NextGeometry& g) {
    const int D = (int) g.n_embd;
    auto& w = b.w;
    b.gate_inp = make_lin(0, D, g.n_expert, 2.0f);
    b.exp_probs_b = gvec((size_t) g.n_expert, 0.1f);
    w.gate_inp = b.gate_inp.dev;
    w.exp_probs_b = up_f(b.exp_probs_b);
    const int64_t F = g.n_ff_exp;
    w.expert_layout = k::native_expert_layout(23, 23, D, F);
    const size_t slot = (w.expert_layout.bytes + 255) & ~(size_t) 255;
    uint8_t* blobs = nullptr;
    check(cudaMalloc(&blobs, slot * (size_t) g.n_expert), "blobs");
    g_free.push_back(blobs);
    vector<unsigned long long> ptrs;
    for (int e = 0; e < g.n_expert; ++e) {
        const Lin eg = make_lin(23, D, F), eu = make_lin(23, D, F), ed = make_lin(23, F, D);
        uint8_t* dst = blobs + slot * (size_t) e;
        const size_t gb = (size_t) w.expert_layout.up_off, db = w.expert_layout.bytes - w.expert_layout.down_off;
        check(cudaMemcpy(dst, eg.dev.data, gb, cudaMemcpyDeviceToDevice), "blob gate");
        check(cudaMemcpy(dst + w.expert_layout.up_off, eu.dev.data, gb, cudaMemcpyDeviceToDevice), "blob up");
        check(cudaMemcpy(dst + w.expert_layout.down_off, ed.dev.data, db, cudaMemcpyDeviceToDevice), "blob down");
        b.e_gate.push_back(eg.w); b.e_up.push_back(eu.w); b.e_down.push_back(ed.w);
        ptrs.push_back((unsigned long long) (uintptr_t) dst);
    }
    w.expert_ptrs = upload(ptrs);
    g_free.push_back((void*) w.expert_ptrs);
    w.expert_base = blobs;
    w.expert_stride = slot;
    b.sh_gate = make_lin(23, D, g.n_ff_shexp); b.sh_up = make_lin(23, D, g.n_ff_shexp);
    b.sh_down = make_lin(8, g.n_ff_shexp, D);
    w.sh_gate = b.sh_gate.dev; w.sh_up = b.sh_up.dev; w.sh_down = b.sh_down.dev;
}

/// The reference's state of block il for a new sequence: all zero, as GlmModel::reset leaves the device's.
[[maybe_unused]] void ref_state_zero(Block& b, const Glm5NextGeometry& g, int64_t il, int64_t max_ctx) {
    const size_t di = (size_t) (g.n_head * 128);
    if (!g.is_mla[(size_t) il]) {
        b.conv_hist.assign(3 * di * 3, 0.0);
        b.kda_state.assign(di * 128, 0.0);
        return;
    }
    b.latent.assign((size_t) max_ctx * 512, 0);
    b.idx_cache.assign((size_t) max_ctx * 256, 0.0);
    b.pooled.assign((size_t) (max_ctx / 4 + 1) * 128, 0.0);
}

// ---------------------------------------------------------------- whole blocks
bool full_idx(const Glm5NextGeometry& g, int64_t il) {
    return il >= g.n_layers || g.idx_full.empty() || g.idx_full[(size_t) il] != 0;
}

/// A main block on the streams X [4][n_embd], in place: the body of llama.cpp's graph loop.
void block_ref(Block& b, const Glm5NextGeometry& g, int64_t il, int64_t pos, vector<double>& X, Ref& r) {
    vector<double> h;
    double post[4], comb[4][4];
    mhc_pre_ref(X, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base, g, h, post, comb);
    rms_w(h, b.attn_norm, g.rms_eps);
    const vector<double> f = g.is_mla[(size_t) il] ? mla_ref(b, g, h, pos, full_idx(g, il), r) : kda_ref(b, g, h);
    mhc_post_ref(f, post, comb, X);
    mhc_pre_ref(X, b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base, g, h, post, comb);
    rms_w(h, b.ffn_norm, g.rms_eps);
    const vector<double> ff = ffn_ref(b, g, h, il, r);
    mhc_post_ref(ff, post, comb, X);
}

/// The MTP block on x [n_embd], in place: graph_mtp between eh_proj and the head norm (a plain residual).
[[maybe_unused]] void mtp_block_ref(Block& b, const Glm5NextGeometry& g, int64_t il, int64_t pos, vector<double>& x, Ref& r) {
    vector<double> h = x;
    rms_w(h, b.attn_norm, g.rms_eps);
    const vector<double> f = mla_ref(b, g, h, pos, true, r);
    for (size_t d = 0; d < x.size(); ++d) x[d] += f[d];
    h = x;
    rms_w(h, b.ffn_norm, g.rms_eps);
    const vector<double> ff = ffn_ref(b, g, h, il, r);
    for (size_t d = 0; d < x.size(); ++d) x[d] += ff[d];
}

/// An MLA block's attention in the reference reads the device's latent cache, rows [0, pos] (call before block_ref,
/// after the device wrote row pos), as the block reads the device's streams: the reference's own FP16 roundings of a
/// row differ from the device's in about one value in 5000 (the FP32 values are 1e-7 apart), and over a selection of
/// a few cells one such value moves the softmax by 1e-5, which the q8_1 activations after it (wo, then the FFN)
/// magnify to 1e-2 - found by glm_model_parity, whose MLA blocks were 5e-3 off from position 3 on before this.  The
/// reference still computes the row; latent_row_ok compares it with the device's.
void latent_sync(Block& b, int64_t pos) {
    const vector<uint16_t> dev = download(b.st.latent, (size_t) (pos + 1) * 512);
    std::copy(dev.begin(), dev.begin() + pos * 512, b.latent.begin());
    b.dev_row.assign(dev.begin() + pos * 512, dev.end());
}

/// A KDA block's carried state on the device: the conv history, and the recurrent state in the reference's order.
struct KdaDev {
    vector<float> hist, state;
};
KdaDev kda_dev(const Block& b, const Glm5NextGeometry& g) {
    const int H = (int) g.n_head, Dh = 128;
    KdaDev d;
    d.hist = download(b.st.conv_hist, b.conv_hist.size());
    const vector<float> dst = download(b.st.kda_state, b.kda_state.size());
    d.state.resize(dst.size());
    for (int hh = 0; hh < H; ++hh)   // the reference's [h][j][i]; the device's [h][i][j]
        for (int j = 0; j < Dh; ++j)
            for (int i = 0; i < Dh; ++i)
                d.state[((size_t) hh * Dh + j) * Dh + i] = dst[((size_t) hh * Dh + i) * Dh + j];
    return d;
}
/// The conv history within 1e-6 (it holds the raw projections) and the state within 1e-5.
bool kda_close(const Block& b, const KdaDev& d, double& rh, double& rs) {
    rh = rel_l1(b.conv_hist, d.hist);
    rs = rel_l1(b.kda_state, d.state);
    return rh <= 1e-6 && rs <= 1e-5;
}
bool kda_state_ok(const Block& b, const KdaDev& d, int64_t pos, int64_t il) {
    double rh, rs;
    const bool ok = kda_close(b, d, rh, rs);
    std::printf("  pos %lld block %lld: KDA conv history rel %.3e, state rel %.3e %s\n", (long long) pos,
                (long long) il, rh, rs, ok ? "" : "*** over ***");
    return ok;
}

/// The latent row the token wrote, the device's against the reference's: the values further apart than one FP16 step
/// or 1e-6 of the row's largest value (the projection's FP32 error, which is absolute: on a value near zero it is
/// several steps); `steps` counts those apart by less.
int latent_row_far(const Block& b, int* steps = nullptr) {
    double top = 0;
    for (uint16_t v : b.ref_row) top = std::max(top, std::fabs((double) k::f32_from_f16(v)));
    int near = 0, far = 0;
    for (size_t c = 0; c < b.dev_row.size(); ++c) {
        const double a = k::f32_from_f16(b.dev_row[c]), want = k::f32_from_f16(b.ref_row[c]);
        // one FP16 step at the larger magnitude (2^-24 below the normals), by value: across zero the neighbours' bit
        // patterns are not adjacent
        const double big = std::max(std::fabs(a), std::fabs(want));
        const double step = big < 6.103515625e-05 ? 5.960464477539063e-08 : std::ldexp(1.0, std::ilogb(big) - 10);
        if (a != want) ++(std::fabs(a - want) <= std::max(step, 1e-6 * top) ? near : far);
    }
    if (steps) *steps = near;
    return far;
}
bool latent_row_ok(Block& b, int64_t pos, long long il) {
    int steps = 0;
    const int far = latent_row_far(b, &steps);
    if (steps || far)
        std::printf("  pos %lld block %lld: the latent row: %d values a rounding apart%s\n", (long long) pos, il, steps,
                    far ? ", *** others further ***" : "");
    b.dev_row.clear();
    return far == 0;
}

/// The reference's state of a block and the selection and experts it leaves, for verify's reruns.
struct Snap {
    vector<double> conv_hist, kda_state, idx_cache, pooled;
    vector<uint16_t> latent;
    vector<int32_t> sel, eids;
};
Snap save(const Block& b, const Ref& r) { return {b.conv_hist, b.kda_state, b.idx_cache, b.pooled, b.latent, r.sel, r.eids}; }
void restore(Block& b, Ref& r, const Snap& s) {
    b.conv_hist = s.conv_hist; b.kda_state = s.kda_state; b.idx_cache = s.idx_cache; b.pooled = s.pooled;
    b.latent = s.latent; r.sel = s.sel; r.eids = s.eids;
}

/// What a reference run gives: the result's relative difference from the device's, and whether everything else the
/// run produced matches the device's (the state the block carries: the latent row, the KDA history and state).
struct Fit {
    double rel;
    bool rest = true;
};

/// A device result against the reference: `run` computes the reference from the state `undo` restores.  It passes
/// when the result is within 1e-4 and the rest matches.  Where it does not and the reference met q8_1 roundings on a
/// boundary (g_near), the device's FP32 activation, ~1e-7 from the reference's, may have gone the other way at one of
/// them - and every activation after it moves too: one FP16 step of a scale moved a block 8.5e-4, a rounding before a
/// projection that a few cells' attention reads moved the FFN behind it 1e-2.  So `run` is repeated with every set of
/// one, two or three of those roundings taken the other way, and it passes only if one set explains the result and
/// the state together: an explanation checked, not assumed (a wrong result does not land within 1e-4 of the
/// reference with a specific other rounding).  The state left is the accepted run's, else the plain reference's.
bool g_quiet = false;   ///< verify prints only the results that needed a search (or failed)
template <class Run, class Undo>
bool verify(const char* what, int64_t pos, long long il, Run run, Undo undo) {
    auto good = [](const Fit& f) { return f.rel <= 1e-4 && f.rest; };
    // FP32 summation against double: within this the reference took the device's roundings.  A result between it and
    // 1e-4 may still be a rounding taken the other way whose effect is small here but carried in the state - a KDA
    // block's conv history, which a prompt shows only at its end - so the search runs for it too, and keeps the plain
    // reference when nothing comes closer.
    auto exact = [](const Fit& f) { return f.rel <= 1e-6 && f.rest; };
    auto run_with = [&](const vector<int>& s) {
        undo();
        g_flip = s;
        g_near = 0;
        g_q8_memo.clear();
        return run();
    };
    g_flip.clear();
    g_near = 0;
    g_q8_memo.clear();
    const Fit f0 = run();
    const int n = g_near;
    Fit f = f0;
    vector<int> taken;
    if (!exact(f0) && n > 0) {
        vector<vector<int>> sets;
        for (int a = 0; a < n; ++a) sets.push_back({a});
        for (int a = 0; a < n; ++a)
            for (int c = a + 1; c < n; ++c) sets.push_back({a, c});
        if (n <= 24)
            for (int a = 0; a < n; ++a)
                for (int c = a + 1; c < n; ++c)
                    for (int e = c + 1; e < n; ++e) sets.push_back({a, c, e});
        bool found = false, have_good = false;
        vector<int> good_set;
        for (const vector<int>& s : sets) {
            const Fit fs = run_with(s);
            if (exact(fs)) { f = fs; taken = s; found = true; break; }
            if (!have_good && good(fs)) { have_good = true; good_set = s; }
        }
        if (!found) {
            if (!good(f0) && have_good) {   // the closest acceptable explanation
                taken = good_set;
                f = run_with(taken);
            } else {                          // the plain reference (its state)
                f = run_with({});
            }
        }
    }
    g_flip.clear();
    if (g_quiet && exact(f0)) return true;
    if (g_quiet && good(f0) && taken.empty()) return true;
    std::printf("  pos %lld block %lld: %s rel %.3e%s", (long long) pos, il, what, f.rel, f.rest ? "" : " (and its state)");
    if (!taken.empty())
        std::printf("  (%.1e%s as rounded; the reference matches with %zu of its %d q8_1 roundings on a boundary taken "
                    "the other way)", f0.rel, f0.rest ? "" : " and its state", taken.size(), n);
    std::printf("%s\n", good(f) ? "" : "  *** over 1e-4, or its state differs ***");
    return good(f);
}

/// glm_model_parity's model, which glm_loader_parity writes to a GGUF: five layers (0 KDA + dense FFN, 1 MLA with its
/// own indexer, 2 KDA, 3 MLA sharing 1's selection, 4 KDA; MoE from 1) and an MTP block, at n_embd 1024 with a
/// vocabulary of 2048; an IQ4_XS embedding, a Q8_0 head, a BF16 eh_proj, the MTP block's own head norm.  Built in one
/// fixed order from g_rng, so both tests get the same numbers.
struct TestModel {
    Glm5NextGeometry g;
    vector<Block> blocks;
    Lin tok, out, eh;
    vector<float> out_norm, enorm, hnorm, head_norm;
    glm::GlmModelWeights w;
};
[[maybe_unused]] TestModel build_test_model(int64_t max_ctx) {
    TestModel tm;
    Glm5NextGeometry& g = tm.g;
    vector<Block>& blocks = tm.blocks;
    g.n_layers = 5; g.n_nextn = 1; g.n_embd = 1024; g.n_vocab = 2048; g.n_ctx_train = max_ctx;
    g.rms_eps = 1e-5; g.norm_eps = 1e-6;
    g.is_mla = {0, 1, 0, 1, 0, 1};
    g.n_head = 8;
    g.kda_head_dim = 128; g.ssm_d_conv = 4; g.has_kda_gate_lower_bound = true; g.kda_gate_lower_bound = -5;
    g.q_lora_rank = 512; g.kv_lora_rank = 512; g.n_rot = 0; g.head_dim_k = 256; g.head_dim_v = 256;
    g.idx_heads = 32; g.idx_key_dim = 128; g.idx_top_k = 2048; g.idx_kpool = 4; g.idx_kpool_select_tail = true;
    g.idx_full = {1, 1, 1, 0, 1};
    g.hc = 4; g.hc_sinkhorn_iters = 20; g.hc_eps = 1e-6;
    g.n_dense_lead = 1; g.n_ff_dense = 512; g.n_expert = 16; g.n_expert_used = 4; g.n_expert_shared = 1;
    g.n_ff_exp = 256; g.n_ff_shexp = 256; g.expert_gating = 2; g.expert_weights_scale = 2.5;
    g.expert_weights_norm = true; g.swiglu_clamp_exp = 10; g.swiglu_clamp_shexp = 10;
    const int D = (int) g.n_embd, V = (int) g.n_vocab;
    const int64_t NL = g.n_layers;

    blocks.resize((size_t) g.n_blocks());
    for (int64_t il = 0; il < g.n_blocks(); ++il) {
        Block& b = blocks[(size_t) il];
        build_mhc(b, D, il < NL);
        if (g.is_mla[(size_t) il]) build_mla_attn(b, g, full_idx(g, il));
        else build_kda_attn(b, g);
        if (il < g.n_dense_lead) build_dense_ffn(b, g);
        else build_moe(b, g);
        ref_state_zero(b, g, il, max_ctx);
    }
    // the model's own weights
    tm.tok = make_lin(23, D, V, std::sqrt((float) D));   // V rows of D: an embedding table, std ~1
    tm.out = make_lin(8, D, V);
    tm.eh = make_lin(30, 2 * D, D);
    tm.out_norm = gvec((size_t) D, 0.1f, 1.0f);
    tm.enorm = gvec((size_t) D, 0.1f, 1.0f);
    tm.hnorm = gvec((size_t) D, 0.1f, 1.0f);
    tm.head_norm = gvec((size_t) D, 0.1f, 1.0f);
    const Lin &tok = tm.tok, &out = tm.out, &eh = tm.eh;
    glm::GlmModelWeights& w = tm.w;
    w.tok_embd = {23, tok.dev.data, k::iq_row_bytes(23, D)};
    w.output_norm = up_f(tm.out_norm);
    w.output = out.dev;
    for (const Block& b : blocks) w.blocks.push_back(b.w);
    w.eh_proj = eh.dev;
    w.enorm = up_f(tm.enorm);
    w.hnorm = up_f(tm.hnorm);
    w.mtp_head_norm = up_f(tm.head_norm);
    return tm;
}

}  // namespace

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
