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
// The reference is exact arithmetic on the same numbers: the quantized weights read back through iq_dequant_f32,
// and before a quantized projection the activation rounded to q8_1 as the kernels round it (a scale amax / 127 kept
// in FP16, the values rounded to integers), so what remains is FP32 summation against double.
//
// Each block's reference starts from the streams the device gave it, and the state a block carries from token to
// token (the KDA conv history and recurrent state; the caches through the indexer's choices) is compared on its own.
// q8_1 roundings are steps: where the reference rounds a code or a block's FP16 scale on a boundary, the device's
// value, 1e-7 away, can step the other way.  A block is held to 1e-4, or to 2e-3 only where the reference counted
// such a rounding in it (on this fixture one block in twelve: a scale one FP16 step apart, 8.5e-4).
#include "strata/glm/glm_layer.hpp"
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

namespace {

constexpr int64_t P0 = 2100;     // cells of history
constexpr int STEPS = 4;
constexpr int64_t MAX_CTX = 2304;

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
/// go the other way: a code (x / d within 1e-4 of .5), or a block's FP16 scale (d within 1e-5 of the midpoint
/// between two FP16 values - then all 32 values of the block move by one FP16 step, 4.9e-4).  Found by this test:
/// at 2103 a block's amax was 0.79767389 in the reference and 0.79767448 on the device, either side of one.
int g_near = 0;
bool scale_on_boundary(float d) {
    const uint16_t h = k::f16_from_f32(d);
    const float v = k::f32_from_f16(h);
    if (d == v || h == 0) return false;
    const float nb = d > v ? k::f32_from_f16((uint16_t) (h + 1)) : k::f32_from_f16((uint16_t) (h - 1));
    const float mid = 0.5f * (v + nb);
    return std::fabs(d - mid) < 1e-5f * d;
}

/// The q8_1 rounding of an activation the quantized projections and the grouped experts read.
vector<double> q8_1(const vector<double>& x) {
    vector<double> out(x.size());
    for (size_t b = 0; b < x.size(); b += 32) {
        float amax = 0.0f;
        for (size_t i = b; i < b + 32; ++i) amax = std::max(amax, std::fabs((float) x[i]));
        const float d = amax / 127.0f;
        const double dh = k::f32_from_f16(k::f16_from_f32(d));
        if (amax != 0.0f) g_near += scale_on_boundary(d);
        for (size_t i = b; i < b + 32; ++i) {
            const float q = amax == 0.0f ? 0.0f : std::round((float) x[i] / d);
            if (amax != 0.0f) {
                const double fr = (double) x[i] / d - std::floor((double) x[i] / d);
                g_near += std::fabs(fr - 0.5) < 1e-4;
            }
            out[i] = q * dh;
        }
    }
    return out;
}
vector<double> lin_ref(const Lin& L, const vector<double>& x) {
    const vector<double> xa = L.quant ? q8_1(x) : x;
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
    vector<double> idx_cache, pooled;             // [ctx][256], [pools][128]
    glm::GlmLayerState st;
};

const float* up_f(const vector<float>& v) {
    float* p = upload(v);
    g_free.push_back(p);
    return p;
}

void build_mhc(Block& b, int D) {
    const int K = 4 * D;
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
    const vector<double> xq = q8_1(h);
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
        const vector<double> s = q8_1(swiglu_clamp(gg, uu, g.swiglu_clamp_exp));
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
        b.wq = make_lin(23, D, di); b.wk = make_lin(8, D, di); b.wv = make_lin(30, D, di);
        b.f_a = make_lin(0, D, Dh); b.f_b = make_lin(0, Dh, di); b.beta = make_lin(30, D, H);
        b.g_a = make_lin(0, D, Dh); b.g_b = make_lin(30, Dh, di); b.wo = make_lin(8, di, D);
        b.conv_q = gvec((size_t) di * 4, 0.5f); b.conv_k = gvec((size_t) di * 4, 0.5f); b.conv_v = gvec((size_t) di * 4, 0.5f);
        b.ssm_a.resize((size_t) H);
        for (auto& a : b.ssm_a) a = -std::exp(unit() * std::log(16.0f));
        b.dt_bias = gvec((size_t) di, 0.5f);
        b.o_norm = gvec((size_t) Dh, 0.1f, 1.0f);
        b.ffn_gate = make_lin(8, D, g.n_ff_dense); b.ffn_up = make_lin(23, D, g.n_ff_dense);
        b.ffn_down = make_lin(23, g.n_ff_dense, D);
        auto& w = b.w;
        w.wq = b.wq.dev; w.wk = b.wk.dev; w.wv = b.wv.dev; w.f_a = b.f_a.dev; w.f_b = b.f_b.dev; w.beta = b.beta.dev;
        w.g_a = b.g_a.dev; w.g_b = b.g_b.dev; w.wo = b.wo.dev;
        w.conv_q = up_f(b.conv_q); w.conv_k = up_f(b.conv_k); w.conv_v = up_f(b.conv_v);
        w.ssm_a = up_f(b.ssm_a); w.dt_bias = up_f(b.dt_bias); w.o_norm = up_f(b.o_norm);
        w.ffn_gate = b.ffn_gate.dev; w.ffn_up = b.ffn_up.dev; w.ffn_down = b.ffn_down.dev;
        // state: a conv history and a recurrent state as after some tokens
        const vector<float> hist = gvec((size_t) 3 * di * 3, 1.0f), stt = gvec((size_t) Dh * H * Dh, 0.05f);
        b.conv_hist.assign(hist.begin(), hist.end());
        b.kda_state.assign(stt.size(), 0.0);
        vector<float> dev_state(stt.size());
        for (int hh = 0; hh < H; ++hh)   // stt is the reference's [h][j][i]; the device's is [i][h][j]
            for (int j = 0; j < Dh; ++j)
                for (int i = 0; i < Dh; ++i) {
                    const float v = stt[((size_t) hh * Dh + j) * Dh + i];
                    b.kda_state[((size_t) hh * Dh + j) * Dh + i] = v;
                    dev_state[((size_t) i * H + hh) * Dh + j] = v;
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
        // the MoE: router F32, 16 IQ4_XS experts (blobs [gate | up | down]), the shared expert
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
        b.sh_gate = make_lin(23, D, g.n_ff_shexp); b.sh_up = make_lin(23, D, g.n_ff_shexp);
        b.sh_down = make_lin(8, g.n_ff_shexp, D);
        w.sh_gate = b.sh_gate.dev; w.sh_up = b.sh_up.dev; w.sh_down = b.sh_down.dev;
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
            g_near = 0;
            // The reference takes the block's input from the device: the q8_1 roundings make a quantized projection
            // jump where an input sits on a rounding boundary, so a 1e-6 difference carried in from the blocks before
            // would sometimes show as a 1e-3 one here.  Each block is checked on its own arithmetic; the state it
            // carries from token to token is checked separately below.
            {
                const vector<float> xin = download(d_x, X.size());
                X.assign(xin.begin(), xin.end());
            }
            if (!glm::glm_layer_decode(g, il, b.w, b.st, ctx, d_x, stream, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            check(cudaStreamSynchronize(stream), "layer");
            // the reference
            vector<double> h;
            double post[4], comb[4][4];
            mhc_pre_ref(X, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base, g, h, post, comb);
            rms_w(h, b.attn_norm, g.rms_eps);
            const bool mla = g.is_mla[(size_t) il] != 0, full = g.idx_full[(size_t) il] != 0;
            const vector<double> f = mla ? mla_ref(b, g, h, pos, full, r) : kda_ref(b, g, h);
            mhc_post_ref(f, post, comb, X);
            if (mla && full) {
                vector<int32_t> got = download(ctx.sel_ids, r.sel.size());
                std::sort(got.begin(), got.end());
                const bool ok = got == r.sel;
                if (!ok) ++bad;
                std::printf("  pos %lld block %d: the indexer's %zu cells %s\n", (long long) pos, il, r.sel.size(),
                            ok ? "match" : "*** DIFFER ***");
            }
            mhc_pre_ref(X, b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base, g, h, post, comb);
            rms_w(h, b.ffn_norm, g.rms_eps);
            const vector<double> ff = ffn_ref(b, g, h, il, r);
            mhc_post_ref(ff, post, comb, X);
            if (il >= g.n_dense_lead) {
                const vector<int32_t> got = download(ctx.eids, (size_t) g.n_expert_used);
                const bool ok = got == r.eids;
                if (!ok) ++bad;
                std::printf("  pos %lld block %d: experts %d %d %d %d %s\n", (long long) pos, il, r.eids[0], r.eids[1],
                            r.eids[2], r.eids[3], ok ? "match, in order" : "*** DIFFER ***");
            }
            if (!mla) {   // the carried state: the conv history and the recurrent state
                const vector<float> hist = download(b.st.conv_hist, b.conv_hist.size());
                const vector<float> dst = download(b.st.kda_state, b.kda_state.size());
                vector<float> st_ref_order(dst.size());
                for (int hh = 0; hh < H; ++hh)
                    for (int j = 0; j < Dh; ++j)
                        for (int i = 0; i < Dh; ++i)
                            st_ref_order[((size_t) hh * Dh + j) * Dh + i] = dst[((size_t) i * H + hh) * Dh + j];
                const double rh = rel_l1(b.conv_hist, hist), rs = rel_l1(b.kda_state, st_ref_order);
                const bool ok = rh <= 1e-6 && rs <= 1e-5;
                if (!ok) ++bad;
                std::printf("  pos %lld block %d: KDA conv history rel %.3e, state rel %.3e %s\n", (long long) pos, il,
                            rh, rs, ok ? "" : "*** over ***");
            }
            const double rel = rel_l1(X, download(d_x, X.size()));
            // 1e-4; or 2e-3 where the reference rounded a q8_1 code or scale on a boundary (the device may have gone
            // the other way: one FP16 step of a block's scale moved this one 8.5e-4).  Unexplained, it fails.
            const bool tight = rel <= 1e-4, boundary = g_near > 0 && rel <= 2e-3;
            if (!tight && !boundary) ++bad;
            std::printf("  pos %lld block %d: the four streams rel %.3e%s\n", (long long) pos, il, rel,
                        tight ? "" : boundary ? "  (a q8_1 rounding on a boundary in this block)" : "  *** over 1e-4 ***");
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
