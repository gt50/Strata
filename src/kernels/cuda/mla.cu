// src/kernels/cuda/mla.cu - see include/strata/kernels/mla.hpp.
#include "strata/kernels/mla.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int L = kMlaLatent;
constexpr int CH = kMlaChunk;
constexpr int HG = kMlaHeadGroup;
constexpr int LW = L / 2;   // a latent row in 32-bit words (FP16 pairs)
static_assert(L == 512 && CH == 32 && HG == 8, "mla_decode_partial_kernel: a warp per head, a lane per position, "
                                               "a thread per word of the row");

__device__ __forceinline__ float bf16_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ float2 f16x2(uint32_t u) {
    return make_float2(__half2float(__ushort_as_half((unsigned short) (u & 0xffffu))),
                       __half2float(__ushort_as_half((unsigned short) (u >> 16))));
}

/// A warp per (head, output row): the row's n_in BF16 weights into registers once (8 a lane per 256), then a dot
/// with each token's input, summed across the warp.
template <int NV>   // n_in = 256 NV
__global__ void __launch_bounds__(256) mla_head_gemv_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                                            float* __restrict__ y, int heads, int n_out, int n_tok) {
    constexpr int n_in = 256 * NV;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = (int64_t) blockIdx.x * 8 + warp;   // h * n_out + o
    if (row >= (int64_t) heads * n_out) return;
    const int h = (int) (row / n_out), o = (int) (row % n_out);
    float wf[8 * NV];
    const uint4* w4 = reinterpret_cast<const uint4*>(w + row * n_in);
#pragma unroll
    for (int v = 0; v < NV; ++v) {   // elements 8 (lane + 32 v) .. + 7
        const uint4 u = w4[lane + 32 * v];
        wf[8 * v + 0] = bf16_lo(u.x); wf[8 * v + 1] = bf16_hi(u.x);
        wf[8 * v + 2] = bf16_lo(u.y); wf[8 * v + 3] = bf16_hi(u.y);
        wf[8 * v + 4] = bf16_lo(u.z); wf[8 * v + 5] = bf16_hi(u.z);
        wf[8 * v + 6] = bf16_lo(u.w); wf[8 * v + 7] = bf16_hi(u.w);
    }
    for (int t = 0; t < n_tok; ++t) {
        const float4* x4 = reinterpret_cast<const float4*>(x + ((size_t) t * heads + h) * n_in);
        float acc = 0.0f;
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const float4 a = x4[2 * (lane + 32 * v)], b = x4[2 * (lane + 32 * v) + 1];
            acc = fmaf(wf[8 * v + 0], a.x, acc); acc = fmaf(wf[8 * v + 1], a.y, acc);
            acc = fmaf(wf[8 * v + 2], a.z, acc); acc = fmaf(wf[8 * v + 3], a.w, acc);
            acc = fmaf(wf[8 * v + 4], b.x, acc); acc = fmaf(wf[8 * v + 5], b.y, acc);
            acc = fmaf(wf[8 * v + 6], b.z, acc); acc = fmaf(wf[8 * v + 7], b.w, acc);
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) y[((size_t) t * heads + h) * n_out + o] = acc;
    }
}

/// A block per token: the row's RMS (FP32, over L), then the scaled row rounded to FP16 into the cache.
__global__ void __launch_bounds__(256) mla_kv_store_kernel(const float* __restrict__ kv, const float* __restrict__ gamma,
                                                           float eps, uint16_t* __restrict__ cache, int64_t row0,
                                                           const int32_t* __restrict__ step) {
    if (step) row0 = step[kStepPos];
    __shared__ float part[8];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float2 v = reinterpret_cast<const float2*>(kv + (size_t) t * L)[tid];
    float ss = v.x * v.x + v.y * v.y;
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    if ((tid & 31) == 0) part[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = rsqrtf(tot / (float) L + eps);
    const float2 g = reinterpret_cast<const float2*>(gamma)[tid];
    const uint32_t lo = f16_from_f32(v.x * inv * g.x), hi = f16_from_f32(v.y * inv * g.y);
    reinterpret_cast<uint32_t*>(cache + (size_t) (row0 + t) * L)[tid] = lo | (hi << 16);
}

/// The CH latent rows p0 .. p0 + CH - 1 of a selection (ids, or the rows themselves) into `kv`, word w of row p at
/// p * LW + (w ^ (p & 31)): a lane per row reading one word, or a thread per word of one row, both land on 32
/// different banks, without padding.  Rows at or past n_sel are zeros.  256 threads.
__device__ __forceinline__ void mla_stage_rows(uint32_t* kv, const uint16_t* __restrict__ cache,
                                               const int32_t* __restrict__ ids, int64_t p0, int64_t n_sel) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int k = 0; k < CH * LW / 4 / 256; ++k) {   // uint4 = 4 words; 64 of them a row
        const int idx = tid + 256 * k, p = idx / (LW / 4), w = (idx % (LW / 4)) * 4;
        uint4 u = make_uint4(0u, 0u, 0u, 0u);
        if (p0 + p < n_sel) {
            const int64_t r = ids ? (int64_t) ids[p0 + p] : p0 + p;
            u = reinterpret_cast<const uint4*>(cache + (size_t) r * L)[w / 4];
        }
        uint32_t* row = kv + p * LW;
        const int sw = p & 31;
        row[(w + 0) ^ sw] = u.x; row[(w + 1) ^ sw] = u.y; row[(w + 2) ^ sw] = u.z; row[(w + 3) ^ sw] = u.w;
    }
}

/// The unscaled score of head `warp`'s query row (qs, [HG][L]) with staged row `lane`.
__device__ __forceinline__ float mla_score(const float* qs, const uint32_t* kv, int warp, int lane) {
    const float2* q2 = reinterpret_cast<const float2*>(qs + warp * L);
    const uint32_t* row = kv + lane * LW;
    float s = 0.0f;
#pragma unroll 8
    for (int w = 0; w < LW; ++w) {
        const float2 c = f16x2(row[w ^ lane]);
        const float2 q = q2[w];
        s = fmaf(q.x, c.x, s);
        s = fmaf(q.y, c.y, s);
    }
    return s;
}

/// A block per (chunk of CH selected positions, HG heads), 256 threads.
///   staging: the HG query rows (FP32) and the chunk's CH latent rows (FP16, mla_stage_rows): exactly 48 KB, CUDA's
///            static limit
///   scores:  a warp per head, a lane per position; the chunk's max m and sum l of exp(s - m) per head
///   values:  a thread per word (2 columns) for all HG heads: acc[h] = sum_p exp(s_hp - m_h) c_p
/// The partial (m, l, acc) goes to scratch for mla_decode_merge_kernel.  The count is n_sel, or step[kStepWidth] when
/// `step` is given; the grid may then hold more chunks than the count needs (a captured step sizes it by the
/// capacity), and those return at once.
__global__ void __launch_bounds__(256) mla_decode_partial_kernel(const float* __restrict__ q_abs,
                                                                 const uint16_t* __restrict__ cache,
                                                                 const int32_t* __restrict__ ids, int64_t n_sel_arg,
                                                                 const int32_t* __restrict__ step, float scale,
                                                                 float* __restrict__ part_acc,
                                                                 float* __restrict__ part_ml, int heads) {
    __shared__ __align__(16) float qs[HG * L];   // the query rows; then the weights exp(s - m) [HG][CH]
    __shared__ uint32_t kv[CH * LW];
    const int64_t n_sel = step ? (int64_t) step[kStepWidth] : n_sel_arg;
    const int64_t chunk = blockIdx.x;
    const int h0 = blockIdx.y * HG, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int64_t p0 = chunk * CH;
    if (p0 >= n_sel) return;   // uniform over the block
    {
        const float4* src = reinterpret_cast<const float4*>(q_abs + (size_t) h0 * L);
#pragma unroll
        for (int k = 0; k < HG * L / 4 / 256; ++k) reinterpret_cast<float4*>(qs)[tid + 256 * k] = src[tid + 256 * k];
    }
    mla_stage_rows(kv, cache, ids, p0, n_sel);
    __syncthreads();
    // scores: head h0 + warp, position p0 + lane
    const bool valid = p0 + lane < n_sel;
    float s = mla_score(qs, kv, warp, lane);
    s = valid ? s * scale : -INFINITY;
    float m = s;
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    const float e = valid ? expf(s - m) : 0.0f;
    float l = e;
    for (int off = 16; off > 0; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
    const int h = h0 + warp;
    if (lane == 0) {
        part_ml[(chunk * heads + h) * 2 + 0] = m;
        part_ml[(chunk * heads + h) * 2 + 1] = l;
    }
    __syncthreads();   // every read of the query rows done: the weights go over them
    float* ps = qs;    // [HG][CH]
    ps[warp * CH + lane] = e;
    __syncthreads();
    // values: word tid of every row, for all HG heads
    float acc[HG][2] = {};
#pragma unroll 4
    for (int p = 0; p < CH; ++p) {
        const float2 c = f16x2(kv[p * LW + (tid ^ (p & 31))]);
#pragma unroll
        for (int hh = 0; hh < HG; ++hh) {
            const float wgt = ps[hh * CH + p];
            acc[hh][0] = fmaf(wgt, c.x, acc[hh][0]);
            acc[hh][1] = fmaf(wgt, c.y, acc[hh][1]);
        }
    }
#pragma unroll
    for (int hh = 0; hh < HG; ++hh)
        reinterpret_cast<float2*>(part_acc + (chunk * heads + h0 + hh) * L)[tid] = make_float2(acc[hh][0], acc[hh][1]);
}

/// A block per head, a thread per two columns: the chunks' partials merged with the log-sum-exp rescale.  The chunk
/// count from n_chunks, or from step[kStepWidth].
__global__ void __launch_bounds__(256) mla_decode_merge_kernel(const float* __restrict__ part_acc,
                                                               const float* __restrict__ part_ml, int64_t n_chunks_arg,
                                                               const int32_t* __restrict__ step, float* __restrict__ o,
                                                               int heads) {
    const int64_t n_chunks = step ? ((int64_t) step[kStepWidth] + CH - 1) / CH : n_chunks_arg;
    const int h = blockIdx.x, tid = threadIdx.x;
    float M = -INFINITY;
    for (int64_t c = 0; c < n_chunks; ++c) M = fmaxf(M, part_ml[(c * heads + h) * 2]);
    float lsum = 0.0f, a0 = 0.0f, a1 = 0.0f;
    for (int64_t c = 0; c < n_chunks; ++c) {
        const float wgt = expf(part_ml[(c * heads + h) * 2] - M);
        lsum = fmaf(wgt, part_ml[(c * heads + h) * 2 + 1], lsum);
        const float2 a = reinterpret_cast<const float2*>(part_acc + (c * heads + h) * L)[tid];
        a0 = fmaf(wgt, a.x, a0);
        a1 = fmaf(wgt, a.y, a1);
    }
    const float inv = 1.0f / lsum;
    reinterpret_cast<float2*>(o + (size_t) h * L)[tid] = make_float2(a0 * inv, a1 * inv);
}

/// A block per (query, HG heads), 256 threads: the query's selection walked in chunks of CH rows with an online
/// softmax (flash attention's running max and sum), so a prompt's many queries need no scratch.  Per chunk: the rows
/// staged as in decode; a warp per head scores a lane per row; the running max m moves to max(m, the chunk's), the
/// sums so far are rescaled by exp(m_old - m_new), and each thread adds its 2 columns for all HG heads.  The query's
/// count is steps[q][kStepWidth]; ids [nq][cap] or null (rows 0 .. count - 1).  Shared memory: the query rows 16 KB,
/// the chunk 32 KB, the weights and rescales 1 KB (past CUDA's static 48 KB: dynamic, opted in by the host).
__global__ void __launch_bounds__(256) mla_prompt_attn_kernel(const float* __restrict__ q_abs,
                                                              const uint16_t* __restrict__ cache,
                                                              const int32_t* __restrict__ ids,
                                                              const int32_t* __restrict__ steps, int64_t cap,
                                                              float scale, float* __restrict__ o, int heads) {
    extern __shared__ __align__(16) float psm[];
    float* qs = psm;                                              // [HG][L]
    uint32_t* kv = reinterpret_cast<uint32_t*>(qs + HG * L);      // [CH][LW]
    float* ps = reinterpret_cast<float*>(kv + CH * LW);           // [HG][CH]
    float* alpha = ps + HG * CH;                                  // [HG] this chunk's rescale of the running sums
    float* lsum = alpha + HG;                                     // [HG]
    const int64_t q = blockIdx.x;
    const int h0 = blockIdx.y * HG, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int64_t n_sel = steps[q * kStepCount + kStepWidth];
    const int32_t* qids = ids ? ids + q * cap : nullptr;
    {
        const float4* src = reinterpret_cast<const float4*>(q_abs + ((size_t) q * heads + h0) * L);
#pragma unroll
        for (int k = 0; k < HG * L / 4 / 256; ++k) reinterpret_cast<float4*>(qs)[tid + 256 * k] = src[tid + 256 * k];
    }
    float m_run = -INFINITY, l_run = 0.0f;   // head h0 + warp's, the same in every lane
    float acc[HG][2] = {};
    for (int64_t p0 = 0; p0 < n_sel; p0 += CH) {
        __syncthreads();   // the last chunk's reads of the rows and weights done
        mla_stage_rows(kv, cache, qids, p0, n_sel);
        __syncthreads();
        const bool valid = p0 + lane < n_sel;
        float s = mla_score(qs, kv, warp, lane);
        s = valid ? s * scale : -INFINITY;
        float m_c = s;
        for (int off = 16; off > 0; off >>= 1) m_c = fmaxf(m_c, __shfl_xor_sync(0xffffffffu, m_c, off));
        const float m_new = fmaxf(m_run, m_c);   // finite: every chunk has a valid row
        const float a = expf(m_run - m_new);     // 0 on the first chunk
        const float e = valid ? expf(s - m_new) : 0.0f;
        float l_c = e;
        for (int off = 16; off > 0; off >>= 1) l_c += __shfl_xor_sync(0xffffffffu, l_c, off);
        l_run = fmaf(l_run, a, l_c);
        m_run = m_new;
        ps[warp * CH + lane] = e;
        if (lane == 0) alpha[warp] = a;
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < HG; ++hh) {
            acc[hh][0] *= alpha[hh];
            acc[hh][1] *= alpha[hh];
        }
#pragma unroll 4
        for (int p = 0; p < CH; ++p) {
            const float2 c = f16x2(kv[p * LW + (tid ^ (p & 31))]);
#pragma unroll
            for (int hh = 0; hh < HG; ++hh) {
                const float wgt = ps[hh * CH + p];
                acc[hh][0] = fmaf(wgt, c.x, acc[hh][0]);
                acc[hh][1] = fmaf(wgt, c.y, acc[hh][1]);
            }
        }
    }
    if (lane == 0) lsum[warp] = l_run;
    __syncthreads();
#pragma unroll
    for (int hh = 0; hh < HG; ++hh) {
        const float inv = lsum[hh] > 0.0f ? 1.0f / lsum[hh] : 0.0f;   // an empty selection: zeros, not NaN
        reinterpret_cast<float2*>(o + ((size_t) q * heads + h0 + hh) * L)[tid] =
            make_float2(acc[hh][0] * inv, acc[hh][1] * inv);
    }
}
constexpr size_t kPromptSmem = sizeof(float) * (HG * L + CH * LW + HG * CH + 2 * HG);

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void mla_head_gemv(const float* x, const uint16_t* w, float* y, int heads, int n_in, int n_out, int n_tok,
                   void* stream) {
    if (!x || !w || !y || heads <= 0 || n_out <= 0 || n_tok <= 0 || (n_in != 256 && n_in != 512)) {
        std::fprintf(stderr, "mla_head_gemv: invalid arguments (n_in %d: 256 or 512)\n", n_in);
        std::exit(1);
    }
    const int64_t rows = (int64_t) heads * n_out;
    const unsigned blocks = (unsigned) ((rows + 7) / 8);
    if (n_in == 256)
        mla_head_gemv_kernel<1><<<blocks, 256, 0, (cudaStream_t) stream>>>(x, w, y, heads, n_out, n_tok);
    else
        mla_head_gemv_kernel<2><<<blocks, 256, 0, (cudaStream_t) stream>>>(x, w, y, heads, n_out, n_tok);
    check_launch("mla_head_gemv");
}

void mla_kv_store(const float* kv, const float* gamma, float eps, uint16_t* cache, int64_t row0, int n_tok,
                  void* stream, const int32_t* step) {
    if (!kv || !gamma || !cache || row0 < 0 || n_tok <= 0 || (step && n_tok != 1)) {
        std::fprintf(stderr, "mla_kv_store: invalid arguments\n");
        std::exit(1);
    }
    mla_kv_store_kernel<<<(unsigned) n_tok, 256, 0, (cudaStream_t) stream>>>(kv, gamma, eps, cache, row0, step);
    check_launch("mla_kv_store");
}

uint64_t mla_decode_attn_scratch_floats(int heads, int64_t max_sel) {
    if (heads <= 0 || max_sel <= 0) return 0;
    const uint64_t n_chunks = (uint64_t) ((max_sel + CH - 1) / CH);
    return n_chunks * (uint64_t) heads * (uint64_t) (L + 2);
}

void mla_decode_attn(const float* q_abs, const uint16_t* cache, const int32_t* ids, int64_t n_sel, float scale,
                     float* scratch, float* o, int heads, void* stream) {
    if (!q_abs || !cache || !scratch || !o || n_sel <= 0 || heads <= 0 || heads % HG != 0) {
        std::fprintf(stderr, "mla_decode_attn: invalid arguments\n");
        std::exit(1);
    }
    const int64_t n_chunks = (n_sel + CH - 1) / CH;
    float* part_acc = scratch;
    float* part_ml = scratch + (size_t) n_chunks * heads * L;
    const cudaStream_t s = (cudaStream_t) stream;
    mla_decode_partial_kernel<<<dim3((unsigned) n_chunks, (unsigned) (heads / HG)), 256, 0, s>>>(
        q_abs, cache, ids, n_sel, nullptr, scale, part_acc, part_ml, heads);
    check_launch("mla_decode_attn (partial)");
    mla_decode_merge_kernel<<<(unsigned) heads, LW, 0, s>>>(part_acc, part_ml, n_chunks, nullptr, o, heads);
    check_launch("mla_decode_attn (merge)");
}

void mla_decode_attn_step(const float* q_abs, const uint16_t* cache, const int32_t* ids, const int32_t* step,
                          int64_t cap, float scale, float* scratch, float* o, int heads, void* stream) {
    if (!q_abs || !cache || !step || !scratch || !o || cap <= 0 || heads <= 0 || heads % HG != 0) {
        std::fprintf(stderr, "mla_decode_attn_step: invalid arguments\n");
        std::exit(1);
    }
    const int64_t max_chunks = (cap + CH - 1) / CH;   // the grid: by capacity, so a captured step stays valid
    float* part_acc = scratch;
    float* part_ml = scratch + (size_t) max_chunks * heads * L;
    const cudaStream_t s = (cudaStream_t) stream;
    mla_decode_partial_kernel<<<dim3((unsigned) max_chunks, (unsigned) (heads / HG)), 256, 0, s>>>(
        q_abs, cache, ids, 0, step, scale, part_acc, part_ml, heads);
    check_launch("mla_decode_attn_step (partial)");
    mla_decode_merge_kernel<<<(unsigned) heads, LW, 0, s>>>(part_acc, part_ml, 0, step, o, heads);
    check_launch("mla_decode_attn_step (merge)");
}

void mla_prompt_attn(const float* q_abs, const uint16_t* cache, const int32_t* ids, const int32_t* steps, int64_t cap,
                     int64_t nq, float scale, float* o, int heads, void* stream) {
    if (!q_abs || !cache || !steps || !o || (ids && cap <= 0) || nq <= 0 || heads <= 0 || heads % HG != 0 ||
        heads / HG > 65535) {
        std::fprintf(stderr, "mla_prompt_attn: invalid arguments\n");
        std::exit(1);
    }
    {   // 49 KB of dynamic shared memory: past the 48 KB a CUDA kernel gets without asking, once a device
        static bool attrs[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64 && !attrs[dev]) {
            if (cudaFuncSetAttribute(mla_prompt_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int) kPromptSmem) != cudaSuccess) {
                std::fprintf(stderr, "mla_prompt_attn: %zu bytes of shared memory: %s\n", kPromptSmem,
                             cudaGetErrorString(cudaGetLastError()));
                std::exit(1);
            }
            attrs[dev] = true;
        }
    }
    mla_prompt_attn_kernel<<<dim3((unsigned) nq, (unsigned) (heads / HG)), 256, kPromptSmem, (cudaStream_t) stream>>>(
        q_abs, cache, ids, steps, cap, scale, o, heads);
    check_launch("mla_prompt_attn");
}

}  // namespace strata::kernels
