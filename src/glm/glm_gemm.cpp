// src/glm/glm_gemm.cpp - see include/strata/glm/glm_gemm.hpp.
#include "strata/glm/glm_gemm.hpp"

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/glm_moe.hpp"
#include "strata/prefill/gemm.hpp"
#if defined(STRATA_PREFILL_MMQ)
#include "strata/prefill/moe_mmq.hpp"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::glm {
namespace k = strata::kernels;
namespace {

/// MMQ's q8_1 layout for a weight type (llama.cpp's mmq_get_q8_1_ds_layout): 0 D4 (FP32 scale), 1 DS4 (FP16 scale
/// and sum), -1 another (D2S6) or unknown - not taken here, the reference has no reading of it.
int mmq_layout(int type) {
    switch (type) {
        case 6: case 8: case 11: case 14:                         // Q5_0 Q8_0 Q3_K Q6_K
        case 16: case 17: case 18: case 20: case 21: case 22: case 23:   // the i-quants but IQ1_S / IQ1_M
            return 0;
        case 2: case 3: case 7: case 12: case 13:                 // Q4_0 Q4_1 Q5_1 Q4_K Q5_K
            return 1;
        default:
            return -1;
    }
}

uint64_t block_bytes(int type) {   // bytes of one block, for the experts' stride (MMQ steps it in blocks)
    const int64_t per = type == 8 || type == 2 || type == 3 || type == 6 || type == 7 || type == 20 ? 32 : 256;
    return k::iq_row_bytes(type, per);
}

void die(const char* what) {
    const cudaError_t e = cudaGetLastError();
    std::fprintf(stderr, "GlmPromptGemm: %s%s%s\n", what, e != cudaSuccess ? ": " : "",
                 e != cudaSuccess ? cudaGetErrorString(e) : "");
    std::exit(1);
}

}  // namespace

bool glm_mmq_fp16_scale(int type) { return mmq_layout(type) == 1; }

GlmProjPath glm_prompt_path(int type, int64_t n_in, int64_t n_out) {
    if (type == 30) return GlmProjPath::Bf16Gemm;
#if defined(STRATA_PREFILL_MMQ)
    namespace mmq = strata::prefill::mmq;
    if (mmq_layout(type) >= 0 && n_in % 256 == 0 && mmq::built() && mmq::supported(type) && mmq::fits(type, n_out))
        return GlmProjPath::Mmq;
#else
    (void) n_in;
    (void) n_out;
#endif
    return GlmProjPath::Gemv;
}

bool glm_prompt_experts_mmq(const k::NativeExpertLayout& L, uint64_t stride) {
    if (stride == 0 || L.bytes == 0) return false;
    const uint64_t gb = block_bytes(L.gu_type), db = block_bytes(L.d_type);
    if (gb == 0 || db == 0 || stride % gb != 0 || stride % db != 0 || L.down_off % db != 0) return false;
    return glm_prompt_path(L.gu_type, L.n_embd, 2 * L.n_ff) == GlmProjPath::Mmq &&
           glm_prompt_path(L.d_type, L.n_ff, L.n_embd) == GlmProjPath::Mmq;
}

GlmPromptGemm::~GlmPromptGemm() { release(); }

void GlmPromptGemm::release() {
#if defined(STRATA_PREFILL_MMQ)
    delete static_cast<strata::prefill::mmq::Context*>(mmq_);
#endif
    delete static_cast<strata::prefill::Gemm*>(gemm_);
    mmq_ = gemm_ = nullptr;
    for (void* p : {xq_, xq2_, (void*) ident_, (void*) bounds1_, (void*) xb_, (void*) gu_, (void*) hh_})
        if (p) cudaFree(p);
    xq_ = xq2_ = nullptr;
    ident_ = bounds1_ = nullptr;
    xb_ = nullptr;
    gu_ = hh_ = nullptr;
}

bool GlmPromptGemm::init(int64_t max_tokens, int64_t max_in, int64_t n_embd, int64_t n_ff, int64_t k,
                         std::string& err) {
    release();
    max_tokens_ = max_tokens;
    max_in_ = max_in;
    n_embd_ = n_embd;
    n_ff_ = n_ff;
    k_ = k;
    const int64_t T = max_tokens, E = std::max<int64_t>(T * k, 1);
    size_t xq_bytes = 0, xq2_bytes = 0;
#if defined(STRATA_PREFILL_MMQ)
    namespace mmq = strata::prefill::mmq;
    xq_bytes = std::max(mmq::q8_bytes(T, max_in), mmq::q8_bytes(E, n_embd));
    xq2_bytes = mmq::q8_bytes(E, std::max<int64_t>(n_ff, 1));
#endif
    auto alloc = [&](void** p, size_t bytes, const char* what) {
        if (cudaMalloc(p, std::max<size_t>(bytes, 256)) == cudaSuccess) return true;
        cudaGetLastError();
        err = std::string("GlmPromptGemm: ") + what + ", " + std::to_string(bytes) + " bytes";
        return false;
    };
    if (!alloc(&xq_, xq_bytes, "MMQ activations") || !alloc(&xq2_, xq2_bytes, "MMQ expert activations") ||
        !alloc((void**) &ident_, (size_t) E * 4, "row table") || !alloc((void**) &bounds1_, 8, "bounds") ||
        !alloc((void**) &xb_, (size_t) (T * max_in) * 2, "BF16 activations") ||
        !alloc((void**) &gu_, (size_t) (E * 2 * n_ff) * 4, "expert gate / up") ||
        !alloc((void**) &hh_, (size_t) (E * n_ff) * 4, "expert hidden")) {
        release();
        return false;
    }
#if defined(STRATA_PREFILL_MMQ)
    mmq::iota(ident_, E, nullptr);
    mmq_ = new mmq::Context();
#endif
    return cudaDeviceSynchronize() == cudaSuccess;
}

void GlmPromptGemm::linear(const GlmLinear& w, const float* x, float* y, int n, void* q8, void* stream) {
    if (n > max_tokens_ || w.n_in > max_in_) die("a call wider than init's");
    // one row: the GEMV (rocBLAS has no kernel for a one-row BF16 product on gfx1151: hipErrorInvalidDeviceFunction)
    switch (n == 1 ? GlmProjPath::Gemv : glm_prompt_path(w.type, w.n_in, w.n_out)) {
        case GlmProjPath::Gemv:
            k::glm_linear(w, x, y, n, q8, stream);
            return;
        case GlmProjPath::Bf16Gemm: {
            if (!gemm_ || gemm_stream_ != stream) {   // prefill::Gemm binds its handle to one stream
                delete static_cast<strata::prefill::Gemm*>(gemm_);
                auto* g = new strata::prefill::Gemm();
                std::string e;
                if (!g->init(stream, 0, e)) {
                    std::fprintf(stderr, "GlmPromptGemm: %s\n", e.c_str());
                    std::exit(1);
                }
                gemm_ = g;
                gemm_stream_ = stream;
            }
            k::f32_to_bf16_bulk(x, xb_, (int64_t) n * w.n_in, stream);
            static_cast<strata::prefill::Gemm*>(gemm_)->bf16(xb_, (const uint16_t*) w.data, y, n, w.n_out, w.n_in);
            return;
        }
        case GlmProjPath::Mmq: {
#if defined(STRATA_PREFILL_MMQ)
            namespace mmq = strata::prefill::mmq;
            const int32_t b[2] = {0, n};   // pageable: staged before the call returns
            cudaMemcpyAsync(bounds1_, b, sizeof b, cudaMemcpyHostToDevice, (cudaStream_t) stream);
            mmq::quantize(x, nullptr, xq_, w.type, w.n_in, w.n_in, n, stream);
            mmq::Product p;
            p.w = w.data;
            p.type = w.type;
            p.w_rows = w.n_out;
            p.w_cols = w.n_in;
            p.expert_bytes = mmq::matrix_bytes(w.type, w.n_out, w.n_in);
            p.n = 1;
            p.xq = xq_;
            p.bounds = bounds1_;
            p.ids = ident_;
            p.total_rows = n;
            p.max_rows = n;
            p.dst = y;
            p.ld_dst = w.n_out;
            static_cast<mmq::Context*>(mmq_)->run(p, stream);
#endif
            return;
        }
    }
}

void GlmPromptGemm::experts(const k::NativeExpertLayout& L, const uint8_t* base, uint64_t stride, int n_slots,
                            const int32_t* bounds, const int32_t* tok, const int32_t* ent, int rows, int max_rows,
                            const float* h, float* parts, float swiglu_limit, void* stream) {
    if (rows <= 0) return;
    if (rows > max_tokens_ * k_ || L.n_embd != n_embd_ || L.n_ff != n_ff_) die("experts wider than init's");
#if defined(STRATA_PREFILL_MMQ)
    namespace mmq = strata::prefill::mmq;
    const int64_t D = L.n_embd, F = L.n_ff;
    mmq::quantize(h, tok, xq_, L.gu_type, D, D, rows, stream);   // row r: token tok[r]'s h
    mmq::Product gu;
    gu.w = base;
    gu.type = L.gu_type;
    gu.w_rows = 2 * F;   // gate rows, then up rows
    gu.w_cols = D;
    gu.expert_bytes = stride;
    gu.n = n_slots;
    gu.xq = xq_;
    gu.bounds = bounds;
    gu.ids = ident_;
    gu.total_rows = rows;
    gu.max_rows = max_rows;
    gu.dst = gu_;
    gu.ld_dst = 2 * F;
    static_cast<mmq::Context*>(mmq_)->run(gu, stream);
    k::swiglu_clamp_rows(gu_, hh_, rows, F, swiglu_limit, stream);
    mmq::quantize(hh_, nullptr, xq2_, L.d_type, F, F, rows, stream);
    mmq::Product dn = gu;
    dn.w = base + L.down_off;
    dn.type = L.d_type;
    dn.w_rows = D;
    dn.w_cols = F;
    dn.xq = xq2_;
    dn.ids = ent;   // the row goes to its entry's part
    dn.dst = parts;
    dn.ld_dst = D;
    static_cast<mmq::Context*>(mmq_)->run(dn, stream);
#else
    (void) L; (void) base; (void) stride; (void) n_slots; (void) bounds; (void) tok; (void) ent; (void) max_rows;
    (void) h; (void) parts; (void) swiglu_limit; (void) stream;
    die("no MMQ in this build");
#endif
}

}  // namespace strata::glm
