// include/strata/glm/glm_loader.hpp - GLM-5.3-Flash's weights from its GGUF (llama.cpp architecture glm5-next) into
// the GlmModelWeights that GlmModel runs.
//
// The tensors are the ones llama.cpp's loader creates (upstream src/models/glm5-next.cpp, load_arch_tensors), under
// its names (llama-arch.cpp) and in the shapes and layouts its converter writes (conversion/glm.py, Glm5NextModel):
//   - projections ([n_out][n_in] rows, dim 0 = n_in): uploaded as they are, any type glm_linear takes
//   - the norms, the mHC tensors, ssm_a (-exp(A_log), the converter's), ssm_dt.bias, exp_probs_b, the indexer's
//     LayerNorm, compressor gate and positional bias (F32 by the converter's tensor_force_quant), the conv1d weights
//     ({d_conv, 1, d_inner, 1}: [d_inner][d_conv]): as F32 vectors, from F32, F16 or BF16
//   - attn_k_b {DK, kv_lora, H} and attn_v_b {kv_lora, DV, H}: already the [H][kv_lora][DK] and [H][DV][kv_lora]
//     mla_head_gemv reads, converted to the BF16 it takes (from F32 / F16 by rounding, from a quantized type through
//     iq_dequant_f32 - a BF16 matrix where llama.cpp multiplies the quantized one)
//   - the experts, three stacked tensors a layer ({n_embd, n_ff, n_expert} gate and up, {n_ff, n_embd, n_expert}
//     down): repacked into one [gate rows | up rows | down rows] blob per expert (native_expert_layout), in device
//     memory or in mapped host memory the GPU reads in place
//
// glm_plan checks the whole file against the geometry from the tensor directory alone - every tensor there, its
// shape, a type the kernels take, its bytes inside the file - before anything is allocated, and counts the bytes.
// The MTP block (blk.<n_layers>) is loaded when the geometry has one and the options ask for it.
//
// Experts larger than memory stream from the file through a GlmExpertCache (GlmLoadOptions::expert_cache_slots).
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/geometry_glm5_next.hpp"
#include "strata/glm/glm_expert_cache.hpp"
#include "strata/glm/glm_model.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::glm {

enum class GlmExpertHome {
    Device,       ///< device memory
    MappedHost,   ///< pinned host memory mapped into the device's address space (on an APU, the same DRAM)
};

struct GlmLoadOptions {
    GlmExpertHome experts = GlmExpertHome::Device;
    bool mtp = true;   ///< load the MTP block when the geometry has one
    /// > 0: the experts are not loaded but streamed from the file through a GlmExpertCache of this many blobs in
    /// `experts`' home (the GgufModel must then outlive the loaded weights); filled before the first token when
    /// `cache_fill`, copying with `cache_threads` threads
    int64_t expert_cache_slots = 0;
    bool cache_fill = true;
    int cache_threads = 8;
};

/// What glm_plan found: the bytes each home needs, and the types the file uses per role (for a report).
struct GlmPlan {
    uint64_t device_bytes = 0;    ///< everything but the experts, as loaded (F32 vectors, BF16 k_b / v_b)
    uint64_t expert_bytes = 0;    ///< the expert blobs
    uint64_t file_bytes = 0;      ///< the payloads read
    bool mtp = false;             ///< the MTP block will be loaded
    int tensors = 0;
    /// "role: type" -> tensors, e.g. "projection: Q4_K" -> 300
    std::vector<std::pair<std::string, int>> types;
};

/// Checks `m` against `g` (read_geometry's, from the same file).  Returns "" or the first problem, naming the tensor.
std::string glm_plan(const GgufModel& m, const core::Glm5NextGeometry& g, const GlmLoadOptions& opt, GlmPlan& plan);

/// The weights on the device (and the experts where the options put them); weights() points into them.
class GlmLoadedWeights {
public:
    GlmLoadedWeights() = default;
    ~GlmLoadedWeights();
    GlmLoadedWeights(const GlmLoadedWeights&) = delete;
    GlmLoadedWeights& operator=(const GlmLoadedWeights&) = delete;

    /// Plans, allocates and copies.  On failure nothing stays allocated and `err` names the problem.
    bool load(const GgufModel& m, const core::Glm5NextGeometry& g, const GlmLoadOptions& opt, std::string& err);
    const GlmModelWeights& weights() const { return w_; }
    /// The expert cache when the options streamed the experts (pass it to GlmModel::init), else null.
    GlmExpertCache* expert_cache() const { return cache_.get(); }
    const GlmPlan& plan() const { return plan_; }

private:
    GlmModelWeights w_;
    GlmPlan plan_;
    std::vector<void*> device_, host_;
    std::unique_ptr<GlmExpertCache> cache_;
    void release();
};

}  // namespace strata::glm
