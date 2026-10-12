// include/strata/core/geometry_glm5_next.hpp - GLM-5.3-Flash's geometry (llama.cpp architecture "glm5-next"), read
// from its GGUF metadata by `strata::read_geometry` (gguf_reader.hpp).
//
// The engine has no forward pass for this architecture yet: this is what a file declares, read under the key names
// llama.cpp's converter writes (conversion/glm.py, Glm5NextModel) and its loader reads (src/models/glm5-next.cpp).
// GLM-5.3-Flash, from zai-org/GLM-5.3-Flash's config.json: 45 layers, 34 KDA linear-attention layers and 11
// MLA layers with a DeepSeek-style sparse indexer (3, 7, ... 43), a 4-stream mHC residual, 288 routed experts
// (8 per token, sigmoid routing, x2.5) plus one shared expert, the first 3 layers dense, and one MTP layer.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::core {

struct Glm5NextGeometry {
    int64_t n_layers = 0;             ///< main layers: block_count less the MTP layers (45)
    int64_t n_nextn = 0;              ///< nextn_predict_layers: MTP layers after the main ones (1)
    int64_t n_embd = 0;               ///< embedding_length (4096)
    int64_t n_vocab = 0;              ///< vocab_size, else the length of tokenizer.ggml.tokens (154880)
    int64_t n_ctx_train = 0;          ///< context_length (1048576)
    double rms_eps = 0;               ///< attention.layer_norm_rms_epsilon
    /// attention.layer_norm_epsilon (1e-6): the indexer key's LayerNorm.  Optional as in llama.cpp, whose value-
    /// initialized f_norm_eps is then 0
    double norm_eps = 0;

    /// Per block, MTP layers included: 1 = MLA attention, 0 = KDA linear attention (attention.head_count_kv, an
    /// array: 0 marks a linear layer, as llama.cpp's is_recr).
    std::vector<uint8_t> is_mla;
    int64_t n_head = 0;               ///< attention.head_count, of both the MLA and the KDA layers (64)

    // KDA (Kimi delta attention)
    int64_t kda_head_dim = 0;         ///< kda.head_dim (128)
    int64_t ssm_d_conv = 0;           ///< ssm.conv_kernel (4)
    bool has_kda_gate_lower_bound = false;
    double kda_gate_lower_bound = 0;  ///< kda.gate_lower_bound (-5)

    // MLA, absorbed: the cache holds the kv_lora_rank-wide latent (plus n_rot rotated dims; GLM has none)
    int64_t q_lora_rank = 0;          ///< attention.q_lora_rank (1536)
    int64_t kv_lora_rank = 0;         ///< attention.kv_lora_rank (512)
    int64_t n_rot = 0;                ///< rope.dimension_count (0: no RoPE at all)
    int64_t head_dim_k = 0;           ///< attention.key_length_mla: per-head q.k width, nope + rope (256)
    int64_t head_dim_v = 0;           ///< attention.value_length_mla (256)

    // the sparse indexer, keys pooled kpool cells at a time
    int64_t idx_heads = 0;            ///< attention.indexer.head_count (32)
    int64_t idx_key_dim = 0;          ///< attention.indexer.key_length (128)
    int64_t idx_top_k = 0;            ///< attention.indexer.top_k (2048)
    int64_t idx_kpool = 0;            ///< attention.indexer.kpool (4)
    bool idx_kpool_select_tail = true;  ///< attention.indexer.kpool_select_tail
    std::vector<uint8_t> idx_full;    ///< attention.indexer.types per main layer (1 = full); absent = all full

    // mHC
    int64_t hc = 0;                   ///< hyper_connection.count (4)
    int64_t hc_sinkhorn_iters = 0;    ///< hyper_connection.sinkhorn_iterations (20)
    double hc_eps = 0;                ///< hyper_connection.epsilon (1e-6)

    // MoE
    int64_t n_dense_lead = 0;         ///< leading_dense_block_count: dense FFN layers before the MoE ones (3)
    int64_t n_ff_dense = 0;           ///< feed_forward_length: the dense layers' width (12288)
    int64_t n_expert = 0;             ///< expert_count (288)
    int64_t n_expert_used = 0;        ///< expert_used_count (8)
    int64_t n_expert_shared = 0;      ///< expert_shared_count (1)
    int64_t n_ff_exp = 0;             ///< expert_feed_forward_length (2048)
    /// expert_shared_feed_forward_length (2048; Unsloth's files carry it, llama.cpp's converter does not write it):
    /// the shared expert's width, else n_ff_exp x n_expert_shared as DeepSeek-style shared experts are sized
    int64_t n_ff_shexp = 0;
    int64_t expert_gating = 2;        ///< expert_gating_func: 1 softmax, 2 sigmoid (llama.cpp's default here), 4 sqrtsoftplus
    double expert_weights_scale = 1;  ///< expert_weights_scale (2.5)
    bool expert_weights_norm = false; ///< expert_weights_norm (true)
    double swiglu_clamp_exp = 0;      ///< swiglu_clamp_exp, one value for every layer (10); 0 = no clamp
    double swiglu_clamp_shexp = 0;    ///< swiglu_clamp_shexp, else the routed experts' clamp

    int64_t n_blocks() const { return n_layers + n_nextn; }
    int64_t n_mla_layers() const {
        int64_t n = 0;
        for (int64_t l = 0; l < n_layers; ++l) n += is_mla[(size_t) l];
        return n;
    }
    int64_t n_kda_layers() const { return n_layers - n_mla_layers(); }
};

}  // namespace strata::core
