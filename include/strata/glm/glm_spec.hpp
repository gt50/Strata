// include/strata/glm/glm_spec.hpp - generation on a GlmModel, greedy or sampled (GlmModel::set_sampling), with or
// without speculative decoding: the MTP block drafts, a verify window checks the drafts against the main model, the
// accepted prefix is kept.
//
// One round, the next token t at position p (the main model has run everything before it):
//
//     d_1 .. d_k   the MTP block's chain: d_1 from the main model's h at p - 1 and t, d_j from its own h and d_{j-1},
//                  each drawn as the window row that judges it will draw (the same Philox(seed, position) and history)
//     g_0 .. g_k   GlmModel::verify([t, d_1 .. d_k], p): the pick after each - bitwise decode()'s logits, the same
//                  draw (the token after position q draws Philox(seed, q), whatever else ran)
//     a            the drafts accepted: d_i == g_{i-1} for i <= a
//     out         += t, d_1 .. d_a;  commit(1 + a, g_a);  the next round starts at p + 1 + a with t = g_a
//
// So the output is token for token what decoding a token at a time gives - greedy, or sampled with the same seed -
// whatever the drafts: a wrong draft costs only the work spent on it.  (A draft is kept only when the sample equals
// it, not with probability min(1, p / q): the output stays the plain run's exactly, at some acceptance.)
//
// Measured on the Radeon 8060S (glm_graph_parity --spec 45: GLM-5.3-Flash's shapes, 8 resident IQ4_XS experts a
// layer), a decode token 50.7 ms: a window of 2 tokens 55.4 ms, 4 67.5, 8 100.7; commit 1.9-4.0 ms.  The window's
// experts are grouped by expert, each read once for its tokens (glm_expert_groups_window) - with a group an entry it
// was 71.2 / 114.8 / 199.6.  That model's tokens all choose the same 8 experts, so this is the most the grouping can
// save: GLM-5.3-Flash's tokens choose 8 of 288, and share fewer.  Not yet captured as a graph, not with an expert
// provider's cache past what it holds at once, and the MTP block's acceptance on the real weights not yet measured
// (the synthetic model's random MTP block drafts nothing that holds).
#pragma once

#include "strata/glm/glm_model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::glm {

struct GlmSpecStats {
    int64_t rounds = 0;     ///< verify windows
    int64_t drafted = 0;    ///< drafts verified
    int64_t accepted = 0;   ///< drafts accepted
};

/// n_gen tokens after `prompt` (n_prompt tokens at positions 0 ..; the model reset first), picked as the model's
/// sampling says (greedy by default; the penalties count the prompt and every token since): `out` holds them.
/// n_draft 0, or a model without an MTP block: decode() a token at a time.  Otherwise rounds of up to n_draft drafts
/// (and at most max_verify() - 1).  The model needs init_verify() either way (its picks).
bool glm_generate(GlmModel& model, const int32_t* prompt, int64_t n_prompt, int64_t n_gen, int n_draft,
                  std::vector<int32_t>& out, void* stream, std::string& err, GlmSpecStats* stats = nullptr);

}  // namespace strata::glm
