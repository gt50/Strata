// include/strata/glm/glm_spec.hpp - greedy generation on a GlmModel, with or without speculative decoding: the MTP
// block drafts, a verify window checks the drafts against the main model, the accepted prefix is kept.
//
// One round, the next token t at position p (the main model has run everything before it):
//
//     d_1 .. d_k   the MTP block's chain: d_1 from the main model's h at p - 1 and t, d_j from its own h and d_{j-1}
//     g_0 .. g_k   GlmModel::verify([t, d_1 .. d_k], p): the greedy pick after each - bitwise what decode() gives
//     a            the drafts accepted: d_i == g_{i-1} for i <= a
//     out         += t, d_1 .. d_a;  commit(1 + a, g_a);  the next round starts at p + 1 + a with t = g_a
//
// So the output is token for token plain greedy decoding's, whatever the drafts: a wrong draft costs only the work
// spent on it.
//
// Measured on the Radeon 8060S (glm_graph_parity --spec 45: GLM-5.3-Flash's shapes, 8 resident IQ4_XS experts a
// layer), a decode token 50.7 ms: a window of 2 tokens 71.2 ms, 4 114.8, 8 199.6 (21 ms a token more: each token's
// experts are read for it alone - a group an entry, as decode groups them); commit 2.1-3.1 ms.  Not yet captured as a
// graph, not with an expert provider's cache past what it holds at once, and the MTP block's acceptance on the real
// weights not yet measured (the synthetic model's random MTP block drafts nothing that holds).
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

/// Greedy generation of n_gen tokens after `prompt` (n_prompt tokens at positions 0 ..; the model reset first): `out`
/// holds them.  n_draft 0, or a model without an MTP block: decode() a token at a time.  Otherwise rounds of up to
/// n_draft drafts (and at most max_verify() - 1).  The model needs init_verify() either way (its greedy picks).
bool glm_generate(GlmModel& model, const int32_t* prompt, int64_t n_prompt, int64_t n_gen, int n_draft,
                  std::vector<int32_t>& out, void* stream, std::string& err, GlmSpecStats* stats = nullptr);

}  // namespace strata::glm
