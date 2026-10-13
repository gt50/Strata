// src/glm/glm_spec.cpp - see include/strata/glm/glm_spec.hpp.
#include "strata/glm/glm_spec.hpp"

#include <algorithm>

namespace strata::glm {

bool glm_generate(GlmModel& model, const int32_t* prompt, int64_t n_prompt, int64_t n_gen, int n_draft,
                  std::vector<int32_t>& out, void* stream, std::string& err, GlmSpecStats* stats) {
    out.clear();
    if (!prompt || n_prompt < 1 || n_gen < 1 || n_prompt + n_gen > model.max_ctx()) {
        err = "glm_generate: " + std::to_string(n_prompt) + " prompt tokens and " + std::to_string(n_gen) +
              " more do not fit the context of " + std::to_string(model.max_ctx());
        return false;
    }
    model.reset(stream);
    if (!model.prompt(prompt, n_prompt, 0, stream, err)) return false;
    int32_t t = 0;
    if (!model.argmax(model.logits(), 1, &t, stream, err)) return false;
    int64_t pos = n_prompt;   // t's position
    const int k_max = std::min(n_draft, model.max_verify() - 1);
    if (k_max < 1 || !model.has_mtp()) {   // a token at a time
        for (;;) {
            out.push_back(t);
            if ((int64_t) out.size() == n_gen) return true;
            if (!model.decode(t, pos, stream, err) || !model.argmax(model.logits(), 1, &t, stream, err)) return false;
            ++pos;
        }
    }
    // the first draft: the MTP block over the prompt's last position, with t
    if (!model.mtp_decode(t, pos - 1, model.h_nextn(), stream, err)) return false;
    std::vector<int32_t> window, picks((size_t) k_max + 1);
    for (;;) {
        const int64_t left = n_gen - (int64_t) out.size();   // t among them
        const int k = (int) std::min<int64_t>(k_max, left - 1);
        window.assign(1, t);
        for (int j = 1; j <= k; ++j) {   // d_j from the MTP block's last output; then its chain
            int32_t d = 0;
            if (!model.argmax(model.mtp_logits(), 1, &d, stream, err)) return false;
            window.push_back(d);
            if (j < k && !model.mtp_decode(d, pos + j - 1, model.mtp_h(), stream, err)) return false;
        }
        if (!model.verify(window.data(), k + 1, pos, picks.data(), stream, err)) return false;
        int a = 0;
        while (a < k && window[(size_t) a + 1] == picks[(size_t) a]) ++a;
        for (int i = 0; i <= a; ++i) out.push_back(window[(size_t) i]);
        if (stats) {
            ++stats->rounds;
            stats->drafted += k;
            stats->accepted += a;
        }
        if ((int64_t) out.size() >= n_gen) {
            out.resize((size_t) n_gen);
            return true;
        }
        const int32_t next = picks[(size_t) a];
        if (!model.commit(1 + a, next, stream, err)) return false;
        pos += 1 + a;
        t = next;
    }
}

}  // namespace strata::glm
