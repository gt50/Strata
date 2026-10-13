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
    // every token before the next pick, the last at position seq.size() - 1: the penalties' history
    std::vector<int32_t> seq(prompt, prompt + n_prompt);
    auto pick_after = [&](const float* logits, int32_t& tok) {   // the token after seq's last
        return model.pick(logits, 1, (int64_t) seq.size() - 1, seq.data(), (int64_t) seq.size() - 1, &seq.back(), &tok,
                          stream, err);
    };
    int32_t t = 0;
    if (!pick_after(model.logits(), t)) return false;
    int64_t pos = n_prompt;   // t's position
    const int k_max = std::min(n_draft, model.max_verify() - 1);
    if (k_max < 1 || !model.has_mtp()) {   // a token at a time
        for (;;) {
            out.push_back(t);
            if ((int64_t) out.size() == n_gen) return true;
            seq.push_back(t);
            if (!model.decode(t, pos, stream, err) || !pick_after(model.logits(), t)) return false;
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
            // drawn as the window's row j - 1 will draw (Philox(seed, pos + j - 1), the same history): when the two
            // distributions are close the draws agree more often than independent ones would
            seq.insert(seq.end(), window.begin(), window.end() - 1);
            int32_t d = 0;
            const bool ok = model.pick(model.mtp_logits(), 1, pos + j - 1, seq.data(), (int64_t) seq.size(),
                                       &window.back(), &d, stream, err);
            seq.resize((size_t) pos);
            if (!ok) return false;
            window.push_back(d);
            if (j < k && !model.mtp_decode(d, pos + j - 1, model.mtp_h(), stream, err)) return false;
        }
        if (!model.verify(window.data(), k + 1, pos, picks.data(), stream, err, seq.data(), (int64_t) seq.size()))
            return false;
        int a = 0;
        while (a < k && window[(size_t) a + 1] == picks[(size_t) a]) ++a;
        for (int i = 0; i <= a; ++i) {
            out.push_back(window[(size_t) i]);
            seq.push_back(window[(size_t) i]);
        }
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
