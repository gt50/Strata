// src/glm/glm_spec.cpp - see include/strata/glm/glm_spec.hpp.
#include "strata/glm/glm_spec.hpp"

#include <algorithm>

namespace strata::glm {

bool GlmGenerator::start(const int32_t* prompt, int64_t n, int n_draft, void* stream, std::string& err) {
    started_ = false;
    owed_ = false;
    st_ = GlmSpecStats{};
    if (!prompt || n < 1 || n >= m_.max_ctx()) {
        err = "GlmGenerator: a prompt of " + std::to_string(n) + " tokens for a context of " +
              std::to_string(m_.max_ctx());
        return false;
    }
    m_.reset(stream);
    if (!m_.prompt(prompt, n, 0, stream, err)) return false;
    seq_.assign(prompt, prompt + n);
    if (!pick_after(m_.logits(), t_, stream, err)) return false;
    pos_ = n;
    k_max_ = std::min(n_draft, m_.max_verify() - 1);
    spec_ = k_max_ >= 1 && m_.has_mtp();
    // the first draft: the MTP block over the prompt's last position, with t
    if (spec_ && !m_.mtp_decode(t_, pos_ - 1, m_.h_nextn(), stream, err)) return false;
    started_ = true;
    return true;
}

bool GlmGenerator::pick_after(const float* logits, int32_t& tok, void* stream, std::string& err) {
    return m_.pick(logits, 1, (int64_t) seq_.size() - 1, seq_.data(), (int64_t) seq_.size() - 1, &seq_.back(), &tok,
                   stream, err);
}

bool GlmGenerator::next(int64_t limit, std::vector<int32_t>& out, void* stream, std::string& err) {
    if (!started_ || limit < 1) {
        err = "GlmGenerator::next: no sequence started, or nothing asked for";
        return false;
    }
    if (!spec_) {   // a token at a time: the last one returned is decoded now, when the caller wants the one after
        if (owed_) {
            if (pos_ >= m_.max_ctx()) {
                err = "GlmGenerator::next: the context is full";
                return false;
            }
            seq_.push_back(t_);
            if (!m_.decode(t_, pos_, stream, err) || !pick_after(m_.logits(), t_, stream, err)) return false;
            ++pos_;
        }
        out.push_back(t_);
        owed_ = true;
        return true;
    }
    if (owed_) {   // the last round's commit, now that the caller wants more
        if (!m_.commit(keep_, next_, stream, err)) return false;
        pos_ += keep_;
        t_ = next_;
        owed_ = false;
    }
    const int k = (int) std::min<int64_t>({(int64_t) k_max_, limit - 1, m_.max_ctx() - pos_ - 1});
    if (k < 0) {
        err = "GlmGenerator::next: the context is full";
        return false;
    }
    window_.assign(1, t_);
    for (int j = 1; j <= k; ++j) {   // d_j from the MTP block's last output; then its chain
        // drawn as the window's row j - 1 will draw (Philox(seed, pos + j - 1), the same history): when the two
        // distributions are close the draws agree more often than independent ones would
        seq_.insert(seq_.end(), window_.begin(), window_.end() - 1);
        int32_t d = 0;
        const bool ok = m_.pick(m_.mtp_logits(), 1, pos_ + j - 1, seq_.data(), (int64_t) seq_.size(), &window_.back(),
                                &d, stream, err);
        seq_.resize((size_t) pos_);
        if (!ok) return false;
        window_.push_back(d);
        if (j < k && !m_.mtp_decode(d, pos_ + j - 1, m_.mtp_h(), stream, err)) return false;
    }
    picks_.resize((size_t) k + 1);
    if (!m_.verify(window_.data(), k + 1, pos_, picks_.data(), stream, err, seq_.data(), (int64_t) seq_.size()))
        return false;
    int a = 0;
    while (a < k && window_[(size_t) a + 1] == picks_[(size_t) a]) ++a;
    for (int i = 0; i <= a; ++i) {
        out.push_back(window_[(size_t) i]);
        seq_.push_back(window_[(size_t) i]);
    }
    ++st_.rounds;
    st_.drafted += k;
    st_.accepted += a;
    keep_ = 1 + a;
    next_ = picks_[(size_t) a];
    owed_ = true;
    return true;
}

bool glm_generate(GlmModel& model, const int32_t* prompt, int64_t n_prompt, int64_t n_gen, int n_draft,
                  std::vector<int32_t>& out, void* stream, std::string& err, GlmSpecStats* stats) {
    out.clear();
    if (!prompt || n_prompt < 1 || n_gen < 1 || n_prompt + n_gen > model.max_ctx()) {
        err = "glm_generate: " + std::to_string(n_prompt) + " prompt tokens and " + std::to_string(n_gen) +
              " more do not fit the context of " + std::to_string(model.max_ctx());
        return false;
    }
    GlmGenerator gen(model);
    if (!gen.start(prompt, n_prompt, n_draft, stream, err)) return false;
    while ((int64_t) out.size() < n_gen)
        if (!gen.next(n_gen - (int64_t) out.size(), out, stream, err)) return false;
    if (stats) *stats = gen.stats();
    return true;
}

}  // namespace strata::glm
