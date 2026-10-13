// src/glm/glm_spec_parity.cpp - speculative decoding on glm_model_parity's synthetic model (5 layers + MTP, n_embd 1024,
// a vocabulary of 2048): GlmModel::verify / commit against decode(), and glm_generate against plain greedy decoding.
//
//   1. a window of 6 tokens after a 12-token prompt: every row's logits bitwise decode()'s for the same tokens; after
//      commit of all 6, decoding on: bitwise
//   2. a window whose drafts are wrong (random tokens), 2 kept: decoding on from there bitwise as if the window had
//      never run (its caches at the dropped positions are rewritten before anything reads them; KDA's state and
//      history are the kept tokens' alone)
//   3. glm_generate with the MTP block's drafts, 3 and 7 a round: the same tokens as plain greedy (random weights: the
//      drafts are mostly wrong - the rejection path)
//   4. oracle drafts (the greedy tokens themselves, every third round one corrupted): the same tokens, most drafts
//      accepted - the acceptance path
#include "glm_ref.hpp"

#include "strata/glm/glm_spec.hpp"

namespace {

constexpr int64_t MAX_CTX = 96;
int g_bad = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "*** FAIL ***");
    if (!ok) ++g_bad;
}

vector<uint8_t> raw(const void* dev, size_t n) {
    vector<uint8_t> v(n);
    check(cudaMemcpy(v.data(), dev, n, cudaMemcpyDeviceToHost), "download");
    return v;
}

void die(const std::string& err) {
    std::fprintf(stderr, "%s\n", err.c_str());
    std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: glm_spec_parity [--selftest]\n"); return 2; }
    }
    TestModel tm = build_test_model(MAX_CTX);
    const Glm5NextGeometry& g = tm.g;
    glm::GlmModel a, b;
    std::string err;
    if (!a.init(g, tm.w, MAX_CTX, err, nullptr, 16) || !b.init(g, tm.w, MAX_CTX, err, nullptr, 16) ||
        !a.init_verify(8, err) || !b.init_verify(8, err))
        die(err);
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    const int V = (int) g.n_vocab;
    const size_t VB = (size_t) V * 4;
    auto rnd = [&]() { return (int32_t) (unit() * (float) V) % V; };
    vector<int32_t> prompt(12), w(12);
    for (auto& t : prompt) t = rnd();
    for (auto& t : w) t = rnd();
    std::printf("glm_spec: verify windows and commit against decode (5 layers + MTP, n_embd 1024)\n");

    // 0. a window's experts grouped by expert: 3 tokens of 2 choices, experts 5 and 2, 2 and 7, 5 and 2
    {
        const vector<int32_t> ids = {5, 2, 2, 7, 5, 2};
        vector<unsigned long long> table(8);
        for (int e = 0; e < 8; ++e) table[(size_t) e] = 1000 + (unsigned long long) e;
        int32_t* d_ids = upload(ids);
        unsigned long long* d_table = upload(table);
        unsigned long long* d_ptr = nullptr;
        int32_t *d_start = nullptr, *d_ng = nullptr, *d_dst = nullptr, *d_tok = nullptr;
        check(cudaMalloc(&d_ptr, 6 * 8), "ptr");
        check(cudaMalloc(&d_start, 7 * 4), "start");
        check(cudaMalloc(&d_ng, 4), "ng");
        check(cudaMalloc(&d_dst, 6 * 4), "dst");
        check(cudaMalloc(&d_tok, 6 * 4), "tok");
        k::glm_expert_groups_window(d_ids, d_table, 2, 3, d_ptr, d_start, d_ng, d_dst, d_tok, s);
        check(cudaStreamSynchronize(s), "groups");
        int32_t ng = 0;
        check(cudaMemcpy(&ng, d_ng, 4, cudaMemcpyDeviceToHost), "ng");
        vector<unsigned long long> ptr(3);
        vector<int32_t> start(4), dst(6), tok(6);
        check(cudaMemcpy(ptr.data(), d_ptr, 3 * 8, cudaMemcpyDeviceToHost), "ptr");
        check(cudaMemcpy(start.data(), d_start, 4 * 4, cudaMemcpyDeviceToHost), "start");
        check(cudaMemcpy(dst.data(), d_dst, 6 * 4, cudaMemcpyDeviceToHost), "dst");
        check(cudaMemcpy(tok.data(), d_tok, 6 * 4, cudaMemcpyDeviceToHost), "tok");
        // groups 5 {entries 0, 4}, 2 {1, 2, 5}, 7 {3}
        const bool ok = ng == 3 && ptr == vector<unsigned long long>{1005, 1002, 1007} &&
                        start == vector<int32_t>{0, 2, 5, 6} && dst == vector<int32_t>{0, 4, 1, 2, 5, 3} &&
                        tok == vector<int32_t>{0, 2, 0, 1, 2, 1};
        expect(ok, "0. a window's 6 expert choices grouped by expert: 3 groups, entries in token order");
        for (void* p : {(void*) d_ids, (void*) d_table, (void*) d_ptr, (void*) d_start, (void*) d_ng, (void*) d_dst,
                        (void*) d_tok})
            cudaFree(p);
    }

    // the plain run: the prompt, then w[0..12) decoded at 12 .. 23, every token's logits kept
    a.reset(s);
    if (!a.prompt(prompt.data(), 12, 0, s, err)) die(err);
    vector<vector<uint8_t>> ref;
    for (int i = 0; i < 12; ++i) {
        if (!a.decode(w[(size_t) i], 12 + i, s, err)) die(err);
        check(cudaStreamSynchronize(s), "decode");
        ref.push_back(raw(a.logits(), VB));
    }

    // 1. a window of w[0..6), every row; commit all, decode on
    {
        b.reset(s);
        if (!b.prompt(prompt.data(), 12, 0, s, err)) die(err);
        int32_t picks[8];
        if (!b.verify(w.data(), 6, 12, picks, s, err)) die(err);
        bool same = true;
        for (int t = 0; t < 6; ++t) same &= raw(b.window_logits(t), VB) == ref[(size_t) t];
        expect(same, "1. a window of 6 tokens: every row's logits bitwise decode()'s");
        int32_t want[6];
        bool picks_ok = true;
        for (int t = 0; t < 6; ++t) {
            if (!a.argmax(b.window_logits(t), 1, &want[t], s, err)) die(err);
            picks_ok &= want[t] == picks[t];
        }
        expect(picks_ok, "   its greedy picks are its rows' argmax");
        if (!b.commit(6, w[6], s, err)) die(err);
        same = raw(b.logits(), VB) == ref[5];
        for (int i = 6; i < 12; ++i) {
            if (!b.decode(w[(size_t) i], 12 + i, s, err)) die(err);
            check(cudaStreamSynchronize(s), "decode");
            same &= raw(b.logits(), VB) == ref[(size_t) i];
        }
        expect(same, "   commit of all 6, then 6 decodes: logits bitwise the plain run's, every token");
    }

    // 2. wrong drafts, 2 kept
    {
        b.reset(s);
        if (!b.prompt(prompt.data(), 12, 0, s, err)) die(err);
        vector<int32_t> win = {w[0], w[1], rnd(), rnd(), rnd(), rnd()};
        int32_t picks[8];
        if (!b.verify(win.data(), 6, 12, picks, s, err)) die(err);
        bool same = raw(b.window_logits(0), VB) == ref[0] && raw(b.window_logits(1), VB) == ref[1];
        expect(same, "2. a window of 2 kept tokens and 4 wrong drafts: the kept rows bitwise decode()'s");
        if (!b.commit(2, w[2], s, err)) die(err);
        same = true;
        for (int i = 2; i < 12; ++i) {
            if (!b.decode(w[(size_t) i], 12 + i, s, err)) die(err);
            check(cudaStreamSynchronize(s), "decode");
            same &= raw(b.logits(), VB) == ref[(size_t) i];
        }
        expect(same, "   commit of 2, then decoding over the dropped positions: bitwise the plain run's, every token");
    }

    // 3. glm_generate: plain, and with the MTP block's drafts
    vector<int32_t> greedy;
    const int64_t N = 40;
    if (!glm::glm_generate(a, prompt.data(), 12, N, 0, greedy, s, err)) die(err);
    for (const int k : {3, 7}) {
        vector<int32_t> got;
        glm::GlmSpecStats st;
        if (!glm::glm_generate(b, prompt.data(), 12, N, k, got, s, err, &st)) die(err);
        char what[160];
        std::snprintf(what, sizeof what,
                      "3. glm_generate, MTP drafts %d a round: the %lld tokens of plain greedy (%lld rounds, %lld of "
                      "%lld drafts accepted)",
                      k, (long long) N, (long long) st.rounds, (long long) st.accepted, (long long) st.drafted);
        expect(got == greedy, what);
    }

    // 4. oracle drafts through verify / commit: the greedy tokens, every third round one of them corrupted
    {
        b.reset(s);
        if (!b.prompt(prompt.data(), 12, 0, s, err)) die(err);
        vector<int32_t> got;
        int64_t pos = 12, drafted = 0, accepted = 0;
        int32_t t = greedy[0];
        for (int round = 0; (int64_t) got.size() < N; ++round) {
            const int64_t i = (int64_t) got.size();   // t is greedy[i]
            const int k = (int) std::min<int64_t>(5, N - i - 1);
            vector<int32_t> win = {t};
            for (int j = 1; j <= k; ++j) win.push_back(greedy[(size_t) (i + j)]);
            if (round % 3 == 2 && k >= 2) win[2] = (win[2] + 1) % V;   // the second draft wrong
            int32_t picks[8];
            if (!b.verify(win.data(), k + 1, pos, picks, s, err)) die(err);
            int acc = 0;
            while (acc < k && win[(size_t) acc + 1] == picks[acc]) ++acc;
            for (int j = 0; j <= acc; ++j) got.push_back(win[(size_t) j]);
            drafted += k;
            accepted += acc;
            if (!b.commit(1 + acc, picks[acc], s, err)) die(err);
            pos += 1 + acc;
            t = picks[acc];
        }
        got.resize((size_t) N);
        char what[160];
        std::snprintf(what, sizeof what, "4. oracle drafts (every third round one wrong): plain greedy's tokens (%lld of "
                      "%lld drafts accepted)", (long long) accepted, (long long) drafted);
        expect(got == greedy && accepted * 2 > drafted, what);
    }

    // 5. sampled: plain and speculative with the same seed give the same tokens (and not greedy's)
    for (const bool pen : {false, true}) {
        k::SamplerParams sp;
        sp.temperature = 0.9f;
        sp.top_k = 40;
        sp.top_p = 0.9f;
        sp.seed = 1234;
        if (pen) {
            sp.penalty_last_n = 16;
            sp.penalty_repeat = 1.3f;
            sp.penalty_present = 0.2f;
        }
        a.set_sampling(sp);
        b.set_sampling(sp);
        vector<int32_t> plain, spec;
        glm::GlmSpecStats st;
        if (!glm::glm_generate(a, prompt.data(), 12, N, 0, plain, s, err) ||
            !glm::glm_generate(b, prompt.data(), 12, N, 5, spec, s, err, &st))
            die(err);
        char what[200];
        std::snprintf(what, sizeof what,
                      "5. sampled (T 0.9, top_k 40, top_p 0.9%s): MTP drafts 5 a round give plain sampling's %lld "
                      "tokens (%lld of %lld drafts kept)",
                      pen ? ", penalties over 16" : "", (long long) N, (long long) st.accepted, (long long) st.drafted);
        expect(spec == plain && plain != greedy, what);

        // 6. oracle drafts under the same sampling: the plain run's tokens as drafts, every third round one wrong
        b.reset(s);
        if (!b.prompt(prompt.data(), 12, 0, s, err)) die(err);
        vector<int32_t> seq(prompt.begin(), prompt.end()), got;
        int64_t pos = 12, drafted = 0, accepted = 0;
        int32_t t = plain[0];
        for (int round = 0; (int64_t) got.size() < N; ++round) {
            const int64_t i = (int64_t) got.size();
            const int k = (int) std::min<int64_t>(5, N - i - 1);
            vector<int32_t> win = {t};
            for (int j = 1; j <= k; ++j) win.push_back(plain[(size_t) (i + j)]);
            if (round % 3 == 2 && k >= 2) win[2] = (win[2] + 1) % V;
            int32_t picks[8];
            if (!b.verify(win.data(), k + 1, pos, picks, s, err, seq.data(), (int64_t) seq.size())) die(err);
            int acc = 0;
            while (acc < k && win[(size_t) acc + 1] == picks[acc]) ++acc;
            for (int j = 0; j <= acc; ++j) {
                got.push_back(win[(size_t) j]);
                seq.push_back(win[(size_t) j]);
            }
            drafted += k;
            accepted += acc;
            if (!b.commit(1 + acc, picks[acc], s, err)) die(err);
            pos += 1 + acc;
            t = picks[acc];
        }
        got.resize((size_t) N);
        std::snprintf(what, sizeof what, "6. oracle drafts under it (every third round one wrong): plain sampling's "
                      "tokens (%lld of %lld drafts kept)", (long long) accepted, (long long) drafted);
        expect(got == plain && accepted * 2 > drafted, what);
    }

    std::printf("\nglm_spec: %d failures\n", g_bad);
    cudaStreamDestroy(s);
    for (void* p : g_free) cudaFree(p);
    if (g_bad) return 1;
    if (selftest) std::printf("glm_spec_parity OK\n");
    return 0;
}
