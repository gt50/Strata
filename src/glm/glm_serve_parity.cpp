// src/glm/glm_serve_parity.cpp - GlmServer (the `strata --serve` protocol for a GLM GGUF) on glm_model_parity's
// synthetic model written as a glm5-next GGUF, driven in-process with scripted command lines, against glm_generate on
// the same weights built directly:
//
//   1. INFO and READY first; GEN with no sampling keys: the T lines are greedy decoding's tokens, DONE says
//      length and the counts - with 3 drafts a round, and a token at a time
//   2. GEN with temperature, top_k, top_p and seed: the tokens of glm_generate with that sampling
//   3. an end token (--eos-ids): the request ends at its first T line, DONE says stop
//   4. STOP while a request runs: it ends after the tokens already produced, DONE says cancel
//   5. bad commands get ERR and the engine goes on; QUIT ends it with 0
//
// `glm_serve_parity --write-gguf PATH`: writes the synthetic model's GGUF there and prints a prompt's GEN line, for
// running the engine itself on it (strata --serve --native PATH --max-context 96 --spec 3).
#include "glm_test_gguf.hpp"

#include "strata/glm/glm_serve.hpp"
#include "strata/glm/glm_spec.hpp"

#include <sstream>

namespace {

constexpr int64_t MAX_CTX = 96;
int g_bad = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "*** FAIL ***");
    if (!ok) ++g_bad;
}

void die(const std::string& err) {
    std::fprintf(stderr, "%s\n", err.c_str());
    std::exit(1);
}

std::string ids_text(const vector<int32_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}

/// What a script of command lines got back: the lines, and run()'s result.
struct Session {
    vector<std::string> out;
    int rc = -1;
    /// The T lines' tokens of the n-th request (0-based), and its DONE line's fields.
    vector<int32_t> tokens(int n) const {
        vector<int32_t> t;
        int req = 0;
        for (const auto& l : out) {
            if (l.rfind("DONE ", 0) == 0) { if (req++ == n) break; continue; }
            if (req == n && l.rfind("T ", 0) == 0) t.push_back(std::atoi(l.c_str() + 2));
        }
        return t;
    }
    vector<std::string> done(int n) const {
        int req = 0;
        for (const auto& l : out)
            if (l.rfind("DONE ", 0) == 0 && req++ == n) {
                std::istringstream is(l);
                vector<std::string> f;
                for (std::string w; is >> w;) f.push_back(w);
                return f;
            }
        return {};
    }
    int count(const char* prefix) const {
        int c = 0;
        for (const auto& l : out) c += l.rfind(prefix, 0) == 0 ? 1 : 0;
        return c;
    }
};

/// Runs `lines` through `srv`; `on_token` is called at every T line (it may raise the stop flag).
Session drive(glm::GlmServer& srv, const vector<std::string>& lines, std::atomic<bool>& stop,
              const std::function<void(int)>& on_token = {}) {
    Session s;
    size_t i = 0;
    int t = 0;
    s.rc = srv.run([&](std::string& l) {
                       if (i >= lines.size()) return false;
                       l = lines[i++];
                       return true;
                   },
                   stop,
                   [&](const std::string& l) {
                       s.out.push_back(l);
                       if (on_token && l.rfind("T ", 0) == 0) on_token(t++);
                   });
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    std::string write_to;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--write-gguf" && i + 1 < argc) write_to = argv[++i];
        else { std::fprintf(stderr, "usage: glm_serve_parity [--selftest] | --write-gguf PATH\n"); return 2; }
    }
    TestModel tm = build_test_model(MAX_CTX);
    if (!write_to.empty()) {   // the engine itself on the model: the file, and a prompt's GEN line
        write_model_gguf(write_to, tm);
        vector<int32_t> p(12);
        for (auto& t : p) t = (int32_t) (unit() * (float) tm.g.n_vocab) % (int32_t) tm.g.n_vocab;
        std::printf("GEN 24 %s\n", ids_text(p).c_str());
        return 0;
    }
    const Glm5NextGeometry& g = tm.g;
    TempDir dir;
    const fs::path path = dir.path / "glm-test.gguf";
    write_model_gguf(path, tm);
    std::string err;
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");

    // the reference: the same weights built directly
    glm::GlmModel ref;
    if (!ref.init(g, tm.w, MAX_CTX, err, nullptr, 16) || !ref.init_verify(8, err)) die(err);
    const int V = (int) g.n_vocab;
    vector<int32_t> prompt(12);
    for (auto& t : prompt) t = (int32_t) (unit() * (float) V) % V;
    const int64_t N = 24;
    vector<int32_t> greedy, sampled;
    if (!glm::glm_generate(ref, prompt.data(), 12, N, 0, greedy, s, err)) die(err);
    k::SamplerParams sp;
    sp.temperature = 0.9f;
    sp.top_k = 40;
    sp.top_p = 0.9f;
    sp.seed = 1234;
    ref.set_sampling(sp);
    if (!glm::glm_generate(ref, prompt.data(), 12, N, 0, sampled, s, err)) die(err);
    const std::string P = ids_text(prompt);
    std::printf("glm_serve: the serve protocol on a GLM GGUF (5 layers + MTP, n_embd 1024), in-process\n");

    auto logger = [](const std::string& l) { std::fprintf(stderr, "%s\n", l.c_str()); };
    for (const int spec : {3, 0}) {
        glm::GlmServeOptions o;
        o.model = path.string();
        o.max_ctx = MAX_CTX;
        o.n_draft = spec;
        o.max_chunk = 16;
        glm::GlmServer srv;
        if (!srv.load(o, err, logger)) die(err);
        std::atomic<bool> stop{false};
        const Session ss = drive(srv, {"GEN 24 " + P, "GEN 24 temperature=0.9 top_k=40 top_p=0.9 seed=1234 " + P}, stop);
        const bool head = ss.out.size() >= 2 && ss.out[0].rfind("INFO engine=", 0) == 0 &&
                          ss.out[0].find("arch=glm5-next") != std::string::npos &&
                          ss.out[1] == "READY " + std::to_string(MAX_CTX) + " stop";
        char what[160];
        std::snprintf(what, sizeof what, "1. %s: INFO, then READY %lld stop", spec ? "--spec 3" : "no drafts",
                      (long long) MAX_CTX);
        expect(head, what);
        const vector<std::string> d0 = ss.done(0);
        std::snprintf(what, sizeof what, "   GEN 24, no sampling keys: greedy decoding's 24 tokens, DONE 24 12 ... length");
        expect(ss.tokens(0) == greedy && d0.size() >= 8 && d0[1] == "24" && d0[2] == "12" && d0[5] == "length", what);
        std::snprintf(what, sizeof what, "2. GEN 24 temperature=0.9 top_k=40 top_p=0.9 seed=1234: the same sampled tokens");
        expect(ss.tokens(1) == sampled && sampled != greedy, what);
        expect(ss.rc == 0 && ss.count("ERR") == 0, "   the input's end: run() returns 0, no ERR");
    }

    // 3. an end token: the 6th greedy token (its first place in the output)
    {
        glm::GlmServeOptions o;
        o.model = path.string();
        o.max_ctx = MAX_CTX;
        o.n_draft = 3;
        o.max_chunk = 16;
        o.eos_ids = {greedy[5]};
        glm::GlmServer srv;
        if (!srv.load(o, err, logger)) die(err);
        std::atomic<bool> stop{false};
        const Session ss = drive(srv, {"GEN 24 " + P}, stop);
        const size_t at = (size_t) (std::find(greedy.begin(), greedy.end(), greedy[5]) - greedy.begin());
        const vector<int32_t> want(greedy.begin(), greedy.begin() + (ptrdiff_t) at + 1);
        const vector<std::string> d = ss.done(0);
        expect(ss.tokens(0) == want && d.size() >= 6 && d[5] == "stop",
               "3. --eos-ids: the request ends at the end token's T line, DONE says stop");

        // 4. STOP while it runs: raised at the first T line
        const Session sc = drive(srv, {"GEN 24 " + P}, stop, [&](int t) { if (t == 0) stop.store(true); });
        const vector<int32_t> got = sc.tokens(0);
        const vector<std::string> dc = sc.done(0);
        const bool prefix = !got.empty() && got.size() < 24 &&
                            std::equal(got.begin(), got.end(), greedy.begin());
        expect(prefix && dc.size() >= 6 && dc[5] == "cancel",
               "4. STOP during a request: it ends after the tokens produced (a prefix of greedy's), DONE says cancel");

        // 5. bad commands, then QUIT
        const Session sb = drive(srv, {"GEN x", "BGEN 0 4 1,2", "GEN 4 " + std::to_string(V + 5), "GENI 4 f 1,2",
                                       "GEN 4 " + P, "QUIT", "GEN 4 " + P},
                                 stop);
        expect(sb.count("ERR") == 4 && sb.count("DONE") == 1 && sb.rc == 0,
               "5. 4 bad commands get ERR, a good one runs, QUIT ends run() with 0 (the line after it unread)");
    }

    std::printf("\nglm_serve: %d failures\n", g_bad);
    cudaStreamDestroy(s);
    for (void* p : g_free) cudaFree(p);
    if (g_bad) return 1;
    if (selftest) std::printf("glm_serve_parity OK\n");
    return 0;
}
