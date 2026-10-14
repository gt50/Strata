// src/glm/glm_loader_parity.cpp - the GGUF loader (glm_loader.hpp) on glm_model_parity's synthetic model written as a
// glm5-next GGUF: the tensors under llama.cpp's names, in its converter's shapes and layouts (the experts stacked
// three tensors a layer, attn_k_b / attn_v_b split per head, the conv1d weights {d_conv, 1, d_inner, 1}), with the
// metadata read_geometry reads.
//
//   1. the geometry read back is the one the model was built with
//   2. loaded with the experts in device memory, and again in mapped host memory, GlmModel gives bitwise the logits,
//      h_nextn and MTP logits of the model built directly from the same weights, for every token; and the same with
//      the experts streamed through GlmExpertCache (24 slots, 4 slots - every lookup a miss - and 40 filled ahead),
//      and a cache too small for one layer's experts refused
//   3. the conversions, on a second file: an F16 and a BF16 vector to F32, a Q8_0 attn_k_b and an F32 attn_v_b to
//      BF16, a conv1d of three dimensions - each loaded value the one expected, bitwise
//   4. what glm_plan refuses, naming the tensor: one missing, a shape the geometry does not give, a type no kernel
//      takes, experts whose gate and up types differ, a missing MTP block (and the same file loads without it)
//
// `glm_loader_parity --plan SHARD`: the plan of a real model (any shard of it) - the types per role and the bytes.
// `glm_loader_parity --bench`: the expert cache's costs at GLM-5.3-Flash's expert size (see bench()).
#include "glm_ref.hpp"

#include "glm_test_gguf.hpp"
#include "strata/glm/glm_loader.hpp"
#include "strata/glm/glm_serve.hpp"

#include <chrono>
#include <filesystem>

namespace {

constexpr int64_t MAX_CTX = 64;
constexpr int STEPS = 6;
int g_bad = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "*** FAIL ***");
    if (!ok) ++g_bad;
}

fixture::Tensor* find_t(std::vector<fixture::Tensor>& ts, const std::string& name) {
    for (auto& t : ts)
        if (t.name == name) return &t;
    std::fprintf(stderr, "no tensor %s\n", name.c_str());
    std::exit(1);
}

/// Writes `ts` and runs glm_plan on it: "" or its error.
std::string plan_of(const fs::path& path, const Glm5NextGeometry& g, const std::vector<fixture::Tensor>& ts,
                    const glm::GlmLoadOptions& opt, glm::GlmPlan* out = nullptr) {
    fixture::write(path, model_keys(g), ts);
    const strata::GgufModel m({path.string()});
    glm::GlmPlan plan;
    const std::string e = glm::glm_plan(m, g, opt, plan);
    if (out) *out = plan;
    return e;
}

void print_plan(const glm::GlmPlan& p) {
    std::printf("  %d tensors, %.1f MB read; %.1f MB of weights and %.1f MB of experts as loaded%s\n", p.tensors,
                (double) p.file_bytes / 1e6, (double) p.device_bytes / 1e6, (double) p.expert_bytes / 1e6,
                p.mtp ? ", the MTP block included" : "");
    for (const auto& [what, n] : p.types) std::printf("    %-40s %d\n", what.c_str(), n);
}

int plan_real(const char* shard) {
    try {
        const strata::GgufModel m = strata::GgufModel::open(shard);
        Glm5NextGeometry g;
        if (const std::string e = strata::read_geometry(m.meta(), g); !e.empty()) {
            std::fprintf(stderr, "geometry: %s\n", e.c_str());
            return 1;
        }
        glm::GlmPlan plan;
        const std::string e = glm::glm_plan(m, g, glm::GlmLoadOptions{}, plan);
        std::printf("%s: %zu shards, %d layers + %d MTP\n", shard, m.size(), (int) g.n_layers, (int) g.n_nextn);
        print_plan(plan);
        if (!e.empty()) {
            std::printf("refused: %s\n", e.c_str());
            return 1;
        }
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}

/// `--bench-file SHARD [slots]`: the expert cache's reads on a real model's file, the cache empty at start (its own
/// slots only; the weights loaded as for serving).  Rounds of 64 different experts of one layer (resident(), a prompt's
/// pattern), then decode's pattern: 8 experts a MoE layer, layer after layer (experts(), the ids on the device), for
/// 3 tokens of random choices.  STRATA_GLM_MAPPED_READS=1 times the mapped path instead.
int bench_file(const char* shard, int64_t slots) {
    try {
        const strata::GgufModel m = strata::GgufModel::open(shard);
        Glm5NextGeometry g;
        if (const std::string e = strata::read_geometry(m.meta(), g); !e.empty()) {
            std::fprintf(stderr, "geometry: %s\n", e.c_str());
            return 1;
        }
        glm::GlmLoadOptions opt;
        opt.expert_cache_slots = slots;
        opt.cache_fill = false;
        glm::GlmLoadedWeights lw;
        std::string err;
        if (!lw.load(m, g, opt, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        glm::GlmExpertCache& c = *lw.expert_cache();
        std::printf("%s: %lld slots of %.1f MB, %s reads\n", shard, (long long) c.slots(), c.slot_bytes() / 1e6,
                    c.direct() ? "unbuffered" : "mapped");
        cudaStream_t s;
        check(cudaStreamCreate(&s), "stream");
        std::mt19937 rng(7);
        const int64_t l0 = g.n_dense_lead, nl = g.n_layers - g.n_dense_lead;
        std::vector<int32_t> all((size_t) g.n_expert);
        for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) all[(size_t) e] = e;
        for (int round = 0; round < 6; ++round) {   // a prompt round: 64 different experts of one layer
            const int64_t il = l0 + (int64_t) (rng() % (uint32_t) nl);
            std::shuffle(all.begin(), all.end(), rng);
            const auto st0 = c.stats();
            const auto t0 = std::chrono::steady_clock::now();
            if (!c.resident(il, all.data(), 64, s, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
            check(cudaStreamSynchronize(s), "sync");
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const auto st1 = c.stats();
            std::printf("  64 experts of layer %2lld: %llu misses, %.2f GB in %6.0f ms = %.2f GB/s\n", (long long) il,
                        (unsigned long long) (st1.misses - st0.misses), (st1.bytes - st0.bytes) / 1e9, ms,
                        (st1.bytes - st0.bytes) / 1e6 / ms);
        }
        int32_t* d_ids = nullptr;
        check(cudaMalloc(&d_ids, 64), "ids");
        for (int tok = 0; tok < 3; ++tok) {   // decode: 8 a layer, one layer after another
            const auto st0 = c.stats();
            const auto t0 = std::chrono::steady_clock::now();
            for (int64_t il = l0; il < g.n_layers; ++il) {
                std::shuffle(all.begin(), all.end(), rng);
                check(cudaMemcpy(d_ids, all.data(), 8 * sizeof(int32_t), cudaMemcpyHostToDevice), "ids");
                if (!c.experts(il, d_ids, 8, s, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
            }
            check(cudaStreamSynchronize(s), "sync");
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const auto st1 = c.stats();
            std::printf("  a token, 8 experts x %lld layers: %llu misses, %.2f GB in %6.0f ms = %.2f GB/s, %.1f ms a layer\n",
                        (long long) nl, (unsigned long long) (st1.misses - st0.misses), (st1.bytes - st0.bytes) / 1e9, ms,
                        (st1.bytes - st0.bytes) / 1e6 / ms, ms / (double) nl);
        }
        cudaFree(d_ids);
        cudaStreamDestroy(s);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}

/// `--logits SHARD PROMPT_IDS OUT_PREFIX N_NEW [slots]`: the real model as serving runs it - the prompt's ids (a text
/// file of integers) through prompt(), then N_NEW greedy tokens through decode() (stopping at an end token) - writing
/// OUT_PREFIX.ids (every id, prompt and answer, as text) and OUT_PREFIX.logits (float32 [answer tokens][n_vocab]: row r
/// the logits that chose answer token r).  For comparing with another implementation on the same ids
/// (tools/glm_compare_logits.py).
///
/// Teacher-forced instead, when PROMPT_IDS has a second line P (the format OUT_PREFIX.ids has): the first P ids through
/// prompt(), then every later id through decode() whatever the model would have chosen, a row of logits before each -
/// a text's per-position predictions (perplexity) on the path serving takes.  N_NEW is then not used.
/// STRATA_GLM_TOP_K=N replaces the indexer's top_k (cells; a multiple of 4): one above the context reads every cell,
/// for how much the sparse selection changes a long text's logits.  The prompt runs layer by layer in groups of
/// STRATA_GLM_PROMPT_GROUP tokens (serving's default, --glm-prompt-group; 0: chunk by chunk) when the experts stream.
int logits_real(const char* shard, const char* ids_path, const std::string& out, int n_new, int64_t slots) {
    try {
        std::vector<int32_t> ids, forced;
        {
            std::FILE* f = std::fopen(ids_path, "r");
            if (!f) { std::fprintf(stderr, "%s: cannot open\n", ids_path); return 1; }
            std::vector<std::string> lines;
            char buf[1 << 16];
            std::string cur;
            while (std::fgets(buf, sizeof buf, f)) {
                cur += buf;
                if (!cur.empty() && cur.back() == '\n') { lines.push_back(cur); cur.clear(); }
            }
            if (!cur.empty()) lines.push_back(cur);
            std::fclose(f);
            std::vector<int32_t> all;
            if (!lines.empty()) {
                const char* p = lines[0].c_str();
                char* end = nullptr;
                for (long v = std::strtol(p, &end, 10); end != p; v = std::strtol(p, &end, 10)) {
                    all.push_back((int32_t) v);
                    p = end;
                }
            }
            const long P = lines.size() > 1 ? std::strtol(lines[1].c_str(), nullptr, 10) : 0;
            if (P > 0 && P < (long) all.size()) {   // teacher-forced: the rest of the ids are the answer
                ids.assign(all.begin(), all.begin() + P);
                forced.assign(all.begin() + P, all.end());
                n_new = (int) forced.size();
            } else {
                ids = all;
            }
        }
        const strata::GgufModel m = strata::GgufModel::open(shard);
        Glm5NextGeometry g;
        if (const std::string e = strata::read_geometry(m.meta(), g); !e.empty()) {
            std::fprintf(stderr, "geometry: %s\n", e.c_str());
            return 1;
        }
        if (const char* e = std::getenv("STRATA_GLM_TOP_K")) {
            g.idx_top_k = std::atoll(e);
            std::printf("indexer top_k %lld (STRATA_GLM_TOP_K)\n", (long long) g.idx_top_k);
        }
        glm::GlmLoadOptions opt;
        opt.expert_cache_slots = slots;
        glm::GlmLoadedWeights lw;
        std::string err;
        if (!lw.load(m, g, opt, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        glm::GlmModel model;
        const int64_t n_prompt = (int64_t) ids.size();
        if (!model.init(g, lw.weights(), n_prompt + n_new + 16, err, lw.expert_cache(), 256)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        model.set_mtp_over_prompt(false);
        {   // as serving runs a prompt (--glm-prompt-group); STRATA_GLM_PROMPT_GROUP=0: chunk by chunk
            const char* e = std::getenv("STRATA_GLM_PROMPT_GROUP");
            const int64_t group = e ? std::atoll(e) : glm::GlmServeOptions{}.prompt_group;
            if (slots > 0 && !model.init_prompt_group(group, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            std::printf("prompt group %lld tokens\n", (long long) model.prompt_group());
        }
        cudaStream_t s;
        check(cudaStreamCreate(&s), "stream");
        model.reset(s);
        const size_t V = (size_t) g.n_vocab;
        std::vector<float> row(V);
        std::FILE* fl = std::fopen((out + ".logits").c_str(), "wb");
        if (!fl) { std::fprintf(stderr, "%s.logits: cannot write\n", out.c_str()); return 1; }
        const auto t0 = std::chrono::steady_clock::now();
        if (!model.prompt(ids.data(), n_prompt, 0, s, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        check(cudaStreamSynchronize(s), "prompt");
        std::printf("prompt of %lld tokens in %.1f s\n", (long long) n_prompt,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        const std::vector<int32_t> ends = {154820, 154827, 154829};   // <|endoftext|> <|user|> <|observation|>
        for (int t = 0; t < n_new; ++t) {
            check(cudaMemcpyAsync(row.data(), model.logits(), V * 4, cudaMemcpyDeviceToHost, s), "logits");
            check(cudaStreamSynchronize(s), "sync");
            std::fwrite(row.data(), 4, V, fl);
            const int32_t next = forced.empty() ? (int32_t) (std::max_element(row.begin(), row.end()) - row.begin())
                                                : forced[(size_t) t];
            ids.push_back(next);
            if ((forced.empty() && std::find(ends.begin(), ends.end(), next) != ends.end()) || t + 1 == n_new) break;
            if (!model.decode(next, (int64_t) ids.size() - 1, s, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
        }
        std::fclose(fl);
        std::FILE* fi = std::fopen((out + ".ids").c_str(), "w");
        for (size_t i = 0; i < ids.size(); ++i) std::fprintf(fi, "%d%c", ids[i], i + 1 < ids.size() ? ' ' : '\n');
        std::fprintf(fi, "%lld\n", (long long) n_prompt);   // second line: the prompt's length
        std::fclose(fi);
        std::printf("%lld prompt ids, %lld answer tokens in %.1f s\n", (long long) n_prompt,
                    (long long) ids.size() - n_prompt,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        cudaStreamDestroy(s);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}

/// `--bench`: GlmExpertCache at GLM-5.3-Flash's expert size (n_embd 4096, n_ff 2048, IQ4_XS: 13.4 MB a blob), one
/// layer of 32 experts written to a GGUF and mapped (its pages are in the OS cache after the write: a warm file).
/// The copy of a call's 8 misses into device or mapped host slots by 1, 4 and 8 threads, and a call whose 8 experts
/// all hit (the ids' trip to the host and the stream synchronization: the cost every MoE layer pays).
int bench() {
    const uint64_t D = 4096, F = 2048, NE = 32;
    TempDir dir;
    const fs::path path = dir.path / "glm-experts.gguf";
    fixture::write(path, {fixture::str("general.architecture", "glm5-next")},
                   {tensor("blk.0.ffn_gate_exps.weight", {D, F, NE}, 23, {}),
                    tensor("blk.0.ffn_up_exps.weight", {D, F, NE}, 23, {}),
                    tensor("blk.0.ffn_down_exps.weight", {F, D, NE}, 23, {})});
    const strata::GgufModel m({path.string()});
    glm::GlmExpertSource src;
    src.gate = m.shard(0).tensor_data(*m.find("blk.0.ffn_gate_exps.weight"));
    src.up = m.shard(0).tensor_data(*m.find("blk.0.ffn_up_exps.weight"));
    src.down = m.shard(0).tensor_data(*m.find("blk.0.ffn_down_exps.weight"));
    src.layout = k::native_expert_layout(23, 23, (int64_t) D, (int64_t) F);
    std::printf("GlmExpertCache, one layer of %llu IQ4_XS experts of %.1f MB (GLM-5.3-Flash's size), a warm file:\n",
                (unsigned long long) NE, (double) src.layout.bytes / 1e6);
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    int32_t* d_ids = nullptr;
    check(cudaMalloc(&d_ids, 64), "ids");
    std::string err;
    glm::GlmExpertSource dsrc = src;   // the same experts, read unbuffered from the file
    dsrc.gate_file = dsrc.up_file = dsrc.down_file = 0;
    dsrc.gate_off = m.shard(0).data_start() + m.find("blk.0.ffn_gate_exps.weight")->offset;
    dsrc.up_off = m.shard(0).data_start() + m.find("blk.0.ffn_up_exps.weight")->offset;
    dsrc.down_off = m.shard(0).data_start() + m.find("blk.0.ffn_down_exps.weight")->offset;
    for (bool mapped : {false, true})
        for (int threads : {0, 1, 4, 8}) {   // 0: unbuffered reads (the default when the cache has the files)
            glm::GlmExpertCache c;
            if (!(threads ? c.init({src}, (int64_t) NE, 8, mapped, threads, err)
                          : c.init({dsrc}, (int64_t) NE, 8, mapped, 1, err, {path.string()}))) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            // 4 calls of 8 misses each (8 slots: every call evicts the last), the first one not timed
            double ms = 0;
            for (int call = 0; call < 5; ++call) {
                int32_t ids[8];
                for (int i = 0; i < 8; ++i) ids[i] = (call % 4) * 8 + i;
                check(cudaMemcpy(d_ids, ids, sizeof ids, cudaMemcpyHostToDevice), "ids");
                const auto t0 = std::chrono::steady_clock::now();
                if (!c.experts(0, d_ids, 8, s, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
                check(cudaStreamSynchronize(s), "sync");
                if (call) ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            }
            const double mb = 4 * 8 * (double) src.layout.bytes / 1e6;
            if (threads)
                std::printf("  8 misses into %-11s slots, mapped file, %d thread%s: %6.2f ms a call, %5.2f GB/s\n",
                            mapped ? "mapped host" : "device", threads, threads > 1 ? "s" : " ", ms / 4, mb / ms);
            else
                std::printf("  8 misses into %-11s slots, unbuffered reads:      %6.2f ms a call, %5.2f GB/s\n",
                            mapped ? "mapped host" : "device", ms / 4, mb / ms);
        }
    {
        glm::GlmExpertCache c;
        if (!c.init({src}, (int64_t) NE, 8, false, 8, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        int32_t ids[8];
        for (int i = 0; i < 8; ++i) ids[i] = i;
        check(cudaMemcpy(d_ids, ids, sizeof ids, cudaMemcpyHostToDevice), "ids");
        c.experts(0, d_ids, 8, s, err);
        const int N = 2000;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) c.experts(0, d_ids, 8, s, err);
        check(cudaStreamSynchronize(s), "sync");
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
        std::printf("  8 hits, an idle stream: %.1f us a call (x 43 MoE layers = %.2f ms a token)\n", us, us * 43 / 1000);
    }
    // the grouped expert kernel (one decode token, 8 experts, the SwiGLU clamp) reading its blobs from each home
    for (bool mapped : {false, true}) {
        glm::GlmExpertCache c;
        if (!c.init({src}, (int64_t) NE, 8, mapped, 8, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        int32_t ids[8];
        for (int i = 0; i < 8; ++i) ids[i] = i;
        check(cudaMemcpy(d_ids, ids, sizeof ids, cudaMemcpyHostToDevice), "ids");
        const unsigned long long* table = c.experts(0, d_ids, 8, s, err);
        float* x = upload(gvec(D, 1.0f));
        void *xq = nullptr, *scr = nullptr;
        float* parts = nullptr;
        unsigned long long* grp_ptr = nullptr;
        int32_t *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr;
        check(cudaMalloc(&xq, D / 32 * 36 + 256), "xq");
        check(cudaMalloc(&scr, k::native_expert_scratch_bytes(8, (int64_t) F)), "scratch");
        check(cudaMalloc(&parts, 8 * D * 4), "parts");
        check(cudaMalloc(&grp_ptr, 64), "grp");
        check(cudaMalloc(&grp_start, 64), "grp");
        check(cudaMalloc(&n_groups, 4), "grp");
        check(cudaMalloc(&ent_dst, 64), "ent");
        check(cudaMalloc(&ent_tok, 64), "ent");
        auto run = [&]() {
            k::glm_expert_groups_decode(d_ids, table, 8, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, s);
            k::quantize_q8_1_rows(x, 1, (int64_t) D, xq, s);
            k::native_expert_grouped_clamped(src.layout, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, 8, 8, xq, scr,
                                             parts, 10.0f, s);
        };
        for (int i = 0; i < 5; ++i) run();
        check(cudaStreamSynchronize(s), "warm");
        const int N = 50;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) run();
        check(cudaStreamSynchronize(s), "kernel");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N;
        std::printf("  the grouped expert kernel, 8 experts in %-11s memory: %.3f ms (%.0f GB/s of blobs)\n",
                    mapped ? "mapped host" : "device", ms, 8 * (double) src.layout.bytes / 1e6 / ms);
        for (void* p : {(void*) x, xq, scr, (void*) parts, (void*) grp_ptr, (void*) grp_start, (void*) n_groups,
                        (void*) ent_dst, (void*) ent_tok})
            cudaFree(p);
    }
    cudaFree(d_ids);
    cudaStreamDestroy(s);
    return 0;
}

/// Runs `a` and `b` over the same tokens, comparing logits, h_nextn and the MTP logits bitwise.
void same_outputs(glm::GlmModel& a, glm::GlmModel& b, const Glm5NextGeometry& g, const char* what) {
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    a.reset(s);
    b.reset(s);
    std::string err;
    bool same = true;
    const int32_t tokens[STEPS + 1] = {1513, 265, 1973, 1060, 658, 670, 1695};
    for (int t = 0; t < STEPS; ++t) {
        for (glm::GlmModel* m : {&a, &b}) {
            if (!m->decode(tokens[t], t, s, err) || !m->mtp_decode(tokens[t + 1], t, m->h_nextn(), s, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                std::exit(1);
            }
        }
        check(cudaStreamSynchronize(s), "decode");
        const size_t V = (size_t) g.n_vocab, D = (size_t) g.n_embd;
        same &= raw(a.logits(), V * 4) == raw(b.logits(), V * 4);
        same &= raw(a.h_nextn(), D * 4) == raw(b.h_nextn(), D * 4);
        same &= raw(a.mtp_logits(), V * 4) == raw(b.mtp_logits(), V * 4);
    }
    cudaStreamDestroy(s);
    expect(same, std::string(what) + ": logits, h_nextn and MTP logits bitwise the direct model's, " +
                     std::to_string(STEPS) + " tokens");
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--plan" && i + 1 < argc) return plan_real(argv[++i]);
        else if (a == "--bench") return bench();
        else if (a == "--logits" && i + 4 < argc) {
            const char* shard = argv[i + 1];
            const char* ids = argv[i + 2];
            const std::string out = argv[i + 3];
            const int n_new = std::atoi(argv[i + 4]);
            return logits_real(shard, ids, out, n_new, i + 5 < argc ? std::atoll(argv[i + 5]) : 4000);
        } else if (a == "--bench-file" && i + 1 < argc) {
            const char* shard = argv[++i];
            return bench_file(shard, i + 1 < argc ? std::atoll(argv[i + 1]) : 1024);
        } else {
            std::fprintf(stderr, "usage: glm_loader_parity [--selftest] | --plan SHARD | --bench | --bench-file SHARD [slots] | --logits SHARD PROMPT_IDS OUT N_NEW [slots]\n");
            return 2;
        }
    }
    TestModel tm = build_test_model(MAX_CTX);
    const Glm5NextGeometry& g0 = tm.g;
    TempDir dir;
    const fs::path path = dir.path / "glm-test.gguf";
    std::vector<fixture::Tensor> ts = model_tensors(tm);
    fixture::write(path, model_keys(g0), ts);
    std::printf("glm_loader: the synthetic model (5 layers + MTP) as a glm5-next GGUF of %.1f MB, %zu tensors\n",
                (double) fs::file_size(path) / 1e6, ts.size());

    // 1. the geometry
    Glm5NextGeometry g;
    {
        const strata::GgufModel m({path.string()});
        const std::string e = strata::read_geometry(m.meta(), g);
        expect(e.empty() && g.n_layers == g0.n_layers && g.n_nextn == g0.n_nextn && g.is_mla == g0.is_mla &&
                   g.idx_full == g0.idx_full && g.n_embd == g0.n_embd && g.n_vocab == g0.n_vocab &&
                   g.n_expert == g0.n_expert && g.n_ff_exp == g0.n_ff_exp && g.n_dense_lead == g0.n_dense_lead &&
                   (float) g.rms_eps == (float) g0.rms_eps && g.n_ff_shexp == g0.n_ff_shexp,
               "the geometry read back is the model's" + (e.empty() ? std::string() : ": " + e));
    }

    // 2. loaded, both expert homes, against the model built directly
    glm::GlmModel direct;
    std::string err;
    if (!direct.init(g0, tm.w, MAX_CTX, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    for (glm::GlmExpertHome home : {glm::GlmExpertHome::Device, glm::GlmExpertHome::MappedHost}) {
        const bool dev = home == glm::GlmExpertHome::Device;
        const strata::GgufModel m({path.string()});
        glm::GlmLoadedWeights lw;
        glm::GlmLoadOptions opt;
        opt.experts = home;
        if (!lw.load(m, g, opt, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        if (dev) print_plan(lw.plan());
        glm::GlmModel loaded;
        if (!loaded.init(g, lw.weights(), MAX_CTX, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        same_outputs(direct, loaded, g, dev ? "experts in device memory" : "experts in mapped host memory");
    }

    // 2b. streamed through a cache smaller than the 80 experts (5 MoE blocks of 16): every eviction and refill must
    // leave the outputs bitwise the same
    struct CacheCase {
        int64_t slots;
        bool fill, mapped;
        const char* what;
    };
    for (const CacheCase& c0 : {CacheCase{24, false, false, "a cache of 24 slots in device memory, empty at start"},
                                CacheCase{4, false, true, "a cache of 4 slots (one layer's k) in mapped host memory"},
                                CacheCase{40, true, false, "a cache of 40 slots filled before the first token"}})
    for (const int how : {0, 1, 2}) {   // unbuffered reads, through the mapping, unbuffered and reading ahead
        const bool direct_reads = how != 1, ahead = how == 2;
        if (ahead && c0.slots < 8) continue;   // 4 slots: one layer's k, none to spare for a guess
        const std::string label = std::string(c0.what) +
                                  (ahead ? ", reading ahead" : direct_reads ? ", unbuffered reads" : ", through the mapping");
        const CacheCase c{c0.slots, c0.fill, c0.mapped, label.c_str()};
        const strata::GgufModel m({path.string()});
        glm::GlmLoadedWeights lw;
        glm::GlmLoadOptions opt;
        opt.experts = c.mapped ? glm::GlmExpertHome::MappedHost : glm::GlmExpertHome::Device;
        opt.expert_cache_slots = c.slots;
        opt.cache_fill = c.fill;
        opt.direct_reads = direct_reads;
        opt.prefetch = ahead ? 2 : 0;
        if (!lw.load(m, g, opt, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        expect(lw.expert_cache()->direct() == direct_reads, label + ": the read path asked for");
        glm::GlmModel loaded;
        if (!loaded.init(g, lw.weights(), MAX_CTX, err, lw.expert_cache())) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        const int64_t resident0 = lw.expert_cache()->resident();
        same_outputs(direct, loaded, g, c.what);
        const glm::GlmExpertCache::Stats& st = lw.expert_cache()->stats();
        std::printf("    %lld resident at start; %llu lookups: %llu hits, %llu misses (%.1f MB read)\n",
                    (long long) resident0, (unsigned long long) (st.hits + st.misses), (unsigned long long) st.hits,
                    (unsigned long long) st.misses, (double) st.bytes / 1e6);
        expect(st.misses > 0 && (c.fill ? st.hits > 0 : true) && resident0 == (c.fill ? 40 : 0),
               std::string(c.what) + ": misses were served" + (c.fill ? ", and the fill gave hits" : ""));
        if (ahead) {
            std::printf("    guesses %llu of %llu right; %llu read ahead, %llu of them then hit\n",
                        (unsigned long long) st.guessed_right, (unsigned long long) st.guessed,
                        (unsigned long long) st.prefetched, (unsigned long long) st.prefetch_hits);
            expect(lw.expert_cache()->prefetching() && st.guessed > 0 && st.prefetched > 0 && st.prefetch_hits > 0,
                   std::string(c.what) + ": guesses were made, read ahead and then hit");
        }
    }
    {   // fewer slots than one layer's experts: refused at the first MoE layer, by name
        const strata::GgufModel m({path.string()});
        glm::GlmLoadedWeights lw;
        glm::GlmLoadOptions opt;
        opt.expert_cache_slots = 3;
        opt.cache_fill = false;
        if (!lw.load(m, g, opt, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        glm::GlmModel loaded;
        if (!loaded.init(g, lw.weights(), MAX_CTX, err, lw.expert_cache())) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        cudaStream_t s;
        check(cudaStreamCreate(&s), "stream");
        loaded.reset(s);
        std::string e;
        const bool ok = loaded.decode(1513, 0, s, e);
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
        expect(!ok && e.find("3 slots cannot hold one layer's 4 experts") != std::string::npos,
               "a cache of 3 slots for k = 4: \"" + e + "\"");
    }

    // 3. the conversions
    {
        std::vector<fixture::Tensor> cv = ts;
        const Block& b0 = tm.blocks[0];
        const Block& b1 = tm.blocks[1];
        const Block& b3 = tm.blocks[3];
        // blk.0.ssm_norm F16, blk.1.attn_norm BF16
        vector<uint16_t> f16v(b0.o_norm.size()), bf16v(b1.attn_norm.size());
        for (size_t i = 0; i < f16v.size(); ++i) f16v[i] = k::f16_from_f32(b0.o_norm[i]);
        for (size_t i = 0; i < bf16v.size(); ++i) bf16v[i] = bf16_bits(b1.attn_norm[i]);
        auto bytes16 = [](const vector<uint16_t>& v) {
            vector<uint8_t> r(v.size() * 2);
            std::memcpy(r.data(), v.data(), r.size());
            return r;
        };
        *find_t(cv, "blk.0.ssm_norm.weight") = tensor("blk.0.ssm_norm.weight", {128}, 1, bytes16(f16v));
        *find_t(cv, "blk.1.attn_norm.weight") = tensor("blk.1.attn_norm.weight", {1024}, 30, bytes16(bf16v));
        // blk.1.attn_k_b Q8_0: 32 values a block, an FP16 scale amax / 127, the values rounded
        const size_t nk = b1.k_b.size();
        vector<uint8_t> q8(nk / 32 * 34);
        vector<uint16_t> kb_want(nk);
        for (size_t blk = 0; blk < nk / 32; ++blk) {
            float amax = 0;
            for (size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(b1.k_b[blk * 32 + i]));
            const uint16_t dh = k::f16_from_f32(amax / 127.0f);
            const float d = k::f32_from_f16(dh), id = d != 0 ? 1.0f / d : 0.0f;
            std::memcpy(&q8[blk * 34], &dh, 2);
            for (size_t i = 0; i < 32; ++i) {
                const int8_t q = (int8_t) std::lround(b1.k_b[blk * 32 + i] * id);
                q8[blk * 34 + 2 + i] = (uint8_t) q;
                kb_want[blk * 32 + i] = bf16_bits((float) q * d);
            }
        }
        *find_t(cv, "blk.1.attn_k_b.weight") = tensor("blk.1.attn_k_b.weight", {256, 512, 8}, 8, q8);
        // blk.3.attn_v_b F32 (BF16 values, so the rounding gives them back), blk.0.ssm_conv1d_q {4, 1, d_inner}
        *find_t(cv, "blk.3.attn_v_b.weight") = f32_t("blk.3.attn_v_b.weight", {512, 256, 8}, b3.v_b);
        *find_t(cv, "blk.0.ssm_conv1d_q.weight") = f32_t("blk.0.ssm_conv1d_q.weight", {4, 1, 1024}, b0.conv_q);
        const fs::path cpath = dir.path / "glm-conv.gguf";
        fixture::write(cpath, model_keys(g0), cv);
        const strata::GgufModel m({cpath.string()});
        glm::GlmLoadedWeights lw;
        if (!lw.load(m, g, glm::GlmLoadOptions{}, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const glm::GlmModelWeights& w = lw.weights();
        bool ok = true;
        const vector<float> on = download(w.blocks[0].o_norm, f16v.size());
        for (size_t i = 0; i < f16v.size(); ++i) ok &= on[i] == k::f32_from_f16(f16v[i]);
        expect(ok, "an F16 vector (blk.0.ssm_norm) loaded as its F32 values");
        ok = true;
        const vector<float> an = download(w.blocks[1].attn_norm, bf16v.size());
        for (size_t i = 0; i < bf16v.size(); ++i) ok &= an[i] == bf16_val(bf16v[i]);
        expect(ok, "a BF16 vector (blk.1.attn_norm) loaded as its F32 values");
        expect(download(w.blocks[1].k_b, nk) == kb_want,
               "a Q8_0 attn_k_b loaded as the BF16 of its dequantized values");
        vector<uint16_t> vb_want(b3.v_b.size());
        for (size_t i = 0; i < vb_want.size(); ++i) vb_want[i] = bf16_bits(b3.v_b[i]);
        expect(download(w.blocks[3].v_b, vb_want.size()) == vb_want, "an F32 attn_v_b loaded as its BF16 values");
        expect(download(w.blocks[0].conv_q, b0.conv_q.size()) == b0.conv_q,
               "a conv1d of three dimensions {4, 1, d_inner} loaded as [d_inner][4]");
    }

    // 4. what the plan refuses
    {
        const fs::path epath = dir.path / "glm-bad.gguf";
        auto refused = [&](std::vector<fixture::Tensor> bad, const std::string& want, const char* what,
                           const glm::GlmLoadOptions& opt = {}) {
            const std::string e = plan_of(epath, g0, bad, opt);
            expect(e.find(want) != std::string::npos, std::string(what) + ": \"" + e + "\"");
        };
        std::vector<fixture::Tensor> bad = ts;
        bad.erase(bad.begin() + (find_t(bad, "blk.3.exp_probs_b.bias") - bad.data()));
        refused(bad, "blk.3.exp_probs_b.bias: missing", "a tensor missing");
        bad = ts;
        *find_t(bad, "blk.1.attn_kv_a_mqa.weight") = tensor("blk.1.attn_kv_a_mqa.weight", {1024, 256}, 23, {});
        refused(bad, "blk.1.attn_kv_a_mqa.weight: shape {1024, 256}", "a shape the geometry does not give");
        bad = ts;
        *find_t(bad, "blk.0.ssm_norm.weight") = tensor("blk.0.ssm_norm.weight", {128}, 8, {});
        refused(bad, "blk.0.ssm_norm.weight: Q8_0 is not a type", "a type no kernel takes");
        bad = ts;
        *find_t(bad, "blk.2.ffn_up_exps.weight") = tensor("blk.2.ffn_up_exps.weight", {1024, 256, 16}, 8, {});
        refused(bad, "blk.2.ffn_gate_exps.weight: IQ4_XS and its up_exps Q8_0", "experts' gate and up of two types");
        bad = ts;
        for (const char* n : {"nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
                              "nextn.shared_head_norm.weight"})
            bad.erase(bad.begin() + (find_t(bad, std::string("blk.5.") + n) - bad.data()));
        refused(bad, "blk.5.nextn.eh_proj.weight: missing", "the MTP block missing");
        glm::GlmLoadOptions no_mtp;
        no_mtp.mtp = false;
        glm::GlmPlan p;
        const std::string e = plan_of(epath, g0, bad, no_mtp, &p);
        expect(e.empty() && !p.mtp, "the same file planned without the MTP block" + (e.empty() ? "" : ": " + e));
    }

    std::printf("\nglm_loader: %d failures\n", g_bad);
    for (void* p : g_free) cudaFree(p);
    if (g_bad) return 1;
    if (selftest) std::printf("glm_loader_parity OK\n");
    return 0;
}
