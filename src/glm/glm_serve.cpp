// src/glm/glm_serve.cpp - see include/strata/glm/glm_serve.hpp.
#include "strata/glm/glm_serve.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/glm/glm_loader.hpp"
#include "strata/glm/glm_spec.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>

namespace strata::glm {
namespace {

std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

/// "a,b,c" -> ints; false on anything else.
bool parse_ids(const char* s, std::vector<int32_t>& out) {
    out.clear();
    while (*s == ' ') ++s;
    if (*s == '\0') return false;
    for (;;) {
        char* end = nullptr;
        const long long v = std::strtoll(s, &end, 10);
        if (end == s || v < 0 || v > 0x7fffffff) return false;
        out.push_back((int32_t) v);
        s = end;
        if (*s == ',') { ++s; continue; }
        while (*s == ' ') ++s;
        return *s == '\0';
    }
}

/// The GGUF's end tokens: tokenizer.ggml.eos / eot / eom_token_id, those present.
std::vector<int32_t> gguf_end_tokens(const GgufFile& f) {
    std::vector<int32_t> ids;
    for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"})
        if (const MetaValue* v = f.get(key); v && v->is_num()) {
            const int32_t id = (int32_t) v->num();
            if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
        }
    return ids;
}

}  // namespace

bool glm_is_glm_gguf(const std::string& path) {
    try {
        const GgufFile f(path);
        const MetaValue* a = f.get("general.architecture");
        return a && a->type == MetaType::STRING && a->s == "glm5-next";
    } catch (const std::exception&) {
        return false;
    }
}

bool glm_serve_options(int argc, char** argv, GlmServeOptions& o, std::string& err) {
    // the engine's flags that take a value (skipped with it when GLM does not use them)
    static const char* const kValued[] = {
        "--pack", "--mtp", "--prefill", "--batch", "--slots", "--pcie-frac", "--vram-segment-mib", "--config",
        "--expert-cache-mib", "--cvec", "--cvec-dir", "--peer-device", "--peer-reserve-mib", "--peer-slots",
        "--prompt-cache-root", "--prompt-cache-every", "--aux-cpus", "--lookup-chain", "--mtp-window", "--threads"};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                err = std::string(name) + " needs a value";
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--serve") continue;
        if (a == "--native") {
            const char* v = value("--native");
            if (!v) return false;
            o.model = v;
        } else if (a == "--max-context") {
            const char* v = value("--max-context");
            if (!v) return false;
            o.max_ctx = std::atoll(v);
        } else if (a == "--spec") {
            const char* v = value("--spec");
            if (!v) return false;
            o.n_draft = std::max(0, std::atoi(v));
        } else if (a == "--eos-ids") {
            const char* v = value("--eos-ids");
            if (!v || !parse_ids(v, o.eos_ids)) {
                err = "--eos-ids: a list of token ids, comma-separated";
                return false;
            }
        } else if (a == "--glm-chunk") {
            const char* v = value("--glm-chunk");
            if (!v) return false;
            o.max_chunk = std::max<int64_t>(16, std::atoll(v));
        } else if (a == "--glm-expert-cache") {
            const char* v = value("--glm-expert-cache");
            if (!v) return false;
            o.expert_cache_slots = std::atoll(v);
        } else if (a == "--glm-experts-host") {
            o.experts_host = true;
        } else {
            const bool valued = std::find_if(std::begin(kValued), std::end(kValued),
                                             [&](const char* k) { return a == k; }) != std::end(kValued);
            std::fprintf(stderr, "strata serve (GLM): %s%s is not used for GLM models\n", a.c_str(),
                         valued && i + 1 < argc ? (std::string(" ") + argv[i + 1]).c_str() : "");
            if (valued) ++i;
        }
    }
    if (o.model.empty()) {
        err = "--native <gguf> is required";
        return false;
    }
    if (o.max_ctx < 64) {
        err = "--max-context: at least 64";
        return false;
    }
    return true;
}

struct GlmServer::Impl {
    GlmServeOptions o;
    std::unique_ptr<GgufModel> gguf;
    core::Glm5NextGeometry g;
    GlmLoadedWeights weights;
    GlmModel model;
    std::vector<int32_t> eos;
    cudaStream_t stream = nullptr;
    bool loaded = false;
    ~Impl() {
        if (stream) cudaStreamDestroy(stream);
    }
};

GlmServer::GlmServer() : p_(std::make_unique<Impl>()) {}
GlmServer::~GlmServer() = default;

bool GlmServer::load(const GlmServeOptions& o, std::string& err, const std::function<void(const std::string&)>& log) {
    Impl& p = *p_;
    p.o = o;
    try {
        p.gguf = std::make_unique<GgufModel>(GgufModel::open(o.model));
    } catch (const std::exception& e) {
        err = std::string("GLM: ") + o.model + ": " + e.what();
        return false;
    }
    if (const std::string e = read_geometry(p.gguf->meta(), p.g); !e.empty()) {
        err = "GLM: " + e;
        return false;
    }
    GlmLoadOptions lo;
    lo.experts = o.experts_host ? GlmExpertHome::MappedHost : GlmExpertHome::Device;
    lo.expert_cache_slots = std::max<int64_t>(0, o.expert_cache_slots);
    GlmPlan plan;
    if (const std::string e = glm_plan(*p.gguf, p.g, lo, plan); !e.empty()) {
        err = "GLM: " + e;
        return false;
    }
    if (o.expert_cache_slots < 0 && !o.experts_host) {   // resident when they fit, else as many slots as do
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            cudaGetLastError();
            free_b = 0;
        }
        const uint64_t margin = (uint64_t) 3 << 30;   // the context's state and scratch, the runtime's own
        const int64_t n_moe = p.g.n_layers - p.g.n_dense_lead;
        const uint64_t blobs = (uint64_t) std::max<int64_t>(1, n_moe * p.g.n_expert);
        const uint64_t blob = plan.expert_bytes / blobs;
        if (plan.device_bytes + plan.expert_bytes + margin > free_b && blob > 0) {
            const uint64_t room = free_b > plan.device_bytes + margin ? free_b - plan.device_bytes - margin : 0;
            lo.expert_cache_slots = std::max<int64_t>(64, (int64_t) (room / blob));
            log(fmt("strata serve (GLM): the experts (%.1f GB) do not fit beside the rest (%.1f GB of %.1f GB free): "
                    "%lld cache slots", plan.expert_bytes / 1e9, plan.device_bytes / 1e9, free_b / 1e9,
                    (long long) lo.expert_cache_slots));
        }
    }
    log(fmt("strata serve (GLM): loading %s: %d tensors, %.1f GB besides the experts, %.1f GB of experts%s",
            o.model.c_str(), plan.tensors, plan.device_bytes / 1e9, plan.expert_bytes / 1e9,
            lo.expert_cache_slots > 0 ? " (streamed)" : ""));
    const auto t0 = std::chrono::steady_clock::now();
    if (!p.weights.load(*p.gguf, p.g, lo, err)) return false;
    const int max_t = std::max(1, std::min(o.n_draft + 1, 8));
    const int64_t chunk = std::max<int64_t>(o.max_chunk, max_t + 1);
    if (!p.model.init(p.g, p.weights.weights(), o.max_ctx, err, p.weights.expert_cache(), chunk) ||
        !p.model.init_verify(max_t, err))
        return false;
    if (cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        err = "GLM: a stream";
        return false;
    }
    p.eos = o.eos_ids.empty() ? gguf_end_tokens(p.gguf->meta()) : o.eos_ids;
    log(fmt("strata serve (GLM): loaded in %.1f s; context %lld, MTP %s, %d drafts a round, end tokens %zu",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), (long long) o.max_ctx,
            p.model.has_mtp() ? "yes" : "no", p.model.has_mtp() ? std::min(o.n_draft, max_t - 1) : 0, p.eos.size()));
    p.loaded = true;
    return true;
}

int GlmServer::run(const std::function<bool(std::string&)>& next_line, std::atomic<bool>& stop,
                   const std::function<void(const std::string&)>& emit) {
    Impl& p = *p_;
    if (!p.loaded) {
        emit("ERR the GLM model is not loaded");
        return 1;
    }
    const int drafts = p.model.has_mtp() ? std::min(p.o.n_draft, p.model.max_verify() - 1) : 0;
#ifdef STRATA_VERSION
    const char* version = STRATA_VERSION;
#else
    const char* version = "dev";
#endif
    emit(fmt("INFO engine=%s arch=glm5-next context=%lld mtp=%d spec=%d", version, (long long) p.o.max_ctx,
             p.model.has_mtp() ? 1 : 0, drafts));
    emit(fmt("READY %lld stop", (long long) p.o.max_ctx));
    GlmGenerator gen(p.model);
    std::string line, err;
    std::vector<int32_t> ids, toks;
    while (next_line(line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line == "STOP") continue;   // a STOP with no request running
        if (line == "QUIT") return 0;
        if (line.rfind("GEN ", 0) != 0) {
            emit("ERR the GLM engine takes GEN, STOP and QUIT");
            continue;
        }
        // GEN <max_new> [key=value ...] <ids>
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &endp, 10);
        kernels::SamplerParams sp;
        float temperature = 0.0f;
        sp.top_k = 20;
        sp.top_p = 1.0f;
        for (;;) {
            while (endp && *endp == ' ') ++endp;
            if (!endp || *endp == '\0') break;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            const std::string tok(start, (size_t) (endp - start));
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) {
                endp = const_cast<char*>(start);
                break;
            }
            const std::string key = tok.substr(0, eq);
            const char* v = tok.c_str() + eq + 1;
            if (key == "temperature") temperature = std::strtof(v, nullptr);
            else if (key == "top_p") sp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") sp.top_k = std::atoi(v);
            else if (key == "min_p") sp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") sp.penalty_last_n = std::atoi(v);
            else if (key == "penalty_repeat") sp.penalty_repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") sp.penalty_freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") sp.penalty_present = std::strtof(v, nullptr);
            else if (key == "seed") sp.seed = std::strtoull(v, nullptr, 10);
            // other keys (logit_bias, cvec, ckpt, pin, ...): not taken here
        }
        if (max_new < 1 || !endp || !parse_ids(endp, ids)) {
            emit("ERR bad request: GEN <max_new> [key=value ...] <id,id,...>");
            continue;
        }
        const int64_t n = (int64_t) ids.size();
        if (std::any_of(ids.begin(), ids.end(), [&](int32_t t) { return t >= p.g.n_vocab; })) {
            emit("ERR bad request: a token id outside the vocabulary");
            continue;
        }
        if (n >= p.o.max_ctx) {
            emit(fmt("ERR the prompt (%lld tokens) does not fit the context (%lld)", (long long) n,
                     (long long) p.o.max_ctx));
            continue;
        }
        sp.temperature = temperature;
        sp.greedy = temperature <= 0.0f;
        p.model.set_sampling(sp);
        stop.store(false);
        const int64_t budget = std::min<int64_t>(max_new, p.o.max_ctx - n);
        const auto t0 = std::chrono::steady_clock::now();
        if (!gen.start(ids.data(), n, drafts, p.stream, err)) {
            emit("ERR " + err);
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        int64_t produced = 0;
        const char* finish = "length";
        bool failed = false;
        while (produced < budget) {
            toks.clear();
            if (!gen.next(budget - produced, toks, p.stream, err)) {
                failed = true;
                break;
            }
            bool end = false;
            for (int32_t t : toks) {
                emit(fmt("T %d", (int) t));
                ++produced;
                if (std::find(p.eos.begin(), p.eos.end(), t) != p.eos.end()) {
                    end = true;
                    break;
                }
            }
            if (end) {
                finish = "stop";
                break;
            }
            if (stop.load()) {
                finish = "cancel";
                break;
            }
        }
        if (failed) {
            emit("ERR " + err);
            return 1;
        }
        const auto t2 = std::chrono::steady_clock::now();
        const GlmSpecStats& st = gen.stats();
        emit(fmt("DONE %lld %lld %.1f %.1f %s %lld %lld 0 0 0 0 0 0.0 %lld 0", (long long) produced, (long long) n,
                 std::chrono::duration<double, std::milli>(t1 - t0).count(),
                 std::chrono::duration<double, std::milli>(t2 - t1).count(), finish, (long long) st.accepted,
                 (long long) st.drafted, (long long) n));
    }
    return 0;
}

int glm_serve_stdio(int argc, char** argv) {
    GlmServeOptions o;
    std::string err;
    if (!glm_serve_options(argc, argv, o, err)) {
        std::fprintf(stderr, "strata serve (GLM): %s\n", err.c_str());
        return 2;
    }
    auto log = [](const std::string& s) {
        std::fprintf(stderr, "%s\n", s.c_str());
        std::fflush(stderr);
    };
    GlmServer srv;
    if (!srv.load(o, err, log)) {
        std::fprintf(stderr, "strata serve (GLM): %s\n", err.c_str());
        return 1;
    }
    // stdin on a thread: STOP raises the flag at once (a request is running), every other line queues
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::thread([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") {
                stop.store(true);
                continue;
            }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(l);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    }).detach();
    auto next_line = [&](std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !lines.empty() || eof; });
        if (lines.empty()) return false;
        out = std::move(lines.front());
        lines.pop_front();
        return true;
    };
    auto emit = [](const std::string& s) {
        std::fputs(s.c_str(), stdout);
        std::fputc('\n', stdout);
        std::fflush(stdout);
    };
    const int rc = srv.run(next_line, stop, emit);
    std::fflush(stdout);
    std::_Exit(rc);   // the reader thread may still block on stdin
}

}  // namespace strata::glm
