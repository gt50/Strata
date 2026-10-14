// include/strata/glm/glm_serve.hpp - `strata --serve` for a GLM-5.3-Flash GGUF (general.architecture glm5-next):
// the line protocol serve/server.py speaks to the engine (src/program/generate.cpp's --serve), on GlmModel.
//
//   engine -> server   INFO engine=<version> arch=glm5-next context=<n> mtp=<0|1> spec=<drafts>, then READY <ctx> stop
//   server -> engine   GEN <max_new> [key=value ...] <id,id,...>    temperature, top_p, top_k, min_p, penalty_last_n,
//                                                                    penalty_repeat, penalty_freq, penalty_present, seed
//                                                                    (absent: greedy); other keys are skipped
//   engine -> server   T <id> a token (the end token too, then the request ends), and
//                      DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> <drafts accepted>
//                           <drafts offered> 0 0 0 0 0 0.0 <prompt> 0
//                      (the Qwen engine's columns: nothing reused, no expert-tier counts)
//   STOP               while a request runs: it ends after the tokens already produced (finish cancel)
//   QUIT               the engine exits; any other command: ERR
//
// A request starts from an empty context (no prompt reuse yet), one at a time (no batch slots).  The model is
// GlmLoadedWeights' from the GGUF, the experts resident when they fit in the device's free memory, else streamed
// through a GlmExpertCache; speculative decoding with --spec N (the MTP block's drafts, N a round; GlmGenerator).
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::glm {

struct GlmServeOptions {
    std::string model;                ///< --native: the GGUF (any shard of a split one)
    int64_t max_ctx = 32768;          ///< --max-context
    int n_draft = 0;                  ///< --spec N: drafts a round (0: none)
    std::vector<int32_t> eos_ids;     ///< --eos-ids a,b,...; empty: the GGUF's end tokens
    int64_t max_chunk = 256;          ///< --glm-chunk: a prompt's tokens a pass
    int64_t prompt_group = 8192;      ///< --glm-prompt-group: with streamed experts, a prompt's tokens run layer by
                                      ///< layer (GlmModel::init_prompt_group; 0: chunk by chunk)
    int64_t expert_cache_slots = -1;  ///< --glm-expert-cache N: experts streamed through N slots; 0 all resident;
                                      ///< -1 (default) resident when they fit, else as many slots as fit
    bool experts_host = false;        ///< --glm-experts-host: the experts (or their slots) in mapped host memory
    double host_reserve_gb = 12.0;    ///< --glm-host-reserve-gb: on an integrated GPU the automatic count leaves the
                                      ///< host this much RAM available (its device memory past the carve-out is RAM)
};

/// Whether the GGUF at `path` (any shard) is a GLM-5.3-Flash one: what generate.cpp dispatches on.
bool glm_is_glm_gguf(const std::string& path);

/// The flags of the engine's command line that GLM serving takes (the Qwen-only ones are skipped, named on stderr).
bool glm_serve_options(int argc, char** argv, GlmServeOptions& o, std::string& err);

class GlmServer {
public:
    GlmServer();
    ~GlmServer();
    /// Opens and loads the model; `log` gets progress lines (stderr in the engine).
    bool load(const GlmServeOptions& o, std::string& err, const std::function<void(const std::string&)>& log);
    /// The protocol until QUIT or the end of the input: `next_line` blocks for the next command (false at the end),
    /// `stop` is raised by a STOP that arrives while a request runs (cleared when one starts), `emit` writes a line
    /// (READY first).  0 at QUIT / the end; 1 when the engine cannot go on.
    int run(const std::function<bool(std::string&)>& next_line, std::atomic<bool>& stop,
            const std::function<void(const std::string&)>& emit);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

/// `strata --serve --native <glm gguf> ...`: load, then the protocol on stdin / stdout (a reader thread turns STOP
/// into the stop flag).  The process's exit code.
int glm_serve_stdio(int argc, char** argv);

}  // namespace strata::glm
