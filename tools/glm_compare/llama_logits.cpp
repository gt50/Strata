// tools/glm_compare/llama_logits.cpp - llama.cpp's logits on a fixed token sequence, for comparing Strata's GLM engine
// with the reference implementation on the same ids (tools/glm_compare_logits.py).
//
//   llama_logits MODEL IDS OUT [threads] [ubatch]
//
// IDS is what `glm_loader_parity --logits` writes (OUT_PREFIX.ids): line 1 every id, prompt and answer; line 2 the
// prompt's length P.  The whole sequence runs as one batch (teacher-forced), on the CPU (n_gpu_layers 0: the reference,
// no GPU kernels), and OUT gets float32 [n - P][n_vocab]: the logits at positions P - 1 .. n - 2, each the prediction
// of the next id - the same rows Strata wrote.  `ubatch` 1 runs the tokens one at a time through the same batch (as
// decoding does): llama.cpp's own second path, for how far two correct paths can differ at a position.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: llama_logits MODEL IDS OUT [threads] [ubatch]\n");
        return 2;
    }
    const int threads = argc > 4 ? std::atoi(argv[4]) : 16;
    const int ubatch = argc > 5 ? std::atoi(argv[5]) : 0;   // 0: the whole sequence at once
    std::vector<llama_token> ids;
    long long n_prompt = 0;
    {
        std::FILE* f = std::fopen(argv[2], "r");
        if (!f) {
            std::fprintf(stderr, "%s: cannot open\n", argv[2]);
            return 1;
        }
        std::vector<long long> all;
        long long v;
        while (std::fscanf(f, "%lld", &v) == 1) all.push_back(v);
        std::fclose(f);
        if (all.size() < 3) {
            std::fprintf(stderr, "%s: no ids\n", argv[2]);
            return 1;
        }
        n_prompt = all.back();   // the second line's single number
        all.pop_back();
        for (long long x : all) ids.push_back((llama_token) x);
    }
    const int n = (int) ids.size();
    if (n_prompt < 1 || n_prompt >= n) {
        std::fprintf(stderr, "prompt length %lld of %d ids\n", n_prompt, n);
        return 1;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;   // 157 GB on a 64 GB PC: mapped, read as the tokens need it
    mp.use_extra_bufts = false;            // no repacking (it would copy the weights into RAM)
    mp.load_mtp = false;                   // the main model only: the comparison is of its logits
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "%s: the model did not load\n", argv[1]);
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) (n + 16);
    cp.n_batch = (uint32_t) n;
    cp.n_ubatch = (uint32_t) (ubatch > 0 ? ubatch : n);
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "no context\n");
        return 1;
    }
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = ids[(size_t) i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = i >= n_prompt - 1 && i < n - 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) {
        std::fprintf(stderr, "llama_decode failed\n");
        return 1;
    }
    std::FILE* out = std::fopen(argv[3], "wb");
    if (!out) {
        std::fprintf(stderr, "%s: cannot write\n", argv[3]);
        return 1;
    }
    for (int i = (int) n_prompt - 1; i < n - 1; ++i) std::fwrite(llama_get_logits_ith(ctx, i), 4, (size_t) n_vocab, out);
    std::fclose(out);
    std::printf("%d ids (prompt %lld): %d rows of %d logits\n", n, n_prompt, n - (int) n_prompt, n_vocab);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
