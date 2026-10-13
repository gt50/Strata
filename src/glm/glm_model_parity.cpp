// src/glm/glm_model_parity.cpp - GlmModel (glm_model.hpp): a synthetic GLM-5.3-Flash of five layers and an MTP
// block, run from a new sequence, against glm_ref.hpp's double-precision transcription of llama.cpp's glm5-next
// graph and graph_mtp.
//
// The model has GLM's structure at glm_layer_parity's width (n_embd 1024, 8 heads, 16 experts of 256, 4 used, a
// vocabulary of 2048):
//   0 KDA + dense FFN,  1 MLA with its own indexer + MoE,  2 KDA + MoE,  3 MLA sharing 1's selection + MoE,
//   4 KDA + MoE,  then the MTP block: MLA with its own indexer + MoE, no mHC
// The token embedding is IQ4_XS, the output head Q8_0, eh_proj BF16; the MTP block has its own head norm and uses
// the model's embedding and head (llama.cpp's fallbacks).
//
// Twelve tokens from position 0, after GlmModel::reset (so the KDA state starts at zero and the selection is every
// cell).  For every token:
//   the embedding and mhc_init: the four streams are bitwise the IQ4_XS row
//   every block: from the streams the device gave it (the taps) and its latent history, as glm_layer_parity does, so
//     each block is checked on its own arithmetic, with the state it carries (the latent row it writes, the KDA conv
//     history and state); then the last indexer's cells and the last router's experts
//   the head: h_nextn = rms_norm(mean of the streams) * output_norm, and the logits and their argmax
//   the MTP block on the next token and the device's h_nextn: its h and logits, its latent row, cells and experts
// and on the last token a second draft chained on the MTP block's own h (mtp_decode(h = mtp_h())).
//
// Every result is held to 1e-4 against the reference as rounded, or with a set of up to three of its q8_1 roundings on
// a boundary taken the other way (glm_ref.hpp's verify).  This fixture needs it more than glm_layer_parity's: over a
// selection of a few cells (not 2051) a rounding that moves kv or o_mla shows through the attention, and every q8_1
// activation behind it multiplies the difference - 4.4e-3 as rounded, 1.1e-7 with the device's two roundings.
#include "glm_ref.hpp"

#include "strata/glm/glm_model.hpp"

namespace {

constexpr int64_t MAX_CTX = 64;
constexpr int STEPS = 12;

size_t argmax(const vector<float>& v) { return (size_t) (std::max_element(v.begin(), v.end()) - v.begin()); }
size_t argmax(const vector<double>& v) { return (size_t) (std::max_element(v.begin(), v.end()) - v.begin()); }

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: glm_model_parity [--selftest]\n"); return 2; }
    }
    TestModel tm = build_test_model(MAX_CTX);
    const Glm5NextGeometry& g = tm.g;
    vector<Block>& blocks = tm.blocks;
    const Lin &tok = tm.tok, &out = tm.out, &eh = tm.eh;
    const vector<float> &out_norm = tm.out_norm, &enorm = tm.enorm, &hnorm = tm.hnorm, &head_norm = tm.head_norm;
    const glm::GlmModelWeights& w = tm.w;
    const int D = (int) g.n_embd, V = (int) g.n_vocab;
    const int64_t NL = g.n_layers, MTP = g.n_layers;

    int bad = 0;
    // the checks before anything is allocated: a weight missing, a projection of the wrong shape
    for (int c = 0; c < 2; ++c) {
        glm::GlmModelWeights w2 = w;
        if (c == 0) w2.blocks[3].exp_probs_b = nullptr;
        else w2.blocks[MTP].wkv_a.n_out = 256;
        glm::GlmModel m2;
        std::string e2;
        const bool refused = !m2.init(g, w2, MAX_CTX, e2);
        if (!refused) ++bad;
        std::printf("init with %s: %s\n", c == 0 ? "block 3's exp_probs_b missing" : "the MTP block's wkv_a 256 wide",
                    refused ? e2.c_str() : "*** accepted ***");
    }
    glm::GlmModel model;
    std::string err;
    if (!model.init(g, w, MAX_CTX, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    for (int64_t il = 0; il < g.n_blocks(); ++il) blocks[(size_t) il].st = model.state(il);
    float* taps = nullptr;
    check(cudaMalloc(&taps, (size_t) (NL + 1) * 4 * D * sizeof(float)), "taps");
    glm::GlmTaps gtaps;
    gtaps.x = taps;
    model.set_taps(&gtaps);
    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "stream");
    model.reset(stream);

    vector<int32_t> tokens(STEPS + 2);
    for (auto& t : tokens) t = (int32_t) (unit() * (float) V) % V;
    Ref r{g, {}, {}};
    const glm::GlmContext& ctx = model.context();
    auto sel_ok = [&](int64_t pos, long long il) {
        vector<int32_t> got = download(ctx.sel_ids, r.sel.size());
        std::sort(got.begin(), got.end());
        const bool ok = got == r.sel;
        if (!ok) ++bad;
        std::printf("  pos %lld block %lld: the indexer's %zu cells %s\n", (long long) pos, il, r.sel.size(),
                    ok ? "match" : "*** DIFFER ***");
    };
    auto experts_ok = [&](int64_t pos, long long il) {
        const vector<int32_t> got = download(ctx.eids, (size_t) g.n_expert_used);
        const bool ok = got == r.eids;
        if (!ok) ++bad;
        std::printf("  pos %lld block %lld: experts %d %d %d %d %s\n", (long long) pos, il, r.eids[0], r.eids[1],
                    r.eids[2], r.eids[3], ok ? "match, in order" : "*** DIFFER ***");
    };
    // the head on a device h: the norm checked against the reference's, the logits from the device's h
    auto head_ok = [&](const char* what, int64_t pos, const vector<double>& h_ref, const float* d_h,
                       const float* d_logits) {
        const vector<float> h = download(d_h, (size_t) D), lg = download(d_logits, (size_t) V);
        const double rh = rel_l1(h_ref, h);
        vector<double> lg_ref;
        const bool lg_ok = verify("logits", pos, -1, [&] {
            lg_ref = lin_ref(out, vector<double>(h.begin(), h.end()));
            return Fit{rel_l1(lg_ref, lg)};
        }, [] {});
        const size_t am = argmax(lg), am_ref = argmax(lg_ref);
        // the argmax may differ only on a near-tie
        const bool am_ok = am == am_ref || std::fabs(lg_ref[am] - lg_ref[am_ref]) < 1e-4 * std::fabs(lg_ref[am_ref]);
        if (rh > 1e-6 || !am_ok || !lg_ok) ++bad;
        std::printf("  pos %lld %s: h rel %.3e%s, argmax %zu %s\n", (long long) pos, what, rh,
                    rh <= 1e-6 ? "" : " *** over 1e-6 ***", am, am_ok ? "matches" : "*** DIFFERS ***");
    };
    // the MTP block from the device's embedding row and h: [rms(e) enorm ; rms(h) hnorm] -> eh_proj -> block -> norm
    auto mtp_ok = [&](int32_t token, int64_t pos, const vector<float>& h_in) {
        vector<double> e(tok.w.begin() + (size_t) token * D, tok.w.begin() + (size_t) (token + 1) * D);
        vector<double> hh(h_in.begin(), h_in.end());
        rms_w(e, enorm, g.rms_eps);
        rms_w(hh, hnorm, g.rms_eps);
        e.insert(e.end(), hh.begin(), hh.end());
        Block& b = blocks[(size_t) MTP];
        latent_sync(b, pos);
        const vector<float> mh = download(model.mtp_h(), (size_t) D), lg = download(model.mtp_logits(), (size_t) V);
        const Snap snap = save(b, r);
        if (!verify("MTP block's h", pos, MTP, [&] {
                vector<double> x = lin_ref(eh, e);
                mtp_block_ref(b, g, MTP, pos, x, r);
                rms_w(x, head_norm, g.rms_eps);   // mtp_h(): after the head norm
                return Fit{rel_l1(x, mh), latent_row_far(b) == 0};
            }, [&] { restore(b, r, snap); }))
            ++bad;
        if (!latent_row_ok(b, pos, MTP)) ++bad;
        sel_ok(pos, MTP);
        experts_ok(pos, MTP);
        if (!verify("MTP logits", pos, MTP,
                    [&] { return Fit{rel_l1(lin_ref(out, vector<double>(mh.begin(), mh.end())), lg)}; }, [] {}))
            ++bad;
    };

    std::printf("GlmModel: 5 layers (KDA + dense, MLA + indexer, KDA, MLA + shared indexer, KDA; MoE) and an MTP "
                "block,\nn_embd %d, vocabulary %d, %d tokens from a new sequence:\n", D, V, STEPS);
    for (int s = 0; s < STEPS; ++s) {
        const int64_t pos = s;
        if (!model.decode(tokens[(size_t) s], pos, stream, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        check(cudaStreamSynchronize(stream), "decode");
        const vector<float> tp = download(taps, (size_t) (NL + 1) * 4 * D);
        // the embedding: four bitwise copies of the row
        {
            bool same = true;
            for (int i = 0; i < 4; ++i)
                for (int d = 0; d < D; ++d)
                    same &= tp[(size_t) (i * D + d)] == tok.w[(size_t) tokens[(size_t) s] * D + d];
            if (!same) ++bad;
            std::printf("  pos %lld token %d: the four streams %s the embedding row\n", (long long) pos,
                        tokens[(size_t) s], same ? "are bitwise" : "*** are not ***");
        }
        for (int64_t il = 0; il < NL; ++il) {
            Block& b = blocks[(size_t) il];
            const bool mla = g.is_mla[(size_t) il] != 0;
            if (mla) latent_sync(b, pos);
            const vector<float> got(tp.begin() + (size_t) (il + 1) * 4 * D, tp.begin() + (size_t) (il + 2) * 4 * D);
            const KdaDev kd = mla ? KdaDev{} : kda_dev(b, g);
            const Snap snap = save(b, r);
            if (!verify("the four streams", pos, il, [&] {
                    vector<double> X(tp.begin() + (size_t) il * 4 * D, tp.begin() + (size_t) (il + 1) * 4 * D);
                    block_ref(b, g, il, pos, X, r);
                    double rh, rs;
                    return Fit{rel_l1(X, got), mla ? latent_row_far(b) == 0 : kda_close(b, kd, rh, rs)};
                }, [&] { restore(b, r, snap); }))
                ++bad;
            if (mla && !latent_row_ok(b, pos, il)) ++bad;
            if (!mla && !kda_state_ok(b, kd, pos, il)) ++bad;
        }
        sel_ok(pos, 1);   // the last full indexer before the head: block 1 (3 shares it)
        experts_ok(pos, NL - 1);
        // the head
        vector<double> hm((size_t) D, 0.0);
        for (int d = 0; d < D; ++d) {
            for (int i = 0; i < 4; ++i) hm[(size_t) d] += tp[(size_t) (NL * 4 * D + i * D + d)];
            hm[(size_t) d] /= 4;
        }
        rms_w(hm, out_norm, g.rms_eps);
        head_ok("head", pos, hm, model.h_nextn(), model.logits());
        // the MTP block on the next token
        const vector<float> h_dev = download(model.h_nextn(), (size_t) D);
        if (!model.mtp_decode(tokens[(size_t) s + 1], pos, model.h_nextn(), stream, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        check(cudaStreamSynchronize(stream), "mtp");
        mtp_ok(tokens[(size_t) s + 1], pos, h_dev);
    }
    // a draft chained on the MTP block's own h, at the next position
    {
        const int64_t pos = STEPS;
        const vector<float> h_dev = download(model.mtp_h(), (size_t) D);
        if (!model.mtp_decode(tokens[(size_t) STEPS + 1], pos, model.mtp_h(), stream, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        check(cudaStreamSynchronize(stream), "mtp chain");
        std::printf("  the chained draft (h = mtp_h()):\n");
        mtp_ok(tokens[(size_t) STEPS + 1], pos, h_dev);
    }

    std::printf("\nglm_model: %d failures\n", bad);
    for (void* p : g_free) cudaFree(p);
    cudaFree(taps);
    cudaStreamDestroy(stream);
    if (bad) return 1;
    if (selftest) std::printf("glm_model_parity OK\n");
    return 0;
}
