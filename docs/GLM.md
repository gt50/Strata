# GLM-5.3-Flash (experimental)

**Experimental: the engine path is written and checked piece by piece, but the real model has not been run whole on
Strata yet.** Strata's GLM-5.3-Flash support runs [Z.ai's GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
from [Unsloth's UD-IQ4_XS GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) on its own engine path (`src/glm`):
another architecture than Qwen3.8-Flash-Next, with none of the Qwen engine's packs, memory modes or draft layer. Every
piece below was checked against a double-precision transcription of llama.cpp's `glm5-next` graph on small synthetic
models of GLM's shapes, and timed on GLM's real sizes, all on one AMD Radeon 8060S (Strix Halo, gfx1151, Windows).
Back to [Which model?](MODELS.md).

> **On this page:** [Status](#status) · [The model](#the-model) · [Install](#install) ·
> [Engine options](#engine-options) · [Chat, thinking and tools](#chat-thinking-and-tools) ·
> [How the engine runs it](#how-the-engine-runs-it) · [Measured](#measured) · [Tests](#tests) · [Not done yet](#not-done-yet)

## Status

| | |
| --- | --- |
| **Runs the real file** | not yet: the 157 GB download has not been run whole; only its metadata shard was read (geometry, tokenizer, chat template) |
| **GPUs** | AMD, built with HIP and run on a Radeon 8060S (gfx1151) only. The CUDA build of the GLM kernels has not been compiled; Intel (SYCL) has no GLM path |
| **Engine** | not in a ready-made release: build it from a checkout that has `src/glm` (below) |
| **Setup** | `--family glm` (one size, UD-IQ4_XS), marked experimental |
| **Server** | chat, thinking levels, tool calls (streamed too), the OpenAI and Anthropic APIs; no images |
| **One GPU** | no layer split, no batch slots (one request at a time), no calibration |

## The model

From the GGUF's metadata (`glm5-next.*`, Unsloth revision `a38483c8`):

| | |
| --- | --- |
| Layers | 46: 3 dense, then 43 MoE; 34 KDA (linear attention) and 12 MLA (layers 3, 7, ..., 43 and 45) |
| Experts | 288 per MoE layer, 8 used per token, 1 shared; 4096 x 2048 each |
| Width | 4096 (hidden), 64 attention heads; four residual streams mixed by mHC (20 Sinkhorn rounds) |
| MLA | latent 512 per token and layer (q rank 1536), no rotary part; a k-pool indexer (32 heads, pools of 4 cells, the best 512 pools) chooses the rows each query reads |
| KDA | 64 heads of 128, short convolution of 4 |
| Draft layer | 1 MTP block in the file (`nextn_predict_layers`) |
| Context | 1,048,576 tokens (trained) |
| Vocabulary | 154,880 tokens, pre-tokenizer `glm4` |
| License | MIT (`general.license`) |

The UD-IQ4_XS file is five shards, 156.8 GB together; shard 1 (9.4 MB) holds only metadata. Setup pins the five sizes
and SHA-256 from the Hub's LFS pointers (`GLM_SHARDS` in `setup.py`).

## Install

```
START-HERE.bat --setup --family glm
```

Linux: `./setup.sh --setup --family glm`. Setup:

1. Asks only the context: 8K to 256K are offered, 32K recommended (the engine's default). `--context` takes up to
   1,048,576. Strata's GLM tests use a few thousand tokens, so long contexts are untested.
2. Checks the engine before the download: the GLM path must be in it (`setup.glm_engine` looks for the GLM start line's
   text in the binary, because a build with the GLM path and a release without it can carry the same version). Today's
   ready-made engines do not have it. Build one: on Windows with AMD, `tools\hip\build_windows.bat` (it builds for
   gfx1151 among others, [AMD_HIP.md](AMD_HIP.md)), then `START-HERE.bat --setup --family glm --backend hip
   --prebuilt dist\`; on Linux with AMD, setup compiles the engine from the checkout itself.
3. Downloads the five shards (157 GB; files already in another Strata folder or given with `--gguf-dir` are reused)
   and checks their sizes and SHA-256.
4. Extracts the tokenizer from shard 1 into `Strata-data/packs/glm-ud-iq4_xs/tokenizer/`, with a round trip over a
   test corpus. Nothing else is prepared: the engine reads the GGUF as it is.
5. Writes `strata-glm-ud-iq4_xs.json` (`--native <shard 1> --max-context N`, `"arch": "glm5-next"`) and
   `run-glm-ud-iq4_xs.bat` / `.sh`.

Setup options made for the Qwen models (`--kv`, `--vision`, `--draft-vocab`, `--parallel`, `--low-ram`,
`--resident-budget-gib`, `--kv-streaming`, `--vram-reserve-mib`, ...) are said and left out.

**Memory.** The engine puts the weights other than the routed experts on the GPU (their size is not known yet: shard 1
has no tensors), then all the experts if they fit the free GPU memory beside them and 3 GB of room, else as many
cache slots as fit (at least 64). Experts not in a slot are copied from the mapped model files when a token chooses
them, so the OS file cache and the SSD carry the rest. How fast that is with the real file is not measured.

## Engine options

The config's `args`; the server adds `--serve`.

| Option | Default | |
| --- | --- | --- |
| `--native <shard 1>` | | the model; the other shards are found beside it |
| `--max-context N` | 32768 | the context (the MLA cache is sized for it) |
| `--spec N` | 0 | MTP drafts per round (speculative decoding). Off by default: not measured on the real model, and a window reads more experts, which costs more when they come from the SSD |
| `--glm-expert-cache N` | auto | `0`: every expert resident; `N`: stream them through N slots; auto: resident when they fit (above) |
| `--glm-experts-host` | off | the experts (or their slots) in mapped host memory instead of device memory |
| `--glm-chunk N` | 256 | prompt tokens per pass (at least 16) |
| `--eos-ids a,b,c` | the file's | end tokens; the file's eos, eot and eom: `<|endoftext|>`, `<|user|>`, `<|observation|>` |

Other engine flags are named on stderr and skipped. The engine speaks the same line protocol as the Qwen engine
(`INFO`, `READY`, `GEN` with the sampling keys, `T`, `DONE`, `STOP`, `QUIT`), so `serve/server.py` runs it unchanged.

## Chat, thinking and tools

The server recognizes GLM's chat template (it starts with `[gMASK]`) and follows its format:

- **Thinking level.** GLM's template writes `Reasoning Effort: Low | High | Max` in the system turn. The API's
  `reasoning_effort` maps `low` to Low, `medium` to High, `high` (and `xhigh`, `max`) to Max; without one the template
  uses its default, Max. Thinking off (`"none"`) closes the `<think>` block before the answer, since the template has
  no switch for it.
- **Tool calls.** GLM writes `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`; the server
  turns it into the form its parser reads, so streamed arguments, typed values and `tool_choice` work as with the Qwen
  models. Tool results go back as `<|observation|>` turns.
- **End tokens.** A reply ends at `<|endoftext|>`, `<|user|>` or `<|observation|>`.
- **Tokenizer.** The `glm4` pre-tokenizer is llama.cpp's Llama 3 expression; Strata's tokenizer gives llama.cpp's ids on
  all 47 strings of its Llama 3 test, and every special token in GLM's template is one token.
- Qwen-only server features stay off for GLM: the effort turn at the end (`effort_position`) and reasoning-loop
  recovery. Images are not wired.

## How the engine runs it

One decode token, in llama.cpp's `glm5-next` order: the embedding into four streams; per layer mHC's mix (a GEMV, the
Sinkhorn, the input `h` and the norm in one finalize kernel), KDA or MLA, mHC's merge, again for the FFN (dense,
or router + experts + shared expert); the streams' mean, the output norm and the head. Then the MTP block for a draft.

| Part | Files | |
| --- | --- | --- |
| KDA | `kernels/cuda/kda.cu` | convolution + L2 norm, gate, the state scan and output norm; state (64, 128, 128) per layer |
| MLA | `kernels/cuda/mla.cu` | absorbed projections (BF16), FP16 latent cache, split-K decode attention, online-softmax prompt attention |
| Indexer | `kernels/cuda/kpool.cu` | key store, pooling and scores; the selection by QSA's `qsa_block_topk` |
| mHC | `kernels/cuda/mhc.cu` | pre (mix, Sinkhorn, h, the sublayer's norm), post, init, mean |
| MoE | `kernels/cuda/glm_moe.cu`, `glm_linear.cu` | sigmoid router with the selection bias, the clamped SwiGLU, grouped GGUF experts |
| Model | `src/glm/glm_layer.cpp`, `glm_model.cpp` | a block and the model loop, the MTP block |
| Prompts | `src/glm/glm_gemm.cpp` | projections and experts as matrix products (llama.cpp's MMQ, BF16 GEMM) |
| Loading | `src/glm/glm_loader.cpp`, `glm_expert_cache.cpp` | the plan from the tensor directory, the upload, the expert slots |
| Drafts | `src/glm/glm_spec.cpp` | verify windows bitwise decode's, the commit, sampling keyed by position |
| Server | `src/glm/glm_serve.cpp` | the line protocol |

Speculative decoding keeps a draft when it equals the token its window row picks; each position's draw is
Philox(seed, position), so sampled output is token for token the plain sampled run's.

## Measured

Radeon 8060S, Windows, HIP, timed on the host clock (HIP's events close early on this host). The model is a synthetic
one of GLM-5.3-Flash's shapes: 45 layers, the vocabulary of 154,880, 8 resident IQ4_XS experts per layer that every
token uses. So these are kernel times, not the real model's speed: real tokens choose 8 of 288 experts, and most of them
come from slots or the SSD.

| | |
| --- | --- |
| Decode, one token | 50.7-52.8 ms (73 ms before the kernel work below); the step reads 9.6 GB of weights |
| Decode kernels | KDA scan 59 -> 45 us a layer; mHC finalize 10 -> 5.4 us and mix 11 -> 10 us a sublayer; KDA's BF16 f_b / g_b 14 -> 10.5 us; norms fused into mHC (90 launches of 28-30 us gone) |
| Verify windows | 2 / 4 / 8 tokens: 55.4 / 67.5 / 100.7 ms (best case: all tokens share their experts); commit 2-3 ms |
| Prompt products | 4096 -> 8192 IQ4_XS: 0.86 ms at 256 tokens (4.6x the GEMV path), 3.13 ms at 1024 (5.2x) |
| One MoE layer, prompt | 288 IQ4_XS experts: 22.7 ms at 256 tokens, 39.0 ms at 1024 (1.85x / 3.3x the GEMV path) |
| Expert slots | 8 misses copied from a warm file: 3.6 ms into mapped host slots (30 GB/s), 8.1 ms into device slots (13 GB/s); the expert kernel 220 GB/s from device slots, 190 GB/s from mapped host; a layer with every expert cached 29-33 us |
| MLA prompt attention | 2048 queries over 2051 selected rows: 215 ms a layer (about 2.7 TFLOP/s in FP32) |
| Indexer, one query | scores + selection 88 us at 32K cells, 213 us at 128K, 2.2 ms at 1M |

## Tests

The GPU tests (ctest, HIP build; each `--selftest`): `kda_parity`, `mla_parity`, `kpool_parity`, `glm_moe_parity`,
`mhc_parity` for the kernels; `glm_layer_parity`, `glm_model_parity`, `glm_prompt_parity`, `glm_loader_parity`,
`glm_graph_parity`, `glm_spec_parity` and `glm_serve_parity` for the model on synthetic weights against
`src/glm/glm_ref.hpp`. Most have `--bench`. `glm_serve_parity --write-gguf PATH` writes the synthetic model as a GGUF
to run `strata --native PATH --serve` on.

Without a GPU:

```
python -m unittest serve.test_glm_dialect tools.test_strata_tokenizer_glm4
python tools/test_setup_glm.py
```

`tools/test_strata_tokenizer_glm4.py` also runs llama.cpp's Llama 3 tokenizer test (with a llama.cpp checkout in
`third_party/llama.cpp` or `$STRATA_LLAMA_CPP`) and checks the real tokenizer when `$STRATA_GLM_GGUF` names shard 1.

## Not done yet

- A run of the real model: speed, expert hit rates, quality against llama.cpp on the same file.
- The CUDA build of the GLM kernels, and other AMD architectures than gfx1151.
- Images, batch slots, prompt reuse between requests (each request reads its whole prompt), `logit_bias`, progress
  lines while a prompt is read.
- Several GPUs, calibration, a ready-made engine with the GLM path.
