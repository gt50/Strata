# GLM-5.3-Flash (experimental)

**Experimental: the real model runs and answers correctly on one PC, but slowly (2.2-3 tokens per second, 4-12 s
before a short prompt's first token) while many of its experts come from the SSD.** Strata's GLM-5.3-Flash support runs [Z.ai's GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
from [Unsloth's UD-IQ4_XS GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) on its own engine path (`src/glm`):
another architecture than Qwen3.8-Flash-Next, with none of the Qwen engine's packs, memory modes or draft layer. Every
piece below was checked against a double-precision transcription of llama.cpp's `glm5-next` graph on small synthetic
models of GLM's shapes, and timed on GLM's real sizes, all on one AMD Radeon 8060S (Strix Halo, gfx1151, Windows);
the whole file was then run there ([the first run](#the-first-run-of-the-real-file)). Back to [Which model?](MODELS.md).

> **On this page:** [Status](#status) · [The model](#the-model) · [Install](#install) ·
> [Engine options](#engine-options) · [Chat, thinking and tools](#chat-thinking-and-tools) ·
> [How the engine runs it](#how-the-engine-runs-it) · [Measured](#measured) · [Tests](#tests) · [Not done yet](#not-done-yet)

## Status

| | |
| --- | --- |
| **Runs the real file** | yes, on one PC (Radeon 8060S, 64 GB of RAM and a 64 GB GPU carve-out): correct answers, thinking off, a streamed tool call; 2.2-3.0 tokens/s, 4-12 s for a 20-token prompt and 94 s for a 3,712-token one, bound by reading experts from the SSD; its next-token choices and perplexity match llama.cpp's on the same file ([below](#against-llamacpp-on-the-same-file)). UD-IQ2_XXS (102 GB) fits there whole: 16-24 tokens/s and 0.6-1.0 s for a short prompt, for 6% higher perplexity ([below](#ud-iq2_xxs-the-experts-resident)) |
| **GPUs** | AMD, built with HIP and run on a Radeon 8060S (gfx1151) only. The CUDA build of the GLM kernels has not been compiled; Intel (SYCL) has no GLM path |
| **Engine** | not in a ready-made release: build it from a checkout that has `src/glm` (below) |
| **Setup** | `--family glm` (one size, UD-IQ4_XS), marked experimental |
| **Server** | chat, thinking levels, tool calls (streamed too), the OpenAI and Anthropic APIs; no images |
| **One GPU** | no layer split, no batch slots (one request at a time), no calibration |

## The model

From the GGUF's metadata (`glm5-next.*`, Unsloth revision `a38483c8`):

| | |
| --- | --- |
| Layers | 45 (`block_count` 46 less the MTP block): 3 dense, then 42 MoE; 34 KDA (linear attention) and 11 MLA (layers 3, 7, ..., 43) |
| Experts | 288 per MoE layer, 8 used per token, 1 shared; 4096 x 2048 each |
| Width | 4096 (hidden), 64 attention heads; four residual streams mixed by mHC (20 Sinkhorn rounds) |
| MLA | latent 512 per token and layer (q rank 1536), no rotary part; a k-pool indexer (32 heads, pools of 4 cells, the best 512 pools) chooses the rows each query reads |
| KDA | 64 heads of 128, short convolution of 4 |
| Draft layer | 1 MTP block in the file (`nextn_predict_layers`), as layer 45: MLA and MoE, no mHC |
| Context | 1,048,576 tokens (trained) |
| Vocabulary | 154,880 tokens, pre-tokenizer `glm4` |
| License | MIT (`general.license`) |

The UD-IQ4_XS file is five shards, 156.8 GB together; shard 1 (9.4 MB) holds only metadata. Setup pins the five sizes
and SHA-256 from the Hub's LFS pointers (`GLM_SHARDS` in `setup.py`).

**UD-IQ2_XXS** (four shards, 101.8 GB) runs too: 7.0 GB of weights besides the experts and 95.2 GB of experts,
IQ2_XXS gate / up over IQ3_XXS down in 39 of the 42 MoE layers, IQ2_S / IQ4_XS in layer 11, IQ2_XXS / IQ4_XS in 12 and
44, Q2_K / Q3_K in the MTP block (`glm_moe_parity` section 4 checks those pairs). On the 8060S all its experts fit, so
nothing is read from the SSD: about 8x the speed of UD-IQ4_XS, for 6% higher perplexity
([below](#ud-iq2_xxs-the-experts-resident)). Setup does not offer it yet.

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

**Memory.** The engine puts the 10 GB of weights other than the routed experts on the GPU, then all 147 GB of
experts if they fit the free GPU memory beside them and 3 GB of room, else as many cache slots of 15.8 MB as fit (at
least 64); experts not in a slot are read from the model files when a token chooses them. On an integrated GPU (an
APU) the runtime's free figure counts the system RAM the GPU can share (the 8060S with 64 GB carved out of 128
reports 115 GB free), and memory past the carve-out is taken from Windows' RAM. So there the engine measures the room
at start: it allocates and writes 4 GB pieces until the RAM Windows has available would fall below the reserve
(`--glm-host-reserve-gb`, 12 GB, plus 3 GB of room for the context and 4 GB for the engine's own host memory), then
frees them and sizes the slots to that: on the 8060S 90.2 GB, 4,874 slots, and 12.3 GB of RAM left available once
loaded. After loading, the engine trims its working set (Windows), so the dense weights' file pages count as
available again.

## Engine options

The config's `args`; the server adds `--serve`.

| Option | Default | |
| --- | --- | --- |
| `--native <shard 1>` | | the model; the other shards are found beside it |
| `--max-context N` | 32768 | the context (the MLA cache is sized for it) |
| `--spec N` | 0 | MTP drafts per round (speculative decoding). Off by default: not measured on the real model, and a window reads more experts, which costs more when they come from the SSD |
| `--glm-expert-cache N` | auto | `0`: every expert resident; `N`: stream them through N slots; auto: resident when they fit (above) |
| `--glm-experts-host` | off | the experts (or their slots) in mapped host memory instead of device memory |
| `--glm-host-reserve-gb N` | 12 | an integrated GPU: the RAM the automatic slot count leaves Windows (or Linux) available |
| `--glm-chunk N` | 256 | prompt tokens per pass (at least 16) |
| `--glm-prompt-group N` | 8192 | with streamed experts, up to N prompt tokens run layer by layer, each layer's experts read once for them ([below](#long-prompts-layer-by-layer)); `0`: chunk by chunk |
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

## The first run of the real file

Radeon 8060S (Strix Halo, 128 GB: 64 GB for Windows, 64 GB carved out for the GPU), Windows, the model on a Samsung
990 EVO Plus NVMe, `--max-context 8192`, `--glm-expert-cache 3000` (3,000 slots of 15.8 MB, 47 GB: with the 10 GB of
other weights, inside the carve-out), greedy, through `serve/server.py`:

- **Loading:** the plan reads 1,412 tensors: 10.0 GB of weights besides the routed experts and 147.1 GB of experts.
  Unsloth's file stores the mHC matrices (`hc_*_fn`) in Q8_0, dequantized to F32 at load, and some expert layers in
  types the grouped kernels gained for it: Q6_K downs (layers 11, 12, 44), and the MTP block's Q3_K gate/up and Q4_K
  down. Loaded in 45.7 s.
- **Answers:** "What is the capital of France?" (effort low): "The capital of France is Paris."; a haiku with
  thinking off; a tool call streamed as `get_weather` with `{"city": "Paris", "days": 2}` (the number typed).
- **Speed:** almost all the time goes to reading missed experts from the file into slots. First through the file's
  mapping (each 4 KB page faulted in), at 0.45-0.9 GB/s:

| Request | Prompt | Its experts | Answer | Its experts |
| --- | --- | --- | --- | --- |
| 24 tokens, cold start | 61.9 s | 39 hits, 3,590 misses (42.8 GB) | 9 tokens, 11.3 s | 1,900 hits, 788 misses (9.4 GB) |
| 21 tokens | 34.8 s | 640 hits, 2,667 misses (31.8 GB) | 22 tokens, 39.3 s | 5,035 hits, 2,021 misses (24.1 GB) |
| 24 tokens, again | 80.8 s | 607 hits, 3,022 misses (36.0 GB) | 9 tokens, 17.7 s | 1,892 hits, 796 misses (9.5 GB) |
| 180 tokens (a tool) | 229.1 s | 435 hits, 9,068 misses (107.8 GB) | 18 tokens, 58.3 s | 3,322 hits, 2,390 misses (28.4 GB) |

Then with unbuffered reads (the default now: `platform::DirectFile`, each expert's three slabs into pinned staging
buffers, up to 48 reads in flight, then to the slot on a copy stream), the same requests and the same experts, at
2.3-2.6 GB/s; loading (filling the slots) 45.7 -> 20-26 s:

| Request | Prompt | Answer | Through the mapping |
| --- | --- | --- | --- |
| 24 tokens, cold start | 17.2 s | 9 tokens, 4.6 s (1.92 tokens/s) | 61.9 s, 0.76 tokens/s |
| 21 tokens | 13.4 s | 22 tokens, 12.4 s (1.69 tokens/s) | 34.8 s, 0.53 tokens/s |
| 24 tokens, again | 14.2 s | 9 tokens, 4.7 s (1.90 tokens/s) | 80.8 s, 0.49 tokens/s |
| 180 tokens (a tool) | 43.8 s | 18 tokens, 13.7 s (1.86 tokens/s) | 229.1 s, 0.44 tokens/s |

The drive reads 3.6-4.0 GB/s unbuffered at this read size with 8-32 reads in flight (1.2 GB/s with one), so the
cache gets about two thirds of it; `STRATA_IO_THREADS` 8 or 16 changed nothing. A token of an answer misses about 88
experts (1 GB), about 2 per layer, one layer after another: the reads wait for each layer's router. A 24-token prompt
touches about 3,600 different experts, more than the 3,000 slots, so the same prompt again is no faster. The engine
prints these lines per request in its log; `STRATA_GLM_MAPPED_READS=1` reads through the mapping as before.

**Prefetching experts (opt-in, off by default: slower here).** With `STRATA_GLM_PREFETCH=1`, a decode step guesses a
later block's experts (that block's router on this block's FFN input) and a thread of its own reads the guessed ones
that are not cached while this block computes; `STRATA_GLM_PREFETCH=measure` only counts the guesses, and
`STRATA_GLM_PREFETCH_AHEAD=N` guesses N blocks on. Same PC, same requests (24-token and 21-token prompts, 9 and 22
answer tokens):

| | Guesses right | An answer's misses (9 tokens) | Bytes read (9 tokens) | Decode |
| --- | --- | --- | --- | --- |
| off | - | 788 | 9.4 GB | 1.60-1.92 tokens/s |
| 1 block on | 75% | 316 + 772 read ahead (518 then hit) | 13.0 GB | 1.40 tokens/s |
| 2 blocks on | 64-68% | 435 + 767 read ahead (415 then hit) | 14.3 GB | 1.35 tokens/s |
| 4 blocks on | 55-59% | 552 + 710 read ahead (303 then hit) | 15.1 GB | 1.31 tokens/s |

The guesses are good, but every wrong one is a read, and it pushes a cached expert out; the drive is already the
limit for these scattered reads, so 38-60% more bytes cost more time than the overlap saves. `glm_loader_parity
--bench-file SHARD` times the cache's reads on a real file alone: 2.5-5.4 GB/s in rounds of 64 experts of one layer
(it differs by layer, within one shard), 2.4 GB/s in decode's pattern (8 experts a layer, layer after layer). The
default speed also varies from run to run with the drive: 4.0-4.5 s to read the same 9.4 GB.

**More slots.** On the APU the slots can go past the carve-out into Windows' RAM, which the unbuffered reads leave
otherwise idle (no file cache to feed). The same requests, by slot count (`--glm-expert-cache N`; the last row the
automatic count above), the RAM Windows had available once loaded:

| Slots | RAM available | 24-token prompt | 21-token prompt | Decode (9 / 22 tokens) | An answer's misses (9 tokens) |
| --- | --- | --- | --- | --- | --- |
| 3,000 (47 GB) | 31.3 GB | 17.2 s | 13.5 s | 1.83 / 1.73 tokens/s | 788 |
| 3,800 (60 GB) | 23.4 GB | 15.7 s | 13.1 s | 2.64 / 1.96 tokens/s | 515 |
| 4,600 (73 GB) | 11.0 GB | 12.7 s | 4.2 s | 2.91 / 2.31 tokens/s | 462 |
| 4,874 (77 GB, automatic) | 12.3 GB | 11.9 s | 4.2 s | 2.95 / 2.39 tokens/s | 450 |

The 180-token prompt with a tool: 39.2 s, then 2.25 tokens/s (43.8 s and 1.86 with 3,000 slots). The fill reads the
slots' experts at start (an equal share of every layer), so loading takes 39-46 s instead of 20-26.

**Which experts the slots keep.** `STRATA_GLM_EXPERT_TRACE=FILE` has the engine write every expert-cache call (its
block, decode or prompt round, the ids), and `tools/glm_trace_replay.py` replays a trace through slot policies. Two
traces of six short requests each (code, arithmetic, a translation, Chinese, an explanation; 60 answer tokens, thinking
off) on the 8060S with 4,874 slots; trace B replayed, A as the profile:

| Policy | Decode misses | Prompt misses |
| --- | --- | --- |
| one LRU over all slots, an equal fill (the cache) | 18.3% | 60.0% |
| the same, without the MTP block over prompts | 17.8% | 57.2% |
| an LRU per layer, equal shares | 19.6% | 48.6% |
| an LRU per layer, shares sized from trace A | 17.6% | 48.3% |
| filled with trace A's most used experts, then LRU | 25.9% overall | (26.4% overall now) |
| Belady's optimum (knows the future: the bound) | 8.3% | 29.8% |

The experts are used broadly: trace A's 4,874 most used cover 59% of trace B's lookups. No policy that does not know
the future comes close to the bound for decode; per-layer shares help prompts most. One change was free: the engine ran
the MTP block over every prompt even without drafts (`--spec 0`), reading its experts for nothing - it now does so only
with drafts. Measured on trace B's requests: prompt misses 14,943 -> 14,248, prompt time about 5% less, an answer's
misses 18,880 -> 18,364.

### Long prompts, layer by layer

A prompt runs 256 tokens a pass (`--glm-chunk`), and chunk by chunk each pass went through all 45 layers. A pass of
256 tokens uses most of every layer's experts, more than the slots hold, so each pass read most of the 147 GB from
the SSD again. With streamed experts the engine now runs up to `--glm-prompt-group` tokens (8,192) layer by layer:
every chunk through layer 0, then every chunk through layer 1, and so on, keeping the chunks' four streams and their
selections between layers. The calls are the same, in another order, so the logits are the same bit for bit
(`glm_prompt_parity` scenario E; on the real file every run below wrote the same logits as chunk by chunk). On the
8060S with 4,874 slots, the prompt alone (`glm_loader_parity --logits`, a cold start each):

| Prompt | Chunk by chunk | Groups of 1,024 | Groups of 4,096 | Groups of 8,192 |
| --- | --- | --- | --- | --- |
| 3,712 tokens (docs/DETAILS.md) | 713.0 s | 245.8 s | 94.3 s | one group, as 4,096 |
| 8,000 tokens (serve/server.py) | | | 193.7 s | 140.2 s |

The times fit about 55 s a group (the experts read once) and 10.7 ms a token of compute, one after the other.
llama.cpp's CPU build took 270-280 s for 4,096 tokens, loading included ([below](#against-llamacpp-on-the-same-file)).
Through the server (`serve/server.py`, a 1,491-token text and "summarize it", thinking off, 4,564-4,602
automatic slots), the first token came after 289.6 s chunk by chunk (59,532 misses, 707.9 GB read) and 69.0 s layer by
layer (11,647 misses, 138.4 GB); the same answer of 192 tokens at 2.0-2.1 tokens/s either way.
A group takes 72 KB of device memory a token (590 MB at 8,192), from the slots' room. `STRATA_GLM_PROMPT_GROUP=N`
sets it for `glm_loader_parity --logits`.

### UD-IQ2_XXS: the experts resident

Unsloth's 2-bit file (101.8 GB, pinned revision `a38483c8`, SHA-256 checked) on the same PC. Its 7.0 + 95.2 GB and
the 3 GB of room fit the 115 GB the runtime reports free, so the engine loads every expert to the GPU, past the 64 GB
carve-out into Windows' RAM, which kept 7.4-8.6 GB available (the 12 GB `--glm-host-reserve-gb` is
for the slot count; a model that fits is not held to it). The same requests through the server, one session, both
files with the automatic setting (`serve/server.py`, greedy, thinking off but the first):

| Request | UD-IQ4_XS, 5,108 slots | UD-IQ2_XXS, all resident |
| --- | --- | --- |
| Loading | 64.0 s | 133.3 s |
| 24 tokens ("the capital of France", effort low), 9-token answer | first token 16.2 s, 2.31 tokens/s | 0.8 s, 22.10 tokens/s |
| 21 tokens (a haiku) | 5.3 s, 1.95 tokens/s | 0.6 s, 19.15 tokens/s |
| 180 tokens (a tool call) | 57.0 s, 1.79 tokens/s | 3.3 s, 23.63 tokens/s |
| 1,491 tokens ("summarize the text") | 86.8 s, 1.79 tokens/s | 23.2 s, 16.43 tokens/s |
| trace B's six requests, 60 tokens each | 6.1-23.5 s, 1.80-2.62 tokens/s | 0.6-1.0 s, 16.51-18.71 tokens/s |

The answers are right in both, as far as 60 tokens go (Paris; the tool call `get_weather`; German, Chinese, the
arithmetic, code); the wording differs where the two files' logits do. The quality cost, measured as above on the three 450-token texts (P =
64, 1,158 positions):

| | English | Python | Chinese | All |
| --- | --- | --- | --- | --- |
| Perplexity, UD-IQ4_XS (Strata) | 11.896 | 5.289 | 13.483 | 9.467 |
| Perplexity, UD-IQ2_XXS (Strata) | 12.279 | 5.423 | 15.202 | 10.041 |
| Perplexity, UD-IQ2_XXS (llama.cpp, one batch) | 12.235 | 5.498 | 15.600 | 10.162 |

UD-IQ2_XXS is 6.1% higher in perplexity over all three (Python 2.5%, English 3.2%, Chinese 12.8%); its top token is
UD-IQ4_XS's at 890 of 1,158 positions (76.9%), KL 0.222 on average - ten times the distance between Strata and
llama.cpp on one file. That distance on this file:

| UD-IQ2_XXS, 1,158 positions | Same top token | KL mean | KL median | KL max |
| --- | --- | --- | --- | --- |
| Strata against llama.cpp, one batch | 1,054 (91.0%) | 0.0274 | 0.0149 | 0.843 |
| Strata against llama.cpp, one token at a time | 1,054 (91.0%) | 0.0263 | 0.0150 | 2.218 |
| llama.cpp one token at a time against one batch | 1,070 (92.4%) | 0.0279 | 0.0142 | 2.694 |

Wider than on UD-IQ4_XS for llama.cpp's own two paths as much as for Strata: the 2-bit weights magnify rounding. Strata
decoded each text's 386 tokens in 24.2 s (UD-IQ4_XS: 195-238 s); llama.cpp's CPU build took 83-85 s a text as one
batch and 289-305 s one token at a time.

## Against llama.cpp on the same file

Upstream llama.cpp (commit 1b70350, which has `glm5-next`; `third_party/llama.cpp` predates it), built for the CPU
- the reference, no GPU kernels - against Strata on the same UD-IQ4_XS file and the same token ids:

1. `glm_loader_parity --logits SHARD PROMPT OUT 48` runs a prompt as serving does (the prompt path, then 48 greedy
   tokens through decode) and writes the ids and every answer position's logits.
2. `tools/glm_compare/llama_logits` (build it with `tools/glm_compare/CMakeLists.txt` against a llama.cpp checkout)
   runs the same ids teacher-forced: the whole sequence as one batch, or with `ubatch` 1 one token at a time, as
   decoding does.
3. `tools/glm_compare_logits.py` compares the rows: the same top token, KL(llama.cpp || Strata), the top-5 overlap.

Four prompts, thinking off (the capital of France, a French translation, three sentences on the Great Wall in Chinese,
a SQL query), 152 answer positions:

| | Same top token | KL mean | KL median | KL max |
| --- | --- | --- | --- | --- |
| Strata against llama.cpp, one token at a time | 151 of 152 (99.3%) | 0.0041 | 0.0001 | 0.132 |
| Strata against llama.cpp, one batch | 149 of 152 (98.0%) | 0.0101 | 0.0000 | 0.826 |
| llama.cpp one token at a time against llama.cpp one batch | 149 of 152 (98.0%) | 0.0103 | 0.0001 | 0.843 |

So Strata differs from llama.cpp's batch exactly as much as llama.cpp's own two paths differ from each other, and is
closer than that to llama.cpp's decoding. The differences are at near ties (the one disagreement left: a Chinese word,
0.289 against 0.264) and at one knife-edge position: after `GROUP BY c.id, c.name` in the SQL answer the model picks
a second `GROUP` - Strata decoding 0.536, llama.cpp decoding 0.745, while llama.cpp's batch puts `JOIN` first (0.572)
and Strata's prompt path `GROUP` (0.564). The doubled `GROUP BY` that answer shows is the model's own choice when
decoding, not Strata's.

**Perplexity over text.** Given a file whose second line is a prompt length P (the `.ids` format), `--logits`
teacher-forces instead: the first P ids through the prompt path, every later id through decode whatever the model
would have chosen, a row of logits before each. Three texts of 450 tokens from this repository, each `[gMASK]<sop>`
and then English prose (README.md), Python (tools/glm_trace_replay.py) and Chinese prose (README.zh-CN.md), P = 64,
1,158 predicted positions:

| | English | Python | Chinese | All |
| --- | --- | --- | --- | --- |
| Perplexity, Strata | 11.896 | 5.289 | 13.483 | 9.467 |
| Perplexity, llama.cpp one batch | 11.975 | 5.322 | 13.575 | 9.529 |
| Perplexity, llama.cpp one token at a time | 12.085 | 5.303 | 13.509 | 9.531 |

| | Same top token | KL mean | KL median | KL max |
| --- | --- | --- | --- | --- |
| Strata against llama.cpp, one batch | 1,073 of 1,158 (92.7%) | 0.0204 | 0.0103 | 1.763 |
| Strata against llama.cpp, one token at a time | 1,065 of 1,158 (92.0%) | 0.0222 | 0.0108 | 1.687 |
| llama.cpp one token at a time against one batch | 1,073 of 1,158 (92.7%) | 0.0202 | 0.0102 | 0.468 |

On text the next token is often uncertain, so the top token differs more often than in the answers above - for
llama.cpp's own two paths as much as for Strata. One position is further apart than llama.cpp's two paths ever are:
the digit after "Q2_0 用的是引擎 " in the Chinese README (a version number), where Strata puts `2` first (0.374) and
llama.cpp `57` (0.249); 99% of positions are under KL 0.14.

**Past the indexer's 2,048 cells.** Two texts of 4,096 tokens, English prose (docs/DETAILS.md) and Python
(serve/server.py), P = 3,712: each of the 768 scored queries has 3,712 to 4,095 cells and reads 2,048 of them (the best
512 pools of 4) and the tail. llama.cpp's `glm5-next` makes the same selection. `STRATA_GLM_TOP_K=8192` makes
Strata read every cell instead, to see whether the comparison notices the selection at all:

| | Same top token | KL mean | KL median | KL max | Perplexity English / Python / both |
| --- | --- | --- | --- | --- | --- |
| Strata against llama.cpp, one batch | 726 of 768 (94.5%) | 0.0184 | 0.0101 | 0.456 | 14.203 / 3.895 / 7.438 |
| Strata reading every cell, against llama.cpp | 710 of 768 (92.4%) | 0.0248 | 0.0145 | 0.594 | 14.258 / 3.830 / 7.390 |
| llama.cpp, one batch | | | | | 14.260 / 3.932 / 7.488 |

Past 2,048 cells Strata is as close to llama.cpp as on the short texts. Reading every cell moves it away: with the
selection Strata is closer to llama.cpp at 519 of 768 positions, by KL 0.0065 on average (95% bootstrap interval
0.0040-0.0089). So the comparison sees the selection, and Strata's matches llama.cpp's. Reading every cell is not
better text prediction either: lower perplexity on the Python, higher on the English.

Each text took Strata 934-943 s on the 8060S after loading: about 720 s for the 3,712-token prompt chunk by chunk
(each 256-token chunk reading most of the experts from the SSD again), then 1.8 tokens/s through decode. llama.cpp's
CPU build took 270-280 s for the whole 4,096 as one batch, loading included, reading each expert once. Layer by layer
([above](#long-prompts-layer-by-layer)) the same prompt takes 94.3 s, and the English text's 384 rows of logits are
byte for byte the ones in the table.

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

- Per-layer slot shares (prompt misses 60% -> 48% in the replay above, decode 18.3% -> 17.6%): they need a
  per-layer table measured on the model; the shares were not learned well online.
- Quality with thinking on, and on contexts much longer than 4,096 tokens (where the selection keeps a smaller share).
- Reading the next layer's experts while a long prompt's layer computes: now about 55 s of reads a group and 10.7 ms
  a token of compute, one after the other.
- The CUDA build of the GLM kernels, and other AMD architectures than gfx1151.
- Images, batch slots, prompt reuse between requests (each request reads its whole prompt), `logit_bias`, progress
  lines while a prompt is read.
- Several GPUs, calibration, a ready-made engine with the GLM path.
- UD-IQ2_XXS in setup (`--family glm` offers UD-IQ4_XS only); on an integrated GPU, whether a model whose experts all
  fit should still keep `--glm-host-reserve-gb` available (it streams then, and is slower).
