"""Strata's GLM-5.3-Flash logits against llama.cpp's on the same ids.

    glm_loader_parity --logits SHARD PROMPT_IDS OUT N    -> OUT.ids, OUT.logits   (Strata: prompt, then greedy decode)
    llama_logits SHARD OUT.ids OUT.llama [threads] [ubatch]   (tools/glm_compare: llama.cpp, CPU, teacher-forced)
    python tools/glm_compare_logits.py OUT [OUT ...] [--tokenizer DIR] [--llama .llama1]

Per answer position: whether both pick the same top token, the KL divergence KL(llama.cpp || Strata) of the softmax,
and the top-5 overlap; and each one's perplexity on the ids that follow (for a teacher-forced text, its perplexity;
for Strata's own greedy answer, how sure each is of it).  With --tokenizer (a Strata tokenizer folder), the answer is printed and each position where the
two disagree on the top token shows both choices.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

V = 154880


def load(prefix: str, llama_suffix: str = ".llama"):
    lines = Path(prefix + ".ids").read_text().split("\n")
    ids = [int(x) for x in lines[0].split()]
    n_prompt = int(lines[1])
    s = np.fromfile(prefix + ".logits", dtype=np.float32).reshape(-1, V)
    ll = np.fromfile(prefix + llama_suffix, dtype=np.float32).reshape(-1, V)
    rows = min(len(s), len(ll), len(ids) - n_prompt)
    return ids, n_prompt, s[:rows].astype(np.float64), ll[:rows].astype(np.float64)


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def decoder(tok_dir):
    if not tok_dir:
        return None
    vocab = json.loads((Path(tok_dir) / "vocab.json").read_text(encoding="utf-8"))
    inv = {i: t for t, i in vocab.items()}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import strata_tokenizer as ST
    byte = {u: b for b, u in ST.BYTE_TO_UNICODE.items()}

    def text(ids):
        return b"".join(bytes(byte.get(c, ord("?")) for c in inv.get(i, "")) for i in ids).decode("utf-8", "replace")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--tokenizer")
    ap.add_argument("--llama", default=".llama", help="the llama.cpp file's suffix (e.g. .llama1: a run of ubatch 1)")
    a = ap.parse_args()
    text = decoder(a.tokenizer)
    all_kl, all_same, n_all = [], 0, 0
    all_nll_s, all_nll_l = [], []
    for p in a.prefixes:
        ids, n_prompt, s, ll = load(p, a.llama)
        ls, lll = log_softmax(s), log_softmax(ll)
        kl = (np.exp(lll) * (lll - ls)).sum(axis=-1)
        top_s, top_l = s.argmax(axis=-1), ll.argmax(axis=-1)
        same = top_s == top_l
        top5 = [len(set(np.argsort(-s[r])[:5]) & set(np.argsort(-ll[r])[:5])) for r in range(len(s))]
        nxt = np.array(ids[n_prompt:n_prompt + len(s)])
        ppl_s = float(np.exp(-ls[np.arange(len(s)), nxt].mean()))
        ppl_l = float(np.exp(-lll[np.arange(len(s)), nxt].mean()))
        all_nll_s.append(-ls[np.arange(len(s)), nxt])
        all_nll_l.append(-lll[np.arange(len(s)), nxt])
        print("%s: %d answer tokens; top token the same at %d of %d; KL mean %.4f, max %.4f (at %d); top-5 overlap "
              "mean %.2f; perplexity Strata %.3f, llama.cpp %.3f" % (
                  Path(p).name, len(s), int(same.sum()), len(s), kl.mean(), kl.max(), int(kl.argmax()),
                  float(np.mean(top5)), ppl_s, ppl_l))
        if text:
            print("  answer: %r" % text(ids[n_prompt:n_prompt + len(s)]))
            for r in np.nonzero(~same)[0]:
                ps, pl = np.exp(ls[r]), np.exp(lll[r])
                print("  position %d after %r: Strata %r (p %.3f, llama.cpp %.3f), llama.cpp %r (p %.3f, Strata %.3f)" % (
                    r, text(ids[n_prompt + max(0, r - 6):n_prompt + r]), text([top_s[r]]), ps[top_s[r]], pl[top_s[r]],
                    text([top_l[r]]), pl[top_l[r]], ps[top_l[r]]))
        all_kl += list(kl)
        all_same += int(same.sum())
        n_all += len(s)
    if len(a.prefixes) > 1:
        print("all: top token the same at %d of %d (%.1f%%); KL mean %.4f, median %.4f, max %.4f; perplexity Strata "
              "%.3f, llama.cpp %.3f" % (
                  all_same, n_all, 100.0 * all_same / n_all, float(np.mean(all_kl)), float(np.median(all_kl)),
                  float(np.max(all_kl)), float(np.exp(np.concatenate(all_nll_s).mean())),
                  float(np.exp(np.concatenate(all_nll_l).mean()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
