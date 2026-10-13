"""GLM-5.3-Flash's tokenizer (pre-tokenizer `glm4`): the pattern table, and two checks against outside truth.

    python -m unittest tools.test_strata_tokenizer_glm4

1. llama.cpp's own test of the same expression: GLM4's regex is llama.cpp's LLAMA3 one (`llama-bpe`), so the
   tokenizer runs over llama.cpp's llama-bpe vocabulary and test strings, models/ggml-vocab-llama-bpe.gguf.inp, and
   must give exactly the ids in .out (HF's tokenizer's, which llama.cpp's test-tokenizer-0 holds itself to) - every
   string, including the one only `ignore_merges` gets right.  Needs a llama.cpp checkout: $STRATA_LLAMA_CPP or the
   one setup unpacks into third_party/llama.cpp; skipped without.
2. The real tokenizer: $STRATA_GLM_GGUF, a GLM-5.3-Flash GGUF (the first shard of a split one holds it): `pre` is
   glm4, every special token the template writes is one id, and a corpus round-trips.  Skipped without.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402


def llama_cpp() -> Path | None:
    for p in (os.environ.get("STRATA_LLAMA_CPP"), ROOT / "third_party" / "llama.cpp"):
        if p and (Path(p) / "models" / "ggml-vocab-llama-bpe.gguf.out").exists():
            return Path(p)
    return None


def toy(pre: str) -> ST.Tokenizer:
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    return ST.Tokenizer(tokens, [], pre=pre)


class Patterns(unittest.TestCase):
    def test_pre_selects_the_split(self):
        # digits: qwen35 one at a time, glm4 in runs of up to three
        self.assertEqual(toy("qwen35")._re.findall("12345"), ["1", "2", "3", "4", "5"])
        self.assertEqual(toy("glm4")._re.findall("12345"), ["123", "45"])
        # a combining mark: qwen35 keeps it in the letter run, glm4 ends the run before it (no \p{M} there): the
        # mark opens the next piece as its optional non-letter lead
        self.assertEqual(toy("qwen35")._re.findall("éx"), ["éx"])
        self.assertEqual(toy("glm4")._re.findall("éx"), ["e", "́x"])

    def test_unknown_pre_is_refused(self):
        with self.assertRaises(ValueError):
            toy("gpt-4o")

    def test_ignore_merges_only_where_llama_cpp_sets_it(self):
        self.assertFalse(toy("glm4").ignore_merges)
        self.assertFalse(toy("qwen35").ignore_merges)
        self.assertTrue(toy("llama-bpe").ignore_merges)


@unittest.skipIf(llama_cpp() is None, "no llama.cpp checkout ($STRATA_LLAMA_CPP or third_party/llama.cpp)")
class LlamaCppFixture(unittest.TestCase):
    def test_llama_bpe_strings_give_hf_ids(self):
        base = llama_cpp() / "models" / "ggml-vocab-llama-bpe.gguf"
        tk = ST.Tokenizer.from_gguf(base)
        self.assertEqual(tk.pre, "llama-bpe")
        self.assertEqual(ST.PRE_PATTERNS["llama-bpe"][0], ST.PRE_PATTERNS["glm4"][0])
        inp = Path(str(base) + ".inp").read_text(encoding="utf-8").split("\n__ggml_vocab_test__\n")
        if inp[-1].endswith("\n"):
            inp[-1] = inp[-1][:-1]
        out = Path(str(base) + ".out").read_text(encoding="utf-8").split("\n")
        self.assertGreaterEqual(len(out), len(inp))
        bad = [ascii(s[:40]) for s, want in zip(inp, out) if tk.encode(s) != [int(x) for x in want.split()]]
        self.assertEqual(bad, [], f"{len(bad)} of {len(inp)} strings differ")


GLM_SPECIALS = ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|endoftext|>",
                "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
                "</arg_value>", "<tool_response>", "</tool_response>"]


@unittest.skipIf(not os.environ.get("STRATA_GLM_GGUF"), "no GLM-5.3-Flash GGUF ($STRATA_GLM_GGUF)")
class GlmTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tk = ST.Tokenizer.from_gguf(os.environ["STRATA_GLM_GGUF"])

    def test_pre_and_vocabulary(self):
        self.assertEqual(self.tk.pre, "glm4")
        self.assertEqual(len(self.tk.tokens), 154880)
        self.assertFalse(self.tk.ignore_merges)

    def test_template_specials_are_one_id_each(self):
        for s in GLM_SPECIALS:
            self.assertEqual(len(self.tk.encode(s, parse_special=True)), 1, s)
        ids = self.tk.encode("[gMASK]<sop><|user|>hi<|assistant|><think>", parse_special=True)
        self.assertEqual(self.tk.decode(ids), "[gMASK]<sop><|user|>hi<|assistant|><think>")

    def test_round_trip(self):
        corpus = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb", "def f(x):\n\treturn x  # c\n",
                  "你好，世界", "مرحبا", "\U0001f600\U0001f680",
                  "é́ combining", "\x00\x01\x7f control", " nbsp ", "1234567890",
                  "Cửa Việt", "\r\n\r\n", "x" * 3000]
        for s in corpus:
            self.assertEqual(self.tk.decode(self.tk.encode(s)), s, ascii(s[:40]))


if __name__ == "__main__":
    unittest.main()
