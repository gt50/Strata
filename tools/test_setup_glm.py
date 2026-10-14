"""Tests for setup.py's GLM-5.3-Flash choice (EXPERIMENTAL, install_glm): its two sizes' pinned shards (UD-IQ2_XXS four,
UD-IQ4_XS five) with their sizes and SHA-256, the size recommended by what fits the GPU, the engine check before the
download, the tokenizer with its glm4 split, a config that runs the GGUF directly
(no pack, no MTP download, no Qwen memory modes), one GPU, and the Qwen questions left out.  Mocked - no GPU, no
downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_glm
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

M = "UD-IQ4_XS"
M2 = "UD-IQ2_XXS"
REV = "a38483c8cd5df544f53d70fb281afe97369d5ab6"
SHARDS = [f"GLM-5.3-Flash-UD-IQ4_XS-0000{i}-of-00005.gguf" for i in range(1, 6)]
SHARDS2 = [f"GLM-5.3-Flash-UD-IQ2_XXS-0000{i}-of-00004.gguf" for i in range(1, 5)]


def quiet(fn, *args, **kw):
    """-> (exit code or None, printed text)."""
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            code = fn(*args, **kw)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class Pins(unittest.TestCase):
    def test_the_shards_of_both_sizes(self):
        self.assertEqual(list(setup.GLM_SHARDS), SHARDS2 + SHARDS)
        fam = setup.FAMILIES["glm"]
        self.assertEqual(list(fam["models"]), [M2, M])       # the menu: the smaller first
        for m, names in ((M, SHARDS), (M2, SHARDS2)):
            size = fam["models"][m]
            self.assertEqual(round(sum(setup.GLM_SHARDS[n][0] for n in names) / 1e9, 1), size["download_gb"])
            self.assertEqual([fam["file"].format(q=m, i=i, n=size["shards"]) for i in range(1, size["shards"] + 1)],
                             names)
        self.assertEqual(setup.HF_REVISIONS["unsloth/GLM-5.3-Flash-GGUF"], REV)
        self.assertTrue(fam["experimental"])

    def test_what_each_size_needs_to_fit(self):
        # glm_loader_parity --plan: 7.0 + 95.2 GB and 10.0 + 147.1 GB, and the engine's 3 GB of room
        sizes = setup.FAMILIES["glm"]["models"]
        self.assertEqual(sizes[M2]["fit_gb"], 105.2)
        self.assertEqual(sizes[M]["fit_gb"], 160.1)

    def test_not_a_size_of_the_other_families(self):
        # its UD-IQ4_XS is GLM's own: the Qwen one in MODELS keeps its three shards
        self.assertNotIn("glm", setup.MODELS[M].get("families", ()))
        self.assertEqual(setup.MODELS[M]["shards"], 3)
        self.assertIsNone(setup.gguf_choice(SHARDS[0]))


class EngineMark(unittest.TestCase):
    def test_mark(self):
        with tempfile.TemporaryDirectory() as t:
            yes, no = Path(t) / "a.exe", Path(t) / "b.exe"
            yes.write_bytes(b"\0" * 100 + b"INFO engine=%s arch=glm5-next context=%lld" + b"\0" * 100)
            no.write_bytes(b"\0" * 100 + b"engine=0.1.43.1\n")
            self.assertTrue(setup.glm_engine(yes))
            self.assertFalse(setup.glm_engine(no))
            self.assertFalse(setup.glm_engine(Path(t) / "missing.exe"))

    def test_the_engine_source_prints_it(self):
        src = (ROOT / "src" / "glm" / "glm_serve.cpp").read_text(encoding="utf-8")
        self.assertIn(setup.GLM_ENGINE_MARK.decode(), src)


class Base(unittest.TestCase):
    """setup.main() for the GLM choice, every outside effect mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.downloads, self.runs, self.verified = [], [], []

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, argv, ram=127.8, glm=True, n_gpus=1, model=True, family=True, answers=None, free=500.0, size=M,
             vram=24.0):
        """answers: None = --yes; else {words of a question: its answer} (Enter for the others), every question asked
        kept in self.asked.  glm=False: an engine without the GLM path."""
        self.asked = []

        def fake_input(prompt=""):
            if answers is None:
                raise AssertionError(f"asked {prompt!r}")
            self.asked.append(prompt)
            return next((v for k, v in answers.items() if k in prompt), "")

        eng = self.t / "engine"
        eng.mkdir(exist_ok=True)
        (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.43.1", "source": "local"}))
        (eng / setup.EXE).write_bytes(b"engine=0.1.43.1\n" + (setup.GLM_ENGINE_MARK if glm else b""))
        found = [{"index": i, "name": "NVIDIA GeForce RTX 4090", "vram_gb": vram, "arch": "89", "driver": "580.97"}
                 for i in range(n_gpus)]

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def fake_run(cmd, *a, **k):
            self.runs.append([str(x) for x in cmd])
            if cmd[1].endswith("strata_tokenizer.py"):
                out = Path(cmd[cmd.index("--out") + 1]) / "tokenizer"
                out.mkdir(parents=True, exist_ok=True)
                (out / "vocab.json").write_text("{}")
                (out / "tokenizer.json").write_text(json.dumps({"pre": "glm4"}))

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "amd_gpus", lambda *a: []),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "cpu_cores", lambda: None),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "free_gb", lambda p: free),
            mock.patch.object(setup, "is_wsl", lambda: False),
            mock.patch.object(setup, "pcie_link", lambda i: None),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", lambda *a, **k: eng),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "run", fake_run),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"start-{tag}.bat"),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.object(sys, "argv", ["setup.py", *(["--family", "glm"] if family else []),
                                            *(["--model", size] if model else []),
                                            *(["--yes"] if answers is None else []), "--no-start",
                                            "--models-dir", str(self.t / "models"), *argv]),
            mock.patch("builtins.input", fake_input),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = quiet(setup.main)
        cfgs = sorted(self.t.glob("strata-glm-*.json"))
        self.assertLessEqual(len(cfgs), 1)
        return code, out, (json.loads(cfgs[0].read_text()) if cfgs else None)


class Install(Base):
    def test_install(self):
        code, out, cfg = self.main([])
        self.assertEqual(code, 0, out)
        self.assertIn("EXPERIMENTAL", out)
        base = f"https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/{REV}/UD-IQ4_XS/"
        self.assertEqual(self.downloads, [base + n for n in SHARDS])          # no image encoder, no MTP tensors
        self.assertEqual([v[0] for v in self.verified], SHARDS)
        self.assertEqual([v[1:] for v in self.verified], [setup.GLM_SHARDS[n] for n in SHARDS])
        # the tokenizer only: no pack, no draft layer
        self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
        tok = self.runs[0]
        self.assertTrue(tok[tok.index("--gguf") + 1].endswith(SHARDS[0]))
        self.assertIn("--check", tok)
        models = self.t / "models" / "glm-UD-IQ4_XS"
        self.assertEqual(cfg["args"], ["--native", str(models / SHARDS[0]), "--max-context", "32768"])
        self.assertEqual(cfg["arch"], "glm5-next")
        self.assertTrue(setup.is_glm_config(cfg))
        self.assertTrue(cfg["gpus_asked"])
        self.assertEqual(cfg["model_name"], "glm-5.3-flash-ud-iq4_xs")
        self.assertEqual(Path(cfg["tokenizer"]), self.t / "data" / "packs" / "glm-ud-iq4_xs" / "tokenizer")
        self.assertEqual(cfg["exe"], str(self.t / "engine" / setup.EXE))
        for key in ("vision", "parallel", "draft_vocab", "backend", "layer_split"):
            self.assertNotIn(key, cfg)

    def test_choices_from_config(self):
        code, out, cfg = self.main([])
        self.assertEqual(code, 0, out)
        ch = setup.choices_from_config(self.t / f"strata-glm-{M.lower()}.json")
        self.assertEqual((ch["family"], ch["model"], ch["context"]), ("glm", M, 32768))

    def test_install_the_2_bit_size(self):
        code, out, cfg = self.main([], size=M2)
        self.assertEqual(code, 0, out)
        base = f"https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/{REV}/UD-IQ2_XXS/"
        self.assertEqual(self.downloads, [base + n for n in SHARDS2])
        self.assertEqual([v[0] for v in self.verified], SHARDS2)
        self.assertEqual([v[1:] for v in self.verified], [setup.GLM_SHARDS[n] for n in SHARDS2])
        models = self.t / "models" / "glm-UD-IQ2_XXS"
        self.assertEqual(cfg["args"], ["--native", str(models / SHARDS2[0]), "--max-context", "32768"])
        self.assertEqual(cfg["model_name"], "glm-5.3-flash-ud-iq2_xxs")
        self.assertEqual(Path(cfg["tokenizer"]), self.t / "data" / "packs" / "glm-ud-iq2_xxs" / "tokenizer")
        # a run of setup again reads its answers back (the size is in no other family's list)
        ch = setup.choices_from_config(self.t / "strata-glm-ud-iq2_xxs.json")
        self.assertEqual((ch["family"], ch["model"]), ("glm", M2))

    def test_engine_without_glm_stops_before_the_download(self):
        code, out, cfg = self.main([], glm=False)
        self.assertEqual(code, 1)
        self.assertIn("has no GLM-5.3-Flash path", out)
        self.assertEqual(self.downloads, [])
        self.assertIsNone(cfg)

    def test_files_already_there_are_not_downloaded(self):
        d = self.t / "models" / "glm-UD-IQ4_XS"
        d.mkdir(parents=True)
        for n in SHARDS[:2]:
            (d / n).write_bytes(b"")
            setup.mark(d / n)
        code, out, cfg = self.main([])
        self.assertEqual(code, 0, out)
        self.assertEqual([u.rsplit("/", 1)[1] for u in self.downloads], SHARDS[2:])
        self.assertEqual(len(self.verified), 5)                  # every shard checked (once: the mark keeps it)

    def test_disk_space(self):
        code, out, cfg = self.main([], free=100.0)
        self.assertEqual(code, 1)
        self.assertIn("need ~158 GB", out)
        self.assertEqual(self.downloads, [])
        code, out, cfg = self.main([], free=100.0, size=M2)
        self.assertEqual(code, 1)
        self.assertIn("need ~103 GB", out)

    def test_a_tokenizer_without_the_glm_split_is_extracted_again(self):
        tok = self.t / "data" / "packs" / "glm-ud-iq4_xs" / "tokenizer"
        tok.mkdir(parents=True)
        (tok / "vocab.json").write_text("{}")                    # an extraction from before "pre" was written
        code, out, cfg = self.main([])
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.runs), 1)
        code, out, cfg = self.main([])                           # now with "pre": glm4, kept
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.runs), 1)


class Choices(Base):
    def test_context(self):
        code, out, cfg = self.main(["--context", "131072"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "131072")
        self.assertNotIn("--rope-scaling", cfg["args"])

    def test_context_past_the_trained_length(self):
        code, out, cfg = self.main(["--context", str(2 * setup.GLM_TRAINED_CTX)])
        self.assertEqual(code, 1)
        self.assertIn("64 to 1048576 tokens", out)
        self.assertIsNone(cfg)

    def test_qwen_options_are_left_out(self):
        code, out, cfg = self.main(["--kv", "q4_0", "--vision", "yes", "--draft-vocab", "en", "--parallel", "2",
                                    "--experimental-speed-projection", "on"])
        self.assertEqual(code, 0, out)
        self.assertIn("--kv, --vision, --experimental-speed-projection, --draft-vocab, --parallel: not used for "
                      "GLM-5.3-Flash", out)
        self.assertEqual(len(cfg["args"]), 4)
        for key in ("vision", "parallel", "draft_vocab"):
            self.assertNotIn(key, cfg)

    def test_recommended_size(self):
        """--yes without --model: the best size that fits the GPU whole, else the smallest."""
        for vram, want in ((24.0, M2), (110.0, M2), (170.0, M)):
            code, out, cfg = self.main([], model=False, vram=vram)
            self.assertEqual(code, 0, out)
            self.assertIn(f"size: {want}", out)
            line = next(ln for ln in out.splitlines() if "(recommended)" in ln)
            self.assertIn(want, line)
            for p in self.t.glob("strata-glm-*.json"):
                p.unlink()
        self.assertIn("UD-IQ2_XXS ~2-bit", out)
        self.assertIn("fits this GPU whole", out)

    def test_the_room_of_an_apu(self):
        # the 8060S: 64 GB carved out, 63.6 left to Windows - the runtime reported 115 GB free and UD-IQ2_XXS fit
        apu = {"uma": True, "dedicated_gb": 64.0, "shared_gb": 31.8, "vram_gb": 95.8}
        room = setup.glm_gpu_room(apu, 63.6)
        sizes = setup.FAMILIES["glm"]["models"]
        self.assertGreaterEqual(room, sizes[M2]["fit_gb"])
        self.assertLess(room, sizes[M]["fit_gb"])
        self.assertEqual(setup.glm_gpu_room({"vram_gb": 24.0}, 128.0), 24.0)   # a card: its VRAM

    def test_its_size_named_for_another_family(self):
        # --model UD-IQ2_XXS alone: the first family has no such size; the message names GLM's
        code, out, cfg = self.main(["--family", "qwen"], size=M2, family=False)
        self.assertEqual(code, 1)
        self.assertIn("--family glm --model UD-IQ2_XXS", out)

    def test_unknown_size(self):
        code, out, cfg = self.main(["--model", "Q2_0"], model=False)
        self.assertEqual(code, 1)
        self.assertIn("GLM-5.3-Flash has no Q2_0 model file", out)

    def test_one_gpu(self):
        code, out, cfg = self.main(["--gpus", "0,1"], n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertIn("runs on one GPU so far", out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertNotIn("layer_split", cfg)

    def test_menus(self):
        """Interactive: the family menu lists GLM last, marked [experimental]; then its size and the context (none of
        the Qwen questions), Enter = the recommended size and the engine's default 32K."""
        code, out, cfg = self.main([], model=False, family=False, answers={"Which model": "5"})
        self.assertEqual(code, 0, out)
        line = next(ln for ln in out.splitlines() if ln.startswith("  5) "))
        self.assertIn("GLM-5.3-Flash", line)
        self.assertIn("[experimental]", line)
        self.assertEqual(len(self.asked), 3, self.asked)
        self.assertTrue(self.asked[0].startswith("Which model?"))
        self.assertTrue(self.asked[1].startswith("Which size?"))
        self.assertTrue(self.asked[2].startswith("Context?"))
        self.assertEqual(cfg["model_name"], "glm-5.3-flash-ud-iq2_xxs")   # neither fits a 24 GB card: the smaller
        self.assertEqual(cfg["args"][-1], "32768")


class Start(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "strata.exe").write_bytes(b"")
        self.first = self.t / SHARDS[0]
        for n in SHARDS:
            (self.t / n).write_bytes(b"x" * 1000)
        self.cfg = self.t / "strata-glm-ud-iq4_xs.json"
        self.cfg.write_text(json.dumps({"exe": str(self.t / "strata.exe"), "args": ["--native", str(self.first),
                                        "--max-context", "32768"], "arch": "glm5-next", "gpus_asked": True,
                                        "model_name": "glm-5.3-flash-ud-iq4_xs"}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_several_gpus_refused(self):
        code, out = quiet(setup.start, self.cfg, 8080, [0, 1])
        self.assertEqual(code, 1)
        self.assertIn("runs on one GPU so far", out)

    def test_start_says_what_it_maps(self):
        with mock.patch.object(setup, "gpus", lambda: []), mock.patch.object(setup.subprocess, "call",
                                                                             lambda cmd: 0) as _:
            code, out = quiet(setup.start, self.cfg, 8080)
        self.assertEqual(code, 0, out)
        self.assertIn("it maps the model files (about 0 GB)", out)
        self.assertNotIn("34-55 GB", out)


class Calibrate(unittest.TestCase):
    def test_glm_is_not_tuned(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = Path(t) / "strata-glm-ud-iq4_xs.json"
            cfg.write_text(json.dumps({"exe": "x", "args": [], "arch": "glm5-next"}))
            started = mock.Mock(return_value=0)
            with mock.patch.object(setup, "installed_configs", lambda: [cfg]), \
                    mock.patch.object(setup, "data_folder", lambda d: (Path(t) / "data", [])), \
                    mock.patch.object(setup, "calibrate_config", mock.Mock(side_effect=AssertionError("tuned"))), \
                    mock.patch.object(setup, "start", started), \
                    mock.patch.object(sys, "argv", ["setup.py", "--calibrate"]):
                code, out = quiet(setup.main)
        self.assertEqual(code, 0, out)
        self.assertIn("has no tuning yet", out)
        self.assertEqual(started.call_args[0][0], cfg)


if __name__ == "__main__":
    unittest.main()
