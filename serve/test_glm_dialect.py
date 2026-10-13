"""serve/test_glm_dialect.py - serving GLM-5.3-Flash: its chat template (serve/fixtures/glm5_chat_template.jinja, the
one its GGUF carries), its call format and its end tokens, against a byte tokenizer with GLM's special tokens and the
mock engine (no GPU, no model).

    python -m unittest serve.test_glm_dialect -v

  - GlmCallRewriter: GLM's `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>` -> the form
    OutputParser reads, the same text whole or a character at a time; parse_glm_call's typed values
  - GlmOutputParser: reasoning, content and a call, streamed a character at a time
  - ChatTemplate on GLM's template: the dialect, the effort levels mapped onto GLM's (low / high / max), thinking off
    as an empty thinking block, the capabilities detected, a literal call tag in a message kept as text
  - forced_call in GLM's form
  - the server end to end: a reply with reasoning and an answer, a tool call (non-streamed and streamed), effort, the
    end tokens (<|user|>, <|observation|>) ending the reply
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import (GLM_TOOL_TAGS, ChatTemplate, GlmCallRewriter, GlmOutputParser, effort_kwargs,  # noqa: E402
                            forced_call, literal_tags, mark_think_literals, parse_glm_call)
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "serve/fixtures/glm5_chat_template.jinja"
CTX = 8192

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "The weather forecast for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                                    "units": {"type": "object"}}, "required": ["city"]}}}
CALL = ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key>"
        "<arg_value>3</arg_value><arg_key>units</arg_key><arg_value>{\"t\": \"C\"}</arg_value></tool_call>")


class GlmTokenizer(ByteTokenizer):
    """ByteTokenizer with GLM-5.3-Flash's special tokens: the turn markers CONTROL (parsed in a prompt only), the
    thinking and call tags type 4 (matched everywhere), as in its GGUF."""
    SPECIALS = ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|endoftext|>",
                "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
                "</arg_value>", "<tool_response>", "</tool_response>"]
    ALWAYS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
              "</arg_value>", "<tool_response>", "</tool_response>")


class GlmEngine(MockEngine):
    """MockEngine whose replies end as GLM's do: `<|user|>` after an answer, `<|observation|>` after a tool call."""

    def __init__(self, tokenizer, scripts, max_context=CTX):
        super().__init__(tokenizer, scripts, max_context=max_context)
        self.scripts = [tokenizer.encode(x, parse_special=True) +
                        tokenizer.encode("<|observation|>" if "</tool_call>" in x else "<|user|>", parse_special=True)
                        for x in ([scripts] if isinstance(scripts, str) else scripts)]
        self.script = self.scripts[0]
        self.prompts = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def prompt_text(self, i=-1):
        return self.tok.decode(self.prompts[i])


class Rewriter(unittest.TestCase):
    WANT = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n"
            "</parameter>\n<parameter=units>\n{\"t\": \"C\"}\n</parameter>\n</function>\n</tool_call>")

    def test_whole_and_char_by_char(self):
        rw = GlmCallRewriter()
        self.assertEqual(rw.push("a " + CALL + " b") + rw.flush(), "a " + self.WANT + " b")
        rw = GlmCallRewriter()
        self.assertEqual("".join(rw.push(c) for c in "a " + CALL + " b") + rw.flush(), "a " + self.WANT + " b")

    def test_no_arguments_and_whitespace(self):
        rw = GlmCallRewriter()
        self.assertEqual(rw.push("<tool_call>now\n</tool_call>") + rw.flush(),
                         "<tool_call>\n<function=now>\n</function>\n</tool_call>")
        rw = GlmCallRewriter()
        got = rw.push("<tool_call>f\n<arg_key>a</arg_key>\n<arg_value>1</arg_value>\n</tool_call>") + rw.flush()
        self.assertEqual(got, "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>")

    def test_held_tag_prefix(self):
        rw = GlmCallRewriter()
        self.assertEqual(rw.push("text <tool_"), "text ")      # a split tag waits
        self.assertEqual(rw.push("x"), "<tool_x")              # ... and was not one

    def test_parse_glm_call_typed_values(self):
        call = parse_glm_call(CALL[len("<tool_call>"):-len("</tool_call>")], WEATHER["function"])
        self.assertEqual((call.name, call.arguments), ("get_weather", {"city": "Paris", "days": 3, "units": {"t": "C"}}))


class Parser(unittest.TestCase):
    def test_reasoning_content_and_call_streamed(self):
        p = GlmOutputParser(thinking=True, tools=[WEATHER["function"]], stream_tools=True)
        evs = []
        for c in "Check the weather.</think>Let me look." + CALL:
            evs += p.feed(c)
        evs += p.finish()
        reasoning = "".join(e.text for e in evs if e.kind == "reasoning")
        content = "".join(e.text for e in evs if e.kind == "content")
        calls = [e.call for e in evs if e.kind == "tool_call"]
        self.assertEqual(reasoning, "Check the weather.")
        self.assertEqual(content, "Let me look.")
        self.assertEqual([(c.name, c.arguments) for c in calls],
                         [("get_weather", {"city": "Paris", "days": 3, "units": {"t": "C"}})])
        self.assertTrue(any(e.kind == "tool_start" for e in evs))


class Template(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = ChatTemplate(TEMPLATE)

    def render(self, **kw):
        return self.t.render([{"role": "user", "content": "hi"}], **kw)

    def test_dialect_and_prompt(self):
        self.assertEqual(self.t.dialect, "glm")
        p = self.render()
        self.assertTrue(p.startswith("[gMASK]<sop>"))
        self.assertIn("Reasoning Effort: Max", p)                  # the template's default
        self.assertTrue(p.endswith("<|user|>hi<|assistant|><think>"))

    def test_effort_levels(self):
        for given, want in (("low", "Low"), ("medium", "High"), ("high", "Max"), ("max", "Max")):
            self.assertIn(f"Reasoning Effort: {want}", self.render(**effort_kwargs(given)), given)

    def test_thinking_off(self):
        self.assertTrue(self.render(**effort_kwargs("none")).endswith("<|assistant|><think></think>"))

    def test_capabilities(self):
        self.assertEqual(self.t.caps, {"supports_tools": True, "supports_tool_calls": True,
                                       "supports_system_role": True, "supports_parallel_tool_calls": True,
                                       "supports_preserve_reasoning": True})

    def test_literal_call_tags_stay_text(self):
        tags = literal_tags(GlmTokenizer().control_tokens, GLM_TOOL_TAGS)
        msgs = [{"role": "user", "content": "write <arg_key>x</arg_key> for me"}]
        marked, _, changed = mark_think_literals(msgs, None, tags)
        self.assertTrue(changed)
        self.assertNotIn("<arg_key>", marked[0]["content"])


class Forced(unittest.TestCase):
    def test_glm_openings(self):
        tools = [WEATHER["function"], {"name": "other"}]
        self.assertEqual(forced_call("required", tools, "glm"), "<tool_call>")
        self.assertEqual(forced_call({"type": "function", "function": {"name": "get_weather"}}, tools, "glm"),
                         "<tool_call>get_weather")
        self.assertEqual(forced_call("required", [WEATHER["function"]], "glm"), "<tool_call>get_weather")


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = GlmTokenizer()
        cls.engine = GlmEngine(tok, ["Thinking it over.</think>Hello there.",
                                     "Check the weather.</think>" + CALL])
        cls.svc = Service(cls.engine, tok, ChatTemplate(TEMPLATE))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body, raw=False):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
                return r.status, (data.decode() if raw else json.loads(data))
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def setUp(self):
        self.engine.turns = 0

    def test_end_tokens(self):
        tok = GlmTokenizer()
        self.assertEqual(self.svc.dialect, "glm")
        self.assertEqual(self.svc.stop_ids, {tok.encode(t, parse_special=True)[0]
                                             for t in ("<|endoftext|>", "<|user|>", "<|observation|>")})

    def test_answer(self):
        s, b = self.post({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200, b)
        m = b["choices"][0]["message"]
        self.assertEqual((m.get("reasoning_content"), m["content"]), ("Thinking it over.", "Hello there."))
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")
        self.assertTrue(self.engine.prompt_text().startswith("[gMASK]<sop>"))
        self.assertTrue(self.engine.prompt_text().endswith("<|user|>hi<|assistant|><think>"))

    def test_tool_call(self):
        self.engine.turns = 1
        s, b = self.post({"model": "m", "messages": [{"role": "user", "content": "weather?"}], "tools": [WEATHER]})
        self.assertEqual(s, 200, b)
        m = b["choices"][0]["message"]
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
        fn = m["tool_calls"][0]["function"]
        self.assertEqual((fn["name"], json.loads(fn["arguments"])),
                         ("get_weather", {"city": "Paris", "days": 3, "units": {"t": "C"}}))
        self.assertIn("<arg_key>{arg-key-1}</arg_key>", self.engine.prompt_text())   # the template's instructions

    def test_tool_call_streamed(self):
        self.engine.turns = 1
        s, text = self.post({"model": "m", "messages": [{"role": "user", "content": "weather?"}], "tools": [WEATHER],
                             "stream": True}, raw=True)
        self.assertEqual(s, 200)
        name, args = None, ""
        for line in text.splitlines():
            if not line.startswith("data: {"):
                continue
            for tc in json.loads(line[6:])["choices"][0]["delta"].get("tool_calls") or []:
                name = tc.get("function", {}).get("name") or name
                args += tc.get("function", {}).get("arguments") or ""
        self.assertEqual((name, json.loads(args)), ("get_weather", {"city": "Paris", "days": 3, "units": {"t": "C"}}))

    def test_effort_reaches_the_prompt(self):
        s, b = self.post({"model": "m", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"})
        self.assertEqual(s, 200, b)
        self.assertIn("Reasoning Effort: Low", self.engine.prompt_text())


if __name__ == "__main__":
    unittest.main()
