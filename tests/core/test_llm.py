"""Tests for the LLM client abstraction. No network: MockLLM, plus AnthropicLLM with a
fake injected SDK client (so we verify request shape without the anthropic package)."""

import pytest

from scholarloop.core.llm import DEFAULT_MODEL, AnthropicLLM, MockLLM


def test_default_model_is_opus_4_8():
    assert DEFAULT_MODEL == "claude-opus-4-8"


def test_mock_text_and_json_fifo_and_logging():
    llm = MockLLM(texts=["hello"], jsons=[{"ok": True}])
    assert llm.complete("hi", system="sys") == "hello"
    assert llm.complete_json("give json", {"type": "object"}) == {"ok": True}
    assert [c["kind"] for c in llm.calls] == ["text", "json"]
    assert llm.calls[0]["system"] == "sys"


def test_mock_router_is_prompt_dependent():
    llm = MockLLM(router=lambda p: {"echo": p} if "{" in p else p.upper())
    assert llm.complete("abc") == "ABC"
    assert llm.complete_json("{...}", {}) == {"echo": "{...}"}


def test_mock_raises_when_script_exhausted():
    with pytest.raises(AssertionError):
        MockLLM().complete("x")


# ---- AnthropicLLM with a fake client (no anthropic package needed) ----
class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]


class _FakeMessages:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        # echo a JSON body so complete_json can parse it
        return _Resp('{"answer": 42}')


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_anthropic_complete_builds_request():
    fake = _FakeClient()
    llm = AnthropicLLM(client=fake)
    out = llm.complete("question", system="be terse", max_tokens=512)
    assert out == '{"answer": 42}'
    kw = fake.messages.last_kwargs
    assert kw["model"] == "claude-opus-4-8"
    assert kw["max_tokens"] == 512
    assert kw["system"] == "be terse"
    assert kw["messages"] == [{"role": "user", "content": "question"}]
    assert "output_config" not in kw          # plain completion sets no format


def test_anthropic_complete_json_sets_schema_and_parses():
    fake = _FakeClient()
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    out = AnthropicLLM(client=fake).complete_json("q", schema)
    assert out == {"answer": 42}
    fmt = fake.messages.last_kwargs["output_config"]["format"]
    assert fmt == {"type": "json_schema", "schema": schema}


def test_anthropic_omits_system_when_none():
    fake = _FakeClient()
    AnthropicLLM(client=fake).complete("q")
    assert "system" not in fake.messages.last_kwargs
