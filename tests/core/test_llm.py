"""Tests for the LLM client abstraction. No network: MockLLM, plus AnthropicLLM with a
fake injected SDK client (so we verify request shape without the anthropic package)."""

import pytest

from scholarloop.core.llm import DEFAULT_MODEL, AnthropicLLM, MockLLM, RefusalError, model_from_env
from scholarloop.experiment.governor import MODEL_PRICES


def test_default_model_is_opus_5_5_and_priced():
    assert DEFAULT_MODEL == "claude-opus-5-5"
    assert MODEL_PRICES[DEFAULT_MODEL] == (4.0, 20.0)      # the budget gate can meter the default


def test_model_from_env(monkeypatch):
    monkeypatch.delenv("SCHOLARLOOP_MODEL", raising=False)
    assert model_from_env() == DEFAULT_MODEL
    monkeypatch.setenv("SCHOLARLOOP_MODEL", "claude-sonnet-5-5")
    assert model_from_env() == "claude-sonnet-5-5"


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
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.stop_reason, self.stop_details = stop_reason, None


class _FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.last_kwargs = None
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        # echo a JSON body so complete_json can parse it
        return _Resp('{"answer": 42}', self.stop_reason)


class _FakeBeta:
    def __init__(self, messages):
        self.messages = messages


class _FakeClient:
    """Both endpoints share one recorder: the beta one carries the refusal fallback."""
    def __init__(self, stop_reason="end_turn"):
        self.messages = _FakeMessages(stop_reason)
        self.beta = _FakeBeta(self.messages)


def test_anthropic_complete_builds_request():
    fake = _FakeClient()
    llm = AnthropicLLM(client=fake)
    out = llm.complete("question", system="be terse", max_tokens=512)
    assert out == '{"answer": 42}'
    kw = fake.messages.last_kwargs
    assert kw["model"] == DEFAULT_MODEL
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
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


def test_fallback_only_on_models_that_support_it_and_effort_is_passed():
    fake = _FakeClient()
    AnthropicLLM("claude-opus-4-8", client=fake, effort="high").complete_json("q", {"type": "object"})
    kw = fake.messages.last_kwargs
    assert "fallbacks" not in kw and "betas" not in kw          # Opus 4.8 has no server-side fallback
    assert kw["output_config"]["effort"] == "high" and "format" in kw["output_config"]
    AnthropicLLM(client=fake, fallbacks=False).complete("q")
    assert "fallbacks" not in fake.messages.last_kwargs


def test_refusal_raises_a_retryable_error():
    llm = AnthropicLLM(client=_FakeClient(stop_reason="refusal"))
    with pytest.raises(RefusalError):
        llm.complete_json("q", {"type": "object"})
    assert issubclass(RefusalError, ValueError)                 # the Agent harness retries ValueErrors


# ---- shared hard spend cap -------------------------------------------------------------------
class _Usage:
    input_tokens, output_tokens = 100_000, 10_000          # $0.40 + $0.20 on the default model


class _MeteredMessages(_FakeMessages):
    def create(self, **kwargs):
        resp = super().create(**kwargs)
        resp.usage = _Usage()
        return resp


def _metered():
    fake = _FakeClient()
    fake.messages = _MeteredMessages()
    return fake


def test_spend_cap_is_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("SCHOLARLOOP_SPEND_CAP", raising=False)
    monkeypatch.chdir(tmp_path)
    AnthropicLLM(client=_metered(), fallbacks=False).complete("q")
    assert not (tmp_path / "scholarloop_spend.json").exists()


def test_spend_cap_is_shared_across_clients_and_refuses_before_sending(monkeypatch, tmp_path):
    import json
    from scholarloop.core.llm import SpendCapExceeded
    path = tmp_path / "spend.json"
    monkeypatch.setenv("SCHOLARLOOP_SPEND_CAP", "1.0")
    monkeypatch.setenv("SCHOLARLOOP_SPEND_FILE", str(path))
    AnthropicLLM(client=_metered(), fallbacks=False).complete("q")           # $0.60
    assert json.loads(path.read_text())["usd"] == pytest.approx(0.6)
    second = _metered()
    AnthropicLLM(client=second, fallbacks=False).complete("q")               # a new client, same file: $1.20
    third = _metered()
    with pytest.raises(SpendCapExceeded, match="spend cap reached"):
        AnthropicLLM(client=third, fallbacks=False).complete("q")
    assert third.messages.last_kwargs is None                               # nothing was sent
    assert json.loads(path.read_text())["requests"] == 2


def test_spend_cap_refuses_an_unpriced_model(monkeypatch, tmp_path):
    from scholarloop.core.llm import SpendCapExceeded
    monkeypatch.setenv("SCHOLARLOOP_SPEND_CAP", "1.0")
    monkeypatch.setenv("SCHOLARLOOP_SPEND_FILE", str(tmp_path / "spend.json"))
    fake = _metered()
    with pytest.raises(SpendCapExceeded, match="no price"):
        AnthropicLLM("some-unpriced-model", client=fake).complete("q")
    assert fake.messages.last_kwargs is None
