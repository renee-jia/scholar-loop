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


def test_api_schema_drops_only_the_array_bounds_the_api_rejects():
    from scholarloop.core.llm import api_schema
    schema = {"type": "object", "properties": {
        "maxItems": {"type": "integer"},                       # a property that happens to share the name
        "seed": {"type": "integer", "minimum": 0, "maximum": 9},
        "queries": {"type": "array", "minItems": 3, "maxItems": 5, "uniqueItems": True,
                    "items": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}}}}}
    out = api_schema(schema)
    assert out["properties"]["queries"] == {"type": "array", "minItems": 1, "items": {
        "type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}}}
    assert out["properties"]["maxItems"] == {"type": "integer"}
    assert out["properties"]["seed"] == {"type": "integer"}
    assert schema["properties"]["queries"]["maxItems"] == 5    # the full schema is left for local validation


def test_long_outputs_stream_and_a_truncated_response_is_not_retried():
    from types import SimpleNamespace
    from scholarloop.core.llm import AnthropicLLM, OutputTruncated
    seen = []

    def message(stop):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"a": 1}')], stop_reason=stop,
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    class Stream:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def get_final_message(self): return message("end_turn")

    class Messages:
        def create(self, **kw):
            seen.append(("create", kw["max_tokens"]))
            return message("max_tokens")
        def stream(self, **kw):
            seen.append(("stream", kw["max_tokens"]))
            return Stream()

    llm = AnthropicLLM(model="claude-haiku-4-5", client=SimpleNamespace(messages=Messages()), fallbacks=False)
    assert llm.complete_json("p", {"type": "object"}, max_tokens=64000) == {"a": 1}
    with pytest.raises(OutputTruncated):
        llm.complete_json("p", {"type": "object"})
    assert seen == [("stream", 64000), ("create", 16000)]


# ---- shared context block and prompt-cache accounting ----------------------------------------
def test_context_is_sent_as_a_cached_block_ahead_of_the_prompt():
    fake = _FakeClient()
    AnthropicLLM(client=fake, fallbacks=False).complete_json("the question", {"type": "object"},
                                                             context="a large dossier")
    content = fake.messages.last_kwargs["messages"][0]["content"]
    assert content == [{"type": "text", "text": "a large dossier", "cache_control": {"type": "ephemeral"}},
                       {"type": "text", "text": "the question"}]
    AnthropicLLM(client=fake, fallbacks=False).complete_json("q", {"type": "object"})
    assert fake.messages.last_kwargs["messages"][0]["content"] == "q"          # unchanged without one


class _CachedUsage:
    input_tokens, output_tokens = 1_000, 0
    cache_creation_input_tokens, cache_read_input_tokens = 100_000, 1_000_000


def test_cache_writes_and_reads_are_priced_and_counted(monkeypatch, tmp_path):
    import json
    from scholarloop.experiment.governor import cost_of
    path = tmp_path / "spend.json"
    monkeypatch.setenv("SCHOLARLOOP_SPEND_CAP", "5.0")
    monkeypatch.setenv("SCHOLARLOOP_SPEND_FILE", str(path))
    fake = _metered()
    monkeypatch.setattr(_Usage, "input_tokens", 1_000)
    monkeypatch.setattr(_Usage, "output_tokens", 0)
    monkeypatch.setattr(_Usage, "cache_creation_input_tokens", 100_000, raising=False)
    monkeypatch.setattr(_Usage, "cache_read_input_tokens", 1_000_000, raising=False)
    llm = AnthropicLLM(client=fake, fallbacks=False)
    llm.complete_json("q", {"type": "object"}, context="c")
    # $4/M input: 1k plain + 100k written at 1.25x + 1M read at 0.05x
    expected = (1_000 + 125_000 + 50_000) / 1e6 * 4.0
    assert json.loads(path.read_text())["usd"] == pytest.approx(expected)
    assert llm.usage["cache_read_input_tokens"] == 1_000_000
    assert cost_of(llm.usage, llm.model) == pytest.approx(expected)
    assert cost_of({"input_tokens": 0, "cache_read_input_tokens": 1_000_000}, "claude-haiku-5-5") == \
        pytest.approx(0.1 * 0.1)                                               # default read rate
