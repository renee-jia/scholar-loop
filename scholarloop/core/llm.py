"""LLM client abstraction for the framework's reasoning/idea components.

ScholarLoop runs as a standalone process, so it can NOT call Claude Code skills at
runtime — it needs its own LLM client. Everything that needs a model (the Reasoner,
the Lit Scout, the writing agents) depends on the `LLMClient` interface here, so:
  - tests run against `MockLLM` (deterministic, no API, no cost),
  - real runs use `AnthropicLLM` (Anthropic SDK, default model claude-opus-4-8).

`complete()` returns free text; `complete_json()` constrains the response to a JSON
schema (via the Messages API `output_config.format`) and returns the parsed dict — so
callers like the Reasoner get a validated object, not prose to parse.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

# Default model: the most capable Opus-tier model. See the claude-api reference.
DEFAULT_MODEL = "claude-opus-4-8"


class LLMClient(ABC):
    """Minimal interface every reasoning component depends on."""

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 16000) -> str:
        """Return the model's free-text completion for `prompt`."""

    @abstractmethod
    def complete_json(self, prompt: str, schema: dict, *, system: str | None = None,
                      max_tokens: int = 16000) -> dict:
        """Return a dict conforming to `schema` (JSON-schema-constrained output)."""


class MockLLM(LLMClient):
    """Deterministic, scriptable client for tests — no network, no cost.

    Provide FIFO queues of canned replies; each call pops the next one and records the
    request in `.calls` for assertions. A `router` callable can compute replies from the
    prompt instead, for tests that need prompt-dependent behavior.
    """

    def __init__(self, *, texts: list[str] | None = None,
                 jsons: list[dict] | None = None,
                 router=None):
        self._texts = list(texts or [])
        self._jsons = list(jsons or [])
        self._router = router
        self.calls: list[dict] = []

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 16000) -> str:
        self.calls.append({"kind": "text", "prompt": prompt, "system": system})
        if self._router is not None:
            return str(self._router(prompt))
        if not self._texts:
            raise AssertionError("MockLLM.complete called with no scripted texts left")
        return self._texts.pop(0)

    def complete_json(self, prompt: str, schema: dict, *, system: str | None = None,
                      max_tokens: int = 16000) -> dict:
        self.calls.append({"kind": "json", "prompt": prompt, "system": system, "schema": schema})
        if self._router is not None:
            return self._router(prompt)
        if not self._jsons:
            raise AssertionError("MockLLM.complete_json called with no scripted jsons left")
        return self._jsons.pop(0)


class AnthropicLLM(LLMClient):
    """Anthropic SDK-backed client. The `anthropic` package is imported lazily so the
    framework (and its tests) run without it installed; only real runs need it.

    Credentials resolve from the environment (ANTHROPIC_API_KEY or an `ant` profile) —
    never hardcode a key.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, client=None):
        self.model = model
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}   # cumulative, for cost tracking
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - exercised only without the dep
                raise RuntimeError(
                    "AnthropicLLM needs the 'anthropic' package: pip install scholarloop[llm]"
                ) from e
            self._client = anthropic.Anthropic()

    def _create(self, **kwargs):
        resp = self._client.messages.create(**kwargs)
        u = getattr(resp, "usage", None)
        if u is not None:
            self.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
            self.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
        self.usage["calls"] += 1
        return resp

    def _text_of(self, response) -> str:
        return next((b.text for b in response.content if b.type == "text"), "")

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 16000) -> str:
        kwargs = {"model": self.model, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]}
        if system is not None:
            kwargs["system"] = system
        return self._text_of(self._create(**kwargs))

    def complete_json(self, prompt: str, schema: dict, *, system: str | None = None,
                      max_tokens: int = 16000) -> dict:
        kwargs = {"model": self.model, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}],
                  "output_config": {"format": {"type": "json_schema", "schema": schema}}}
        if system is not None:
            kwargs["system"] = system
        return json.loads(self._text_of(self._create(**kwargs)))
