"""LLM client abstraction for the framework's reasoning/idea components.

ScholarLoop runs as a standalone process, so it can NOT call Claude Code skills at
runtime — it needs its own LLM client. Everything that needs a model (the Reasoner,
the Lit Scout, the writing agents) depends on the `LLMClient` interface here, so:
  - tests run against `MockLLM` (deterministic, no API, no cost),
  - real runs use `AnthropicLLM` (Anthropic SDK, default model `DEFAULT_MODEL`).

`complete()` returns free text; `complete_json()` constrains the response to a JSON
schema (via the Messages API `output_config.format`) and returns the parsed dict — so
callers like the Reasoner get a validated object, not prose to parse.
"""

from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path

# The one place the default model is named: every CLI and example reads it from here (or from
# $SCHOLARLOOP_MODEL via `model_from_env`). Prices for the cost gate live in governor.MODEL_PRICES.
DEFAULT_MODEL = "claude-opus-5-5"

# Models that accept the server-side refusal fallback (`fallbacks: "default"`): on a policy decline
# the API re-runs the same request on a fallback model inside the same call.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def model_from_env() -> str:
    """$SCHOLARLOOP_MODEL if set, else DEFAULT_MODEL."""
    import os
    return os.environ.get("SCHOLARLOOP_MODEL") or DEFAULT_MODEL


class RefusalError(ValueError):
    """The model (and any fallback) declined the request. A ValueError, so the agent harness
    treats it like any other unusable reply: retried with feedback, then surfaced as AgentError."""


class SpendCapExceeded(RuntimeError):
    """The shared dollar cap ($SCHOLARLOOP_SPEND_CAP) is reached; no request was sent."""


_spend_mutex = threading.Lock()


def _spend(model: str, usage: dict | None = None) -> None:
    """A hard, cross-process dollar cap on real API requests, off unless $SCHOLARLOOP_SPEND_CAP
    (USD) is set. Estimated spend accumulates in $SCHOLARLOOP_SPEND_FILE (default
    ./scholarloop_spend.json), so ideation, the campaign, review, writing and evaluation
    commands pointed at one file share one budget. Checked before every request and charged
    after it: the cap can be overshot by at most the request in flight. Spend is an estimate
    from token counts and `governor.MODEL_PRICES`; a model without a price is refused, since an
    unpriced request cannot be capped. With `usage` it records a finished request; without, it
    checks the cap."""
    cap = os.environ.get("SCHOLARLOOP_SPEND_CAP")
    if not cap:
        return
    from scholarloop.experiment.governor import MODEL_PRICES
    price = MODEL_PRICES.get(model)
    if price is None:
        raise SpendCapExceeded(f"SCHOLARLOOP_SPEND_CAP is set but model {model!r} has no price")
    path = Path(os.environ.get("SCHOLARLOOP_SPEND_FILE") or "scholarloop_spend.json")
    with _spend_mutex:
        state = json.loads(path.read_text()) if path.exists() else {"usd": 0.0, "requests": 0}
        if usage is None:
            if state["usd"] >= float(cap):
                raise SpendCapExceeded(f"spend cap reached: ${state['usd']:.2f} of ${float(cap):.2f} "
                                       f"({path}); saved work is kept, raise the cap to continue")
            return
        state["usd"] += usage["input"] / 1e6 * price[0] + usage["output"] / 1e6 * price[1]
        state["requests"] += 1
        state["cap_usd"] = float(cap)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, path)


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


_STREAM_ABOVE = 16000


class OutputTruncated(RuntimeError):
    """The model's output was cut off at the token limit."""


_API_UNSUPPORTED = ("maxItems", "uniqueItems", "minimum", "maximum", "exclusiveMinimum",
                    "exclusiveMaximum", "multipleOf")


def api_schema(schema):
    """The schema as the structured-output API accepts it. The API rejects some array bounds
    (`maxItems`, `uniqueItems`, `minItems` above 1) and numeric bounds (`minimum`, `maximum`,
    ...), so they are dropped from the request only:
    the Agent harness still validates every response against the full schema and retries with
    the violation, so the bounds stay enforced."""
    if isinstance(schema, list):
        return [api_schema(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):    # property names are not keywords
            out[key] = {name: api_schema(sub) for name, sub in value.items()}
        elif key not in _API_UNSUPPORTED:
            out[key] = api_schema(value)
    if isinstance(out.get("minItems"), int) and out["minItems"] > 1 and out.get("type") == "array":
        out["minItems"] = 1
    return out


class AnthropicLLM(LLMClient):
    """Anthropic SDK-backed client. The `anthropic` package is imported lazily so the
    framework (and its tests) run without it installed; only real runs need it.

    Credentials resolve from the environment (ANTHROPIC_API_KEY or an `ant` profile) —
    never hardcode a key.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, client=None, effort: str | None = None,
                 fallbacks: bool = True):
        """`effort` sets output_config.effort (None = the model's default; Opus 5.5 defaults to
        medium). `fallbacks` opts supported models into the server-side refusal fallback; the
        cost gate still prices every call at `model`'s rate."""
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks and model in _FALLBACK_MODELS
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
        if self.effort is not None:
            kwargs["output_config"] = {**kwargs.get("output_config", {}), "effort": self.effort}
        _spend(self.model)                              # refuse before sending once the cap is reached
        api = self._client.beta.messages if self.fallbacks else self._client.messages
        extra = {"betas": [_FALLBACK_BETA], "fallbacks": "default"} if self.fallbacks else {}
        if kwargs.get("max_tokens", 0) > _STREAM_ABOVE:    # long outputs must stream or the SDK refuses them
            with api.stream(**extra, **kwargs) as stream:
                resp = stream.get_final_message()
        else:
            resp = api.create(**extra, **kwargs)
        u = getattr(resp, "usage", None)
        _spend(self.model, {"input": getattr(u, "input_tokens", 0) or 0,
                            "output": getattr(u, "output_tokens", 0) or 0})
        if u is not None:
            self.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
            self.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
        self.usage["calls"] += 1
        if getattr(resp, "stop_reason", None) == "max_tokens":
            # Not a ValueError: a retry would be cut off at the same place and billed again.
            raise OutputTruncated(f"the response hit max_tokens={kwargs.get('max_tokens')} before it was "
                                  "complete; raise the agent's max_tokens or ask for less output")
        if getattr(resp, "stop_reason", None) == "refusal":
            details = getattr(resp, "stop_details", None)
            raise RefusalError(f"the model declined the request (category "
                               f"{getattr(details, 'category', None)!r})")
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
                  "output_config": {"format": {"type": "json_schema", "schema": api_schema(schema)}}}
        if system is not None:
            kwargs["system"] = system
        return json.loads(self._text_of(self._create(**kwargs)))
