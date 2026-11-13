"""Agent harness — the scaffolding every LLM agent in ScholarLoop runs on.

Design principles (the "harnessing" that matters more than the agent count):
  1. Deterministic control flow lives in the orchestrator; an Agent is a stateless,
     pure function of its context — all durable state lives in the ledger, never in an agent.
  2. Every agent has a typed I/O contract (a JSON schema). The harness validates the
     output at the boundary and retries with feedback on a structural failure, so callers
     downstream always receive a well-formed object — never prose to parse.
  3. Checkable work (dedup, calibration, constraint enforcement) stays in code, not the
     agent. The agent only does the open-ended part; the harness keeps it honest.
  4. Every call is traced (agent name + attempts + output) for auditability; token/cost
     accounting reads `AnthropicLLM.usage` (see `governor.cost_of`).

Concrete agents (Reasoner, LitScout, …) subclass `Agent`, declare `schema` + `system`,
and implement `build_prompt`; the harness owns validate → retry → trace.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
import threading
import time
import uuid
from pathlib import Path

from jsonschema import Draft7Validator

from scholarloop.core.llm import LLMClient


class AgentError(RuntimeError):
    """Raised when an agent cannot produce a valid output within its retry budget."""

    def __init__(self, agent: str, errors: list[str], last_output: dict):
        super().__init__(f"{agent} failed after retries: {errors}")
        self.agent, self.errors, self.last_output = agent, errors, last_output


@dataclass
class AgentCall:
    agent: str
    attempts: int          # how many tries it took (1 == first-shot)
    output: dict


@dataclass
class AgentTrace:
    """An append-only log of agent calls — the harness's observability surface."""
    calls: list[AgentCall] = field(default_factory=list)

    path: str | Path | None = None
    events: list[dict] = field(default_factory=list)
    _lock: object = field(default_factory=threading.Lock, repr=False)

    def event(self, event: dict) -> None:
        with self._lock:
            if self.path is not None:
                path = Path(self.path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            self.events.append(event)

    def record(self, call: AgentCall) -> None:
        self.calls.append(call)


def _accepts_context(llm) -> bool:
    try:
        params = inspect.signature(llm.complete_json).parameters
    except (TypeError, ValueError):
        return False
    return "context" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


class Agent(ABC):
    name: str = "agent"
    system: str = ""
    schema: dict = {}
    max_retries: int = 2
    max_tokens: int = 0     # output limit for this agent's calls; 0 keeps the client's default

    def __init__(self, llm: LLMClient, *, trace: AgentTrace | None = None,
                 max_retries: int | None = None):
        self.llm = llm
        self.trace = trace
        if max_retries is not None:
            self.max_retries = max_retries

    @abstractmethod
    def build_prompt(self, ctx: dict) -> str:
        """Render the prompt for this agent from its context."""

    def build_context(self, ctx: dict) -> str | None:
        """Optional large block shared by several calls (other agents, retries). It is sent
        ahead of the prompt and cached by clients that support it. To share it, agents must
        also share `system` and `schema`: put what differs between them in the prompt."""
        return None

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        """Optional deterministic validation beyond the schema (override as needed).

        Return a list of human-readable problems; an empty list means the output is good.
        Use this for *structural* requirements the schema can't express — NOT for policy
        decisions (those are the orchestrator's deterministic gate).
        """
        return []

    def run(self, ctx: dict) -> dict:
        """Produce a schema-valid, post-checked output. Retries with feedback; traces the call."""
        prompt = self.build_prompt(ctx)
        context = self.build_context(ctx)
        if context and not _accepts_context(self.llm):
            prompt, context = context + "\n\n" + prompt, None     # a client without a shared block
        validator = Draft7Validator(self.schema) if self.schema else None
        output: dict = {}
        errors: list[str] = []
        call_id = uuid.uuid4().hex
        def fingerprint(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            identity = {"version": 1, "call_id": call_id, "agent": self.name, "attempt": attempt + 1}
            if self.trace is not None:
                self.trace.event({**identity, "event": "started", "timestamp": time.time(),
                    "prompt_hash": fingerprint([context, prompt] if context else prompt), "system_hash": fingerprint(self.system),
                    "schema_hash": fingerprint(self.schema), "context_hash": fingerprint(ctx),
                    "model": getattr(self.llm, "model", None),
                    "client": type(self.llm).__module__ + "." + type(self.llm).__qualname__,
                    "client_usage": dict(getattr(self.llm, "usage", {}) or {})})
            status, output, errors = "exception", {}, []
            try:
                try:
                    limit = {"max_tokens": self.max_tokens} if self.max_tokens else {}
                    if context:
                        limit["context"] = context
                    output = self.llm.complete_json(prompt, self.schema, system=self.system, **limit)
                except ValueError as exc:
                    errors = [f"response was not valid JSON ({exc})"]
                    status = "invalid_json"
                else:
                    errors = [f"schema: {e.message}" for e in validator.iter_errors(output)] if validator else []
                    if not errors:
                        errors += self.postcheck(output, ctx)
                    status = "rejected" if errors else "accepted"
            except BaseException as exc:
                errors = [f"{type(exc).__name__}: {exc}"]
                raise
            finally:
                if self.trace is not None:
                    self.trace.event({**identity, "event": "finished", "timestamp": time.time(),
                        "elapsed_seconds": time.monotonic() - started, "status": status,
                        "output": output, "errors": errors,
                        "client_usage": dict(getattr(self.llm, "usage", {}) or {}),
                        "usage_scope": "client_cumulative_not_per_attempt"})
            if not errors:
                if self.trace is not None:
                    self.trace.record(AgentCall(self.name, attempt + 1, output))
                return output
            prompt = self._retry_prompt(prompt, errors)
        if self.trace is not None:
            self.trace.record(AgentCall(self.name, self.max_retries + 1, output))
        raise AgentError(self.name, errors, output)

    @staticmethod
    def _retry_prompt(prompt: str, errors: list[str]) -> str:
        return (prompt + "\n\nYour previous response was rejected:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\nReturn a corrected response that fixes these problems.")


def summarize_trace(path):
    """Read persisted attempts without turning missing/corrupt records into successes."""
    starts, finishes, errors = {}, {}, []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
                key = (event["call_id"], event["attempt"])
                if event.get("version") != 1 or event["event"] not in ("started", "finished"):
                    raise ValueError("unknown event")
                target = starts if event["event"] == "started" else finishes
                if key in target:
                    raise ValueError("duplicate event")
                if event["event"] == "finished" and event.get("status") not in (
                        "accepted", "rejected", "invalid_json", "exception"):
                    raise ValueError("unknown attempt outcome")
                target[key] = event
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"line {number}: {exc}")
    orphan = set(finishes) - set(starts)
    errors.extend(f"finish without start: {key}" for key in sorted(orphan))
    outcomes = {status: sum(e["status"] == status for k, e in finishes.items() if k in starts)
                for status in ("accepted", "rejected", "invalid_json", "exception")}
    return {"attempts_started": len(starts), "calls_started": len({k[0] for k in starts}),
            "unfinished_attempts": len(set(starts) - set(finishes)), "outcomes": outcomes,
            "errors": errors, "complete_record": not errors and set(starts) == set(finishes)}
