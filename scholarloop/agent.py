"""Agent harness — the scaffolding every LLM agent in ScholarLoop runs on.

Design principles (the "harnessing" that matters more than the agent count):
  1. Deterministic control flow lives in the orchestrator; an Agent is a stateless,
     pure function of its context — all durable state lives in the ledger, never in an agent.
  2. Every agent has a typed I/O contract (a JSON schema). The harness validates the
     output at the boundary and retries with feedback on a structural failure, so callers
     downstream always receive a well-formed object — never prose to parse.
  3. Checkable work (dedup, calibration, constraint enforcement) stays in code, not the
     agent. The agent only does the open-ended part; the harness keeps it honest.
  4. Every call is traced (agent name + attempts + output) for auditability. This is also
     where token/cost accounting plugs in once the LLMClient surfaces usage.

Concrete agents (Reasoner, LitScout, …) subclass `Agent`, declare `schema` + `system`,
and implement `build_prompt`; the harness owns validate → retry → trace.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from jsonschema import Draft7Validator

from scholarloop.llm import LLMClient


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

    def record(self, call: AgentCall) -> None:
        self.calls.append(call)


class Agent(ABC):
    name: str = "agent"
    system: str = ""
    schema: dict = {}
    max_retries: int = 2

    def __init__(self, llm: LLMClient, *, trace: AgentTrace | None = None,
                 max_retries: int | None = None):
        self.llm = llm
        self.trace = trace
        if max_retries is not None:
            self.max_retries = max_retries

    @abstractmethod
    def build_prompt(self, ctx: dict) -> str:
        """Render the prompt for this agent from its context."""

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
        validator = Draft7Validator(self.schema) if self.schema else None
        output: dict = {}
        errors: list[str] = []
        for attempt in range(self.max_retries + 1):
            try:
                output = self.llm.complete_json(prompt, self.schema, system=self.system)
            except ValueError as e:          # json.JSONDecodeError: a truncated / non-JSON reply
                output, errors = {}, [f"response was not valid JSON ({e})"]
                prompt = self._retry_prompt(prompt, errors)
                continue
            errors = []
            if validator is not None:
                errors += [f"schema: {e.message}" for e in validator.iter_errors(output)]
            errors += self.postcheck(output, ctx)
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
