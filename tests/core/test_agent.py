"""Agent harness: schema validation, retry-with-feedback, postcheck, tracing, AgentError."""

import pytest

from scholarloop.core.agent import Agent, AgentError, AgentTrace
from scholarloop.core.llm import MockLLM

_SCHEMA = {"type": "object", "additionalProperties": False,
           "properties": {"x": {"type": "integer"}}, "required": ["x"]}


class _Echo(Agent):
    name = "echo"
    schema = _SCHEMA

    def build_prompt(self, ctx):
        return ctx["q"]


class _EvenOnly(_Echo):
    name = "even"

    def postcheck(self, output, ctx):
        return [] if output["x"] % 2 == 0 else ["x must be even"]


def test_first_shot_success_traces_one_attempt():
    trace = AgentTrace()
    out = _Echo(MockLLM(jsons=[{"x": 7}]), trace=trace).run({"q": "give x"})
    assert out == {"x": 7}
    assert trace.calls[0].attempts == 1


def test_retries_on_schema_failure_then_succeeds():
    trace = AgentTrace()
    # first reply is missing required "x" -> schema fails -> harness retries
    out = _Echo(MockLLM(jsons=[{}, {"x": 3}]), trace=trace).run({"q": "give x"})
    assert out == {"x": 3}
    assert trace.calls[0].attempts == 2


def test_postcheck_drives_retry():
    out = _EvenOnly(MockLLM(jsons=[{"x": 1}, {"x": 4}])).run({"q": "even x"})
    assert out == {"x": 4}


def test_exhausting_retries_raises_agent_error():
    with pytest.raises(AgentError) as ei:
        _Echo(MockLLM(jsons=[{}, {}]), max_retries=1).run({"q": "give x"})
    assert ei.value.agent == "echo"
    assert ei.value.errors  # carries the last validation problems


# ---- shared context block ---------------------------------------------------------------------
class _WithContext(Agent):
    name = "with_context"
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

    def build_context(self, ctx):
        return "SHARED"

    def build_prompt(self, ctx):
        return "tail"

    def postcheck(self, output, ctx):
        return [] if output["ok"] else ["say ok"]


def test_context_is_passed_separately_and_retries_leave_it_untouched():
    llm = MockLLM(jsons=[{"ok": False}, {"ok": True}])
    _WithContext(llm).run({})
    assert [c["context"] for c in llm.calls] == ["SHARED", "SHARED"]
    assert llm.calls[0]["prompt"] == "tail" and "say ok" in llm.calls[1]["prompt"]


def test_a_client_without_a_context_parameter_gets_one_prompt():
    class Plain:
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, schema, system=None):
            self.prompts.append(prompt)
            return {"ok": True}

    llm = Plain()
    _WithContext(llm).run({})
    assert llm.prompts == ["SHARED\n\ntail"]
