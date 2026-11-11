import pytest

from scholarloop.core.agent import Agent
from scholarloop.core.call_budget import CallLimitedLLM, CallBudgetExceeded
from scholarloop.core.llm import MockLLM


class InvalidAgent(Agent):
    schema = {"type": "object", "required": ["ok"]}

    def build_prompt(self, ctx):
        return "test retry accounting"


def test_retries_consume_budget_before_dispatch(tmp_path):
    client = MockLLM(jsons=[{}, {}, {"ok": True}])
    path = tmp_path / "budget.json"
    with pytest.raises(CallBudgetExceeded):
        InvalidAgent(CallLimitedLLM(client, 2, path=path)).run({})
    assert len(client.calls) == 2
    with pytest.raises(CallBudgetExceeded):
        CallLimitedLLM(client, 2, path=path).complete_json("resume", {})
    assert len(client.calls) == 2
    # An explicit increased cap retains the two already reserved attempts.
    assert CallLimitedLLM(client, 3, path=path).complete_json("resume", {}) == {"ok": True}


def test_transport_failure_still_reserves_request(tmp_path):
    path = tmp_path / "budget.json"
    client = CallLimitedLLM(MockLLM(), 1, path=path)
    with pytest.raises(AssertionError):
        client.complete("test")
    with pytest.raises(CallBudgetExceeded):
        CallLimitedLLM(MockLLM(texts=["unexpected"]), 1, path=path).complete("resume")
