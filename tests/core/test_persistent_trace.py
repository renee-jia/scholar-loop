import json

import pytest

from scholarloop.core.agent import AgentTrace, summarize_trace, AgentError
from scholarloop.core.llm import MockLLM
from tests.core.test_agent import _Echo


def test_retry_history_survives_new_trace_instance(tmp_path):
    path = tmp_path / "agent.agents.jsonl"
    _Echo(MockLLM(jsons=[{}, {"x": 1}]), trace=AgentTrace(path=path)).run({"q": "private prompt"})
    _Echo(MockLLM(jsons=[{"x": 2}]), trace=AgentTrace(path=path)).run({"q": "second"})
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert [r["status"] for r in rows if r["event"] == "finished"] == ["rejected", "accepted", "accepted"]
    assert rows[0]["prompt_hash"] != rows[2]["prompt_hash"]
    assert "private prompt" not in path.read_text()
    summary = summarize_trace(path)
    assert summary["attempts_started"] == 3 and summary["calls_started"] == 2
    assert summary["complete_record"]


@pytest.mark.parametrize("failure", [ValueError("bad JSON"), RuntimeError("provider failed"), KeyboardInterrupt()])
def test_exceptions_preserve_started_and_finished_attempt(tmp_path, failure):
    path = tmp_path / "trace.agents.jsonl"
    def fail(prompt):
        assert json.loads(path.read_text())["event"] == "started"
        raise failure
    expected = AgentError if isinstance(failure, ValueError) else type(failure)
    with pytest.raises(expected):
        _Echo(MockLLM(router=fail), trace=AgentTrace(path=path), max_retries=0).run({"q": "test"})
    summary = summarize_trace(path)
    assert summary["complete_record"] and summary["unfinished_attempts"] == 0
    status = "invalid_json" if isinstance(failure, ValueError) else "exception"
    assert summary["outcomes"][status] == 1


def test_hard_interruption_and_corrupt_tail_are_not_successes(tmp_path):
    path = tmp_path / "trace.agents.jsonl"
    AgentTrace(path=path).event({"version": 1, "call_id": "interrupted", "attempt": 1, "event": "started"})
    with path.open("a") as f:
        f.write('{"partial":')
    summary = summarize_trace(path)
    assert summary["unfinished_attempts"] == 1 and summary["outcomes"]["accepted"] == 0
    assert summary["errors"] and not summary["complete_record"]


def test_diagnostics_read_persisted_attempts(tmp_path):
    from scholarloop.evaluation import audit_run
    from tests.experiment.test_orchestrator import PROFILE
    path = tmp_path / "research.agents.jsonl"
    _Echo(MockLLM(jsons=[{"x": 1}]), trace=AgentTrace(path=path)).run({"q": "test"})
    report = audit_run(PROFILE, tmp_path)
    assert report["agent_traces"][str(path)]["outcomes"]["accepted"] == 1


def test_campaign_frontier_excludes_scientific_controls():
    from dataclasses import replace
    from scholarloop.experiment.compare import campaign_summary
    from tests.experiment.test_orchestrator import PROFILE, _entry
    a = _entry("exploration", {"lr": 0.1}, 4, "kept")
    a.fidelity, a.metric = ["full"], {"full": 4, "seeds": [4]}
    control = replace(a, id="control", purpose="ablation", metric={"full": 0, "seeds": [0]})
    summary = campaign_summary([a, control], PROFILE)
    assert summary["best_full"]["id"] == "exploration"
    assert summary["tiers"]["full"] == 1
