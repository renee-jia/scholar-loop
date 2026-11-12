from types import SimpleNamespace

from examples import compare_search
from scholarloop.experiment.compare import arm_stats
from tests.experiment.test_orchestrator import PROFILE


def args(**kw):
    return SimpleNamespace(**{"repeats": 3, "rounds": 1, "max_cost": None,
                              "topic": "fixture", "warm_start": 0, **kw})


def test_failed_campaigns_stay_in_denominator_and_resume_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_search, "random_orchestrator", lambda *a, **kw: object())
    calls = []
    def failed(*a, **kw):
        calls.append(1)
        raise RuntimeError("provider failed")
    monkeypatch.setattr(compare_search, "run_campaign", failed)
    summaries = compare_search.run_arm("random", PROFILE, tmp_path, args())
    assert len(summaries) == 3 and all(s["status"] == "failed" for s in summaries)
    assert arm_stats(summaries)["missing_full_results"] == 3
    assert compare_search.run_arm("random", PROFILE, tmp_path, args()) == summaries
    assert len(calls) == 3


def test_interrupted_campaign_is_not_deleted(tmp_path, monkeypatch):
    rep = tmp_path / "random" / "rep1"
    rep.mkdir(parents=True)
    marker = rep / "precious_trace"
    marker.write_text("unfinished")
    def forbidden(*a, **kw):
        raise AssertionError("must not restart")
    monkeypatch.setattr(compare_search, "random_orchestrator", forbidden)
    summary = compare_search.run_arm("random", PROFILE, tmp_path, args(repeats=1))[0]
    assert summary["status"] == "interrupted" and marker.read_text() == "unfinished"


def test_budget_skipped_repeats_stay_in_denominator(tmp_path):
    llm = SimpleNamespace(usage={"input_tokens": 1000000}, model="claude-sonnet-4-6")
    summaries = compare_search.run_arm("agent", PROFILE, tmp_path, args(max_cost=0), llm=llm)
    assert len(summaries) == 3
    assert arm_stats(summaries)["statuses"] == {"not_started": 3}


def test_zero_training_target_is_not_replaced_by_rounds(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_search, "random_orchestrator", lambda *a, **kw: object())
    bounds = []
    def empty(orch, **kw):
        bounds.append(kw)
        return []
    monkeypatch.setattr(compare_search, "run_campaign", empty)
    summary = compare_search.run_arm("random", PROFILE, tmp_path, args(repeats=1), max_train_runs=0)[0]
    assert bounds == [{"max_train_runs": 0}] and summary["train_run_overshoot"] == 0


def test_resumed_comparison_requires_same_task_and_settings(tmp_path):
    import pytest
    from dataclasses import replace
    from scholarloop.experiment.compare import freeze_comparison
    settings = {"repeats": 3, "rounds": 2}
    plan = freeze_comparison(tmp_path, PROFILE, settings)
    assert freeze_comparison(tmp_path, PROFILE, settings) == plan
    with pytest.raises(ValueError, match="changed"):
        freeze_comparison(tmp_path, PROFILE, {**settings, "rounds": 4})
    with pytest.raises(ValueError, match="changed"):
        freeze_comparison(tmp_path, replace(PROFILE, cv_folds=5), settings)
