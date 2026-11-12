import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml

from scholarloop.core.training_budget import TrainingBudget, TrainingBudgetExceeded
from scholarloop.experiment.compare import run_campaign, train_runs
from scholarloop.experiment.suite import freeze_suite, run_suite
from examples.compare_search import random_orchestrator
from tests.experiment.test_orchestrator import PROFILE


def test_budget_reservations_survive_restart_and_concurrent_calls(tmp_path):
    path = tmp_path / "budget.json"
    budget = TrainingBudget(path, 2)
    def reserve(_):
        try:
            budget.reserve()
            return True
        except TrainingBudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve, range(8))) == 2
    with pytest.raises(TrainingBudgetExceeded):
        TrainingBudget(path, 2).reserve()
    TrainingBudget(path, 3).reserve()
    assert TrainingBudget(path, 3).used == 3


@pytest.mark.parametrize("limit", [0, 1, 2, 5])
def test_real_funnel_never_exceeds_training_cap(tmp_path, limit):
    orch = random_orchestrator(PROFILE, tmp_path, seed=0)
    entries = run_campaign(orch, max_train_runs=limit)
    assert train_runs(entries) <= limit
    assert orch.training_budget.used <= limit
    before = list(entries)
    assert run_campaign(orch, max_train_runs=limit) == before


def manifest(tmp_path):
    root = tmp_path / "project"
    shutil.copytree(PROFILE.root / "engines" / "vision", root / "engines" / "vision")
    (root / "profiles").mkdir()
    raw = yaml.safe_load(PROFILE.source_path.read_text())
    tasks = []
    for split in ("development", "evaluation"):
        path = root / "profiles" / f"{split}.yaml"
        path.write_text(yaml.safe_dump({**raw, "datasets": [f"synthetic_fixture_{split}"]}))
        tasks.append({"id": split, "split": split, "dataset_group": split,
                      "profile": str(path), "seeds": [0, 1], "train_budget": 1})
    return {"version": 1, "tasks": tasks}


def test_suite_runs_all_repeats_and_preserves_frozen_resume(tmp_path):
    spec = manifest(tmp_path)
    report = run_suite(spec, tmp_path / "suite")
    assert report["planned_repeats"] == 4
    assert all(r["train_runs"] <= 1 and r["reserved_training_attempts"] <= 1 for r in report["results"])
    assert run_suite(spec, tmp_path / "suite") == report
    changed = copy.deepcopy(spec)
    changed["tasks"][0]["train_budget"] = 2
    with pytest.raises(ValueError, match="changed"):
        run_suite(changed, tmp_path / "suite")


def test_suite_rejects_declared_data_leakage(tmp_path):
    spec = manifest(tmp_path)
    spec["tasks"][1]["profile"] = spec["tasks"][0]["profile"]
    with pytest.raises(ValueError, match="overlap"):
        freeze_suite(spec, tmp_path / "suite")
    assert not (tmp_path / "suite" / "suite_plan.json").exists()


def test_suite_failures_remain_in_denominator(tmp_path):
    spec = manifest(tmp_path)
    def broken(*args):
        raise RuntimeError("execution failed")
    report = run_suite(spec, tmp_path / "suite", arm="broken", factory=broken)
    assert report["planned_repeats"] == 4
    assert all(r["status"] == "failed" for r in report["results"])


def test_suite_rejects_model_arm_without_persistent_request_budget(tmp_path):
    from types import SimpleNamespace
    from scholarloop.core.llm import MockLLM
    llm = MockLLM()
    report = run_suite(manifest(tmp_path), tmp_path / "suite", arm="agent",
                       factory=lambda *args: SimpleNamespace(llm=llm))
    assert llm.calls == []
    assert all(r["status"] == "failed" and "persistent CallLimitedLLM" in r["reason"]
               for r in report["results"])
