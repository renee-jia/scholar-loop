"""Protocols execute scientific controls even when optimization would prune them."""

import copy
import json
from dataclasses import replace
from statistics import mean

import pytest

from scholarloop.core.ledger import Ledger
from scholarloop.core.llm import MockLLM
from scholarloop.experiment.agenda import ExperimentAgenda
from scholarloop.experiment.reasoning import _scored_entries
from scholarloop.experiment.runner import RunError
from scholarloop.review.contracts import ReviewBlocked
from scholarloop.review.dossier import build_dossier
from tests.test_scientific_review import ARGUMENT, PASS, REVISE, session, setup


SPEC = {
    "purpose": "ablation", "issue_ids": ["r1:experimental:1"],
    "hypothesis": "The recipe's learning rate explains its validation error.",
    "predicted_observation": "Changing the learning rate will increase validation error.",
    "falsification_outcome": "Matched errors remain similar despite the learning-rate change.",
    "conditions": [
        {"name": "control", "description": "Reference recipe", "config": [{"name": "lr", "value": 0.05}], "edits": []},
        {"name": "ablation", "description": "Alter learning rate", "config": [{"name": "lr", "value": 100.0}], "edits": []}],
    "reference": "control", "metric": "val_top1_err", "fidelity": "verify", "seeds": [3, 7],
    "holdout": False, "analysis": "paired_mean_difference",
    "interpretation": "Descriptive change; paired observations do not prove significance.",
    "budget_seconds": 20000, "data_files": [],
    "limitations": ["Stub engine only; data content fingerprints are unavailable."],
}


def agenda(tmp_path):
    paths = setup(tmp_path)
    s = session(paths, MockLLM(jsons=[PASS, REVISE, PASS, PASS]))
    # Fold mapping is frozen and matched between both conditions.
    s.profile = replace(s.profile, cv_folds=5)
    s.review(argument=ARGUMENT)
    return ExperimentAgenda(s)


def _reporting_every_arm(dossier):
    """The shared argument plus one observation claim per protocol arm, and a passing review of
    it: every arm of an analysed protocol must be reported, whatever is concluded from it."""
    arms = {m["run_id"]: (ref, m["value"]) for ref, m in dossier["measurements"].items()
            if m["run_id"].startswith("protocol_")}
    claims = [{"id": f"claim_arm_{i}", "text": f"This arm recorded {value} percent.", "kind": "observation",
               "evidence": [run], "measurements": [ref], "depends_on": [], "scope": "This arm only."}
              for i, (run, (ref, value)) in enumerate(arms.items())]
    checks = [{**PASS["claim_checks"][0], "claim_id": c["id"], "evidence": c["evidence"]} for c in claims]
    return ({**ARGUMENT, "claims": [*ARGUMENT["claims"], *claims]},
            {**PASS, "claim_checks": [*PASS["claim_checks"], *checks]})


def test_real_stub_protocol_runs_negative_conditions_and_rereviews(tmp_path):
    a = agenda(tmp_path)
    state = a.plan(spec=copy.deepcopy(SPEC))
    pid = state["protocol_id"]
    state = a.execute(pid, author="Researcher", max_conditions=1)
    assert state["status"] == "PAUSED" and len(state["results"]) == 1
    # Resume from disk; a completed condition is never trained or appended twice.
    a = ExperimentAgenda(a.review)
    state = a.execute(pid, author="Researcher")
    assert state["status"] == "EXECUTED"
    assert a.execute(pid, author="Researcher") == state
    entries = list(Ledger(a.review.ledger_path).read_all())
    assert len(entries) == 3 and entries[-1].verdict == "discarded"
    assert entries[-1].purpose == "ablation"
    assert entries[-1].metric["seed_ids"] == [3, 7]
    assert entries[-1].metric["folds"] == [3, 2]
    assert _scored_entries(entries) == _scored_entries(entries[:1])
    assert Ledger(a.review.ledger_path).best(a.review.profile.name) is None
    analysis = json.loads((a.protocols / f"{pid}.analysis.json").read_text())
    diff = mean(b - v for v, b in zip(entries[1].metric["seeds"], entries[2].metric["seeds"]))
    assert analysis["comparisons"][0]["mean_difference"] == diff
    assert analysis["scientific_outcome"] == "unreviewed"
    with pytest.raises(ReviewBlocked):
        a.review.require_release()
    assert a.review.load()["issues"][0]["status"] == "open"
    dossier = build_dossier(a.review.inputs())
    assert not dossier["checks"]
    assert dossier["measurements"][f"protocol:{pid}::val_top1_err.difference.ablation"]["value"] == round(diff, 10)
    with pytest.raises(ReviewBlocked, match="ran but no claim reports their recorded value"):
        a.review.review(argument=ARGUMENT)                 # refused before any reviewer is called
    argument, passing = _reporting_every_arm(dossier)
    disposition = {**passing, "dispositions": [{"issue_id": "r1:experimental:1", "status": "verified",
                    "reason": "The matched comparison is now recorded, within the narrow stated scope.",
                    "evidence": [f"protocol:{pid}"]}]}
    a.review.llm = MockLLM(jsons=[passing, disposition, passing, passing])
    state = a.review.review(argument=argument)
    assert state["status"] == "READY_FOR_AUTHOR"
    with pytest.raises(ReviewBlocked):
        a.review.require_release()  # experiments/critics never act as the author


@pytest.mark.parametrize("change", ["budget", "metric", "scorer", "seeds", "holdout", "issue", "knob"])
def test_invalid_protocol_spends_no_compute(tmp_path, change):
    a = agenda(tmp_path)
    spec = copy.deepcopy(SPEC)
    if change == "budget":
        spec["budget_seconds"] = 1
    elif change == "metric":
        spec["metric"] = "invented"
    elif change == "scorer":
        spec["conditions"][1]["edits"] = [{"path": a.review.profile.fixed_module, "content": "pass"}]
    elif change == "seeds":
        spec["seeds"] = [3, 3]
    elif change == "holdout":
        spec["holdout"] = True
    elif change == "issue":
        spec["issue_ids"] = ["unknown"]
    else:
        spec["conditions"][1]["config"][0]["name"] = "invented"
    with pytest.raises(ReviewBlocked):
        a.plan(spec=spec)
    assert len(list(Ledger(a.review.ledger_path).read_all())) == 1


@pytest.mark.parametrize("change", ["source", "data", "protocol", "ledger", "registry"])
def test_frozen_protocol_detects_changed_inputs(tmp_path, change):
    a = agenda(tmp_path)
    data = a.review.profile.root / "dataset.txt"
    data.write_text("original")
    spec = copy.deepcopy(SPEC)
    spec["data_files"] = ["dataset.txt"]
    state = a.plan(spec=spec)
    pid = state["protocol_id"]
    path = {"source": a.review.profile.root / a.review.profile.train_entrypoint,
            "data": data, "protocol": a.protocols / f"{pid}.json",
            "ledger": a.review.ledger_path,
            "registry": a.review.registry_dir / "exp_0001.json"}[change]
    if change == "protocol":
        envelope = json.loads(path.read_text())
        envelope["spec"]["seeds"] = [1, 2]
        path.write_text(json.dumps(envelope))
    else:
        path.write_text(path.read_text() + "changed")
    with pytest.raises(ReviewBlocked):
        a.execute(pid, author="Researcher")
    if change != "protocol":
        assert a.load(pid)[0]["results"] == []


def test_missing_seed_is_retained_and_cannot_create_complete_comparison(tmp_path, monkeypatch):
    a = agenda(tmp_path)
    pid = a.plan(spec=SPEC)["protocol_id"]
    def fake_run(*args, **kwargs):
        if args[2] == 7:
            raise RunError("seed crash")
        assert kwargs["holdout"] is False
        return {"value": 8.0, "fold": 3}
    monkeypatch.setattr("scholarloop.experiment.runner._run_one_seed", fake_run)
    state = a.execute(pid, author="Researcher")
    assert state["status"] == "EXECUTED"
    assert state["results"][0]["entry"]["metric"]["seeds_failed"] == 1
    analysis = json.loads((a.protocols / f"{pid}.analysis.json").read_text())
    assert analysis["comparisons"][0]["status"] == "incomplete"
    assert analysis["comparisons"][0]["mean_difference"] is None
    assert a.review.load()["issues"][0]["status"] == "open"


def test_corrupt_derived_analysis_blocks_review(tmp_path):
    a = agenda(tmp_path)
    pid = a.plan(spec=SPEC)["protocol_id"]
    a.execute(pid, author="Researcher")
    path = a.protocols / f"{pid}.analysis.json"
    data = json.loads(path.read_text())
    data["comparisons"][0]["mean_difference"] = -999
    path.write_text(json.dumps(data))
    assert any("derived analysis" in s for s in build_dossier(a.review.inputs())["checks"])


def test_interrupted_condition_never_silently_repeats(tmp_path, monkeypatch):
    a = agenda(tmp_path)
    pid = a.plan(spec=SPEC)["protocol_id"]
    def fail(*args, **kwargs):
        raise RuntimeError("interrupted")
    monkeypatch.setattr("scholarloop.experiment.agenda.run_experiment", fail)
    with pytest.raises(RuntimeError):
        a.execute(pid, author="Researcher")
    with pytest.raises(ReviewBlocked, match="interrupted condition"):
        a.execute(pid, author="Researcher")


def test_protocol_requires_explicit_actor(tmp_path):
    a = agenda(tmp_path)
    pid = a.plan(spec=SPEC)["protocol_id"]
    with pytest.raises(ReviewBlocked):
        a.execute(pid, author="")
    assert len(list(Ledger(a.review.ledger_path).read_all())) == 1


def test_declared_data_change_invalidates_later_writing_release(tmp_path):
    a = agenda(tmp_path)
    data = a.review.profile.root / "dataset.txt"
    data.write_text("initial data")
    spec = copy.deepcopy(SPEC)
    spec["data_files"] = ["dataset.txt"]
    pid = a.plan(spec=spec)["protocol_id"]
    a.execute(pid, author="Researcher")
    argument, passing = _reporting_every_arm(build_dossier(a.review.inputs()))
    disposition = {**passing, "dispositions": [{"issue_id": "r1:experimental:1", "status": "verified",
                    "reason": "The control was executed under the frozen protocol.", "evidence": [f"protocol:{pid}"]}]}
    a.review.llm = MockLLM(jsons=[passing, disposition, passing, passing])
    state = a.review.review(argument=argument)
    a.review.approve(author="Researcher", review_hash=state["review_hash"])
    a.review.require_release()
    data.write_text("different data")
    with pytest.raises(ReviewBlocked, match="stale"):
        a.review.require_release()
