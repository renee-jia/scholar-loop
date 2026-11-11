import copy
import json
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from scholarloop.core.ledger import Ledger
from scholarloop.core.llm import MockLLM
from scholarloop.ideation.litscout import ArxivClient
from scholarloop.ideation.pilot import PilotAgenda
from scholarloop.ideation.session import ResearchSession
from scholarloop.review.contracts import ReviewBlocked
from scholarloop.review.dossier import build_dossier
from tests.experiment.test_agenda import SPEC
from tests.ideation.test_session import ATOM, QUERIES, REVIEW, _idea
from tests.test_scientific_review import setup


def fixture(tmp_path, *, first_experiment=False):
    profile, ledger, registry, _ = setup(tmp_path)
    profile = replace(profile, cv_folds=5)
    if first_experiment:
        ledger.unlink()
        (registry / "exp_0001.json").unlink()
    research = ResearchSession(MockLLM(jsons=[QUERIES, REVIEW,
                               {"ideas": [_idea("Mechanism"), _idea("Alternative")]}]),
                               profile, tmp_path / "research", topic="Mechanism tests",
                               sources=[ArxivClient(fetcher=lambda q, n: ATOM)])
    research.review()
    ids = [i.idea_id for i in research.ideate(2)]
    agenda = PilotAgenda(None, profile, research.out, ledger_path=ledger, registry_dir=registry, out_dir=tmp_path)
    spec = copy.deepcopy(SPEC)
    spec.pop("issue_ids")
    spec.update(purpose="feasibility", fidelity="smoke", seeds=[0, 1], idea_ids=ids)
    spec["conditions"][0]["idea_id"] = "control"
    spec["conditions"][1]["idea_id"] = ids[0]
    spec["conditions"].append({**copy.deepcopy(spec["conditions"][1]), "name": "alternative", "idea_id": ids[1]})
    return research, agenda, spec


def test_portfolio_pilot_runs_all_conditions_and_feeds_generation(tmp_path):
    research, agenda, spec = fixture(tmp_path, first_experiment=True)
    pid = agenda.plan(spec["idea_ids"], spec=spec)["protocol_id"]
    assert agenda.execute(pid, author="Researcher", max_conditions=1)["status"] == "PAUSED"
    assert not (research.out / "pilot_feedback.json").exists()
    state = agenda.execute(pid, author="Researcher")
    assert state["status"] == "EXECUTED"
    entries = list(Ledger(agenda.review.ledger_path).read_all())
    assert len(entries) == 3 and all(e.purpose == "feasibility" for e in entries)
    assert entries[-1].verdict == "discarded" and all("holdout" not in e.metric for e in entries)
    assert not build_dossier(agenda.review.inputs())["checks"]
    feedback = json.loads((research.out / "pilot_feedback.json").read_text())
    assert {o["idea_id"] for o in feedback} == set(spec["idea_ids"])
    assert all(o["hypothesis_status"] == "unresolved" for o in feedback)
    assert len(agenda.execute(pid, author="Researcher")["results"]) == 3
    assert len(json.loads((research.out / "pilot_feedback.json").read_text())) == 2
    assert not (agenda.review.out / "scientific_review.json").exists()
    resumed = ResearchSession(MockLLM(jsons=[{"ideas": [_idea("Revised mechanism")]}]),
                               research.profile, research.out, sources=[])
    resumed.ideate(1, feedback="Explain the disappointing pilot")
    assert "Observed pilot results" in resumed.llm.calls[0]["prompt"]
    assert "not confirmations or refutations" in resumed.llm.calls[0]["prompt"]


@pytest.mark.parametrize("change", ["test", "full", "seed_count", "missing_candidate", "wrong_control", "budget"])
def test_invalid_pilot_is_rejected_before_execution(tmp_path, change):
    _, agenda, spec = fixture(tmp_path)
    if change == "test":
        spec["holdout"] = True
    elif change == "full":
        spec["fidelity"] = "full"
    elif change == "seed_count":
        spec["seeds"] = [0, 1, 2, 3]
    elif change == "missing_candidate":
        spec["conditions"].pop()
    elif change == "wrong_control":
        spec["reference"] = "ablation"
    else:
        spec["budget_seconds"] = 1
    with pytest.raises(ReviewBlocked):
        agenda.plan(spec["idea_ids"], spec=spec)
    assert len(list(Ledger(agenda.review.ledger_path).read_all())) == 1


def test_human_rejection_invalidates_planned_pilot(tmp_path):
    research, agenda, spec = fixture(tmp_path)
    pid = agenda.plan(spec["idea_ids"], spec=spec)["protocol_id"]
    research.decide("reject", [spec["idea_ids"][0]], actor="Researcher", reason="The mechanism is redundant")
    with pytest.raises(ReviewBlocked, match="active"):
        agenda.execute(pid, author="Researcher")
    assert len(list(Ledger(agenda.review.ledger_path).read_all())) == 1


def test_failed_pilot_is_not_marked_as_hypothesis_refutation(tmp_path, monkeypatch):
    from scholarloop.experiment.runner import RunError
    research, agenda, spec = fixture(tmp_path)
    pid = agenda.plan(spec["idea_ids"], spec=spec)["protocol_id"]
    def fail(*args, **kwargs):
        raise RunError("implementation does not produce an artifact")
    monkeypatch.setattr("scholarloop.experiment.runner._run_one_seed", fail)
    assert agenda.execute(pid, author="Researcher")["status"] == "EXECUTED"
    observations = json.loads((research.out / "pilot_feedback.json").read_text())
    assert all(o["execution_status"] == "failed" and o["hypothesis_status"] == "unresolved" for o in observations)
    assert all(o["comparison"]["mean_difference"] is None for o in observations)


def test_pilot_revokes_old_selection_and_new_choice_binds_observations(tmp_path):
    research, agenda, spec = fixture(tmp_path)
    research.select(1)
    pid = agenda.plan(spec["idea_ids"], spec=spec)["protocol_id"]
    agenda.execute(pid, author="Researcher")
    resumed = ResearchSession(MockLLM(), research.profile, research.out)
    assert resumed.stage == "ideated" and resumed.handoff is None
    assert not (research.out / "handoff.json").exists()
    assert json.loads((research.out / "idea_pending_decisions.json").read_text())[0]["kind"] == "interpret_pilot"
    decision = resumed.select(1)["selection_decision"]
    assert decision["pilot_protocol_ids"] == [pid] and decision["pilot_observation_hash"]


@pytest.mark.skipif(not all(importlib.util.find_spec(p) for p in ("torch", "sklearn")), reason="optional torch engine")
def test_pilot_with_real_cpu_engine_never_scores_test_split(tmp_path):
    from scholarloop.core.profile import load_profile
    profile = load_profile(Path(__file__).resolve().parents[2] / "profiles/digits-mlp.yaml")
    research = ResearchSession(MockLLM(jsons=[QUERIES, REVIEW, {"ideas": [_idea("Learning-rate pilot")]}]),
                               profile, tmp_path, topic="Learning-rate feasibility",
                               sources=[ArxivClient(fetcher=lambda q, n: ATOM)])
    research.review()
    idea = research.ideate(1)[0]
    agenda = PilotAgenda(None, profile, tmp_path, ledger_path=tmp_path / "experiments.jsonl",
                         registry_dir=tmp_path / "registry")
    spec = copy.deepcopy(SPEC)
    spec.pop("issue_ids")
    spec.update(purpose="feasibility", fidelity="smoke", seeds=[0], idea_ids=[idea.idea_id])
    for condition in spec["conditions"]:
        condition["idea_id"] = "control" if condition["name"] == spec["reference"] else idea.idea_id
        condition["config"] = [{"name": "lr", "value": 0.05 if condition["idea_id"] == "control" else 0.1},
                                {"name": "epochs", "value": 1}, {"name": "hidden", "value": 8}]
    pid = agenda.plan([idea.idea_id], spec=spec)["protocol_id"]
    state = agenda.execute(pid, author="Test researcher")
    assert len(state["results"]) == 2
    for result in state["results"]:
        metric = result["entry"]["metric"]
        assert metric["seed_ids"] == [0] and metric["folds"] == [0]
        assert 0 <= metric["smoke"] <= 100 and "holdout" not in metric


def test_runner_itself_refuses_full_tier_feasibility(tmp_path):
    from scholarloop.core.ledger import Hypothesis
    from scholarloop.experiment.runner import run_experiment
    research, agenda, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="Feasibility"):
        run_experiment(research.profile, "full", "unexpected", Hypothesis("Pilot", "source"),
                       purpose="feasibility", ledger_path=agenda.review.ledger_path,
                       registry_dir=agenda.review.registry_dir)
    assert not (agenda.review.registry_dir / "unexpected.json").exists()
