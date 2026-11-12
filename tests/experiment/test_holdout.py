"""Explicit final-test access and campaign closure after recorded confirmation attempts."""

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from scholarloop.core.ledger import Hypothesis, Ledger
from scholarloop.core.llm import MockLLM
from scholarloop.core.profile import load_profile
from scholarloop.core.registry import VerifiedRegistry
from scholarloop.experiment.orchestrator import Orchestrator
from scholarloop.experiment.runner import run_experiment

ROOT = Path(__file__).resolve().parents[2]
STUB = dataclasses.replace(load_profile(ROOT / "profiles" / "image-classification.yaml"), holdout=True)
IDEAL = {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}
_NO_TORCH = importlib.util.find_spec("torch") is None or importlib.util.find_spec("sklearn") is None


def _run(profile, fidelity, eid, tmp_path, config=IDEAL):
    return run_experiment(profile, fidelity, eid, Hypothesis(claim="probe", source="arXiv:1"),
                          config_override=config, ledger_path=tmp_path / "l.jsonl",
                          registry_dir=tmp_path / "reg", purpose="confirmation",
                          holdout_override=profile.holdout and fidelity == "full")


def test_full_tier_scores_holdout_apart_from_selection_metric(tmp_path):
    e = _run(STUB, "full", "exp_0001", tmp_path)
    held = e.metric["holdout"]
    assert len(held["seeds"]) == len(e.metric["seeds"]) == 5          # one holdout score per seed
    assert e.holdout_score() == held["value"]
    assert e.primary_score() == e.metric["full"] != e.holdout_score()   # selection still reads val
    reg = VerifiedRegistry.load(tmp_path / "reg" / "exp_0001.json")
    assert reg.is_grounded(e.holdout_score())                          # a paper may cite it


@pytest.mark.parametrize("fidelity", ["smoke", "verify"])
def test_lower_tiers_never_touch_the_holdout(tmp_path, fidelity):
    e = _run(STUB, fidelity, "exp_0001", tmp_path)
    assert "holdout" not in e.metric and e.holdout_score() is None


def test_holdout_off_by_default(tmp_path):
    e = _run(dataclasses.replace(STUB, holdout=False), "full", "exp_0001", tmp_path)
    assert e.holdout_score() is None


def test_engine_without_holdout_support_warns_but_keeps_the_val_result(tmp_path, capsys):
    # a misconfigured profile must not turn an idea into a "killed" run the loop learns from
    cheater = dataclasses.replace(load_profile(ROOT / "profiles" / "cheater.yaml"), holdout=True)
    honest = _run(dataclasses.replace(cheater, holdout=False), "full", "exp_0001", tmp_path, config=None)
    e = _run(cheater, "full", "exp_0002", tmp_path, config=None)
    assert e.verdict == honest.verdict != "killed" and e.primary_score() == honest.primary_score()
    assert e.holdout_score() is None and "test" in e.metric["holdout"]["error"]
    assert "no held-out score" in capsys.readouterr().err


def test_one_failed_holdout_seed_keeps_every_val_seed(tmp_path):
    from scholarloop.experiment import runner
    from scholarloop.core.registry import VerifiedRegistry as Reg
    results = [{"value": 4.0, "holdout_value": 5.0}, {"value": 4.2, "holdout_error": "seed 1: timeout"},
               {"value": 4.1, "holdout_value": 5.2}]
    blob = runner._aggregate_holdout(results, Reg(tmp_path / "r.json", exp_id="x"), "err", "x")
    assert blob == {"value": 5.1, "seeds": [5.0, 5.2], "std": 0.1414, "seeds_failed": 1}


def test_no_agent_prompt_ever_contains_the_holdout_number(tmp_path):
    # round 1 climbs to full (the stub's ideal config) and records a holdout score; round 2's
    # Reasoner reads that ledger. Neither the value nor the word may reach any prompt.
    def proposal(cfg):
        return {"rationale": "t", "config": [{"name": k, "value": v} for k, v in cfg.items()],
                "hypothesis": {"claim": "c", "source": "arXiv:1", "predicted_effect": "lower"},
                "predicted_delta": -0.1}
    llm = MockLLM(jsons=[proposal(IDEAL), proposal({**IDEAL, "depth": 22})])
    orch = Orchestrator(llm, STUB, ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "reg")
    orch.funnel_step()
    orch.funnel_step()
    held = [e.holdout_score() for e in Ledger(tmp_path / "l.jsonl").read_all() if e.holdout_score() is not None]
    assert not held, "exploration must never consume the final test split"
    assert any(e.fidelity == ["full"] for e in Ledger(tmp_path / "l.jsonl").read_all())
    prompts = "\n".join(c["prompt"] + (c["system"] or "") for c in llm.calls)
    assert len(llm.calls) == 2
    for v in held:
        assert str(v) not in prompts
    assert "holdout" not in prompts.lower() and "held-out" not in prompts.lower()


def test_writing_evidence_reports_the_holdout(tmp_path):
    from scholarloop.writing.paper import gather_findings
    from scholarloop.writing.session import EvidenceBank

    e = _run(STUB, "full", "exp_0001", tmp_path)
    assert gather_findings([e])[0]["heldout_test_score"] == e.holdout_score()
    bank = EvidenceBank.build([e], tmp_path / "reg", STUB)
    assert f"held-out test val_top1_err={e.holdout_score()}" in bank.describe("exp_0001")
    assert bank.ungrounded(f"On held-out data the error is {e.holdout_score()}.") == []


@pytest.mark.skipif(_NO_TORCH, reason="torch/sklearn not installed")
def test_torch_engine_serves_a_real_holdout(tmp_path):
    profile = load_profile(ROOT / "profiles" / "digits-mlp.yaml")
    assert profile.holdout
    e = _run(profile, "full", "exp_0001", tmp_path, config={"epochs": 10})
    assert 0.0 <= e.holdout_score() <= 100.0 and len(e.metric["holdout"]["seeds"]) == 5


@pytest.mark.parametrize("purpose", ["exploration", "baseline", "ablation", "falsification", "feasibility"])
def test_non_confirmation_cannot_request_test_access(tmp_path, purpose):
    with pytest.raises(ValueError):
        run_experiment(STUB, "full", "blocked", Hypothesis("probe", "fixture"),
                       purpose=purpose, holdout_override=True,
                       ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")
    assert not (tmp_path / "ledger.jsonl").exists()
    assert not (tmp_path / "registry").exists()


def test_confirmation_requires_explicit_test_opt_in(tmp_path):
    e = run_experiment(STUB, "full", "validation_only", Hypothesis("probe", "fixture"),
                       purpose="confirmation", ledger_path=tmp_path / "ledger.jsonl",
                       registry_dir=tmp_path / "registry")
    assert e.holdout_score() is None


def test_test_access_closes_exploration_before_any_model_call(tmp_path):
    _run(STUB, "full", "confirmed", tmp_path)
    llm = MockLLM()
    orch = Orchestrator(llm, STUB, ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "reg")
    with pytest.raises(ValueError, match="Final test access"):
        orch.step()
    assert llm.calls == []
    with pytest.raises(ValueError, match="Final test access"):
        run_experiment(STUB, "smoke", "late", Hypothesis("probe", "fixture"),
                       ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "reg")
    assert len(list(Ledger(tmp_path / "l.jsonl").read_all())) == 1


def test_test_access_blocks_new_planning_before_loading_or_prompting(tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from scholarloop.experiment.agenda import ExperimentAgenda
    from scholarloop.review.contracts import ReviewBlocked
    _run(STUB, "full", "confirmed", tmp_path)
    review = SimpleNamespace(out=tmp_path, registry_dir=tmp_path / "reg",
                             ledger_path=tmp_path / "l.jsonl", _lock=nullcontext)
    agenda = ExperimentAgenda(review)
    with pytest.raises(ReviewBlocked, match="Final test access"):
        agenda.plan()
    with pytest.raises(ReviewBlocked, match="Final test access"):
        agenda._store_plan({})


def test_failed_test_access_only_allows_original_confirmation_protocol(tmp_path):
    from scholarloop.core.ledger import LedgerEntry
    from scholarloop.experiment.test_boundary import require_unopened_test
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(LedgerEntry(id="attempt", domain=STUB.name, metric_name=STUB.metric.name,
                  hypothesis=Hypothesis("probe", "fixture"), verdict="killed",
                  purpose="confirmation", protocol={"id": "frozen-before-test"},
                  run_spec={"holdout": True}))
    with pytest.raises(ValueError):
        require_unopened_test(ledger.path)
    with pytest.raises(ValueError):
        require_unopened_test(ledger.path, continuing_protocol="new-after-test")
    require_unopened_test(ledger.path, continuing_protocol="frozen-before-test")
