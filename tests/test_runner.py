"""End-to-end smoke test for the Phase-0 runner: profile -> subprocess -> registry -> ledger."""

import dataclasses
from pathlib import Path

import pytest

from scholarloop.ledger import Hypothesis, Ledger
from scholarloop.profile import load_profile
from scholarloop.registry import VerifiedRegistry
from scholarloop.runner import run_experiment

ROOT = Path(__file__).resolve().parent.parent


def test_run_experiment_produces_ledger_and_registry(tmp_path):
    profile = load_profile(ROOT / "profiles" / "image-classification.yaml")
    entry = run_experiment(
        profile, "smoke", "exp_test_0001",
        Hypothesis(claim="baseline probe", source="arXiv:1512.03385"),
        ledger_path=tmp_path / "ledger.jsonl",
        registry_dir=tmp_path / "registry",
    )

    # verdict is decided against the must-beat baseline; the stub baseline config loses
    assert entry.verdict in ("kept", "discarded")
    assert entry.primary_score() is not None

    # the ledger got exactly one durable record
    rows = list(Ledger(tmp_path / "ledger.jsonl").read_all())
    assert len(rows) == 1 and rows[0].id == "exp_test_0001"

    # the registry froze the measured number, and it is grounded
    reg = VerifiedRegistry.load(tmp_path / "registry" / "exp_test_0001.json")
    assert reg.status == "ok"
    assert reg.is_grounded(entry.primary_score())
    assert reg.audit_text("fabricated 1.23 result") == ["1.23"]


def test_config_override_varies_outcome_and_feeds_reasoning(tmp_path):
    # Drive several configs through the runner (no source mutation), then check the
    # deterministic reasoner mines real signal from the resulting ledger.
    from scholarloop.ledger import Ledger
    from scholarloop.reasoning import analyze_search_space

    profile = load_profile(ROOT / "profiles" / "image-classification.yaml")
    ledger_path = tmp_path / "ledger.jsonl"
    # near-optimal lr (0.1) vs far lr (0.9); ideal config gives a much lower error
    configs = {
        "exp_lo1": {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5},
        "exp_lo2": {"lr": 0.1, "depth": 18, "weight_decay": 5e-4, "warmup": 5},
        "exp_hi1": {"lr": 0.9, "depth": 20, "weight_decay": 5e-4, "warmup": 5},
        "exp_hi2": {"lr": 0.9, "depth": 18, "weight_decay": 5e-4, "warmup": 5},
    }
    for eid, cfg in configs.items():
        run_experiment(profile, "verify", eid,
                       Hypothesis(claim="sweep lr", source="arXiv:1608.03983"),
                       config_override=cfg, ledger_path=ledger_path,
                       registry_dir=tmp_path / "registry")

    entries = list(Ledger(ledger_path).read_all())
    assert len(entries) == 4
    # the effective (merged) config was captured, not the file default lr=0.05
    assert {e.config["lr"] for e in entries} == {0.1, 0.9}

    c = analyze_search_space(entries, profile)
    assert "lr" in c.focus                                    # lr clearly moves the metric
    assert any(r.startswith("lr>") for r in c.ruled_out)      # high lr region is exhausted


def test_run_seeds_tolerates_partial_failure():
    from scholarloop.runner import RunError, _run_seeds

    def run_one(s):                                  # seed 1 crashes; 0 and 2 succeed
        if s == 1:
            raise RunError("seed 1 crashed")
        return {"value": 5.0 + s}

    results, failures = _run_seeds([0, 1, 2], run_one)
    assert [r["value"] for r in results] == [5.0, 7.0]   # the survivors are kept
    assert len(failures) == 1

    def always_fail(s):
        raise RunError("boom")

    with pytest.raises(RunError):                    # every seed failing still kills the run
        _run_seeds([0, 1], always_fail)


def test_run_experiment_kill_path(tmp_path):
    # a train entrypoint that cannot be imported -> nonzero exit -> "killed" verdict,
    # registry marked killed with no measurements (not a clean empty run).
    profile = load_profile(ROOT / "profiles" / "image-classification.yaml")
    broken = dataclasses.replace(profile, train_entrypoint="engines/vision/does_not_exist.py")
    entry = run_experiment(
        broken, "smoke", "exp_kill_0001",
        Hypothesis(claim="will fail", source="arXiv:0000.00000"),
        ledger_path=tmp_path / "ledger.jsonl",
        registry_dir=tmp_path / "registry",
    )
    assert entry.verdict == "killed"
    assert entry.primary_score() is None

    reg = VerifiedRegistry.load(tmp_path / "registry" / "exp_kill_0001.json")
    assert reg.status == "killed"
    assert reg.measurements == {}
