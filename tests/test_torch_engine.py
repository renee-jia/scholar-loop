"""The real torch digits engine runs end-to-end through the runner (skipped if torch absent)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_MISSING = importlib.util.find_spec("torch") is None or importlib.util.find_spec("sklearn") is None


@pytest.mark.skipif(_MISSING, reason="torch/sklearn not installed")
def test_digits_engine_real_training_run(tmp_path):
    from scholarloop.ledger import Hypothesis
    from scholarloop.profile import load_profile
    from scholarloop.runner import run_experiment

    profile = load_profile(ROOT / "profiles" / "digits-mlp.yaml")
    entry = run_experiment(
        profile, "smoke", "exp_torch",
        Hypothesis(claim="probe a real MLP", source="arXiv:1502.01852"),
        config_override={"epochs": 10},      # keep the test fast
        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry",
    )
    assert entry.verdict in ("kept", "discarded")        # a real measurement, compared to baseline
    assert 0.0 <= entry.primary_score() <= 100.0          # a real error percentage
    assert entry.config["epochs"] == 10                   # the config override actually drove the run


@pytest.mark.skipif(_MISSING, reason="torch/sklearn not installed")
def test_diabetes_regression_engine_real_run(tmp_path):
    # a SECOND domain (regression, RMSE metric) runs through the SAME runner — zero orchestrator change
    from scholarloop.ledger import Hypothesis
    from scholarloop.profile import load_profile
    from scholarloop.runner import run_experiment

    profile = load_profile(ROOT / "profiles" / "diabetes-mlp.yaml")
    entry = run_experiment(
        profile, "smoke", "exp_reg", Hypothesis(claim="probe a regressor", source="arXiv:1"),
        config_override={"epochs": 50}, ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "r")
    assert entry.verdict in ("kept", "discarded")
    assert entry.metric_name == "val_rmse" and entry.primary_score() > 0   # a real RMSE
    assert entry.config["epochs"] == 50
