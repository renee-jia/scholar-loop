"""The frozen-scoring guard: the runner trusts only the frozen scorer's number,
so an edited train.py can neither fabricate the metric nor be the source of truth."""

from pathlib import Path

from scholarloop.ledger import Hypothesis
from scholarloop.profile import load_profile
from scholarloop.runner import run_experiment

ROOT = Path(__file__).resolve().parent.parent


def test_fabricated_train_output_is_ignored(tmp_path):
    # cheater train.py prints a fake "0.1"; the frozen scorer reports the honest 50.0
    profile = load_profile(ROOT / "profiles" / "cheater.yaml")
    entry = run_experiment(
        profile, "smoke", "exp_cheat",
        Hypothesis(claim="claims a perfect score", source="arXiv:0000.0"),
        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")
    assert entry.primary_score() == 50.0     # the frozen number, NOT the fabricated 0.1
    assert entry.verdict == "discarded"      # 50 > baseline 10 -> the cheat did not win


def test_recorded_metric_comes_from_the_frozen_scorer(tmp_path):
    # the recorded number must equal an independent recompute of the frozen evaluate
    from engines.vision import prepare as stub_prepare

    profile = load_profile(ROOT / "profiles" / "image-classification.yaml")
    entry = run_experiment(
        profile, "smoke", "exp_s", Hypothesis(claim="default", source="arXiv:1"),
        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")
    assert entry.primary_score() == stub_prepare.evaluate(entry.config, 0)
