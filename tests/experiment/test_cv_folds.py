"""K-fold cross-validation for small datasets: seed s trains and validates on fold s % K, the fold
reaches the frozen scorer through the runner's argv, and the held-out test split stays outside
every fold."""

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from scholarloop.core.ledger import Hypothesis
from scholarloop.core.profile import ProfileError, load_profile
from scholarloop.experiment.runner import fold_for_seed, run_experiment

ROOT = Path(__file__).resolve().parents[2]
STUB = load_profile(ROOT / "profiles" / "image-classification.yaml")
IDEAL = {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}
_NO_TORCH = importlib.util.find_spec("torch") is None or importlib.util.find_spec("sklearn") is None


def _run(profile, fidelity, tmp_path, eid="exp_0001"):
    return run_experiment(profile, fidelity, eid, Hypothesis(claim="probe", source="arXiv:1"),
                          config_override=IDEAL, ledger_path=tmp_path / "l.jsonl",
                          registry_dir=tmp_path / "reg")


def test_seed_fold_pairing():
    assert [fold_for_seed(s, 5) for s in range(7)] == [0, 1, 2, 3, 4, 0, 1]
    assert fold_for_seed(3, 0) is None


def test_full_tier_covers_every_fold_and_records_them(tmp_path):
    cv = _run(dataclasses.replace(STUB, cv_folds=5), "full", tmp_path)
    plain = _run(STUB, "full", tmp_path, "exp_0002")
    assert cv.metric["folds"] == [0, 1, 2, 3, 4] and "folds" not in plain.metric
    # fold 0 is the plain val split; every other seed is scored on a different fold
    assert cv.metric["seeds"][0] == plain.metric["seeds"][0]
    assert cv.metric["seeds"][1:] != plain.metric["seeds"][1:]


def test_cv_and_holdout_together(tmp_path):
    e = _run(dataclasses.replace(STUB, cv_folds=5, holdout=True), "full", tmp_path)
    assert len(e.metric["holdout"]["seeds"]) == 5 and e.holdout_score() is not None


def test_engine_without_fold_support_kills_the_run(tmp_path):
    # a misconfigured cv_folds must not silently fall back to one split
    cheater = dataclasses.replace(load_profile(ROOT / "profiles" / "cheater.yaml"), cv_folds=5)
    e = run_experiment(cheater, "smoke", "exp_0001", Hypothesis(claim="c", source="arXiv:1"),
                       ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "reg")
    assert e.verdict == "killed"


def test_one_fold_is_rejected(tmp_path):
    raw = (ROOT / "profiles" / "image-classification.yaml").read_text() + "\ncv_folds: 1\n"
    (tmp_path / "p.yaml").write_text(raw)
    with pytest.raises(ProfileError):
        load_profile(tmp_path / "p.yaml")


@pytest.mark.skipif(_NO_TORCH, reason="torch/sklearn not installed")
@pytest.mark.parametrize("engine", ["torch_vision", "torch_regression"])
def test_torch_folds_partition_the_pool_and_never_touch_test(engine):
    import importlib

    import numpy as np
    prep = importlib.import_module(f"engines.{engine}.prepare")
    key = lambda xs: {r.tobytes() for r in xs}          # noqa: E731
    vals, test = [], None
    for k in range(prep.N_FOLDS):
        x_tr, x_val, x_te, *_ = prep._split(k)
        assert not key(x_tr) & key(x_val)                       # train never contains its val fold
        assert not (key(x_tr) | key(x_val)) & key(x_te)         # nor the held-out test
        vals.append(key(x_val))
        test = x_te if test is None else test
        assert np.array_equal(test, x_te)                      # the test split is fold-independent
    assert sum(map(len, vals)) == len(set().union(*vals))      # the val folds are disjoint ...
    assert len(set().union(*vals)) == len(x_tr) + len(x_val)   # ... and cover the pool


@pytest.mark.skipif(_NO_TORCH, reason="torch/sklearn not installed")
def test_torch_score_args():
    from engines.torch_vision.prepare import parse_score_args
    assert parse_score_args(["score", "a"]) == ("a", "val", 0)
    assert parse_score_args(["score", "a", "val", "3"]) == ("a", "val", 3)
    assert parse_score_args(["score", "a", "test", "2"]) == ("a", "test", 2)
    for bad in (["score", "a", "val", "5"], ["score", "a", "val", "-1"], ["score", "a", "train"],
                ["score"], ["eval", "a"]):
        assert parse_score_args(bad) is None


@pytest.mark.skipif(_NO_TORCH, reason="torch/sklearn not installed")
def test_load_train_follows_the_fold_env(monkeypatch):
    from engines.torch_regression import prepare
    monkeypatch.setenv("SCHOLARLOOP_FOLD", "3")
    x_tr, _ = prepare.load_train()
    assert {r.tobytes() for r in x_tr} == {r.tobytes() for r in prepare._split(3)[0]}
