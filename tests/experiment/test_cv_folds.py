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
def test_torch_folds_partition_the_pool_and_never_touch_test():
    import numpy as np
    from engines.torch_vision import prepare as prep
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
    from engines.torch_vision import prepare
    monkeypatch.setenv("SCHOLARLOOP_FOLD", "3")
    x_tr, _ = prepare.load_train()
    assert {r.tobytes() for r in x_tr} == {r.tobytes() for r in prepare._split(3)[0]}


def _entry(eid, tier, seeds, verdict="kept"):
    from scholarloop.core.ledger import LedgerEntry
    return LedgerEntry(id=eid, domain=STUB.name, hypothesis=Hypothesis(claim="c", source="arXiv:1"),
                       metric_name=STUB.metric.name, fidelity=[tier], verdict=verdict,
                       metric={"name": STUB.metric.name, tier: round(sum(seeds) / len(seeds), 4),
                               "seeds": seeds, "folds": list(range(len(seeds)))})


def test_tiers_are_compared_on_shared_folds(tmp_path):
    """Fold 4 is hard here, as on digits. A 3-fold verify mean never sees it, so comparing it with
    a 5-fold mean flatters it: the frontier stays the full run, and gates use its first folds."""
    from scholarloop.core.ledger import Ledger
    from scholarloop.experiment.orchestrator import Orchestrator
    cv = dataclasses.replace(STUB, cv_folds=5, baselines=[])
    ledger = Ledger(tmp_path / "l.jsonl")
    full = ledger.append(_entry("exp_0001", "full", [3.0, 3.0, 3.0, 3.0, 5.0]))      # mean 3.4
    ledger.append(_entry("exp_0002", "verify", [3.1, 3.1, 3.1]))                     # 3.1 < 3.4 ...
    orch = Orchestrator(None, cv, ledger_path=ledger.path, registry_dir=tmp_path / "r")
    assert orch._parent([]).id == full.id                  # ... but worse than full on folds 0-2
    candidate = _entry("exp_0003", "verify", [3.05, 3.05, 3.05])
    assert orch._gate_for(candidate, full) == 3.0          # frontier on folds 0-2, not 3.4
    assert not orch._promote(candidate, "verify", orch._gate_for(candidate, full))
    assert orch._gate_for(_entry("exp_0004", "smoke", [2.0]), full) == 3.0
    plain = Orchestrator(None, dataclasses.replace(cv, cv_folds=0), ledger_path=ledger.path,
                         registry_dir=tmp_path / "r")
    assert plain._gate_for(candidate, full) == 3.4         # no CV: the old, unmatched comparison


def test_a_climb_trains_each_seed_once_and_still_scores_the_holdout(tmp_path):
    """verify re-scores smoke's model and full re-scores verify's: same numbers, fewer trainings,
    and the full tier's held-out score covers every seed, reused ones included."""
    from scholarloop.experiment.runner import _artifact_key
    cv = dataclasses.replace(STUB, cv_folds=5, holdout=True)
    cache = tmp_path / "models"
    common = dict(config_override=IDEAL, ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "reg",
                  artifact_cache=cache)
    h = Hypothesis(claim="probe", source="arXiv:1")
    verify = run_experiment(cv, "verify", "exp_0001", h, **common)
    full = run_experiment(cv, "full", "exp_0002", h, **common)
    fresh = _run(cv, "full", tmp_path, "exp_0003")                       # no cache: trains all five
    assert "reused_seeds" not in verify.metric and full.metric["reused_seeds"] == 3
    assert full.metric["seeds"] == fresh.metric["seeds"]                 # re-scoring changes nothing
    assert full.metric["seeds"][:3] == verify.metric["seeds"]
    assert len(full.metric["holdout"]["seeds"]) == 5
    assert len(list(cache.iterdir())) == 5                               # one model per seed
    other = {**IDEAL, "lr": 0.2}                                         # a different config never hits
    assert _artifact_key(other, [], ROOT / cv.train_entrypoint) != _artifact_key(IDEAL, [], ROOT / cv.train_entrypoint)
