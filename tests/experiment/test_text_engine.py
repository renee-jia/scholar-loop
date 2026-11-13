"""The AG News engine runs end-to-end through the runner, including a train-script edit.

Skipped unless the cached subset exists (the first real run downloads it), so CI stays offline.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_MISSING = importlib.util.find_spec("torch") is None or importlib.util.find_spec("sklearn") is None


def _cached() -> bool:
    from engines.text_cls import prepare
    return prepare._data_path().exists()


pytestmark = pytest.mark.skipif(_MISSING or not _cached(), reason="torch/sklearn or the AG News cache is absent")


def _run(tmp_path, exp_id, **kwargs):
    from scholarloop.core.ledger import Hypothesis
    from scholarloop.core.profile import load_profile
    from scholarloop.experiment.runner import run_experiment
    return run_experiment(load_profile(ROOT / "profiles" / "agnews-bow.yaml"), "smoke", exp_id,
                          Hypothesis(claim="probe", source="arXiv:1509.01626"),
                          ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry", **kwargs)


def test_encoding_is_stable_and_padded():
    from engines.text_cls import prepare
    ids = prepare.encode("Wall St. Bears claw back")
    assert ids.shape == (prepare.MAX_LEN,) and ids[5] == 0 and (ids[:5] > 0).all()
    assert (ids == prepare.encode("wall st bears CLAW back")).all()
    assert 0 < ids.max() <= prepare.VOCAB


def test_splits_are_disjoint_in_size_and_stratified():
    from engines.text_cls import prepare
    x_tr, x_va, x_te, y_tr, y_va, y_te = prepare._split(0)
    assert (len(x_tr), len(x_va), len(x_te)) == (6400, 1600, 2000)
    assert all((y_va == c).sum() == 400 and (y_te == c).sum() == 500 for c in range(prepare.N_CLASSES))
    assert len(prepare.load_train()[0]) == 6400


def test_real_training_run(tmp_path):
    entry = _run(tmp_path, "exp_text", config_override={"epochs": 2})
    assert entry.verdict in ("kept", "discarded")
    assert 5.0 < entry.primary_score() < 40.0          # a trained 4-class model, not chance (75)
    assert entry.config["epochs"] == 2


def test_train_script_edit_runs_through_the_edit_channel(tmp_path):
    """A method change (label smoothing) is applied as a full train-file replacement in an
    isolated copy; the frozen scorer still produces the number."""
    path = "engines/text_cls/train.py"
    source = (ROOT / path).read_text()
    edited = source.replace("nn.CrossEntropyLoss()", "nn.CrossEntropyLoss(label_smoothing=0.1)")
    assert edited != source
    entry = _run(tmp_path, "exp_edit", config_override={"epochs": 2},
                 edits=[{"path": path, "content": edited}])
    assert 5.0 < entry.primary_score() < 40.0
    assert (ROOT / path).read_text() == source          # the repository file is untouched
