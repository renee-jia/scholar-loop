"""The seq-distill engine: disjoint frozen splits, a validity scorer, and a real run through the
runner in both off-policy and on-policy mode.

Training tests are skipped unless the cached teacher exists (the first real run trains it for
about a minute), so CI stays fast.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is absent")


def _teacher_cached() -> bool:
    from engines.seq_distill import prepare
    return prepare._teacher_path().exists()


needs_teacher = pytest.mark.skipif(importlib.util.find_spec("torch") is None or not _teacher_cached(),
                                   reason="the cached teacher is absent")


def test_splits_are_disjoint_and_the_teacher_never_sees_them():
    from engines.seq_distill import prepare
    as_set = lambda a: {tuple(r) for r in a.tolist()}
    train, val, test = (as_set(a) for a in prepare._split(0))
    teacher = as_set(prepare._inputs()["teacher"])
    assert (len(train), len(val), len(test)) == (2000, 500, 2000)
    assert not (train & val or train & test or val & test or teacher & (train | val | test))
    assert len(teacher) == prepare.ALPHABET ** prepare.LENGTH - 4500
    assert as_set(prepare._split(1)[1]) != val


def test_scorer_accepts_each_mode_and_rejects_a_mixture():
    import torch
    from engines.seq_distill import prepare

    class Scripted(torch.nn.Module):
        """Emits a fixed output per input: mode k, or the first half of copy and the rest of reverse."""
        def __init__(self, mode):
            super().__init__()
            self.mode = mode

        def forward(self, ids):
            x, t = ids[:, :prepare.LENGTH], ids.shape[1] - prepare.LENGTH - 1
            outs = prepare.valid_outputs(x)
            y = outs[self.mode] if self.mode < 3 else torch.where(
                torch.arange(prepare.LENGTH) < 3, outs[0], outs[1] + 1)
            logits = torch.zeros(len(ids), ids.shape[1], prepare.VOCAB)
            logits[torch.arange(len(ids)), -1, y[:, t]] = 1.0
            return logits

    assert [prepare.evaluate(Scripted(k)) for k in range(3)] == [0.0, 0.0, 0.0]
    assert prepare.evaluate(Scripted(3)) > 90.0
    assert prepare.parse_score_args(["score", "a.pt", "test"]) == ("a.pt", "test", 0)
    assert prepare.parse_score_args(["score", "a.pt", "val", "9"]) is None


def _run(tmp_path, exp_id, **kwargs):
    from scholarloop.core.ledger import Hypothesis
    from scholarloop.core.profile import load_profile
    from scholarloop.experiment.runner import run_experiment
    return run_experiment(load_profile(ROOT / "profiles" / "seq-distill.yaml"), "smoke", exp_id,
                          Hypothesis(claim="probe", source="arXiv:2306.13649"),
                          ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry", **kwargs)


@needs_teacher
def test_teacher_is_frozen_and_nearly_always_valid():
    from engines.seq_distill import prepare
    teacher = prepare.load_teacher()
    assert not any(p.requires_grad for p in teacher.parameters())
    assert prepare.evaluate(teacher, "test") < 2.0


@needs_teacher
@pytest.mark.parametrize("frac", [0.0, 1.0])
def test_real_distillation_run_off_and_on_policy(tmp_path, frac, monkeypatch):
    monkeypatch.setenv("SCHOLARLOOP_DISTILL_STEPS", "150")
    entry = _run(tmp_path, "exp_distill", config_override={"on_policy_frac": frac})
    assert entry.verdict in ("kept", "discarded")
    assert 0.0 <= entry.primary_score() <= 100.0
    assert entry.config["on_policy_frac"] == frac and entry.metric_name == "val_invalid_rate"
    # the training script's own report travels with the result, one value per seed
    assert entry.metric["self_reported"]["on_policy_batches"] == [150.0 * frac]


def test_student_size_cannot_be_changed_by_a_run():
    from engines.seq_distill import prepare
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(prepare.build_model({"d_model": 512, "n_layers": 8})) == count(prepare.build_model())
    from scholarloop.core.profile import load_profile
    assert set(load_profile(ROOT / "profiles" / "seq-distill.yaml").knobs) == {"lr", "on_policy_frac", "temperature"}
