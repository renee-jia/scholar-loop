"""FROZEN module for the digits-mlp engine — data, model construction, and evaluation.
Never edit this file. It is the trust anchor:

  1. `load_train()` hands the agent ONLY the training split — the val set never leaves here.
  2. `build_model()` is frozen, so the model the scorer reconstructs is defined by trusted code;
     the agent shapes it through hyperparameters, not by injecting classes the scorer must unpickle.
  3. `score(artifact)` loads a state_dict with `weights_only=True` (no arbitrary-object unpickle,
     so a malicious artifact can't run code in the trusted scorer) into the frozen architecture,
     evaluates on the frozen val split, and emits the trusted number. The runner runs this from
     pristine ROOT, so an edited train.py cannot reach it.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, train_test_split

METRIC_NAME = "val_top1_err"
_IN_DIM = 64       # digits are 8x8
_N_CLASSES = 10
N_FOLDS = 5        # the pool outside the test split is cut into this many CV folds


def _split(fold: int = 0):
    """Train / val / held-out test. A fixed, stratified 20% is the held-out test split (scored only
    on full-tier runs, never selected on). The other 80% is cut into N_FOLDS stratified folds:
    `fold` is the val split (the metric the loop selects on) and the rest is training data. Without
    cross-validation the runner never passes a fold, so val is fold 0."""
    d = load_digits()
    x = (d.data.astype("float32") / 16.0)
    y = d.target.astype("int64")
    x_pool, x_te, y_pool, y_te = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(x_pool, y_pool))
    tr, va = folds[fold]
    return x_pool[tr], x_pool[va], x_te, y_pool[tr], y_pool[va], y_te


def _fold_from_env() -> int:
    """The CV fold the runner assigned this training run ($SCHOLARLOOP_FOLD, default 0)."""
    return int(os.environ.get("SCHOLARLOOP_FOLD", "0"))


def load_train():
    """Training data only — the agent never receives the validation split."""
    x_tr, _x_val, _x_te, y_tr, _y_val, _y_te = _split(_fold_from_env())
    return x_tr, y_tr


def _eval_split(split: str = "val", fold: int = 0):
    _x_tr, x_val, x_te, _y_tr, y_val, y_te = _split(fold)
    return (x_te, y_te) if split == "test" else (x_val, y_val)


def build_model(hparams: dict):
    """Frozen architecture, parameterized by hyperparameters (depth / hidden width)."""
    import torch.nn as nn

    layers, d = [], _IN_DIM
    for _ in range(int(hparams["depth"])):
        layers += [nn.Linear(d, int(hparams["hidden"])), nn.ReLU()]
        d = int(hparams["hidden"])
    layers += [nn.Linear(d, _N_CLASSES)]
    return nn.Sequential(*layers)


def evaluate(model, split: str = "val", fold: int = 0) -> float:
    """Top-1 error (%) of `model` on a frozen split (val fold `fold`, or the held-out test). Lower
    is better."""
    import torch

    x_val, y_val = _eval_split(split, fold)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_val)).argmax(1).numpy()
    return round(float((pred != y_val).mean() * 100.0), 4)


def emit_result(value: float, seeds: list[float] | None = None, config: dict | None = None) -> None:
    print("SCHOLARLOOP_RESULT " + json.dumps(
        {"metric_name": METRIC_NAME, "value": float(value),
         "seeds": seeds or [value], "config": config or {}}))


def score(artifact_path: str, split: str = "val", fold: int = 0) -> None:
    """Trusted metric: rebuild the frozen model from config, load the agent's weights (safely),
    and score on the frozen val set."""
    import torch

    bundle = torch.load(artifact_path, weights_only=True)   # state_dict only — no code execution
    cfg = bundle["config"]
    model = build_model(cfg)
    model.load_state_dict(bundle["state_dict"])
    emit_result(evaluate(model, split, fold), config=cfg)


def _main(argv: list[str]) -> int:
    parsed = parse_score_args(argv)
    if parsed is None:
        print("usage: python -m engines.torch_vision.prepare score <artifact> [test | val [fold]]",
              file=sys.stderr)
        return 2
    score(*parsed)
    return 0


def parse_score_args(argv: list[str]):
    """`score <artifact> [val|test] [fold]` -> (artifact, split, fold), or None if malformed. The
    fold is the one the model was trained without (the val split; for test it only selects the
    training statistics the model was fit with)."""
    if not 2 <= len(argv) <= 4 or argv[0] != "score":
        return None
    split = argv[2] if len(argv) > 2 else "val"
    fold = argv[3] if len(argv) > 3 else "0"
    if split not in ("val", "test") or not fold.isdigit() or int(fold) >= N_FOLDS:
        return None
    return argv[1], split, int(fold)

if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
