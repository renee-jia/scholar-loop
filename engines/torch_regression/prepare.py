"""FROZEN module for the diabetes-mlp regression engine — data, model, evaluation.
Never edit this file. Same trust guarantees as the other engines: the agent gets
the training split only, and the trusted RMSE is computed here from a state_dict loaded with
weights_only=True.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, train_test_split

METRIC_NAME = "val_rmse"
_IN_DIM = 10
N_FOLDS = 5        # the pool outside the test split is cut into this many CV folds


def _split(fold: int = 0):
    """A fixed 20% held-out test split (full tier only, never selected on); the other 80% is cut
    into N_FOLDS folds, `fold` is val (selected on) and the rest is training data. With 442 samples
    one val split is ~70 points, too noisy to select on alone; that is what cv_folds is for."""
    d = load_diabetes()
    x = d.data.astype("float32")          # sklearn diabetes features are already mean-centered/scaled
    y = d.target.astype("float32")        # continuous disease-progression score (~25–346)
    x_pool, x_te, y_pool, y_te = train_test_split(x, y, test_size=0.2, random_state=0)
    tr, va = list(KFold(N_FOLDS, shuffle=True, random_state=0).split(x_pool))[fold]
    return x_pool[tr], x_pool[va], x_te, y_pool[tr], y_pool[va], y_te


def _fold_from_env() -> int:
    """The CV fold the runner assigned this training run ($SCHOLARLOOP_FOLD, default 0)."""
    return int(os.environ.get("SCHOLARLOOP_FOLD", "0"))


def _y_stats(fold: int = 0):
    """Mean/std of the fold's training target — used to standardize for stable training."""
    y_tr = _split(fold)[3]
    return float(y_tr.mean()), float(y_tr.std())


def load_train():
    fold = _fold_from_env()
    x_tr, _xv, _xt, y_tr, _yv, _yt = _split(fold)
    m, s = _y_stats(fold)
    return x_tr, ((y_tr - m) / s).astype("float32")   # standardized target → stable gradients


def _eval_split(split: str = "val", fold: int = 0):
    _x_tr, x_val, x_te, _y_tr, y_val, y_te = _split(fold)
    return (x_te, y_te) if split == "test" else (x_val, y_val)   # original-unit target


def build_model(hparams: dict):
    """Frozen architecture (regressor: single output), parameterized by hyperparameters."""
    import torch.nn as nn

    layers, d = [], _IN_DIM
    for _ in range(int(hparams["depth"])):
        layers += [nn.Linear(d, int(hparams["hidden"])), nn.ReLU()]
        d = int(hparams["hidden"])
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


def evaluate(model, split: str = "val", fold: int = 0) -> float:
    """RMSE on a frozen split (val fold `fold`, or the held-out test). Lower is better. The model
    was trained on fold `fold`'s standardized target, so predictions are un-standardized with it."""
    import torch

    x_val, y_val = _eval_split(split, fold)
    m, s = _y_stats(fold)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_val)).squeeze(-1).numpy() * s + m   # back to original units
    return round(float(np.sqrt(np.mean((pred - y_val) ** 2))), 4)


def emit_result(value: float, seeds: list[float] | None = None, config: dict | None = None) -> None:
    print("SCHOLARLOOP_RESULT " + json.dumps(
        {"metric_name": METRIC_NAME, "value": float(value),
         "seeds": seeds or [value], "config": config or {}}))


def score(artifact_path: str, split: str = "val", fold: int = 0) -> None:
    import torch

    bundle = torch.load(artifact_path, weights_only=True)
    cfg = bundle["config"]
    model = build_model(cfg)
    model.load_state_dict(bundle["state_dict"])
    emit_result(evaluate(model, split, fold), config=cfg)


def _main(argv: list[str]) -> int:
    parsed = parse_score_args(argv)
    if parsed is None:
        print("usage: python -m engines.torch_regression.prepare score <artifact> [test | val [fold]]",
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
