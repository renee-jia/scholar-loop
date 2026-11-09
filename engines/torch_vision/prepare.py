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
import sys

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

METRIC_NAME = "val_top1_err"
_IN_DIM = 64       # digits are 8x8
_N_CLASSES = 10


def _split():
    """Train / val / held-out test. The 25% split off the data is halved into val (the metric the
    loop selects on, every run) and test (scored only on full-tier runs, never selected on)."""
    d = load_digits()
    x = (d.data.astype("float32") / 16.0)
    y = d.target.astype("int64")
    x_tr, x_ev, y_tr, y_ev = train_test_split(x, y, test_size=0.25, random_state=0, stratify=y)
    x_val, x_te, y_val, y_te = train_test_split(x_ev, y_ev, test_size=0.5, random_state=0, stratify=y_ev)
    return x_tr, x_val, x_te, y_tr, y_val, y_te


def load_train():
    """Training data only — the agent never receives the validation split."""
    x_tr, _x_val, _x_te, y_tr, _y_val, _y_te = _split()
    return x_tr, y_tr


def _eval_split(split: str = "val"):
    _x_tr, x_val, x_te, _y_tr, y_val, y_te = _split()
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


def evaluate(model, split: str = "val") -> float:
    """Top-1 error (%) of `model` on a frozen split (val, or the held-out test). Lower is better."""
    import torch

    x_val, y_val = _eval_split(split)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_val)).argmax(1).numpy()
    return round(float((pred != y_val).mean() * 100.0), 4)


def emit_result(value: float, seeds: list[float] | None = None, config: dict | None = None) -> None:
    print("SCHOLARLOOP_RESULT " + json.dumps(
        {"metric_name": METRIC_NAME, "value": float(value),
         "seeds": seeds or [value], "config": config or {}}))


def score(artifact_path: str, split: str = "val") -> None:
    """Trusted metric: rebuild the frozen model from config, load the agent's weights (safely),
    and score on the frozen val set."""
    import torch

    bundle = torch.load(artifact_path, weights_only=True)   # state_dict only — no code execution
    cfg = bundle["config"]
    model = build_model(cfg)
    model.load_state_dict(bundle["state_dict"])
    emit_result(evaluate(model, split), config=cfg)


def _main(argv: list[str]) -> int:
    if len(argv) in (2, 3) and argv[0] == "score" and (argv[2:] or ["val"])[0] in ("val", "test"):
        score(argv[1], *argv[2:])
        return 0
    print("usage: python -m engines.torch_vision.prepare score <artifact> [val|test]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
