"""FROZEN module for the diabetes-mlp regression engine — data, model, evaluation.
Never edit this file. Same trust guarantees as the other engines: the agent gets
the training split only, and the trusted RMSE is computed here from a state_dict loaded with
weights_only=True.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

METRIC_NAME = "val_rmse"
_IN_DIM = 10


def _split():
    d = load_diabetes()
    x = d.data.astype("float32")          # sklearn diabetes features are already mean-centered/scaled
    y = d.target.astype("float32")        # continuous disease-progression score (~25–346)
    return train_test_split(x, y, test_size=0.25, random_state=0)


def _y_stats():
    """Mean/std of the training target — used to standardize for stable training."""
    _x_tr, _x_val, y_tr, _y_val = _split()
    return float(y_tr.mean()), float(y_tr.std())


def load_train():
    x_tr, _x_val, y_tr, _y_val = _split()
    m, s = _y_stats()
    return x_tr, ((y_tr - m) / s).astype("float32")   # standardized target → stable gradients


def _val():
    _x_tr, x_val, _y_tr, y_val = _split()
    return x_val, y_val                                # original-unit target for the metric


def build_model(hparams: dict):
    """Frozen architecture (regressor: single output), parameterized by hyperparameters."""
    import torch.nn as nn

    layers, d = [], _IN_DIM
    for _ in range(int(hparams["depth"])):
        layers += [nn.Linear(d, int(hparams["hidden"])), nn.ReLU()]
        d = int(hparams["hidden"])
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


def evaluate(model) -> float:
    """Validation RMSE on the frozen val split. Lower is better."""
    import torch

    x_val, y_val = _val()
    m, s = _y_stats()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_val)).squeeze(-1).numpy() * s + m   # back to original units
    return round(float(np.sqrt(np.mean((pred - y_val) ** 2))), 4)


def emit_result(value: float, seeds: list[float] | None = None, config: dict | None = None) -> None:
    print("SCHOLARLOOP_RESULT " + json.dumps(
        {"metric_name": METRIC_NAME, "value": float(value),
         "seeds": seeds or [value], "config": config or {}}))


def score(artifact_path: str) -> None:
    import torch

    bundle = torch.load(artifact_path, weights_only=True)
    cfg = bundle["config"]
    model = build_model(cfg)
    model.load_state_dict(bundle["state_dict"])
    emit_result(evaluate(model), config=cfg)


def _main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "score":
        score(argv[1])
        return 0
    print("usage: python -m engines.torch_regression.prepare score <artifact>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
