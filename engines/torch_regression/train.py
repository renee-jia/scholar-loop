"""EDITABLE training script for the diabetes-mlp regression engine.

Builds the (frozen) regressor via `prepare.build_model`, trains it with MSE on the FROZEN training
split, and saves a state_dict to $SCHOLARLOOP_ARTIFACT. The trusted RMSE is computed separately by
the frozen `prepare score` step. The runner injects a hyperparameter override via $SCHOLARLOOP_CONFIG.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from engines.torch_regression import prepare

HPARAMS = {
    "lr": 0.1,
    "hidden": 32,
    "depth": 1,
    "weight_decay": 1e-4,
    "epochs": 200,
}


def effective_hparams() -> dict:
    return {**HPARAMS, **json.loads(os.environ.get("SCHOLARLOOP_CONFIG", "{}"))}


def train_model(seed: int, h: dict) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr, y_tr = prepare.load_train()
    model = prepare.build_model(h)
    opt = torch.optim.SGD(model.parameters(), lr=float(h["lr"]), momentum=0.9,
                          weight_decay=float(h["weight_decay"]))
    loss_fn = nn.MSELoss()
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    model.train()
    for _ in range(int(h["epochs"])):
        opt.zero_grad()
        loss_fn(model(xt).squeeze(-1), yt).backward()
        opt.step()
    return model


def main() -> None:
    seed = int(os.environ.get("SCHOLARLOOP_SEED", "0"))
    h = effective_hparams()
    model = train_model(seed, h)
    torch.save({"config": h, "state_dict": dict(model.state_dict())},
               os.environ["SCHOLARLOOP_ARTIFACT"])


if __name__ == "__main__":
    main()
