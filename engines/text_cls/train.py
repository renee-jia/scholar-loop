"""EDITABLE training script for the agnews-bow engine.

Builds the (frozen) model via `prepare.build_model`, trains it on the FROZEN training folds
(`prepare.load_train()` never returns val or test), and saves a `state_dict` + config to
$SCHOLARLOOP_ARTIFACT. It does NOT compute or report the metric: the runner runs the frozen
`prepare score` step from pristine ROOT on the artifact. The runner injects a hyperparameter
override via $SCHOLARLOOP_CONFIG.

Everything about the training procedure is yours to change here: the optimizer and schedule,
the loss, regularization, and how token-id batches are sampled or perturbed before the model
sees them. Inputs are int64 token ids of shape [N, 64] where 0 is padding. The architecture is
fixed by `prepare.build_model(hparams)`; `model.pool(ids)` returns the pooled text embedding
and `model.head(z)` maps it to logits, so a method may act on either side of the pooling.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from engines.text_cls import prepare

HPARAMS = {
    "lr": 0.003,
    "embed_dim": 64,
    "hidden": 64,
    "depth": 1,
    "weight_decay": 0.0001,
    "epochs": 8,
}
BATCH = 128


def effective_hparams() -> dict:
    return {**HPARAMS, **json.loads(os.environ.get("SCHOLARLOOP_CONFIG", "{}"))}


def train_model(seed: int, h: dict) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr, y_tr = prepare.load_train()           # training folds only, no val leakage
    model = prepare.build_model(h)              # frozen architecture, shaped by hparams
    torch.manual_seed(seed)
    for p in model.parameters():                # seed-dependent initialisation
        if p.dim() > 1:
            nn.init.normal_(p, std=0.1)
    with torch.no_grad():
        model.embed.weight[0].zero_()
    opt = torch.optim.Adam(model.parameters(), lr=float(h["lr"]), weight_decay=float(h["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    model.train()
    for _ in range(int(h["epochs"])):
        order = torch.randperm(len(xt))
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
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
