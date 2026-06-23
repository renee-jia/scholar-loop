"""EDITABLE training script for the image-classification engine (dependency-free STUB).

This is the single file the L1 agent may modify. It "trains" (the stub does nothing but pick
hyperparameters) and saves its config to $SCHOLARLOOP_ARTIFACT. It does NOT compute or report
the metric — the runner runs the frozen `prepare score` step on the artifact for the trusted
number, so an edited train.py cannot fabricate a result. The runner injects a hyperparameter
override via $SCHOLARLOOP_CONFIG.

Two edit channels: the runner sweeps the hparam space via the override
(parallel-safe, no source mutation); real architecture changes are made by editing this file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HPARAMS = {
    "lr": 0.05,
    "depth": 18,
    "weight_decay": 1e-4,
    "warmup": 0,
}


def effective_hparams() -> dict:
    override = json.loads(os.environ.get("SCHOLARLOOP_CONFIG", "{}"))
    return {**HPARAMS, **override}


def main() -> None:
    seed = int(os.environ.get("SCHOLARLOOP_SEED", "0"))
    hparams = effective_hparams()
    # A real engine trains a model and saves it here; the stub saves only its config choice.
    Path(os.environ["SCHOLARLOOP_ARTIFACT"]).write_text(
        json.dumps({"config": hparams, "seed": seed}))


if __name__ == "__main__":
    main()
