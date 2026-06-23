"""FROZEN module for the image-classification engine (the profile's `fixed_module`).

The agent must NEVER edit this file. It owns data preparation and evaluation — the
parts that, if editable, would let an agent game the metric (the #1 reward-hacking
risk). Keeping eval here, behind a stable interface, is the structural guard.

PHASE-0 NOTE: this is a dependency-free STUB. `evaluate()` does not train a real
network; it deterministically maps a hyperparameter dict to a plausible validation
error so the full orchestration pipeline (budget -> run -> metric -> registry ->
ledger) can be exercised on a laptop with no GPU. Replace with a real torch
train/eval in Phase 3 — the interface (`evaluate(hparams, seed) -> float`) stays.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

METRIC_NAME = "val_top1_err"

# A frozen "ground-truth" optimum the stub rewards configs for approaching. The agent
# cannot see this; it can only observe the returned error. Acts as a hidden objective.
_IDEAL = {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}


def _seed_noise(seed: int, hparams: dict) -> float:
    """Deterministic per-seed jitter so multi-seed runs show realistic variance."""
    h = hashlib.sha256(f"{seed}:{json.dumps(hparams, sort_keys=True)}".encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF - 0.5) * 0.4  # +/- 0.2 abs error


def evaluate(hparams: dict, seed: int = 0) -> float:
    """Return validation top-1 error (%) for a hyperparameter config. Lower is better.

    STUB: distance from a hidden ideal config, squashed into a ~[3.5, 9.0] error band,
    plus deterministic per-seed noise. Real engines replace this body with training.
    """
    dist = 0.0
    for k, ideal in _IDEAL.items():
        v = float(hparams.get(k, ideal))
        scale = abs(ideal) or 1.0
        dist += ((v - ideal) / scale) ** 2
    base = 3.8 + 5.0 * (1.0 - math.exp(-dist))   # 3.8 (ideal) .. ~8.8 (far)
    return round(base + _seed_noise(seed, hparams), 4)


def emit_result(value: float, seeds: list[float] | None = None, config: dict | None = None) -> None:
    """Print the result in the protocol the runner parses (a single tagged JSON line).

    `config` is the hyperparameter snapshot this run used; the runner stores it in the
    ledger so the Reasoning Layer can prune the search space across runs.
    """
    payload = {"metric_name": METRIC_NAME, "value": float(value),
               "seeds": seeds or [value], "config": config or {}}
    print("SCHOLARLOOP_RESULT " + json.dumps(payload))


def score(artifact_path: str) -> None:
    """Trusted metric path: recompute the (frozen) evaluate from the saved config+seed.

    For this stub the 'model' is just the config the agent chose; the score is recomputed
    here so a fabricated result line from train.py is ignored — only this output is trusted.
    """
    data = json.loads(Path(artifact_path).read_text())
    cfg = data.get("config", {})
    emit_result(evaluate(cfg, int(data.get("seed", 0))), config=cfg)


def _main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "score":
        score(argv[1])
        return 0
    print("usage: python -m engines.vision.prepare score <artifact>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
