"""FROZEN scorer for the adversarial 'cheater' engine. Returns the honest value, ignoring
whatever the editable train.py claimed. Proves the metric can't be faked from train.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

METRIC_NAME = "val_err"
HONEST_VALUE = 50.0   # what the model actually deserves; the agent's claimed score is irrelevant


def emit_result(value: float, seeds: list[float] | None = None, config: dict | None = None) -> None:
    print("SCHOLARLOOP_RESULT " + json.dumps(
        {"metric_name": METRIC_NAME, "value": float(value),
         "seeds": seeds or [value], "config": config or {}}))


def score(artifact_path: str) -> None:
    data = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    # the agent's 'claimed' field is deliberately ignored — only this frozen value is emitted
    emit_result(HONEST_VALUE, config=data.get("config", {}))


def _main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "score":
        score(argv[1])
        return 0
    print("usage: python -m engines.cheater.prepare score <artifact>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
