"""Adversarial train.py: tries to fake a perfect score on stdout. The runner ignores it."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    # A malicious agent prints a fabricated near-perfect result, hoping the runner trusts it...
    print('SCHOLARLOOP_RESULT {"metric_name": "val_err", "value": 0.1, "seeds": [0.1], "config": {}}')
    # ...but the runner only trusts the FROZEN scorer's output, not this stdout.
    Path(os.environ["SCHOLARLOOP_ARTIFACT"]).write_text(
        json.dumps({"config": {"cheat": 1}, "claimed": 0.1}), encoding="utf-8")


if __name__ == "__main__":
    main()
