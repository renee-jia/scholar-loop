"""60-second ScholarLoop quickstart — the autonomous loop, no API key, no GPU.

Runs one idea through the multi-fidelity funnel on the dependency-free stub engine, driven by a
scripted MockLLM. Shows the whole shape of the system (propose → run smoke/verify/full → record)
in well under a second. For the real thing (real torch + the full agent chain) see campaign_demo.py.

Run from the repo root:  python examples/quickstart.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from scholarloop.core.ledger import Ledger
from scholarloop.core.llm import MockLLM
from scholarloop.experiment.orchestrator import Orchestrator
from scholarloop.core.profile import load_profile

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    profile = load_profile(ROOT / "profiles" / "image-classification.yaml")
    tmp = Path(tempfile.mkdtemp(prefix="quickstart_"))

    # A scripted "reasoner": propose one near-optimal config (in a real run an LLM does this).
    idea = {"rationale": "raise lr, deepen, add warmup per the cosine-SGD recipe",
            "config": [{"name": "lr", "value": 0.1}, {"name": "depth", "value": 20},
                       {"name": "weight_decay", "value": 5e-4}, {"name": "warmup", "value": 5}],
            "hypothesis": {"claim": "cosine + warmup beats the default",
                           "source": "arXiv:1608.03983", "predicted_effect": "lower val error"},
            "predicted_delta": -1.2}

    orch = Orchestrator(MockLLM(jsons=[idea]), profile,
                        ledger_path=tmp / "ledger.jsonl", registry_dir=tmp / "registry")

    print(f"baseline to beat: {profile.best_baseline()}% {profile.metric.name}\n")
    produced = orch.funnel_step()    # smoke → verify → full, each gated by a promotion check

    for e in produced:
        print(f"  {e.fidelity[0]:<6} {e.metric_name}={e.primary_score()}%  [{e.verdict}]")
    print(f"\nthe idea climbed {len(produced)} fidelity tier(s); "
          f"{sum(e.verdict == 'kept' for e in produced)} kept.")
    print(f"ledger written to {tmp/'ledger.jsonl'} "
          f"({len(list(Ledger(tmp/'ledger.jsonl').read_all()))} records)")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
