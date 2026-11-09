"""Loop engineering demo: a PARALLEL POPULATION FUNNEL under a self-stopping GOVERNOR.

Deterministic and free — scripted by `MockLLM` on the dependency-free stub engine, so it runs in
well under a second with no API key and no GPU. It shows the two loop-engineering pieces:

  - `population_step(k, max_workers=...)` — propose k ideas, smoke-screen them ALL in parallel,
    then climb only the survivors up verify -> full (expensive tiers spent on a pre-screened few).
  - `Governor(...)` — the campaign loop stops itself on a budget / round cap / convergence
    (`dry_patience` rounds with no frontier improvement), instead of running a fixed length.

    python examples/governed_campaign.py
"""

from __future__ import annotations

from pathlib import Path
import tempfile

from scholarloop.experiment.governor import Governor
from scholarloop.core.llm import MockLLM
from scholarloop.experiment.orchestrator import Orchestrator
from scholarloop.core.profile import load_profile

ROOT = Path(__file__).resolve().parent.parent


def _proposal(config: dict, *, delta=-0.2):
    return {"reasoning_trace": "ledger headroom; trying a recipe",
            "config": [{"name": k, "value": v} for k, v in config.items()],
            "hypothesis": {"claim": "tuned recipe", "source": "arXiv:1608.03983",
                           "predicted_effect": "lower val error"},
            "predicted_delta": delta}


def main() -> int:
    profile = load_profile(ROOT / "profiles" / "image-classification.yaml")   # stub engine: instant
    tmp = Path(tempfile.mkdtemp(prefix="governed_"))

    # Three ideas per round: two beat the 4.9 baseline, one is far off and dies at the smoke screen.
    good1 = {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}    # ~3.76
    good2 = {"lr": 0.1, "depth": 22, "weight_decay": 5e-4, "warmup": 5}    # ~3.92
    bad = {"lr": 0.05, "depth": 18, "weight_decay": 1e-4, "warmup": 0}     # ~7.9 -> dies at smoke
    round_ideas = [good1, good2, bad]
    # Enough scripted proposals for several rounds; the Governor decides when to stop.
    llm = MockLLM(jsons=[_proposal(c) for _ in range(6) for c in round_ideas])

    orch = Orchestrator(llm, profile, ledger_path=tmp / "ledger.jsonl", registry_dir=tmp / "registry")
    # loop-until-dry: after the first round finds the frontier, no further round improves it -> stop.
    gov = Governor(max_rounds=5, dry_patience=1)

    print("=== GOVERNED POPULATION FUNNEL · image-classification (stub) · baseline 4.9% ===\n")
    produced = orch.run(governor=gov, population=3, max_workers=3)

    by_round = {}
    for e in produced:
        by_round.setdefault(e.id, e)
    print(f"  proposed in parallel, smoke-screened, survivors climbed verify->full:\n")
    for e in produced:
        tier = e.fidelity[0]
        mark = "kept " if e.verdict == "kept" else e.verdict
        print(f"    {e.id}  {tier:6s}  err={e.primary_score()}%  [{mark}]")
    print(f"\n  governor stopped after {gov.rounds} round(s); {len(produced)} experiments total.")
    print("  (smoke fanned out in parallel; only survivors burned verify/full.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
