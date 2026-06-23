"""The Advisor agent — post-run loop control.

After an experiment, the Advisor reads the result against its parent and recent history and
decides the loop's next move:
  - PROCEED — the frontier advanced; keep iterating from here.
  - REFINE  — the direction is promising but this config missed; tweak and retry (bounded:
              the orchestrator forces a PIVOT after N consecutive refines).
  - PIVOT   — this direction is exhausted; change direction (orchestrator re-grounds via the
              Lit Scout / Director).

It only steers control flow; the metric still decides quality. Runs on the agent harness.
"""

from __future__ import annotations

from scholarloop.agent import Agent
from scholarloop.ledger import LedgerEntry

ADVISOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["proceed", "refine", "pivot"]},
        "rationale": {"type": "string"},
    },
    "required": ["decision", "rationale"],
}

_SYSTEM = (
    "You steer an automated ML research loop after each experiment. Decide proceed / refine / "
    "pivot: proceed if the frontier advanced, refine if the direction is promising but the "
    "config missed, pivot if the direction looks exhausted. Give a one-line rationale that the "
    "next reasoning step can act on. You control flow only — the measured metric judges quality."
)


class Advisor(Agent):
    name = "advisor"
    system = _SYSTEM
    schema = ADVISOR_SCHEMA

    def build_prompt(self, ctx: dict) -> str:
        e: LedgerEntry = ctx["entry"]
        parent: LedgerEntry | None = ctx.get("parent")
        lines = [
            f"Latest experiment {e.id}: {e.metric_name}={e.primary_score()}, verdict={e.verdict}.",
            f"Config: {e.config}.  Prediction: {e.prediction}.",
        ]
        if parent is not None:
            improved = (e.primary_score() is not None and parent.primary_score() is not None
                        and e.primary_score() < parent.primary_score())
            lines.append(f"Frontier {parent.id} scored {parent.primary_score()} "
                         f"({'improved' if improved else 'did not improve'}).")
        lines.append(f"Recent verdicts: {ctx.get('recent_verdicts', [])}.")
        lines.append("\nDecide proceed / refine / pivot with a one-line rationale.")
        return "\n".join(lines)

    def advise(self, entry: LedgerEntry, parent: LedgerEntry | None,
               recent_verdicts: list[str] | None = None) -> dict:
        return self.run({"entry": entry, "parent": parent,
                         "recent_verdicts": recent_verdicts or []})
