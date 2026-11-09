"""The Research Director (L4) — sets the campaign's direction.

Given the ledger so far (what's been tried, the current frontier, recent verdicts) and any
literature hotspots, the Director picks the next research direction and a concrete `topic` for
the Lit Scout to search (budget is the Governor's job). It runs once at the start of a campaign
and again on a PIVOT, so the loop has a sense of *where to look*, not just *how to tweak*.

Higher-level than the Advisor (which only nudges the next config): the Director changes the
question being asked. Runs on the agent harness.
"""

from __future__ import annotations

from scholarloop.core.agent import Agent
from scholarloop.core.ledger import LedgerEntry, idea_chains
from scholarloop.core.profile import Profile

DIRECTOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "direction": {"type": "string"},        # the strategic direction, used as Reasoner guidance
        "topic": {"type": "string"},             # a concrete search query for the Lit Scout
        "rationale": {"type": "string"},
    },
    "required": ["direction", "topic", "rationale"],
}

_SYSTEM = (
    "You are the research director of an automated ML lab. From the experiment ledger and the "
    "literature, choose the next research direction worth pursuing and a concrete search topic "
    "for the literature scout. Favor directions with evidence of headroom; abandon exhausted "
    "ones. Be specific — the topic is fed verbatim into an arXiv search."
)


class Director(Agent):
    name = "director"
    system = _SYSTEM
    schema = DIRECTOR_SCHEMA

    def __init__(self, llm, profile: Profile, *, trace=None):
        super().__init__(llm, trace=trace)
        self.profile = profile

    def _summary(self, entries: list[LedgerEntry]) -> str:
        if not entries:
            return "No experiments yet."
        kept = [e for e in entries if e.verdict == "kept" and e.primary_score() is not None]
        best = min(kept, key=lambda e: e.primary_score(), default=None) \
            if self.profile.metric.direction == "minimize" \
            else max(kept, key=lambda e: e.primary_score(), default=None)
        return (f"{len(idea_chains(entries))} distinct ideas over {len(entries)} runs; {len(kept)} kept. "
                f"Best {self.profile.metric.name}: "
                f"{best.primary_score() if best else 'none'} (config {best.config if best else '-'}). "
                f"Recent verdicts: {[e.verdict for e in entries[-5:]]}.")

    def build_prompt(self, ctx: dict) -> str:
        p = self.profile
        parts = [
            f"Domain: {p.name} ({p.paradigm}). Metric: {p.metric.name}, {p.metric.direction}. "
            f"Must-beat baseline: {p.best_baseline()}.",
            f"Ledger summary: {self._summary(ctx['entries'])}",
        ]
        if ctx.get("lit_hotspots"):
            parts += [f"Literature hotspots: {ctx['lit_hotspots']}"]
        parts += ["\nChoose the next direction, a concrete arXiv search topic, and a rationale."]
        return "\n".join(parts)

    def direct(self, entries: list[LedgerEntry], *, lit_hotspots: str = "") -> dict:
        return self.run({"entries": entries, "lit_hotspots": lit_hotspots})
