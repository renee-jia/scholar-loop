"""Debate panel — diverse/adversarial perspectives before spending a GPU.

Several persona critics (from `profile.debate_roles`, e.g. Innovator / Pragmatist /
Contrarian) each examine a proposal through a distinct lens and vote run / revise / reject.
The Contrarian specifically attacks the search-space reasoning.

IMPORTANT framing: the panel gates whether a proposal is *worth running* (saves budget on
low-value bets), NOT whether it is *good* — the ground-truth metric decides that after the
run. So the debate is never an LLM-judge used as an optimization target; it's a cheap filter
in front of the real evaluator. Each critic runs on the agent harness (schema-validated, traced).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scholarloop.agent import Agent, AgentTrace
from scholarloop.llm import LLMClient

CRITIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["run", "revise", "reject"]},
        "concern": {"type": "string"},
    },
    "required": ["verdict", "concern"],
}

# Lens per known persona; unknown roles get a neutral lens.
_LENSES = {
    "Innovator": ("Is this proposal ambitious and novel enough, or a timid local tweak? "
                  "Push for bolder, literature-backed moves; reject only true non-experiments."),
    "Pragmatist": ("Given the ledger evidence and budget, is this likely to actually improve "
                   "the metric and run cleanly? Flag fragile, redundant, or low-probability bets."),
    "Contrarian": ("Attack the proposal's assumptions and the search-space reasoning. Is a "
                   "ruled-out region actually worth revisiting? What is being overlooked?"),
}
_DEFAULT_LENS = "Judge whether this experiment is worth running now versus revising or rejecting."


class Critic(Agent):
    schema = CRITIC_SCHEMA

    def __init__(self, llm: LLMClient, role: str, *, trace: AgentTrace | None = None):
        super().__init__(llm, trace=trace)
        self.role = role
        self.name = f"critic:{role}"
        lens = _LENSES.get(role, _DEFAULT_LENS)
        self.system = (f"You are the {role} on a research review panel. {lens} You decide "
                       "whether the proposed experiment is worth a GPU run — not whether it is "
                       "correct (the measured metric decides that). Vote run, revise, or reject "
                       "with one concise concern.")

    def build_prompt(self, ctx: dict) -> str:
        p = ctx["proposal"]
        return (
            f"Proposed experiment:\n"
            f"  config: {p.config}\n"
            f"  hypothesis: {p.hypothesis.claim} (source {p.hypothesis.source})\n"
            f"  predicted_delta: {p.predicted_delta}\n"
            f"  reasoning: {p.reasoning_trace}\n\n"
            f"{p.constraints.to_prompt_block()}\n\n"
            f"As the {self.role}, vote run/revise/reject and give one concern."
        )


@dataclass
class PanelVerdict:
    decision: str                       # "run" | "reject"
    votes: list[dict] = field(default_factory=list)      # [{role, verdict, concern}]
    concerns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"decision": self.decision, "votes": self.votes, "concerns": self.concerns}


class DebatePanel:
    """A panel of persona critics. Rejects a proposal only on a strict majority of rejects."""

    def __init__(self, llm: LLMClient, roles, *, trace: AgentTrace | None = None):
        self.critics = [Critic(llm, r, trace=trace) for r in roles]

    def review(self, proposal) -> PanelVerdict:
        votes, concerns, rejects = [], [], 0
        for critic in self.critics:
            out = critic.run({"proposal": proposal})
            votes.append({"role": critic.role, "verdict": out["verdict"], "concern": out["concern"]})
            if out["concern"]:
                concerns.append(f"{critic.role}: {out['concern']}")
            if out["verdict"] == "reject":
                rejects += 1
        # reject on at least half the panel (fail-closed on an even-split tie); an empty panel
        # has no standing to reject. One skeptic on a 3+ panel still can't veto a backed bet.
        decision = "reject" if self.critics and rejects * 2 >= len(self.critics) else "run"
        return PanelVerdict(decision=decision, votes=votes, concerns=concerns)
