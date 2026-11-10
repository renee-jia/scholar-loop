"""The Reasoner agent — ReAct-light "reason → act".

Given the experiment ledger, the Reasoner:
  1. mines search-space constraints from history (deterministic, `reasoning.py`),
  2. assembles a prompt — constraints + literature + skills + the editable knobs + the
     must-beat baseline — and asks the LLM (via the agent harness) for a structured next
     experiment,
  3. ENFORCES the constraints on the returned config (dedup + ruled-out regions), so the
     search-space limitation actually binds rather than being a polite suggestion.

It runs on the `Agent` harness (schema-validated, retried, traced) and returns a `Proposal`;
the orchestrator runs admissible proposals and skips violating ones. The LLM only does what
it's good at (pick a promising, literature-grounded move); the checkable parts (region
pruning, dedup, calibration) stay deterministic and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scholarloop.core.agent import Agent, AgentTrace
from scholarloop.core.ledger import Hypothesis, LedgerEntry
from scholarloop.core.llm import LLMClient
from scholarloop.core.profile import Profile
from scholarloop.experiment.reasoning import SearchSpaceConstraints, analyze_search_space

# JSON-schema for the structured proposal. `config` is a list of {name,value} entries rather
# than an open dict because the structured-output API requires additionalProperties:false.
REASONER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rationale": {"type": "string"},
        "config": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                "required": ["name", "value"],
            },
        },
        "hypothesis": {
            "type": "object", "additionalProperties": False,
            "properties": {"claim": {"type": "string"}, "source": {"type": "string"},
                           "predicted_effect": {"type": "string"}},
            "required": ["claim", "source", "predicted_effect"],
        },
        "predicted_delta": {"type": "number"},
        "edits": {                       # optional source-diff channel (full-file replacements)
            "type": "array",
            "items": {"type": "object", "additionalProperties": False,
                      "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]},
        },
    },
    "required": ["rationale", "config", "hypothesis", "predicted_delta"],
}

def _ranges_line(knobs: dict) -> str:
    """The search range of every knob, the same one the random-search control arm samples from,
    so the agent and the control search one space."""
    from scholarloop.experiment.random_search import knob_range
    def fmt(v):
        return f"{v:.3g}" if isinstance(v, float) else str(v)
    spans = []
    for name, default in knobs.items():
        kind, lo, hi = knob_range(default)
        spans.append(f"{name} {fmt(lo)}..{fmt(hi)}" + (" (log scale)" if kind in ("log", "logint") else ""))
    return "Search ranges: " + "; ".join(spans) + "."


_SYSTEM = (
    "You plan experiments for an automated ML research loop. Propose the single next "
    "experiment as one config edit, grounded in the ledger evidence and the cited literature. "
    "Never propose a config in a ruled-out region or one already tried. Every hypothesis MUST "
    "cite a real source. Predict the metric change before the experiment runs."
)


@dataclass
class Proposal:
    config: dict                       # the hparam override to run
    hypothesis: Hypothesis
    predicted_delta: float             # expected change in the primary metric (signed)
    reasoning_trace: str
    constraints: SearchSpaceConstraints
    edits: list[dict] = field(default_factory=list)       # source-diff channel: [{path, content}]
    violations: list[str] = field(default_factory=list)   # empty == admissible to run

    @property
    def admissible(self) -> bool:
        return not self.violations

    def reasoning_blob(self) -> dict:
        """The `reasoning` dict stored on the ledger entry."""
        blob = {"trace": self.reasoning_trace,
                "search_space_constraints": self.constraints.to_dict()}
        if self.edits:
            blob["edits"] = [e["path"] for e in self.edits]
        return blob


class Reasoner(Agent):
    name = "reasoner"
    system = _SYSTEM
    schema = REASONER_SCHEMA

    def __init__(self, llm: LLMClient, profile: Profile, *, trace: AgentTrace | None = None,
                 explore_ideas: int = 3):
        super().__init__(llm, trace=trace)
        self.profile = profile
        # the first `explore_ideas` ideas of a campaign are asked to spread out over the knob
        # ranges, not to refine the default: measured against random search on digits, an agent
        # that starts by nudging one knob near the default never finds the good region (§9.1.2)
        self.explore_ideas = explore_ideas

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        # structural: a proposal must change something — a config knob or a source edit
        return [] if (output.get("config") or output.get("edits")) \
            else ["proposal must set a config knob or provide an edit"]

    def build_prompt(self, ctx: dict) -> str:
        constraints: SearchSpaceConstraints = ctx["constraints"]
        lit_context, skills = ctx.get("lit_context", ""), ctx.get("skills", "")
        guidance = ctx.get("guidance", "")
        p = self.profile
        head = (f"Domain: {p.name} ({p.paradigm}). Metric: {p.metric.name}, "
                + ("minimize (lower is better)." if p.metric.direction == "minimize" else "maximize."))
        parts = [
            head,
            f"Must-beat baseline: {p.best_baseline()} (your config should aim to beat it).",
            (f"Tunable hyperparameters — set any of these in `config` by their exact names "
             f"(current defaults shown): {p.knobs}." if p.knobs
             else f"Editable areas: {list(p.allowed_edits)}."),
            *([_ranges_line(p.knobs)] if p.knobs else []),
            "",
            constraints.to_prompt_block(),
        ]
        if guidance:
            parts += ["", "Director/Advisor guidance for this step:", guidance]
        if lit_context:
            parts += ["", "Relevant literature:", lit_context]
        if skills:
            parts += ["", "Lessons from past runs (skill library):", skills]
        calibration = ctx.get("calibration", "")
        if calibration:
            parts += ["", "Agent track record so far (predict-then-verify — trust accordingly):", calibration]
        n_ideas = ctx.get("n_ideas", 0)
        if p.knobs and n_ideas < self.explore_ideas:
            parts += ["", (f"Phase: EXPLORE (idea {n_ideas + 1} of the first {self.explore_ideas}). The ledger "
                           "is too thin to refine anything yet. Propose a bold config that changes "
                           "several knobs at once, anywhere in their ranges, in a region no tried config "
                           "covers. Do not stay near the defaults: they are a starting point, not a prior.")]
        elif p.knobs:
            parts += ["", ("Phase: REFINE. Build on the best region the ledger has found, but move far "
                           "enough to be measurable: a change smaller than the seed-to-seed spread "
                           "in the ledger cannot be ranked by the funnel, so it wastes a run. Prefer "
                           "moving the knobs the evidence says matter, by a factor of 1.5 or more, over "
                           "nudging one knob a few percent.")]
        parts += [
            "",
            "Propose the next experiment as a `config`: overrides to the tunable hyperparameters "
            "above, using their exact names (e.g. one or more of them). Use the `edits` channel "
            "ONLY to rewrite the train script itself for a genuine architecture change — otherwise "
            "leave it out. Also give a hypothesis {claim, source, predicted_effect}, a one-paragraph "
            "rationale (why this experiment, citing the ledger evidence), and predicted_delta (signed expected change in the metric; negative = "
            "improvement for a minimize metric). Stay out of the ruled-out regions and don't repeat a tried config.",
        ]
        return "\n".join(parts)

    def propose(self, entries: list[LedgerEntry], *, lit_context: str = "",
                skills: str = "", guidance: str = "", lit_priors: list[str] | None = None,
                calibration: str = "") -> Proposal:
        constraints = analyze_search_space(entries, self.profile, lit_priors=lit_priors)
        n_ideas = sum(1 for e in entries if e.fidelity == ["smoke"])   # ideas tried so far
        raw = self.run({"constraints": constraints, "lit_context": lit_context, "n_ideas": n_ideas,
                        "skills": skills, "guidance": guidance, "calibration": calibration})

        config = {item["name"]: item["value"] for item in raw.get("config", [])}
        edits = raw.get("edits", []) or []
        h = raw.get("hypothesis", {})
        hypothesis = Hypothesis(claim=h.get("claim", ""), source=h.get("source", ""),
                                predicted_effect=h.get("predicted_effect", ""))

        violations = constraints.violations(config, defaults=self.profile.knobs)   # effective config, as the engine runs it
        if edits:   # an edit makes the experiment novel even at a repeated config — drop config-dedup
            violations = [v for v in violations if v != "config already tried"]
        edit_paths = [e["path"] for e in edits]
        forbidden = set(self.profile.touches_forbidden(edit_paths))
        violations += [f"forbidden edit (frozen surface): {p}" for p in forbidden]   # the §7 guard, now in the run path
        violations += [f"edit outside the train entrypoint: {p}" for p in edit_paths
                       if p not in forbidden and p != self.profile.train_entrypoint]

        return Proposal(
            config=config, hypothesis=hypothesis,
            predicted_delta=float(raw.get("predicted_delta", 0.0)),
            reasoning_trace=raw.get("rationale", ""),
            constraints=constraints, edits=edits, violations=violations,
        )
