"""The Reflector agent — compounds intuition after each run.

Given a completed experiment and its parent, the Reflector asks: what is the one transferable
lesson here? It weighs the prediction against the measurement (a large `calibration_error`
means the reasoning mispredicted — itself a strong signal), the verdict, and the config, then
emits a `{category, severity, mitigation}` lesson the Skill Library stores with time-decay.

Not every run yields a lesson — `worth_recording: false` skips it, so the library stays signal.
Runs on the agent harness (schema-validated, retried, traced). The lesson's timestamp is the
experiment's own `ts`, so decay is deterministic and tied to when the learning actually happened.
"""

from __future__ import annotations

from scholarloop.agent import Agent
from scholarloop.ledger import LedgerEntry
from scholarloop.skills import Skill

REFLECTOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "worth_recording": {"type": "boolean"},
        "category": {"type": "string"},
        "severity": {"type": "number"},        # (0, 1]; validated in postcheck
        "mitigation": {"type": "string"},
    },
    "required": ["worth_recording", "category", "severity", "mitigation"],
}

_SYSTEM = (
    "You are the reflective memory of an automated ML research loop. After an experiment, "
    "distill at most ONE transferable lesson for future experiments — a concrete, reusable "
    "mitigation, not a restatement of the result. A large gap between predicted and measured "
    "effect is a strong signal worth recording. If there is no generalizable lesson, set "
    "worth_recording=false. severity is how strongly the lesson should weigh in, in (0, 1]."
)


class Reflector(Agent):
    name = "reflector"
    system = _SYSTEM
    schema = REFLECTOR_SCHEMA

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        if not output.get("worth_recording"):
            return []   # nothing to validate when no lesson is recorded
        errs = []
        s = output.get("severity")
        if not isinstance(s, (int, float)) or not (0.0 < s <= 1.0):
            errs.append("severity must be in (0, 1]")
        if not output.get("category") or not output.get("mitigation"):
            errs.append("category and mitigation must be non-empty when worth_recording")
        return errs

    def build_prompt(self, ctx: dict) -> str:
        e: LedgerEntry = ctx["entry"]
        parent: LedgerEntry | None = ctx.get("parent")
        pred = e.prediction or {}
        lines = [
            f"Experiment {e.id} on {e.domain}: verdict={e.verdict}, {e.metric_name}={e.primary_score()}.",
            f"Config: {e.config}.",
            f"Hypothesis: {e.hypothesis.claim} (source {e.hypothesis.source}).",
            f"Predicted effect: {pred.get('predicted')}, measured: {pred.get('measured')}, "
            f"calibration_error: {pred.get('calibration_error')}.",
        ]
        if parent is not None:
            lines.append(f"Parent {parent.id} scored {parent.primary_score()}.")
        lines.append("\nWhat is the single transferable lesson (category, severity in (0,1], "
                     "mitigation)? If none, set worth_recording=false.")
        return "\n".join(lines)

    def reflect(self, entry: LedgerEntry, parent: LedgerEntry | None) -> Skill | None:
        out = self.run({"entry": entry, "parent": parent})
        if not out.get("worth_recording"):
            return None
        return Skill.make(category=out["category"], severity=float(out["severity"]),
                          mitigation=out["mitigation"], source=entry.id, ts=entry.ts)
