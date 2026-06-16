"""Universal predict-then-verify (DESIGN §4.5 mechanism 1, generalized).

Self-calibration is not special to the Reasoner: ANY generative step that commits to a checkable
claim can be scored once ground truth arrives, and the running accuracy fed back into the loop.
This module is the shared substrate for that — a tiny, pure scoring layer plus an accumulating
`CalibrationLog` that summarizes how well each agent's claims have held up.

Two claim kinds today:
  - **delta**  — a signed predicted change in the metric (the Reasoner's `predicted_delta`). Scored
    against the measured delta: the error is the absolute gap; a "hit" is getting the *direction* right.
  - **binary** — a go/no-go bet (e.g. the Debate panel voting "run" = "this idea will beat the gate").
    Scored against the realized outcome (did it end up kept?).

The log renders into a prompt block so the next round's Reasoner sees, e.g., "debate panel: 40%
of approved ideas actually beat the gate" — the verifier loop, closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Verdict:
    agent: str
    kind: str            # "delta" | "binary"
    predicted: float
    actual: float
    error: float         # |pred - actual| for delta; 0.0/1.0 for binary
    hit: bool            # direction-correct (delta) or correct (binary)


def score_delta(agent: str, predicted: float | None, measured: float | None) -> Verdict | None:
    """Score a signed-magnitude prediction against the measured change. None if not checkable yet."""
    if predicted is None or measured is None:
        return None
    hit = (predicted <= 0) == (measured <= 0)        # did the prediction get the direction right?
    return Verdict(agent, "delta", float(predicted), float(measured),
                   round(abs(predicted - measured), 4), hit)


def score_binary(agent: str, predicted_true: bool, actual_true: bool) -> Verdict:
    """Score a go/no-go bet against the realized outcome."""
    hit = bool(predicted_true) == bool(actual_true)
    return Verdict(agent, "binary", float(bool(predicted_true)), float(bool(actual_true)),
                   0.0 if hit else 1.0, hit)


@dataclass
class CalibrationLog:
    verdicts: list = field(default_factory=list)

    def record(self, verdict: Verdict | None) -> None:
        if verdict is not None:
            self.verdicts.append(verdict)

    def by_agent(self) -> dict:
        """Per-agent {n, hit_rate, mean_error}, computed over all recorded verdicts."""
        out: dict = {}
        for v in self.verdicts:
            g = out.setdefault(v.agent, {"n": 0, "hits": 0, "err_sum": 0.0, "kind": v.kind})
            g["n"] += 1
            g["hits"] += int(v.hit)
            g["err_sum"] += v.error
        return {a: {"n": g["n"], "hit_rate": round(g["hits"] / g["n"], 2),
                    "mean_error": round(g["err_sum"] / g["n"], 3), "kind": g["kind"]}
                for a, g in out.items()}

    def render(self) -> str:
        """A prompt block summarizing each agent's track record, or '' when nothing is scored yet."""
        stats = self.by_agent()
        if not stats:
            return ""
        lines = []
        for agent, s in sorted(stats.items()):
            if s["kind"] == "delta":
                lines.append(f"- {agent}: predicted the right direction {int(s['hit_rate'] * 100)}% of "
                             f"{s['n']} times; mean magnitude error {s['mean_error']}.")
            else:
                lines.append(f"- {agent}: {int(s['hit_rate'] * 100)}% of {s['n']} go/no-go calls were "
                             f"correct.")
        return "\n".join(lines)
