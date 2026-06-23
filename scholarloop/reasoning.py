"""Reasoning Layer — ReAct-light.

This module holds the *deterministic* substrate the LLM Reasoner stands on:

  - `analyze_search_space` — mine the ledger for which hyperparameter regions are
    exhausted (ruled out) and which knobs are high-leverage (worth focusing edits on).
    This is mechanism 2 ("Search-Space Reasoning"): it bounds the next experiment so the
    agent does not re-explore losing regions or fiddle low-impact knobs.
  - `calibration_error` / `score_prediction` — mechanism 1 ("Predict-then-Verify"): turn a
    predicted effect and the measured effect into a number, so reasoning that doesn't help
    is demoted by measurement rather than trusted on faith.
  - `is_duplicate_config` — config-level novelty so the same config isn't run twice.

The LLM Reasoner consumes a `SearchSpaceConstraints` (rendered via `to_prompt_block`) plus
literature/skills to write the trace + act; the Reflector calls `score_prediction` after.
Everything here is pure and testable with no LLM and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev

from scholarloop.ledger import LedgerEntry
from scholarloop.profile import Profile


@dataclass
class SearchSpaceConstraints:
    ruled_out: list[str] = field(default_factory=list)        # human-readable, e.g. "lr>0.3 (3 losses)"
    ruled_out_bounds: list[dict] = field(default_factory=list)  # machine-checkable, e.g. {"knob":"lr","op":">","value":0.3}
    focus: list[str] = field(default_factory=list)             # high-leverage knob names
    priors: list[str] = field(default_factory=list)            # literature-derived hard bounds
    tried: list[dict] = field(default_factory=list)            # configs already evaluated

    def to_dict(self) -> dict:
        return {"ruled_out": self.ruled_out, "ruled_out_bounds": self.ruled_out_bounds,
                "focus": self.focus, "priors": self.priors, "tried": self.tried}

    def violations(self, config: dict) -> list[str]:
        """Return the reasons `config` violates the constraints (empty == admissible).

        This is what makes search-space limitation actually BIND: a proposed
        config inside a ruled-out region — or already tried — is rejected, not merely discouraged.
        """
        out = []
        for b in self.ruled_out_bounds:
            if b["knob"] in config:
                v = config[b["knob"]]
                if (b["op"] == ">" and v > b["value"]) or (b["op"] == "<" and v < b["value"]):
                    out.append(f"{b['knob']}{b['op']}{b['value']} is ruled out")
        if any(config == t for t in self.tried):
            out.append("config already tried")
        return out

    def to_prompt_block(self) -> str:
        """Render for injection into the Reasoner's prompt."""
        lines = ["Search-space constraints derived from the experiment ledger:"]
        lines.append(f"  RULED OUT (do not propose): {self.ruled_out or 'none yet'}")
        lines.append(f"  FOCUS (high-leverage knobs): {self.focus or 'unknown — explore'}")
        if self.priors:
            lines.append(f"  LITERATURE PRIORS (hard bounds): {self.priors}")
        lines.append(f"  ALREADY TRIED ({len(self.tried)} configs): do not repeat any of them.")
        return "\n".join(lines)


def _scored_entries(entries: list[LedgerEntry]) -> list[tuple[dict, float]]:
    """(config, score) pairs for entries that actually produced a measurement + config."""
    out = []
    for e in entries:
        score = e.primary_score()
        if e.verdict in ("kept", "discarded") and e.config and score is not None:
            out.append((dict(e.config), score))
    return out


def _numeric_keys(pairs: list[tuple[dict, float]]) -> list[str]:
    keys = set()
    for cfg, _ in pairs:
        for k, v in cfg.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    return sorted(keys)


def analyze_search_space(
    entries: list[LedgerEntry],
    profile: Profile,
    *,
    lit_priors: list[str] | None = None,
    max_focus: int = 3,
) -> SearchSpaceConstraints:
    """Derive search-space constraints from the ledger (deterministic, no LLM).

    - ruled_out: for each numeric knob, if every losing run sits strictly to one side of
      every winning run (>=2 losers), that outer region is exhausted.
    - focus: knobs whose value changes move the metric the most (per-value mean spread).
    """
    pairs = _scored_entries(entries)
    constraints = SearchSpaceConstraints(priors=list(lit_priors or []),
                                         tried=[cfg for cfg, _ in pairs])
    if len(pairs) < 2:
        return constraints  # not enough signal to bound anything yet

    baseline = profile.best_baseline()
    if baseline is None:
        # no must-beat baseline: split winners/losers at the median score
        scores = sorted(s for _, s in pairs)
        baseline = scores[len(scores) // 2]

    def is_win(score: float) -> bool:
        return profile.metric.is_better(score, baseline)

    winners = [(c, s) for c, s in pairs if is_win(s)]
    losers = [(c, s) for c, s in pairs if not is_win(s)]

    leverage: dict[str, float] = {}
    for k in _numeric_keys(pairs):
        vals = [(c[k], s) for c, s in pairs if k in c]
        # --- sensitivity / leverage: spread of per-value mean scores ---
        by_val: dict[float, list[float]] = {}
        for v, s in vals:
            by_val.setdefault(v, []).append(s)
        if len(by_val) >= 2:
            means = [mean(v) for v in by_val.values()]
            leverage[k] = max(means) - min(means)

        # --- region pruning: losers all strictly outside the winning range ---
        if winners and len(losers) >= 2:
            win_vals = [c[k] for c, _ in winners if k in c]
            lose_vals = [c[k] for c, _ in losers if k in c]
            if win_vals and len(lose_vals) >= 2:
                hi = max(win_vals)
                lo = min(win_vals)
                above = [v for v in lose_vals if v > hi]
                below = [v for v in lose_vals if v < lo]
                if len(above) >= 2 and len(above) == len(lose_vals):
                    constraints.ruled_out.append(f"{k}>{hi} ({len(above)} losses)")
                    constraints.ruled_out_bounds.append({"knob": k, "op": ">", "value": hi})
                elif len(below) >= 2 and len(below) == len(lose_vals):
                    constraints.ruled_out.append(f"{k}<{lo} ({len(below)} losses)")
                    constraints.ruled_out_bounds.append({"knob": k, "op": "<", "value": lo})

    # focus = knobs with leverage at/above the mean leverage (most impactful first)
    if leverage:
        avg = mean(leverage.values())
        ranked = sorted((k for k, lv in leverage.items() if lv >= avg and lv > 0),
                        key=lambda k: leverage[k], reverse=True)
        constraints.focus = ranked[:max_focus]
    return constraints


def is_duplicate_config(config: dict, entries: list[LedgerEntry]) -> bool:
    """True if this exact config was already evaluated (config-level novelty check)."""
    return any(e.config == config for e in entries if e.config)


def confidence_bound(values, direction: str, z: float = 1.0) -> float:
    """Pessimistic z·SEM bound on the mean across seeds — the side that must still clear a gate
    for the improvement to count as real rather than noise (statistical-significance gating).

    For a `minimize` metric this is mean + z·SEM (the high/worse side); for `maximize`, mean − z·SEM.
    With a single value (or z=0) it's just the mean.
    """
    vals = list(values)
    m = mean(vals)
    if len(vals) < 2 or z == 0:
        return round(m, 4)
    sem = stdev(vals) / (len(vals) ** 0.5)
    return round(m + z * sem if direction == "minimize" else m - z * sem, 4)


def calibration_error(predicted_delta: float, measured_delta: float) -> float:
    """Absolute gap between a predicted metric change and the measured one. Lower = better."""
    return round(abs(predicted_delta - measured_delta), 4)


def prediction_from_scores(predicted_delta: float | None, score: float | None,
                           parent_score: float | None) -> dict:
    """Build the `prediction` blob from raw scores (mechanism 1, Predict-then-Verify).

    measured_delta = this run's score minus its parent's (negative = improvement for a
    minimize metric). With no parent/score, measured_delta is unknown and calibration skipped.
    """
    if predicted_delta is None or score is None or parent_score is None:
        return {"predicted": predicted_delta, "measured": None, "calibration_error": None}
    measured_delta = round(score - parent_score, 4)
    return {"predicted": predicted_delta, "measured": measured_delta,
            "calibration_error": calibration_error(predicted_delta, measured_delta)}


def score_prediction(entry: LedgerEntry, parent: LedgerEntry | None,
                     predicted_delta: float) -> dict:
    """Build the `prediction` blob for a ledger entry, resolving scores from the entries."""
    parent_score = parent.primary_score() if parent is not None else None
    return prediction_from_scores(predicted_delta, entry.primary_score(), parent_score)
