"""Loop governor — the autonomous loop's stop conditions and budget control.

A long-running campaign should not run a fixed number of steps blindly: it should stop when it has
spent its money, exhausted its ideas, or run long enough. The `Governor` is a small, pure
state machine the Orchestrator consults once per round. It never calls an LLM, so it is fully
unit-testable on its own.

Three independent stop conditions (any one fires):
  - **budget** — a USD ceiling on cumulative LLM spend (with 50/80/100% alerts on the way up),
  - **rounds** — a hard cap on iterations,
  - **convergence** — `dry_patience` consecutive rounds with no frontier improvement ("loop-until-dry":
    keep going while ideas still help, stop once the search has clearly plateaued).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per 1M tokens (input, output). Mirrors the public price list; unknown models cost None,
# which disables the budget gate (e.g. the free MockLLM used in tests).
MODEL_PRICES = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def cost_of(usage: dict, model: str | None) -> float | None:
    """Estimated USD for a usage blob ({input_tokens, output_tokens}) on `model`, or None if the
    model has no known price (so the caller treats spend as unmetered)."""
    price = MODEL_PRICES.get(model or "")
    if price is None:
        return None
    return usage.get("input_tokens", 0) / 1e6 * price[0] + usage.get("output_tokens", 0) / 1e6 * price[1]


def _better(a: float, b: float, direction: str) -> bool:
    return a < b if direction == "minimize" else a > b


@dataclass
class Governor:
    max_cost: float | None = None        # USD ceiling on cumulative LLM spend
    max_rounds: int | None = None        # hard iteration cap
    dry_patience: int | None = None      # stop after this many rounds with no frontier improvement
    _rounds: int = 0
    _dry: int = 0
    _best: float | None = None
    _alerted: set = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.dry_patience is not None and self.dry_patience < 1:
            raise ValueError("dry_patience must be >= 1 (rounds with no improvement before stopping)")
        if self.max_cost is None and self.max_rounds is None and self.dry_patience is None:
            raise ValueError("Governor needs at least one stop condition "
                             "(max_cost / max_rounds / dry_patience) or the loop never ends")

    def update_frontier(self, score: float | None, direction: str) -> bool:
        """Fold a fresh result into the tracked frontier. Returns True iff it improved the best so far."""
        if score is None:
            return False
        if self._best is None or _better(score, self._best, direction):
            self._best = score
            return True
        return False

    def record_round(self, improved: bool) -> None:
        """Close out a round: bump the counter and the dry streak (reset on any improvement)."""
        self._rounds += 1
        self._dry = 0 if improved else self._dry + 1

    def alerts(self, spent: float | None) -> list[str]:
        """Newly-crossed budget thresholds (50/80/100%), each emitted at most once."""
        if spent is None or self.max_cost is None or self.max_cost <= 0:
            return []
        out = []
        for pct in (50, 80, 100):
            if spent >= self.max_cost * pct / 100 and pct not in self._alerted:
                self._alerted.add(pct)
                out.append(f"budget {pct}% — spent ${spent:.2f} of ${self.max_cost:.2f}")
        return out

    def should_stop(self, spent: float | None) -> tuple[bool, str]:
        """Whether to end the campaign now, and why. Checked at the TOP of each round."""
        if self.max_cost is not None and spent is not None and spent >= self.max_cost:
            return True, f"budget exhausted (spent ${spent:.2f} ≥ ${self.max_cost:.2f})"
        if self.max_rounds is not None and self._rounds >= self.max_rounds:
            return True, f"round cap reached ({self._rounds} rounds)"
        if self.dry_patience is not None and self._dry >= self.dry_patience:
            return True, f"converged — {self._dry} rounds with no frontier improvement"
        return False, ""

    @property
    def rounds(self) -> int:
        return self._rounds
