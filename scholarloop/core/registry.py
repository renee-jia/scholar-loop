"""VerifiedRegistry: the #1 anti-hallucination / anti reward-hacking lever.

At runtime the experiment harness captures every measured number (per-condition
mean/std, per-seed scores) and freezes it here, keyed by a stable claim name. Later,
any number appearing in a generated artifact (a verdict, a paper draft) is checked
against the registry: if a reported number is not a captured measurement, the
conclusion is REJECTED. Prose can never override the measured direction.

Two guarantees:
  1. Grounding   - every claimed number must trace to a captured artifact.
  2. Consistency - a verdict's claimed direction must match the measured direction.

Number scanning is token-based (not a raw regex over prose) so identifiers like
arXiv ids ("arXiv:1609.04836") and hyphenated terms ("top-1", "resnet18") are NOT
mistaken for measurements. Grounding is checked at the *reported* number's decimal
precision, so a draft that rounds a 5.8333 mean to "5.83" still grounds.

Usage:
    reg = VerifiedRegistry("registry/exp_0427.json")
    reg.capture("val_top1_err.smoke", 6.2, seeds=[6.2])
    reg.capture("val_top1_err.verify", 5.8, seeds=[5.8, 5.9, 5.7])
    reg.save()

    bad = reg.audit_text("We reach 4.1 top1 error, beating the baseline.")
    # -> ["4.1"]  (4.1 was never measured -> reject the draft)
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# A token is a pure number only if, after peeling surrounding punctuation, the whole
# token matches this. "top-1", "arXiv:1609.04836", "resnet18" do NOT match.
_PURE_NUMBER_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$")
# Punctuation that may wrap a number in prose (but NOT '.', which is part of decimals).
_WRAP = "([{<>}]),;:%\"'*`|~$\u2248\u2020"
# A number written with a unit ("2.3x", "3.10pp") or joined to another ("3.10\u00b10.02", "4.1/5.2",
# "3.10\u20133.20") is still a number the text states.
_UNIT_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)(?:x|pp|\u00d7)")
_JOIN_RE = re.compile(r"[\u00b1/\u2013\u2014]")
# Delimiters to split on, in addition to whitespace, so embedded numbers like "err=4.1" or
# "(4.1,5.2)" are caught. Deliberately EXCLUDES ':' '.' '-' so arXiv ids ("arXiv:1609.04836"),
# decimals, and hyphenated terms ("top-1") stay single, non-numeric tokens.
_SPLIT_RE = re.compile(r"[\s=,;()\[\]<>{}]+")


def _number_tokens(text: str):
    """Yield the bare numeric tokens in `text` (e.g. "5.8", "-0.3", "1e-4").

    Splits on whitespace and structural delimiters, then peels wrapping punctuation, so
    standalone numbers — including ones embedded after '=' or inside parens/commas — are
    yielded. Ranges like "5.5-4.9" and identifiers like "top-1" / "arXiv:1609.04836" are skipped.
    """
    for raw in _SPLIT_RE.split(text):
        # Peel wrappers and trailing sentence periods until stable: "4.1%." and "**4.1**." both
        # reduce to "4.1". A Unicode minus counts as a minus sign.
        tok = raw.replace("\u2212", "-")
        while True:
            peeled = tok.strip(_WRAP)
            if peeled.endswith(".") and not _PURE_NUMBER_RE.match(peeled):
                peeled = peeled[:-1]
            if peeled == tok:
                break
            tok = peeled
        if _PURE_NUMBER_RE.match(tok):
            yield tok
            continue
        unit = _UNIT_RE.fullmatch(tok)
        if unit:
            yield unit.group(1)
            continue
        parts = [_UNIT_RE.sub(r"\1", p.strip(_WRAP)) if _UNIT_RE.fullmatch(p.strip(_WRAP)) else p.strip(_WRAP)
                 for p in _JOIN_RE.split(tok)]
        if len(parts) > 1 and all(_PURE_NUMBER_RE.match(p) for p in parts):
            yield from parts


def _decimals(tok: str) -> int:
    """Number of significant fractional digits a numeric token carries, exponent included:
    "5.83" -> 2, "3e-4" -> 4, "2.5e-3" -> 4, "2.5e3" -> 0. Without the exponent term "3e-4" would
    be matched at integer precision, where every measurement below 0.5 "grounds" it."""
    mantissa, _, exp = tok.lower().partition("e")
    frac = len(mantissa.split(".")[1]) if "." in mantissa else 0
    try:
        return max(0, frac - int(exp)) if exp else frac
    except ValueError:
        return frac


@dataclass
class Measurement:
    value: float
    seeds: list[float] = field(default_factory=list)
    std: float | None = None

    def to_dict(self) -> dict:
        return {"value": self.value, "seeds": self.seeds, "std": self.std}


@dataclass
class VerifiedRegistry:
    """A frozen record of every number an experiment actually measured."""

    path: str | Path
    exp_id: str = ""
    measurements: dict[str, Measurement] = field(default_factory=dict)
    status: str = "ok"          # "ok" | "killed" — a killed run captured no measurements
    tol: float = 1e-6           # absolute tolerance for an exact-precision match

    # ---- capture (write side, called by the harness) ----
    def capture(self, key: str, value: float, seeds: list[float] | None = None,
                std: float | None = None) -> None:
        """Freeze one measured number under `key` (e.g. "val_top1_err.verify")."""
        self.measurements[key] = Measurement(float(value), list(seeds or []), std)

    def save(self) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"exp_id": self.exp_id, "status": self.status,
                   "measurements": {k: m.to_dict() for k, m in self.measurements.items()}}
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VerifiedRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        reg = cls(path=path, exp_id=data.get("exp_id", ""), status=data.get("status", "ok"))
        for k, m in data.get("measurements", {}).items():
            reg.measurements[k] = Measurement(m["value"], m.get("seeds", []), m.get("std"))
        return reg

    # ---- verify (read side, called before trusting an artifact) ----
    def _all_values(self) -> list[float]:
        vals: list[float] = []
        for m in self.measurements.values():
            vals.append(m.value)
            vals.extend(m.seeds)
        return vals

    exact_integers = False   # set by checks where "7" must not stand for a recorded 7.25

    def is_grounded(self, number: float, decimals: int | None = None) -> bool:
        """True if `number` matches a captured measurement.

        When `decimals` is given, the match is made at that precision (so a measured
        5.8333 grounds a reported "5.83"). Otherwise an absolute tolerance is used.
        """
        for v in self._all_values():
            if decimals is None:
                if abs(number - v) <= self.tol:
                    return True
            elif decimals == 0 and self.exact_integers:
                if abs(number - v) < 1e-9:
                    return True
            # Either rounding convention is accepted: 0.8125 grounds both "0.812" and "0.813".
            elif round(v, decimals) == round(number, decimals) or abs(v - number) <= 0.5 * 10 ** -decimals + 1e-12:
                return True
        return False

    def audit_text(self, text: str, ignore: set[float] | None = None) -> list[str]:
        """Return the list of ungrounded numeric tokens found in `text`.

        An empty list means every number in the text traces to a real measurement.
        `ignore` lets callers whitelist structural constants (e.g. years, section nums).
        """
        ignore = ignore or set()
        bad: list[str] = []
        for tok in _number_tokens(text):
            num = float(tok)
            if num in ignore:
                continue
            if not self.is_grounded(num, decimals=_decimals(tok)):
                bad.append(tok)
        return bad

    def check_direction(self, key: str, baseline: float, claim: str,
                        metric_direction: str) -> bool:
        """Verify a 'beats baseline' style claim against the measured direction.

        `claim` in {"beats", "loses"}. Returns False on an inverted verdict — a draft
        that says "supported" while the measurement points the other way is rejected
        regardless of its prose.
        """
        if key not in self.measurements:
            return False
        measured = self.measurements[key].value
        better = (measured < baseline) if metric_direction == "minimize" else (measured > baseline)
        if claim == "beats":
            return better
        if claim == "loses":
            return not better
        raise ValueError(f"claim must be 'beats' or 'loses', got {claim!r}")


def _main(argv: list[str]) -> int:
    # python -m scholarloop.core.registry audit <registry.json> "<text to check>"
    if len(argv) != 3 or argv[0] != "audit":
        print('usage: python -m scholarloop.core.registry audit <registry.json> "<text>"', file=sys.stderr)
        return 2
    reg = VerifiedRegistry.load(argv[1])
    bad = reg.audit_text(argv[2])
    if bad:
        print(f"REJECTED: ungrounded numbers {bad}", file=sys.stderr)
        return 1
    print("OK: every number is grounded in a measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
