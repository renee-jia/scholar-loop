"""Experiment Ledger: the framework's cross-run memory.

Every experiment appends exactly one JSON line to `ledger.jsonl`. Nothing about an
experiment lives only in an agent's context window (which compresses away over long
runs); the ledger is the durable source of truth that the Director, novelty dedup,
and meta-analysis all read from.

An entry records the hypothesis (with its literature source), the diff, the metric at
each fidelity tier, and the final verdict. The append is atomic-per-line so concurrent
workers never interleave partial records.

CLI:
    python -m scholarloop.ledger show ledger.jsonl
    python -m scholarloop.ledger best ledger.jsonl --domain image-classification
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator

# Serializes appends within a process so parallel workers (the population funnel) can't interleave
# records even when an entry exceeds the OS atomic-write size (PIPE_BUF). Cross-process writers still
# rely on O_APPEND atomicity, which holds for the common small-line case.
_WRITE_LOCK = threading.Lock()

VERDICTS = ("kept", "discarded", "killed", "pending")
FIDELITIES = ("smoke", "verify", "full")


@dataclass
class Hypothesis:
    claim: str
    source: str                      # e.g. "arXiv:2401.xxxxx"; "" is rejected by validate()
    predicted_effect: str = ""
    how_to_implement: str = ""


@dataclass
class LedgerEntry:
    id: str
    domain: str
    hypothesis: Hypothesis
    metric_name: str
    parent: str | None = None
    diff: str = ""
    config: dict[str, Any] = field(default_factory=dict)     # hparam snapshot this run used (for search-space reasoning)
    fidelity: list[str] = field(default_factory=list)        # tiers this entry reached
    metric: dict[str, Any] = field(default_factory=dict)     # {"smoke": 6.2, "verify": 5.8, "seeds": [...]}
    reasoning: dict[str, Any] = field(default_factory=dict)  # ReasoningTrace (§4.5): trace/act/search_space_constraints
    prediction: dict[str, Any] = field(default_factory=dict) # {"predicted":..,"measured":..,"calibration_error":..}
    verdict: str = "pending"
    registry_id: str | None = None
    ts: float = field(default_factory=time.time)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("entry.id is required")
        if not self.hypothesis.source:
            raise ValueError(f"{self.id}: hypothesis.source is required (no ungrounded ideas in the ledger)")
        if self.verdict not in VERDICTS:
            raise ValueError(f"{self.id}: verdict {self.verdict!r} not in {VERDICTS}")
        for fid in self.fidelity:
            if fid not in FIDELITIES:
                raise ValueError(f"{self.id}: fidelity {fid!r} not in {FIDELITIES}")

    def primary_score(self) -> float | None:
        """Best (highest-fidelity) recorded score for the primary metric."""
        for fid in reversed(FIDELITIES):
            if fid in self.metric:
                return float(self.metric[fid])
        return None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "LedgerEntry":
        # Tolerate schema evolution: keep only fields this version knows about, so an
        # older/newer ledger line (or a hand edit) does not break read-back.
        known = {f.name for f in fields(cls)}
        hyp_known = {f.name for f in fields(Hypothesis)}
        d = {k: v for k, v in d.items() if k in known}
        d["hypothesis"] = Hypothesis(**{k: v for k, v in d.get("hypothesis", {}).items() if k in hyp_known})
        return cls(**d)


class Ledger:
    """Append-only JSONL ledger with atomic per-line writes."""

    def __init__(self, path: str | Path = "ledger.jsonl"):
        self.path = Path(path)

    def append(self, entry: LedgerEntry) -> LedgerEntry:
        entry.validate()
        line = entry.to_json() + "\n"
        # O_APPEND writes are atomic for a single line below PIPE_BUF, so concurrent
        # workers cannot interleave partial records.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
        return entry

    def read_all(self) -> Iterator[LedgerEntry]:
        """Yield every parseable entry. A single corrupt line is skipped (warned on
        stderr), never aborting the read of an append-only source of truth."""
        if not self.path.exists():
            return
        with self.path.open() as f:
            for lineno, ln in enumerate(f, 1):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield LedgerEntry.from_dict(json.loads(ln))
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    print(f"ledger: skipping corrupt line {lineno} in {self.path}: {e}", file=sys.stderr)

    def best(self, domain: str, direction: str = "minimize") -> LedgerEntry | None:
        """Best kept entry for a domain under the given metric direction."""
        kept = [e for e in self.read_all() if e.domain == domain and e.verdict == "kept"
                and e.primary_score() is not None]
        if not kept:
            return None
        key = (lambda e: e.primary_score())
        return min(kept, key=key) if direction == "minimize" else max(kept, key=key)


_TIER_ORDER = {"smoke": 0, "verify": 1, "full": 2}


def idea_chains(entries: list[LedgerEntry]) -> list[dict]:
    """Roll the flat ledger up to the level of *ideas*: group the funnel tiers of one idea
    (which share a config) and report, per idea, the tiers it reached and its highest-fidelity
    result. Useful for campaign summaries (Director, write-up) and analysis."""
    groups: dict[str, list[LedgerEntry]] = {}
    for e in entries:
        groups.setdefault(json.dumps(e.config, sort_keys=True), []).append(e)
    out = []
    for tiers in groups.values():
        top = max(tiers, key=lambda e: _TIER_ORDER.get(e.fidelity[0] if e.fidelity else "smoke", 0))
        out.append({
            "config": tiers[0].config,
            "source": tiers[0].hypothesis.source,
            "tiers": [e.fidelity[0] for e in tiers if e.fidelity],
            "top_fidelity": top.fidelity[0] if top.fidelity else None,
            "score": top.primary_score(),
            "verdict": top.verdict,
        })
    return out


def _main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m scholarloop.ledger {show|best} <ledger.jsonl> [--domain D] [--direction minimize|maximize]",
              file=sys.stderr)
        return 0 if argv else 2
    cmd, path = argv[0], argv[1] if len(argv) > 1 else "ledger.jsonl"
    ledger = Ledger(path)
    if cmd == "show":
        for e in ledger.read_all():
            print(f"{e.id:>10}  {e.domain:<22} {e.verdict:<10} "
                  f"{e.metric_name}={e.primary_score()}  src={e.hypothesis.source}")
        return 0
    if cmd == "best":
        domain = _flag(argv, "--domain")
        direction = _flag(argv, "--direction") or "minimize"
        if not domain:
            print("best requires --domain", file=sys.stderr)
            return 2
        b = ledger.best(domain, direction)
        print("none" if b is None else f"{b.id}  {b.metric_name}={b.primary_score()}  {b.hypothesis.claim}")
        return 0
    print(f"unknown command {cmd!r}", file=sys.stderr)
    return 2


def _flag(argv: list[str], name: str) -> str | None:
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
