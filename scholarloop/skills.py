"""Skill Library — the PhD's accumulating intuition.

A deterministic store of lessons distilled from past runs. Each lesson decays in influence
over time (half-life ~30 days), so recent experience outweighs stale experience without ever
being deleted. The Reflector writes lessons here; the Reasoner reads the top-weighted ones
into its prompt next round. No training, backbone-agnostic — just prompt overlays.

Dedup is by content hash: the same lesson (category + mitigation) maps to the same file, so a
judge-rejected direction can't silently reappear as a fresh skill every round — it refreshes
in place. The store is plain files under a directory (`arc-<id>.json`), trivially inspectable.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Tiny lexical relevance, no embedding model needed (deterministic, offline). Used to surface the
# lessons that actually bear on the CURRENT direction, not just the globally heaviest ones.
_STOP = {"the", "a", "an", "to", "of", "and", "or", "is", "in", "on", "for", "with", "at", "by",
         "be", "it", "this", "that", "from", "as", "are", "was", "not", "but", "than", "then",
         "when", "its", "into", "rather", "use", "using", "dont", "do", "you", "your", "can"}


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2 and t not in _STOP}


def _relevance(skill: "Skill", query_terms: set[str]) -> float:
    """Fraction of the query's terms this lesson touches (0 when there's no query)."""
    if not query_terms:
        return 0.0
    st = _terms(f"{skill.category} {skill.mitigation}")
    return len(st & query_terms) / len(query_terms)


@dataclass
class Skill:
    id: str
    category: str
    severity: float       # (0, 1] — how strongly this lesson should weigh in
    mitigation: str       # the actionable lesson, written for the next reasoning prompt
    source: str           # the experiment id that produced it
    ts: float             # when the originating experiment ran (decay is measured from here)

    @staticmethod
    def make(category: str, severity: float, mitigation: str, source: str, ts: float) -> "Skill":
        key = f"{category.strip().lower()}|{mitigation.strip().lower()}"
        sid = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        return Skill(id=sid, category=category, severity=float(severity),
                     mitigation=mitigation, source=source, ts=ts)

    def weight(self, now: float, half_life_days: float = 30.0) -> float:
        """Time-decayed influence: severity halves every `half_life_days`."""
        dt_days = max(0.0, (now - self.ts) / 86400.0)
        return self.severity * (2.0 ** (-dt_days / half_life_days))


class SkillLibrary:
    def __init__(self, path: str | Path, *, half_life_days: float = 30.0):
        self.dir = Path(path)
        self.half_life_days = half_life_days

    @classmethod
    def for_domain(cls, domain: str, *, root: str | Path | None = None,
                   half_life_days: float = 30.0) -> "SkillLibrary":
        """A library persisted per-domain so intuition compounds ACROSS campaigns (not just within
        one run). Defaults to ~/.scholarloop/skills/<domain>/; skills stay isolated per domain."""
        base = Path(root) if root is not None else (Path.home() / ".scholarloop" / "skills")
        return cls(base / domain, half_life_days=half_life_days)

    def add(self, skill: Skill) -> Skill:
        """Write (or refresh in place, by content id) a lesson."""
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"arc-{skill.id}.json").write_text(
            json.dumps(asdict(skill), indent=2, ensure_ascii=False))
        return skill

    def all(self) -> list[Skill]:
        if not self.dir.exists():
            return []
        out = []
        for f in sorted(self.dir.glob("arc-*.json")):
            try:
                out.append(Skill(**json.loads(f.read_text())))
            except (json.JSONDecodeError, TypeError):
                continue  # skip a corrupt skill file rather than break the read
        return out

    def active(self, now: float | None = None, *, top_k: int | None = None,
               min_weight: float = 0.0, query: str | None = None,
               relevance_boost: float = 2.0) -> list[tuple[Skill, float]]:
        """(skill, weight) pairs, capped/filtered. Ranked by decayed weight, and — when a `query` is
        given — boosted by lexical relevance to it, so the lessons shown bear on the current
        direction rather than just being the globally heaviest. query=None reproduces pure weight order."""
        now = time.time() if now is None else now
        qterms = _terms(query) if query else set()
        scored = []
        for s in self.all():
            w = s.weight(now, self.half_life_days)
            if w <= min_weight:
                continue
            rank = w * (1.0 + relevance_boost * _relevance(s, qterms))   # relevance lifts, weight floors
            scored.append((s, w, rank))
        scored.sort(key=lambda x: x[2], reverse=True)
        pairs = [(s, w) for s, w, _ in scored]
        return pairs[:top_k] if top_k is not None else pairs

    def render(self, now: float | None = None, *, top_k: int = 5, query: str | None = None) -> str:
        """Render the top lessons as a prompt block for the Reasoner's `skills` slot. Pass `query`
        (the current topic/guidance/priors) to bias selection toward relevant lessons."""
        rows = self.active(now, top_k=top_k, query=query)
        if not rows:
            return ""
        return "\n".join(f"- [{s.category}, w={w:.2f}] {s.mitigation}" for s, w in rows)
