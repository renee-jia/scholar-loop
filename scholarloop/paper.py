"""L5 Writer + Reviewer — turn confirmed findings into a draft, grounded and peer-reviewed.

Three pieces, on the agent harness:
  - `Writer` drafts a short paper from the kept ledger findings.
  - **Number grounding** (the anti-hallucination gate): every number in the draft must trace
    to a recorded fact — a captured measurement (VerifiedRegistry) or a logged config value.
    Any ungrounded number is flagged, so prose can never invent results.
  - `Reviewer` critiques the draft (strengths / weaknesses / score / recommendation).

The grounding is deterministic code, not an agent — the Writer proposes, the registry verifies.
"""

from __future__ import annotations

from pathlib import Path

from scholarloop.agent import Agent, AgentTrace
from scholarloop.ledger import LedgerEntry
from scholarloop.llm import LLMClient
from scholarloop.registry import VerifiedRegistry

WRITER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "abstract": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False,
                      "properties": {"heading": {"type": "string"}, "body": {"type": "string"}},
                      "required": ["heading", "body"]},
        },
    },
    "required": ["title", "abstract", "sections"],
}

REVIEWER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "number"},                  # 1-10, validated in postcheck
        "recommendation": {"type": "string", "enum": ["accept", "weak_accept", "reject"]},
    },
    "required": ["summary", "strengths", "weaknesses", "score", "recommendation"],
}


def gather_findings(entries: list[LedgerEntry]) -> list[dict]:
    """Confirmed results: kept entries with a measured score."""
    out = []
    for e in entries:
        if e.verdict == "kept" and e.primary_score() is not None:
            out.append({"id": e.id, "score": e.primary_score(), "metric": e.metric_name,
                        "config": e.config, "source": e.hypothesis.source,
                        "claim": e.hypothesis.claim})
    return out


def grounded_registry(entries: list[LedgerEntry], registry_dir: str | Path,
                      extra: list[float] | None = None) -> VerifiedRegistry:
    """Every number a paper may cite: each kept entry's config values, its score, and its
    captured measurements, plus any `extra` recorded facts (e.g. the must-beat baseline). The
    audit then rejects any number not in this set."""
    reg = VerifiedRegistry(path=":grounded:", exp_id="paper")
    for v in (extra or []):
        reg.capture(f"extra.{v}", float(v))
    for e in entries:
        if e.verdict != "kept":
            continue
        for k, v in (e.config or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                reg.capture(f"{e.id}.cfg.{k}", float(v))
        s = e.primary_score()
        if s is not None:
            reg.capture(f"{e.id}.score", s)
        p = Path(registry_dir) / f"{e.registry_id or e.id}.json"
        if p.exists():
            for k, m in VerifiedRegistry.load(p).measurements.items():
                reg.measurements[f"{e.id}.{k}"] = m
    return reg


def audit_draft(draft: dict, reg: VerifiedRegistry, ignore: set[float] | None = None) -> list[str]:
    """Ungrounded numbers anywhere in the draft (empty == every number is backed by a fact).

    Citation years (bare integers in 1900–2099) are ignored — they're references, not results.
    """
    text = "\n".join([draft.get("title", ""), draft.get("abstract", "")]
                     + [f"{s['heading']}\n{s['body']}" for s in draft.get("sections", [])])
    ignore = set(ignore or []) | {float(y) for y in range(1900, 2100)}
    return reg.audit_text(text, ignore=ignore)


class Writer(Agent):
    name = "writer"
    system = ("You write concise ML research papers from confirmed experimental findings. Use "
              "ONLY the numbers given to you (scores and config values) — never invent a result. "
              "Cite the arXiv source for each idea. Keep it short: abstract + method + results.")
    schema = WRITER_SCHEMA

    def build_prompt(self, ctx: dict) -> str:
        rows = "\n".join(
            f"- {f['id']}: {f['metric']}={f['score']} with config {f['config']} "
            f"(idea: {f['claim']}, source {f['source']})" for f in ctx["findings"])
        return ("Write a short paper from these confirmed findings. Report only these numbers; "
                "do not fabricate any value.\n\nFindings:\n" + rows)

    def draft(self, findings: list[dict]) -> dict:
        return self.run({"findings": findings})


class Reviewer(Agent):
    name = "reviewer"
    system = ("You are a peer reviewer. Critique the draft for soundness, clarity, and "
              "significance. Give strengths, weaknesses, a score in 1-10, and a recommendation.")
    schema = REVIEWER_SCHEMA

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        s = output.get("score")
        return [] if isinstance(s, (int, float)) and 1 <= s <= 10 else ["score must be in 1-10"]

    def build_prompt(self, ctx: dict) -> str:
        d = ctx["draft"]
        body = "\n\n".join(f"## {s['heading']}\n{s['body']}" for s in d.get("sections", []))
        return (f"Review this draft.\n\nTitle: {d['title']}\nAbstract: {d['abstract']}\n\n{body}\n\n"
                "Give summary, strengths, weaknesses, score (1-10), recommendation.")

    def review(self, draft: dict) -> dict:
        return self.run({"draft": draft})


class PaperPipeline:
    """Draft → ground → review. Returns the draft, ungrounded numbers, and the review."""

    def __init__(self, writer_llm: LLMClient, reviewer_llm: LLMClient, *,
                 registry_dir: str | Path = "registry", trace: AgentTrace | None = None):
        self.writer = Writer(writer_llm, trace=trace)
        self.reviewer = Reviewer(reviewer_llm, trace=trace)
        self.registry_dir = Path(registry_dir)

    def run(self, entries: list[LedgerEntry], extra_grounded: list[float] | None = None) -> dict:
        findings = gather_findings(entries)
        draft = self.writer.draft(findings)
        reg = grounded_registry(entries, self.registry_dir, extra=extra_grounded)
        ungrounded = audit_draft(draft, reg)
        review = self.reviewer.review(draft)
        return {"draft": draft, "ungrounded": ungrounded,
                "grounded": not ungrounded, "review": review}
