"""Literature Review — the first deliverable of a research session.

Before any experiment is proposed, the loop should *understand the field*: what the main lines of
work are, what each key paper contributes and where it stops, and which questions are still open.
That understanding is a document a human can read and push back on, not a hidden prompt.

Two agents on the harness, one deterministic retrieval step between them:
  - `QueryPlanner`  — topic (+ human feedback) → several concrete search queries, so retrieval
    covers the neighbourhood of the topic instead of one literal string.
  - `retrieve()`    — plain HTTP against the configured sources (arXiv / OpenAlex / …), merged,
    de-duplicated, citation-ranked with slots reserved for fresh preprints (`litscout.retrieve`).
  - `LitReviewer`   — the corpus → a structured review: overview, themes, per-paper notes, open
    gaps, trends. **Citation grounding** is enforced deterministically: every paper id the review
    cites must be one of the retrieved papers, or the output is rejected and retried — the
    literature analogue of the VerifiedRegistry's number grounding.

`render_markdown` turns the review into the `lit_review.md` a human reads at the first checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from scholarloop.core.agent import Agent
from scholarloop.ideation.litscout import Paper, render_papers

QUERY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"queries": {"type": "array", "items": {"type": "string"}},
                   "rationale": {"type": "string"}},
    "required": ["queries", "rationale"],
}

REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "overview": {"type": "string"},
        "themes": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False,
                      "properties": {"name": {"type": "string"}, "summary": {"type": "string"},
                                     "papers": {"type": "array", "items": {"type": "string"}}},
                      "required": ["name", "summary", "papers"]},
        },
        "paper_notes": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False,
                      "properties": {"id": {"type": "string"}, "contribution": {"type": "string"},
                                     "limitation": {"type": "string"}},
                      "required": ["id", "contribution", "limitation"]},
        },
        "gaps": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False,
                      "properties": {"gap": {"type": "string"}, "why_open": {"type": "string"},
                                     "sources": {"type": "array", "items": {"type": "string"}}},
                      "required": ["gap", "why_open", "sources"]},
        },
        "trends": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["overview", "themes", "paper_notes", "gaps", "trends"],
}


class QueryPlanner(Agent):
    name = "query_planner"
    system = ("You plan literature searches for an ML research project. Turn a topic into a few "
              "concrete, diverse search queries (3-6 words each) that together cover the topic's "
              "core methods, its strongest baselines, and adjacent ideas worth borrowing. Queries "
              "go verbatim into arXiv/OpenAlex keyword search, so use real terminology, not prose.")
    schema = QUERY_SCHEMA
    max_queries = 6

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        qs = [q.strip() for q in output.get("queries", []) if isinstance(q, str) and q.strip()]
        if not qs:
            return ["give at least one non-empty query"]
        if len(qs) > self.max_queries:
            return [f"give at most {self.max_queries} queries"]
        return []

    def build_prompt(self, ctx: dict) -> str:
        parts = [f"Topic: {ctx['topic']}"]
        if ctx.get("domain"):
            parts.append(f"Experimental setting: {ctx['domain']}")
        if ctx.get("previous"):
            parts.append(f"Queries already searched (do not repeat): {ctx['previous']}")
        if ctx.get("feedback"):
            parts.append(f"Human feedback to cover: {ctx['feedback']}")
        parts.append(f"Return 3-{self.max_queries} search queries and a one-line rationale.")
        return "\n".join(parts)

    def plan(self, topic: str, *, domain: str = "", previous: list[str] | None = None,
             feedback: str = "") -> list[str]:
        out = self.run({"topic": topic, "domain": domain, "previous": previous or [],
                        "feedback": feedback})
        seen, queries = set(), []
        for q in out["queries"]:
            key = " ".join(q.lower().split())
            if key and key not in seen and key not in {p.lower() for p in (previous or [])}:
                seen.add(key)
                queries.append(q.strip())
        return queries


@dataclass
class LitReview:
    topic: str
    queries: list[str]
    papers: list[Paper]
    overview: str
    themes: list[dict] = field(default_factory=list)       # {name, summary, papers:[id]}
    paper_notes: list[dict] = field(default_factory=list)  # {id, contribution, limitation}
    gaps: list[dict] = field(default_factory=list)         # {gap, why_open, sources:[id]}
    trends: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["papers"] = [asdict(p) for p in self.papers]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LitReview":
        d = dict(d)
        d["papers"] = [Paper(**p) for p in d.get("papers", [])]
        return cls(**d)

    def paper(self, pid: str) -> Paper | None:
        return next((p for p in self.papers if p.arxiv_id == pid), None)

    def to_prompt_block(self, *, max_notes: int = 25) -> str:
        """A compact rendering for downstream agents (the Ideator) — themes, gaps, notes."""
        lines = [f"Overview: {self.overview}", "", "Themes:"]
        for t in self.themes:
            lines.append(f"- {t['name']}: {t['summary']} [{', '.join(t['papers'])}]")
        lines += ["", "Open gaps:"]
        for i, g in enumerate(self.gaps, 1):
            lines.append(f"{i}. {g['gap']} — {g['why_open']} [{', '.join(g['sources'])}]")
        if self.trends:
            lines += ["", "Trends: " + "; ".join(self.trends)]
        lines += ["", "Paper notes:"]
        for n in self.paper_notes[:max_notes]:
            p = self.paper(n["id"])
            title = f" {p.title}" if p else ""
            lines.append(f"- [{n['id']}]{title}: {n['contribution']} Limitation: {n['limitation']}")
        return "\n".join(lines)


class LitReviewer(Agent):
    name = "lit_reviewer"
    system = ("You are a careful researcher writing a literature review for a colleague who will "
              "decide what to work on next. Organize the papers into themes, state what each paper "
              "contributes and where it stops, and name the open gaps — questions the papers raise "
              "but do not settle, or combinations nobody has tried. Cite papers ONLY by the bracketed "
              "ids given; never cite anything not in the list. Be concrete and skeptical: a gap must "
              "be something an experiment could address.")
    schema = REVIEW_SCHEMA

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        ids = {p.arxiv_id for p in ctx["papers"]}
        errs = []
        cited = [pid for t in output.get("themes", []) for pid in t.get("papers", [])]
        cited += [n.get("id") for n in output.get("paper_notes", [])]
        cited += [pid for g in output.get("gaps", []) for pid in g.get("sources", [])]
        unknown = sorted({c for c in cited if c not in ids})
        if unknown:                                    # citation grounding — hallucinated ids are rejected
            errs.append(f"unknown paper ids (cite only the bracketed ids given): {unknown}")
        if not output.get("themes"):
            errs.append("give at least one theme")
        if not output.get("gaps"):
            errs.append("give at least one open gap")
        if not output.get("paper_notes"):
            errs.append("give notes for the key papers")
        return errs

    def build_prompt(self, ctx: dict) -> str:
        parts = [f"Topic: {ctx['topic']}"]
        if ctx.get("domain"):
            parts.append(f"Experimental setting the reader works in: {ctx['domain']}")
        if ctx.get("feedback"):
            parts += ["", "The reader reviewed a previous version and asked for changes — address them:",
                      ctx["feedback"]]
        parts += ["", "Papers (cite by the bracketed id):", "", render_papers(ctx["papers"]), "",
                  "Write the review: overview, themes (each listing its paper ids), a note per key "
                  "paper (contribution + limitation), the open gaps (each with why it is still open "
                  "and the source ids that expose it), and the trends you see."]
        return "\n".join(parts)

    def review(self, topic: str, papers: list[Paper], *, queries: list[str] | None = None,
               domain: str = "", feedback: str = "") -> LitReview:
        out = self.run({"topic": topic, "papers": papers, "domain": domain, "feedback": feedback})
        return LitReview(topic=topic, queries=list(queries or []), papers=list(papers),
                         overview=out["overview"], themes=out["themes"],
                         paper_notes=out["paper_notes"], gaps=out["gaps"], trends=out["trends"])


def render_markdown(r: LitReview) -> str:
    """`lit_review.md` — the human-facing report for the first checkpoint."""
    by_id = {p.arxiv_id: p for p in r.papers}

    def cite(pid: str) -> str:
        p = by_id.get(pid)
        if p is None:
            return f"`{pid}`"
        meta = ", ".join(x for x in [str(p.year) if p.year else "",
                                     f"{p.citations} cites" if p.citations is not None else ""] if x)
        link = f"[{pid}]({p.url})" if p.url else f"`{pid}`"
        return f"{link} *{p.title}*" + (f" ({meta})" if meta else "")

    out = [f"# Literature review — {r.topic}", "",
           f"_{len(r.papers)} papers · {len(r.themes)} themes · {len(r.gaps)} open gaps · "
           f"queries: {' · '.join(f'`{q}`' for q in r.queries) or '—'}_", "",
           "## Overview", "", r.overview, "", "## Themes", ""]
    for t in r.themes:
        out += [f"### {t['name']}", "", t["summary"], ""]
        out += [f"- {cite(pid)}" for pid in t["papers"]] + [""]
    out += ["## Open gaps", ""]
    for i, g in enumerate(r.gaps, 1):
        srcs = ", ".join(f"`{s}`" for s in g["sources"]) or "—"
        out += [f"{i}. **{g['gap']}**  ", f"   {g['why_open']} (sources: {srcs})", ""]
    if r.trends:
        out += ["## Trends", ""] + [f"- {t}" for t in r.trends] + [""]
    out += ["## Paper notes", "", "| paper | contribution | limitation |", "|---|---|---|"]
    for n in r.paper_notes:
        out.append(f"| {cite(n['id'])} | {n['contribution']} | {n['limitation']} |")
    out += ["", "## Corpus", ""]
    for p in r.papers:
        out.append(f"- {cite(p.arxiv_id)}")
    out += ["", "*Retrieved from arXiv / OpenAlex and reviewed by ScholarLoop's LitReviewer agent. "
            "Every cited id is checked against the retrieved corpus.*", ""]
    return "\n".join(out)
