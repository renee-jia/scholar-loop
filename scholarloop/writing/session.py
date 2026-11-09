"""Step-by-step writing — outline → key points → prose, with AI discussion and human sign-off.

A one-shot "write the paper" call produces fluent text with weak arguments. This module makes the
argument explicit *before* any prose exists, and lets different agents argue about it:

    evidence bank ──▶ [1] OUTLINE      Outliner proposes ─▶ panel (Skeptic · Editor · Methodologist)
                           │           argues ─▶ Outliner revises … until the panel accepts or the
                           │           round cap hits ──▶ outline.md   ◀── human: approve / feedback
                           ▼
                      [2] KEY POINTS   per section: claim + the evidence ids that back it, same
                           │           discussion loop ──▶ key_points.md   ◀── human: approve / feedback
                           ▼
                      [3] DRAFT        one SectionDrafter call per section, from the agreed points,
                           │           under the bundled style guide ──▶ paper.md
                           ▼
                      [4] REVIEW       three persona reviewers score it ──▶ review.md

What is enforced in code, not asked for in a prompt:
  - **evidence grounding** — every evidence id a key point cites is a real run id, a reviewed paper
    id, or `baseline`; unknown ids are rejected and the agent retried;
  - **number grounding** — every number in a key point or a drafted section traces to a recorded
    measurement, a logged config value, or the baseline (`VerifiedRegistry`);
  - **prose lint** (`lint_prose`) — machine-writing tells from `skills/style.md` (banned
    vocabulary, em-dashes, "not X but Y", hype, stacked connectives) are rejected and retried;
  - **coverage** — a drafted section that drops one of its agreed key points is flagged.
The LLM does only the open-ended work: arguing, organizing, and writing sentences. The bundled
`skills/*.md` files are the agents' system-prompt overlays (see that folder's README).

State lives in `writing.json` under the output directory (`new → outlined → keypointed → drafted →
reviewed`), so a session runs interactively (`WritingSession.interactive()`), from the CLI across
invocations, or from another program. `examples/run_to_paper.py` ends every live campaign in it.

    python -m scholarloop.writing --profile profiles/digits-mlp.yaml \
        --ledger runs/s1/experiments.jsonl --registry runs/s1/registry \
        --research runs/s1 --out runs/s1            # interactive
    ... --approve                                   # advance one stage, non-interactive
    ... --feedback "lead with the negative result"  # revise the current stage (after the draft:
                                                    #   the key points; the outline is kept)
    ... --auto                                      # every stage, no sign-off
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from scholarloop.core.agent import Agent, AgentError, AgentTrace
from scholarloop.core.ledger import Ledger, LedgerEntry
from scholarloop.ideation.litreview import LitReview
from scholarloop.core.profile import Profile
from scholarloop.core.registry import VerifiedRegistry
from scholarloop.core.text import containment

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def load_skill(*names: str) -> str:
    """Concatenate bundled writing-skill files (e.g. load_skill("narrative", "style"))."""
    return "\n\n".join((SKILLS_DIR / f"{n}.md").read_text() for n in names)


# ---------------------------------------------------------------------------------------------
# Prose lint — the enforceable half of skills/style.md
# ---------------------------------------------------------------------------------------------
# Words with no legitimate technical sense in a results paper. "foster" (a surname) and
# "underscore" (the character) are NOT here; their machine-writing senses are caught as phrases.
_BANNED_WORDS = re.compile(
    r"\b(delv(?:e|es|ed|ing)|leverag(?:e|es|ed|ing)|tapestry|testament|"
    r"pivotal|showcas(?:e|es|ed|ing)|realm|seamless(?:ly)?|multifaceted|"
    r"paradigm shift|groundbreaking|intricate(?:ly)?|crucial(?:ly)?|vital(?:ly)?)\b", re.IGNORECASE)
_BANNED_PHRASES = [
    "underscores the", "underscore the", "underscoring the", "fosters a", "foster a", "fostering a",
    "it is worth noting", "it should be noted", "it is important to note", "in recent years",
    "with the rapid development", "despite recent advances", "pave the way", "paves the way",
    "shed light on", "sheds light on", "bridge the gap", "bridges the gap", "open new avenues",
    "opens new avenues", "crucial step", "potential to revolutionize", "to the best of our knowledge",
    "for the first time", "extensive experiments", "in order to", "the landscape of",
]
_CONTRAST = [re.compile(r"\bnot (?:only|just|merely|simply)\b[^.?!]{0,80}\bbut\b", re.IGNORECASE),
             re.compile(r"\b(?:is|are|was|were|does|do|did)\s+not\b[^.?!]{0,60},\s*but\b", re.IGNORECASE)]
_OPENERS = ("moreover", "furthermore", "additionally", "notably", "importantly")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def lint_prose(text: str) -> list[str]:
    """Hard failures: patterns that read as machine-written (see skills/style.md).
    Deterministic; returned strings are the retry feedback."""
    errs: list[str] = []
    if "—" in text:
        errs.append("remove em-dashes (recast with a comma, colon, parentheses or a new sentence)")
    words = sorted({m.group(0).lower() for m in _BANNED_WORDS.finditer(text)})
    if words:
        errs.append(f"banned vocabulary: {words}")
    phrases = [p for p in _BANNED_PHRASES if p in text.lower()]
    if phrases:
        errs.append(f"banned phrases: {phrases}")
    for rx in _CONTRAST:
        m = rx.search(text)
        if m:
            errs.append(f'"not X, but Y" contrast frame: "{m.group(0)[:70]}" (state Y directly)')
            break
    sents = _sentences(text)
    for a, b in zip(sents, sents[1:]):
        if a.split(" ", 1)[0].rstrip(",").lower() in _OPENERS and \
                b.split(" ", 1)[0].rstrip(",").lower() in _OPENERS:
            errs.append(f'consecutive sentences open with connectives ("{a[:30]}…" / "{b[:30]}…")')
            break
    return errs


def prose_warnings(text: str, max_words: int = 40) -> list[str]:
    """Soft signals surfaced to the human, not retried: overlong sentences, unquantified
    "significantly", "novel" more than once."""
    warns: list[str] = []
    long = [s for s in _sentences(text) if len(s.split()) > max_words]
    if long:
        warns.append(f"{len(long)} sentence(s) over {max_words} words (first: \"{long[0][:60]}…\")")
    for s in _sentences(text):
        if "significant" in s.lower() and not re.search(r"\d", s):
            warns.append(f'"significant" without a number: "{s[:60]}…"')
            break
    if len(re.findall(r"\bnovel\b", text, re.IGNORECASE)) > 1:
        warns.append('"novel" used more than once')
    soft = ["strong results", "a novel framework", "the landscape", "performance"]
    hits = [w for w in soft if re.search(r"\b" + re.escape(w) + r"\b", text, re.IGNORECASE)]
    if hits:
        warns.append(f"vague wording, name the metric or the claim instead: {hits}")
    return warns


# ---------------------------------------------------------------------------------------------
# Evidence bank — the universe of things a paper may claim
# ---------------------------------------------------------------------------------------------
_YEARS = {float(y) for y in range(1900, 2100)}
# Tiny integers are counts (5 seeds, 3 tiers, section 2), not results: a fabricated metric is never
# a bare 0-10 integer in these domains, while blocking "five seeds" written as "5" is pure friction.
_SMALL_INTS = {float(i) for i in range(0, 11)}
_ID_RX = re.compile(r"(arXiv:[\w.\-/]+|doi:[\w./\-()]+|openalex:\w+|s2:\w+|exp_\d{4,})")


@dataclass
class EvidenceBank:
    findings: list[dict]                      # one row per scored or killed run
    papers: dict[str, str]                    # citation id -> title (from the lit review, if any)
    baseline: float | None
    metric_name: str
    direction: str
    registry: VerifiedRegistry
    idea: dict | None = None                  # the chosen idea (session handoff), if any
    notes: dict[str, dict] = field(default_factory=dict)   # paper id -> {contribution, limitation}

    @classmethod
    def build(cls, entries: list[LedgerEntry], registry_dir: str | Path, profile: Profile, *,
              review: LitReview | None = None, idea: dict | None = None) -> "EvidenceBank":
        reg = VerifiedRegistry(path=":writing:", exp_id="paper")
        baseline = profile.best_baseline()
        if baseline is not None:
            reg.capture("baseline", float(baseline))
        for tier, sec in (("smoke", profile.budget.smoke_sec), ("verify", profile.budget.verify_sec),
                          ("full", profile.budget.full_sec)):
            reg.capture(f"budget.{tier}_sec", float(sec))            # structural facts a paper may state
        for k, v in profile.knobs.items():
            reg.capture(f"knob.default.{k}", float(v))
        rows = []
        for e in entries:
            score = e.primary_score()
            row = {"id": e.id, "fidelity": e.fidelity[0] if e.fidelity else "?", "verdict": e.verdict,
                   "score": score, "seeds": list(e.metric.get("seeds") or []), "std": None,
                   "config": dict(e.config or {}), "claim": e.hypothesis.claim,
                   "source": e.hypothesis.source, "parent": e.parent,
                   "error": e.metric.get("error")}
            for k, v in (e.config or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    reg.capture(f"{e.id}.cfg.{k}", float(v))
            if score is not None:
                reg.capture(f"{e.id}.score", score, seeds=row["seeds"])
                reg.capture(f"{e.id}.n_seeds", float(len(row["seeds"])))
            p = Path(registry_dir) / f"{e.registry_id or e.id}.json"
            if p.exists():
                for k, m in VerifiedRegistry.load(p).measurements.items():
                    reg.measurements[f"{e.id}.{k}"] = m
                    if m.std is not None and k.endswith(row["fidelity"]):
                        row["std"] = m.std
                        reg.capture(f"{e.id}.std", m.std)
            rows.append(row)
        papers = {p.arxiv_id: p.title for p in (review.papers if review else [])}
        notes = {n["id"]: n for n in (review.paper_notes if review else [])}
        return cls(rows, papers, baseline, profile.metric.name, profile.metric.direction, reg,
                   idea=idea, notes=notes)

    def ids(self) -> set[str]:
        return {r["id"] for r in self.findings} | set(self.papers) | {"baseline"}

    def unknown(self, ids) -> list[str]:
        known = self.ids()
        return sorted({i for i in ids if i not in known})

    def ungrounded(self, text: str) -> list[str]:
        return self.registry.audit_text(text, ignore=_YEARS | _SMALL_INTS)

    def describe(self, eid: str) -> str:
        """One line a drafter can quote for an evidence id."""
        if eid == "baseline":
            return f"baseline: {self.baseline} {self.metric_name} (must beat; {self.direction})"
        if eid in self.papers:
            n = self.notes.get(eid)
            return f"[{eid}] {self.papers[eid]}" + (f" — {n['contribution']}" if n else "")
        r = next((r for r in self.findings if r["id"] == eid), None)
        if r is None:
            return eid
        if r["score"] is None:
            return f"{eid} ({r['fidelity']}): killed — {r['error']}"
        spread = f" ± {r['std']} std" if r["std"] is not None else ""
        return (f"{eid} ({r['fidelity']}, {len(r['seeds'])} seed(s)): {self.metric_name}={r['score']}{spread}, "
                f"verdict {r['verdict']}, config {r['config']}, idea: {r['claim']} (source {r['source']})")

    def to_prompt_block(self) -> str:
        lines = [f"Metric: {self.metric_name} ({self.direction}). Baseline to beat: {self.baseline} "
                 f"(cite as `baseline`).", "", "Runs (cite by id; every number you use must come from here):"]
        for r in self.findings:
            lines.append("- " + self.describe(r["id"]))
        if self.papers:
            lines += ["", "Papers you may cite (by the bracketed id only):"]
            lines += [f"- [{pid}] {t}" for pid, t in self.papers.items()]
        if self.idea:
            lines += ["", f"The idea this campaign tested: {self.idea.get('title')} — "
                      f"{self.idea.get('description', '')} Hypothesis: {self.idea.get('hypothesis', {}).get('claim', '')}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------------------------
OUTLINE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "question": {"type": "string"},
        "nugget": {"type": "string"},
        "contributions": {"type": "array", "items": {"type": "string"}},
        "sections": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"heading": {"type": "string"}, "purpose": {"type": "string"},
                           "evidence": {"type": "array", "items": {"type": "string"}}},
            "required": ["heading", "purpose", "evidence"]}},
    },
    "required": ["title", "question", "nugget", "contributions", "sections"],
}

PANEL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["accept", "revise"]},
        "issues": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"target": {"type": "string"}, "issue": {"type": "string"},
                           "suggestion": {"type": "string"}},
            "required": ["target", "issue", "suggestion"]}},
    },
    "required": ["verdict", "issues"],
}

POINT_ROLES = ["context", "challenge", "method", "result", "comparison", "limitation",
               "future_work", "transition"]
_ROLES_NEEDING_EVIDENCE = {"method", "result", "comparison"}

KEYPOINTS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"sections": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"heading": {"type": "string"}, "points": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"claim": {"type": "string"},
                           "evidence": {"type": "array", "items": {"type": "string"}},
                           "role": {"type": "string", "enum": POINT_ROLES}},
            "required": ["claim", "evidence", "role"]}}},
        "required": ["heading", "points"]}}},
    "required": ["sections"],
}

SECTION_SCHEMA = {"type": "object", "additionalProperties": False,
                  "properties": {"body": {"type": "string"}}, "required": ["body"]}

REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "number"},
        "recommendation": {"type": "string", "enum": ["accept", "weak_accept", "weak_reject", "reject"]},
    },
    "required": ["summary", "strengths", "weaknesses", "questions", "score", "recommendation"],
}

PERSONAS = {
    "Skeptic": ("You are the skeptical reviewer. Check claim-evidence discipline: is every claim "
                "backed by a named run or paper, is any verb stronger than its evidence, does any "
                "conclusion reach beyond the dataset / seeds / budget that was actually run?"),
    "Editor": ("You are the narrative editor. Is there exactly one question and one nugget, and "
               "does every section serve them? Is the order the reader's order (what and why "
               "before how)? Flag redundancy, missing transitions, and sections that do not earn "
               "their place."),
    "Methodologist": ("You are the methodologist. Could an expert reproduce this from what is "
                      "planned: data, split, model shape, hyperparameters, seeds, budget, who "
                      "computes the metric? Are the strongest baseline, the spread across seeds, "
                      "and the failures (killed / discarded runs) reported? Is a limitations "
                      "section planned?"),
}

REVIEWER_PERSONAS = {
    "harsh-but-fair": "You are a harsh but fair reviewer who expects experiments that lead to insight.",
    "harsh-critical": "You are a harsh, critical reviewer looking for a result that would matter to the field.",
    "open-minded": "You are an open-minded reviewer looking for a non-obvious idea or finding.",
}


def _fmt_issues(issues: list[dict]) -> str:
    return "\n".join(f"- [{i['target']}] {i['issue']} → {i['suggestion']}" for i in issues)


class Outliner(Agent):
    name = "outliner"
    schema = OUTLINE_SCHEMA

    def __init__(self, llm, bank: EvidenceBank, *, trace=None):
        super().__init__(llm, trace=trace)
        self.bank = bank
        self.system = ("You plan ML papers. Produce an outline for a short paper from the evidence "
                       "given: a one-sentence question, the nugget, specific contributions, and the "
                       "sections (each with its purpose and the evidence ids it will use). Include "
                       "an Abstract section first and a Limitations section. Use only the evidence "
                       "ids given.\n\n" + load_skill("narrative", "sections"))

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        errs = []
        if not output.get("question", "").strip() or not output.get("nugget", "").strip():
            errs.append("question and nugget must be non-empty one-sentence statements")
        secs = output.get("sections", [])
        heads = [s["heading"].strip() for s in secs]
        if len(secs) < 3:
            errs.append("plan at least three sections")
        if len({h.lower() for h in heads}) != len(heads):
            errs.append("section headings must be distinct")
        if not any("abstract" in h.lower() for h in heads):
            errs.append("include an Abstract section")
        if not any("limitation" in h.lower() for h in heads):
            errs.append("include a Limitations section")
        unknown = self.bank.unknown(i for s in secs for i in s.get("evidence", []))
        if unknown:
            errs.append(f"unknown evidence ids {unknown}; use only the run ids, paper ids, or `baseline`")
        if not output.get("contributions"):
            errs.append("list at least one contribution")
        return errs

    def build_prompt(self, ctx: dict) -> str:
        parts = ["Evidence:", self.bank.to_prompt_block()]
        if ctx.get("review"):
            parts += ["", "Literature review (gaps and themes to position against):", ctx["review"]]
        if ctx.get("previous"):
            parts += ["", "Your previous outline:", json.dumps(ctx["previous"], ensure_ascii=False)]
        if ctx.get("issues"):
            parts += ["", "The discussion panel raised these issues — address every one:", _fmt_issues(ctx["issues"])]
        if ctx.get("feedback"):
            parts += ["", "The human author asked for these changes — follow them:", ctx["feedback"]]
        parts += ["", "Write the outline: title, question, nugget, contributions (each naming a specific "
                  "result), and sections with purpose + evidence ids. Abstract first; include Limitations."]
        return "\n".join(parts)


class Panelist(Agent):
    schema = PANEL_SCHEMA

    def __init__(self, llm, role: str, *, trace=None):
        super().__init__(llm, trace=trace)
        self.role = role
        self.name = f"panel:{role}"
        self.system = (PERSONAS.get(role, f"You are the {role} on a paper-planning panel.")
                       + " Vote accept if the plan is sound as it stands; vote revise and list concrete, "
                         "actionable issues (target + issue + suggestion) otherwise. Do not request "
                         "experiments that were not run; work with the evidence given.\n\n"
                       + load_skill("review"))

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        if output.get("verdict") == "revise" and not output.get("issues"):
            return ["a revise vote must list at least one issue"]
        return []

    def build_prompt(self, ctx: dict) -> str:
        return (f"Stage: {ctx['stage']}.\n\nEvidence available:\n{ctx['evidence']}\n\n"
                f"Proposal under discussion:\n{ctx['artifact']}\n\n"
                f"As the {self.role}: accept, or revise with issues.")

    def critique(self, stage: str, artifact: str, evidence: str) -> dict:
        return self.run({"stage": stage, "artifact": artifact, "evidence": evidence})


class KeyPointer(Agent):
    name = "key_pointer"
    schema = KEYPOINTS_SCHEMA

    def __init__(self, llm, bank: EvidenceBank, *, trace=None):
        super().__init__(llm, trace=trace)
        self.bank = bank
        self.system = ("You turn a paper outline into key points: for each section, the ordered "
                       "claims it will make, each with the evidence ids that back it and its role. "
                       "Claims are specific sentences a drafter can expand, with the actual numbers "
                       "from the evidence. Method/result/comparison claims must cite evidence. Use "
                       "the outline's section headings exactly.\n\n" + load_skill("narrative", "sections"))

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        errs = []
        want = [s["heading"] for s in ctx["outline"]["sections"]]
        got = [s["heading"] for s in output.get("sections", [])]
        if [h.lower() for h in got] != [h.lower() for h in want]:
            errs.append(f"sections must match the outline exactly, in order: {want}")
        for s in output.get("sections", []):
            if not s.get("points"):
                errs.append(f"section {s['heading']!r} needs at least one key point")
            for i, p in enumerate(s.get("points", []), 1):
                unknown = self.bank.unknown(p.get("evidence", []))
                if unknown:
                    errs.append(f"{s['heading']} point {i}: unknown evidence ids {unknown}")
                if p.get("role") in _ROLES_NEEDING_EVIDENCE and not p.get("evidence"):
                    errs.append(f"{s['heading']} point {i}: a {p['role']} claim must cite evidence")
                bad = self.bank.ungrounded(p.get("claim", ""))
                if bad:
                    errs.append(f"{s['heading']} point {i}: numbers not in the evidence: {bad}")
        return errs

    def build_prompt(self, ctx: dict) -> str:
        parts = ["Outline:", render_outline_block(ctx["outline"]), "", "Evidence:", self.bank.to_prompt_block()]
        if ctx.get("previous"):
            parts += ["", "Your previous key points:", json.dumps(ctx["previous"], ensure_ascii=False)]
        if ctx.get("issues"):
            parts += ["", "The discussion panel raised these issues — address every one:", _fmt_issues(ctx["issues"])]
        if ctx.get("feedback"):
            parts += ["", "The human author asked for these changes — follow them:", ctx["feedback"]]
        parts += ["", f"Write the key points for every section (roles: {POINT_ROLES}). Every number "
                  "must appear in the evidence exactly."]
        return "\n".join(parts)


class SectionDrafter(Agent):
    name = "section_drafter"
    schema = SECTION_SCHEMA

    def __init__(self, llm, bank: EvidenceBank, *, trace=None):
        super().__init__(llm, trace=trace)
        self.bank = bank
        self.system = ("You write one section of an ML paper from agreed key points. Cover every "
                       "point, in order, as flowing paragraphs (no bullet lists), citing evidence "
                       "ids inline as [exp_0005] or [arXiv:...]. Use only numbers that appear in the "
                       "key points or evidence. Do not add claims that are not in the points.\n\n"
                       + load_skill("style", "sections"))

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        body = output.get("body", "")
        errs = []
        if not body.strip():
            errs.append("empty section")
        errs += lint_prose(body)
        bad = self.bank.ungrounded(body)
        if bad:
            errs.append(f"numbers not in the evidence: {bad}")
        unknown = self.bank.unknown(_ID_RX.findall(body))
        if unknown:
            errs.append(f"unknown citation ids: {unknown}")
        return errs

    def build_prompt(self, ctx: dict) -> str:
        o = ctx["outline"]
        pts = "\n".join(f"{i}. ({p['role']}) {p['claim']}  evidence: {p['evidence'] or '—'}"
                        for i, p in enumerate(ctx["points"], 1))
        ev_ids = sorted({e for p in ctx["points"] for e in p["evidence"]})
        ev = "\n".join("- " + self.bank.describe(e) for e in ev_ids)
        parts = [f"Paper: {o['title']}", f"Question: {o['question']}", f"Nugget: {o['nugget']}",
                 f"Sections (in order): {[s['heading'] for s in o['sections']]}", "",
                 f"Section to write: {ctx['heading']}", f"Purpose: {ctx['purpose']}", "",
                 "Key points to cover, in order:", pts, "", "Evidence details:", ev or "—"]
        if ctx.get("so_far"):
            parts += ["", "Sections already written (first sentences, for continuity):", ctx["so_far"]]
        parts += ["", "Write the section body as prose. Abstract: five sentences, no citations needed. "
                  "Other sections: one paragraph per one or two points, each paragraph opening with "
                  "its message."]
        return "\n".join(parts)


class PaperReviewer(Agent):
    schema = REVIEW_SCHEMA

    def __init__(self, llm, persona: str, *, trace=None):
        super().__init__(llm, trace=trace)
        self.persona = persona
        self.name = f"reviewer:{persona}"
        self.system = (REVIEWER_PERSONAS.get(persona, f"You are a {persona} reviewer.")
                       + " Review the draft on quality, clarity, significance and originality; run "
                         "the self-consistency audit; score 1-10 honestly.\n\n" + load_skill("review"))

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        s = output.get("score")
        return [] if isinstance(s, (int, float)) and 1 <= s <= 10 else ["score must be in 1-10"]

    def build_prompt(self, ctx: dict) -> str:
        return (f"Draft:\n\n{ctx['paper']}\n\nEvidence the draft was written from:\n{ctx['evidence']}\n\n"
                "Give summary, specific strengths, specific weaknesses (section + sentence + what would "
                "fix it), questions for the authors, score (1-10), recommendation.")


# ---------------------------------------------------------------------------------------------
# Discussion loop — deterministic control over proposer ⇄ panel
# ---------------------------------------------------------------------------------------------
def discuss(propose, panel: list[Panelist], stage: str, render, evidence: str,
            *, max_rounds: int = 2) -> tuple[dict, list[dict]]:
    """propose(issues|None) -> artifact; each panelist critiques render(artifact); the proposer
    revises against the union of issues until every panelist accepts or `max_rounds` revisions
    have been made. Returns (final artifact, transcript)."""
    transcript: list[dict] = []
    artifact = propose(None)
    for rnd in range(max_rounds + 1):
        reviews = [{"role": p.role, **p.critique(stage, render(artifact), evidence)} for p in panel]
        issues = [i for r in reviews if r["verdict"] == "revise" for i in r["issues"]]
        transcript.append({"round": rnd + 1, "artifact": artifact, "reviews": reviews,
                           "accepted": not issues})
        if not issues or rnd == max_rounds:
            break
        artifact = propose(issues)
    return artifact, transcript


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------
def render_outline_block(o: dict) -> str:
    lines = [f"Title: {o['title']}", f"Question: {o['question']}", f"Nugget: {o['nugget']}",
             "Contributions:"] + [f"- {c}" for c in o["contributions"]] + ["Sections:"]
    lines += [f"{i}. {s['heading']} — {s['purpose']} [evidence: {', '.join(s['evidence']) or '—'}]"
              for i, s in enumerate(o["sections"], 1)]
    return "\n".join(lines)


def render_keypoints_block(kp: dict) -> str:
    out = []
    for s in kp["sections"]:
        out.append(f"## {s['heading']}")
        out += [f"- ({p['role']}) {p['claim']}  [{', '.join(p['evidence']) or '—'}]" for p in s["points"]]
    return "\n".join(out)


def _render_transcript(transcript: list[dict]) -> list[str]:
    out = ["## Discussion", ""]
    for t in transcript:
        status = "✅ accepted by the whole panel" if t["accepted"] else "🔁 revised"
        out.append(f"### Round {t['round']} — {status}")
        for r in t["reviews"]:
            out.append(f"- **{r['role']}**: {r['verdict']}")
            out += [f"  - [{i['target']}] {i['issue']} → *{i['suggestion']}*" for i in r.get("issues", [])]
        out.append("")
    return out


def render_outline_md(o: dict, transcript: list[dict], feedback: list[str]) -> str:
    out = [f"# Outline — {o['title']}", "",
           f"**Question this paper answers:** {o['question']}  ", f"**Nugget:** {o['nugget']}", "",
           "**Contributions**", ""] + [f"- {c}" for c in o["contributions"]] + ["", "| # | section | purpose | evidence |", "|---|---|---|---|"]
    out += [f"| {i} | **{s['heading']}** | {s['purpose']} | {', '.join(f'`{e}`' for e in s['evidence']) or '—'} |"
            for i, s in enumerate(o["sections"], 1)]
    out.append("")
    if feedback:
        out += ["> Author feedback applied: " + " / ".join(feedback), ""]
    out += _render_transcript(transcript)
    out += ["*Proposed by the Outliner, argued by the panel, settled by you. Approve to move to key points.*", ""]
    return "\n".join(out)


def render_keypoints_md(o: dict, kp: dict, transcript: list[dict], feedback: list[str]) -> str:
    out = [f"# Key points — {o['title']}", "", f"_Question: {o['question']}_", ""]
    for s in kp["sections"]:
        out += [f"## {s['heading']}", ""]
        for i, p in enumerate(s["points"], 1):
            ev = ", ".join(f"`{e}`" for e in p["evidence"]) or "—"
            out.append(f"{i}. *{p['role']}* — {p['claim']}  \n   evidence: {ev}")
        out.append("")
    if feedback:
        out += ["> Author feedback applied: " + " / ".join(feedback), ""]
    out += _render_transcript(transcript)
    out += ["*Every evidence id is a real run, paper, or the baseline; every number was checked against "
            "the registry. Approve to draft the prose.*", ""]
    return "\n".join(out)


def render_paper_md(o: dict, sections: list[dict], review: dict | None = None) -> str:
    out = [f"# {o['title']}", ""]
    if review:
        scores = " · ".join(f"{r['persona']} {r['score']}" for r in review["reviews"])
        out += [f"> **Panel review:** {review['recommendation']} · mean score {review['mean_score']}/10 "
                f"({scores})", ""]
    for s in sections:
        out += [f"## {s['heading']}", "", s["body"], ""]
    warns = [(s["heading"], w) for s in sections for w in s.get("warnings", [])]
    if warns:
        out += ["---", "", "### Drafting notes (for the author)", ""]
        out += [f"- **{h}**: {w}" for h, w in warns] + [""]
    out += ["*Written section by section from author-approved key points; every number and citation "
            "id checked against the evidence bank; prose linted for machine-writing tells.*", ""]
    return "\n".join(out)


def render_review_md(review: dict) -> str:
    out = [f"# Review — {review['title']}", "",
           f"**Recommendation:** {review['recommendation']} · **mean score:** {review['mean_score']}/10", "",
           "| reviewer | score | recommendation |", "|---|---|---|"]
    out += [f"| {r['persona']} | {r['score']} | {r['recommendation']} |" for r in review["reviews"]]
    out += ["", "## Consensus weaknesses", ""] + [f"- {w}" for w in review["weaknesses"]]
    out += ["", "## Strengths", ""] + [f"- {s}" for s in review["strengths"]]
    out += ["", "## Questions for the authors", ""] + [f"- {q}" for q in review["questions"]]
    out += ["", "## Reviewer summaries", ""]
    out += [f"- **{r['persona']}**: {r['summary']}" for r in review["reviews"]]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------------------------
STAGES = ("new", "outlined", "keypointed", "drafted", "reviewed")


@dataclass
class WritingState:
    stage: str = "new"
    outline: dict | None = None
    outline_discussion: list = field(default_factory=list)
    key_points: dict | None = None
    keypoints_discussion: list = field(default_factory=list)
    sections: list = field(default_factory=list)      # [{heading, body, warnings}]
    review: dict | None = None
    feedback: list = field(default_factory=list)      # [{stage, text, ts}]

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "WritingState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class WritingSession:
    """outline → key points → draft → review, with the human signing off the first two."""

    STATE_FILE = "writing.json"

    def __init__(self, llm, profile: Profile, out_dir: str | Path, *, ledger_path: str | Path,
                 registry_dir: str | Path, research_dir: str | Path | None = None,
                 trace: AgentTrace | None = None, max_rounds: int = 2,
                 panel_roles=("Skeptic", "Editor", "Methodologist"),
                 reviewer_personas=tuple(REVIEWER_PERSONAS)):
        self.llm, self.profile = llm, profile
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.trace = trace or AgentTrace()
        self.max_rounds = max_rounds
        entries = list(Ledger(ledger_path).read_all())
        review, idea = None, None
        if research_dir is not None:
            rd = Path(research_dir)
            if (rd / "lit_review.json").exists():
                review = LitReview.from_dict(json.loads((rd / "lit_review.json").read_text()))
            if (rd / "handoff.json").exists():
                idea = json.loads((rd / "handoff.json").read_text()).get("idea")
        self.review_doc = review
        self.bank = EvidenceBank.build(entries, registry_dir, profile, review=review, idea=idea)
        self.outliner = Outliner(llm, self.bank, trace=self.trace)
        self.key_pointer = KeyPointer(llm, self.bank, trace=self.trace)
        self.drafter = SectionDrafter(llm, self.bank, trace=self.trace)
        self.panel = [Panelist(llm, r, trace=self.trace) for r in panel_roles]
        self.reviewers = [PaperReviewer(llm, p, trace=self.trace) for p in reviewer_personas]
        sp = self.out / self.STATE_FILE
        self.state = WritingState.from_dict(json.loads(sp.read_text())) if sp.exists() else WritingState()
        self._save()

    def _save(self) -> None:
        (self.out / self.STATE_FILE).write_text(json.dumps(self.state.to_dict(), indent=2, ensure_ascii=False))

    @property
    def stage(self) -> str:
        return self.state.stage

    def _note(self, stage: str, text: str) -> None:
        if text:
            self.state.feedback.append({"stage": stage, "text": text, "ts": time.time()})

    def _feedback(self, stage: str) -> list[str]:
        return [f["text"] for f in self.state.feedback if f["stage"] == stage]

    # ---- [1] outline ----------------------------------------------------------------------------
    def outline(self, feedback: str = "") -> dict:
        self._note("outline", feedback)
        fb = "\n".join(f"- {t}" for t in self._feedback("outline"))
        lit = self.review_doc.to_prompt_block() if self.review_doc else ""
        prev = self.state.outline

        def propose(issues):
            nonlocal prev
            prev = self.outliner.run({"review": lit, "previous": prev, "issues": issues, "feedback": fb})
            return prev

        outline, transcript = discuss(propose, self.panel, "outline", render_outline_block,
                                      self.bank.to_prompt_block(), max_rounds=self.max_rounds)
        self.state.outline, self.state.outline_discussion = outline, transcript
        self.state.key_points, self.state.keypoints_discussion = None, []
        self.state.sections, self.state.review = [], None
        self.state.stage = "outlined"
        self._save()
        (self.out / "outline.md").write_text(render_outline_md(outline, transcript, self._feedback("outline")))
        return outline

    # ---- [2] key points -------------------------------------------------------------------------
    def key_points(self, feedback: str = "") -> dict:
        if self.state.outline is None:
            raise RuntimeError("run outline() first")
        self._note("key_points", feedback)
        fb = "\n".join(f"- {t}" for t in self._feedback("key_points"))
        prev = self.state.key_points
        outline = self.state.outline

        def propose(issues):
            nonlocal prev
            prev = self.key_pointer.run({"outline": outline, "previous": prev, "issues": issues, "feedback": fb})
            return prev

        kp, transcript = discuss(propose, self.panel, "key points", render_keypoints_block,
                                 self.bank.to_prompt_block(), max_rounds=self.max_rounds)
        self.state.key_points, self.state.keypoints_discussion = kp, transcript
        self.state.sections, self.state.review = [], None
        self.state.stage = "keypointed"
        self._save()
        (self.out / "key_points.md").write_text(
            render_keypoints_md(outline, kp, transcript, self._feedback("key_points")))
        return kp

    # ---- [3] draft -------------------------------------------------------------------------------
    def draft(self) -> list[dict]:
        if self.state.key_points is None:
            raise RuntimeError("run key_points() first")
        o, kp = self.state.outline, self.state.key_points
        purposes = {s["heading"].lower(): s["purpose"] for s in o["sections"]}
        sections: list[dict] = []
        for s in kp["sections"]:
            so_far = "\n".join(f"- {d['heading']}: {_sentences(d['body'])[0] if d['body'].strip() else ''}"
                               for d in sections)
            ctx = {"outline": o, "heading": s["heading"], "purpose": purposes.get(s["heading"].lower(), ""),
                   "points": s["points"], "so_far": so_far}
            warnings: list[str] = []
            try:
                body = self.drafter.run(ctx)["body"]
            except AgentError as e:                      # keep the best attempt, surface the problems
                body = e.last_output.get("body", "")
                warnings += [f"unresolved after retries: {err}" for err in e.errors]
            warnings += prose_warnings(body)
            for i, p in enumerate(s["points"], 1):
                if containment(p["claim"], body) < 0.4:
                    warnings.append(f"key point {i} may not be covered: \"{p['claim'][:70]}…\"")
            sections.append({"heading": s["heading"], "body": body, "warnings": warnings})
        self.state.sections, self.state.review = sections, None
        self.state.stage = "drafted"
        self._save()
        (self.out / "paper.md").write_text(render_paper_md(o, sections))
        return sections

    # ---- [4] review ------------------------------------------------------------------------------
    def review(self) -> dict:
        if not self.state.sections:
            raise RuntimeError("run draft() first")
        o = self.state.outline
        paper = render_paper_md(o, self.state.sections)
        ev = self.bank.to_prompt_block()
        reviews = [{"persona": r.persona, **r.run({"paper": paper, "evidence": ev})} for r in self.reviewers]
        seen, weaknesses, strengths, questions = set(), [], [], []
        for r in reviews:
            for k, bucket in (("weaknesses", weaknesses), ("strengths", strengths), ("questions", questions)):
                for item in r[k]:
                    key = " ".join(item.lower().split())
                    if key not in seen:
                        seen.add(key)
                        bucket.append(item)
        order = ["reject", "weak_reject", "weak_accept", "accept"]
        recs = sorted((r["recommendation"] for r in reviews), key=order.index)
        result = {"title": o["title"], "mean_score": round(mean(r["score"] for r in reviews), 1),
                  "recommendation": recs[len(recs) // 2],          # the median reviewer's call
                  "reviews": reviews, "weaknesses": weaknesses, "strengths": strengths, "questions": questions}
        self.state.review = result
        self.state.stage = "reviewed"
        self._save()
        (self.out / "paper.md").write_text(render_paper_md(o, self.state.sections, result))
        (self.out / "review.md").write_text(render_review_md(result))
        return result

    # ---- interactive driver ----------------------------------------------------------------------
    def interactive(self, *, input_fn=input, print_fn=print) -> dict | None:
        say = print_fn
        if self.stage == "new":
            say("[1/4] outlining — the panel will argue over it ...")
            self.outline()
        while self.stage == "outlined":
            o = self.state.outline
            rounds = len(self.state.outline_discussion)
            say(f"[1/4] outline → {self.out / 'outline.md'}  ({len(o['sections'])} sections · "
                f"{rounds} discussion round(s) · {'panel accepted' if self.state.outline_discussion[-1]['accepted'] else 'round cap hit'})")
            say(f"      question: {o['question']}")
            say(f"      nugget:   {o['nugget']}")
            cmd, arg = _parse(input_fn("      (a)pprove · (r)efine <feedback> · (q)uit > "))
            if cmd == "a":
                say("[2/4] key points — the panel will argue over them ...")
                self.key_points()
            elif cmd == "r" and arg:
                self.outline(feedback=arg)
            elif cmd == "q":
                return None
            else:
                say("      type a, r <what to change>, or q")
        while self.stage == "keypointed":
            kp = self.state.key_points
            n = sum(len(s["points"]) for s in kp["sections"])
            say(f"[2/4] key points → {self.out / 'key_points.md'}  ({n} points · "
                f"{len(self.state.keypoints_discussion)} discussion round(s))")
            cmd, arg = _parse(input_fn("      (a)pprove · (r)efine <feedback> · (q)uit > "))
            if cmd == "a":
                say("[3/4] drafting section by section ...")
                self.draft()
            elif cmd == "r" and arg:
                self.key_points(feedback=arg)
            elif cmd == "q":
                return None
            else:
                say("      type a, r <what to change>, or q")
        if self.stage == "drafted":
            warns = sum(len(s["warnings"]) for s in self.state.sections)
            say(f"[3/4] draft → {self.out / 'paper.md'}  ({len(self.state.sections)} sections · {warns} drafting note(s))")
            say("[4/4] review panel ...")
            self.review()
        r = self.state.review
        say(f"[4/4] review → {self.out / 'review.md'}  {r['recommendation']} · {r['mean_score']}/10")
        return r


def _parse(line: str) -> tuple[str, str]:
    line = (line or "").strip()
    if not line:
        return "", ""
    head, _, rest = line.partition(" ")
    return head[:1].lower(), rest.strip()


def _main(argv: list[str]) -> int:
    import argparse
    import os

    from scholarloop.core.llm import AnthropicLLM
    from scholarloop.core.profile import load_profile

    ap = argparse.ArgumentParser(prog="scholarloop.writing",
                                 description="outline → key points → draft → review, with author sign-off")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--ledger", required=True, help="experiments.jsonl of the campaign to write up")
    ap.add_argument("--registry", required=True, help="registry dir with the per-run measurements")
    ap.add_argument("--out", required=True, help="writing directory (resumes if writing.json exists)")
    ap.add_argument("--research", help="a ResearchSession dir (lit_review.json + handoff.json) to cite from")
    ap.add_argument("--model", default=os.environ.get("SCHOLARLOOP_MODEL", "claude-opus-4-8"))
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--feedback", help="non-interactive: feedback for the CURRENT stage, then stop")
    ap.add_argument("--approve", action="store_true", help="non-interactive: advance one stage")
    ap.add_argument("--auto", action="store_true", help="non-interactive: run every stage without sign-off")
    a = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 1
    s = WritingSession(AnthropicLLM(model=a.model), load_profile(a.profile), a.out, ledger_path=a.ledger,
                       registry_dir=a.registry, research_dir=a.research, max_rounds=a.max_rounds)
    if not any([a.feedback, a.approve, a.auto]):
        return 0 if s.interactive() else 1
    if a.feedback:
        # Feedback lands on the last stage the author signs off: the outline while it is being
        # settled, the key points afterwards. Past the draft it re-derives points → draft → review
        # and keeps the agreed outline; it never silently throws the whole session away.
        target = s.outline if s.stage in ("new", "outlined") else s.key_points
        target(feedback=a.feedback)
        print(f"stage {s.stage} → {s.out}")
        return 0
    steps = {"new": s.outline, "outlined": s.key_points, "keypointed": s.draft, "drafted": s.review}
    while s.stage in steps:
        steps[s.stage]()
        print(f"stage {s.stage} → {s.out}")
        if not a.auto:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
