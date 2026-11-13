"""Ideation — from a literature review to candidate research ideas a human can choose between.

The `Ideator` agent reads the review (themes, gaps, paper notes), the experimental setting it must
be testable in (the profile's metric, baseline, knobs, editable surfaces), what the ledger has
already tried, and any human feedback — and proposes several *distinct* ideas. Each idea names the
gap it attacks, cites its supporting papers, states a falsifiable hypothesis, and sketches the
experiment that would test it on this engine.

Two deterministic checks sit behind the agent:
  - **citation grounding** — every cited id must be in the reviewed corpus (rejected + retried),
  - **novelty check** (`novelty_check`) — lexical proximity of each idea to (a) every reviewed
    paper and (b) every hypothesis already in the ledger. It is a *prompt for human judgement*,
    not a verdict: a high overlap means "this looks like paper X / experiment Y — are you sure?"

`render_markdown` writes the `ideas.md` the human reads at the second checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from scholarloop.core.agent import Agent
from scholarloop.core.ledger import LedgerEntry
from scholarloop.ideation.litreview import LitReview
from scholarloop.core.profile import Profile
from scholarloop.core.text import containment, jaccard

IDEAS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "ideas": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "gap": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "hypothesis": {"type": "object", "additionalProperties": False,
                                   "properties": {"claim": {"type": "string"},
                                                  "predicted_effect": {"type": "string"}},
                                   "required": ["claim", "predicted_effect"]},
                    "experiment_plan": {"type": "string"},
                    "novelty_rationale": {"type": "string"},
                    "feasibility": {"type": "number"},
                    "impact": {"type": "number"},
                    # Present when the author chose directions (direction.py): the id of the
                    # direction this idea belongs to and how it follows the author's steer.
                    "direction": {"type": "string"},
                    "steer_link": {"type": "string"},
                },
                "required": ["title", "description", "gap", "sources", "hypothesis",
                             "experiment_plan", "novelty_rationale", "feasibility", "impact"],
            },
        },
    },
    "required": ["ideas"],
}


@dataclass
class Idea:
    title: str
    description: str
    gap: str
    sources: list[str]
    hypothesis: dict                      # {claim, predicted_effect}
    experiment_plan: str
    novelty_rationale: str
    feasibility: float                    # 1-5, the agent's own estimate
    impact: float                         # 1-5, the agent's own estimate
    novelty: dict = field(default_factory=dict)   # filled by novelty_check(): {closest, overlap, flag, ...}
    tournament: dict = field(default_factory=dict)   # filled by Tournament: {elo, rank, wins, ...}
    selected: bool = False
    idea_id: str = ""
    strategy: str = "general"
    design: dict = field(default_factory=dict)
    novelty_review: dict = field(default_factory=dict)
    status: str = "candidate"
    direction: str = ""                   # id of the author-chosen direction, when one was chosen
    steer_link: str = ""                  # how the idea follows the author's steer

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("direction", "steer_link"):     # absent when unsteered, so older idea ids hold
            if not d[k]:
                del d[k]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Idea":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def flagged(self) -> bool:
        return bool(self.novelty.get("flag")) or self.novelty_review.get("status") == "overlap"

    def rank_key(self) -> float:
        """Deterministic ranking used by non-interactive selection, with a flagged novelty check
        pushed below every unflagged idea. The tournament's Elo when one ran (pairwise judgement
        by other agents); otherwise the Ideator's own feasibility + impact (a self-score)."""
        if "elo" in self.tournament:
            return float(self.tournament["elo"]) - (1e6 if self.flagged else 0.0)
        return self.feasibility + self.impact - (10.0 if self.flagged else 0.0)


class Ideator(Agent):
    name = "ideator"
    system = ("You generate research ideas for an automated ML lab from a literature review. Each "
              "idea must attack a named open gap, cite the reviewed papers it builds on (by the "
              "bracketed ids only), state a falsifiable hypothesis, and come with an experiment that "
              "can run in the given experimental setting. Prefer ideas that combine or transfer "
              "techniques in ways the reviewed papers have not tried over re-tuning what they did. "
              "Ideas must be mutually distinct. Score feasibility and impact honestly on 1-5.")
    schema = IDEAS_SCHEMA

    def __init__(self, llm, profile: Profile, *, trace=None):
        super().__init__(llm, trace=trace)
        self.profile = profile

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        ideas = output.get("ideas", [])
        ids = {p.arxiv_id for p in ctx["review"].papers}
        errs = []
        if not ideas:
            errs.append("propose at least one idea")
        for i, idea in enumerate(ideas, 1):
            if not idea.get("sources"):
                errs.append(f"idea {i}: cite at least one reviewed paper")
            unknown = sorted(s for s in idea.get("sources", []) if s not in ids)
            if unknown:
                errs.append(f"idea {i}: unknown paper ids {unknown} (cite only the bracketed ids)")
            for k in ("feasibility", "impact"):
                v = idea.get(k)
                if not isinstance(v, (int, float)) or not 1 <= v <= 5:
                    errs.append(f"idea {i}: {k} must be a number in 1-5")
        titles = [" ".join(i.get("title", "").lower().split()) for i in ideas]
        if len(set(titles)) != len(titles):
            errs.append("ideas must have distinct titles")
        chosen = [d["id"] for d in ctx.get("directions") or []]
        if chosen:
            for i, idea in enumerate(ideas, 1):
                if idea.get("direction") not in chosen:
                    errs.append(f"idea {i}: `direction` must be one of the chosen ids {chosen}")
                if not (idea.get("steer_link") or "").strip():
                    errs.append(f"idea {i}: say in `steer_link` how it follows the author's steer")
            unused = [d for d in chosen if d not in {i.get("direction") for i in ideas}]
            if unused and len(ideas) >= len(chosen):
                errs.append(f"no idea belongs to chosen direction(s) {unused}")
        return errs

    def _setting(self) -> str:
        p = self.profile
        knobs = f" Tunable knobs (name: default): {p.knobs}." if p.knobs else ""
        return (f"{p.name} ({p.paradigm}); datasets {list(p.datasets) or 'n/a'}; metric "
                f"{p.metric.name} ({p.metric.direction}); must-beat baseline {p.best_baseline()}; "
                f"the agent may edit {list(p.allowed_edits)} in the train script and set "
                f"hyperparameters.{knobs}")

    def build_prompt(self, ctx: dict) -> str:
        review: LitReview = ctx["review"]
        n = ctx.get("n", 5)
        parts = [f"Topic: {review.topic}",
                 f"Experimental setting every idea must be testable in: {self._setting()}", "",
                 "Literature review:", review.to_prompt_block()]
        if ctx.get("directions"):
            from scholarloop.ideation.direction import directions_block, steer_block
            parts += ["", "The author's steer:", steer_block(ctx["steer"]), "",
                      "Directions the author CHOSE. Generate ideas only inside these, spread "
                      "across them, and never what the author asked to avoid or already knows:",
                      directions_block(ctx["directions"])]
            if ctx.get("direction_note"):
                parts += [f"The author's note on the choice: {ctx['direction_note']}"]
        if ctx.get("tried"):
            parts += ["", "Hypotheses the lab has ALREADY tested (do not re-propose):"]
            parts += [f"- {t}" for t in ctx["tried"]]
        if ctx.get("previous"):
            parts += ["", "Ideas proposed in the previous round:"]
            parts += [f"- {t}" for t in ctx["previous"]]
        if ctx.get("meta_review"):
            parts += ["", "The idea committee's recurring critiques of earlier rounds — address them:"]
            parts += [f"- {r}" for r in ctx["meta_review"]]
        if ctx.get("lessons"):
            parts += ["", "Lessons the lab learned from its own past experiments (weight = recency x "
                      "severity) — do not propose what they warn against:", ctx["lessons"]]
        if ctx.get("feedback"):
            parts += ["", "Human feedback on the previous round — follow it:", ctx["feedback"]]
        parts += ["", f"Propose {n} distinct ideas. For each: title, description (one paragraph), the "
                  "gap it attacks, sources (bracketed ids), hypothesis {claim, predicted_effect}, an "
                  "experiment_plan concrete enough to run in the setting above (which knobs or edits, "
                  "what to compare against), a novelty_rationale (why the reviewed papers have not "
                  "done this), feasibility 1-5, impact 1-5."]
        if ctx.get("directions"):
            parts += ["Also give each idea `direction` (the id of the chosen direction it belongs "
                      "to) and `steer_link` (one sentence: which part of the author's steer it "
                      "acts on, and what the author would learn from it). An idea is a specific, "
                      "testable step inside a direction, not a restatement of the direction."]
        return "\n".join(parts)

    def ideate(self, review: LitReview, *, n: int = 5, tried: list[str] | None = None,
               previous: list[str] | None = None, feedback: str = "",
               meta_review: list[str] | None = None, lessons: str = "",
               steer: dict | None = None, directions: list[dict] | None = None,
               direction_note: str = "") -> list[Idea]:
        out = self.run({"review": review, "n": n, "tried": tried or [],
                        "previous": previous or [], "feedback": feedback,
                        "meta_review": meta_review or [], "lessons": lessons,
                        "steer": steer or {}, "directions": directions or [],
                        "direction_note": direction_note})
        return [Idea(title=i["title"], description=i["description"], gap=i["gap"],
                     sources=list(i["sources"]), hypothesis=dict(i["hypothesis"]),
                     experiment_plan=i["experiment_plan"], novelty_rationale=i["novelty_rationale"],
                     feasibility=float(i["feasibility"]), impact=float(i["impact"]),
                     direction=i.get("direction", ""), steer_link=i.get("steer_link", ""))
                for i in out["ideas"]]


def tried_claims(entries: list[LedgerEntry]) -> list[str]:
    """Distinct hypothesis claims already in the ledger (what the lab has actually tested)."""
    seen, out = set(), []
    for e in entries:
        c = " ".join(e.hypothesis.claim.split())
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def novelty_check(idea: Idea, review: LitReview, tried: list[str] | None = None, *,
                  paper_threshold: float = 0.6, tried_threshold: float = 0.7) -> dict:
    """Deterministic lexical proximity of an idea to the reviewed papers and to the ledger.

    Against papers: the fraction of the idea's own terms (title + description + claim) that appear
    in the paper (title + abstract) — asymmetric, since an abstract is much longer than an idea.
    Against the ledger: symmetric Jaccard between the idea's claim and each already-tested claim
    (both short). The closest match is reported; the idea is *flagged* past a threshold. Lexical
    overlap is a heuristic — it surfaces "this reads like paper X" for a human to judge, it does
    not decide."""
    text = f"{idea.title} {idea.description} {idea.hypothesis.get('claim', '')}"
    closest, best = None, 0.0
    for p in review.papers:
        s = containment(text, f"{p.title} {p.summary}")
        if s > best:
            closest, best = p, s
    closest_tried, best_tried = None, 0.0
    for t in (tried or []):
        s = jaccard(idea.hypothesis.get("claim") or idea.title, t)
        if s > best_tried:
            closest_tried, best_tried = t, s
    report = {
        "closest_paper": closest.arxiv_id if closest else None,
        "closest_title": closest.title if closest else None,
        "paper_overlap": round(best, 2),
        "closest_tried": closest_tried,
        "tried_overlap": round(best_tried, 2),
        "flag": best >= paper_threshold or best_tried >= tried_threshold,
    }
    reasons = []
    if best >= paper_threshold:
        reasons.append(f"reads close to {closest.arxiv_id} ({best:.0%} of its terms appear there)")
    if best_tried >= tried_threshold:
        reasons.append(f"overlaps an already-tested hypothesis: {closest_tried!r}")
    report["reason"] = "; ".join(reasons)
    return report


def render_markdown(topic: str, ideas: list[Idea], *, feedback: list[str] | None = None,
                    meta_review: list[dict] | None = None, directions: list[dict] | None = None) -> str:
    """`ideas.md` — the human-facing candidate list for the second checkpoint."""
    ranked = any("elo" in i.tournament for i in ideas)
    names = {d["id"]: d["name"] for d in directions or []}
    how = ("ranked by a pairwise Elo tournament (each pair judged twice, order swapped)" if ranked
           else "ranked by the Ideator's own feasibility + impact")
    out = [f"# Research ideas — {topic}", "",
           f"_{len(ideas)} candidates · {how} · novelty check is lexical and advisory_", ""]
    if names:
        out += ["> Generated inside the directions you chose ([directions.md](directions.md)): "
                + " / ".join(names.values()), ""]
    if feedback:
        out += ["> Feedback applied: " + " / ".join(feedback), ""]
    elo_h, elo_sep = (" Elo (W-D-L) |", "---|") if ranked else ("", "")
    out += [f"| # | idea | gap |{elo_h} feasibility | impact | novelty check |",
            f"|---|---|---|{elo_sep}---|---|---|"]
    for i, idea in enumerate(ideas, 1):
        nov = idea.novelty
        if not nov:
            n = "—"
        elif nov.get("flag"):
            n = f"⚠️ {nov.get('reason', 'flagged')}"
        else:
            n = f"ok (closest {nov.get('closest_paper')}, {nov.get('paper_overlap', 0):.0%})"
        mark = "✅ " if idea.selected else ""
        t = idea.tournament
        elo = (f" {t['elo']:g} ({t['wins']}-{t['draws']}-{t['losses']}) |" if "elo" in t
               else " — |") if ranked else ""
        out.append(f"| {i} | {mark}**{idea.title}** | {idea.gap} |{elo} {idea.feasibility:g} | "
                   f"{idea.impact:g} | {n} |")
    out.append("")
    for i, idea in enumerate(ideas, 1):
        out += [f"## {i}. {idea.title}" + ("  ✅ selected" if idea.selected else ""), "",
                idea.description, "",
                *([f"- **Direction:** {names.get(idea.direction, idea.direction)}",
                   f"- **Follows your steer:** {idea.steer_link}"] if idea.direction else []),
                f"- **Gap:** {idea.gap}",
                f"- **Hypothesis:** {idea.hypothesis.get('claim', '')} → *{idea.hypothesis.get('predicted_effect', '')}*",
                f"- **Experiment:** {idea.experiment_plan}",
                f"- **Builds on:** {', '.join(f'`{s}`' for s in idea.sources)}",
                f"- **Why novel (agent):** {idea.novelty_rationale}"]
        if idea.idea_id:
            out += [f"- **Stable ID:** `{idea.idea_id}`; strategy: {idea.strategy}; status: {idea.status}"]
        if idea.design:
            out += [f"- **Mechanism:** {idea.design['mechanism']}",
                    f"- **Intervention:** {idea.design['intervention']}",
                    f"- **Assumptions:** {'; '.join(idea.design['assumptions'])}",
                    f"- **Minimum test:** {idea.design['minimal_test']}",
                    f"- **Kill criterion:** {idea.design['kill_criterion']}",
                    f"- **Confound controls:** {'; '.join(idea.design['confound_controls'])}",
                    f"- **Resource risk:** {idea.design['resource_risk']}"]
        if idea.novelty_review:
            nr = idea.novelty_review
            out += [f"- **Mechanism novelty review ({nr['evidence_scope']}):** {nr['status']}; {nr['summary']}"]
            out += [f"  - `{c['paper_id']}`: shared mechanism: {c['shared_mechanism']}; "
                    f"difference: {c['difference']}; abstract excerpt: “{c['supporting_excerpt']}”"
                    for c in nr["comparisons"]]
            out += [f"- **Unresolved search:** {'; '.join(nr['remaining_search'])}"]
        if idea.tournament.get("near_duplicates"):
            out.append("- **Near-duplicate of:** " + ", ".join(
                f"#{j}" for j in idea.tournament["near_duplicates"]) + " (title + claim overlap)")
        if idea.novelty:
            nov = idea.novelty
            line = (f"- **Novelty check (lexical):** closest paper `{nov.get('closest_paper')}` "
                    f"({nov.get('paper_overlap', 0):.0%} term overlap)")
            if nov.get("closest_tried"):
                line += f"; closest tested claim {nov.get('tried_overlap', 0):.0%}"
            if nov.get("flag"):
                line += f" — ⚠️ {nov.get('reason')}"
            out.append(line)
        out.append("")
    if meta_review:
        out += ["## What the idea committee kept saying", "",
                "_Recurring critiques from the tournament, distilled by the meta-reviewer; the next "
                "round of ideas is written against them. Support = the matches that show it._", ""]
        out += [f"{k}. {r['text']}  _(support: {', '.join(r['support'])})_"
                for k, r in enumerate(meta_review, 1)]
        out.append("")
    out += ["*Generated by ScholarLoop's Ideator from the literature review; pick one (or send "
            "feedback) to continue to experiments.*", ""]
    return "\n".join(out)
