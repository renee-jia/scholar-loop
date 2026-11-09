"""Meta-review: turn the idea panel's recurring critiques into a few standing recommendations.

Each tournament leaves a log of judge rationales ("B wins: A's effect is too small to measure on
450 val images"). One critique is noise; the same critique across several matches is a pattern
the Ideator should hear before its next round. Google's AI co-scientist has a meta-review agent
for this, ShinkaEvolve a periodic meta-scratchpad (at most 5 recommendations), and ACE updates
its playbook with incremental edits, not full rewrites, so the playbook doesn't lose earlier entries.

`MetaReviewer` does the same with the harness split:
  - the agent reads this round's match rationales plus the standing recommendations and returns
    an updated list: keep, sharpen, merge or drop each old one, add new ones;
  - code enforces the contract: at most `max_recs` recommendations, each backed by evidence. A
    new recommendation must cite at least `min_support` distinct matches (that is what makes it
    *recurring*); a carried-over one cites its previous id (`prev:k`), and each `prev:k` may be
    carried once, so it cannot launder new, unsupported text. Unknown ids are rejected and the
    agent retried.
  - the stored support is a resolvable audit trail: this round's `m<k>` become `r<round>:m<k>`
    (rounds are kept in tournament.json) and a `prev:k` inherits that recommendation's support.

The session stores the list, renders it in ideas.md and feeds it to the Ideator on the next round.
"""

from __future__ import annotations

from scholarloop.core.agent import Agent

META_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "recommendations": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"text": {"type": "string"},
                           "support": {"type": "array", "items": {"type": "string"}}},
            "required": ["text", "support"]}},
    },
    "required": ["recommendations"],
}


def match_lines(matches: list[dict], titles: list[str]) -> list[str]:
    """Render a tournament's match log as cite-able lines `m<k>: ...` (k is 1-based)."""
    out = []
    for k, m in enumerate(matches, 1):
        a, b = titles[m["a"] - 1], titles[m["b"] - 1]
        if m["result"] == "void":
            out.append(f"m{k}: {a!r} vs {b!r} is void (the judge failed to answer); no evidence.")
            continue
        verdict = {"a": f"preferred {a!r}", "b": f"preferred {b!r}", "draw": "split (a draw)"}[m["result"]]
        out.append(f"m{k}: {m['judge']} judged {a!r} vs {b!r}, {verdict}. "
                   f"Shown {a!r} first: {m['rationale_a_first']} Shown {b!r} first: {m['rationale_b_first']}")
    return out


class MetaReviewer(Agent):
    name = "meta_reviewer"
    system = ("You are the meta-reviewer of a research lab's idea committee. You read how the judges "
              "compared candidate ideas and extract what they KEEP objecting to or rewarding, as "
              "short, actionable recommendations for whoever writes the next round of ideas. A "
              "recommendation states what to do differently (e.g. 'size the predicted effect "
              "against seed noise'), not a verdict on one idea. Keep the list short; merge overlaps.")
    schema = META_SCHEMA

    def __init__(self, llm, *, trace=None, max_recs: int = 5, min_support: int = 2):
        super().__init__(llm, trace=trace)
        self.max_recs, self.min_support = max_recs, min_support

    def build_prompt(self, ctx: dict) -> str:
        parts = [f"Topic: {ctx['topic']}", ""]
        if ctx["previous"]:
            parts += ["Standing recommendations from earlier rounds (cite as prev:<k> to keep one):"]
            parts += [f"prev:{k}: {r}" for k, r in enumerate(ctx["previous"], 1)]
            parts.append("")
        parts += ["This round's matches (cite as m<k>):", *ctx["lines"], "",
                  f"Return at most {self.max_recs} recommendations. Each new one must cite at least "
                  f"{self.min_support} different matches that show the pattern; a kept or revised "
                  "standing recommendation cites its prev:<k> (plus any matches that reinforce it). "
                  "Drop standing ones this round contradicts. An empty list is fine if nothing recurs."]
        return "\n".join(parts)

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        recs = output.get("recommendations", [])
        known = {f"m{k}" for k in range(1, len(ctx["lines"]) + 1)} | \
                {f"prev:{k}" for k in range(1, len(ctx["previous"]) + 1)}
        errs = []
        carried = [s for r in recs for s in set(r.get("support", [])) if s.startswith("prev:")]
        twice = sorted({s for s in carried if carried.count(s) > 1})
        if twice:
            errs.append(f"each standing recommendation may be carried once; {twice} are cited by "
                        "several (merge them, or support the new text with matches)")
        if len(recs) > self.max_recs:
            errs.append(f"at most {self.max_recs} recommendations (got {len(recs)}); merge or drop")
        for i, r in enumerate(recs, 1):
            sup = set(r.get("support", []))
            unknown = sorted(sup - known)
            if unknown:
                errs.append(f"recommendation {i}: unknown support ids {unknown}")
            matches = {s for s in sup if s.startswith("m")}
            if not any(s.startswith("prev:") for s in sup) and len(matches) < self.min_support:
                errs.append(f"recommendation {i}: a new recommendation needs >= {self.min_support} "
                            f"supporting matches (got {sorted(matches)}); it is not recurring yet")
            if not r.get("text", "").strip():
                errs.append(f"recommendation {i}: empty text")
        return errs

    def update(self, topic: str, previous: list[dict], matches: list[dict], titles: list[str],
               *, round_no: int = 1) -> list[dict]:
        """The new standing list [{text, support}] after tournament round `round_no`. `previous`
        is the standing list; support ids are stored resolved (see the module docstring). With no
        evidence this round (no matches, or fewer usable ones than a new recommendation needs and
        nothing standing) the standing list is returned without a call."""
        prev = [r if isinstance(r, dict) else {"text": r, "support": []} for r in previous]
        usable = sum(m["result"] != "void" for m in matches)
        if usable == 0 or (usable < self.min_support and not prev):
            return prev
        out = self.run({"topic": topic, "previous": [r["text"] for r in prev],
                        "lines": match_lines(matches, titles)})

        def resolve(sid: str) -> list[str]:
            if sid.startswith("prev:"):
                return list(prev[int(sid.split(":")[1]) - 1]["support"])
            return [f"r{round_no}:{sid}"]
        return [{"text": r["text"].strip(),
                 "support": sorted({x for sid in set(r["support"]) for x in resolve(sid)})}
                for r in out["recommendations"]]
