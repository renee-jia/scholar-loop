"""Research Session — the human-in-the-loop front half of ScholarLoop.

The autonomous loop (Orchestrator) is good at *running* ideas. Deciding *which* idea deserves the
compute is where a human still adds the most value, so the session puts two checkpoints in front
of the loop and persists everything between them:

    topic ──▶ [1] literature review ──▶ lit_review.md   ◀── read it; approve, or send feedback
                     │                                       (feedback re-plans queries, re-reviews)
                     ▼
              [2] candidate ideas ──▶ ideas.md         ◀── pick one, or send feedback
                     │                                       (feedback re-ideates, keeps the corpus)
                     ▼
              [3] handoff.json ──▶ Orchestrator.adopt(handoff) ──▶ smoke → verify → full …

With a `Tournament`, step [2] also ranks the ideas by pairwise, position-swapped judgement
(tournament.py), so idea 1 in ideas.md is the panel's pick rather than the Ideator's self-score.

Control flow is deterministic code; the agents (QueryPlanner, LitReviewer, Ideator, Judge) are stateless
and schema-checked; all state lives in `session.json` under the output directory, so a session can
be driven interactively (`interactive()`), from the CLI across separate invocations
(`python -m scholarloop.ideation ... --feedback/--select`), or by another program.

    python -m scholarloop.ideation --profile profiles/digits-mlp.yaml \
        --topic "small MLP regularization on digits" --out runs/s1
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from scholarloop.ideation import ideator, litreview
from scholarloop.core.agent import AgentTrace
from scholarloop.ideation.ideator import Idea, Ideator, novelty_check, tried_claims
from scholarloop.core.ledger import Ledger
from scholarloop.ideation.litreview import LitReview, LitReviewer, QueryPlanner
from scholarloop.ideation.litscout import ArxivClient, OpenAlexClient, dedup_key, retrieve
from scholarloop.core.llm import LLMClient
from scholarloop.core.profile import Profile
from scholarloop.ideation.tournament import Tournament

STAGES = ("new", "reviewed", "ideated", "selected")


@dataclass
class SessionState:
    topic: str
    stage: str = "new"
    review: dict | None = None
    ideas: list[dict] = field(default_factory=list)
    selected: int | None = None                      # 1-based index into ideas
    feedback: list[dict] = field(default_factory=list)   # [{stage, text, ts}]
    handoff: dict | None = None

    def to_dict(self) -> dict:
        return {"topic": self.topic, "stage": self.stage, "review": self.review,
                "ideas": self.ideas, "selected": self.selected, "feedback": self.feedback,
                "handoff": self.handoff}

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _domain_blurb(profile: Profile) -> str:
    return (f"{profile.name} ({profile.paradigm}; datasets {list(profile.datasets) or 'n/a'}; "
            f"metric {profile.metric.name}, {profile.metric.direction})")


class ResearchSession:
    STATE_FILE = "session.json"

    def __init__(self, llm: LLMClient, profile: Profile, out_dir: str | Path, *,
                 topic: str | None = None, sources: list | None = None,
                 ledger_path: str | Path | None = None, trace: AgentTrace | None = None,
                 max_papers: int = 20, per_query: int = 8, tournament: Tournament | None = None):
        self.llm, self.profile = llm, profile
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sources = sources if sources is not None else [OpenAlexClient(), ArxivClient()]
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.trace = trace or AgentTrace()
        self.max_papers, self.per_query = max_papers, per_query
        self.planner = QueryPlanner(llm, trace=self.trace)
        self.reviewer = LitReviewer(llm, trace=self.trace)
        self.ideator = Ideator(llm, profile, trace=self.trace)
        self.tournament = tournament            # None: rank by the Ideator's self-scores
        state_path = self.out / self.STATE_FILE
        if state_path.exists():
            self.state = SessionState.from_dict(json.loads(state_path.read_text()))
            if topic and topic != self.state.topic:
                print(f"session: resuming {state_path} on its saved topic {self.state.topic!r} "
                      f"(ignoring --topic)", file=sys.stderr)
        else:
            if not topic:
                raise ValueError("a new session needs a topic")
            self.state = SessionState(topic=topic)
            self._save()

    # ---- persistence ------------------------------------------------------------------------
    def _save(self) -> None:
        (self.out / self.STATE_FILE).write_text(json.dumps(self.state.to_dict(), indent=2,
                                                            ensure_ascii=False))

    @property
    def stage(self) -> str:
        return self.state.stage

    @property
    def topic(self) -> str:
        return self.state.topic

    @property
    def review_obj(self) -> LitReview | None:
        return LitReview.from_dict(self.state.review) if self.state.review else None

    @property
    def ideas(self) -> list[Idea]:
        return [Idea.from_dict(d) for d in self.state.ideas]

    def _note_feedback(self, stage: str, text: str) -> None:
        if text:
            self.state.feedback.append({"stage": stage, "text": text, "ts": time.time()})

    def _feedback_for(self, stage: str) -> list[str]:
        return [f["text"] for f in self.state.feedback if f["stage"] == stage]

    # ---- [1] literature review ----------------------------------------------------------------
    def review(self, feedback: str = "") -> LitReview:
        """Plan queries → retrieve → review. With `feedback` (a second pass), new queries are
        planned around it, their papers are ADDED to the existing corpus, and the review is
        rewritten to address the feedback. Writes lit_review.md / lit_review.json."""
        self._note_feedback("review", feedback)
        prev = self.review_obj
        previous_queries = prev.queries if prev else []
        queries = self.planner.plan(self.topic, domain=_domain_blurb(self.profile),
                                    previous=previous_queries, feedback=feedback)
        fresh = retrieve(self.sources, queries, per_query=self.per_query, max_total=self.max_papers)
        papers = list(prev.papers) if prev else []
        known = {dedup_key(p) for p in papers}               # version-insensitive, like merge_papers
        papers += [p for p in fresh if dedup_key(p) not in known]
        all_feedback = "\n".join(f"- {t}" for t in self._feedback_for("review"))
        rev = self.reviewer.review(self.topic, papers, queries=previous_queries + queries,
                                   domain=_domain_blurb(self.profile), feedback=all_feedback)
        self.state.review = rev.to_dict()
        self.state.stage = "reviewed"
        self.state.ideas, self.state.selected, self.state.handoff = [], None, None   # downstream is stale
        self._save()
        (self.out / "lit_review.md").write_text(litreview.render_markdown(rev))
        (self.out / "lit_review.json").write_text(json.dumps(rev.to_dict(), indent=2, ensure_ascii=False))
        return rev

    # ---- [2] ideas ----------------------------------------------------------------------------
    def ideate(self, n: int = 5, feedback: str = "") -> list[Idea]:
        """Generate `n` candidate ideas from the review, novelty-check each against the corpus and
        the ledger. With `feedback`, the previous round's titles + the feedback go into the prompt.
        Writes ideas.md / ideas.json."""
        rev = self.review_obj
        if rev is None:
            raise RuntimeError("run review() before ideate()")
        self._note_feedback("ideas", feedback)
        tried = tried_claims(list(Ledger(self.ledger_path).read_all())) if self.ledger_path else []
        previous = [i.title for i in self.ideas]
        ideas = self.ideator.ideate(rev, n=n, tried=tried, previous=previous,
                                    feedback="\n".join(f"- {t}" for t in self._feedback_for("ideas")))
        for idea in ideas:
            idea.novelty = novelty_check(idea, rev, tried)
        if self.tournament is not None and len(ideas) > 1:
            ideas = self._rank(ideas)
        self.state.ideas = [i.to_dict() for i in ideas]
        self.state.stage = "ideated"
        self.state.selected, self.state.handoff = None, None
        self._save()
        self._write_ideas()
        return ideas

    def _rank(self, ideas: list[Idea]) -> list[Idea]:
        """Run the tournament, store each idea's standing, and return the ideas best-first (flagged
        novelty last), so idea 1 in ideas.md is the tournament's pick. The match log goes to
        tournament.json for audit."""
        res = self.tournament.run(self.topic, ideas)
        for i, idea in enumerate(ideas):
            idea.tournament = res.summary(i)
        order = sorted(range(len(ideas)), key=lambda i: (-ideas[i].rank_key(), i))
        new_pos = {old: new + 1 for new, old in enumerate(order)}   # renumber for the human
        for idea in ideas:
            idea.tournament["near_duplicates"] = sorted(new_pos[j - 1] for j in idea.tournament["near_duplicates"])
        matches = [{**m, "a": new_pos[m["a"] - 1], "b": new_pos[m["b"] - 1]} for m in res.matches]
        (self.out / "tournament.json").write_text(json.dumps(
            {"ideas": [ideas[i].title for i in order], "matches": matches}, indent=2, ensure_ascii=False))
        return [ideas[i] for i in order]

    def _write_ideas(self) -> None:
        (self.out / "ideas.md").write_text(
            ideator.render_markdown(self.topic, self.ideas, feedback=self._feedback_for("ideas")))
        (self.out / "ideas.json").write_text(json.dumps(self.state.ideas, indent=2, ensure_ascii=False))

    # ---- [3] selection → handoff --------------------------------------------------------------
    def select(self, index: int) -> dict:
        """Pick idea `index` (1-based) and build the handoff the Orchestrator adopts."""
        ideas = self.ideas
        if not ideas:
            raise RuntimeError("run ideate() before select()")
        if not 1 <= index <= len(ideas):
            raise ValueError(f"pick an idea in 1..{len(ideas)}")
        for i, idea in enumerate(ideas, 1):
            idea.selected = (i == index)
        self.state.ideas = [i.to_dict() for i in ideas]
        self.state.selected = index
        self.state.handoff = build_handoff(ideas[index - 1], self.review_obj)
        self.state.stage = "selected"
        self._save()
        self._write_ideas()
        (self.out / "handoff.json").write_text(json.dumps(self.state.handoff, indent=2, ensure_ascii=False))
        return self.state.handoff

    def auto_select(self) -> dict:
        """Non-interactive: the top idea by the deterministic rank (tournament Elo when one ran,
        else feasibility + impact; flagged novelty pushed down)."""
        ideas = self.ideas
        if not ideas:
            raise RuntimeError("run ideate() before auto_select()")
        best = max(range(len(ideas)), key=lambda i: (ideas[i].rank_key(), -i))
        return self.select(best + 1)

    @property
    def handoff(self) -> dict | None:
        return self.state.handoff

    # ---- interactive driver -------------------------------------------------------------------
    def interactive(self, *, n_ideas: int = 5, input_fn=input, print_fn=print) -> dict | None:
        """Drive the session from a terminal (or any `input_fn`, e.g. a scripted one in tests).
        Returns the handoff, or None if the human quits."""
        say = print_fn
        if self.stage == "new":
            say(f"[1/3] reviewing the literature on {self.topic!r} ...")
            self.review()
        while self.stage == "reviewed":
            rev = self.review_obj
            say(f"[1/3] literature review → {self.out / 'lit_review.md'}  "
                f"({len(rev.papers)} papers · {len(rev.themes)} themes · {len(rev.gaps)} gaps)")
            for i, g in enumerate(rev.gaps, 1):
                say(f"      gap {i}: {g['gap']}")
            cmd, arg = _parse(input_fn("      (a)pprove · (r)efine <feedback> · (q)uit > "))
            if cmd == "a":
                say(f"[2/3] generating {n_ideas} ideas ...")
                self.ideate(n_ideas)
            elif cmd == "r" and arg:
                say("      re-reviewing with your feedback ...")
                self.review(feedback=arg)
            elif cmd == "q":
                return None
            else:
                say("      type a, r <what to change>, or q")
        while self.stage == "ideated":
            say(f"[2/3] ideas → {self.out / 'ideas.md'}")
            for i, idea in enumerate(self.ideas, 1):
                flag = "  ⚠️ " + idea.novelty.get("reason", "") if idea.flagged else ""
                t = idea.tournament
                elo = f"Elo {t['elo']:g} · " if "elo" in t else ""
                say(f"      {i}. {idea.title}  ({elo}feasibility {idea.feasibility:g} · impact {idea.impact:g}){flag}")
            cmd, arg = _parse(input_fn("      (s)elect <n> · (r)efine <feedback> · (q)uit > "))
            if cmd == "s" and arg.isdigit() and 1 <= int(arg) <= len(self.ideas):
                self.select(int(arg))
            elif cmd == "r" and arg:
                say("      re-ideating with your feedback ...")
                self.ideate(n_ideas, feedback=arg)
            elif cmd == "q":
                return None
            else:
                say(f"      type s <1..{len(self.ideas)}>, r <what to change>, or q")
        say(f"[3/3] selected idea {self.state.selected}: {self.ideas[self.state.selected - 1].title}")
        say(f"      handoff → {self.out / 'handoff.json'}  (adopt it with Orchestrator.adopt, or "
            f"SCHOLARLOOP_HANDOFF=... examples/run_to_paper.py)")
        return self.state.handoff


def _parse(line: str) -> tuple[str, str]:
    line = (line or "").strip()
    if not line:
        return "", ""
    head, _, rest = line.partition(" ")
    return head[:1].lower(), rest.strip()


def build_handoff(idea: Idea, review: LitReview | None) -> dict:
    """What the experiment loop needs from a chosen idea: a topic, Reasoner guidance (the plan and
    the hypothesis), the literature context restricted to the idea's own sources (plus the gap it
    attacks), and the priors the search-space analysis records. Pure; no LLM."""
    notes = {n["id"]: n for n in (review.paper_notes if review else [])}
    lines = []
    for sid in idea.sources:
        p = review.paper(sid) if review else None
        n = notes.get(sid)
        title = f" {p.title}" if p else ""
        contrib = f": {n['contribution']}" if n else ""
        lines.append(f"- [{sid}]{title}{contrib}")
    lit_context = "\n".join([f"Chosen idea: {idea.title}. {idea.description}",
                             f"Gap it attacks: {idea.gap}", "Supporting papers:", *lines])
    priors = [f"{idea.title} ({sid})" for sid in idea.sources]
    guidance = (f"Test this idea: {idea.hypothesis.get('claim', idea.title)} "
                f"(expected: {idea.hypothesis.get('predicted_effect', '')}). "
                f"Experiment plan: {idea.experiment_plan}")
    return {"topic": idea.title, "guidance": guidance, "lit_context": lit_context,
            "lit_priors": priors, "sources": list(idea.sources), "idea": idea.to_dict()}


# ---- CLI ---------------------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    import argparse
    import os

    from scholarloop.core.llm import AnthropicLLM
    from scholarloop.core.profile import load_profile

    ap = argparse.ArgumentParser(prog="scholarloop.ideation",
                                 description="literature review → ideas → human choice → handoff")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True, help="session directory (resumes if session.json exists)")
    ap.add_argument("--topic", help="required for a new session")
    ap.add_argument("--model", default=os.environ.get("SCHOLARLOOP_MODEL", "claude-opus-4-8"))
    ap.add_argument("--ledger", help="an existing ledger.jsonl, so ideas avoid what was already tested")
    ap.add_argument("--n-ideas", type=int, default=5)
    ap.add_argument("--no-tournament", action="store_true",
                    help="skip the pairwise idea tournament; rank by the Ideator's self-scores")
    ap.add_argument("--max-papers", type=int, default=20)
    ap.add_argument("--feedback", help="non-interactive: feedback for the CURRENT stage, then advance")
    ap.add_argument("--select", type=int, help="non-interactive: pick this idea (1-based)")
    ap.add_argument("--auto", action="store_true",
                    help="non-interactive: run review → ideas → pick the top-ranked idea")
    ap.add_argument("--stop-after", choices=["review", "ideas"],
                    help="non-interactive: produce this artifact and exit for a human to read")
    a = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 1
    llm, profile = AnthropicLLM(model=a.model), load_profile(a.profile)
    s = ResearchSession(llm, profile, a.out, topic=a.topic, ledger_path=a.ledger,
                        max_papers=a.max_papers,
                        tournament=None if a.no_tournament else Tournament(llm, profile))

    non_interactive = any([a.feedback, a.select, a.auto, a.stop_after])
    if not non_interactive:
        return 0 if s.interactive(n_ideas=a.n_ideas) else 1

    if a.feedback:                                   # feedback applies to whatever stage we're at
        if s.stage in ("new", "reviewed"):
            s.review(feedback=a.feedback)
        else:
            s.ideate(a.n_ideas, feedback=a.feedback)
    if s.stage == "new":
        s.review()
    if a.stop_after == "review":
        print(f"literature review → {s.out / 'lit_review.md'}")
        return 0
    if s.stage == "reviewed":
        s.ideate(a.n_ideas)
    if a.stop_after == "ideas":
        print(f"ideas → {s.out / 'ideas.md'}")
        return 0
    if a.select:
        s.select(a.select)
    elif a.auto and s.stage != "selected":
        s.auto_select()
    if s.stage == "selected":
        print(f"selected idea {s.state.selected} → {s.out / 'handoff.json'}")
        return 0
    print(f"session at stage {s.stage!r}; pass --select N, --auto, or run interactively", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
