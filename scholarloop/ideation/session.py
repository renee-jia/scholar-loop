"""Research Session — the human-in-the-loop front half of ScholarLoop.

The autonomous loop (Orchestrator) is good at *running* ideas. Deciding *which* idea deserves the
compute is where a human still adds the most value, so the session puts two checkpoints in front
of the loop and persists everything between them:

    topic ──▶ [1] literature review ──▶ lit_review.md   ◀── read it; approve, or send feedback
                     │                                       (feedback re-plans queries, re-reviews)
                     ▼
              [·] direction map ──▶ directions.md    ◀── choose where to dig, or write your own
                     │                                       (needs your steer; see direction.py)
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
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scholarloop.ideation import direction, ideator, litreview
from scholarloop.core.agent import AgentError, AgentTrace
from scholarloop.ideation.ideator import Idea, Ideator, novelty_check, tried_claims
from scholarloop.core.ledger import Ledger
from scholarloop.ideation.litreview import LitReview, LitReviewer, QueryPlanner
from scholarloop.ideation.litscout import ArxivClient, OpenAlexClient, dedup_key, retrieve
from scholarloop.core.llm import LLMClient
from scholarloop.core.profile import Profile
from scholarloop.experiment.skills import SkillLibrary
from scholarloop.ideation.metareview import MetaReviewer
from scholarloop.ideation.tournament import Tournament
from scholarloop.ideation.graph import new_graph, record_round, record_decision, context_hash, validate_brief
from scholarloop.review.contracts import atomic_json

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
    meta_review: list[dict] = field(default_factory=list)   # standing [{text, support}] across rounds
    tournament_rounds: int = 0                               # rounds logged in tournament.json
    idea_graph: dict = field(default_factory=new_graph)
    deep_ideation: bool = False
    research_brief: dict = field(default_factory=dict)
    brief_history: list[dict] = field(default_factory=list)
    review_pending: bool = False
    steer: dict = field(default_factory=dict)                # the author's directional input
    direction_map: list[dict] = field(default_factory=list)  # what the DirectionMapper laid out
    chosen_directions: list[str] = field(default_factory=list)   # dir_* ids the author chose
    direction_note: str = ""
    direction_log: list[dict] = field(default_factory=list)  # append-only steer/choice decisions
    unsteered: bool = False                                  # the author explicitly waived the steer

    def to_dict(self) -> dict:
        return {"topic": self.topic, "stage": self.stage, "review": self.review,
                "ideas": self.ideas, "selected": self.selected, "feedback": self.feedback,
                "handoff": self.handoff, "meta_review": self.meta_review,
                "tournament_rounds": self.tournament_rounds, "idea_graph": self.idea_graph,
                "deep_ideation": self.deep_ideation, "research_brief": self.research_brief,
                "brief_history": self.brief_history, "review_pending": self.review_pending,
                "steer": self.steer, "direction_map": self.direction_map,
                "chosen_directions": self.chosen_directions, "direction_note": self.direction_note,
                "direction_log": self.direction_log, "unsteered": self.unsteered}

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
                 max_papers: int = 20, per_query: int = 8, tournament: Tournament | None = None,
                 meta_reviewer: MetaReviewer | None = None, skills: SkillLibrary | None = None,
                 deep_ideation: bool = False, require_direction: bool = False):
        self.llm, self.profile = llm, profile
        # With require_direction, ideate() refuses to run until the author has given a steer and
        # chosen directions (or explicitly waived both). The CLI always sets it.
        self.require_direction = require_direction
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sources = sources if sources is not None else [OpenAlexClient(), ArxivClient()]
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.trace = trace or AgentTrace(path=self.out / "ideation.agents.jsonl")
        self.max_papers, self.per_query = max_papers, per_query
        self.planner = QueryPlanner(llm, trace=self.trace)
        self.reviewer = LitReviewer(llm, trace=self.trace)
        self.ideator = Ideator(llm, profile, trace=self.trace)
        self.mapper = direction.DirectionMapper(llm, self.ideator._setting(), trace=self.trace)
        self.tournament = tournament            # None: rank by the Ideator's self-scores
        self.meta_reviewer = meta_reviewer      # distills the tournament's rationales; needs a tournament
        self.skills = skills                    # the experiment loop's lessons, read-only here
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
        self.state.deep_ideation = self.state.deep_ideation or deep_ideation
        # Import an older session without dropping its currently visible candidates.
        if self.state.ideas and not self.state.idea_graph["rounds"]:
            imported = self.ideas
            record_round(self.state.idea_graph, imported, review=self.review_obj, profile=self.profile,
                         feedback="Imported legacy candidates; prior lineage is unavailable.")
            self.state.ideas = [i.to_dict() for i in imported]
        self._save()

    # ---- persistence ------------------------------------------------------------------------
    def _save(self) -> None:
        atomic_json(self.out / self.STATE_FILE, self.state.to_dict())
        atomic_json(self.out / "idea_graph.json", self.state.idea_graph)
        handoff_path = self.out / "handoff.json"
        if self.state.handoff is None and handoff_path.exists():
            from scholarloop.review.contracts import digest
            archive = self.out / "handoff_history"
            archive.mkdir(exist_ok=True)
            handoff_path.replace(archive / f"{digest(handoff_path.read_text())}.json")
        if self.state.stage != "ideated":
            waiting = self.state.stage == "reviewed" and self.state.direction_map and not self.state.chosen_directions
            atomic_json(self.out / "idea_pending_decisions.json", [{
                "kind": "choose_directions", "direction_ids": [d["id"] for d in self.state.direction_map],
                "choices": ["choose", "add_own", "remap", "pause"]}] if waiting else [])

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

    # ---- the author's steer and the direction map ---------------------------------------------
    @property
    def chosen(self) -> list[dict]:
        return [d for d in self.state.direction_map if d["id"] in self.state.chosen_directions]

    def _context_brief(self) -> dict:
        """The brief the idea graph hashes and the judges read. The steer and the chosen directions
        are part of it, so candidates generated under another choice cannot be selected."""
        brief = self.state.research_brief
        if not self.state.steer:
            return brief
        return {**brief, "steer": self.state.steer, "directions": self.chosen,
                "direction_note": self.state.direction_note}

    def _log_direction(self, action, *, actor, kind="human", **fields):
        if not actor.strip():
            raise ValueError("A direction decision needs an actor")
        self.state.direction_log.append({"action": action, "actor": actor, "kind": kind,
                                         **fields, "timestamp": time.time()})

    def _reset_directions(self):
        self.state.direction_map, self.state.chosen_directions, self.state.direction_note = [], [], ""

    def set_steer(self, steer, *, actor):
        """Record what the author wants to find out. A new steer invalidates the direction map and
        the candidates; an existing review is kept (send review feedback to extend the corpus)."""
        steer = direction.validate_steer(steer)
        if steer == self.state.steer:
            return
        self._log_direction("steer", actor=actor, steer=steer)
        self.state.steer, self.state.unsteered = steer, False
        self._reset_directions()
        self.state.stage = "reviewed" if self.state.review else "new"
        self.state.ideas, self.state.selected, self.state.handoff = [], None, None
        self._save()
        atomic_json(self.out / "steer.json", steer)

    def waive_steer(self, *, actor, reason, kind="human"):
        """Let ideation run with no steer. Explicit, attributed and written to the log."""
        if not reason.strip():
            raise ValueError("Waiving the steer needs a reason")
        self._log_direction("waive", actor=actor, kind=kind, reason=reason)
        self.state.unsteered = True
        self._save()

    def map_directions(self, feedback: str = "") -> list[dict]:
        """Lay out research directions from the review and the steer. Writes directions.md."""
        rev = self.review_obj
        if rev is None or self.state.review_pending:
            raise RuntimeError("run review() before map_directions()")
        if not self.state.steer:
            raise ValueError("Give a steer before mapping directions (set_steer / --steer)")
        self._note_feedback("directions", feedback)
        previous = [f"{d['name']}: {d['thesis']}" for d in self.state.direction_map]
        own = [d for d in self.state.direction_map if d["origin"] == "author"]
        mapped = self.mapper.map(rev, self.state.steer, previous=previous,
                                 feedback="\n".join(f"- {t}" for t in self._feedback_for("directions")))
        self.state.direction_map = mapped + own               # the author's own directions survive a remap
        self.state.chosen_directions, self.state.direction_note = [], ""
        self.state.ideas, self.state.selected, self.state.handoff = [], None, None
        self.state.stage = "reviewed"
        self._save()
        self._write_directions()
        return self.state.direction_map

    def choose_directions(self, picks=(), *, actor, note="", own=(), kind="human") -> list[dict]:
        """Choose where ideas will be generated: 1-based positions on the map and/or directions
        the author writes (`own`). Replaces any earlier choice and drops its candidates."""
        if not self.state.steer:
            raise ValueError("Give a steer before choosing directions")
        added = [direction.own_direction(text) for text in own]
        known = {d["id"] for d in self.state.direction_map}
        board = self.state.direction_map + [d for d in added if d["id"] not in known]
        picks = list(picks)
        if any(not isinstance(i, int) or not 1 <= i <= len(self.state.direction_map) for i in picks):
            raise ValueError(f"Choose directions by their number on the map (1..{len(self.state.direction_map)})")
        ids = list(dict.fromkeys([self.state.direction_map[i - 1]["id"] for i in picks] + [d["id"] for d in added]))
        if not ids:
            raise ValueError("Choose at least one direction, or write your own")
        self._log_direction("choose", actor=actor, kind=kind, direction_ids=ids, note=note)
        self.state.direction_map, self.state.chosen_directions, self.state.direction_note = board, ids, note
        self.state.ideas, self.state.selected, self.state.handoff = [], None, None
        self.state.stage = "reviewed"
        self._save()
        self._write_directions()
        return self.chosen

    def _write_directions(self) -> None:
        (self.out / "directions.md").write_text(direction.render_markdown(
            self.topic, self.state.steer, self.state.direction_map, self.state.chosen_directions,
            self.state.direction_note))
        atomic_json(self.out / "directions.json", {"steer": self.state.steer, "map": self.state.direction_map,
                                                   "chosen": self.state.chosen_directions,
                                                   "note": self.state.direction_note, "log": self.state.direction_log})

    def set_brief(self, brief, *, actor):
        validate_brief(brief)
        if not actor.strip():
            raise ValueError("A research brief needs an author")
        if brief == self.state.research_brief:
            return
        self.state.brief_history.append({"brief": brief, "actor": actor, "timestamp": time.time()})
        self.state.research_brief = brief
        self.state.stage = "reviewed" if self.state.review else "new"
        self.state.ideas, self.state.selected, self.state.handoff = [], None, None
        self._save()
        atomic_json(self.out / "research_brief.json", brief)

    # ---- [1] literature review ----------------------------------------------------------------
    def review(self, feedback: str = "") -> LitReview:
        """Plan queries → retrieve → review. With `feedback` (a second pass), new queries are
        planned around it, their papers are ADDED to the existing corpus, and the review is
        rewritten to address the feedback. Writes lit_review.md / lit_review.json."""
        self._note_feedback("review", feedback)
        self.state.selected, self.state.handoff = None, None
        self.state.review_pending = True
        self.state.stage = "reviewed" if self.state.review else "new"
        self._save()
        prev = self.review_obj
        previous_queries = prev.queries if prev else []
        domain = _domain_blurb(self.profile)
        if self.state.research_brief:
            domain += "\nResearch brief: " + json.dumps(self.state.research_brief, ensure_ascii=False)
        if self.state.steer:       # queries and the review's gaps are planned around the author's steer
            domain += "\n" + direction.steer_block(self.state.steer)
        queries = self.planner.plan(self.topic, domain=domain,
                                    previous=previous_queries, feedback="\n".join(self._feedback_for("review")))
        fresh = retrieve(self.sources, queries, per_query=self.per_query, max_total=self.max_papers)
        papers = list(prev.papers) if prev else []
        known = {dedup_key(p) for p in papers}               # version-insensitive, like merge_papers
        papers += [p for p in fresh if dedup_key(p) not in known]
        all_feedback = "\n".join(f"- {t}" for t in self._feedback_for("review"))
        rev = self.reviewer.review(self.topic, papers, queries=previous_queries + queries,
                                   domain=domain, feedback=all_feedback)
        self.state.review = rev.to_dict()
        self.state.review_pending = False
        self.state.stage = "reviewed"
        self.state.ideas, self.state.selected, self.state.handoff = [], None, None   # downstream is stale
        self._reset_directions()                               # a map drawn from the old review is stale too
        self._save()
        (self.out / "lit_review.md").write_text(litreview.render_markdown(rev))
        (self.out / "lit_review.json").write_text(json.dumps(rev.to_dict(), indent=2, ensure_ascii=False))
        return rev

    # ---- [2] ideas ----------------------------------------------------------------------------
    def ideate(self, n: int = 5, feedback: str = "", *, parents=(), relation="refines") -> list[Idea]:
        """Generate `n` candidate ideas from the review, novelty-check each against the corpus and
        the ledger. With `feedback`, the previous round's titles + the feedback go into the prompt.
        Writes ideas.md / ideas.json."""
        rev = self.review_obj
        if rev is None:
            raise RuntimeError("run review() before ideate()")
        if self.state.review_pending:
            raise RuntimeError("The requested literature update is incomplete; finish review() before ideation")
        if self.require_direction and not self.state.unsteered:
            if not self.state.steer:
                raise ValueError("Ideation starts from your steer: say what you want to find out "
                                 "(--steer FILE or --interest \"...\"), or waive it with --unsteered")
            if not self.chosen:
                raise ValueError("Choose the directions to generate ideas in first "
                                 "(see directions.md; --directions 1,3 or --own-direction \"...\")")
        self._note_feedback("ideas", feedback)
        self.state.selected, self.state.handoff = None, None
        self.state.stage = "ideated" if self.state.ideas else "reviewed"
        self.state.ideas = [{**i, "selected": False} for i in self.state.ideas]
        self._save()
        tried = tried_claims([e for e in Ledger(self.ledger_path).read_all() if e.purpose == "exploration"]) if self.ledger_path else []
        previous = [i.title for i in self.ideas]
        lessons = self.skills.render(query=self.topic) if self.skills is not None else ""
        standing = [r["text"] for r in self.state.meta_review]
        graph = self.state.idea_graph
        observations = graph.get("observations", [])[-20:]
        if observations:
            lessons += ("\nObserved pilot results (not confirmations or refutations; inspect failures and "
                        "alternative explanations before revising the mechanism):\n"
                        + json.dumps(observations, ensure_ascii=False))
        parent_ideas = [graph["nodes"][pid]["idea"] for pid in parents]
        context = {"tried": tried, "previous": previous,
                   "feedback": "\n".join(f"- {t}" for t in self._feedback_for("ideas")),
                   "meta_review": standing, "lessons": lessons}
        if self.state.research_brief:
            context["feedback"] += "\nAuthor's research brief: " + json.dumps(self.state.research_brief, ensure_ascii=False)
        if self.chosen:
            context.update(steer=self.state.steer, directions=self.chosen,
                           direction_note=self.state.direction_note)
        audits = []
        if self.state.deep_ideation or parents:
            from scholarloop.ideation.deep import generate
            # Retain decisions with the hypotheses they concerned, not only opaque ids.
            context["history"] = [{**d, "ideas": [graph["nodes"][pid]["idea"] for pid in d["idea_ids"]]}
                                  for d in graph["decisions"][-20:]]
            ideas, audits = generate(self, rev, n=n, context=context, parents=parent_ideas)
        else:
            ideas = self.ideator.ideate(rev, n=n, **context)
        for idea in ideas:
            idea.novelty = novelty_check(idea, rev, tried)
        if self.tournament is not None and len(ideas) > 1:
            ideas = self._rank(ideas)
        record_round(graph, ideas, review=rev, profile=self.profile, feedback=feedback,
                     parents=parents, relation=relation, audits=audits, brief=self._context_brief())
        self.state.ideas = [i.to_dict() for i in ideas]
        self.state.stage = "ideated"
        self.state.selected, self.state.handoff = None, None
        self._save()
        self._write_ideas()
        return ideas

    def _rank(self, ideas: list[Idea]) -> list[Idea]:
        """Run the tournament, store each idea's standing, and return the ideas best-first (flagged
        novelty last), so idea 1 in ideas.md is the tournament's pick. Every round's match log is
        appended to tournament.json for audit (meta-review support ids point into it)."""
        res = self.tournament.run(self.topic, ideas, evidence={
            "brief": self._context_brief(),
            "dissent": self.state.meta_review[-5:],
            "papers": [asdict(p) for p in self.review_obj.papers] if self.review_obj else []})
        for i, idea in enumerate(ideas):
            idea.tournament = res.summary(i)
        order = sorted(range(len(ideas)), key=lambda i: (-ideas[i].rank_key(), i))
        new_pos = {old: new + 1 for new, old in enumerate(order)}   # renumber for the human
        for i, idea in enumerate(ideas):
            idea.tournament["rank"] = new_pos[i]                     # the position shown, flags included
            idea.tournament["near_duplicates"] = sorted(new_pos[j - 1] for j in idea.tournament["near_duplicates"])
        matches = [{**m, "a": new_pos[m["a"] - 1], "b": new_pos[m["b"] - 1]} for m in res.matches]
        titles = [ideas[i].title for i in order]
        self.state.tournament_rounds += 1
        rnd = self.state.tournament_rounds
        log_path = self.out / "tournament.json"
        log = json.loads(log_path.read_text()) if log_path.exists() else {}
        rounds = [r for r in log.get("rounds", []) if r.get("round") != rnd]
        rounds.append({"round": rnd, "ideas": titles, "matches": matches})
        log_path.write_text(json.dumps({"rounds": rounds}, indent=2, ensure_ascii=False))
        if self.meta_reviewer is not None:       # incremental: the standing list is revised, not rewritten
            try:
                self.state.meta_review = self.meta_reviewer.update(
                    self.topic, self.state.meta_review, matches, titles, round_no=rnd)
            except AgentError as e:              # keep the ranked ideas; the standing list stays as it was
                print(f"session: meta-review skipped this round ({e}); keeping the standing list",
                      file=sys.stderr)
        return [ideas[i] for i in order]

    def _write_ideas(self) -> None:
        rendered = ideator.render_markdown(self.topic, self.ideas, feedback=self._feedback_for("ideas"),
                                           meta_review=self.state.meta_review, directions=self.chosen)
        if self.state.idea_graph.get("observations"):
            rendered += "\nPilot observations: [pilot_feedback.json](pilot_feedback.json). Hypotheses remain unresolved.\n"
        (self.out / "ideas.md").write_text(rendered)
        (self.out / "ideas.json").write_text(json.dumps(self.state.ideas, indent=2, ensure_ascii=False))
        from scholarloop.ideation.deep import portfolio
        atomic_json(self.out / "idea_portfolio.json", {"selection_rule": "strategy/mechanism diversity, then quality rank",
                                                       "shortlist": portfolio(self.ideas)})
        atomic_json(self.out / "idea_pending_decisions.json", [{"kind": "choose_research_direction",
            "context_hash": self.state.idea_graph["rounds"][-1]["context_hash"],
            "idea_ids": [i.idea_id for i in self.ideas if i.status == "candidate"],
            "choices": ["select", "reject", "refine", "merge", "revive", "pause"]}]
            if self.stage == "ideated" and self.state.idea_graph["rounds"] else [])

    def decide(self, action, idea_ids, *, actor, reason, n=3):
        """Reject in place, or generate an explicit child branch from archived parents."""
        if action not in {"reject", "refine", "merge", "revive"}:
            raise ValueError("Choose reject, refine, merge or revive")
        if action == "merge" and len(idea_ids) < 2:
            raise ValueError("Merge requires at least two parent ideas")
        if action in {"refine", "revive"} and len(idea_ids) != 1:
            raise ValueError("Refine/revive requires exactly one parent idea")
        record_decision(self.state.idea_graph, action, idea_ids, actor=actor, reason=reason)
        self._save()  # even an unsuccessful model call preserves the human's request
        if action == "reject":
            ideas = self.ideas
            for idea in ideas:
                if idea.idea_id in idea_ids:
                    idea.status, idea.selected = "rejected", False
            self.state.ideas = [i.to_dict() for i in ideas]
            self.state.selected, self.state.handoff = None, None
            self.state.stage = "ideated" if ideas else "reviewed"
            self._save()
            self._write_ideas()
            return ideas
        return self.ideate(n, feedback=reason, parents=idea_ids,
                           relation={"refine": "refines", "merge": "merges", "revive": "revisits"}[action])

    def inject(self, raw, *, actor, reason):
        """Store a human-authored candidate without asking a model to rewrite it."""
        from jsonschema import Draft7Validator
        from scholarloop.ideation.deep import DEEP_SCHEMA
        if not isinstance(raw, dict):
            raise ValueError("An injected idea must be a JSON object")
        if self.review_obj is None:
            raise ValueError("Create a literature review before injecting a candidate")
        if self.state.review_pending:
            raise ValueError("Finish the pending literature update before injecting a candidate")
        if not actor.strip() or not reason.strip():
            raise ValueError("An injected idea needs an actor and a reason")
        schema = DEEP_SCHEMA if "design" in raw else ideator.IDEAS_SCHEMA
        output = {"ideas": [raw]}
        errors = [e.message for e in Draft7Validator(schema).iter_errors(output)]
        if not errors:
            errors = self.ideator.postcheck(output, {"review": self.review_obj})
            # A human may submit an ungrounded seed, but it cannot yet be selected/executed.
            errors = [e for e in errors if "cite at least one reviewed paper" not in e]
        if errors:
            raise ValueError("Invalid human idea: " + "; ".join(errors))
        idea = Idea.from_dict({**raw, "strategy": "human",
                               "status": "candidate" if raw["sources"] else "needs_grounding"})
        idea.novelty = novelty_check(idea, self.review_obj)
        candidates = self.ideas
        rounds = self.state.idea_graph["rounds"]
        if rounds and rounds[-1]["context_hash"] != context_hash(self.review_obj, self.profile, self._context_brief()):
            candidates = []  # injecting one idea must not revalidate stale peers
        for candidate in candidates:
            candidate.selected = False
        candidates.append(idea)
        record_round(self.state.idea_graph, candidates, review=self.review_obj, profile=self.profile,
                     feedback=reason, brief=self._context_brief())
        # Exact repeats retain their stable identity and do not duplicate the visible list.
        candidates = list({i.idea_id: i for i in candidates}.values())
        record_decision(self.state.idea_graph, "inject", [idea.idea_id], actor=actor, reason=reason)
        self.state.ideas = [i.to_dict() for i in candidates]
        self.state.selected, self.state.handoff, self.state.stage = None, None, "ideated"
        self._save()
        self._write_ideas()
        return idea

    # ---- [3] selection → handoff --------------------------------------------------------------
    def select(self, index: int, *, actor="local-user", reason="Selected after inspecting the candidate list", kind="human") -> dict:
        """Pick idea `index` (1-based) and build the handoff the Orchestrator adopts."""
        ideas = self.ideas
        if self.state.review_pending:
            raise ValueError("Finish the pending literature update before choosing a direction")
        if not ideas:
            raise RuntimeError("run ideate() before select()")
        if not 1 <= index <= len(ideas):
            raise ValueError(f"pick an idea in 1..{len(ideas)}")
        if ideas[index - 1].status == "rejected":
            raise ValueError("This idea was rejected; explicitly revive/re-evaluate it first")
        if ideas[index - 1].status != "candidate":
            raise ValueError("This human seed needs literature grounding; refine it before selection")
        if self.state.idea_graph["rounds"][-1]["context_hash"] != context_hash(self.review_obj, self.profile, self._context_brief()):
            raise ValueError("Review/profile changed since ideation; generate candidates again")
        record_decision(self.state.idea_graph, "select", [ideas[index - 1].idea_id],
                         actor=actor, reason=reason, kind=kind)
        from scholarloop.review.contracts import digest
        observations = [o for o in self.state.idea_graph.get("observations", [])
                        if o["idea_id"] == ideas[index - 1].idea_id]
        self.state.idea_graph["decisions"][-1].update(
            pilot_protocol_ids=[o["protocol_id"] for o in observations],
            pilot_observation_hash=digest(observations))
        for i, idea in enumerate(ideas, 1):
            idea.selected = (i == index)
        self.state.ideas = [i.to_dict() for i in ideas]
        self.state.selected = index
        self.state.handoff = build_handoff(ideas[index - 1], self.review_obj)
        self.state.handoff["selection_decision"] = self.state.idea_graph["decisions"][-1]
        self.state.handoff["research_brief"] = self.state.research_brief
        if self.state.steer:
            picked = ideas[index - 1]
            self.state.handoff["author_direction"] = {
                "steer": self.state.steer, "note": self.state.direction_note,
                "direction": next((d for d in self.chosen if d["id"] == picked.direction), None),
                "steer_link": picked.steer_link}
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
        eligible = [i for i in range(len(ideas)) if ideas[i].status == "candidate"]
        if not eligible:
            raise ValueError("No eligible candidates (rejected or awaiting grounding); refine or revive candidates")
        best = max(eligible, key=lambda i: (ideas[i].rank_key(), -i))
        return self.select(best + 1, actor="auto-selector", reason="Highest eligible deterministic rank", kind="automated")

    @property
    def handoff(self) -> dict | None:
        return self.state.handoff

    # ---- interactive driver -------------------------------------------------------------------
    def interactive(self, *, n_ideas: int = 5, input_fn=input, print_fn=print) -> dict | None:
        """Drive the session from a terminal (or any `input_fn`, e.g. a scripted one in tests).
        Returns the handoff, or None if the human quits."""
        say = print_fn
        if self.require_direction and not self.state.steer and not self.state.unsteered:
            self.set_steer(direction.ask_steer(input_fn, say), actor="local-user")
        if self.stage == "new" or self.state.review_pending:
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
                if self.state.steer and not self.chosen and not self._choose_interactively(input_fn, say):
                    return None
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
            if self.state.idea_graph.get("observations"):
                say(f"      pilot observations → {self.out / 'pilot_feedback.md'} (hypotheses remain unresolved)")
            for i, idea in enumerate(self.ideas, 1):
                flag = "  ⚠️ " + idea.novelty.get("reason", "") if idea.flagged else ""
                t = idea.tournament
                elo = f"Elo {t['elo']:g} · " if "elo" in t else ""
                say(f"      {i}. {idea.title} [{idea.status}]  ({elo}feasibility {idea.feasibility:g} · impact {idea.impact:g}){flag}")
            cmd, arg = _parse(input_fn("      (s)elect <n> · (r)efine all <feedback> · "
                                      "(f)ork <n> <reason> · (m)erge <n,n> <reason> · "
                                      "(x) reject <n> <reason> · (q)uit > "))
            if cmd == "s" and arg.isdigit() and 1 <= int(arg) <= len(self.ideas):
                try:
                    self.select(int(arg))
                except ValueError as exc:
                    say(str(exc))
            elif cmd in {"f", "m", "x"}:
                indices, _, reason = arg.partition(" ")
                try:
                    selected = [int(x) for x in indices.split(",")]
                    if not selected or any(not 1 <= i <= len(self.ideas) for i in selected):
                        raise ValueError("Use current candidate numbers")
                    self.decide({"f": "refine", "m": "merge", "x": "reject"}[cmd],
                                [self.ideas[i - 1].idea_id for i in selected],
                                actor="local-user", reason=reason.strip(), n=n_ideas)
                except ValueError as exc:
                    say(str(exc))
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


def _choose_interactively(self, input_fn, say) -> bool:
    """The direction checkpoint of interactive(): map, show, and wait for the author's choice."""
    if not self.state.direction_map:
        say("      mapping research directions from the review and your steer ...")
        self.map_directions()
    own = []
    while True:
        say(f"      directions → {self.out / 'directions.md'}")
        for i, d in enumerate(self.state.direction_map, 1):
            tag = "yours" if d["origin"] == "author" else d["stance"]
            say(f"      {i}. [{tag}] {d['name']}: {d['thesis']}")
        for text in own:
            say(f"      +  [yours] {text}")
        cmd, arg = _parse(input_fn("      (c)hoose <n,n> [| note] · (o)wn <your direction> · "
                                  "(r)emap <feedback> · (q)uit > "))
        try:
            if cmd == "c":
                picks, _, note = arg.partition("|")
                self.choose_directions(_numbers(picks), actor="local-user", note=note.strip(), own=own)
                return True
            if cmd == "o" and arg:
                direction.own_direction(arg)
                own.append(arg)
            elif cmd == "r" and arg:
                say("      remapping with your feedback ...")
                self.map_directions(feedback=arg)
            elif cmd == "q":
                return False
            else:
                say("      type c 1,3 (c alone keeps only your own), o <direction>, r <feedback>, or q")
        except ValueError as exc:
            say("      " + str(exc))


ResearchSession._choose_interactively = _choose_interactively


def _numbers(text: str) -> list[int]:
    try:
        return [int(x) for x in text.replace(",", " ").split()]
    except ValueError:
        raise ValueError("Give direction numbers such as 1,3") from None


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
    if idea.design:
        guidance += (f" Mechanism: {idea.design['mechanism']}. Intervention: {idea.design['intervention']}. "
                     f"Minimum discriminating test: {idea.design['minimal_test']}. "
                     f"Kill criterion: {idea.design['kill_criterion']}. "
                     f"Confound controls: {'; '.join(idea.design['confound_controls'])}.")
    return {"topic": idea.title, "guidance": guidance, "lit_context": lit_context,
            "lit_priors": priors, "sources": list(idea.sources), "idea": idea.to_dict()}


# ---- CLI ---------------------------------------------------------------------------------------
def _run_cli(argv: list[str]) -> int:
    import argparse
    import os

    from scholarloop.core.llm import AnthropicLLM, model_from_env
    from scholarloop.core.call_budget import CallLimitedLLM
    from scholarloop.core.profile import load_profile

    ap = argparse.ArgumentParser(prog="scholarloop.ideation",
                                 description="literature review → ideas → human choice → handoff")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True, help="session directory (resumes if session.json exists)")
    ap.add_argument("--topic", help="required for a new session")
    ap.add_argument("--brief", help="JSON research brief: question, success_criteria, constraints, non_goals, resources, literature_scope")
    ap.add_argument("--steer", help="JSON steer: interest (required), hunches, avoid, contribution, known "
                    "(see examples/steer.json). Ideation does not start without a steer")
    ap.add_argument("--interest", help="inline steer: what you want to find out and why")
    ap.add_argument("--hunch", action="append", default=[], help="inline steer: a hunch to test (repeatable)")
    ap.add_argument("--avoid", action="append", default=[], help="inline steer: a direction to stay away from (repeatable)")
    ap.add_argument("--directions", help="choose directions by number on directions.md, e.g. 1,3")
    ap.add_argument("--own-direction", action="append", default=[], help="add and choose a direction you write (repeatable)")
    ap.add_argument("--direction-note", default="", help="a note the Ideator reads with your choice")
    ap.add_argument("--unsteered", action="store_true",
                    help="waive the steer and the direction choice; recorded with --actor and --reason")
    ap.add_argument("--model", default=model_from_env())
    ap.add_argument("--max-llm-calls", type=int, default=60,
                    help="cumulative request cap for this ideation directory, including retries")
    ap.add_argument("--ledger", help="an existing ledger.jsonl, so ideas avoid what was already tested")
    ap.add_argument("--n-ideas", type=int, default=5)
    ap.add_argument("--deep", action="store_true", help="multi-strategy designs, targeted retrieval and mechanism novelty review")
    ap.add_argument("--inject-idea", help="JSON human candidate in the Idea schema; no generation call")
    ap.add_argument("--decision", choices=["reject", "refine", "merge", "revive"])
    ap.add_argument("--idea-ids", nargs="+", default=[], help="stable idea_* ids from ideas.md / idea_graph.json")
    ap.add_argument("--actor", default="local-user")
    ap.add_argument("--reason", default="", help="required explanation for an idea decision")
    ap.add_argument("--no-tournament", action="store_true",
                    help="skip the pairwise idea tournament (and its meta-review); rank by self-scores")
    ap.add_argument("--skills", help="a skill-library dir whose lessons the Ideator reads (default: "
                    "~/.scholarloop/skills/<domain> when SCHOLARLOOP_PERSIST_SKILLS=1, else none)")
    ap.add_argument("--max-papers", type=int, default=20)
    ap.add_argument("--feedback", help="non-interactive: feedback for the CURRENT stage, then advance")
    ap.add_argument("--select", type=int, help="non-interactive: pick this idea (1-based)")
    ap.add_argument("--auto", action="store_true",
                    help="non-interactive: run review → ideas → pick the top-ranked idea")
    ap.add_argument("--stop-after", choices=["review", "directions", "ideas"],
                    help="non-interactive: produce this artifact and exit for a human to read")
    a = ap.parse_args(argv)
    if sum(bool(x) for x in (a.select, a.decision, a.inject_idea)) > 1:
        ap.error("Choose one of --select, --decision or --inject-idea")

    local_decision = bool(a.select or a.decision == "reject" or a.inject_idea) and not a.feedback
    if not local_decision and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 1
    llm, profile = (None if local_decision else AnthropicLLM(model=a.model)), load_profile(a.profile)
    if llm is not None:
        llm = CallLimitedLLM(llm, a.max_llm_calls, path=Path(a.out) / "ideation_call_budget.json")
    skills = (SkillLibrary(a.skills) if a.skills else SkillLibrary.for_domain(profile.name)
              if os.environ.get("SCHOLARLOOP_PERSIST_SKILLS") == "1" else None)
    s = ResearchSession(llm, profile, a.out, topic=a.topic, ledger_path=a.ledger,
                        max_papers=a.max_papers, skills=skills, deep_ideation=a.deep, require_direction=True,
                        tournament=None if a.no_tournament else Tournament(llm, profile),
                        meta_reviewer=None if a.no_tournament else MetaReviewer(llm))
    if a.brief:
        s.set_brief(json.loads(Path(a.brief).read_text()), actor=a.actor)
    try:
        if a.steer or a.interest:
            steer = json.loads(Path(a.steer).read_text()) if a.steer else {}
            if a.interest:
                steer["interest"] = a.interest
            for key, extra in (("hunches", a.hunch), ("avoid", a.avoid)):
                if extra:
                    steer[key] = list(steer.get(key, [])) + extra
            s.set_steer(steer, actor=a.actor)
        elif a.hunch or a.avoid:
            ap.error("--hunch/--avoid add to a steer; give --interest or --steer as well")
        if a.unsteered and not s.state.steer and not s.state.unsteered:
            s.waive_steer(actor=a.actor, reason=a.reason or "Waived on the command line")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if a.inject_idea:
        try:
            idea = s.inject(json.loads(Path(a.inject_idea).read_text()), actor=a.actor, reason=a.reason)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"human idea {idea.idea_id} ({idea.status}) → {s.out / 'ideas.md'}")
        return 0

    if a.decision:
        try:
            s.decide(a.decision, a.idea_ids, actor=a.actor, reason=a.reason, n=a.n_ideas)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"decision recorded → {s.out / 'idea_graph.json'}; candidates → {s.out / 'ideas.md'}")
        return 0
    if local_decision:
        try:
            s.select(a.select, actor=a.actor, reason=a.reason or "Selected after inspecting the candidate list")
        except (ValueError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"selected idea {s.state.selected} → {s.out / 'handoff.json'}")
        return 0

    choosing = bool(a.directions or a.own_direction)
    non_interactive = any([a.feedback, a.select, a.auto, a.stop_after, choosing])
    if not non_interactive:
        return 0 if s.interactive(n_ideas=a.n_ideas) else 1

    steered = bool(s.state.steer)
    if not steered and not s.state.unsteered and s.stage in ("new", "reviewed"):
        print("Ideation starts from your steer. Say what you want to find out with --steer FILE or "
              "--interest \"...\" (add --hunch/--avoid), run without flags to be asked, or pass "
              "--unsteered to waive it.", file=sys.stderr)
        return 2
    try:
        if a.feedback:                               # feedback applies to whatever stage we're at
            if s.stage == "new" or (s.stage == "reviewed" and not s.state.direction_map):
                s.review(feedback=a.feedback)
            elif s.stage == "reviewed":              # a map exists: the feedback redraws it
                s.map_directions(feedback=a.feedback)
            else:
                s.ideate(a.n_ideas, feedback=a.feedback)
        if s.stage == "new" or s.state.review_pending:
            s.review()
        if a.stop_after == "review":
            print(f"literature review → {s.out / 'lit_review.md'}")
            return 0
        if s.stage == "reviewed" and steered:
            if not s.state.direction_map and not (a.own_direction and not a.directions):
                s.map_directions()
            if choosing:
                s.choose_directions(_numbers(a.directions or ""), actor=a.actor, note=a.direction_note,
                                    own=a.own_direction)
            elif a.auto and not s.chosen:
                s.choose_directions(range(1, len(s.state.direction_map) + 1), actor="auto-selector",
                                    kind="automated", note="Every mapped direction, chosen without a human")
            if not s.chosen or a.stop_after == "directions":
                print(f"directions → {s.out / 'directions.md'}\n"
                      "read them, then choose: --directions 1,3  (or --own-direction \"...\", "
                      "or --feedback \"...\" for a new map)")
                return 0
        if s.stage == "reviewed":
            s.ideate(a.n_ideas)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
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


def _main(argv):
    from scholarloop.core.call_budget import CallBudgetExceeded
    try:
        return _run_cli(argv)
    except (CallBudgetExceeded, AgentError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
