"""The minimal orchestration loop (DESIGN §3) — one ReAct-light iteration per step.

step():
  1. read the ledger,
  2. Reasoner proposes the next experiment (reason → act under search-space constraints),
  3. ENFORCE admissibility — a proposal in a ruled-out region or already tried is skipped,
     never run (this is the search-space limit binding),
  4. run it under the fidelity budget,
  5. the runner records the measured result + the prediction's calibration_error (Reflect).

This is the first point where the system actually researches by itself: it invents a
literature-grounded config, edits, validates, and remembers — with no human in the loop. It
runs end-to-end against MockLLM (no API, no GPU), so the whole cycle is unit-testable.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scholarloop.advisor import Advisor
from scholarloop.agent import AgentTrace
from scholarloop.debate import DebatePanel
from scholarloop.director import Director
from scholarloop.ledger import Ledger, LedgerEntry
from scholarloop.litscout import LitScout
from scholarloop.llm import LLMClient
from scholarloop.profile import Profile
from scholarloop.reasoner import Reasoner
from scholarloop.governor import Governor, cost_of
from scholarloop.reasoning import confidence_bound
from scholarloop.reflector import Reflector
from scholarloop.skills import SkillLibrary


class Orchestrator:
    def __init__(self, llm: LLMClient, profile: Profile, *,
                 lit_scout: LitScout | None = None, topic: str | None = None,
                 debate_panel: DebatePanel | None = None,
                 reflector: Reflector | None = None,
                 skill_library: SkillLibrary | None = None,
                 advisor: Advisor | None = None,
                 director: Director | None = None,
                 max_refines: int = 2,
                 promote_z: float = 1.0,   # how many SEMs of margin a verify result needs to reach full
                 smoke_slack: float = 0.25,  # smoke is single-seed & noisy: promote within this band of the gate
                 trace: AgentTrace | None = None,
                 ledger_path: str | Path = "ledger.jsonl",
                 registry_dir: str | Path = "registry"):
        self.profile = profile
        self.llm = llm                           # shared client; usage accumulates here for the Governor
        self.trace = trace or AgentTrace()       # one shared trace: every agent call is logged
        self.reasoner = Reasoner(llm, profile, trace=self.trace)
        self.lit_scout = lit_scout
        self.debate_panel = debate_panel
        self.reflector = reflector
        self.skill_library = skill_library
        self.advisor = advisor
        self.director = director
        self.max_refines = max_refines
        self.promote_z = promote_z
        self.smoke_slack = smoke_slack
        # one shared trace across EVERY agent — full auditability of the loop
        for agent in (lit_scout, reflector, advisor, director):
            if agent is not None:
                agent.trace = self.trace
        if debate_panel is not None:
            for critic in debate_panel.critics:
                critic.trace = self.trace
        self.topic = topic or profile.name
        self.ledger_path = Path(ledger_path)
        self.registry_dir = Path(registry_dir)
        self._lit: tuple[str, list[str]] | None = None   # cached (lit_context, lit_priors)
        self.guidance = ""               # Director direction / Advisor rationale for the next step
        self.last_advice: dict | None = None
        self._refine_count = 0
        self._directed = False

    def _literature(self) -> tuple[str, list[str]]:
        """Fetch literature grounding once (cached). Empty when no Lit Scout is configured."""
        if self.lit_scout is None:
            return "", []
        if self._lit is None:
            self.lit_scout.trace = self.trace
            self._lit = self.lit_scout.scout(self.topic)
        return self._lit

    @staticmethod
    def _max_suffix(entries: list[LedgerEntry]) -> int:
        # the largest exp_NNNN suffix seen — robust to resumed/gapped ledgers (not a count)
        nums = [int(e.id.split("_", 1)[1]) for e in entries
                if e.id.startswith("exp_") and e.id.split("_", 1)[1].isdigit()]
        return max(nums, default=0)

    def _next_id(self, entries: list[LedgerEntry]) -> str:
        return f"exp_{self._max_suffix(entries) + 1:04d}"

    def _best_of(self, entries: list[LedgerEntry]):
        """The best-scoring entry in a batch by the metric direction (None scores ignored)."""
        scored = [e for e in entries if e.primary_score() is not None]
        if not scored:
            return None                                 # all killed/unscored — no result to reflect on
        return min(scored, key=lambda e: e.primary_score()) if self.profile.metric.direction == "minimize" \
            else max(scored, key=lambda e: e.primary_score())

    def _parent(self, entries: list[LedgerEntry]):
        """The frontier to iterate from: the current best kept entry, else the last run."""
        best = Ledger(self.ledger_path).best(self.profile.name, self.profile.metric.direction)
        if best is not None:
            return best
        return entries[-1] if entries else None

    def _propose(self, entries: list[LedgerEntry], *, lit_context: str = "", skills: str = "",
                 lit_priors: list[str] | None = None):
        """Propose → admissibility gate → debate gate. Returns (proposal, reasoning) or None."""
        import time

        if not lit_context and lit_priors is None:
            lit_context, lit_priors = self._literature()   # fall back to the Lit Scout
        if not skills and self.skill_library is not None:
            skills = self.skill_library.render(time.time())   # inject accumulated intuition
        proposal = self.reasoner.propose(entries, lit_context=lit_context, skills=skills,
                                         guidance=self.guidance, lit_priors=lit_priors)
        if not proposal.admissible:
            print(f"orchestrator: skipping inadmissible proposal {proposal.config} "
                  f"({proposal.violations})", file=sys.stderr)
            return None
        reasoning = proposal.reasoning_blob()
        if self.debate_panel is not None:                  # diverse-perspective gate before GPU
            verdict = self.debate_panel.review(proposal)
            reasoning["debate"] = verdict.to_dict()
            if verdict.decision == "reject":
                print(f"orchestrator: panel rejected {proposal.config} ({verdict.concerns})",
                      file=sys.stderr)
                return None
        return proposal, reasoning

    def _run_tier(self, proposal, reasoning, fidelity, exp_id, parent_id, parent_score, predicted_delta):
        from scholarloop.runner import run_experiment   # imported lazily (heavy deps)
        return run_experiment(
            self.profile, fidelity, exp_id, proposal.hypothesis,
            config_override=proposal.config, edits=proposal.edits, reasoning=reasoning,
            predicted_delta=predicted_delta, parent=parent_id, parent_score=parent_score,
            ledger_path=self.ledger_path, registry_dir=self.registry_dir)

    def _post_run(self, entry, frontier, entries):
        if self.reflector is not None and self.skill_library is not None:   # Reflect → skill library
            skill = self.reflector.reflect(entry, frontier)
            if skill is not None:
                self.skill_library.add(skill)
        if self.advisor is not None:                       # post-run loop control
            recent = [e.verdict for e in entries[-5:]]
            self.last_advice = self.advisor.advise(entry, frontier, recent_verdicts=recent)
            self._apply_advice(entries)

    def step(self, *, fidelity: str = "smoke", lit_context: str = "", skills: str = "",
             lit_priors: list[str] | None = None) -> LedgerEntry | None:
        """Run one autonomous iteration at a single fidelity. Returns the entry, or None if skipped."""
        entries = list(Ledger(self.ledger_path).read_all())
        prepared = self._propose(entries, lit_context=lit_context, skills=skills, lit_priors=lit_priors)
        if prepared is None:
            return None
        proposal, reasoning = prepared
        parent = self._parent(entries)
        entry = self._run_tier(proposal, reasoning, fidelity, self._next_id(entries),
                               parent.id if parent else None,
                               parent.primary_score() if parent else None,
                               proposal.predicted_delta)
        self._post_run(entry, parent, entries)
        return entry

    def _climb_from(self, smoke_entry, proposal, reasoning, tiers, gate, known) -> list[LedgerEntry]:
        """Climb an already-smoked idea up the remaining tiers (verify→full), chaining via `parent`
        and stopping the moment a tier fails its promotion gate. Returns the entries produced."""
        produced: list[LedgerEntry] = []
        prev = smoke_entry
        for fidelity in tiers:
            entry = self._run_tier(proposal, reasoning, fidelity,
                                   self._next_id(known + produced), prev.id, prev.primary_score(), None)
            produced.append(entry)
            if not self._promote(entry, fidelity, gate):
                break
            prev = entry
        return produced

    def funnel_step(self, *, tiers=("smoke", "verify", "full"), lit_context: str = "",
                    skills: str = "", lit_priors: list[str] | None = None) -> list[LedgerEntry]:
        """Funnel one idea up the fidelity ladder (DESIGN §3): each tier must clear a promotion
        gate to advance, so cheap smoke runs kill most ideas before any expensive full run.
        The tiers chain via `parent`; calibration is measured against the frontier. Returns the
        chain of entries produced (1-3)."""
        entries = list(Ledger(self.ledger_path).read_all())
        prepared = self._propose(entries, lit_context=lit_context, skills=skills, lit_priors=lit_priors)
        if prepared is None:
            return []
        proposal, reasoning = prepared
        frontier = self._parent(entries)
        frontier_score = frontier.primary_score() if frontier else None
        gate = self._gate_score(frontier_score)        # must beat the frontier (or baseline) to climb
        smoke = self._run_tier(proposal, reasoning, tiers[0], self._next_id(entries),
                               frontier.id if frontier else None, frontier_score, proposal.predicted_delta)
        produced = [smoke]
        if self._promote(smoke, tiers[0], gate):
            produced += self._climb_from(smoke, proposal, reasoning, tiers[1:], gate, entries + produced)
        self._post_run(produced[-1], frontier, entries + produced)
        return produced

    def population_step(self, k: int = 4, *, tiers=("smoke", "verify", "full"),
                        max_workers: int = 1, lit_context: str = "", skills: str = "",
                        lit_priors: list[str] | None = None) -> list[LedgerEntry]:
        """Population funnel (parallel fan-out): propose up to `k` distinct ideas, smoke-screen them
        ALL (concurrently when max_workers>1), then climb only the smoke survivors up verify→full.
        Expensive tiers are spent on a pre-screened few, not on every idea. Returns every entry run."""
        entries = list(Ledger(self.ledger_path).read_all())
        frontier = self._parent(entries)
        frontier_score = frontier.primary_score() if frontier else None
        gate = self._gate_score(frontier_score)

        # Phase 1 — propose up to k admissible, mutually-distinct ideas.
        batch: list[tuple] = []
        seen: list[dict] = []
        for _ in range(k):
            prepared = self._propose(entries, lit_context=lit_context, skills=skills, lit_priors=lit_priors)
            if prepared is None:
                continue
            proposal, reasoning = prepared
            if any(proposal.config == c for c in seen):
                continue                                # don't smoke the same config twice in one round
            seen.append(proposal.config)
            batch.append((proposal, reasoning))
        if not batch:
            return []

        # Phase 2 — smoke every idea (ids assigned up front so parallel writers never collide).
        base = self._max_suffix(entries)
        smoke_tier = tiers[0]
        parent_id = frontier.id if frontier else None

        def smoke_one(item):
            i, (proposal, reasoning) = item
            return self._run_tier(proposal, reasoning, smoke_tier, f"exp_{base + 1 + i:04d}",
                                  parent_id, frontier_score, proposal.predicted_delta)

        items = list(enumerate(batch))
        if max_workers > 1 and len(items) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                smoked = list(ex.map(smoke_one, items))
        else:
            smoked = [smoke_one(it) for it in items]
        produced = list(smoked)

        # Phase 3 — climb only the smoke survivors, best-first, through the remaining tiers.
        survivors = [(e, batch[i][0], batch[i][1]) for i, e in enumerate(smoked)
                     if self._promote(e, smoke_tier, gate)]
        survivors.sort(key=lambda t: t[0].primary_score() if t[0].primary_score() is not None else float("inf"),
                       reverse=(self.profile.metric.direction == "maximize"))
        for smoke_entry, proposal, reasoning in survivors:
            produced += self._climb_from(smoke_entry, proposal, reasoning, tiers[1:], gate,
                                         entries + produced)

        best = self._best_of(produced)
        if best is not None:
            self._post_run(best, frontier, entries + produced)
        return produced

    def _gate_score(self, frontier_score: float | None) -> float | None:
        """The promotion threshold: must beat the current best (frontier), or the baseline if
        no frontier yet — whichever is more stringent. None means no gate (climb freely)."""
        baseline = self.profile.best_baseline()
        if frontier_score is None:
            return baseline
        if baseline is None:
            return frontier_score
        return frontier_score if self.profile.metric.is_better(frontier_score, baseline) else baseline

    def _relaxed_gate(self, gate: float) -> float:
        """The smoke screen's bar: the gate widened by `smoke_slack` (a fraction of its magnitude).
        Smoke is a single noisy seed, so it should only discard ideas that are *clearly* worse than
        the gate, not borderline ones that a multi-seed verify might confirm. Scale-free, so it works
        for both error% (gate≈5) and RMSE (gate≈60). The relative band assumes a gate magnitude away
        from 0 (true for all error/RMSE-style metrics); a gate of exactly 0 degenerates to no slack."""
        margin = abs(gate) * self.smoke_slack
        return gate + margin if self.profile.metric.direction == "minimize" else gate - margin

    def _promote(self, entry: LedgerEntry, fidelity: str, gate: float | None) -> bool:
        """Promotion gate (deterministic): does this tier earn a more expensive confirmation?
        Smoke screens with slack (high recall); verify enforces the strict significance gate
        (high precision) — the standard multi-fidelity shape: cheap stages keep candidates, expensive
        stages confirm them."""
        if fidelity == "full":
            return False                                   # full is terminal
        score = entry.primary_score()
        if score is None:                                  # killed
            return False
        if gate is None:
            return True                                    # nothing to gate on yet — climb
        if fidelity == "smoke":                            # coarse screen: only kill the clearly-worse
            return self.profile.metric.is_better(score, self._relaxed_gate(gate))
        if not self.profile.metric.is_better(score, gate):
            return False                                   # didn't clear the bar — drop it cheaply
        if fidelity == "verify":                           # statistical-significance gate (not just the mean)
            seeds = entry.metric.get("seeds") or [score]
            if len(seeds) < 2:                             # a degraded verify (seeds crashed) isn't the
                return False                               # multi-seed robustness check full deserves
            bound = confidence_bound(seeds, self.profile.metric.direction, self.promote_z)
            return self.profile.metric.is_better(bound, gate)   # promote only if robust to seed noise
        return False                                       # default-deny: only smoke/verify ever climb

    def _apply_advice(self, entries: list[LedgerEntry]) -> None:
        decision = (self.last_advice or {}).get("decision", "proceed")
        rationale = (self.last_advice or {}).get("rationale", "")
        if decision == "refine":
            self._refine_count += 1
            self.guidance = rationale
            if self._refine_count >= self.max_refines:     # bounded: stop refining a dead idea
                self._pivot(entries, rationale)            # carry the real rationale into the pivot
        elif decision == "pivot":
            self._pivot(entries, rationale)
        else:  # "proceed", or any unrecognized decision -> safe default (never an accidental pivot)
            self._refine_count, self.guidance = 0, ""

    def _pivot(self, entries: list[LedgerEntry], rationale: str) -> None:
        """Change direction: reset refines, drop cached literature, re-ground.

        With a Director this genuinely re-directs (new topic + direction). Without one the topic
        is unchanged, so we at least instruct the Reasoner to leave the exhausted region.
        """
        self._refine_count = 0
        self._lit = None
        if self.director is not None:
            d = self.director.direct(entries)
            self.topic = d.get("topic", self.topic)
            self.guidance = d.get("direction", rationale)
        else:
            self.guidance = (f"{rationale} — change approach: explore a different architecture/"
                             "optimizer outside the current focus and ruled-out regions.").strip()

    def _spent(self) -> float | None:
        """Cumulative LLM spend (USD) for the Governor, or None when the model has no known price."""
        usage = getattr(self.llm, "usage", None) or {}
        return cost_of(usage, getattr(self.llm, "model", None))

    def _round(self, *, funnel, population, fidelity, max_workers, lit_context, skills) -> list[LedgerEntry]:
        """One campaign round: a population funnel, a single-idea funnel, or one flat run."""
        if population > 1:
            return self.population_step(population, max_workers=max_workers,
                                        lit_context=lit_context, skills=skills)
        if funnel:
            return self.funnel_step(lit_context=lit_context, skills=skills)
        entry = self.step(fidelity=fidelity, lit_context=lit_context, skills=skills)
        return [entry] if entry is not None else []

    def run(self, n_steps: int | None = None, *, fidelity: str = "smoke", funnel: bool = False,
            population: int = 1, max_workers: int = 1, governor: Governor | None = None,
            lit_context: str = "", skills: str = "") -> list[LedgerEntry]:
        """Run autonomous rounds; returns the entries that actually ran.

        Bounding the campaign (at least one is required, else it's a single round):
          - `n_steps` — a fixed round count (the classic mode), and/or
          - `governor` — a `Governor` that stops on a USD budget, a round cap, or convergence
            (`dry_patience` rounds with no frontier improvement). Checked at the top of each round.

        Each round is shaped by:
          - `population>1` — a parallel population funnel (propose N, smoke all, climb survivors),
          - else `funnel=True` — one idea through smoke→verify→full,
          - else a single flat run at `fidelity`.
        """
        if self.director is not None and not self._directed:   # set the campaign direction once
            d = self.director.direct(list(Ledger(self.ledger_path).read_all()))
            self.topic, self.guidance, self._directed = d.get("topic", self.topic), \
                d.get("direction", ""), True
        direction = self.profile.metric.direction
        if governor is not None:                               # seed convergence tracking from any prior best
            fr = self._parent(list(Ledger(self.ledger_path).read_all()))
            if fr is not None:
                governor.update_frontier(fr.primary_score(), direction)
        out: list[LedgerEntry] = []
        round_idx = 0
        while True:
            if governor is not None:
                stop, why = governor.should_stop(self._spent())
                if stop:
                    print(f"governor: stopping — {why}", file=sys.stderr)
                    break
            if n_steps is not None and round_idx >= n_steps:
                break
            if n_steps is None and governor is None:               # nothing bounds the loop -> one round
                if round_idx >= 1:
                    break

            produced = self._round(funnel=funnel, population=population, fidelity=fidelity,
                                    max_workers=max_workers, lit_context=lit_context, skills=skills)
            out.extend(produced)
            round_idx += 1

            if governor is not None:
                kept = [e for e in produced if e.verdict == "kept"]
                improved = any(governor.update_frontier(e.primary_score(), direction) for e in kept)
                governor.record_round(improved)
                for alert in governor.alerts(self._spent()):
                    print(f"governor: {alert}", file=sys.stderr)
        return out
