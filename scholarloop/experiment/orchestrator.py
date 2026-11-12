"""The minimal orchestration loop — one ReAct-light iteration per step.

step():
  1. read the ledger,
  2. Reasoner proposes the next experiment (reason → act under search-space constraints),
  3. ENFORCE admissibility — a duplicate candidate or forbidden source edit is skipped,
     while observational warnings remain revisitable,
  4. run it under the fidelity budget,
  5. the runner records the measured result + the prediction's calibration_error (Reflect).

This is the first point where the system actually researches by itself: it invents a
literature-grounded config, edits, validates, and remembers — with no human in the loop. It
runs end-to-end against MockLLM (no API, no GPU), so the whole cycle is unit-testable.

Above step(): `funnel_step` climbs one idea smoke → verify → full; `population_step(k)` screens k
ideas in parallel and climbs the survivors; `run(governor=...)` repeats rounds until the Governor
stops it (budget / round cap / convergence; `stop_reason` records why). `adopt(handoff)` starts
from a human-chosen idea (a ResearchSession handoff) instead of asking the Director.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

from scholarloop.experiment.advisor import Advisor
from scholarloop.core.agent import AgentTrace
from scholarloop.experiment.debate import DebatePanel
from scholarloop.experiment.director import Director
from scholarloop.core.ledger import Ledger, LedgerEntry
from scholarloop.core.candidates import CandidateStore, make_snapshot
from scholarloop.experiment.evidence import best_comparable, search_entries, scope, reference_score
from scholarloop.ideation.litscout import LitScout
from scholarloop.core.llm import LLMClient
from scholarloop.core.profile import Profile
from scholarloop.experiment.reasoner import Reasoner
from scholarloop.experiment.calibration import CalibrationLog, score_binary, score_delta
from scholarloop.experiment.governor import Governor, cost_of
from scholarloop.experiment.reasoning import confidence_bound
from scholarloop.experiment.reflector import Reflector
from scholarloop.experiment.skills import SkillLibrary


def _fold_scores(entry: LedgerEntry) -> dict[int, float] | None:
    """{fold: val score} for a cross-validated entry, or None (no CV, killed, or no fold record).
    Seeds that crashed are absent from both lists, so they stay aligned."""
    folds, seeds = (entry.metric or {}).get("folds"), (entry.metric or {}).get("seeds")
    if not folds or not seeds or len(folds) != len(seeds) or len(set(folds)) != len(folds):
        return None
    return dict(zip(folds, seeds))


def _fold_matched(entry: LedgerEntry, frontier: LedgerEntry | None) -> float | None:
    """The frontier's mean over the folds `entry` saw, or None when that can't be formed."""
    mine, theirs = _fold_scores(entry), _fold_scores(frontier) if frontier is not None else None
    if not mine or not theirs or not set(mine) <= set(theirs):
        return None
    return round(mean(theirs[f] for f in mine), 4)


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
                 registry_dir: str | Path = "registry",
                 calibration_path: str | Path | None = None,
                 warm_start: int = 0,    # the first N ideas are random draws from the knob ranges
                 warm_seed: int = 0):
        self.profile = profile
        self.llm = llm                           # shared client; usage accumulates here for the Governor
        self.trace = trace or AgentTrace(path=Path(ledger_path).with_name(Path(ledger_path).name + ".agents.jsonl"))       # one shared trace: every agent call is logged
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
        # Warm start: measured against random search on digits, the agent's priors steered it
        # away from the good region (§9.1.2). Its first `warm_start` ideas are drawn at random
        # from the same ranges the control arm uses, so it refines from spread-out evidence.
        self.warm_start, self.warm_seed = warm_start, warm_seed
        self._round_cache: str | None = None   # trained models shared by one idea's tiers
        self._warm = None
        self.training_budget = None
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
        self.candidates = CandidateStore(self.registry_dir)
        self._lit: tuple[str, list[str]] | None = None   # cached (lit_context, lit_priors)
        self.guidance = ""               # Director direction / Advisor rationale for the next step
        self.last_advice: dict | None = None
        self._refine_count = 0
        self._directed = False
        self.adopted: dict | None = None     # a human-chosen idea (ResearchSession handoff), if any
        self._base_guidance = ""             # standing guidance a "proceed" falls back to (the adopted plan)
        self.stop_reason: str | None = None  # why the last governed `run()` ended, for run logs
        # universal predict-then-verify across agents (B); with a path it survives a resumed campaign
        self.calibration = (CalibrationLog.open(calibration_path) if calibration_path is not None
                            else CalibrationLog())

    def adopt(self, handoff: dict) -> None:
        """Start the loop from a human-chosen idea (a `ResearchSession` handoff): its title becomes
        the topic, its plan + hypothesis the Reasoner guidance, and its own sources the literature
        context. The Director is NOT consulted for an opening direction — the human already set it;
        it is still used on a PIVOT, when the idea's direction has been exhausted."""
        self.adopted = handoff
        self.topic = handoff.get("topic", self.topic)
        self.guidance = self._base_guidance = handoff.get("guidance", "")
        self._lit = (handoff.get("lit_context", ""), list(handoff.get("lit_priors", [])))
        self._directed = True

    def _literature(self) -> tuple[str, list[str]]:
        """Literature grounding, cached: an adopted idea's own sources, else one Lit Scout pass.
        Empty when neither is available."""
        if self._lit is not None:
            return self._lit
        if self.lit_scout is None:
            return "", []
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
        """Select within the widest comparable validation scope in this batch."""
        return best_comparable(entries, self.profile)

    def _parent(self, entries: list[LedgerEntry]):
        """The frontier to iterate from: the current best kept entry, else the last run.

        With cross-validation a tier's mean covers only its own folds (smoke fold 0, verify 0-2,
        full 0-4), and folds differ in difficulty, so means from different tiers are not
        comparable: a 3-fold verify mean that skips a hard fold "beats" a 5-fold full mean. The
        frontier is then the best kept entry among those that cover the most folds."""
        entries = search_entries(entries, self.profile)
        best = best_comparable(list(Ledger(self.ledger_path).read_all()), self.profile, kept_only=True)
        if best is not None:
            return best
        return entries[-1] if entries else None

    def _gate_for(self, entry: LedgerEntry, frontier: LedgerEntry | None) -> float | None:
        """The promotion threshold for `entry`, with the frontier scored on the same folds.

        Without cross-validation this is `_gate_score(frontier.primary_score())`. With it, the
        frontier's mean is taken over the folds `entry` was validated on, when the frontier has all
        of them, so a candidate is never compared against a mean that includes folds it did not see.
        The baseline is a K-fold mean and has no per-fold scores, so it stays as configured."""
        if frontier is None:
            return self._gate_score(None)
        matched = reference_score(entry, frontier)
        if matched is None and "seed_ids" not in entry.metric and "seed_ids" not in frontier.metric:
            matched = _fold_matched(entry, frontier) if self.profile.cv_folds >= 2 else None
        return self._gate_score(matched)

    def _propose(self, entries: list[LedgerEntry], *, lit_context: str = "", skills: str = "",
                 lit_priors: list[str] | None = None):
        """Propose → admissibility gate → debate gate. Returns (proposal, reasoning) or None."""
        import time
        from scholarloop.experiment.test_boundary import require_unopened_test
        require_unopened_test(self.ledger_path)

        if not lit_context and lit_priors is None:
            lit_context, lit_priors = self._literature()   # fall back to the Lit Scout
        if not skills and self.skill_library is not None:
            # context assembly: bias the (bounded) skill slot toward lessons relevant to the
            # current direction — topic + guidance + literature priors — not just the heaviest.
            query = " ".join(filter(None, [self.topic, self.guidance, *(lit_priors or [])]))
            skills = self.skill_library.render(time.time(), query=query)   # injected intuition, relevance-ranked
        warm = sum(1 for e in entries if e.fidelity == ["smoke"]) < self.warm_start
        if warm:
            from scholarloop.experiment.random_search import RandomProposer
            if self._warm is None:
                self._warm = RandomProposer(self.profile, seed=self.warm_seed)
            proposal = self._warm.propose(entries)
        else:
            frontier = self._parent(entries)
            base = self.candidates.load(frontier.candidate_id) if frontier and frontier.candidate_id else None
            proposal = self.reasoner.propose(entries, lit_context=lit_context, skills=skills,
                                             guidance=self.guidance, lit_priors=lit_priors,
                                             base_candidate=base,
                                             calibration=self.calibration.render())   # close the verifier loop
        if not proposal.admissible:
            print(f"orchestrator: skipping inadmissible proposal {proposal.config} "
                  f"({proposal.violations})", file=sys.stderr)
            return None
        if proposal.snapshot is None:
            proposal.snapshot = make_snapshot(self.profile, proposal.config, proposal.edits)
        reasoning = proposal.reasoning_blob()
        if self.debate_panel is not None and not warm:     # diverse-perspective gate before GPU
                                                           # (a random draw makes no claim to vet)
            verdict = self.debate_panel.review(proposal)
            reasoning["debate"] = verdict.to_dict()
            if verdict.decision == "reject":
                print(f"orchestrator: panel rejected {proposal.config} ({verdict.concerns})",
                      file=sys.stderr)
                return None
        return proposal, reasoning

    def _run_tier(self, proposal, reasoning, fidelity, exp_id, parent_id, parent_score, predicted_delta):
        from scholarloop.experiment.runner import run_experiment   # imported lazily (heavy deps)
        reference = next((e for e in Ledger(self.ledger_path).read_all() if e.id == parent_id), None)
        return run_experiment(
            self.profile, fidelity, exp_id, proposal.hypothesis,
            config_override=proposal.config, edits=proposal.edits, reasoning=reasoning,
            predicted_delta=predicted_delta, parent=parent_id, parent_score=parent_score,
            ledger_path=self.ledger_path, registry_dir=self.registry_dir,
            artifact_cache=self._round_cache, candidate_snapshot=proposal.snapshot,
            prediction_reference=reference, training_budget=self.training_budget)

    def _calibrate(self, entry) -> None:
        """Universal predict-then-verify (B): score every agent claim attached to this run once the
        ground truth (its measured delta / final verdict) is in."""
        pred = entry.prediction or {}
        self.calibration.record(score_delta("reasoner", pred.get("predicted"), pred.get("measured")))
        debate = (entry.reasoning or {}).get("debate") or {}
        if debate.get("decision") == "run":              # the panel bet this idea was worth a GPU run
            self.calibration.record(score_binary("debate", True, entry.verdict == "kept"))

    def _post_run(self, entry, frontier, entries):
        if self.reflector is not None and self.skill_library is not None:   # Reflect → skill library
            skill = self.reflector.reflect(entry, frontier)
            if skill is not None:
                self.skill_library.add(skill)
        if self.advisor is not None:                       # post-run loop control
            recent = [e.verdict for e in entries if e.purpose == "exploration"][-5:]
            from scholarloop.experiment.reasoner import _ranges_line
            self.last_advice = self.advisor.advise(
                entry, frontier, recent_verdicts=recent, direction=self.profile.metric.direction,
                parent_score=reference_score(entry, frontier),
                ranges=_ranges_line(self.profile.knobs) if self.profile.knobs else "")
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
        self._calibrate(entry)                          # entry carries the prediction at this single tier
        self._post_run(entry, parent, entries + [entry])
        return entry

    def _climb_from(self, smoke_entry, proposal, reasoning, tiers, frontier, known) -> list[LedgerEntry]:
        """Climb an already-smoked idea up the remaining tiers (verify→full), chaining via `parent`
        and stopping the moment a tier fails its promotion gate. Returns the entries produced."""
        produced: list[LedgerEntry] = []
        prev = smoke_entry
        for fidelity in tiers:
            entry = self._run_tier(proposal, reasoning, fidelity,
                                   self._next_id(known + produced), prev.id, prev.primary_score(), None)
            produced.append(entry)
            if not self._promote(entry, fidelity, self._gate_for(entry, frontier)):
                break
            prev = entry
        return produced

    def funnel_step(self, **kw) -> list[LedgerEntry]:
        """Funnel one idea up the fidelity ladder (see `_funnel_step`), sharing trained models
        across its tiers: verify re-scores smoke's seed and full re-scores verify's, so an idea
        that reaches full trains 5 models, not 1 + 3 + 5."""
        with self._shared_models():
            return self._funnel_step(**kw)

    def population_step(self, k: int = 4, **kw) -> list[LedgerEntry]:
        """The population funnel (see `_population_step`), sharing trained models across tiers."""
        with self._shared_models():
            return self._population_step(k, **kw)

    @contextmanager
    def _shared_models(self):
        if self._round_cache is not None:                  # nested: reuse the outer cache
            yield
            return
        self._round_cache = tempfile.mkdtemp(prefix="scholarloop_models_")
        try:
            yield
        finally:
            shutil.rmtree(self._round_cache, ignore_errors=True)
            self._round_cache = None

    def _funnel_step(self, *, tiers=("smoke", "verify", "full"), lit_context: str = "",
                     skills: str = "", lit_priors: list[str] | None = None) -> list[LedgerEntry]:
        """Funnel one idea up the fidelity ladder: each tier must clear a promotion
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
        smoke = self._run_tier(proposal, reasoning, tiers[0], self._next_id(entries),
                               frontier.id if frontier else None, frontier_score, proposal.predicted_delta)
        produced = [smoke]
        if self._promote(smoke, tiers[0], self._gate_for(smoke, frontier)):   # beat the frontier (or baseline)
            produced += self._climb_from(smoke, proposal, reasoning, tiers[1:], frontier, entries + produced)
        self._calibrate(smoke)                          # the smoke entry carries the idea's prediction
        self._post_run(produced[-1], frontier, entries + produced)
        return produced

    def _population_step(self, k: int = 4, *, tiers=("smoke", "verify", "full"),
                        max_workers: int = 1, lit_context: str = "", skills: str = "",
                        lit_priors: list[str] | None = None) -> list[LedgerEntry]:
        """Population funnel (parallel fan-out): propose up to `k` distinct ideas, smoke-screen them
        ALL (concurrently when max_workers>1), then climb only the smoke survivors up verify→full.
        Expensive tiers are spent on a pre-screened few, not on every idea. Returns every entry run."""
        entries = list(Ledger(self.ledger_path).read_all())
        frontier = self._parent(entries)
        frontier_score = frontier.primary_score() if frontier else None

        # Phase 1 — propose up to k admissible, mutually-distinct ideas.
        batch: list[tuple] = []
        seen: set[str] = set()
        for _ in range(k):
            prepared = self._propose(entries, lit_context=lit_context, skills=skills, lit_priors=lit_priors)
            if prepared is None:
                continue
            proposal, reasoning = prepared
            if proposal.snapshot is None:
                proposal.snapshot = make_snapshot(self.profile, proposal.config, proposal.edits)
            identity = proposal.snapshot["id"]
            if identity in seen:
                continue                                # same implementation AND effective config
            seen.add(identity)
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
        for e in smoked:                                # each smoke entry carries its idea's prediction
            self._calibrate(e)

        # Phase 3 — climb only the smoke survivors, best-first, through the remaining tiers.
        survivors = [(e, batch[i][0], batch[i][1]) for i, e in enumerate(smoked)
                     if self._promote(e, smoke_tier, self._gate_for(e, frontier))]
        survivors.sort(key=lambda t: t[0].primary_score() if t[0].primary_score() is not None else float("inf"),
                       reverse=(self.profile.metric.direction == "maximize"))
        for smoke_entry, proposal, reasoning in survivors:
            produced += self._climb_from(smoke_entry, proposal, reasoning, tiers[1:], frontier,
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
        Smoke screens with slack; verify uses a pessimistic SEM margin
        as a compute-allocation heuristic, not a significance test. Cheap stages keep
        candidates; expensive stages collect wider validation evidence."""
        if fidelity == "full":
            return False                                   # full is terminal
        score = entry.primary_score()
        if score is None:                                  # killed
            return False
        seeds = entry.metric.get("seeds") or [score]
        if fidelity == "verify" and len(seeds) < 2:        # a degraded verify (seeds crashed) isn't the
            return False                                   # multi-seed robustness check full deserves
        if gate is None:
            return True                                    # nothing to gate on yet — climb
        if fidelity == "smoke":                            # coarse screen: only kill the clearly-worse
            return self.profile.metric.is_better(score, self._relaxed_gate(gate))
        if not self.profile.metric.is_better(score, gate):
            return False                                   # didn't clear the bar — drop it cheaply
        if fidelity == "verify":                           # noise-margin heuristic, not inferential confirmation
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
            self._refine_count, self.guidance = 0, self._base_guidance   # keep steering by the adopted plan

    def _pivot(self, entries: list[LedgerEntry], rationale: str) -> None:
        """Change direction: reset refines, drop cached literature, re-ground.

        With a Director this genuinely re-directs (new topic + direction). Without one the topic
        is unchanged, so we at least instruct the Reasoner to leave the exhausted region.
        """
        self._refine_count = 0
        self._base_guidance = ""             # the adopted plan is exhausted; stop steering by it
        if self.lit_scout is not None:
            self._lit = None                 # re-ground from fresh literature on the new direction
        if self.director is not None:
            d = self.director.direct(entries)
            self.topic = d.get("topic", self.topic)
            self.guidance = d.get("direction", rationale)
        else:
            self.guidance = (f"{rationale} — change approach: explore a different architecture/"
                             "optimizer beyond the current focus; revisit warnings when justified.").strip()

    def _spent(self) -> float | None:
        """Cumulative LLM spend (USD) for the Governor, or None when the model has no known price."""
        usage = getattr(self.llm, "usage", None) or {}
        return cost_of(usage, getattr(self.llm, "model", None))

    def _round(self, *, funnel, population, fidelity, max_workers, lit_context, skills,
               lit_priors=None) -> list[LedgerEntry]:
        """One campaign round: a population funnel, a single-idea funnel, or one flat run."""
        if population > 1:
            return self.population_step(population, max_workers=max_workers,
                                        lit_context=lit_context, skills=skills, lit_priors=lit_priors)
        if funnel:
            return self.funnel_step(lit_context=lit_context, skills=skills, lit_priors=lit_priors)
        entry = self.step(fidelity=fidelity, lit_context=lit_context, skills=skills, lit_priors=lit_priors)
        return [entry] if entry is not None else []

    def run(self, n_steps: int | None = None, *, fidelity: str = "smoke", funnel: bool = False,
            population: int = 1, max_workers: int = 1, governor: Governor | None = None,
            lit_context: str = "", skills: str = "",
            lit_priors: list[str] | None = None) -> list[LedgerEntry]:
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
        from scholarloop.experiment.test_boundary import require_unopened_test
        require_unopened_test(self.ledger_path)
        if self.director is not None and not self._directed:   # set the campaign direction once
            d = self.director.direct(list(Ledger(self.ledger_path).read_all()))
            self.topic, self.guidance, self._directed = d.get("topic", self.topic), \
                d.get("direction", ""), True
        direction = self.profile.metric.direction
        self.stop_reason = None
        if governor is not None and governor.max_cost is not None and self._spent() is None:
            print(f"governor: WARNING — max_cost=${governor.max_cost:.2f} cannot be enforced: no price "
                  f"is known for model {getattr(self.llm, 'model', None)!r}; the budget gate is OFF "
                  "(add it to scholarloop.experiment.governor.MODEL_PRICES).", file=sys.stderr)
        if governor is not None:                               # seed convergence tracking from any prior best
            fr = best_comparable(list(Ledger(self.ledger_path).read_all()), self.profile, kept_only=True)
            if fr is not None:
                governor.update_frontier(fr.primary_score(), direction, scope=scope(fr))
        out: list[LedgerEntry] = []
        round_idx = 0
        while True:
            if governor is not None:
                stop, why = governor.should_stop(self._spent())
                if stop:
                    self.stop_reason = why
                    print(f"governor: stopping — {why}", file=sys.stderr)
                    break
            if n_steps is not None and round_idx >= n_steps:
                break
            if n_steps is None and governor is None:               # nothing bounds the loop -> one round
                if round_idx >= 1:
                    break

            produced = self._round(funnel=funnel, population=population, fidelity=fidelity,
                                    max_workers=max_workers, lit_context=lit_context, skills=skills,
                                    lit_priors=lit_priors)
            out.extend(produced)
            round_idx += 1

            if governor is not None:
                frontier = best_comparable(list(Ledger(self.ledger_path).read_all()), self.profile,
                                           kept_only=True)
                improved = (governor.update_frontier(frontier.primary_score(), direction, scope=scope(frontier))
                            if frontier else False)
                governor.record_round(improved)
                for alert in governor.alerts(self._spent()):
                    print(f"governor: {alert}", file=sys.stderr)
        return out
