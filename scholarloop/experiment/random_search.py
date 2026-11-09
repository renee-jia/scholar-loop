"""Random search: the control arm that the agent loop has to beat.

The common objection to LLM-driven research loops is that random search, given the same budget,
does as well (Gideoni et al. 2026 match code-evolution systems with it; Bergstra & Bengio 2012 is
the classic hyperparameter case). The only fair answer is to measure it, so `RandomProposer`
replaces the Reasoner inside the *same* Orchestrator: the same funnel, promotion gates, seeds,
folds, frozen scorer and held-out test. Only the proposer differs, and it makes no LLM call.

What random search gets, and what it does not:
  - it samples every knob the profile exposes, around the profile's default (see `knob_range`);
  - it never repeats a config already in the ledger (the agent is held to the same rule);
  - it does not get the ruled-out regions, literature priors, skills or debate gate. Those are
    the system's contribution, so they are what the comparison measures.
"""

from __future__ import annotations

import math
import random

from scholarloop.core.ledger import Hypothesis, LedgerEntry
from scholarloop.core.profile import Profile
from scholarloop.experiment.reasoner import Proposal
from scholarloop.experiment.reasoning import SearchSpaceConstraints, same_config

SOURCE = "random-search"        # the hypothesis source every random proposal cites


def knob_range(default) -> tuple:
    """The sampling range for a knob, from its default: (kind, low, high).

    A nonzero default d is searched log-uniformly over [d/4, 4d], two octaves each way, which
    covers the usual ranges for learning rates, widths, depths and epochs. Integers are rounded
    and kept >= 1. A zero default is an off switch: integers are drawn from 0..10, floats are 0
    half the time and log-uniform in [1e-5, 1e-2] otherwise."""
    if isinstance(default, bool):
        return ("choice", False, True)
    if isinstance(default, int):
        return ("int", 0, 10) if default == 0 else ("logint", max(1, default // 4), default * 4)
    d = float(default)
    if d == 0.0:
        return ("zero_or_log", 1e-5, 1e-2)
    return ("log", d / 4, d * 4) if d > 0 else ("uniform", 4 * d, d / 4)


def sample(kind: str, low, high, rng: random.Random):
    if kind == "choice":
        return rng.choice([low, high])
    if kind == "int":
        return rng.randint(low, high)
    if kind == "logint":
        return max(1, round(math.exp(rng.uniform(math.log(low), math.log(high)))))
    if kind == "zero_or_log":
        return 0.0 if rng.random() < 0.5 else float(f"{math.exp(rng.uniform(math.log(low), math.log(high))):.3g}")
    if kind == "log":
        return float(f"{math.exp(rng.uniform(math.log(low), math.log(high))):.3g}")
    return float(f"{rng.uniform(low, high):.3g}")


class RandomProposer:
    """Drop-in for `Reasoner.propose`: a uniformly random, never-repeated config. Set it as
    `orchestrator.reasoner`. `space` overrides `knob_range` per knob: {name: (kind, low, high)}."""

    def __init__(self, profile: Profile, *, seed: int = 0, space: dict | None = None,
                 max_draws: int = 100):
        self.profile = profile
        self.rng = random.Random(seed)
        self.space = {k: (space or {}).get(k) or knob_range(v) for k, v in profile.knobs.items()}
        self.max_draws = max_draws

    def propose(self, entries: list[LedgerEntry], **_context) -> Proposal:
        tried = [e.config for e in entries if e.config]
        config: dict = {}
        for _ in range(self.max_draws):
            config = {k: sample(*spec, self.rng) for k, spec in self.space.items()}
            if not any(same_config({**self.profile.knobs, **config}, t) for t in tried):
                break
        else:                                   # the space is exhausted at this resolution
            return Proposal(config, Hypothesis(claim="random config", source=SOURCE), None,
                            "random search", SearchSpaceConstraints(tried=tried),
                            violations=["config already tried"])
        return Proposal(config=config,
                        hypothesis=Hypothesis(claim="random config", source=SOURCE),
                        predicted_delta=None,           # random search makes no prediction
                        reasoning_trace="random search: knobs sampled independently",
                        constraints=SearchSpaceConstraints(tried=tried))
