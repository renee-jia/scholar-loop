"""Matched-budget comparison of search arms (the agent loop vs random search), with intervals.

One campaign's result is the idea it would report: the best **full-tier** run by val, and that
run's **held-out test** score, which no arm selected on. Arms are compared over several
independent campaigns (repeats), never over one, because a single campaign's best is mostly
luck. The budget is matched in rounds (ideas through the funnel) or in training runs (seed runs
actually consumed); `run_campaign` takes either bound, and every summary reports both, so a
difference in compute is visible rather than hidden.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean, stdev

from scholarloop.core.ledger import Ledger, LedgerEntry
from scholarloop.core.profile import Profile
from scholarloop.experiment.runner import SEEDS_PER_FIDELITY

# two-sided 95% Student-t quantiles by degrees of freedom; repeats are few, so 1.96 would lie
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
        10: 2.23, 15: 2.13, 20: 2.09, 30: 2.04}


def t95(df: int) -> float:
    return _T95.get(df) or _T95[max(k for k in _T95 if k <= df)] if df >= 1 else float("nan")


def train_runs(entries: list[LedgerEntry]) -> int:
    """Models trained for these entries (a killed run still spent its budget). A seed an earlier
    tier already trained is re-scored, not retrained, and is not counted again."""
    return sum(len(SEEDS_PER_FIDELITY[e.fidelity[0]]) - (e.metric or {}).get("reused_seeds", 0)
               for e in entries if e.fidelity)


def run_campaign(orch, *, rounds: int | None = None, max_train_runs: int | None = None,
                 max_rounds: int = 200) -> list[LedgerEntry]:
    """Run funnel rounds on `orch` until `rounds` rounds have run or `max_train_runs` seed runs
    are spent (the round that crosses the budget finishes; there is no partial idea)."""
    if rounds is None and max_train_runs is None:
        raise ValueError("bound the campaign by rounds or by max_train_runs")
    done = 0
    while done < (rounds if rounds is not None else max_rounds):
        entries = list(Ledger(orch.ledger_path).read_all())
        if max_train_runs is not None and train_runs(entries) >= max_train_runs:
            break
        orch.run(1, funnel=True)
        done += 1
    return list(Ledger(orch.ledger_path).read_all())


def campaign_summary(entries: list[LedgerEntry], profile: Profile) -> dict:
    """What one campaign would report, plus what it cost and how its funnel converted."""
    full = [e for e in entries if e.fidelity == ["full"] and e.primary_score() is not None]
    pick = None
    if full:
        key = lambda e: e.primary_score()                       # noqa: E731
        pick = min(full, key=key) if profile.metric.direction == "minimize" else max(full, key=key)
    tiers = Counter(e.fidelity[0] for e in entries if e.fidelity)
    ideas = sum(1 for e in entries if e.fidelity == ["smoke"])
    return {"ideas": ideas, "train_runs": train_runs(entries),
            "tiers": {t: tiers.get(t, 0) for t in ("smoke", "verify", "full")},
            "killed": sum(e.verdict == "killed" for e in entries),
            "best_full": None if pick is None else {
                "id": pick.id, "val": pick.primary_score(), "holdout": pick.holdout_score(),
                "config": pick.config}}


def _interval(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    if len(xs) == 1:
        return {"n": 1, "mean": round(xs[0], 4)}
    m, s = mean(xs), stdev(xs)
    half = t95(len(xs) - 1) * s / math.sqrt(len(xs))
    return {"n": len(xs), "mean": round(m, 4), "std": round(s, 4),
            "ci95": [round(m - half, 4), round(m + half, 4)]}


def arm_stats(summaries: list[dict]) -> dict:
    """Across one arm's repeats: how often it reached the full tier, the reported val and
    held-out scores with 95% intervals, and the compute it used."""
    picks = [s["best_full"] for s in summaries if s["best_full"]]
    return {"campaigns": len(summaries), "reached_full": len(picks),
            "val": _interval([p["val"] for p in picks]),
            "holdout": _interval([p["holdout"] for p in picks if p["holdout"] is not None]),
            "train_runs": _interval([s["train_runs"] for s in summaries]),
            "ideas": _interval([s["ideas"] for s in summaries]),
            "funnel": {t: sum(s["tiers"][t] for s in summaries) for t in ("smoke", "verify", "full")}}


def win_rate(a: list[float], b: list[float], direction: str) -> float | None:
    """P(a random campaign of arm A reports a better score than one of arm B); ties count half.
    0.5 means no difference. This is the common-language effect size (AUC)."""
    if not a or not b:
        return None
    better = (lambda x, y: x < y) if direction == "minimize" else (lambda x, y: x > y)   # noqa: E731
    wins = sum(1.0 if better(x, y) else 0.5 if x == y else 0.0 for x in a for y in b)
    return round(wins / (len(a) * len(b)), 3)


def _held(summaries: list[dict]) -> list[float]:
    return [s["best_full"]["holdout"] for s in summaries
            if s["best_full"] and s["best_full"]["holdout"] is not None]


def render(arms: dict[str, list[dict]], profile: Profile, *, match: str) -> str:
    """A markdown report of the arms, side by side."""
    stats = {name: arm_stats(s) for name, s in arms.items()}
    m = profile.metric
    fmt = lambda iv: ("—" if not iv.get("n") else f"{iv['mean']}" + (                  # noqa: E731
        f" ± {iv['std']} (95% CI {iv['ci95'][0]}..{iv['ci95'][1]})" if "ci95" in iv else ""))
    out = [f"# Matched-budget comparison on {profile.name}", "",
           f"Budget matched by **{match}**. Each campaign reports its best full-tier run by val; the "
           f"held-out column is that run's test score, which no arm selected on. "
           f"Metric: {m.name} ({m.direction}); must-beat baseline {profile.best_baseline()}.", "",
           "| arm | campaigns | reached full | reported val | held-out test | train runs | ideas | funnel smoke/verify/full |",
           "|---|---|---|---|---|---|---|---|"]
    for name, s in stats.items():
        f = s["funnel"]
        out.append(f"| {name} | {s['campaigns']} | {s['reached_full']} | {fmt(s['val'])} | "
                   f"{fmt(s['holdout'])} | {fmt(s['train_runs'])} | {fmt(s['ideas'])} | "
                   f"{f['smoke']}/{f['verify']}/{f['full']} |")
    names = list(arms)
    if len(names) == 2:
        a, b = names
        w = win_rate(_held(arms[a]), _held(arms[b]), m.direction)
        if w is not None:
            out += ["", f"P({a} reports a better held-out score than {b}) = **{w}** over "
                        f"{len(_held(arms[a]))}x{len(_held(arms[b]))} campaign pairs (0.5 = no difference)."]
    out += ["", "## Per campaign", "", "| arm | # | ideas | train runs | best full id | val | held-out | config |",
            "|---|---|---|---|---|---|---|---|"]
    for name, summaries in arms.items():
        for i, s in enumerate(summaries, 1):
            p = s["best_full"] or {}
            out.append(f"| {name} | {i} | {s['ideas']} | {s['train_runs']} | {p.get('id', '—')} | "
                       f"{p.get('val', '—')} | {p.get('holdout', '—')} | `{p.get('config', '')}` |")
    return "\n".join(out) + "\n"
