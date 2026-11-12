"""Bounded, validation-only evidence views shared by search decision makers."""

import json
from statistics import mean

TIERS = {"smoke": 0, "verify": 1, "full": 2}


def search_entries(entries, profile):
    return [e for e in entries if e.purpose == "exploration" and
            e.domain == profile.name and e.metric_name == profile.metric.name]


def scope(entry):
    seeds = entry.metric.get("seed_ids")
    folds = entry.metric.get("folds", [None] * len(seeds or []))
    observations = (tuple(sorted(zip(seeds, folds))) if seeds is not None
                    else tuple(entry.metric.get("folds", [])))
    return (tuple(entry.fidelity), seeds is not None, observations)


def coverage(entry):
    return (len(set(entry.metric.get("folds", []))), len(entry.metric.get("seeds", [])),
            max((TIERS.get(t, -1) for t in entry.fidelity), default=-1))


def best_comparable(entries, profile, *, kept_only=False):
    rows = [e for e in search_entries(entries, profile) if e.primary_score() is not None
            and e.verdict in ("kept", "discarded")]
    order = {e.id: i for i, e in enumerate(rows)}
    candidates = {}
    for e in rows:
        key = e.candidate_id or ("legacy-run", e.id)
        if key not in candidates or coverage(e) >= coverage(candidates[key]):
            candidates[key] = e
    # A candidate rejected by wider evidence cannot remain a frontier via its smoke score.
    rows = [e for e in candidates.values() if not kept_only or e.verdict == "kept"]
    if not rows:
        return None
    # Prefer wider evidence; never rank an easy-fold smoke above a full evaluation.
    widest = max(coverage(e) for e in rows)
    rows = [e for e in rows if coverage(e) == widest]
    anchor = max(rows, key=lambda e: order[e.id])
    rows = [e for e in rows if scope(e) == scope(anchor)]
    key = lambda e: e.primary_score()
    return min(rows, key=key) if profile.metric.direction == "minimize" else max(rows, key=key)


def paired_values(entry, reference):
    """Pair by seed AND fold. Missing/ambiguous identifiers never imply pairing."""
    def values(e):
        if e is None:
            return None
        scores, seeds = e.metric.get("seeds", []), e.metric.get("seed_ids", [])
        folds = e.metric.get("folds", [None] * len(scores))
        if not scores or not (len(scores) == len(seeds) == len(folds)):
            return None
        keys = list(zip(seeds, folds))
        return dict(zip(keys, scores)) if len(set(keys)) == len(keys) else None
    a, b = values(entry), values(reference)
    if not a or not b or not set(a) <= set(b):
        return None
    return list(a.values()), [b[k] for k in a]


def reference_score(entry, reference):
    paired = paired_values(entry, reference)
    if paired:
        return mean(paired[1])
    if reference is not None and scope(entry) == scope(reference):
        return reference.primary_score()
    return None


def evidence_view(entries, profile, *, limit=12):
    if limit < 3:
        raise ValueError("Evidence view needs room for frontier, recent result and failure")
    rows = search_entries(entries, profile)
    frontier = best_comparable(rows, profile, kept_only=True)
    selected = []
    def add(e):
        if e is not None and e.id not in {r.id for r in selected} and len(selected) < limit:
            selected.append(e)
    add(frontier)
    add(rows[-1] if rows else None)
    add(next((e for e in reversed(rows) if e.verdict in ("killed", "discarded")), None))
    for e in reversed(rows):
        add(e)
    observations = []
    for e in selected:
        pair = paired_values(e, frontier) if e is not frontier else None
        observations.append({"run_id": e.id, "candidate_id": e.candidate_id,
            "implementation_id": e.run_spec.get("implementation_id"), "parent_run_id": e.parent,
            "hypothesis": {"claim": e.hypothesis.claim, "source": e.hypothesis.source},
            "provenance": "candidate_snapshot" if e.candidate_id else "legacy_no_snapshot",
            "config": e.config, "fidelity": e.fidelity, "verdict": e.verdict,
            "validation_score": e.primary_score(), "seed_ids": e.metric.get("seed_ids"),
            "folds": e.metric.get("folds"), "seed_scores": e.metric.get("seeds", []),
            "requested_seeds": e.metric.get("requested_seeds"),
            "failed_seeds": e.metric.get("seeds_failed", 0),
            "execution_failed": e.verdict == "killed",
            "paired_difference_to_frontier": mean(a-b for a,b in zip(*pair)) if pair else None})
    return {"frontier_run_id": frontier.id if frontier else None, "total_runs": len(rows),
            "omitted_runs": len(rows) - len(selected), "observations": observations,
            "interpretation": "Validation evidence only. Different fidelities/folds are not interchangeable. "
                              "Paired differences are descriptive, not independent confirmation. "
                              "Failures do not refute a mechanism; legacy code cannot be inferred from current source."}


def render_evidence(entries, profile, **kwargs):
    return json.dumps(evidence_view(entries, profile, **kwargs), ensure_ascii=False, allow_nan=False)
