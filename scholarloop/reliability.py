"""Reliability evaluation of the deterministic guards (citation, number, evidence-id, structure).

Three measurements, each answering a different question. None uses an LLM judge.

  inject    Seeded fault injection into artifacts the guards already accepted. Each variant
            differs from an accepted artifact at exactly one site and is passed through the
            *production* postcheck. Faults come in two tiers: `fabrication` (the artifact
            refers to something that does not exist) and `misattribution` (it refers to
            something real, wrongly). The second tier is outside what a lexical check can
            see; it is included so the report states the boundary instead of hiding it.
            Free and deterministic.

  traces    Natural fault rate, read from persisted agent attempt logs (`*.agents.jsonl`).
            A first attempt the harness rejected is an output that would have been used had
            the guard been off; an accepted later attempt is a recovery.

  baseline  A single-call writer given the same evidence block and the same instructions,
            with no enforced check. Its paper is audited post hoc by the same checkers.
            This isolates "the rule is in the prompt" from "the rule is enforced".
            Uses the configured API and can incur charges.

What the numbers do not show: that a paper passing these checks is scientifically correct,
or that injected faults are distributed like a model's natural errors (`traces` measures
those). Detection of a fabrication the checker was written to catch is expected to be high;
the informative cells are the misses and the per-stage gaps.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from scholarloop.core.registry import _number_tokens

# ---------------------------------------------------------------------------------------------
# Error categories — one classifier, shared by injection and trace analysis
# ---------------------------------------------------------------------------------------------
CATEGORIES = ("citation", "number", "style", "structure", "schema", "invalid_json")
_STYLE_MARKERS = ("em-dash", "banned vocabulary", "banned phrases", "contrast frame",
                  "open with connectives")


def classify(error: str) -> str:
    """Map one harness rejection message to the guard that produced it.

    `citation` covers every unknown identifier (paper, run or evidence id); `structure` is the
    remainder (missing sections, uncited result claims, empty fields). Pinned by tests against
    the production messages, so a reworded guard fails loudly instead of drifting to `structure`.
    """
    low = error.lower()
    if low.startswith("schema:"):
        return "schema"
    if "not valid json" in low:
        return "invalid_json"
    if "unknown" in low and (" ids" in low or " id " in low):
        return "citation"
    if "numbers not in the evidence" in low or "ungrounded" in low:
        return "number"
    if any(m in low for m in _STYLE_MARKERS):
        return "style"
    return "structure"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion; None when there are no trials."""
    if n == 0:
        return None
    p = k / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denom = 1 + z * z / n
    return (round(max(0.0, (centre - half) / denom), 4), round(min(1.0, (centre + half) / denom), 4))


# ---------------------------------------------------------------------------------------------
# Cases — accepted artifacts paired with the production check that accepted them
# ---------------------------------------------------------------------------------------------
@dataclass
class Case:
    target: str                                  # outline | key_points | section | lit_review | ideas
    name: str
    artifact: dict
    check: Callable[[dict], list[str]]
    text_paths: list[tuple]                      # JSON paths to prose the artifact carries
    id_paths: list[tuple]                        # JSON paths to lists of evidence/paper ids


@dataclass
class Context:
    bank: object                                 # EvidenceBank
    papers: list[str]                            # corpus paper ids
    runs: list[str]                              # ledger run ids
    statements: list[str] = field(default_factory=list)   # approved statement ids (reviewed mode)
    mode: str = "legacy"                         # "reviewed": guards bound to an approved release


def _get(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def _set(obj, path, value):
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def load_cases(profile, *, ledger, registry, research, writing) -> tuple[list[Case], Context]:
    """Build cases from one captured run: its ledger, registry, research session and writing state.

    When the writing state is bound to a scientific release that still validates, the guards
    run in `reviewed` mode, exactly as they did in production: prose must cite approved
    statements and numbers are checked per sentence against the statements cited there.
    Otherwise they run in `legacy` mode (numbers and ids checked against the whole evidence
    bank), and `Context.mode` says why."""
    from scholarloop.core.ledger import Ledger
    from scholarloop.ideation.ideator import Ideator
    from scholarloop.ideation.litreview import LitReview, LitReviewer
    from scholarloop.writing.session import EvidenceBank, KeyPointer, Outliner, SectionDrafter

    research = Path(research)
    entries = list(Ledger(ledger).read_all())
    lit_path = research / "lit_review.json"
    raw_review = json.loads(lit_path.read_text()) if lit_path.exists() else None
    review = LitReview.from_dict(raw_review) if raw_review else None
    handoff = research / "handoff.json"
    idea = json.loads(handoff.read_text()).get("idea") if handoff.exists() else None
    bank = EvidenceBank.build(entries, registry, profile, review=review, idea=idea)
    ctx = Context(bank, [p.arxiv_id for p in review.papers] if review else [], [e.id for e in entries])
    state = json.loads(Path(writing).read_text())
    if state.get("release_id"):
        ctx.mode = _bind_release(profile, bank, ctx, ledger=ledger, registry=registry,
                                 research=research, review_dir=Path(writing).parent)
    cases: list[Case] = []

    if review:
        reviewer = LitReviewer(None)
        lit = {k: raw_review[k] for k in ("themes", "paper_notes", "gaps") if k in raw_review}
        cases.append(Case("lit_review", "lit_review", lit,
                          lambda a: reviewer.postcheck(a, {"papers": review.papers}), [],
                          [("themes", i, "papers") for i in range(len(lit.get("themes", [])))]
                          + [("gaps", i, "sources") for i in range(len(lit.get("gaps", [])))]))

    ideas_path = research / "ideas.json"
    if ideas_path.exists() and review:
        ideas = {"ideas": json.loads(ideas_path.read_text())}
        ideator = Ideator(None, profile)
        cases.append(Case("ideas", "ideas", ideas,
                          lambda a: ideator.postcheck(a, {"review": review}), [],
                          [("ideas", i, "sources") for i in range(len(ideas["ideas"]))]))

    outline, points = state.get("outline"), state.get("key_points")
    if outline:
        outliner = Outliner(None, bank)
        cases.append(Case("outline", "outline", outline, lambda a: outliner.postcheck(a, {}),
                          [("nugget",)] + [("contributions", i) for i in range(len(outline["contributions"]))],
                          [("sections", i, "evidence") for i in range(len(outline["sections"]))]))
    if outline and points:
        pointer = KeyPointer(None, bank)
        sites = [(i, j) for i, s in enumerate(points["sections"]) for j in range(len(s["points"]))]
        cases.append(Case("key_points", "key_points", points,
                          lambda a: pointer.postcheck(a, {"outline": outline}),
                          [("sections", i, "points", j, "claim") for i, j in sites],
                          [("sections", i, "points", j, "evidence") for i, j in sites]))
    drafter = SectionDrafter(None, bank)
    for section in state.get("sections") or []:
        body = {"heading": section["heading"], "body": section["body"]}
        cases.append(Case("section", f"section:{section['heading']}", body,
                          lambda a: drafter.postcheck(a, {}), [("body",)], []))
    return cases, ctx


def _bind_release(profile, bank, ctx, *, ledger, registry, research, review_dir) -> str:
    """Attach the approved dossier the writing was bound to, if its release still validates."""
    from scholarloop.review.contracts import ReviewBlocked
    from scholarloop.review.dossier import approved_statements
    from scholarloop.review.session import ScientificReviewSession
    reason = "no release"
    for research_dir in (research, None):           # the review may or may not have had a research dir
        gate = ScientificReviewSession(None, profile, review_dir, ledger_path=ledger,
                                       registry_dir=registry, research_dir=research_dir)
        try:
            dossier, _release = gate.require_release()
        except ReviewBlocked as exc:
            reason = str(exc)
            continue
        bank.approved_dossier = dossier
        ctx.statements = sorted(approved_statements(dossier))
        return "reviewed"
    return f"legacy (the writing cites a release that no longer validates: {reason})"


# ---------------------------------------------------------------------------------------------
# Fault operators — each yields (description, mutated artifact) for every applicable site
# ---------------------------------------------------------------------------------------------
_DECIMAL = re.compile(r"(?<![\w:./])\d+\.\d{2,}(?![\w/]|\.\d)")
_PAPER_ID = re.compile(r"(arXiv:\d{4}\.\d{4,5}|doi:[\w./\-()]+\d|openalex:\w+|s2:\w+)")
_RUN_ID = re.compile(r"exp_\d{4,}")
_SEED_COUNT = re.compile(r"\b([2-9])(?=[ -]seeds?\b)")
_STATEMENT = re.compile(r"(?<=\[)(?:claim_[A-Za-z0-9_]+|scope_question|scope_method|limit_\d+)(?=\])")
_STATEMENT_CITE = re.compile(r"\s*\[(?:claim_[A-Za-z0-9_]+|scope_question|scope_method|limit_\d+)\]")
_FLIPS = [("below", "above"), ("lower", "higher"), ("rose", "fell"), ("larger", "smaller"),
          ("increase", "decrease"), ("worse", "better")]


def _fresh_paper(rng, ctx) -> str:
    while True:
        pid = f"arXiv:{rng.randint(24, 26)}{rng.randint(1, 12):02d}.{rng.randint(10000, 99999)}"
        if pid not in ctx.papers:
            return pid


def _near_miss(rng, pid, ctx) -> str | None:
    digits = [i for i, c in enumerate(pid) if c.isdigit()]
    for _ in range(20):
        i = rng.choice(digits)
        out = pid[:i] + rng.choice([d for d in "0123456789" if d != pid[i]]) + pid[i + 1:]
        if out not in ctx.papers:
            return out
    return None


def _fresh_run(rng, ctx) -> str:
    while True:
        rid = f"exp_{rng.randint(1000, 9999)}"
        if rid not in ctx.runs:
            return rid


def _fabricated_number(rng, tok, ctx) -> str | None:
    """A value near `tok`, at the same precision, that no measurement grounds."""
    decimals = len(tok.split(".")[1])
    for _ in range(40):
        value = float(tok) + rng.choice((-1, 1)) * rng.uniform(0.03, 0.6)
        out = f"{value:.{decimals}f}"
        if value > 0 and out != tok and ctx.bank.ungrounded(out):
            return out
    return None


def _other_measurement(rng, tok, ctx) -> str | None:
    """A different *real* measurement, formatted like `tok` (grounded somewhere, wrong here)."""
    decimals = len(tok.split(".")[1])
    pool = sorted({f"{v:.{decimals}f}" for v in ctx.bank.registry._all_values() if v != int(v)})
    pool = [p for p in pool if p != tok and float(p) != float(tok) and not ctx.bank.ungrounded(p)]
    return rng.choice(pool) if pool else None


def _text_edits(case, pattern, replace):
    """Replace one regex match at a time in every prose field. `replace(match)` → str | None."""
    for path in case.text_paths:
        text = _get(case.artifact, path)
        for m in pattern.finditer(text):
            new = replace(m)
            if new is None or new == m.group(0):
                continue
            mutated = copy.deepcopy(case.artifact)
            _set(mutated, path, text[:m.start()] + new + text[m.end():])
            yield f"{m.group(0)} → {new}", mutated


def _list_edits(case, edit):
    """Edit one id list at a time. `edit(ids)` → (description, new list) | None."""
    for path in case.id_paths:
        result = edit(list(_get(case.artifact, path)))
        if result is None:
            continue
        mutated = copy.deepcopy(case.artifact)
        _set(mutated, path, result[1])
        yield result[0], mutated


def _swap_in_list(ids, pick, replacement):
    """Replace one id chosen by `pick(ids)` with `replacement(id)`; None if nothing applies."""
    candidates = pick(ids)
    if not candidates:
        return None
    old = candidates[0]
    new = replacement(old)
    if new is None or new in ids:
        return None
    return f"{old} → {new}", [new if i == old else i for i in ids]


def fabricated_citation(rng, case, ctx):
    for path in case.text_paths:                         # append a citation to a sentence
        text = _get(case.artifact, path)
        ends = [m.end() for m in re.finditer(r"[a-z\]]\.(?=\s|$)", text)]
        for end in ends[:3] if len(ends) > 3 else ends:
            pid = _fresh_paper(rng, ctx)
            mutated = copy.deepcopy(case.artifact)
            _set(mutated, path, text[:end - 1] + f" [{pid}]." + text[end:])
            yield f"+[{pid}]", mutated

    def add(ids):
        pid = _fresh_paper(rng, ctx)
        return f"+{pid}", ids + [pid]
    yield from _list_edits(case, add)


def near_miss_citation(rng, case, ctx):
    yield from _text_edits(case, _PAPER_ID,
                           lambda m: _near_miss(rng, m.group(0), ctx) if m.group(0) in ctx.papers else None)
    yield from _list_edits(case, lambda ids: _swap_in_list(
        ids, lambda xs: [x for x in xs if x in ctx.papers], lambda x: _near_miss(rng, x, ctx)))


def fabricated_run_id(rng, case, ctx):
    yield from _text_edits(case, _RUN_ID, lambda m: _fresh_run(rng, ctx))
    yield from _list_edits(case, lambda ids: _swap_in_list(
        ids, lambda xs: [x for x in xs if x in ctx.runs], lambda x: _fresh_run(rng, ctx)))


def fabricated_number(rng, case, ctx):
    yield from _text_edits(case, _DECIMAL, lambda m: _fabricated_number(rng, m.group(0), ctx))


def fabricated_statement_id(rng, case, ctx):
    def fresh(_old=None):
        while True:
            sid = f"claim_x{rng.randint(100, 999)}"
            if sid not in ctx.statements:
                return sid
    if not ctx.statements:
        return
    yield from _text_edits(case, _STATEMENT, lambda m: fresh())
    yield from _list_edits(case, lambda ids: _swap_in_list(
        ids, lambda xs: [x for x in xs if x in ctx.statements], fresh))


def uncited_sentence(rng, case, ctx):
    if ctx.statements:
        yield from _text_edits(case, _STATEMENT_CITE, lambda m: "")


def wrong_statement_id(rng, case, ctx):
    def other(sid, exclude=()):
        pool = [s for s in ctx.statements if s != sid and s not in exclude]
        return rng.choice(pool) if pool else None
    yield from _text_edits(case, _STATEMENT, lambda m: other(m.group(0)))

    def swap(ids):
        present = [x for x in ids if x in ctx.statements]
        new = other(present[0], exclude=ids) if present else None
        return None if new is None else (f"{present[0]} → {new}", [new if i == present[0] else i for i in ids])
    yield from _list_edits(case, swap)


def dropped_limitations(rng, case, ctx):
    if case.target != "outline":
        return
    kept = [s for s in case.artifact["sections"] if "limitation" not in s["heading"].lower()]
    if len(kept) != len(case.artifact["sections"]):
        yield "Limitations section removed", {**copy.deepcopy(case.artifact), "sections": copy.deepcopy(kept)}


def uncited_claim(rng, case, ctx):
    from scholarloop.writing.session import _ROLES_NEEDING_EVIDENCE
    if case.target == "key_points":
        for i, s in enumerate(case.artifact["sections"]):
            for j, p in enumerate(s["points"]):
                if p.get("role") in _ROLES_NEEDING_EVIDENCE and p.get("evidence"):
                    mutated = copy.deepcopy(case.artifact)
                    mutated["sections"][i]["points"][j]["evidence"] = []
                    yield f"{s['heading']} point {j + 1}: evidence removed", mutated
    if case.target == "ideas":
        for i in range(len(case.artifact["ideas"])):
            mutated = copy.deepcopy(case.artifact)
            mutated["ideas"][i]["sources"] = []
            yield f"idea {i + 1}: sources removed", mutated


def dropped_section(rng, case, ctx):
    if case.target == "key_points":
        for i, s in enumerate(case.artifact["sections"]):
            mutated = copy.deepcopy(case.artifact)
            del mutated["sections"][i]
            yield f"key points for {s['heading']!r} removed", mutated


def misattributed_number(rng, case, ctx):
    yield from _text_edits(case, _DECIMAL, lambda m: _other_measurement(rng, m.group(0), ctx))


def miscitation(rng, case, ctx):
    def other(pid):
        pool = [p for p in ctx.papers if p != pid]
        return rng.choice(pool) if pool else None
    yield from _text_edits(case, _PAPER_ID, lambda m: other(m.group(0)) if m.group(0) in ctx.papers else None)

    def swap(ids):
        present = [x for x in ids if x in ctx.papers]
        pool = [p for p in ctx.papers if p not in ids]
        if not present or not pool:
            return None
        new = rng.choice(pool)
        return f"{present[0]} → {new}", [new if i == present[0] else i for i in ids]
    yield from _list_edits(case, swap)


def wrong_run_id(rng, case, ctx):
    def other(rid, exclude=()):
        pool = [r for r in ctx.runs if r != rid and r not in exclude]
        return rng.choice(pool) if pool else None
    yield from _text_edits(case, _RUN_ID, lambda m: other(m.group(0)) if m.group(0) in ctx.runs else None)

    def swap(ids):
        present = [x for x in ids if x in ctx.runs]
        new = other(present[0], exclude=ids) if present else None
        return None if new is None else (f"{present[0]} → {new}", [new if i == present[0] else i for i in ids])
    yield from _list_edits(case, swap)


def changed_seed_count(rng, case, ctx):
    yield from _text_edits(case, _SEED_COUNT,
                           lambda m: rng.choice([d for d in "23456789" if d != m.group(1)]))


def direction_flip(rng, case, ctx):
    for a, b in _FLIPS:
        for old, new in ((a, b), (b, a)):
            yield from _text_edits(case, re.compile(rf"\b{old}\b"), lambda m, new=new: new)


# name → (tier, operator). `fabrication`: refers to something absent from the evidence, or omits
# a required element. `misattribution`: every referenced item exists, but is the wrong one.
FAULTS: dict[str, tuple[str, Callable]] = {
    "fabricated_citation": ("fabrication", fabricated_citation),
    "near_miss_citation": ("fabrication", near_miss_citation),
    "fabricated_run_id": ("fabrication", fabricated_run_id),
    "fabricated_number": ("fabrication", fabricated_number),
    "fabricated_statement_id": ("fabrication", fabricated_statement_id),
    "uncited_sentence": ("fabrication", uncited_sentence),
    "uncited_claim": ("fabrication", uncited_claim),
    "dropped_section": ("fabrication", dropped_section),
    "dropped_limitations": ("fabrication", dropped_limitations),
    "misattributed_number": ("misattribution", misattributed_number),
    "miscitation": ("misattribution", miscitation),
    "wrong_run_id": ("misattribution", wrong_run_id),
    "wrong_statement_id": ("misattribution", wrong_statement_id),
    "changed_seed_count": ("misattribution", changed_seed_count),
    "direction_flip": ("misattribution", direction_flip),
}


# ---------------------------------------------------------------------------------------------
# Injection study
# ---------------------------------------------------------------------------------------------
def _rate(k: int, n: int) -> dict:
    return {"n": n, "detected": k, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def run_injection(cases: list[Case], ctx: Context, *, seed: int = 0, per_case: int = 8) -> dict:
    """Inject every fault into every accepted case; at most `per_case` sites per (fault, case).

    A case the guards reject *before* injection is reported as a false positive and excluded
    from injection, so every detection below is caused by the injected change alone.
    """
    controls, usable = [], []
    for case in cases:
        errors = case.check(copy.deepcopy(case.artifact))
        controls.append({"case": case.name, "target": case.target, "errors": errors})
        if not errors:
            usable.append(case)
    trials = []
    for fault, (tier, operator) in FAULTS.items():
        for case in usable:
            rng = random.Random(f"{seed}:{fault}:{case.name}")
            variants = list(operator(rng, case, ctx))
            if len(variants) > per_case:
                variants = [variants[i] for i in sorted(rng.sample(range(len(variants)), per_case))]
            for description, mutated in variants:
                errors = case.check(mutated)
                trials.append({"fault": fault, "tier": tier, "target": case.target, "case": case.name,
                               "change": description, "detected": bool(errors), "errors": errors,
                               "categories": sorted({classify(e) for e in errors})})

    def table(key):
        groups = defaultdict(list)
        for t in trials:
            groups[key(t)].append(t["detected"])
        return {k: _rate(sum(v), len(v)) for k, v in sorted(groups.items())}

    # Counterfactual ablation: with guard G off, a trial stays detected only if some other
    # guard also fired. Computed from the recorded errors, not by editing production code.
    ablation = {}
    for tier in ("fabrication", "misattribution"):
        subset = [t for t in trials if t["tier"] == tier]
        row = {"all_guards": _rate(sum(t["detected"] for t in subset), len(subset))}
        for guard in CATEGORIES:
            still = sum(bool(set(t["categories"]) - {guard}) for t in subset)
            if still != row["all_guards"]["detected"]:
                row[f"without_{guard}"] = _rate(still, len(subset))
        ablation[tier] = row
    false_positives = [c for c in controls if c["errors"]]
    return {"version": 1, "seed": seed, "per_case": per_case, "mode": ctx.mode,
            "interpretation": "Detection of single-site injected faults by the production postchecks. "
                              "Fabrication faults refer to something absent from the evidence; "
                              "misattribution faults use real evidence in the wrong place. In legacy "
                              "mode a lexical check cannot see misattribution. In reviewed mode numbers "
                              "are bound to the statements a sentence cites, so part of it is visible.",
            "controls": {"cases": len(controls), "false_positives": len(false_positives),
                         "details": false_positives},
            "by_tier": table(lambda t: t["tier"]),
            "by_fault": table(lambda t: f"{t['tier']}/{t['fault']}"),
            "by_fault_and_target": table(lambda t: f"{t['tier']}/{t['fault']}/{t['target']}"),
            "ablation": ablation, "trials": trials}


def render_injection(report: dict) -> str:
    def cell(r):
        if not r["n"]:
            return "n/a"
        lo, hi = r["ci95"]
        return f"{r['detected']}/{r['n']} ({100 * r['rate']:.0f}%, 95% CI {100 * lo:.0f}-{100 * hi:.0f})"

    c = report["controls"]
    lines = ["# Guard reliability: fault injection", "",
             report["interpretation"], "",
             f"Guard mode: {report.get('mode', 'legacy')}. "
             f"Seed {report['seed']}, at most {report['per_case']} sites per fault and artifact. "
             f"Unmodified controls: {c['false_positives']} of {c['cases']} accepted artifacts rejected.", "",
             "## Detection by fault", "", "| tier | fault | detected |", "|---|---|---|"]
    for key, r in report["by_fault"].items():
        tier, fault = key.split("/")
        lines.append(f"| {tier} | {fault} | {cell(r)} |")
    lines += ["", "| tier | detected |", "|---|---|"]
    lines += [f"| {tier} | {cell(r)} |" for tier, r in report["by_tier"].items()]
    lines += ["", "## Where faults pass, by pipeline stage", "",
              "| tier | fault | stage | detected |", "|---|---|---|---|"]
    for key, r in report["by_fault_and_target"].items():
        tier, fault, target = key.split("/")
        lines.append(f"| {tier} | {fault} | {target} | {cell(r)} |")
    lines += ["", "## Ablation (one guard disabled at a time)", "",
              "Only guards whose removal changes the count are listed.", "",
              "| tier | configuration | detected |", "|---|---|---|"]
    for tier, row in report["ablation"].items():
        lines += [f"| {tier} | {name} | {cell(r)} |" for name, r in row.items()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Natural fault rate from persisted agent traces
# ---------------------------------------------------------------------------------------------
def natural_faults(paths) -> dict:
    """First-attempt rejection and recovery per agent, from `*.agents.jsonl` attempt logs.

    A call's first attempt is the output a harness without checks would have used. Calls whose
    first attempt never finished, and unparseable lines, are counted separately and excluded
    from the rates. Rejections are split by guard category; one attempt can fall in several.
    """
    calls: dict[str, dict] = {}
    corrupt = 0
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                try:
                    e = json.loads(line)
                    call = calls.setdefault(e["call_id"], {"agent": e["agent"], "attempts": {}})
                    if e["event"] == "finished":
                        call["attempts"][e["attempt"]] = e
                    elif e["event"] == "started":
                        call["attempts"].setdefault(e["attempt"], None)
                    else:
                        raise ValueError("unknown event")
                except (ValueError, KeyError, TypeError):
                    corrupt += 1

    def blank():
        return {"calls": 0, "first_attempt_rejected": 0, "recovered": 0, "unrecovered": 0,
                "first_attempt_unfinished": 0, "attempts": 0, "first_attempt_categories": Counter()}

    agents = defaultdict(blank)
    for call in calls.values():
        for row in (agents[call["agent"]], agents["ALL"]):
            first = call["attempts"].get(1)
            if first is None:
                row["first_attempt_unfinished"] += 1
                continue
            finished = [a for a in call["attempts"].values() if a is not None]
            row["calls"] += 1
            row["attempts"] += len(finished)
            if first["status"] == "accepted":
                continue
            row["first_attempt_rejected"] += 1
            for category in {classify(e) for e in first.get("errors", [])} or {"exception"}:
                row["first_attempt_categories"][category] += 1
            if any(a["status"] == "accepted" for a in finished):
                row["recovered"] += 1
            else:
                row["unrecovered"] += 1
    out = {}
    for name, row in sorted(agents.items()):
        n, rejected = row["calls"], row["first_attempt_rejected"]
        out[name] = {**row, "first_attempt_categories": dict(row["first_attempt_categories"]),
                     "first_attempt_rejection_rate": round(rejected / n, 4) if n else None,
                     "first_attempt_rejection_ci95": wilson(rejected, n),
                     "recovery_rate": round(row["recovered"] / rejected, 4) if rejected else None,
                     "mean_attempts": round(row["attempts"] / n, 3) if n else None}
    return {"version": 1, "corrupt_lines": corrupt,
            "interpretation": "A rejected first attempt would have been used by a harness without "
                              "checks. Rates describe these runs and this model only.",
            "agents": out}


def render_traces(report: dict) -> str:
    lines = ["# Guard reliability: natural fault rate", "", report["interpretation"], "",
             "| agent | calls | first attempt rejected | by guard | recovered | unrecovered | mean attempts |",
             "|---|---|---|---|---|---|---|"]
    for name, r in report["agents"].items():
        if not r["calls"]:
            continue
        cats = ", ".join(f"{k} {v}" for k, v in sorted(r["first_attempt_categories"].items())) or "none"
        lines.append(f"| {name} | {r['calls']} | {r['first_attempt_rejected']} "
                     f"({100 * r['first_attempt_rejection_rate']:.0f}%) | {cats} | {r['recovered']} | "
                     f"{r['unrecovered']} | {r['mean_attempts']} |")
    if report["corrupt_lines"]:
        lines += ["", f"{report['corrupt_lines']} unparseable trace line(s) were excluded."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Single-call baseline, audited post hoc by the same checkers
# ---------------------------------------------------------------------------------------------
BASELINE_SYSTEM = (
    "You write a short ML paper in Markdown from the evidence given: title, Abstract, Introduction, "
    "Related Work, Setup, Results, Discussion, Limitations, Conclusion. Cite evidence ids inline as "
    "[exp_0005] or [arXiv:...]. Use only numbers that appear in the evidence, and cite only the ids "
    "given. Do not add claims the evidence does not support.")


def audit_paper(text: str, bank) -> dict:
    """Apply the pipeline's deterministic checks to a finished paper without rejecting it."""
    from scholarloop.writing.session import _ID_RX, _SMALL_INTS, _YEARS, lint_prose
    checked = [t for t in _number_tokens(text) if float(t) not in _YEARS | _SMALL_INTS]
    ungrounded = bank.ungrounded(text)
    ids = _ID_RX.findall(text)
    unknown = bank.unknown(ids)
    return {"words": len(text.split()), "numbers_checked": len(checked),
            "ungrounded_numbers": ungrounded, "ids_cited": len(ids), "distinct_ids": len(set(ids)),
            "unknown_ids": unknown, "style_errors": lint_prose(text),
            "would_be_rejected": bool(ungrounded or unknown)}


def run_baseline(llm, bank, review_block: str, out_dir, *, repeats: int = 3) -> dict:
    """One model call per paper, same evidence block as the staged writer, no enforced check.

    Completed repeats are reused. A repeat that was started without a saved paper is kept as
    interrupted and never silently rerun, so failures stay in the denominator.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt = "\n".join(["Evidence:", bank.to_prompt_block(), "",
                        "Literature review (gaps and themes to position against):", review_block, "",
                        "Write the paper."])
    papers = []
    for i in range(1, repeats + 1):
        paper, marker = out / f"baseline_{i}.md", out / f"baseline_{i}.started"
        if not paper.exists():
            if marker.exists():
                papers.append({"repeat": i, "status": "interrupted"})
                continue
            marker.write_text("")
            paper.write_text(llm.complete(prompt, system=BASELINE_SYSTEM))
        papers.append({"repeat": i, "status": "complete", **audit_paper(paper.read_text(), bank)})
    done = [p for p in papers if p["status"] == "complete"]
    report = {"version": 1, "model": getattr(llm, "model", None), "repeats": repeats,
              "interpretation": "Single-call writer with the grounding rules stated in the prompt only. "
                                "Counts come from the pipeline's own checkers applied after the fact; "
                                "they measure traceability to the evidence, not scientific quality.",
              "papers": papers,
              "summary": {"complete": len(done),
                          "papers_with_violation": sum(p["would_be_rejected"] for p in done),
                          "ungrounded_numbers": sum(len(p["ungrounded_numbers"]) for p in done),
                          "numbers_checked": sum(p["numbers_checked"] for p in done),
                          "unknown_ids": sum(len(p["unknown_ids"]) for p in done),
                          "ids_cited": sum(p["ids_cited"] for p in done)}}
    (out / "baseline.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse

    from scholarloop.core.profile import load_profile

    ap = argparse.ArgumentParser(prog="scholarloop.reliability", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def run_args(p):
        p.add_argument("--profile", required=True)
        p.add_argument("--run", required=True, help="a captured run directory (experiments.jsonl, "
                       "registry/, lit_review.json, writing.json); override parts with the flags below")
        p.add_argument("--ledger")
        p.add_argument("--registry")
        p.add_argument("--research")
        p.add_argument("--writing")
        p.add_argument("--out", required=True)

    p = sub.add_parser("inject", help="seeded fault injection (free, deterministic)")
    run_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-case", type=int, default=8)
    p = sub.add_parser("traces", help="natural fault rate from *.agents.jsonl (free)")
    p.add_argument("--traces", nargs="+", required=True, help="trace files or directories")
    p.add_argument("--out", required=True)
    p = sub.add_parser("baseline", help="single-call writer, audited post hoc (USES THE API)")
    run_args(p)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--model")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.command == "traces":
        files = [f for t in a.traces for f in
                 (sorted(Path(t).rglob("*.agents.jsonl")) if Path(t).is_dir() else [Path(t)])]
        if not files:
            print("no *.agents.jsonl files found", file=sys.stderr)
            return 1
        report = {**natural_faults(files), "files": [str(f) for f in files]}
        (out / "traces.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        (out / "traces.md").write_text(render_traces(report))
        print(render_traces(report))
        return 0

    run = Path(a.run)
    paths = {"ledger": a.ledger or run / "experiments.jsonl", "registry": a.registry or run / "registry",
             "research": a.research or run, "writing": a.writing or run / "writing.json"}
    profile = load_profile(a.profile)
    cases, ctx = load_cases(profile, **paths)

    if a.command == "inject":
        report = run_injection(cases, ctx, seed=a.seed, per_case=a.per_case)
        (out / "injection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        (out / "injection.md").write_text(render_injection(report))
        print(render_injection(report))
        return 0

    import os

    from scholarloop.core.call_budget import CallLimitedLLM
    from scholarloop.core.llm import AnthropicLLM, model_from_env
    from scholarloop.ideation.litreview import LitReview
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 1
    llm = CallLimitedLLM(AnthropicLLM(model=a.model or model_from_env()), a.repeats,
                         path=out / "baseline_call_budget.json")
    review = LitReview.from_dict(json.loads((Path(paths["research"]) / "lit_review.json").read_text()))
    # The baseline sees the raw evidence and is audited against all of it (does each number
    # trace to some measurement, does each cited id exist), whatever mode the staged writer used.
    ctx.bank.approved_dossier = None
    report = run_baseline(llm, ctx.bank, review.to_prompt_block(), out, repeats=a.repeats)
    state = json.loads(Path(paths["writing"]).read_text())
    if state.get("sections"):
        staged = "\n\n".join(s["body"] for s in state["sections"])
        report["staged_pipeline_paper"] = audit_paper(staged, ctx.bank)
        (out / "baseline.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("summary", "staged_pipeline_paper") if k in report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
