"""Step-by-step writing demo: outline → panel discussion → author sign-off → key points → panel
discussion → author sign-off → section-by-section drafting → reviewer panel.

Deterministic and free: every agent is a scripted `MockLLM`, the author is a scripted input, and
the evidence is a small captured ledger on the stub profile. Run from the repo root:

    python examples/writing_demo.py

With a real key the same flow runs on a real campaign (examples/run_to_paper.py ends in it; the
directory it wrote can be resumed or revised with the CLI):
    python -m scholarloop.writing --profile profiles/digits-mlp.yaml \
        --ledger runs/s1/experiments.jsonl --registry runs/s1/registry \
        --research runs/s1 --out runs/s1
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from scholarloop.core.ledger import Hypothesis, Ledger, LedgerEntry
from scholarloop.core.llm import MockLLM
from scholarloop.core.profile import load_profile
from scholarloop.core.registry import VerifiedRegistry
from scholarloop.writing.session import WritingSession

ROOT = Path(__file__).resolve().parent.parent

RUNS = [("exp_0001", "smoke", 4.2222, [4.2222], "kept", {"lr": 0.1, "depth": 20, "warmup": 5}),
        ("exp_0002", "smoke", 7.3333, [7.3333], "discarded", {"lr": 0.05, "depth": 18, "warmup": 0}),
        ("exp_0003", "verify", 3.9259, [3.9, 3.95, 3.93], "kept", {"lr": 0.1, "depth": 20, "warmup": 5}),
        ("exp_0004", "full", 3.8222, [3.8, 3.86, 3.79, 3.84, 3.82], "kept", {"lr": 0.1, "depth": 20, "warmup": 5})]

OUTLINE_V1 = {
    "title": "Warmup, not depth, closes the gap on digits",
    "question": "Does a short learning-rate warmup let a small MLP beat the 4.9 baseline on digits?",
    "nugget": "On a two-minute budget the schedule, not the architecture, decides whether the MLP converges.",
    "contributions": ["a 5-seed confirmed drop from the 4.9 baseline to 3.8222 val_top1_err with warmup=5 [exp_0004]",
                      "a cheap negative: the no-warmup recipe dies at 7.3333 in the smoke screen [exp_0002]"],
    "sections": [{"heading": "Abstract", "purpose": "the five-sentence summary", "evidence": ["exp_0004", "baseline"]},
                 {"heading": "Setup", "purpose": "everything needed to reproduce", "evidence": ["exp_0004"]},
                 {"heading": "Results", "purpose": "the climb and the failure", "evidence": ["exp_0001", "exp_0002", "exp_0003", "exp_0004"]},
                 {"heading": "Limitations", "purpose": "scope of the claim", "evidence": []}]}
OUTLINE_V2 = {**OUTLINE_V1, "contributions": OUTLINE_V1["contributions"] + ["the spread across seeds at each tier, so the gain is read against noise [exp_0003, exp_0004]"]}
OUTLINE_V3 = {**OUTLINE_V2, "title": "A five-epoch warmup closes the gap on digits"}

SKEPTIC_R1 = {"verdict": "revise", "issues": [{"target": "Results", "issue": "no plan to report the spread across seeds",
                                                "suggestion": "add the std at verify and full as a contribution and a Results point"}]}
ACCEPT = {"verdict": "accept", "issues": []}

KEY_POINTS = {"sections": [
    {"heading": "Abstract", "points": [
        {"claim": "We ask whether a five-epoch warmup lets a small MLP beat the 4.9 baseline on digits under a two-minute budget.", "evidence": ["baseline"], "role": "context"},
        {"claim": "With warmup=5 and lr 0.1 the full 5-seed run reaches 3.8222 val_top1_err.", "evidence": ["exp_0004"], "role": "result"}]},
    {"heading": "Setup", "points": [
        {"claim": "An MLP of depth 20 trained with lr 0.1 and warmup 5; the frozen scorer computes val_top1_err on a held-out split.", "evidence": ["exp_0004"], "role": "method"},
        {"claim": "Each idea climbs smoke (1 seed), verify (3 seeds) and full (5 seeds); a tier must beat the gate to continue.", "evidence": ["exp_0001", "exp_0003", "exp_0004"], "role": "method"}]},
    {"heading": "Results", "points": [
        {"claim": "The warmup recipe scores 4.2222 at smoke, 3.9259 at verify and 3.8222 at full, each below the 4.9 baseline.", "evidence": ["exp_0001", "exp_0003", "exp_0004", "baseline"], "role": "result"},
        {"claim": "The no-warmup recipe at lr 0.05 scores 7.3333 at smoke and is discarded without a verify run.", "evidence": ["exp_0002"], "role": "comparison"},
        {"claim": "The seed spread stays at 0.03 std across the five full-run seeds.", "evidence": ["exp_0004"], "role": "result"}]},
    {"heading": "Limitations", "points": [
        {"claim": "One dataset, one budget, and a validation split that is reconstructable from public data.", "evidence": [], "role": "limitation"}]}]}

BODIES = {
    "Abstract": ("We ask whether a five-epoch learning-rate warmup lets a small MLP beat the 4.9 val_top1_err "
                 "baseline on digits under a two-minute training budget. Small nets on short budgets often fail "
                 "to converge, and the usual response is more depth. We instead keep the depth at 20 and add a "
                 "warmup of 5 epochs at lr 0.1. The recipe climbs a three-tier funnel of 1, 3 and 5 seeds. The "
                 "full run reaches 3.8222 val_top1_err [exp_0004]."),
    "Setup": ("We train an MLP of depth 20 with lr 0.1 and a warmup of 5 epochs, and a frozen scorer computes "
              "val_top1_err on a held-out split the training script never sees [exp_0004]. Each idea climbs a "
              "funnel of three tiers: smoke with 1 seed, verify with 3 seeds and full with 5 seeds, and a tier "
              "must beat the gate to earn the next one [exp_0001, exp_0003, exp_0004]."),
    "Results": ("The warmup recipe scores 4.2222 at smoke, 3.9259 at verify and 3.8222 at full, each below the "
                "4.9 baseline [exp_0001, exp_0003, exp_0004]. The seed spread stays at 0.03 std across the five "
                "full-run seeds, so the gap to the baseline is many times the noise [exp_0004]. The no-warmup "
                "recipe at lr 0.05 scores 7.3333 at smoke and is discarded before any verify run [exp_0002]."),
    "Limitations": ("The claim holds for one dataset and one two-minute budget. The validation split is "
                    "reconstructable from public data, so a hostile training script could overfit it; the "
                    "frozen scorer defends against the realistic case, not that one."),
}

REVIEWS = [
    {"summary": "A clean, small, honest result.", "strengths": ["5-seed confirmation with the spread reported"],
     "weaknesses": ["a single dataset and budget"], "questions": ["does the effect hold at depth 2?"], "score": 5, "recommendation": "weak_accept"},
    {"summary": "Marginal contribution; warmup is well known.", "strengths": ["the negative result is reported"],
     "weaknesses": ["a single dataset and budget", "no comparison to a cosine schedule"], "questions": ["why 5 epochs?"], "score": 3, "recommendation": "weak_reject"},
    {"summary": "Modest but the write-up is traceable.", "strengths": ["every number cites a run id"],
     "weaknesses": ["the nugget overreaches for one setting"], "questions": [], "score": 5, "recommendation": "weak_accept"},
]


def main() -> int:
    profile = load_profile(ROOT / "profiles" / "image-classification.yaml")
    tmp = Path(tempfile.mkdtemp(prefix="writing_"))
    led, reg = tmp / "ledger.jsonl", tmp / "registry"
    for eid, fid, score, seeds, verdict, cfg in RUNS:
        Ledger(led).append(LedgerEntry(id=eid, domain=profile.name, metric_name="val_top1_err",
                                       hypothesis=Hypothesis(claim="warmup lets a higher lr converge", source="arXiv:1706.02677v1"),
                                       config=cfg, fidelity=[fid], metric={fid: score, "seeds": seeds}, verdict=verdict, registry_id=eid))
        r = VerifiedRegistry(reg / f"{eid}.json", exp_id=eid)
        r.capture(f"val_top1_err.{fid}", score, seeds=seeds, std=round(0.03, 4) if len(seeds) == 5 else (0.02 if len(seeds) == 3 else 0.0))
        r.save()

    # Scripted agents, in call order: outliner → panel ×3 → (outliner → panel ×3) → ... → key pointer →
    # panel ×3 → drafter ×4 → reviewers ×3. The Skeptic forces one revision; the author then asks for a
    # sharper title, which re-runs the discussion.
    llm = MockLLM(jsons=[
        OUTLINE_V1, SKEPTIC_R1, ACCEPT, ACCEPT,            # round 1: Skeptic wants the spread
        OUTLINE_V2, ACCEPT, ACCEPT, ACCEPT,                # round 2: accepted
        OUTLINE_V3, ACCEPT, ACCEPT, ACCEPT,                # author feedback: new title, accepted
        KEY_POINTS, ACCEPT, ACCEPT, ACCEPT,
        *({"body": BODIES[h]} for h in ("Abstract", "Setup", "Results", "Limitations")),
        *REVIEWS,
    ])
    author = iter(["r put the five-epoch warmup in the title", "a", "a"])

    print("=== STEP-BY-STEP WRITING · image-classification (captured ledger) ===\n")
    s = WritingSession(llm, profile, tmp / "paper", ledger_path=led, registry_dir=reg, demo=True)

    def ask(prompt):
        a = next(author)
        print(f"{prompt}{a}")
        return a

    s.interactive(input_fn=ask, print_fn=lambda line: print(f"  {line}"))

    print("\n  outline discussion:")
    for t in s.state.outline_discussion:
        votes = ", ".join(f"{r['role']}={r['verdict']}" for r in t["reviews"])
        print(f"    round {t['round']}: {votes}" + ("" if t["accepted"] else f"  → revised ({t['reviews'][0]['issues'][0]['issue']})"))
    print(f"  author feedback: {[f['text'] for f in s.state.feedback]}")
    print(f"  final title: {s.state.outline['title']}")
    print(f"  sections drafted: {[sec['heading'] for sec in s.state.sections]}  ·  "
          f"drafting notes: {sum(len(sec['warnings']) for sec in s.state.sections)}")
    r = s.state.review
    print(f"  review: {r['recommendation']} · mean {r['mean_score']}/10 · weaknesses: {len(r['weaknesses'])}")
    print(f"\n  artifacts: {tmp / 'paper'}  (outline.md · key_points.md · paper.md · review.md · writing.json)")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
