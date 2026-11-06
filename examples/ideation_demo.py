"""Ideation-first demo: literature review → candidate ideas → a human picks → the loop runs it.

Deterministic and free — the agents are scripted by `MockLLM`, the "human" is a scripted input
function, retrieval reads a canned arXiv feed, and the experiment runs on the dependency-free stub
engine — so the whole shape of a human-in-the-loop research session shows in under a second:

  [1/3]  lit_review.md   the human reads it and asks for one change (feedback → re-review)
  [2/3]  ideas.md        the human picks idea 2
  [3/3]  handoff.json    the Orchestrator adopts it and climbs smoke → verify → full

With a real key the same flow runs on real literature and a real LLM:
    python -m scholarloop.session --profile profiles/digits-mlp.yaml --topic "..." --out runs/s1

Run from the repo root:  python examples/ideation_demo.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from scholarloop.litscout import ArxivClient
from scholarloop.llm import MockLLM
from scholarloop.orchestrator import Orchestrator
from scholarloop.profile import load_profile
from scholarloop.session import ResearchSession

ROOT = Path(__file__).resolve().parent.parent

ATOM = """<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/1512.03385v1</id><title>Deep Residual Learning for Image Recognition</title>
    <summary>Residual connections let very deep networks train; identity shortcuts ease optimization.</summary>
    <published>2015-12-10T00:00:00Z</published></entry>
  <entry><id>http://arxiv.org/abs/1608.03983v1</id><title>SGDR: Stochastic Gradient Descent with Warm Restarts</title>
    <summary>A cosine annealing schedule with warm restarts improves convergence of SGD.</summary>
    <published>2016-08-13T00:00:00Z</published></entry>
  <entry><id>http://arxiv.org/abs/1706.02677v1</id><title>Accurate, Large Minibatch SGD</title>
    <summary>Linear scaling rule with a learning-rate warmup phase stabilizes large-batch training.</summary>
    <published>2017-06-08T00:00:00Z</published></entry>
</feed>"""
MIXUP = """<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/1710.09412v1</id><title>mixup: Beyond Empirical Risk Minimization</title>
    <summary>Training on convex combinations of pairs of examples and labels regularizes networks.</summary>
    <published>2017-10-25T00:00:00Z</published></entry>
</feed>"""

R, S, W, M = "arXiv:1512.03385v1", "arXiv:1608.03983v1", "arXiv:1706.02677v1", "arXiv:1710.09412v1"

REVIEW_1 = {
    "overview": "Three lines of work matter for small supervised nets: residual shortcuts, cosine "
                "schedules, and warmup. Each is studied at scale; their interaction on small models is not.",
    "themes": [{"name": "Architecture: shortcuts", "summary": "Identity shortcuts make depth trainable.", "papers": [R]},
               {"name": "Optimization: schedules", "summary": "Cosine decay and warmup shape SGD's trajectory.", "papers": [S, W]}],
    "paper_notes": [{"id": R, "contribution": "residual blocks; depth 18-152", "limitation": "ImageNet scale only"},
                    {"id": S, "contribution": "cosine annealing + warm restarts", "limitation": "restarts unexplored for short runs"},
                    {"id": W, "contribution": "LR warmup for large batches", "limitation": "framed as a batch-size fix"}],
    "gaps": [{"gap": "Warmup is only studied as a large-batch fix; its effect on small nets with cosine decay is open.",
              "why_open": "SGDR and the warmup paper never combine the two", "sources": [S, W]},
             {"gap": "Depth/shortcut trade-offs below 20 layers are unreported.", "why_open": "ResNet starts at 18", "sources": [R]}],
    "trends": ["schedules matter as much as architecture at small scale"],
}
REVIEW_2 = {**REVIEW_1,
            "overview": REVIEW_1["overview"] + " Data-side regularization (mixup) is a fourth, orthogonal lever.",
            "themes": REVIEW_1["themes"] + [{"name": "Regularization: data", "summary": "mixup regularizes by interpolating pairs.", "papers": [M]}],
            "paper_notes": REVIEW_1["paper_notes"] + [{"id": M, "contribution": "interpolated training pairs", "limitation": "interacts with schedules in unknown ways"}],
            "gaps": REVIEW_1["gaps"] + [{"gap": "mixup under a cosine+warmup schedule is untested.", "why_open": "no joint study", "sources": [M, S]}]}

IDEAS = {"ideas": [
    {"title": "Residual shortcuts in a 20-layer net", "description": "Add identity shortcuts to a mid-depth net and re-tune depth.",
     "gap": "shortcut trade-offs below 20 layers", "sources": [R],
     "hypothesis": {"claim": "shortcuts let depth 20 beat depth 18", "predicted_effect": "-0.5 err"},
     "experiment_plan": "depth=20 vs 18 at lr=0.05, same schedule", "novelty_rationale": "ResNet reports 18+ only",
     "feasibility": 4, "impact": 2},
    {"title": "Warmup + cosine on a small net", "description": "Combine a 5-epoch warmup with cosine decay and a higher peak lr.",
     "gap": "warmup x cosine on small nets", "sources": [S, W],
     "hypothesis": {"claim": "warmup lets a higher peak lr converge better", "predicted_effect": "-1.0 err"},
     "experiment_plan": "lr=0.1, warmup=5, weight_decay=5e-4, depth=20; compare to the no-warmup baseline",
     "novelty_rationale": "the two papers never combine them", "feasibility": 5, "impact": 4},
    {"title": "mixup under cosine+warmup", "description": "Apply mixup while training with the warmup+cosine recipe.",
     "gap": "mixup x schedule", "sources": [M, S],
     "hypothesis": {"claim": "mixup adds a further gain on top of the schedule", "predicted_effect": "-0.3 err"},
     "experiment_plan": "requires an augmentation edit in train.py", "novelty_rationale": "no joint study",
     "feasibility": 3, "impact": 3},
]}

PROPOSAL = {"reasoning_trace": "follow the chosen plan: warmup=5 with lr=0.1 and cosine-style decay",
            "config": [{"name": "lr", "value": 0.1}, {"name": "depth", "value": 20},
                       {"name": "weight_decay", "value": 5e-4}, {"name": "warmup", "value": 5}],
            "hypothesis": {"claim": "warmup lets a higher peak lr converge better", "source": W,
                           "predicted_effect": "-1.0 err"},
            "predicted_delta": -1.0}


def main() -> int:
    profile = load_profile(ROOT / "profiles" / "image-classification.yaml")   # stub engine: instant
    tmp = Path(tempfile.mkdtemp(prefix="ideation_"))
    feeds = iter([ATOM, ATOM, MIXUP])                        # 2 planned queries, then 1 feedback query
    session_llm = MockLLM(jsons=[
        {"queries": ["residual networks small models", "cosine schedule warmup"], "rationale": "arch + optim"},
        REVIEW_1,
        {"queries": ["mixup data augmentation"], "rationale": "the reader asked for data-side regularization"},
        REVIEW_2,
        IDEAS,
    ])
    human = iter(["r also cover data-side regularization such as mixup", "a", "s 2"])

    print("=== IDEATION-FIRST SESSION · image-classification (stub engine) ===\n")
    s = ResearchSession(session_llm, profile, tmp / "session", topic="training recipes for small nets",
                        sources=[ArxivClient(fetcher=lambda q, n: next(feeds))])

    def ask(prompt):
        answer = next(human)
        print(f"{prompt}{answer}")
        return answer

    handoff = s.interactive(n_ideas=3, input_fn=ask, print_fn=lambda line: print(f"  {line}"))
    idea = handoff["idea"]
    print(f"\n  chosen: {idea['title']}  ·  builds on {', '.join(idea['sources'])}")
    print(f"  corpus grew to {len(s.review_obj.papers)} papers after the feedback round; "
          f"{len(s.state.feedback)} feedback note(s) recorded\n")

    print("=== EXPERIMENT LOOP · adopts the handoff (no Director / Lit Scout needed) ===\n")
    orch = Orchestrator(MockLLM(jsons=[PROPOSAL]), profile,
                        ledger_path=tmp / "ledger.jsonl", registry_dir=tmp / "registry")
    orch.adopt(handoff)
    produced = orch.run(funnel=True)
    print(f"  topic: {orch.topic}\n  baseline to beat: {profile.best_baseline()}%")
    for e in produced:
        print(f"    {e.fidelity[0]:<6} {e.metric_name}={e.primary_score()}%  [{e.verdict}]")
    print(f"\n  the idea climbed {len(produced)} tier(s); every run cites {produced[0].hypothesis.source}.")
    print(f"  session artifacts: {tmp / 'session'}  (lit_review.md · ideas.md · handoff.json · session.json)")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
