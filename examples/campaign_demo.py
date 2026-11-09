"""A full ScholarLoop campaign on the REAL torch digits-mlp engine, MockLLM-scripted.

Exercises the whole agent chain end-to-end on real experiments:
  Director -> LitScout -> Reasoner -> DebatePanel -> multi-fidelity funnel -> Reflector -> Advisor

The LLM calls are scripted (deterministic, no API) but every experiment is a real torch run.
The script narrates each idea: the Reasoner's config, the debate decision, the funnel tiers
(with real val errors), the reflected lesson, and the advisor's loop-control decision.

Run from the repo root:  python examples/campaign_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from scholarloop.experiment.advisor import Advisor
from scholarloop.experiment.debate import DebatePanel
from scholarloop.experiment.director import Director
from scholarloop.core.ledger import Ledger
from scholarloop.ideation.litscout import ArxivClient, LitScout
from scholarloop.core.llm import MockLLM
from scholarloop.experiment.orchestrator import Orchestrator
from scholarloop.core.profile import load_profile
from scholarloop.experiment.reflector import Reflector
from scholarloop.experiment.skills import SkillLibrary

ROOT = Path(__file__).resolve().parent.parent

# Canned arXiv feed (real network is unavailable in this sandbox; the fetcher is injectable).
ATOM = """<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/1512.03385v1</id><title>Deep Residual Learning</title>
    <summary>Wider/deeper networks with residual connections ease optimization.</summary></entry>
  <entry><id>http://arxiv.org/abs/1608.03983v1</id><title>SGDR: Warm Restarts</title>
    <summary>Cosine schedules with SGD momentum improve convergence.</summary></entry>
</feed>"""


def _proposal(cfg, source, predicted_delta):
    return {"reasoning_trace": f"try {cfg} (grounded in {source})",
            "config": [{"name": k, "value": v} for k, v in cfg.items()],
            "hypothesis": {"claim": "tuned MLP recipe", "source": source,
                           "predicted_effect": "lower val error"},
            "predicted_delta": predicted_delta}


def main() -> None:
    profile = load_profile(ROOT / "profiles" / "digits-mlp.yaml")
    tmp = Path(tempfile.mkdtemp(prefix="campaign_"))

    # --- per-agent scripted LLMs (each agent gets its own) ---
    reasoner_llm = MockLLM(jsons=[
        _proposal({"hidden": 128, "depth": 2, "lr": 0.2, "epochs": 40}, "arXiv:1512.03385", -1.5),
        _proposal({"hidden": 64, "depth": 1, "lr": 0.1, "epochs": 30}, "arXiv:1608.03983", -0.3),
        _proposal({"hidden": 8, "depth": 1, "lr": 0.5, "epochs": 10}, "arXiv:1512.03385", -1.0),
    ])
    director = Director(MockLLM(jsons=[
        {"direction": "scale width/depth and tune the optimizer", "topic": "mlp digit classification",
         "rationale": "the baseline MLP underfits; capacity + lr have headroom"},
        {"direction": "abandon tiny under-capacity models; revisit regularization",
         "topic": "regularization small mlp", "rationale": "tiny models collapsed"},
    ]), profile)
    lit_scout = LitScout(MockLLM(jsons=[{"findings": [
        {"technique": "wider/deeper layers", "source": "arXiv:1512.03385",
         "predicted_effect": "lower error", "rationale": "more capacity"},
        {"technique": "SGD momentum + cosine", "source": "arXiv:1608.03983",
         "predicted_effect": "faster convergence", "rationale": "warm restarts"}]}]),
        ArxivClient(fetcher=lambda q, n: ATOM))
    debate = DebatePanel(MockLLM(jsons=[
        {"verdict": "run", "concern": ""}, {"verdict": "run", "concern": "watch overfit"},
        {"verdict": "run", "concern": ""},                                   # idea 1 -> run
        {"verdict": "reject", "concern": "redundant with baseline capacity"},
        {"verdict": "reject", "concern": "low odds of improvement"},
        {"verdict": "run", "concern": ""},                                   # idea 2 -> REJECT
        {"verdict": "run", "concern": "very small"}, {"verdict": "run", "concern": ""},
        {"verdict": "run", "concern": ""},                                   # idea 3 -> run
    ]), profile.debate_roles)
    reflector = Reflector(MockLLM(jsons=[
        {"worth_recording": True, "category": "capacity", "severity": 0.9,
         "mitigation": "wider+deeper MLP (hidden>=128, depth>=2) clears the baseline"},
        {"worth_recording": True, "category": "capacity", "severity": 0.8,
         "mitigation": "tiny hidden width (<=8) underfits badly — avoid"}]))
    advisor = Advisor(MockLLM(jsons=[
        {"decision": "proceed", "rationale": "new frontier set; keep scaling"},
        {"decision": "pivot", "rationale": "tiny models are a dead end; change direction"}]))
    skills = SkillLibrary(tmp / "skills")

    orch = Orchestrator(reasoner_llm, profile, lit_scout=lit_scout, debate_panel=debate,
                        reflector=reflector, advisor=advisor, director=director,
                        skill_library=skills, ledger_path=tmp / "ledger.jsonl",
                        registry_dir=tmp / "registry")

    print(f"\n=== CAMPAIGN: {profile.name} (real torch) — baseline {profile.best_baseline()}% ===")

    # L4 Director sets the campaign direction (mirrors run()'s first step, but narrated)
    d = orch.director.direct([])
    orch.topic, orch.guidance, orch._directed = d["topic"], d["direction"], True
    print(f"[Director] direction: {d['direction']}\n           topic for Lit Scout: {d['topic']}")
    lc, lp = orch._literature()
    print(f"[LitScout] grounded priors: {lp}")

    for i in range(1, 4):
        print(f"\n--- idea {i} ---")
        produced = orch.funnel_step()
        if not produced:
            critics = [c for c in orch.trace.calls if c.agent.startswith("critic:")][-3:]
            votes = [f"{c.agent.split(':')[1]}={c.output['verdict']}" for c in critics]
            print(f"[Reasoner] proposed an idea\n[Debate] REJECTED {votes} -> skipped, no GPU spent")
            continue
        first = produced[0]
        print(f"[Reasoner] config={first.config}  predicted_delta={first.prediction['predicted']}")
        print(f"[Debate]   decision={first.reasoning['debate']['decision']}")
        tiers = " -> ".join(f"{e.fidelity[0]} {e.primary_score()}% [{e.verdict}]" for e in produced)
        print(f"[Funnel]   {tiers}")
        cal = produced[0].prediction.get("calibration_error")   # smoke tier: idea-vs-frontier prediction
        print(f"[Reflect]  (skill library now holds {len(skills.all())} lessons)")
        print(f"[Advisor]  {orch.last_advice['decision']} — {orch.last_advice['rationale']}"
              + (f"  | predict-vs-measured calib_err={cal}" if cal is not None else ""))

    print("\n=== LEDGER (real torch results) ===")
    for e in Ledger(tmp / "ledger.jsonl").read_all():
        print(f"  {e.id}  {e.fidelity[0]:<6} {e.metric_name}={e.primary_score()}%  {e.verdict:<10}"
              f" parent={e.parent}")
    print("\n=== SKILL LIBRARY (accumulated intuition) ===")
    print(skills.render() or "  (empty)")
    print("\n=== AGENT TRACE (every call, auditable) ===")
    from collections import Counter
    counts = Counter(c.agent for c in orch.trace.calls)
    print("  " + ", ".join(f"{a}:{n}" for a, n in sorted(counts.items())))

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
