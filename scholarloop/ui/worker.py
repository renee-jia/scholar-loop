"""Adapters for stage-specific research actions and resumable experiment campaigns."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def campaign(request):
    from scholarloop.core.call_budget import CallLimitedLLM
    from scholarloop.core.ledger import Ledger
    from scholarloop.core.llm import AnthropicLLM, MockLLM
    from scholarloop.core.profile import load_profile
    from scholarloop.core.training_budget import TrainingBudget
    from scholarloop.experiment.orchestrator import Orchestrator
    from scholarloop.experiment.governor import Governor
    from scholarloop.review.contracts import atomic_json

    run, workspace = Path(request["run"]), Path(request["workspace"])
    config, p = request["config"], request["params"]
    profile = load_profile(workspace / "profiles" / config["profile"])
    ledger = run / "experiments.jsonl"
    if request["action"] == "demo":
        if config["mode"] != "demo" or profile.name != "image-classification":
            raise ValueError("Offline demo requires its own stub-engine research")
        # Explicitly synthetic and never mixed into a live campaign.
        attempt = sum(1 for _ in Ledger(ledger).read_all())
        proposal = {"rationale": "Scripted offline demonstration, not scientific evidence",
                    "config": [{"name": "lr", "value": max(0.06, 0.1 - attempt * 0.001)},
                               {"name": "depth", "value": 20}, {"name": "weight_decay", "value": 0.0005},
                               {"name": "warmup", "value": 5}],
                    "hypothesis": {"claim": "Demonstrate the three-tier execution funnel",
                                   "source": "demo:scripted", "predicted_effect": "lower stub error"},
                    "predicted_delta": -1.0}
        llm = MockLLM(jsons=[proposal])
        orch = Orchestrator(llm, profile, ledger_path=ledger, registry_dir=run / "registry")
        print("OFFLINE DEMO — scripted model and synthetic scores. No API calls. Not research evidence.", flush=True)
        entries = orch.run(n_steps=1, funnel=True)
        (run / "demo.md").write_text("# Offline demonstration\n\nScripted model and synthetic engine scores. "
                                     "These results are not scientific evidence.\n", encoding="utf-8")
    else:
        from scholarloop.experiment.advisor import Advisor
        from scholarloop.experiment.debate import DebatePanel
        from scholarloop.experiment.director import Director
        from scholarloop.experiment.reflector import Reflector
        from scholarloop.experiment.skills import SkillLibrary
        if config["mode"] == "demo":
            raise ValueError("Create a live research project to run real experiments")
        handoff = run / "handoff.json"
        if not handoff.exists():
            raise ValueError("Select a research idea on the Ideas page first")
        llm = CallLimitedLLM(AnthropicLLM(model=config["model"]), config["max_calls"],
                             path=run / "campaign_call_budget.json")
        orch = Orchestrator(llm, profile, ledger_path=ledger, registry_dir=run / "registry",
                            debate_panel=DebatePanel(llm, profile.debate_roles), reflector=Reflector(llm),
                            advisor=Advisor(llm), director=Director(llm, profile),
                            skill_library=SkillLibrary(run / "skills"), calibration_path=run / "calibration.json")
        orch.adopt(json.loads(handoff.read_text(encoding="utf-8")))
        budget_path = ledger.with_name(ledger.name + ".training_budget.json")
        from scholarloop.experiment.compare import train_runs
        used = train_runs(list(Ledger(ledger).read_all()))
        orch.training_budget = TrainingBudget(budget_path, p["training_limit"], initial_used=used)
        print(f"Starting up to {p['rounds']} rounds; prior evidence is retained.", flush=True)
        try:
            entries = orch.run(n_steps=p["rounds"], funnel=True, population=p["population"],
                               max_workers=p["workers"], governor=Governor(max_rounds=p["rounds"], dry_patience=p["patience"]))
        finally:
            # Per-invocation usage is separate from the cross-stage spend file.
            atomic_json(run / "ui_campaign_usage.json", dict(llm.usage))
    for e in entries:
        print(f"{e.id}  {'/'.join(e.fidelity)}  {e.metric_name}={e.primary_score()}  {e.verdict}", flush=True)
    print(f"Completed: {len(entries)} new ledger records.", flush=True)
    return 0


def research(request):
    from scholarloop.core.call_budget import CallLimitedLLM
    from scholarloop.core.llm import AnthropicLLM
    from scholarloop.core.profile import load_profile
    from scholarloop.ideation.session import ResearchSession
    from scholarloop.ideation.tournament import Tournament
    from scholarloop.ideation.metareview import MetaReviewer
    from scholarloop.experiment.skills import SkillLibrary

    run, workspace = Path(request["run"]), Path(request["workspace"])
    p, config, action = request["params"], request["config"], request["action"]
    profile = load_profile(workspace / "profiles" / config["profile"])
    llm = CallLimitedLLM(AnthropicLLM(model=config["model"]), config["max_calls"],
                         path=run / "ideation_call_budget.json")
    session = ResearchSession(llm, profile, run, topic=config["title"],
                              ledger_path=run / "experiments.jsonl", require_direction=True,
                              max_papers=p.get("max_papers", 20), deep_ideation=p.get("deep", False),
                              tournament=Tournament(llm, profile), meta_reviewer=MetaReviewer(llm),
                              skills=SkillLibrary(run / "skills"))
    feedback = p.get("feedback", "")
    if action == "literature":
        session.set_steer({"interest": p["interest"],
                           "hunches": [s.strip() for s in p["hunches"].splitlines() if s.strip()],
                           "avoid": [s.strip() for s in p["avoid"].splitlines() if s.strip()]}, actor="local-user")
        if p["brief"]:
            session.set_brief(p["brief"], actor="local-user")
        session.review(feedback=feedback)
    elif action == "directions":
        session.map_directions(feedback=feedback)
    else:
        if action == "ideas":
            own = [s.strip() for s in p["own_direction"].splitlines() if s.strip()]
            picks = [int(s) for s in p["directions"].replace(",", " ").split()]
            if picks or own:
                session.choose_directions(picks, actor="local-user", note=p["direction_note"], own=own)
            elif not session.chosen:
                raise ValueError("Select at least one direction or enter your own")
        session.ideate(p["n_ideas"], feedback=feedback)
    print(f"Research stage: {session.stage}. Artifacts saved to {run}", flush=True)
    return 0


def main(request):
    return campaign(request) if request["action"] in {"campaign", "demo"} else research(request)


if __name__ == "__main__":
    try:
        raise SystemExit(main(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
