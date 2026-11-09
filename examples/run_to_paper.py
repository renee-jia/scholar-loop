"""Run a COMPLETE ScholarLoop flow with a real LLM — campaign + write-up — and save the artifacts
to one directory (default runs/live/): the real experiments (experiments.jsonl + registry/), a
narrative run log (run.md), and the paper.

The write-up is the step-by-step WritingSession (outline → key points → draft → review): when stdin
is a terminal it asks you to approve or refine the outline and the key points; otherwise it runs
unattended. Its artifacts land next to the run log: outline.md, key_points.md, paper.md, review.md,
writing.json (resume or revise later with `python -m scholarloop.writing --out <dir> ...`).

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/run_to_paper.py
    # start from a human-chosen idea (python -m scholarloop.ideation --out runs/s1 ... -> handoff.json):
    SCHOLARLOOP_HANDOFF=runs/s1/handoff.json SCHOLARLOOP_OUT=runs/s1 python examples/run_to_paper.py

Knobs (environment):
    SCHOLARLOOP_OUT=runs/live            output directory (the shipped examples/sample_runs/* are captures;
                                         point here to overwrite one on purpose)
    SCHOLARLOOP_PROFILE=digits-mlp       profiles/<name>.yaml
    SCHOLARLOOP_MODEL=claude-opus-5-5    SCHOLARLOOP_TOPIC="..."   SCHOLARLOOP_STEPS=3
    SCHOLARLOOP_POPULATION=1             >1: parallel population funnel, with SCHOLARLOOP_MAX_WORKERS
    SCHOLARLOOP_DRY_PATIENCE=2           loop-until-dry convergence (Governor)
    SCHOLARLOOP_MAX_COST=1.0             USD ceiling (Governor; needs a priced model)
    SCHOLARLOOP_WRITING=session|auto|oneshot   session = interactive if a tty else auto;
                                         oneshot = the single-call Writer + Reviewer (paper.py)
    SCHOLARLOOP_PAPER_ONLY=1             skip the campaign; write up the saved ledger in SCHOLARLOOP_OUT
    SCHOLARLOOP_PERSIST_SKILLS=1         ~/.scholarloop/skills/<domain>, so lessons compound across runs
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from scholarloop.experiment.advisor import Advisor
from scholarloop.experiment.debate import DebatePanel
from scholarloop.experiment.director import Director
from scholarloop.experiment.governor import cost_of
from scholarloop.core.ledger import Ledger
from scholarloop.ideation.litscout import ArxivClient, LitScout, OpenAlexClient
from scholarloop.core.llm import AnthropicLLM, model_from_env
from scholarloop.experiment.orchestrator import Orchestrator
from scholarloop.writing.paper import PaperPipeline
from scholarloop.core.profile import load_profile
from scholarloop.experiment.reflector import Reflector
from scholarloop.experiment.skills import SkillLibrary
from scholarloop.writing.session import WritingSession

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "live"


def _cost(model, usage):
    return cost_of(usage, model)          # one price table (scholarloop.experiment.governor), not a copy


def _first(trace, agent):
    return next((c.output for c in trace.calls if c.agent == agent), {})


def fmt_paper(draft, grounded, ungrounded, review) -> str:
    g = "✅ every number traces to a recorded measurement" if grounded \
        else f"⚠️ FLAGGED ungrounded numbers: {ungrounded}"
    out = [f"# {draft['title']}", "",
           f"> **Peer review:** {review['recommendation']} · score {review['score']}/10 — {review['summary']}  ",
           f"> **Number-grounding:** {g}", "",
           "## Abstract", "", draft["abstract"], ""]
    for s in draft.get("sections", []):
        out += [f"## {s['heading']}", "", s["body"], ""]
    out += ["---", "", "### Reviewer notes", "",
            "**Strengths**", ""] + [f"- {x}" for x in review.get("strengths", [])]
    out += ["", "**Weaknesses**", ""] + [f"- {x}" for x in review.get("weaknesses", [])]
    out += ["", "*Drafted by ScholarLoop's L5 Writer and assessed by its Reviewer agent; "
            "every reported number is checked against the experiment registry.*", ""]
    return "\n".join(out)


def _unit(profile) -> str:
    """Error/accuracy metrics are percentages; RMSE and the like are bare numbers."""
    name = profile.metric.name.lower()
    return "%" if ("err" in name or "acc" in name) else ""


def fmt_run(profile, entries, skills, usage, model, trace, calibration: str = "",
            handoff: dict | None = None, stop_reason: str | None = None) -> str:
    d, lit = _first(trace, "director"), _first(trace, "lit_scout")
    cost = _cost(model, usage)
    u = _unit(profile)
    out = [f"# ScholarLoop — autonomous run log", "",
           f"A complete run on the real **{profile.name}** torch engine, driven by **{model}**. "
           f"Every experiment below is a real PyTorch training run; every decision is a real LLM call.", "",
           f"- **{len(entries)}** experiments · **{usage['calls']}** LLM calls · "
           f"{usage['input_tokens']}+{usage['output_tokens']} tokens"
           + (f" · ≈ **${cost:.3f}**" if cost is not None else ""),
           f"- baseline to beat: {profile.best_baseline()}{u} {profile.metric.name}", ""]
    if handoff:
        idea = handoff.get("idea", {})
        out += ["## 💡 Chosen idea (human-selected in a research session)",
                f"> **{idea.get('title', handoff.get('topic'))}** — {idea.get('description', '')}  ",
                f"> *hypothesis:* {idea.get('hypothesis', {}).get('claim', '')}  ",
                f"> *builds on:* {', '.join(handoff.get('sources', []))}", ""]
    else:
        out += ["## 🎯 Director — direction",
                f"> {d.get('direction', '(n/a)')}  ",
                f"> *topic for the Lit Scout:* {d.get('topic', '(n/a)')}", "",
                "## 🔭 Lit Scout — grounded findings (real literature: arXiv + OpenAlex, citation-ranked)"]
        for f in lit.get("findings", []):
            out.append(f"- **{f.get('technique')}** ({f.get('source')}) — {f.get('predicted_effect')}")
    out += ["", "## 🪜 Experiments (real torch · multi-fidelity funnel)", "",
            f"| id | tier | {profile.metric.name} | held-out test | verdict | predicted→measured | grounded source |",
            "|---|---|---|---|---|---|---|"]
    for e in entries:
        p = e.prediction or {}
        pm = f"{p.get('predicted')}→{p.get('measured')}" if p.get("measured") is not None else "—"
        held = f"{e.holdout_score()}{u}" if e.holdout_score() is not None else "—"
        out.append(f"| {e.id} | {e.fidelity[0]} | {e.primary_score()}{u} | {held} | {e.verdict} | {pm} | "
                   f"{e.hypothesis.source[:60]} |")
    if any(e.holdout_score() is not None for e in entries):
        out += ["", "_Held-out test: scored once on full-tier runs, on data the loop never selected on. "
                "The gap to the val column is the optimism that repeated selection on val adds._"]
    out += ["", "## 🧠 Accumulated skills (self-improvement)", "",
            skills.render() or "_(none recorded)_", ""]
    if calibration:
        out += ["## 🎯 Agent calibration (predict-then-verify)", "", calibration, ""]
    if stop_reason:
        out += ["## 🛑 Governor", "", f"The campaign stopped itself: {stop_reason}.", ""]
    out += ["## 🔁 Agent trace (every call, auditable)", "",
            "`" + " · ".join(f"{a}:{n}" for a, n in sorted(Counter(c.agent for c in trace.calls).items())) + "`", "",
            "See [`paper.md`](paper.md) for the write-up this run produced, and "
            "[`experiments.jsonl`](experiments.jsonl) for the raw ledger.", ""]
    return "\n".join(out)


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 1
    model = model_from_env()
    steps = int(os.environ.get("SCHOLARLOOP_STEPS", "3"))
    topic = os.environ.get("SCHOLARLOOP_TOPIC", "small MLP image classification regularization")
    paper_only = bool(os.environ.get("SCHOLARLOOP_PAPER_ONLY"))   # regenerate paper.md from a saved ledger
    profile_name = os.environ.get("SCHOLARLOOP_PROFILE", "digits-mlp")
    population = int(os.environ.get("SCHOLARLOOP_POPULATION", "1"))   # >1 -> parallel population funnel
    max_workers = int(os.environ.get("SCHOLARLOOP_MAX_WORKERS", "1"))
    dry_patience = os.environ.get("SCHOLARLOOP_DRY_PATIENCE")         # loop-until-dry convergence
    max_cost = os.environ.get("SCHOLARLOOP_MAX_COST")                 # USD budget ceiling
    handoff_path = os.environ.get("SCHOLARLOOP_HANDOFF")              # a research session's chosen idea
    handoff = json.loads(Path(handoff_path).read_text()) if handoff_path else None
    research_dir = Path(handoff_path).parent if handoff_path else None   # lit_review.json lives beside it
    writing_mode = os.environ.get("SCHOLARLOOP_WRITING", "session")
    out = Path(os.environ.get("SCHOLARLOOP_OUT", "")) if os.environ.get("SCHOLARLOOP_OUT") else OUT

    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "experiments.jsonl"
    registry_dir = out / "registry"                 # kept: the write-up grounds its numbers here
    skills_dir = Path(tempfile.mkdtemp(prefix="paper_skl_"))
    llm = AnthropicLLM(model=model)
    profile = load_profile(ROOT / "profiles" / f"{profile_name}.yaml")
    # SCHOLARLOOP_PERSIST_SKILLS=1 -> ~/.scholarloop/skills/<domain>, so lessons compound across runs
    skills = (SkillLibrary.for_domain(profile.name) if os.environ.get("SCHOLARLOOP_PERSIST_SKILLS")
              else SkillLibrary(skills_dir))

    if paper_only:
        entries = list(Ledger(ledger_path).read_all())
        print(f"=== paper-only: writing the paper from {len(entries)} saved experiments ... ===")
        orch = None
    else:
        if ledger_path.exists():
            ledger_path.unlink()                   # fresh run
        shutil.rmtree(registry_dir, ignore_errors=True)
        for stale in ("writing.json", "outline.md", "key_points.md", "paper.md", "review.md"):
            (out / stale).unlink(missing_ok=True)
        orch = Orchestrator(
            llm, profile,
            lit_scout=LitScout(llm, sources=[OpenAlexClient(), ArxivClient()]),  # impact-ranked, multi-source
            debate_panel=DebatePanel(llm, profile.debate_roles),
            reflector=Reflector(llm), advisor=Advisor(llm), director=Director(llm, profile),
            skill_library=skills, topic=topic,
            ledger_path=ledger_path, registry_dir=registry_dir)
        if handoff:                                   # the human already chose the direction
            orch.adopt(handoff)
            print(f"=== adopting the chosen idea: {handoff['topic']!r} ===")
        gov = None
        if dry_patience or max_cost:
            from scholarloop.experiment.governor import Governor
            gov = Governor(max_rounds=steps,
                           dry_patience=int(dry_patience) if dry_patience else None,
                           max_cost=float(max_cost) if max_cost else None)
        mode = (f"population={population} (max_workers={max_workers})" if population > 1 else "funnel")
        print(f"=== running campaign on {profile.name} with {model} · {mode}"
              + (f" · governed{' (dry='+dry_patience+')' if dry_patience else ''}" if gov else f" · {steps} steps")
              + " ... ===")
        orch.run(steps, funnel=True, population=population, max_workers=max_workers, governor=gov)
        entries = list(Ledger(ledger_path).read_all())
        kept = [e for e in entries if e.verdict == "kept"]
        print(f"   {len(entries)} experiments, {len(kept)} kept. writing the paper ...")

    if orch is not None:                            # run.md reflects the campaign; keep it on paper-only re-runs
        (out / "run.md").write_text(fmt_run(profile, entries, skills, llm.usage, model, orch.trace,
                                            calibration=orch.calibration.render(), handoff=handoff,
                                            stop_reason=orch.stop_reason))

    if writing_mode == "oneshot":                   # the single-call fallback
        paper = PaperPipeline(llm, llm, registry_dir=registry_dir).run(
            entries, extra_grounded=[profile.best_baseline()])
        (out / "paper.md").write_text(
            fmt_paper(paper["draft"], paper["grounded"], paper["ungrounded"], paper["review"]))
        artifacts = "paper.md · run.md · experiments.jsonl · registry/"
    else:                                           # step by step, with the author in the loop
        session = WritingSession(llm, profile, out, ledger_path=ledger_path, registry_dir=registry_dir,
                                 research_dir=research_dir, trace=orch.trace if orch else None)
        if writing_mode == "session" and sys.stdin.isatty():
            if session.interactive() is None:
                print(f"   writing paused at stage {session.stage!r}; resume with "
                      f"python -m scholarloop.writing --out {out} ...")
        else:
            steps = {"new": session.outline, "outlined": session.key_points,
                     "keypointed": session.draft, "drafted": session.review}
            while session.stage in steps:
                steps[session.stage]()
                print(f"   writing: stage {session.stage}")
        artifacts = "outline.md · key_points.md · paper.md · review.md · run.md · experiments.jsonl · registry/"

    shutil.rmtree(skills_dir, ignore_errors=True)

    cost = _cost(model, llm.usage)
    print(f"\n=== done. artifacts in {out} ===")
    print(f"   {artifacts}")
    print(f"   {llm.usage['calls']} LLM calls"
          + (f" · ≈ ${cost:.3f}" if cost is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
