"""Run a COMPLETE ScholarLoop flow with a real LLM — campaign + L5 write-up — and save the
artifacts to examples/sample_run/ : the paper (paper.md), the real experiments
(experiments.jsonl), and a narrative run log (run.md).

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/run_to_paper.py
    # knobs:  SCHOLARLOOP_MODEL=claude-opus-4-8  SCHOLARLOOP_STEPS=3
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from scholarloop.advisor import Advisor
from scholarloop.debate import DebatePanel
from scholarloop.director import Director
from scholarloop.ledger import Ledger
from scholarloop.litscout import ArxivClient, LitScout, OpenAlexClient
from scholarloop.llm import AnthropicLLM
from scholarloop.orchestrator import Orchestrator
from scholarloop.paper import PaperPipeline
from scholarloop.profile import load_profile
from scholarloop.reflector import Reflector
from scholarloop.skills import SkillLibrary

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "sample_run"
_PRICE = {"claude-opus-4-8": (5.0, 25.0), "claude-sonnet-4-6": (3.0, 15.0),
          "claude-haiku-4-5": (1.0, 5.0)}


def _cost(model, usage):
    if model not in _PRICE:
        return None
    pin, pout = _PRICE[model]
    return usage["input_tokens"] / 1e6 * pin + usage["output_tokens"] / 1e6 * pout


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


def fmt_run(profile, entries, skills, usage, model, trace, calibration: str = "") -> str:
    d, lit = _first(trace, "director"), _first(trace, "lit_scout")
    cost = _cost(model, usage)
    u = _unit(profile)
    out = [f"# ScholarLoop — autonomous run log", "",
           f"A complete run on the real **{profile.name}** torch engine, driven by **{model}**. "
           f"Every experiment below is a real PyTorch training run; every decision is a real LLM call.", "",
           f"- **{len(entries)}** experiments · **{usage['calls']}** LLM calls · "
           f"{usage['input_tokens']}+{usage['output_tokens']} tokens"
           + (f" · ≈ **${cost:.3f}**" if cost is not None else ""),
           f"- baseline to beat: {profile.best_baseline()}{u} {profile.metric.name}", "",
           "## 🎯 Director — direction",
           f"> {d.get('direction', '(n/a)')}  ",
           f"> *topic for the Lit Scout:* {d.get('topic', '(n/a)')}", "",
           "## 🔭 Lit Scout — grounded findings (real literature: arXiv + OpenAlex, citation-ranked)"]
    for f in lit.get("findings", []):
        out.append(f"- **{f.get('technique')}** ({f.get('source')}) — {f.get('predicted_effect')}")
    out += ["", "## 🪜 Experiments (real torch · multi-fidelity funnel)", "",
            f"| id | tier | {profile.metric.name} | verdict | predicted→measured | grounded source |",
            "|---|---|---|---|---|---|"]
    for e in entries:
        p = e.prediction or {}
        pm = f"{p.get('predicted')}→{p.get('measured')}" if p.get("measured") is not None else "—"
        out.append(f"| {e.id} | {e.fidelity[0]} | {e.primary_score()}{u} | {e.verdict} | {pm} | "
                   f"{e.hypothesis.source[:60]} |")
    out += ["", "## 🧠 Accumulated skills (self-improvement)", "",
            skills.render() or "_(none recorded)_", ""]
    if calibration:
        out += ["## 🎯 Agent calibration (predict-then-verify)", "", calibration, ""]
    out += ["## 🔁 Agent trace (every call, auditable)", "",
            "`" + " · ".join(f"{a}:{n}" for a, n in sorted(Counter(c.agent for c in trace.calls).items())) + "`", "",
            "See [`paper.md`](paper.md) for the write-up this run produced, and "
            "[`experiments.jsonl`](experiments.jsonl) for the raw ledger.", ""]
    return "\n".join(out)


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 1
    model = os.environ.get("SCHOLARLOOP_MODEL", "claude-opus-4-8")
    steps = int(os.environ.get("SCHOLARLOOP_STEPS", "3"))
    topic = os.environ.get("SCHOLARLOOP_TOPIC", "small MLP image classification regularization")
    paper_only = bool(os.environ.get("SCHOLARLOOP_PAPER_ONLY"))   # regenerate paper.md from a saved ledger
    profile_name = os.environ.get("SCHOLARLOOP_PROFILE", "digits-mlp")
    out = Path(os.environ.get("SCHOLARLOOP_OUT", "")) if os.environ.get("SCHOLARLOOP_OUT") else OUT

    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "experiments.jsonl"
    registry_dir = Path(tempfile.mkdtemp(prefix="paper_reg_"))
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
        orch = Orchestrator(
            llm, profile,
            lit_scout=LitScout(llm, sources=[OpenAlexClient(), ArxivClient()]),  # impact-ranked, multi-source
            debate_panel=DebatePanel(llm, profile.debate_roles),
            reflector=Reflector(llm), advisor=Advisor(llm), director=Director(llm, profile),
            skill_library=skills, topic=topic,
            ledger_path=ledger_path, registry_dir=registry_dir)
        print(f"=== running {steps}-step campaign on {profile.name} with {model} ... ===")
        orch.run(steps, funnel=True)
        entries = list(Ledger(ledger_path).read_all())
        kept = [e for e in entries if e.verdict == "kept"]
        print(f"   {len(entries)} experiments, {len(kept)} kept. writing the paper ...")

    paper = PaperPipeline(llm, llm, registry_dir=registry_dir).run(
        entries, extra_grounded=[profile.best_baseline()])
    (out / "paper.md").write_text(
        fmt_paper(paper["draft"], paper["grounded"], paper["ungrounded"], paper["review"]))
    if orch is not None:                            # run.md reflects the campaign; keep it on paper-only re-runs
        (out / "run.md").write_text(fmt_run(profile, entries, skills, llm.usage, model, orch.trace,
                                            calibration=orch.calibration.render()))

    shutil.rmtree(registry_dir, ignore_errors=True)
    shutil.rmtree(skills_dir, ignore_errors=True)

    cost = _cost(model, llm.usage)
    print(f"\n=== done. artifacts in {out} ===")
    print(f"   paper.md · run.md · experiments.jsonl")
    print(f"   {llm.usage['calls']} LLM calls"
          + (f" · ≈ ${cost:.3f}" if cost is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
