"""Live end-to-end run against the real Anthropic API.

Swaps the scripted MockLLM for AnthropicLLM and runs the full agent chain — Director, Lit Scout
(real arXiv), Reasoner, Debate, Reflector, Advisor — over the multi-fidelity funnel on the real
torch digits-mlp engine. The experiments are real torch; the agents are a real LLM.

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/e2e_live.py
    # knobs:  SCHOLARLOOP_MODEL=claude-haiku-4-5  SCHOLARLOOP_STEPS=1  to cut cost

A run makes ~6-8 LLM calls per idea; cost is printed at the end (a few cents on Opus).
"""

from __future__ import annotations

import os
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
from scholarloop.profile import load_profile
from scholarloop.reflector import Reflector
from scholarloop.skills import SkillLibrary

ROOT = Path(__file__).resolve().parent.parent

# rough $/1M tokens for cost estimate (Opus 4.8); override-friendly
_PRICE = {"claude-opus-4-8": (5.0, 25.0), "claude-sonnet-4-6": (3.0, 15.0),
          "claude-haiku-4-5": (1.0, 5.0)}


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Set it, then re-run:\n"
              "  export ANTHROPIC_API_KEY=sk-ant-...\n"
              "  python examples/e2e_live.py", file=sys.stderr)
        return 1

    model = os.environ.get("SCHOLARLOOP_MODEL", "claude-opus-4-8")
    steps = int(os.environ.get("SCHOLARLOOP_STEPS", "2"))
    topic = os.environ.get("SCHOLARLOOP_TOPIC", "small MLP image classification regularization")

    llm = AnthropicLLM(model=model)          # one client, shared by every agent
    profile = load_profile(ROOT / "profiles" / "digits-mlp.yaml")
    tmp = Path(tempfile.mkdtemp(prefix="e2e_"))
    # SCHOLARLOOP_PERSIST_SKILLS=1 -> ~/.scholarloop/skills/<domain>, compounding across runs
    skills = (SkillLibrary.for_domain(profile.name) if os.environ.get("SCHOLARLOOP_PERSIST_SKILLS")
              else SkillLibrary(tmp / "skills"))

    orch = Orchestrator(
        llm, profile,
        lit_scout=LitScout(llm, sources=[OpenAlexClient(), ArxivClient()]),  # impact-ranked, multi-source
        debate_panel=DebatePanel(llm, profile.debate_roles),
        reflector=Reflector(llm),
        advisor=Advisor(llm),
        director=Director(llm, profile),
        skill_library=skills,
        topic=topic,
        ledger_path=tmp / "ledger.jsonl", registry_dir=tmp / "registry",
    )

    print(f"=== LIVE E2E · model={model} · {steps} step(s) · real torch digits-mlp ===")
    print(f"    baseline to beat: {profile.best_baseline()}% {profile.metric.name}\n")

    produced = orch.run(steps, funnel=True)

    print("\n--- ledger (real torch results) ---")
    for e in Ledger(tmp / "ledger.jsonl").read_all():
        print(f"  {e.id} {e.fidelity[0]:<6} {e.metric_name}={e.primary_score()}% {e.verdict:<10}"
              f" {e.hypothesis.source}")
    print(f"\n--- skill library ---\n{skills.render() or '  (none)'}")
    print(f"\n--- agent calls (auditable trace) ---\n  "
          f"{dict(Counter(c.agent for c in orch.trace.calls))}")

    u = llm.usage
    cost = ""
    if model in _PRICE:
        pin, pout = _PRICE[model]
        cost = f"  ≈ ${u['input_tokens']/1e6*pin + u['output_tokens']/1e6*pout:.3f}"
    print(f"\n--- cost ---\n  {u['calls']} LLM calls · "
          f"{u['input_tokens']} in + {u['output_tokens']} out tokens{cost}")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
