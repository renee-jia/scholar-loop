"""Does the agent loop beat random search at the same budget? Measure it.

Runs independent campaigns per arm on one profile, through the same Orchestrator funnel, gates,
seeds, CV folds, frozen scorer and held-out test; only the proposer differs (see
scholarloop/experiment/random_search.py). Writes compare.md and compare.json to --out.

    # free: random search only, no API key, no LLM call
    python examples/compare_search.py --arms random --repeats 5 --rounds 8

    # the real comparison (costs API money: one full agent campaign per repeat)
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/compare_search.py --arms agent,random --repeats 3 --rounds 6 --match runs --max-cost 5

`--match rounds` gives both arms the same number of ideas; `--match runs` runs the agent first
and gives random search the agent's mean number of training runs, since the funnel spends more
compute on ideas that climb. Finished campaigns are kept (summary.json per repeat), so an
interrupted comparison picks up where it stopped; delete --out to start over.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scholarloop.core.profile import load_profile                          # noqa: E402
from scholarloop.experiment.compare import campaign_summary, render, run_campaign   # noqa: E402
from scholarloop.experiment.governor import cost_of                       # noqa: E402
from scholarloop.experiment.orchestrator import Orchestrator              # noqa: E402
from scholarloop.experiment.random_search import RandomProposer           # noqa: E402


def random_orchestrator(profile, out: Path, seed: int) -> Orchestrator:
    orch = Orchestrator(None, profile, ledger_path=out / "experiments.jsonl",
                        registry_dir=out / "registry")
    orch.reasoner = RandomProposer(profile, seed=seed)     # the only difference from the agent arm
    return orch


def agent_orchestrator(profile, out: Path, llm, topic: str, *, warm_start: int = 0,
                       seed: int = 0) -> Orchestrator:
    from examples.run_to_paper import make_orchestrator
    from scholarloop.experiment.skills import SkillLibrary
    # a fresh skill library per campaign: repeats must be independent to be averaged
    orch = make_orchestrator(llm, profile, out, topic=topic, skills=SkillLibrary(out / "skills"))
    # warm-start draws use the random arm's seed for the same repeat, so both arms open on the
    # same configs and the difference is what the agent does with them (a paired comparison)
    orch.warm_start, orch.warm_seed = warm_start, seed
    return orch


def run_arm(arm: str, profile, out: Path, args, *, llm=None, max_train_runs=None) -> list[dict]:
    summaries = []
    for k in range(args.repeats):
        rep = out / arm / f"rep{k + 1}"
        done = rep / "summary.json"
        if done.exists():                                   # finished earlier: keep it
            summaries.append(json.loads(done.read_text()))
            continue
        if arm == "agent" and args.max_cost is not None:
            spent = cost_of(llm.usage, llm.model) or 0.0
            if spent >= args.max_cost:
                print(f"   agent: budget ${args.max_cost:.2f} reached after {k} new campaigns; stopping",
                      file=sys.stderr)
                break
        shutil.rmtree(rep, ignore_errors=True)              # an interrupted campaign restarts clean
        rep.mkdir(parents=True)
        before = dict(llm.usage) if llm is not None else {}
        orch = (agent_orchestrator(profile, rep, llm, args.topic, warm_start=args.warm_start, seed=k)
                if arm == "agent"
                else random_orchestrator(profile, rep, seed=k))
        bound = {"max_train_runs": max_train_runs} if max_train_runs else {"rounds": args.rounds}
        entries = run_campaign(orch, **bound)
        s = campaign_summary(entries, profile)
        if llm is not None:
            used = {key: llm.usage.get(key, 0) - before.get(key, 0) for key in llm.usage}
            s["llm"] = {"calls": used.get("calls", 0), "usd": cost_of(used, llm.model)}
        done.write_text(json.dumps(s, indent=1))
        summaries.append(s)
        best = s["best_full"]
        print(f"   {arm} {k + 1}/{args.repeats}: {s['ideas']} ideas, {s['train_runs']} train runs, best "
              + (f"val {best['val']} / held-out {best['holdout']}" if best else "— (nothing reached full)"))
    return summaries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", default="digits-mlp")
    ap.add_argument("--arms", default="random", help="comma-separated: agent, random")
    ap.add_argument("--repeats", type=int, default=5, help="independent campaigns per arm")
    ap.add_argument("--rounds", type=int, default=8, help="funnel rounds (ideas) per campaign")
    ap.add_argument("--match", choices=["rounds", "runs"], default="rounds")
    ap.add_argument("--max-cost", type=float, default=None, help="USD cap on the agent arm")
    ap.add_argument("--warm-start", type=int, default=0,
                    help="the agent's first N ideas are the random arm's draws (paired by repeat)")
    ap.add_argument("--topic", default=os.environ.get("SCHOLARLOOP_TOPIC",
                                                       "small MLP image classification regularization"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "compare"))
    args = ap.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not set(arms) <= {"agent", "random"} or not arms:
        ap.error("--arms takes agent and/or random")
    if args.match == "runs" and set(arms) != {"agent", "random"}:
        ap.error("--match runs needs both arms (random search gets the agent's training runs)")
    profile = load_profile(ROOT / "profiles" / f"{args.profile}.yaml")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    llm = None
    if "agent" in arms:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("the agent arm needs ANTHROPIC_API_KEY (or run --arms random)", file=sys.stderr)
            return 1
        from scholarloop.core.llm import AnthropicLLM, model_from_env
        llm = AnthropicLLM(model=model_from_env())

    results: dict[str, list[dict]] = {}
    for arm in sorted(arms):                                # agent first: --match runs needs it
        budget = None
        if arm == "random" and args.match == "runs":
            agent_runs = [s["train_runs"] for s in results.get("agent", [])]
            if not agent_runs:
                print("no finished agent campaign to match; falling back to --rounds", file=sys.stderr)
            else:
                budget = round(sum(agent_runs) / len(agent_runs))
        print(f"=== {arm}: {args.repeats} campaigns on {profile.name}, "
              + (f"{budget} train runs each" if budget else f"{args.rounds} rounds each") + " ===")
        results[arm] = run_arm(arm, profile, out, args, llm=llm, max_train_runs=budget)

    (out / "compare.json").write_text(json.dumps(results, indent=1))
    (out / "compare.md").write_text(render(results, profile, match=args.match))
    print(f"\n=== wrote {out / 'compare.md'} ===")
    if llm is not None:
        cost = cost_of(llm.usage, llm.model)
        print(f"   {llm.usage.get('calls', 0)} LLM calls" + (f" · ≈ ${cost:.2f}" if cost is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
