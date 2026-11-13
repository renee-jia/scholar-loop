"""Development-only controlled research decisions; no model judge or training run.

The trusted evaluator owns the mechanism and gives policies JSON observations only.
This is an API boundary, NOT a sandbox: an untrusted Python policy can inspect this
process. Published benchmark use requires external isolation and unseen task families.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

from scholarloop.review.contracts import atomic_json, digest, read_json

VERSION = 1
MECHANISMS = ("leakage", "compute", "subgroup", "genuine")
ACTIONS = {
    "clean_split": {"cost": 2, "description": "Remove train/evaluation overlap; retain other settings."},
    "matched_compute": {"cost": 1, "description": "Match training compute and tuning effort between methods."},
    "new_population": {"cost": 2, "description": "Measure the gain in the prespecified second population."},
}
DIAGNOSTIC = {"clean_split": "leakage", "matched_compute": "compute", "new_population": "subgroup"}


def make_tasks(repeats=3, seed=0):
    """Balanced development fixtures; each task has exactly one generating mechanism.

    All mechanisms have indistinguishable initial gains. These deliberately simple
    interventions establish evaluator behavior, not realistic scientific performance.
    """
    if type(repeats) is not int or repeats < 1 or type(seed) is not int:
        raise ValueError("Require positive integer repeats and an integer seed")
    tasks = []
    for repeat in range(repeats):
        for mechanism in MECHANISMS:
            identity = digest({"seed": seed, "repeat": repeat, "mechanism": mechanism, "version": VERSION})
            rng = random.Random(identity)
            observations = {}
            for action in ACTIONS:
                mean = 0.0 if DIAGNOSTIC[action] == mechanism else 8.0
                observations[action] = {"gain": round(mean + rng.gauss(0, 0.25), 4),
                                        "evidence_id": f"{identity[:16]}:{action}"}
            tasks.append({"id": identity[:16], "mechanism": mechanism,
                          "initial_gain": round(8 + rng.gauss(0, 0.25), 4),
                          "observations": observations})
    random.Random(seed).shuffle(tasks)
    return tasks


def public_task(task):
    return {"id": task["id"], "question": "Which single explanation generated the preliminary gain?",
            "hypotheses": list(MECHANISMS), "initial_gain": task["initial_gain"],
            "actions": copy.deepcopy(ACTIONS),
            "scope": "Synthetic single-cause development task; gain units are arbitrary. "
                     "Genuine means robust to these three controls, not universal causal validity."}


def run_episode(task, policy, budget):
    if type(budget) is not int or budget < 0:
        raise ValueError("Budget must be a nonnegative integer")
    public = public_task(task)
    history, spent, decision, status, error = [], 0, None, "completed", None
    # Positive action costs bound the number of calls, including a terminal decision.
    try:
        for _ in range(budget + 1):
            response = policy(copy.deepcopy({"task": public, "history": history,
                                             "remaining_budget": budget - spent}))
            if not isinstance(response, dict):
                raise ValueError("Policy must return an action or decision object")
            if set(response) == {"decision"}:
                decision = response["decision"]
                if decision not in (*MECHANISMS, "abstain"):
                    raise ValueError("Unknown decision")
                break
            if set(response) != {"action"} or response["action"] not in ACTIONS:
                raise ValueError("Unknown action or invalid response")
            action = response["action"]
            cost = ACTIONS[action]["cost"]
            if cost > budget - spent:
                status = "budget_exhausted"
                break
            spent += cost
            # Repeated action returns the same evidence, never a fresh replicate.
            history.append({"action": action, "cost": cost, **task["observations"][action]})
        else:
            status = "budget_exhausted"
    except Exception as exc:
        status, error, decision = "failed", type(exc).__name__ + ": " + str(exc), None
    return {"task_id": task["id"], "status": status, "decision": decision,
            "truth": task["mechanism"], "correct": status == "completed" and decision == task["mechanism"],
            "spent": spent, "budget": budget, "history": history, "error": error}


def summarize(rows):
    n = len(rows)
    completed = [r for r in rows if r["status"] == "completed" and r["decision"] in MECHANISMS]
    artifacts = [r for r in rows if r["truth"] != "genuine"]
    genuine = [r for r in rows if r["truth"] == "genuine"]
    def rate(count, total):
        return count / total if total else None
    return {"planned_tasks": n, "correct": sum(r["correct"] for r in rows),
            "accuracy": rate(sum(r["correct"] for r in rows), n),
            "coverage": rate(len(completed), n),
            "selective_error": rate(sum(not r["correct"] for r in completed), len(completed)),
            "false_genuine_acceptance": rate(sum(r["decision"] == "genuine" for r in artifacts), len(artifacts)),
            "false_genuine_rejection": rate(sum(r["decision"] in MECHANISMS[:-1] for r in genuine), len(genuine)),
            "artifact_tasks": len(artifacts), "genuine_tasks": len(genuine),
            "abstentions": sum(r["decision"] == "abstain" for r in rows),
            "failed": sum(r["status"] == "failed" for r in rows),
            "budget_exhausted": sum(r["status"] == "budget_exhausted" for r in rows),
            "interrupted": sum(r["status"] == "interrupted" for r in rows),
            "unknown_cost_episodes": sum(r["spent"] is None for r in rows),
            "total_cost_units": (sum(r["spent"] for r in rows)
                                 if all(r["spent"] is not None for r in rows) else None)}


def baseline(name):
    if name not in {"accept", "abstain", "fixed", "random"}:
        raise ValueError("Unknown baseline")
    def policy(ctx):
        if name in {"accept", "abstain"}:
            return {"decision": "genuine" if name == "accept" else "abstain"}
        for row in ctx["history"]:
            if row["gain"] < 2:
                return {"decision": DIAGNOSTIC[row["action"]]}
        seen = {r["action"] for r in ctx["history"]}
        if len(seen) == len(ACTIONS):
            return {"decision": "genuine"}
        order = list(ACTIONS)
        if name == "random":
            random.Random(ctx["task"]["id"]).shuffle(order)
        available = [a for a in order if a not in seen and ACTIONS[a]["cost"] <= ctx["remaining_budget"]]
        return {"action": available[0]} if available else {"decision": "abstain"}
    return policy


def run_benchmark(directory, *, repeats=3, seed=0, budget=3, arms=("accept", "abstain", "fixed", "random")):
    if type(budget) is not int or budget < 0:
        raise ValueError("Budget must be a nonnegative integer")
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("Require distinct baseline arms")
    policies = {arm: baseline(arm) for arm in arms}
    tasks = make_tasks(repeats, seed)
    plan = {"version": VERSION, "split": "development", "seed": seed, "repeats": repeats,
            "budget": budget, "arms": list(arms), "tasks_hash": digest(tasks), "actions": ACTIONS,
            "implementation_hash": digest(Path(__file__).read_text())}
    out = Path(directory)
    path = out / "plan.json"
    if path.exists():
        if read_json(path) != plan:
            raise ValueError("Frozen development plan changed; use another directory")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Nonempty directory has no frozen plan")
    else:
        atomic_json(path, plan)
    report = {"plan": plan, "arms": {}, "limitation": "Synthetic development fixtures and trusted "
              "in-process policies only. Not a hidden benchmark, scientific validity evaluation, or RSI result."}
    for arm, policy in policies.items():
        rows = [run_episode(task, policy, budget) for task in tasks]
        report["arms"][arm] = {"summary": summarize(rows), "episodes": rows}
    atomic_json(out / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=3)
    args = parser.parse_args(argv)
    report = run_benchmark(args.out, repeats=args.repeats, seed=args.seed, budget=args.budget)
    print(json.dumps({arm: data["summary"] for arm, data in report["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
