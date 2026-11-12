"""Frozen multi-task evaluation with explicit development/evaluation partitions.

Dataset separation validates declarations, not the semantic independence of real data.
Each task retains its own metric and missing outcomes; raw scores are never averaged
across incompatible tasks. The CLI executes only the free random-search baseline.
"""
import argparse
import json
import re
from pathlib import Path

from scholarloop.core.candidates import make_snapshot
from scholarloop.core.ledger import Ledger
from scholarloop.core.profile import load_profile
from scholarloop.experiment.compare import campaign_summary, run_campaign
from scholarloop.review.contracts import atomic_json


def freeze_suite(spec, directory):
    if set(spec) != {"version", "tasks"} or spec["version"] != 1 or not spec["tasks"]:
        raise ValueError("Expected version=1 and nonempty tasks")
    ids, groups, datasets, tasks = set(), {}, {}, []
    for task in spec["tasks"]:
        if set(task) != {"id", "split", "dataset_group", "profile", "seeds", "train_budget"}:
            raise ValueError("Task fields must be explicit")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task["id"]) or task["id"] in ids:
            raise ValueError("Task ids must be unique safe path components")
        ids.add(task["id"])
        split, group = task["split"], task["dataset_group"]
        if split not in ("development", "evaluation") or not isinstance(group, str) or not group.strip():
            raise ValueError("Each task needs a split and dataset_group")
        seeds = task["seeds"]
        if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds)
                or len(set(seeds)) != len(seeds) or type(task["train_budget"]) is not int or task["train_budget"] < 0):
            raise ValueError("Require distinct nonnegative seeds and a nonnegative integer training budget")
        profile = load_profile(task["profile"])
        if not profile.datasets:
            raise ValueError("Dataset declarations are required for partition checks")
        for mapping, keys in ((groups, [group]), (datasets, profile.datasets)):
            for key in keys:
                if key in mapping and mapping[key] != split:
                    raise ValueError("Development/evaluation dataset declarations overlap")
                mapping[key] = split
        tasks.append({**task, "profile": str(Path(task["profile"]).resolve()),
                      "candidate": make_snapshot(profile)["id"]})
    if {t["split"] for t in tasks} != {"development", "evaluation"}:
        raise ValueError("Declare both development and evaluation partitions")
    plan = {"version": 1, "tasks": tasks}
    directory = Path(directory)
    path = directory / "suite_plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Frozen suite changed; use a separate output directory")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Nonempty suite directory has no frozen plan")
        atomic_json(path, plan)
    return plan


def run_suite(spec, directory, *, arm="random", factory=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", arm):
        raise ValueError("Invalid arm id")
    if factory is None:
        if arm != "random":
            raise ValueError("Non-random arms require an explicit factory and model budget")
        from scholarloop.experiment.orchestrator import Orchestrator
        from scholarloop.experiment.random_search import RandomProposer
        def factory(profile, out, seed):
            orch = Orchestrator(None, profile, ledger_path=out / "experiments.jsonl", registry_dir=out / "registry")
            orch.reasoner = RandomProposer(profile, seed=seed)
            return orch
    directory = Path(directory)
    plan = freeze_suite(spec, directory)
    results = []
    for task in plan["tasks"]:
        profile = load_profile(task["profile"])
        if make_snapshot(profile)["id"] != task["candidate"]:
            raise ValueError("Task inputs changed after suite freeze")
        for seed in task["seeds"]:
            out = directory / arm / task["id"] / f"seed{seed}"
            summary_path = out / "summary.json"
            if summary_path.exists():
                results.append(json.loads(summary_path.read_text()))
                continue
            rows = list(Ledger(out / "experiments.jsonl").read_all())
            status, reason = "completed", None
            if out.exists() and any(out.iterdir()):
                status, reason = "interrupted", "Existing unfinished execution retained"
            else:
                out.mkdir(parents=True)
                try:
                    orch = factory(profile, out, seed)
                    if getattr(orch, "llm", None) is not None:
                        from scholarloop.core.call_budget import CallLimitedLLM
                        if not isinstance(orch.llm, CallLimitedLLM) or orch.llm.path is None:
                            raise ValueError("Suite model arms require a persistent CallLimitedLLM budget")
                    rows = run_campaign(orch, max_train_runs=task["train_budget"])
                except Exception as exc:
                    rows = list(Ledger(out / "experiments.jsonl").read_all())
                    status, reason = "failed", f"{type(exc).__name__}: {exc}"
            summary = {**campaign_summary(rows, profile), "task": task["id"], "split": task["split"],
                       "seed": seed, "arm": arm, "status": status, "reason": reason,
                       "train_budget": task["train_budget"]}
            budget_path = out / "experiments.jsonl.training_budget.json"
            summary["reserved_training_attempts"] = (json.loads(budget_path.read_text())["reserved"]
                                                       if budget_path.exists() else None)
            atomic_json(summary_path, summary)
            results.append(summary)
    report = {"plan": plan, "arm": arm, "planned_repeats": len(results), "results": results,
              "interpretation": "Dataset separation is declaration-based. No cross-task raw-score average; "
                                "missing results remain in the denominator. No final test access is requested."}
    atomic_json(directory / f"{arm}_results.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    spec = json.loads(manifest_path.read_text())
    for task in spec.get("tasks", []):
        if not Path(task["profile"]).is_absolute():
            task["profile"] = str(manifest_path.parent / task["profile"])
    report = run_suite(spec, args.out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
