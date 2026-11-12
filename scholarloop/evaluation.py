"""Read-only workflow diagnostics for later matched-budget research evaluation.

Counts are operational measurements, not novelty/correctness/idea-quality scores.
Missing or invalid artifacts remain explicit; no overall success score is invented.
"""

from collections import Counter
from pathlib import Path

from scholarloop.review.contracts import ReviewBlocked, atomic_json, read_json
from scholarloop.review.session import ScientificReviewSession


def audit_run(profile, directory, *, research_dir=None):
    out = Path(directory)
    errors = []

    def optional(path):
        if not path.exists():
            return None
        try:
            return read_json(path)
        except ReviewBlocked as exc:
            errors.append(str(exc))
            return None

    research = Path(research_dir) if research_dir else out
    session = optional(research / "session.json")
    graph = session.get("idea_graph") if session else None
    science = optional(out / "scientific_review.json")
    result = {"directory": str(out.resolve()), "errors": errors,
              "interpretation": "Workflow diagnostics only. Diversity counts and LLM verdicts do not establish "
                                "research quality; compare blind expert labels and executed evidence at matched cost.",
              "ideation": None, "scientific_review": None, "protocols": [], "request_budgets": {}}
    if graph:
        nodes = list(graph["nodes"].values())
        result["ideation"] = {"rounds": len(graph["rounds"]), "unique_nodes": len(nodes),
            "strategies": dict(Counter(n["idea"].get("strategy", "legacy") for n in nodes)),
            "structured_designs": sum(bool(n["idea"].get("design")) for n in nodes),
            "novelty_statuses": dict(Counter(n["idea"].get("novelty_review", {}).get("status", "not_reviewed") for n in nodes)),
            "human_decisions": sum(d["kind"] == "human" for d in graph["decisions"]),
            "automated_decisions": sum(d["kind"] == "automated" for d in graph["decisions"]),
            "pilot_observations": len(graph.get("observations", [])),
            "pilot_execution_statuses": dict(Counter(o["execution_status"] for o in graph.get("observations", [])))}
    if science:
        check = {"stored_status": science["status"], "current_release_valid": False,
                 "open_issues": [i["id"] for i in science["issues"] if i["status"] == "open"],
                 "reviewer_models": sorted({r["reviewer"]["model"] for r in science["reports"]
                                             if r.get("reviewer", {}).get("model")}),
                 "claim_check_statuses": dict(Counter(c["status"] for r in science["reports"]
                                                       for c in r.get("claim_checks", [])))}
        gate = ScientificReviewSession(None, profile, out, ledger_path=out / "experiments.jsonl",
                                        registry_dir=out / "registry", research_dir=research_dir)
        try:
            gate.require_release()
            check["current_release_valid"] = True
        except ReviewBlocked as exc:
            check["writing_block_reason"] = str(exc)
        result["scientific_review"] = check
    for path in sorted((out / "experiment_agenda").glob("*.json")):
        state = optional(path)
        if state:
            analysis = optional(out / "registry" / "protocols" / f"{state['protocol_id']}.analysis.json")
            result["protocols"].append({"protocol_id": state["protocol_id"], "status": state["status"],
                "recorded_conditions": len(state["results"]), "seconds_used": state["seconds_used"],
                "incomplete_comparisons": (sum(c["status"] != "complete" for c in analysis["comparisons"])
                                           if analysis else None),
                "scientific_outcome": "unreviewed" if not analysis else analysis["scientific_outcome"]})
    for base in {out, research}:
        for path in sorted(base.glob("*_call_budget.json")):
            result["request_budgets"][str(path.resolve())] = optional(path)
    from scholarloop.core.agent import summarize_trace
    result["agent_traces"] = {}
    for base in {out.resolve(), research.resolve()}:
        for path in sorted(base.glob("*.agents.jsonl")):
            result["agent_traces"][str(path)] = summarize_trace(path)
    result["token_usage"] = optional(out / "usage.json")
    return result


def main(argv=None):
    import argparse
    import json
    from scholarloop.core.profile import load_profile
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True, help="campaign directory with experiments.jsonl and registry/")
    ap.add_argument("--research", help="exact research directory used during scientific review, if any")
    ap.add_argument("--output", help="optional JSON report path")
    args = ap.parse_args(argv)
    report = audit_run(load_profile(args.profile), args.out, research_dir=args.research)
    if args.output:
        atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
