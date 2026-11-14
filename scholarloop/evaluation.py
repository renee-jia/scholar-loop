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


def process_report(directory) -> dict:
    """What a run cost and how it converged, from its own records: agent traces, the spend
    file, review history and writing transcripts. Nothing here judges the research.

    A trace records the client's cumulative usage, so a stage's tokens are the last reading of
    each process that wrote to its trace. Requests made outside an agent (the single-call
    baseline) and requests that failed before returning leave no reading; their cost appears
    only as the difference from the spend file."""
    import json
    from scholarloop.experiment.governor import cost_of
    out = Path(directory)
    stages, traced = [], 0.0
    for path in sorted(out.rglob("*.agents.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        done = [r for r in rows if r.get("event") == "finished"]
        model = next((r.get("model") for r in rows if r.get("model")), None)
        readings, previous = [], None
        for r in done:
            usage = r.get("client_usage") or {}
            if previous is not None and usage.get("calls", 0) < previous.get("calls", 0):
                readings.append(previous)                # a new process started counting again
            previous = usage
        if previous:
            readings.append(previous)
        usd = sum(cost_of(u, model) or 0.0 for u in readings)
        traced += usd
        calls = {}
        for r in done:
            calls.setdefault(r["call_id"], []).append(r)
        stages.append({
            "trace": str(path.relative_to(out)), "model": model, "agent_calls": len(calls),
            "attempts": len(done),
            "first_attempt_rejected": sum(a[0]["status"] != "accepted" for a in calls.values()),
            "never_accepted": sum(all(x["status"] != "accepted" for x in a) for a in calls.values()),
            "model_seconds": round(sum(r.get("elapsed_seconds", 0) for r in done)),
            "input_tokens": sum(u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                                + u.get("cache_read_input_tokens", 0) for u in readings),
            "output_tokens": sum(u.get("output_tokens", 0) for u in readings),
            "usd_estimate": round(usd, 2)})
    spend = json.loads((out / "spend.json").read_text()) if (out / "spend.json").exists() else None
    report = {"version": 1, "stages": stages, "usd_traced": round(traced, 2),
              "usd_spend_file": round(spend["usd"], 2) if spend else None,
              "requests_spend_file": spend.get("requests") if spend else None}

    states = [json.loads(p.read_text()) for p in sorted((out / "review_history").glob("*.json"))]
    if (out / "scientific_review.json").exists():
        states.append(json.loads((out / "scientific_review.json").read_text()))
    rounds = {}
    for state in states:                       # a round is archived more than once; keep its last state
        if state.get("status") != "REVIEWING":
            rounds[(state["round"], state["status"] == "READY_TO_WRITE")] = state
    review = []
    for (number, _), state in sorted(rounds.items()):
        new = [i for i in state["issues"] if i["id"].startswith(f"r{number}:")]
        review.append({"round": number, "status": state["status"], "critics_reported": len(state["reports"]),
                       "new_issues": {k: sum(i["severity"] == k for i in new) for k in ("blocker", "major", "minor")},
                       "open_issues": sum(i["status"] == "open" for i in state["issues"]),
                       "author_approvals": len(state["author_decisions"])})
    report["review_rounds"] = review
    report["review_summary"] = {
        "rounds_started": max((r["round"] for r in review), default=0),
        "rounds_with_all_critics": len({r["round"] for r in review if r["critics_reported"] == 4}),
        "rounds_failed_before_critics": len({r["round"] for r in review if r["status"] == "REVIEW_FAILED"}),
        "author_approvals": max((r["author_approvals"] for r in review), default=0)}

    writing = []
    for path in sorted(out.rglob("writing.json")):
        state = json.loads(path.read_text())
        row = {"directory": str(path.parent.relative_to(out)) or ".", "stage_reached": state.get("stage"),
               "sections_drafted": len(state.get("sections") or [])}
        for key in ("outline_discussion", "keypoints_discussion"):
            talk = state.get(key) or []
            row[key] = {"panel_rounds": len(talk), "accepted": bool(talk and talk[-1].get("accepted")),
                        "stopped_for_missing_evidence": bool(talk and talk[-1].get("evidence_gaps"))}
        if state.get("review"):
            scores = [r.get("score") for r in state["review"].get("reviews", []) if r.get("score") is not None]
            row["reviewer_scores"] = scores
        writing.append(row)
    report["writing_sessions"] = writing
    return report


def render_process(report: dict) -> str:
    lines = ["# Run process report", "",
             "Cost and convergence from the run's own records. Dollar figures are estimates from token "
             "counts; the spend file is the total the cap was enforced on.", "",
             "| trace | agent calls | attempts | first attempt rejected | never accepted | model seconds | "
             "input tokens | output tokens | USD |", "|---|---|---|---|---|---|---|---|---|"]
    for s in report["stages"]:
        lines.append(f"| {s['trace']} | {s['agent_calls']} | {s['attempts']} | {s['first_attempt_rejected']} | "
                     f"{s['never_accepted']} | {s['model_seconds']} | {s['input_tokens']} | {s['output_tokens']} | "
                     f"{s['usd_estimate']:.2f} |")
    lines += ["", f"Traced: ${report['usd_traced']:.2f}. Spend file: "
              + (f"${report['usd_spend_file']:.2f} over {report['requests_spend_file']} requests. The difference "
                 "is requests that left no reading in a trace." if report["usd_spend_file"] is not None else "none."),
              "", "## Scientific review", "",
              "| round | status | critics | new blocker | new major | new minor | open | approvals so far |",
              "|---|---|---|---|---|---|---|---|"]
    for r in report["review_rounds"]:
        n = r["new_issues"]
        lines.append(f"| {r['round']} | {r['status']} | {r['critics_reported']} | {n['blocker']} | {n['major']} | "
                     f"{n['minor']} | {r['open_issues']} | {r['author_approvals']} |")
    lines += ["", "## Writing sessions", "",
              "| directory | stage reached | outline rounds | key-point rounds | stopped for missing evidence | "
              "sections | reviewer scores |", "|---|---|---|---|---|---|---|"]
    for w in report["writing_sessions"]:
        o, k = w["outline_discussion"], w["keypoints_discussion"]
        lines.append(f"| {w['directory']} | {w['stage_reached']} | {o['panel_rounds']} | {k['panel_rounds']} | "
                     f"{'yes' if o['stopped_for_missing_evidence'] or k['stopped_for_missing_evidence'] else 'no'} | "
                     f"{w['sections_drafted']} | {w.get('reviewer_scores', '')} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    import argparse
    import json
    from scholarloop.core.profile import load_profile
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True, help="campaign directory with experiments.jsonl and registry/")
    ap.add_argument("--research", help="exact research directory used during scientific review, if any")
    ap.add_argument("--output", help="optional JSON report path")
    ap.add_argument("--process", action="store_true",
                    help="write process.json and process.md (cost and convergence) into --out and exit")
    args = ap.parse_args(argv)
    if args.process:
        report = process_report(args.out)
        atomic_json(Path(args.out) / "process.json", report)
        (Path(args.out) / "process.md").write_text(render_process(report))
        print(render_process(report))
        return 0
    report = audit_run(load_profile(args.profile), args.out, research_dir=args.research)
    if args.output:
        atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
