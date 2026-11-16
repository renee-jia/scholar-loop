from scholarloop.evaluation import audit_run
from tests.test_scientific_review import setup, released


def test_audit_distinguishes_stale_release_from_missing_work(tmp_path):
    paths = setup(tmp_path)
    p, ledger, registry, out = paths
    # Audit uses a campaign layout; use the shared parent for its scientific report.
    paths = p, ledger, registry, tmp_path
    s = released(paths)
    audit = audit_run(p, tmp_path)
    assert audit["scientific_review"]["current_release_valid"] is True
    assert audit["ideation"] is None and audit["token_usage"] is None
    (p.root / p.train_entrypoint).write_text("changed source", encoding="utf-8")
    audit = audit_run(p, tmp_path)
    assert audit["scientific_review"]["current_release_valid"] is False
    assert "stale" in audit["scientific_review"]["writing_block_reason"]


def test_process_report_counts_rounds_and_costs_each_process_once(tmp_path):
    import json
    from scholarloop.evaluation import process_report, render_process
    p, ledger, registry, out = setup(tmp_path)
    released((p, ledger, registry, tmp_path))
    def finished(call, attempt, status, calls, tokens):
        return {"event": "finished", "call_id": call, "attempt": attempt, "status": status,
                "elapsed_seconds": 2.0, "model": "claude-opus-5-5",
                "client_usage": {"calls": calls, "input_tokens": tokens, "output_tokens": tokens}}
    rows = [finished("a", 1, "rejected", 1, 1000), finished("a", 2, "accepted", 2, 2000),   # one process
            finished("b", 1, "accepted", 1, 500)]                                           # a second one
    (tmp_path / "review.agents.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "spend.json").write_text(json.dumps({"usd": 1.0, "requests": 3}), encoding="utf-8")
    report = process_report(tmp_path)
    stage = report["stages"][0]
    # cumulative readings are not summed attempt by attempt: 2000 from the first process, 500 from the second
    assert stage["input_tokens"] == 2500 and stage["agent_calls"] == 2 and stage["first_attempt_rejected"] == 1
    assert report["review_summary"] == {"rounds_started": 1, "rounds_with_all_critics": 1,
                                        "rounds_failed_before_critics": 0, "author_approvals": 1}
    assert "| 1 | READY_TO_WRITE | 4 |" in render_process(report)
