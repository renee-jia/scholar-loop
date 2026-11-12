"""Reliability evaluation: fault injection through the production guards, natural fault rates
from attempt logs, and the post-hoc audit used for the single-call baseline."""

import json

import pytest

from scholarloop.core.agent import Agent, AgentError, AgentTrace
from scholarloop.core.llm import MockLLM
from scholarloop.reliability import (
    FAULTS, audit_paper, classify, load_cases, main, natural_faults, run_baseline, run_injection,
    wilson,
)
from tests.writing.test_writing_session import OUTLINE, PROFILE, REVIEW, _body, _entries, _kp


@pytest.fixture
def run(tmp_path):
    ledger, registry = _entries(tmp_path)
    review = REVIEW.to_dict()
    review["themes"] = [{"name": "schedules", "summary": "s", "papers": ["arXiv:1608.03983v1"]}]
    (tmp_path / "lit_review.json").write_text(json.dumps(review))
    sections = [{"heading": s["heading"], "body": _body(s["heading"])} for s in OUTLINE["sections"]]
    (tmp_path / "writing.json").write_text(json.dumps(
        {"outline": OUTLINE, "key_points": _kp(), "sections": sections}))
    return {"ledger": ledger, "registry": registry, "research": tmp_path,
            "writing": tmp_path / "writing.json"}


def test_accepted_artifacts_pass_and_every_trial_changes_one_site(run):
    cases, ctx = load_cases(PROFILE, **run)
    report = run_injection(cases, ctx, seed=0)
    assert report["controls"] == {"cases": len(cases), "false_positives": 0, "details": []}
    assert {t["fault"] for t in report["trials"]} <= set(FAULTS)
    assert all(t["change"] for t in report["trials"])
    assert report == run_injection(cases, ctx, seed=0)                  # seeded, hence reproducible
    assert report["trials"] != run_injection(cases, ctx, seed=1)["trials"]


def test_fabrications_are_caught_where_the_guards_apply(run):
    cases, ctx = load_cases(PROFILE, **run)
    by = run_injection(cases, ctx)["by_fault_and_target"]
    for key in ("fabrication/fabricated_citation/section", "fabrication/fabricated_citation/lit_review",
                "fabrication/fabricated_citation/outline", "fabrication/fabricated_citation/key_points",
                "fabrication/fabricated_run_id/section", "fabrication/fabricated_run_id/key_points",
                "fabrication/fabricated_number/section", "fabrication/fabricated_number/key_points",
                "fabrication/uncited_claim/key_points", "fabrication/dropped_limitations/outline"):
        assert by[key]["n"] > 0 and by[key]["rate"] == 1.0, key


def test_report_states_what_lexical_checks_cannot_see(run):
    """Misattribution faults reference real evidence, so the guards pass them. The report must
    show that rather than only the cells where detection is high."""
    cases, ctx = load_cases(PROFILE, **run)
    report = run_injection(cases, ctx)
    wrong = [t for t in report["trials"] if t["fault"] == "wrong_run_id" and t["target"] == "section"]
    assert wrong and not any(t["detected"] for t in wrong)
    assert report["by_tier"]["misattribution"]["rate"] < report["by_tier"]["fabrication"]["rate"]
    # an outline may state derived quantities (differences, standard errors), so its numbers
    # are not looked up; the gap stays visible in the per-stage table
    outline = report["by_fault_and_target"].get("fabrication/fabricated_number/outline")
    assert outline is None or outline["rate"] == 0.0


def test_a_case_rejected_before_injection_is_a_false_positive_not_a_detection(run):
    state = json.loads(run["writing"].read_text())
    state["sections"][0]["body"] += " A fabricated 9.8765 appears here."
    run["writing"].write_text(json.dumps(state))
    cases, ctx = load_cases(PROFILE, **run)
    report = run_injection(cases, ctx)
    assert report["controls"]["false_positives"] == 1
    assert not any(t["case"] == "section:Abstract" for t in report["trials"])


def test_ablation_attributes_detections_to_the_guard_that_fired(run):
    cases, ctx = load_cases(PROFILE, **run)
    row = run_injection(cases, ctx)["ablation"]["fabrication"]
    assert row["without_citation"]["detected"] < row["all_guards"]["detected"]
    assert row["without_number"]["detected"] < row["all_guards"]["detected"]
    assert "without_style" not in row                                  # no injected fault is stylistic


def test_classifier_tracks_the_production_messages(run):
    cases, ctx = load_cases(PROFILE, **run)
    seen = {c for t in run_injection(cases, ctx)["trials"] for c in t["categories"]}
    assert {"citation", "number", "structure"} <= seen
    assert classify("schema: 'x' is a required property") == "schema"
    assert classify("response was not valid JSON (boom)") == "invalid_json"
    assert classify("remove em-dashes (recast with a comma)") == "style"
    assert classify("unknown paper ids (cite only the bracketed ids given): ['arXiv:1']") == "citation"


def test_wilson_interval():
    assert wilson(0, 0) is None
    lo, hi = wilson(10, 10)
    assert 0.6 < lo < 1.0 and hi == 1.0
    assert wilson(0, 10)[0] == 0.0


class _Echo(Agent):
    name = "echo"
    schema = {"type": "object", "properties": {"ids": {"type": "array"}}, "required": ["ids"]}

    def build_prompt(self, ctx):
        return "p"

    def postcheck(self, output, ctx):
        bad = [i for i in output["ids"] if i != "ok"]
        return [f"unknown paper ids: {bad}"] if bad else []


def test_natural_faults_counts_first_attempt_rejections_and_recoveries(tmp_path):
    path = tmp_path / "x.agents.jsonl"
    trace = AgentTrace(path=path)
    _Echo(MockLLM(jsons=[{"ids": ["ok"]}]), trace=trace).run({})                       # clean
    _Echo(MockLLM(jsons=[{"ids": ["bad"]}, {"ids": ["ok"]}]), trace=trace).run({})     # recovered
    with pytest.raises(AgentError):                                                             # never recovers
        _Echo(MockLLM(jsons=[{"ids": ["bad"]}] * 3), trace=trace).run({})
    with path.open("a") as stream:
        stream.write("not json\n")
        stream.write(json.dumps({"call_id": "z", "agent": "echo", "attempt": 1, "event": "started"}) + "\n")
    report = natural_faults([path])
    echo = report["agents"]["echo"]
    assert report["corrupt_lines"] == 1
    assert (echo["calls"], echo["first_attempt_rejected"], echo["recovered"], echo["unrecovered"]) == (3, 2, 1, 1)
    assert echo["first_attempt_unfinished"] == 1
    assert echo["first_attempt_categories"] == {"citation": 2}
    assert echo["recovery_rate"] == 0.5 and echo["attempts"] == 6
    assert report["agents"]["ALL"]["calls"] == 3


def test_audit_paper_reports_without_rejecting(run):
    _, ctx = load_cases(PROFILE, **run)
    clean = audit_paper("The full run reaches 3.8222 over 5 seeds [exp_0003].", ctx.bank)
    assert not clean["would_be_rejected"] and clean["numbers_checked"] == 1 and clean["ids_cited"] == 1
    bad = audit_paper("We reach 1.2345 error [exp_0099], as in [arXiv:2401.00001].", ctx.bank)
    assert bad["ungrounded_numbers"] == ["1.2345"]
    assert bad["unknown_ids"] == ["arXiv:2401.00001", "exp_0099"] and bad["would_be_rejected"]


def test_baseline_keeps_interrupted_repeats_in_the_denominator(run, tmp_path):
    _, ctx = load_cases(PROFILE, **run)
    out = tmp_path / "baseline"
    out.mkdir()
    (out / "baseline_2.started").write_text("")                        # a repeat that died mid-call
    llm = MockLLM(texts=["Error is 3.8222 [exp_0003].", "Error is 2.2222 [exp_0003]."])
    report = run_baseline(llm, ctx.bank, "review", out, repeats=3)
    assert [p["status"] for p in report["papers"]] == ["complete", "interrupted", "complete"]
    assert report["summary"]["complete"] == 2 and report["summary"]["papers_with_violation"] == 1
    again = run_baseline(MockLLM(texts=[]), ctx.bank, "review", out, repeats=3)   # reuses saved papers
    assert again["summary"] == report["summary"]


def test_cli_writes_reports(run, tmp_path, capsys):
    out = tmp_path / "report"
    assert main(["inject", "--profile", "profiles/image-classification.yaml", "--run", str(run["research"]),
                 "--ledger", str(run["ledger"]), "--out", str(out)]) == 0
    assert "Detection by fault" in (out / "injection.md").read_text()
    assert json.loads((out / "injection.json").read_text())["version"] == 1
    assert main(["traces", "--traces", str(tmp_path), "--out", str(out)]) == 1   # no traces present
