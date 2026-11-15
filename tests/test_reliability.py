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


# ---- reviewed mode: guards bound to an approved scientific release ----------------------------
@pytest.fixture
def reviewed_run(tmp_path):
    from scholarloop.writing.session import WritingSession
    from tests.test_scientific_review import PANEL_PASS, released, setup, writing_outputs
    paths = setup(tmp_path)
    released(paths)
    profile, ledger, registry, out = paths
    outline, kp, bodies = writing_outputs()
    llm = MockLLM(jsons=[outline, *([PANEL_PASS] * 3), kp, *([PANEL_PASS] * 3), *bodies,
                         {"verdict": "pass", "issues": []}])
    w = WritingSession(llm, profile, out, ledger_path=ledger, registry_dir=registry)
    w.outline(), w.key_points(), w.draft()
    return profile, {"ledger": ledger, "registry": registry, "research": out, "writing": out / "writing.json"}


def test_reviewed_mode_uses_the_release_and_sees_sentence_local_misattribution(reviewed_run):
    profile, paths = reviewed_run
    cases, ctx = load_cases(profile, **paths)
    assert ctx.mode == "reviewed" and "claim_error" in ctx.statements
    report = run_injection(cases, ctx)
    assert report["mode"] == "reviewed" and report["controls"]["false_positives"] == 0
    by = report["by_fault_and_target"]
    assert by["fabrication/fabricated_statement_id/section"]["rate"] == 1.0
    assert by["fabrication/uncited_sentence/section"]["rate"] == 1.0
    assert by["fabrication/fabricated_number/section"]["rate"] == 1.0
    # a sentence about the measured error, re-cited to the limitation, no longer has its number
    wrong = [t for t in report["trials"] if t["fault"] == "wrong_statement_id" and t["target"] == "section"
             and "claim_error →" in t["change"]]
    assert wrong and all(t["detected"] for t in wrong)
    # key points and the outline must copy approved statements, so any edit to them is caught
    assert by["fabrication/fabricated_number/key_points"]["rate"] == 1.0


def test_a_release_that_no_longer_validates_falls_back_and_says_so(reviewed_run):
    profile, paths = reviewed_run
    with paths["ledger"].open("a") as stream:              # evidence changed after approval
        stream.write(paths["ledger"].read_text().splitlines()[0].replace("exp_0001", "exp_0002") + "\n")
    _, ctx = load_cases(profile, **paths)
    assert ctx.mode.startswith("legacy (the writing cites a release that no longer validates")


def test_a_writing_directory_apart_from_its_review_still_runs_in_reviewed_mode(reviewed_run, tmp_path):
    profile, paths = reviewed_run
    apart = tmp_path / "paper_elsewhere"
    apart.mkdir()
    (apart / "writing.json").write_text(paths["writing"].read_text())
    moved = {**paths, "writing": apart / "writing.json"}
    assert load_cases(profile, **moved)[1].mode.startswith("legacy")
    assert load_cases(profile, **moved, review_dir=paths["research"])[1].mode == "reviewed"


def test_a_reviewed_baseline_is_audited_against_the_whole_dossier_not_the_run_measurements(reviewed_run):
    from scholarloop.reliability import DossierEvidence, audit_paper
    profile, paths = reviewed_run
    _, ctx = load_cases(profile, **paths)
    evidence = DossierEvidence(ctx.bank.approved_dossier)
    assert "claim_error" not in evidence.to_prompt_block()          # the argument is not evidence
    # lr=0.1 is a recorded config value and 7.25 a measurement; 9.87 is written nowhere
    ok = audit_paper("With learning rate 0.1 the error is 7.25 percent [exp_0001].", evidence)
    assert not ok["would_be_rejected"]
    bad = audit_paper("The error is 9.87 percent [exp_0001] [arXiv:9999.00001].", evidence)
    assert bad["ungrounded_numbers"] == ["9.87"] and bad["unknown_ids"] == ["arXiv:9999.00001"]


def test_claim_audit_quotes_the_paper_blind_and_counts_claims_beyond_the_evidence(reviewed_run, tmp_path):
    from scholarloop.reliability import blind_paper, run_claim_audit
    profile, paths = reviewed_run
    _, ctx = load_cases(profile, **paths)
    staged = "The validation error is 7.25 percent [claim_error]."
    baseline = "The validation error is 7.25 percent. The recipe is robustly better."
    assert "claim_error" not in blind_paper(staged) and blind_paper(staged).endswith("percent.")
    ok = {"quote": "The validation error is 7.25 percent.", "kind": "observation", "verdict": "supported",
          "reason": "The run records it.", "evidence": ["exp_0001"]}
    over = {"quote": "The recipe is robustly better.", "kind": "interpretation", "verdict": "overstated",
            "reason": "One run records no comparison.", "evidence": ["exp_0001"]}
    invented = {**over, "quote": "The recipe always wins."}          # not in the paper: sent back
    llm = MockLLM(jsons=[{"claims": [ok]}, {"claims": [ok, invented]}, {"claims": [ok, over]}])
    report = run_claim_audit(llm, ctx.bank.approved_dossier, {"staged": staged, "baseline_1": baseline},
                             tmp_path / "claims", passes=1)
    assert report["writers"]["staged"]["beyond_evidence"] == 0 and report["writers"]["staged"]["claims"] == 1
    assert report["writers"]["baseline"]["beyond_evidence"] == 1 and report["writers"]["baseline"]["claims"] == 2
    assert "claim_error" not in ClaimBlock(ctx.bank.approved_dossier)      # the argument is withheld
    again = run_claim_audit(MockLLM(jsons=[]), ctx.bank.approved_dossier,
                            {"staged": staged, "baseline_1": baseline}, tmp_path / "claims", passes=1)
    assert again["rows"] == report["rows"]                                 # finished audits are reused


def ClaimBlock(dossier):
    from scholarloop.reliability import ClaimAuditor
    return ClaimAuditor(MockLLM(jsons=[]), dossier)._block


def test_semantic_study_counts_a_fault_caught_only_when_the_objection_quotes_it(reviewed_run, tmp_path):
    import json
    from scholarloop.reliability import _changed_sentences, _names, run_semantic
    profile, paths = reviewed_run
    cases, ctx = load_cases(profile, **paths)
    state = json.loads(paths["writing"].read_text())
    assert _changed_sentences("A is lower. B holds.", "A is higher. B holds.") == ["A is higher."]
    assert _names("Results: 'the error is higher than the baseline here' flips the direction",
                  "The error is higher than the baseline here [claim_error].")
    assert not _names("Results: something unrelated was changed", "The error is higher than the baseline here.")
    passing = {"verdict": "pass", "issues": []}
    vague = {"verdict": "revise", "issues": ["Draft: something is off"]}       # names no sentence
    llm = MockLLM(jsons=[passing, vague] + [vague] * 50)
    report = run_semantic(llm, cases, ctx, state, tmp_path / "sem", per_fault=2, controls=2)
    assert report["controls"] == {**report["controls"], "reads": 2, "with_objection": 1}
    assert report["all"]["detected"] == 0          # an objection that quotes nothing catches nothing


def test_ablation_writes_each_arm_and_records_a_writer_that_stops(reviewed_run, tmp_path):
    from scholarloop.reliability import DossierEvidence, run_ablation
    from tests.test_scientific_review import PANEL_PASS, writing_outputs
    profile, paths = reviewed_run
    _, ctx = load_cases(profile, **paths)
    evidence = DossierEvidence(ctx.bank.approved_dossier)
    paths = {**paths, "review_dir": paths["writing"].parent, "research": None}   # as the fixture reviewed it
    outline, kp, bodies = writing_outputs()
    script = [outline, *([PANEL_PASS] * 3), kp, *([PANEL_PASS] * 3), *bodies, {"verdict": "pass", "issues": []}]
    out = tmp_path / "ablation"
    full = run_ablation(MockLLM(jsons=list(script)), profile, paths, out, "full", repeats=1, evidence=evidence)
    assert full["completed"] == 1 and (out / "full_1.md").exists()
    assert full["papers"][0]["provenance"]["ungrounded_numbers"] == []
    # Without the deterministic checks an invented number in a section is accepted as written.
    invented = [{"body": b["body"].replace("7.25", "3.14")} for b in bodies]
    loose = [outline, *([PANEL_PASS] * 3), kp, *([PANEL_PASS] * 3), *invented, {"verdict": "pass", "issues": []}]
    bare = run_ablation(MockLLM(jsons=list(loose)), profile, paths, out, "no_guards", repeats=1, evidence=evidence)
    assert bare["completed"] == 1 and "3.14" in (out / "no_guards_1.md").read_text()
    assert bare["papers"][0]["provenance"]["ungrounded_numbers"]
    # The production writer refuses the same script and is recorded as stopped, with nothing published.
    stopped = run_ablation(MockLLM(jsons=list(loose) + invented * 3), profile, paths, out / "strict", "full",
                           repeats=1, evidence=evidence)
    assert stopped["completed"] == 0 and stopped["papers"][0]["stopped_at"] == "draft"
    assert not (out / "strict" / "full_1.md").exists()
    again = run_ablation(MockLLM(jsons=[]), profile, paths, out, "full", repeats=1, evidence=evidence)
    assert again["papers"] == full["papers"]                               # finished papers are reused
