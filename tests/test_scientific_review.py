"""Scientific gate integration: failure/staleness cannot turn into paper generation."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scholarloop.core.agent import AgentError
from scholarloop.core.ledger import Hypothesis, Ledger, LedgerEntry
from scholarloop.core.llm import MockLLM
from scholarloop.core.profile import load_profile
from scholarloop.core.registry import VerifiedRegistry
from scholarloop.review.contracts import ReviewBlocked
from scholarloop.review.dossier import audit_prose, build_dossier, snapshot, validate_argument
from scholarloop.review.session import ScientificReviewSession, main
from scholarloop.writing.paper import PaperPipeline
from scholarloop.writing.session import WritingSession

ROOT = Path(__file__).resolve().parents[1]
CHECK = {"claim_id": "claim_error", "question": "Does the recorded measurement support this scoped observation?",
         "answer": "The owning run records this value; no mechanism is asserted.",
         "status": "supported_in_scope", "evidence": ["exp_0001"]}
PASS = {"verdict": "pass", "summary": "No unresolved concerns in the stated scope.",
        "issues": [], "dispositions": [], "claim_checks": [CHECK]}
ARGUMENT = {
    "question": "How does this recipe behave on the fixed validation split?",
    "method_summary": "We evaluate an MLP using the frozen scorer.",
    "claims": [{"id": "claim_error", "text": "The validation error is 7.25 percent.",
                "kind": "observation", "evidence": ["exp_0001"],
                "measurements": ["exp_0001::val_top1_err.full"], "depends_on": [],
                "scope": "This recipe and validation split only."}],
    "limitations": ["This observation does not establish a causal mechanism."],
    "setup": [{"id": "setup_scorer", "text": "Runs are scored by the frozen scorer.", "evidence": ["scorer"]}],
}
ISSUE = {"target": "claim_error", "severity": "blocker", "observation": "The matched control is missing.",
         "evidence": ["exp_0001"], "required_action": "Run the compute-matched control.",
         "resolution_criterion": "Provide the control result and comparison.", "action": "experiment"}
REVISE = {"verdict": "revise", "summary": "A control is required.", "issues": [ISSUE], "dispositions": [],
          "claim_checks": [{**CHECK, "status": "concern", "answer": "A matched control is missing."}]}
PANEL_PASS = {"verdict": "accept", "issues": []}
PANEL_REVISE = {"verdict": "revise", "issues": [{"target": "Results", "issue": "unclear",
                                                "suggestion": "clarify the scope"}]}


def setup(tmp_path):
    profile_path = tmp_path / "profiles" / "image-classification.yaml"
    profile_path.parent.mkdir()
    profile_path.write_text((ROOT / "profiles/image-classification.yaml").read_text())
    profile = load_profile(profile_path)
    for relative in (profile.train_entrypoint, profile.fixed_module):
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text((ROOT / relative).read_text())
    ledger, registry, out = tmp_path / "experiments.jsonl", tmp_path / "registry", tmp_path / "review"
    e = LedgerEntry("exp_0001", profile.name, Hypothesis("The recipe may help", "paper:source"),
                    profile.metric.name, config={"lr": 0.1}, fidelity=["full"],
                    metric={"full": 7.25, "seeds": [7.0, 7.5]}, verdict="discarded", registry_id="exp_0001")
    Ledger(ledger).append(e)
    r = VerifiedRegistry(registry / "exp_0001.json", exp_id="exp_0001")
    r.capture("val_top1_err.full", 7.25, seeds=[7.0, 7.5], std=0.3536)
    r.save()
    return profile, ledger, registry, out


def session(paths, llm=None):
    p, led, reg, out = paths
    return ScientificReviewSession(llm or MockLLM(jsons=[PASS] * 4), p, out,
                                    ledger_path=led, registry_dir=reg)


def released(paths):
    s = session(paths)
    state = s.review(argument=copy.deepcopy(ARGUMENT))
    assert state["status"] == "READY_FOR_AUTHOR"
    s.approve(author="Researcher", review_hash=state["review_hash"])
    return s


def writing_outputs():
    text = ARGUMENT["claims"][0]["text"]
    outline = {"title": "Recipe evaluation", "question": ARGUMENT["question"], "nugget": text,
               "contributions": [text], "sections": [
                   {"heading": "Abstract", "purpose": "observation", "evidence": ["claim_error"]},
                   {"heading": "Results", "purpose": "measurement", "evidence": ["claim_error"]},
                   {"heading": "Limitations", "purpose": "scope", "evidence": ["limit_1"]}]}
    kp = {"sections": [{"heading": h, "points": [{"statement": cid, "role": role}]}   # ids; text is filled in
                        for h, cid, role in [("Abstract", "claim_error", "result"),
                                             ("Results", "claim_error", "result"),
                                             ("Limitations", "limit_1", "limitation")]]}
    bodies = [{"body": "The validation error is 7.25 percent [claim_error]."}] * 2 + [
        {"body": "This observation does not establish a causal mechanism [limit_1]."}]
    return outline, kp, bodies


def test_no_release_means_zero_writer_calls(tmp_path):
    paths = setup(tmp_path)
    p, led, reg, out = paths
    llm = MockLLM()
    w = WritingSession(llm, p, out, ledger_path=led, registry_dir=reg)
    with pytest.raises(ReviewBlocked):
        w.outline()
    with pytest.raises(ReviewBlocked):
        PaperPipeline(llm, llm, registry_dir=reg).run(list(Ledger(led).read_all()))
    assert llm.calls == [] and not (out / "paper.md").exists()


def test_negative_result_can_be_reviewed_approved_and_written(tmp_path):
    paths = setup(tmp_path)
    s = released(paths)
    dossier, release = s.require_release()
    assert dossier["runs"][0]["verdict"] == "discarded"
    outline, kp, bodies = writing_outputs()
    llm = MockLLM(jsons=[outline, *([PANEL_PASS] * 3), kp, *([PANEL_PASS] * 3),
                         *bodies, {"verdict": "pass", "issues": []}])
    p, led, reg, out = paths
    w = WritingSession(llm, p, out, ledger_path=led, registry_dir=reg)
    w.outline()
    w.key_points()
    w.draft()
    assert w.state.release_id == release["id"]
    assert "7.25 percent [claim_error]" in (out / "paper.md").read_text()


def test_dissent_blocks_approval_and_records_experiment_request(tmp_path):
    paths = setup(tmp_path)
    s = session(paths, MockLLM(jsons=[PASS, REVISE, PASS, PASS]))
    state = s.review(argument=ARGUMENT)
    assert state["status"] == "NEEDS_EXPERIMENTS"
    with pytest.raises(ReviewBlocked):
        s.approve(author="Researcher", review_hash=state["review_hash"])
    assert json.loads((s.out / "experiment_requests.json").read_text())[0]["issue_id"] == "r1:experimental:1"
    assert all("No unresolved concerns" not in c["prompt"] for c in s.llm.calls)  # no peer reviews


def test_previous_issue_cannot_silently_disappear(tmp_path):
    paths = setup(tmp_path)
    s = session(paths, MockLLM(jsons=[PASS, REVISE, PASS, PASS]))
    s.review(argument=ARGUMENT)
    s.llm = MockLLM(jsons=[PASS, PASS, PASS, PASS])
    with pytest.raises(AgentError):
        s.review(argument=ARGUMENT)
    assert s.load()["status"] == "REVIEW_FAILED"
    assert s.load()["issues"][0]["status"] == "open"
    with pytest.raises(ReviewBlocked):
        s.require_release()


def test_reframing_requires_explicit_issue_disposition(tmp_path):
    paths = setup(tmp_path)
    s = session(paths, MockLLM(jsons=[PASS, REVISE, PASS, PASS]))
    s.review(argument=ARGUMENT)
    updated = copy.deepcopy(ARGUMENT)
    updated["limitations"].append("A matched control remains unavailable, so no comparison is claimed.")
    disposition = {**PASS, "dispositions": [{"issue_id": "r1:experimental:1", "status": "out_of_scope",
                                            "reason": "The revised argument explicitly makes no comparison.",
                                            "evidence": ["claim_error", "exp_0001"]}]}
    s.llm = MockLLM(jsons=[PASS, disposition, PASS, PASS])
    state = s.review(argument=updated, feedback="Restrict the paper to the observation.")
    assert state["status"] == "READY_FOR_AUTHOR"
    assert state["issues"][0]["resolution"]["round"] == 2
    assert list((s.out / "review_history").glob("*.json"))


@pytest.mark.parametrize("change", ["ledger", "registry", "train", "scorer", "profile", "argument"])
def test_changes_invalidate_release_even_for_resumed_sessions(tmp_path, change):
    paths = setup(tmp_path)
    s = released(paths)
    p, led, reg, out = paths
    file = {"ledger": led, "registry": reg / "exp_0001.json", "train": p.root / p.train_entrypoint,
            "scorer": p.root / p.fixed_module, "profile": p.source_path,
            "argument": out / s.STATE_FILE}[change]
    if change == "argument":
        data = json.loads(file.read_text())
        data["dossier"]["argument"]["claims"][0]["text"] = "Unsupported interpretation."
        file.write_text(json.dumps(data))
    else:
        file.write_text(file.read_text() + "\n")
    with pytest.raises(ReviewBlocked):
        session(paths).require_release()


def test_failed_rereview_revokes_earlier_author_approval(tmp_path):
    s = released(setup(tmp_path))
    s.llm = MockLLM(jsons=[PASS])
    with pytest.raises(AssertionError):
        s.review(argument=ARGUMENT)
    with pytest.raises(ReviewBlocked):
        s.require_release()
    assert s.load()["release"] is None


def test_approval_requires_exact_review_hash(tmp_path):
    s = session(setup(tmp_path))
    s.review(argument=ARGUMENT)
    with pytest.raises(ReviewBlocked):
        s.approve(author="Researcher", review_hash="old-review")
    with pytest.raises(ReviewBlocked):
        s.require_release()


def test_claim_measurements_cannot_be_borrowed_or_small_integer_exempt(tmp_path):
    p, led, reg, out = setup(tmp_path)
    d = build_dossier(snapshot(p, led, reg))
    d["measurements"]["exp_0002::val_top1_err.full"] = {"run_id": "exp_0002", "key": "val_top1_err.full",
                                                        "value": 2, "seeds": [], "std": None}
    bad = copy.deepcopy(ARGUMENT)
    bad["claims"][0]["text"] = "The validation error is 2 percent."
    assert validate_argument(bad, d)
    bad["claims"][0]["measurements"] = ["exp_0002::val_top1_err.full"]
    assert any("uncited run" in e for e in validate_argument(bad, d))


def test_prose_numeric_grounding_is_sentence_local(tmp_path):
    s = released(setup(tmp_path))
    d, _ = s.require_release()
    assert audit_prose("The error is 7.25 percent [claim_error].", d) == []
    assert audit_prose("The error is 2 percent [claim_error].", d)
    assert audit_prose("The error is 7.25 percent [scope_method].", d)
    assert audit_prose("The error is 7.25 percent.", d)


def test_panel_round_cap_cannot_advance(tmp_path):
    paths = setup(tmp_path)
    released(paths)
    p, led, reg, out = paths
    outline, kp, _ = writing_outputs()
    llm = MockLLM(jsons=[outline, PANEL_REVISE, PANEL_PASS, PANEL_PASS])
    w = WritingSession(llm, p, out, ledger_path=led, registry_dir=reg, max_rounds=0)
    w.outline()
    with pytest.raises(ReviewBlocked, match="Panel concerns"):
        w.key_points()
    assert len(llm.calls) == 4


def test_semantic_draft_failure_does_not_publish(tmp_path):
    paths = setup(tmp_path)
    released(paths)
    p, led, reg, out = paths
    outline, kp, bodies = writing_outputs()
    llm = MockLLM(jsons=[outline, *([PANEL_PASS] * 3), kp, *([PANEL_PASS] * 3), *bodies,
                         {"verdict": "revise", "issues": ["Wrong interpretation."]}])
    w = WritingSession(llm, p, out, ledger_path=led, registry_dir=reg)
    w.outline()
    w.key_points()
    with pytest.raises(ReviewBlocked, match="changed approved claims"):
        w.draft()
    assert not (out / "paper.md").exists() and w.stage == "keypointed"


def test_corrupt_or_missing_measurements_fail_closed(tmp_path):
    paths = setup(tmp_path)
    p, led, reg, out = paths
    (reg / "exp_0001.json").unlink()
    s = session(paths, MockLLM())
    state = s.review(argument=ARGUMENT)
    assert state["status"] == "INSUFFICIENT_EVIDENCE" and s.llm.calls == []
    led.write_text(led.read_text() + "broken json\n")
    with pytest.raises(ReviewBlocked):
        s.review(argument=ARGUMENT)


def test_one_shot_uses_same_gate_and_accepts_negative_evidence(tmp_path):
    paths = setup(tmp_path)
    s = released(paths)
    p, led, reg, out = paths
    writer = MockLLM(jsons=[{"title": "Observation", "abstract": "The error is 7.25 percent [claim_error].",
                             "sections": [{"heading": "Limitations", "body":
                                           "No causal mechanism is established [limit_1]."}]}])
    reviewer = MockLLM(jsons=[{"verdict": "pass", "issues": []},
                               {"summary": "Limited observation", "strengths": [], "weaknesses": [],
                                "score": 5, "recommendation": "weak_accept"}])
    result = PaperPipeline(writer, reviewer, registry_dir=reg, scientific_review=s).run(list(Ledger(led).read_all()))
    assert result["release_id"] and not result["demo"]
    assert "claim_error" in writer.calls[0]["prompt"]


def test_status_and_approval_cli_need_no_api_key(tmp_path, monkeypatch, capsys):
    paths = setup(tmp_path)
    s = session(paths)
    state = s.review(argument=ARGUMENT)
    p, led, reg, out = paths
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    flags = ["--profile", str(p.source_path), "--ledger", str(led), "--registry", str(reg), "--out", str(out)]
    assert main(["status", *flags]) == 0
    assert main(["approve", *flags, "--author", "Researcher", "--review-hash", state["review_hash"]]) == 0
    assert "READY_TO_WRITE" in capsys.readouterr().out


def test_release_change_during_first_outline_cannot_rebind_old_output(tmp_path):
    paths = setup(tmp_path)
    s = released(paths)
    p, led, reg, out = paths
    outline, _, _ = writing_outputs()
    outputs = iter([outline, PANEL_PASS, PANEL_PASS, PANEL_PASS])
    def reply(prompt):
        output = next(outputs)
        if output is outline:
            # A separate review session signs off a new scope while drafting is in flight.
            changed = copy.deepcopy(ARGUMENT)
            changed["limitations"].append("This observation is exploratory.")
            s.llm = MockLLM(jsons=[PASS] * 4)
            state = s.review(argument=changed)
            s.approve(author="Researcher", review_hash=state["review_hash"])
        return output
    w = WritingSession(MockLLM(router=reply), p, out, ledger_path=led, registry_dir=reg)
    with pytest.raises(ReviewBlocked, match="older"):
        w.outline()
    assert not (out / "outline.md").exists()


def test_budget_exhaustion_revokes_release_and_keeps_writer_blocked(tmp_path):
    from scholarloop.core.call_budget import CallLimitedLLM, CallBudgetExceeded
    paths = setup(tmp_path)
    s = released(paths)
    s.llm = CallLimitedLLM(MockLLM(jsons=[PASS] * 4), 2)
    with pytest.raises(CallBudgetExceeded):
        s.review(argument=ARGUMENT)
    assert s.load()["status"] == "REVIEW_FAILED" and s.load()["release"] is None
    with pytest.raises(ReviewBlocked):
        s.require_release()


def test_changed_research_brief_invalidates_writing_release(tmp_path):
    p, led, reg, out = setup(tmp_path)
    research = tmp_path / "research"
    research.mkdir()
    brief = research / "research_brief.json"
    brief.write_text((ROOT / "examples/research_brief.json").read_text())
    s = ScientificReviewSession(MockLLM(jsons=[PASS] * 4), p, out,
                                ledger_path=led, registry_dir=reg, research_dir=research)
    state = s.review(argument=ARGUMENT)
    s.approve(author="Researcher", review_hash=state["review_hash"])
    brief.write_text(brief.read_text().replace("CPU only", "GPU replication required"))
    with pytest.raises(ReviewBlocked, match="stale"):
        s.require_release()


def test_profile_path_alias_does_not_invalidate_the_same_evidence(tmp_path, monkeypatch):
    paths = setup(tmp_path)
    s = released(paths)
    monkeypatch.chdir(tmp_path)
    p, led, reg, out = paths
    alias = load_profile("profiles/image-classification.yaml")
    resumed = ScientificReviewSession(None, alias, out, ledger_path=led, registry_dir=reg)
    assert resumed.require_release()[1]["id"] == s.require_release()[1]["id"]


def test_consistently_wrong_aggregate_is_recomputed_before_critic_calls(tmp_path):
    paths = setup(tmp_path)
    p, led, reg, out = paths
    entry = json.loads(led.read_text())
    entry["metric"]["full"] = 9.0
    led.write_text(json.dumps(entry) + "\n")
    path = reg / "exp_0001.json"
    measured = json.loads(path.read_text())
    measured["measurements"]["val_top1_err.full"]["value"] = 9.0
    path.write_text(json.dumps(measured))
    s = session(paths, MockLLM())
    state = s.review(argument=ARGUMENT)
    assert state["status"] == "INSUFFICIENT_EVIDENCE"
    assert any("aggregate disagrees" in reason for reason in state["dossier"]["checks"])
    assert not s.llm.calls


# ---- what a reviewer is sent ------------------------------------------------------------------
def test_reviewers_share_one_dossier_block_and_one_system_prompt(tmp_path):
    paths = setup(tmp_path)
    llm = MockLLM(jsons=[ARGUMENT, *([PASS] * 4)])
    session(paths, llm).review()
    critics = llm.calls[1:]
    assert len({c["context"] for c in critics}) == 1 and len({c["system"] for c in critics}) == 1
    lenses = [json.loads(c["prompt"])["review_lens"]["role"] for c in critics]
    assert lenses == ["implementation", "experimental", "falsification", "argument"]
    assert "dossier" not in json.loads(critics[0]["prompt"])           # the tail stays small
    assert llm.calls[0]["context"] != critics[0]["context"]            # the builder has no argument yet


def test_prompt_view_states_repeated_text_once_and_leaves_the_dossier_alone(tmp_path):
    from scholarloop.review.dossier import build_dossier, evidence_ids, prompt_view
    dossier = build_dossier(session(setup(tmp_path)).inputs())
    train = dossier["current_code"]["train"]
    path = dossier["profile"]["train_entrypoint"]
    edited = train + "\n# one changed line\n"
    dossier["candidates"] = {"a": {"id": "a", "sources": {path: train}},
                             "b": {"id": "b", "sources": {path: edited}},
                             "c": {"id": "c", "sources": {path: train}}}
    paper = {"arxiv_id": "arXiv:1", "title": "t", "summary": "s"}
    dossier["papers"] = {"arXiv:1": paper}
    dossier["research"] = {"lit_review.json": {"papers": [paper, {"arxiv_id": "arXiv:2", "title": "other"}]}}
    before = json.dumps(dossier, sort_keys=True)
    view = prompt_view(dossier)
    assert json.dumps(dossier, sort_keys=True) == before
    versions = [view["candidates"][k]["source_versions"][path] for k in "abc"]
    assert versions[0] == versions[2] != versions[1]
    assert view["source_files"][versions[0]] == train
    assert view["source_files"][versions[1]]["unified_diff_against"] == versions[0]
    assert "+# one changed line" in view["source_files"][versions[1]]["diff"]
    assert json.dumps(view).count(json.dumps(train)[1:-1]) == 1
    assert view["research"]["lit_review.json"]["papers"] == ["arXiv:1", {"arxiv_id": "arXiv:2", "title": "other"}]
    assert view["current_code"]["train"] == versions[0]
    assert set(view["citable_ids"]["evidence"]) == evidence_ids(dossier)
    assert view["measurements"] == dossier["measurements"] and view["runs"] == dossier["runs"]


def test_a_claim_may_state_the_seed_values_of_a_measurement_it_binds(tmp_path):
    dossier = build_dossier(session(setup(tmp_path)).inputs())
    ref = "exp_0001::val_top1_err.full"
    claim = {**ARGUMENT["claims"][0], "measurements": [ref],
             "text": "The error is 7.25 percent, from seeds at 7.0 and 7.5."}
    assert validate_argument({**ARGUMENT, "claims": [claim]}, dossier) == []
    unbound = {**claim, "measurements": [], "text": "One seed reached 7.5 and another 9.9."}
    (error,) = validate_argument({**ARGUMENT, "claims": [unbound]}, dossier)
    assert f"7.5 is recorded by ['{ref}']" in error and "9.9 is in no measurement" in error


# ---- statements a paper needs besides conclusions ---------------------------------------------
def _supported(tmp_path, setup_text="The reference run used a learning rate of 0.1.", evidence=("exp_0001",)):
    dossier = build_dossier(session(setup(tmp_path)).inputs())
    dossier["papers"] = {"arXiv:1": {"arxiv_id": "arXiv:1", "title": "Warm restarts", "summary": "Uses 200 epochs."}}
    argument = {**ARGUMENT,
                "setup": [{"id": "setup_lr", "text": setup_text, "evidence": list(evidence)}],
                "related_work": [{"id": "related_sgdr", "text": "Warm restarts were studied over 200 epochs.",
                                  "papers": ["arXiv:1"]}]}
    return argument, dossier


def test_setup_and_related_statements_are_approved_statements_a_draft_can_cite(tmp_path):
    from scholarloop.review.dossier import approved_statements
    argument, dossier = _supported(tmp_path)
    assert validate_argument(argument, dossier) == []
    dossier["argument"] = argument
    statements = approved_statements(dossier)
    assert statements["setup_lr"]["evidence"] == ["exp_0001"]
    assert statements["related_sgdr"]["evidence"] == ["arXiv:1"]
    assert audit_prose("Training used a learning rate of 0.1 [setup_lr]. "
                       "Earlier work ran 200 epochs [related_sgdr].", dossier) == []
    assert audit_prose("Training used a learning rate of 0.3 [setup_lr].", dossier)


def test_a_setting_must_be_written_in_the_record_it_cites(tmp_path):
    argument, dossier = _supported(tmp_path)

    def errors(kind, field, value):
        changed = copy.deepcopy(argument)
        changed[kind][0][field] = value
        return validate_argument(changed, dossier)

    (error,) = errors("setup", "text", "The reference run used a learning rate of 0.25.")
    assert "setup_lr: numbers ['0.25'] are not written in ['exp_0001']" in error
    assert "setup_lr: unknown evidence ['exp_9999']" in errors("setup", "evidence", ["exp_9999"])
    assert "related_sgdr: unknown papers ['arXiv:404']" in errors("related_work", "papers", ["arXiv:404"])
    twice = copy.deepcopy(argument)
    twice["setup"].append(dict(twice["setup"][0]))
    assert "setup_lr: statement ids must be unique" in validate_argument(twice, dossier)


def test_key_points_are_chosen_by_id_and_filled_in_from_the_release(tmp_path):
    paths = setup(tmp_path)
    released(paths)
    p, led, reg, out = paths
    outline, kp, _ = writing_outputs()
    unknown = {"sections": [{**s, "points": [{"statement": "claim_invented", "role": "result"}]}
                            for s in kp["sections"]]}
    llm = MockLLM(jsons=[outline, *([PANEL_PASS] * 3), unknown, kp, *([PANEL_PASS] * 3)])
    w = WritingSession(llm, p, out, ledger_path=led, registry_dir=reg)
    w.outline()
    points = w.key_points()["sections"][0]["points"]
    assert points == [{"claim": ARGUMENT["claims"][0]["text"], "evidence": ["claim_error"], "role": "result"}]
    assert "'claim_invented' is not an approved statement id" in llm.calls[5]["prompt"]   # the retry said why


def test_an_outline_section_with_no_approved_statement_is_rejected_at_the_outline(tmp_path):
    paths = setup(tmp_path)
    released(paths)
    p, led, reg, out = paths
    outline, _, _ = writing_outputs()
    hollow = {**outline, "sections": [*outline["sections"],
                                      {"heading": "Related Work", "purpose": "context", "evidence": []}]}
    llm = MockLLM(jsons=[hollow, outline, *([PANEL_PASS] * 3)])
    WritingSession(llm, p, out, ledger_path=led, registry_dir=reg).outline()
    assert "sections ['Related Work'] list no approved statement" in llm.calls[1]["prompt"]


def test_a_failure_after_a_panel_round_keeps_that_round(tmp_path):
    paths = setup(tmp_path)
    released(paths)
    p, led, reg, out = paths
    outline, _, _ = writing_outputs()
    # round one is complete and asks for a revision; the revision then fails
    llm = MockLLM(jsons=[outline, PANEL_REVISE, PANEL_PASS, PANEL_PASS, *([{"title": "broken"}] * 3)])
    w = WritingSession(llm, p, out, ledger_path=led, registry_dir=reg, max_rounds=1)
    with pytest.raises(AgentError):
        w.outline()
    saved = json.loads((out / "writing.json").read_text())
    assert saved["stage"] == "outlined" and len(saved["outline_discussion"]) == 1
    assert saved["outline_discussion"][0]["accepted"] is False
    with pytest.raises(ReviewBlocked, match="Panel concerns"):          # and it still cannot advance
        WritingSession(MockLLM(), p, out, ledger_path=led, registry_dir=reg).key_points()


# ---- can this argument support a paper? asked before reviewers and before writing -----------
def test_an_argument_of_conclusions_alone_is_refused_before_any_reviewer_is_called(tmp_path):
    from scholarloop.review.critics import ArgumentBuilder
    from scholarloop.review.dossier import paper_readiness
    paths = setup(tmp_path)
    bare = {k: v for k, v in ARGUMENT.items() if k != "setup"}
    llm = MockLLM(jsons=[PASS] * 4)
    with pytest.raises(ReviewBlocked, match="cannot support a paper: no setup statements"):
        session(paths, llm).review(argument=bare)
    assert llm.calls == []
    dossier = build_dossier(session(paths).inputs())
    assert paper_readiness(ARGUMENT, dossier) == []
    dossier["papers"] = {"arXiv:1": {"arxiv_id": "arXiv:1", "title": "t", "summary": "s"}}
    (gap,) = paper_readiness(ARGUMENT, dossier)                    # papers were reviewed: relate to them
    assert gap.startswith("no related_work statements")
    builder = ArgumentBuilder(MockLLM(jsons=[bare, ARGUMENT]))     # the builder is sent back for them
    dossier["papers"] = {}
    assert builder.run({"dossier": dossier}) == ARGUMENT
    assert "no setup statements" in builder.llm.calls[1]["prompt"]


def test_a_release_whose_argument_cannot_support_a_paper_does_not_start_writing(tmp_path):
    from scholarloop.review.contracts import digest
    paths = setup(tmp_path)
    s = released(paths)
    state = s.load()
    del state["dossier"]["argument"]["setup"]          # a release approved before setup statements existed
    state["review_hash"] = s._review_hash(state)
    state["release"]["review_hash"] = state["review_hash"]
    state["release"]["dossier_hash"] = digest(state["dossier"])
    s._save(state)
    with pytest.raises(ReviewBlocked, match="the approved argument cannot support a paper"):
        s.require_release()


def test_review_cost_estimate_scales_with_the_dossier_and_the_model(tmp_path):
    from scholarloop.review.session import estimate_review_cost
    dossier = build_dossier(session(setup(tmp_path)).inputs())
    dossier["argument"] = ARGUMENT
    small = estimate_review_cost(dossier, "claude-opus-5-5")
    big = estimate_review_cost({**dossier, "provenance_note": "x" * 500_000}, "claude-opus-5-5")
    assert 0 < small < big
    assert estimate_review_cost(dossier, "claude-opus-5-5", build_argument=False) < small
    assert estimate_review_cost(dossier, "claude-sonnet-5-5") < small
    assert estimate_review_cost(dossier, "no-such-model") is None


def test_a_rejected_id_says_what_is_wrong_and_what_to_write(tmp_path):
    from scholarloop.review.dossier import evidence_ids, explain_unknown
    dossier = build_dossier(session(setup(tmp_path)).inputs())
    known = evidence_ids(dossier)
    note = explain_unknown(["exp_0001::val_top1_err.full", "exp_001", "zzz"], known, dossier)
    assert "'exp_0001::val_top1_err.full' is a measurement reference, not evidence: cite its run 'exp_0001'" in note
    assert "'exp_001' is not an id in the dossier; did you mean 'exp_0001'?" in note
    assert "'zzz' is not an id in the dossier; use a run id" in note
    claim = {**ARGUMENT["claims"][0], "evidence": ["exp_0001", "exp_001"]}
    (error,) = validate_argument({**ARGUMENT, "claims": [claim]}, dossier)
    assert error.startswith("claim_error: unknown evidence: 'exp_001' is not an id")


def test_a_claim_may_quote_what_a_cited_run_reported_about_itself(tmp_path):
    dossier = build_dossier(session(setup(tmp_path)).inputs())
    dossier["runs"][0]["metric"]["self_reported"] = {"switch_step": [700.0, 900.0]}
    claim = {**ARGUMENT["claims"][0], "text": "The error is 7.25 percent; the run reported switching at step 700."}
    assert validate_argument({**ARGUMENT, "claims": [claim]}, dossier) == []
    other = {**claim, "text": "The error is 7.25 percent; the run reported switching at step 800."}
    assert "numbers lack explicitly bound measurements: ['800']" in validate_argument({**ARGUMENT, "claims": [other]}, dossier)[0]


def test_a_round_that_failed_part_way_is_continued_not_repeated(tmp_path):
    paths = setup(tmp_path)
    s = session(paths, MockLLM(jsons=[PASS, REVISE]))          # two reviewers answer, the third call fails
    with pytest.raises(AssertionError):
        s.review(argument=ARGUMENT)
    failed = s.load()
    assert failed["status"] == "REVIEW_FAILED" and [r["role"] for r in failed["reports"]] == ["implementation", "experimental"]
    resumed = MockLLM(jsons=[PASS, PASS])
    state = session(paths, resumed).review(argument=ARGUMENT)
    assert len(resumed.calls) == 2 and state["round"] == failed["round"]          # only the two missing roles
    assert [r["role"] for r in state["reports"]] == ["implementation", "experimental", "falsification", "argument"]
    assert state["resumed_roles"] == ["implementation", "experimental"]
    assert state["status"] == "NEEDS_EXPERIMENTS"                                  # the earlier dissent still counts
    assert [i["id"] for i in state["issues"]] == ["r1:experimental:1"]


def test_a_failed_round_is_not_continued_when_the_argument_or_feedback_changed(tmp_path):
    paths = setup(tmp_path)
    s = session(paths, MockLLM(jsons=[PASS]))
    with pytest.raises(AssertionError):
        s.review(argument=ARGUMENT)
    fresh = MockLLM(jsons=[PASS] * 4)
    state = session(paths, fresh).review(argument=ARGUMENT, feedback="A new instruction.")
    assert len(fresh.calls) == 4 and state["round"] == 2 and "resumed_roles" not in state


def test_a_claim_may_state_the_size_of_a_negative_difference_in_words(tmp_path):
    dossier = build_dossier(session(setup(tmp_path)).inputs())
    dossier["measurements"]["exp_0001::delta"] = {"run_id": "exp_0001", "key": "delta", "value": -4.96,
                                                  "seeds": [-6.2, 1.5], "std": None}
    claim = {**ARGUMENT["claims"][0], "measurements": ["exp_0001::delta"],
             "text": "The arm was lower by 4.96 on average and by 6.2 on one seed."}
    assert validate_argument({**ARGUMENT, "claims": [claim]}, dossier) == []
    wrong = {**claim, "text": "The arm was lower by 5.96 on average."}
    assert validate_argument({**ARGUMENT, "claims": [wrong]}, dossier)
