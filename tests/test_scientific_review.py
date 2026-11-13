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
    kp = {"sections": [{"heading": h, "points": [{"claim": t, "evidence": [cid], "role": role}]}
                        for h, t, cid, role in [
                            ("Abstract", text, "claim_error", "result"),
                            ("Results", text, "claim_error", "result"),
                            ("Limitations", ARGUMENT["limitations"][0], "limit_1", "limitation")]]}
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
