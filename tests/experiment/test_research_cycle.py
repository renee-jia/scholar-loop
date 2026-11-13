import copy
import json

import pytest

from scholarloop.core.agent import AgentError
from scholarloop.core.llm import MockLLM
from scholarloop.experiment.research_cycle import ResearchCycle
from scholarloop.review.contracts import ReviewBlocked
from tests.test_scientific_review import setup

ASSESSMENT = {
    "question": "Does the mechanism improve generalization?", "assessment": "insufficient",
    "interpretation": "The available validation result cannot isolate the mechanism.",
    "evidence": ["exp_0001"], "primary_explanation": "The mechanism changes the representation.",
    "competing_explanations": ["Compute or tuning explains the difference."],
    "next_step": {"kind": "ablation", "question": "Does removing the mechanism remove the effect?",
        "comparison": "Compare matched compute, data and seeds with and without the mechanism.",
        "prediction_if_primary": "Removing the mechanism worsens validation performance.",
        "prediction_if_alternative": "The matched control retains similar performance.",
        "falsifying_result": "The control retains the improvement across the planned paired observations.",
        "target_issue_ids": [], "tested_alternative": "Compute or tuning explains the difference.",
        "decision_if_primary": "Request independent replication before claiming a mechanism.",
        "decision_if_alternative": "Narrow the mechanism claim and investigate compute effects.",
        "decision_if_inconclusive": "Check precision and implementation before another run.",
        "remaining_uncertainty": "This ablation does not establish transfer to new datasets.",
        "max_training_attempts": 6},
    "protocol_appraisals": [],
    "issue_updates": [],
    "claim_scope": "Validation observations only; no mechanism or generalization established.",
    "unresolved_checks": ["Verify implementation correspondence and independent replication."],
}


def cycle(tmp_path):
    p, ledger, registry, out = setup(tmp_path)
    return ResearchCycle(p, out, ledger_path=ledger, registry_dir=registry)


def assess(c, llm):
    return c.assess(llm, question=ASSESSMENT["question"], max_training_attempts=6)


def test_iterations_retain_history_and_negative_evidence_without_releasing_writing(tmp_path):
    c = cycle(tmp_path)
    first = assess(c, MockLLM(jsons=[ASSESSMENT]))
    llm = MockLLM(router=continue_assessment)
    resumed = ResearchCycle(c.profile, c.directory, ledger_path=c.ledger_path, registry_dir=c.registry_dir)
    second = assess(resumed, llm)
    ctx = json.loads(llm.calls[0]["prompt"])
    assert ctx["history"][0] == first and second["round"] == 2
    assert ctx["dossier"]["runs"][0]["verdict"] == "discarded"
    assert second["status"] == "recommendation_requires_scientific_review"
    assert "not verified evidence" in c.review_feedback(second)
    assert not (c.directory / "scientific_review.json").exists()


@pytest.mark.parametrize("change", ["evidence", "budget", "predictions", "stop"])
def test_invalid_research_recommendations_never_enter_history(tmp_path, change):
    c = cycle(tmp_path)
    result = copy.deepcopy(ASSESSMENT)
    if change == "evidence":
        result["evidence"] = ["invented_run"]
    elif change == "budget":
        result["next_step"]["max_training_attempts"] = 7
    elif change == "predictions":
        result["next_step"]["prediction_if_alternative"] = result["next_step"]["prediction_if_primary"]
    else:
        result["next_step"]["kind"] = "stop"
    with pytest.raises(AgentError):
        assess(c, MockLLM(jsons=[result] * 3))
    assert not c.path.exists()


def test_evidence_change_during_assessment_does_not_commit_stale_judgement(tmp_path):
    c = cycle(tmp_path)
    def mutate(prompt):
        source = c.profile.root / c.profile.train_entrypoint
        source.write_text(source.read_text() + "\n# changed\n")
        return ASSESSMENT
    with pytest.raises(ReviewBlocked, match="changed"):
        assess(c, MockLLM(router=mutate))
    assert not c.path.exists()


def test_test_exposure_blocks_research_assessment_before_model_call(tmp_path):
    c = cycle(tmp_path)
    row = json.loads(c.ledger_path.read_text())
    row["metric"]["holdout"] = {"value": 12345}
    c.ledger_path.write_text(json.dumps(row) + "\n")
    llm = MockLLM()
    with pytest.raises(ValueError, match="Final test access"):
        assess(c, llm)
    assert llm.calls == []


def test_assessment_enters_review_then_frozen_experiment_and_reassessment(tmp_path):
    from scholarloop.review.session import ScientificReviewSession
    from scholarloop.experiment.agenda import ExperimentAgenda
    from tests.test_scientific_review import PASS, REVISE, ARGUMENT
    from tests.experiment.test_agenda import SPEC
    c = cycle(tmp_path)
    recommendation = assess(c, MockLLM(jsons=[ASSESSMENT]))
    llm = MockLLM(jsons=[PASS, REVISE, PASS, PASS])
    review = ScientificReviewSession(llm, c.profile, c.directory, ledger_path=c.ledger_path,
                                    registry_dir=c.registry_dir)
    state = review.review(argument=ARGUMENT, iteration=recommendation)
    assert state["status"] == "NEEDS_EXPERIMENTS"
    assert state["research_iteration"] == recommendation
    assert "prediction_if_alternative" in llm.calls[0]["prompt"]
    assert "decision_if_inconclusive" in llm.calls[0]["prompt"]
    assert "remaining_uncertainty" in llm.calls[0]["prompt"]
    agenda = ExperimentAgenda(review)
    planning_llm = MockLLM(jsons=[discriminating_spec()])
    review.llm = planning_llm
    plan = agenda.plan()
    assert json.loads(planning_llm.calls[0]["prompt"])["research_iteration"] == recommendation
    _, envelope = agenda.load(plan["protocol_id"])
    assert envelope["research_iteration"] == recommendation
    report = (agenda.out / f"{plan['protocol_id']}.md").read_text()
    assert "decision_if_inconclusive" in report
    assert ASSESSMENT["next_step"]["remaining_uncertainty"] in report
    assert agenda.execute(plan["protocol_id"], author="Researcher")["status"] == "EXECUTED"
    def address_with_new_evidence(prompt):
        result = continue_assessment(prompt)
        ctx = json.loads(prompt)
        from scholarloop.review.dossier import evidence_ids
        fresh = sorted(evidence_ids(ctx["dossier"]) - set(ctx["tracked_issues"][0]["opening_evidence"]))
        assert fresh
        result["issue_updates"][0].update(status="addressed_pending_review", evidence=[fresh[0]],
                                          rationale="New experiment available for independent review.")
        return result
    second_llm = MockLLM(router=address_with_new_evidence)
    second = assess(c, second_llm)
    assert second["round"] == 2 and second["input_hash"] != recommendation["input_hash"]
    assert json.loads(second_llm.calls[0]["prompt"])["dossier"]["protocols"]
    assert second["assessment"]["protocol_appraisals"][0]["outcome"] == "inconclusive"
    from scholarloop.experiment.research_cycle import tracked_issues
    issues = tracked_issues(json.loads(c.path.read_text())["rounds"])
    assert len(issues) == 1 and issues[0]["status"] == "addressed_pending_review"
    review.llm = MockLLM()
    with pytest.raises(ReviewBlocked, match="stale"):
        review.review(argument=ARGUMENT, iteration=recommendation)
    assert review.llm.calls == []


def continue_assessment(prompt):
    ctx = json.loads(prompt)
    result = copy.deepcopy(ASSESSMENT)
    result["protocol_appraisals"] = [
        {"protocol_id": pid, "outcome": "inconclusive" if protocol["analysis"] is not None else "execution_incomplete",
         "evidence": [pid], "observed_vs_predicted": "Contrast remains insufficient to isolate the mechanism.",
         "remaining_explanations": ["Optimization effects remain plausible."],
         "decision": "Inspect the intervention before expanding the experiment."}
        for pid, protocol in ctx["dossier"].get("protocols", {}).items()]
    result["next_step"]["target_issue_ids"] = [issue["issue_id"] for issue in ctx["tracked_issues"]]
    result["issue_updates"] = [
        {"issue_id": issue["issue_id"], "status": "open", "rationale": "Still needs replication.",
         "evidence": ["exp_0001"]} for issue in ctx["tracked_issues"]]
    return result


def test_open_issues_survive_history_window_and_resume(tmp_path):
    c = cycle(tmp_path)
    assess(c, MockLLM(jsons=[ASSESSMENT]))
    for _ in range(6):
        llm = MockLLM(router=continue_assessment)
        assess(c, llm)
    ctx = json.loads(llm.calls[0]["prompt"])
    assert ctx["omitted_rounds"] == 1
    assert len(ctx["history"]) == 5
    assert len(ctx["tracked_issues"]) == 1
    assert ctx["tracked_issues"][0]["opened_round"] == 1


@pytest.mark.parametrize("mode", ["missing", "duplicate", "invented", "old_evidence"])
def test_issue_dispositions_cannot_silently_drop_or_close_concerns(tmp_path, mode):
    c = cycle(tmp_path)
    assess(c, MockLLM(jsons=[ASSESSMENT]))
    before = c.path.read_bytes()
    def invalid(prompt):
        result = continue_assessment(prompt)
        update = result["issue_updates"][0]
        if mode == "missing":
            result["issue_updates"] = []
        elif mode == "duplicate":
            result["issue_updates"].append(copy.deepcopy(update))
        elif mode == "invented":
            update["issue_id"] = "invented"
        else:
            update["status"] = "addressed_pending_review"
        return result
    with pytest.raises(AgentError):
        assess(c, MockLLM(router=invalid))
    assert c.path.read_bytes() == before


def test_legacy_rounds_reconstruct_concerns_without_state_migration(tmp_path):
    c = cycle(tmp_path)
    assess(c, MockLLM(jsons=[ASSESSMENT]))
    state = json.loads(c.path.read_text())
    del state["rounds"][0]["available_evidence"]
    del state["rounds"][0]["assessment"]["issue_updates"]
    c.path.write_text(json.dumps(state))
    llm = MockLLM(router=continue_assessment)
    assess(c, llm)
    assert json.loads(llm.calls[0]["prompt"])["tracked_issues"][0]["opened_round"] == 1


def test_legacy_issue_cannot_claim_new_evidence_without_opening_inventory(tmp_path):
    from scholarloop.experiment.research_cycle import IterationReviewer
    reviewer = IterationReviewer(MockLLM())
    from scholarloop.review.dossier import build_dossier, snapshot
    c = cycle(tmp_path)
    dossier = build_dossier(snapshot(c.profile, c.ledger_path, c.registry_dir))
    result = copy.deepcopy(ASSESSMENT)
    result["issue_updates"] = [{"issue_id": "legacy", "status": "addressed_pending_review",
                                "rationale": "Claimed new evidence.", "evidence": ["exp_0001"]}]
    errors = reviewer.postcheck(result, {"question": ASSESSMENT["question"], "dossier": dossier,
        "max_training_attempts": 6, "tracked_issues": [{"issue_id": "legacy",
         "opening_evidence": [], "opening_inventory_known": False}]})
    assert any("opening evidence inventory" in error for error in errors)


@pytest.mark.parametrize("mode", ["missing_target", "invented_target", "unlisted_alternative"])
def test_experiment_must_bind_existing_uncertainty_and_named_alternative(tmp_path, mode):
    c = cycle(tmp_path)
    assess(c, MockLLM(jsons=[ASSESSMENT]))
    before = c.path.read_bytes()
    def invalid(prompt):
        result = continue_assessment(prompt)
        if mode == "missing_target":
            result["next_step"]["target_issue_ids"] = []
        elif mode == "invented_target":
            result["next_step"]["target_issue_ids"] = ["invented"]
        else:
            result["next_step"]["tested_alternative"] = "An unnamed explanation"
        return result
    with pytest.raises(AgentError):
        assess(c, MockLLM(router=invalid))
    assert c.path.read_bytes() == before


def test_review_hash_binds_iteration_before_experiment_planning(tmp_path):
    from scholarloop.review.session import ScientificReviewSession
    from scholarloop.experiment.agenda import ExperimentAgenda
    from tests.test_scientific_review import PASS, REVISE, ARGUMENT
    from tests.experiment.test_agenda import SPEC
    c = cycle(tmp_path)
    recommendation = assess(c, MockLLM(jsons=[ASSESSMENT]))
    review = ScientificReviewSession(MockLLM(jsons=[PASS, REVISE, PASS, PASS]), c.profile,
                                    c.directory, ledger_path=c.ledger_path, registry_dir=c.registry_dir)
    state = review.review(argument=ARGUMENT, iteration=recommendation)
    state["research_iteration"]["assessment"]["next_step"]["decision_if_inconclusive"] = "Claim success."
    review._save(state)
    with pytest.raises(ReviewBlocked, match="Source review changed"):
        ExperimentAgenda(review).plan(spec=copy.deepcopy(SPEC))


def discriminating_spec():
    from tests.experiment.test_agenda import SPEC
    spec = copy.deepcopy(SPEC)
    spec["discrimination"] = {
        "reference": "control", "treatment": "ablation",
        "tested_alternative": ASSESSMENT["next_step"]["tested_alternative"],
        "why_discriminating": "Test whether the effect depends on the changed learning rate.",
        "controlled_factors": ["Same data, seeds and architecture."],
        "remaining_confounds": ["Learning rate can affect optimization as well as representation."],
        "inconclusive_outcome": "Small or inconsistent paired differences.",
        "departure_from_recommendation": "Learning-rate ablation only; not a direct mechanism intervention.",
    }
    return spec


@pytest.mark.parametrize("mode", ["missing", "invented_condition", "same_condition", "alternative", "identical_intervention", "default_alias"])
def test_discrimination_mapping_rejects_invalid_executable_contrasts(tmp_path, mode):
    from scholarloop.experiment.spec import validate_spec
    c = cycle(tmp_path)
    spec = discriminating_spec()
    if mode == "missing":
        del spec["discrimination"]
    elif mode == "invented_condition":
        spec["discrimination"]["treatment"] = "invented"
    elif mode == "same_condition":
        spec["discrimination"]["treatment"] = "control"
    elif mode == "alternative":
        spec["discrimination"]["tested_alternative"] = "Different explanation"
    elif mode == "default_alias":
        spec["conditions"][0]["config"] = []
        spec["conditions"][1]["config"] = [{"name": name, "value": value}
                                             for name, value in c.profile.knobs.items()]
    else:
        spec["conditions"][1]["config"] = copy.deepcopy(spec["conditions"][0]["config"])
    issues = [{"id": spec["issue_ids"][0], "status": "open", "action": "experiment"}]
    errors = validate_spec(spec, c.profile, issues, iteration={"assessment": ASSESSMENT})
    assert errors


def test_discrimination_valid_mapping_is_not_a_causal_certificate(tmp_path):
    from scholarloop.experiment.spec import validate_spec
    c = cycle(tmp_path)
    spec = discriminating_spec()
    issues = [{"id": spec["issue_ids"][0], "status": "open", "action": "experiment"}]
    assert validate_spec(spec, c.profile, issues, iteration={"assessment": ASSESSMENT}) == []
    assert spec["discrimination"]["remaining_confounds"]


@pytest.mark.parametrize("mode", ["missing", "duplicate", "invented", "no_analysis", "incomplete", "uncited"])
def test_protocol_appraisals_cannot_omit_or_overclaim_results(tmp_path, mode):
    from scholarloop.experiment.research_cycle import IterationReviewer
    from scholarloop.review.dossier import build_dossier, snapshot
    c = cycle(tmp_path)
    dossier = build_dossier(snapshot(c.profile, c.ledger_path, c.registry_dir))
    dossier["protocols"] = {"protocol:test": {"analysis": {"comparisons": [{"status": "complete"}]}}}
    ctx = {"question": ASSESSMENT["question"], "dossier": dossier, "tracked_issues": [],
           "max_training_attempts": 6}
    result = continue_assessment(json.dumps(ctx))
    appraisal = result["protocol_appraisals"][0]
    if mode == "missing":
        result["protocol_appraisals"] = []
    elif mode == "duplicate":
        result["protocol_appraisals"].append(copy.deepcopy(appraisal))
    elif mode == "invented":
        appraisal["protocol_id"] = "protocol:invented"
    elif mode == "uncited":
        appraisal["evidence"] = ["exp_0001"]
    else:
        appraisal["outcome"] = "consistent_with_primary"
        dossier["protocols"]["protocol:test"]["analysis"] = (
            None if mode == "no_analysis" else {"comparisons": [{"status": "incomplete"}]})
    assert IterationReviewer(MockLLM()).postcheck(result, ctx)
