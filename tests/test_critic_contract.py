import copy
import json

import pytest

from scholarloop.core.agent import AgentError
from scholarloop.core.call_budget import CallBudgetExceeded, CallLimitedLLM
from scholarloop.core.llm import MockLLM
from scholarloop.review.contracts import ROLES, ReviewBlocked
from scholarloop.review.session import ScientificReviewSession, parse_critic_models
from tests.test_scientific_review import ARGUMENT, PASS, REVISE, session, setup, released


@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong_claim", "invented_evidence", "self_reference", "hidden_concern"])
def test_per_claim_review_cannot_skip_or_hide_an_evidence_problem(tmp_path, defect):
    output = copy.deepcopy(PASS)
    if defect == "missing":
        output["claim_checks"] = []
    elif defect == "duplicate":
        output["claim_checks"] *= 2
    elif defect == "wrong_claim":
        output["claim_checks"][0]["claim_id"] = "claim_nonexistent"
    elif defect == "invented_evidence":
        output["claim_checks"][0]["evidence"] = ["invented-run"]
    elif defect == "self_reference":
        output["claim_checks"][0]["evidence"] = ["claim_error"]
    else:
        output["claim_checks"][0]["status"] = "concern"
        output["claim_checks"][0]["answer"] = "The observation cannot be verified."
    s = session(setup(tmp_path), MockLLM(jsons=[output] * 3))
    with pytest.raises(AgentError):
        s.review(argument=ARGUMENT)
    assert s.load()["status"] == "REVIEW_FAILED"
    with pytest.raises(ReviewBlocked):
        s.require_release()


def test_critics_route_to_configured_models_without_peer_reports(tmp_path):
    p, ledger, registry, out = setup(tmp_path)
    base = MockLLM(jsons=[PASS, PASS])
    base.model = "default-model"
    experimental = MockLLM(jsons=[PASS])
    experimental.model = "experimental-model"
    falsification = MockLLM(jsons=[PASS])
    falsification.model = "falsification-model"
    s = ScientificReviewSession(base, p, out, ledger_path=ledger, registry_dir=registry,
                                critic_llms={"experimental": experimental, "falsification": falsification})
    state = s.review(argument=ARGUMENT)
    assert state["status"] == "READY_FOR_AUTHOR"
    models = {r["role"]: r["reviewer"]["model"] for r in state["reports"]}
    assert models == {"implementation": "default-model", "argument": "default-model",
                      "experimental": "experimental-model", "falsification": "falsification-model"}
    assert [len(c.calls) for c in (base, experimental, falsification)] == [2, 1, 1]
    for call in base.calls + experimental.calls + falsification.calls:
        assert "No unresolved concerns" not in call["prompt"]
    s.approve(author="Researcher", review_hash=state["review_hash"])
    raw = json.loads((out / s.STATE_FILE).read_text())
    raw["reports"][0]["reviewer"]["model"] = "different-model"
    (out / s.STATE_FILE).write_text(json.dumps(raw))
    with pytest.raises(ReviewBlocked, match="artifacts changed"):
        s.require_release()


def test_different_models_share_the_same_request_cap(tmp_path):
    p, ledger, registry, out = setup(tmp_path)
    clients = {role: MockLLM(jsons=[PASS]) for role in ROLES}
    critics = {role: CallLimitedLLM(client, 2, path=out / "review_budget.json") for role, client in clients.items()}
    s = ScientificReviewSession(None, p, out, ledger_path=ledger, registry_dir=registry, critic_llms=critics)
    with pytest.raises(CallBudgetExceeded):
        s.review(argument=ARGUMENT)
    assert sum(len(c.calls) for c in clients.values()) == 2
    assert s.load()["status"] == "REVIEW_FAILED" and s.load()["release"] is None
    assert len(s.load()["reports"]) == 2


def test_valid_partial_dissent_survives_a_later_critic_failure(tmp_path):
    s = session(setup(tmp_path), MockLLM(jsons=[REVISE, PASS]))
    with pytest.raises(AssertionError):
        s.review(argument=ARGUMENT)
    state = s.load()
    assert state["status"] == "REVIEW_FAILED" and len(state["reports"]) == 2
    assert state["issues"][0]["id"] == "r1:implementation:1"
    # The same round continues with the two missing reviewers. The dissent already returned is
    # kept as it stands: later passes cannot outvote it or make it disappear.
    s.llm = MockLLM(jsons=[PASS] * 4)
    state = s.review(argument=ARGUMENT)
    assert len(s.llm.calls) == 2 and state["status"] == "NEEDS_EXPERIMENTS"
    assert s.load()["issues"][0]["status"] == "open" and state["reports"][0]["verdict"] == "revise"
    # Reviewing a changed argument is a new round, and there the dissent needs a disposition.
    changed = {**ARGUMENT, "limitations": [*ARGUMENT["limitations"], "One more limit."]}
    s.llm = MockLLM(jsons=[PASS] * 4)
    with pytest.raises(AgentError):
        s.review(argument=changed)
    assert s.load()["issues"][0]["status"] == "open"


def test_old_review_contract_is_archived_but_cannot_release_writing(tmp_path):
    paths = setup(tmp_path)
    s = released(paths)
    raw = s.load()
    raw["schema_version"] = 1
    for report in raw["reports"]:
        report.pop("claim_checks")
        report.pop("reviewer")
    raw["review_hash"] = s._review_hash(raw)
    (s.out / s.STATE_FILE).write_text(json.dumps(raw))
    with pytest.raises(ReviewBlocked, match="contract changed"):
        s.require_release()
    s.llm = MockLLM(jsons=[PASS] * 4)
    current = s.review(argument=ARGUMENT)
    assert current["schema_version"] == 2 and current["status"] == "READY_FOR_AUTHOR"
    archived = [json.loads(p.read_text()) for p in (s.out / "review_history").glob("*.json")]
    assert any(a["schema_version"] == 1 for a in archived)
    with pytest.raises(ReviewBlocked):
        s.require_release()  # new contract still requires a fresh author decision


@pytest.mark.parametrize("values", [["unknown=model"], ["argument="], ["experimental"],
                                   ["argument=a", "argument=b"]])
def test_invalid_reviewer_configuration_is_not_ignored(values):
    with pytest.raises(ReviewBlocked):
        parse_critic_models(values)


def test_reviewer_configuration_is_explicit_and_role_scoped():
    assert parse_critic_models(["experimental=model-a", "falsification=model-b"]) == {
        "experimental": "model-a", "falsification": "model-b"}
