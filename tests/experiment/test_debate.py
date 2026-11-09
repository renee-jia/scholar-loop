"""Debate panel: persona critics vote; rejection requires a strict majority."""

from scholarloop.experiment.debate import DebatePanel
from scholarloop.core.ledger import Hypothesis
from scholarloop.core.llm import MockLLM
from scholarloop.experiment.reasoner import Proposal
from scholarloop.experiment.reasoning import SearchSpaceConstraints

ROLES = ["Innovator", "Pragmatist", "Contrarian"]


def _proposal():
    return Proposal(config={"lr": 0.1}, hypothesis=Hypothesis("tune lr", "arXiv:1"),
                    predicted_delta=-0.2, reasoning_trace="trying lower lr",
                    constraints=SearchSpaceConstraints())


def test_panel_runs_when_no_majority_rejects():
    llm = MockLLM(jsons=[{"verdict": "run", "concern": ""},
                         {"verdict": "revise", "concern": "tight budget"},
                         {"verdict": "run", "concern": ""}])
    v = DebatePanel(llm, ROLES).review(_proposal())
    assert v.decision == "run"
    assert len(v.votes) == 3


def test_panel_rejects_on_strict_majority():
    llm = MockLLM(jsons=[{"verdict": "reject", "concern": "already explored"},
                         {"verdict": "reject", "concern": "low odds"},
                         {"verdict": "run", "concern": ""}])
    v = DebatePanel(llm, ROLES).review(_proposal())
    assert v.decision == "reject"
    assert any("Innovator" in c for c in v.concerns)


def test_single_skeptic_does_not_veto():
    llm = MockLLM(jsons=[{"verdict": "reject", "concern": "risky"},
                         {"verdict": "run", "concern": ""},
                         {"verdict": "run", "concern": ""}])
    assert DebatePanel(llm, ROLES).review(_proposal()).decision == "run"
