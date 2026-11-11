import copy
import json
from dataclasses import replace

import pytest

from scholarloop.core.agent import AgentError
from scholarloop.core.llm import MockLLM
from scholarloop.ideation.deep import NoveltyCritic, portfolio
from scholarloop.ideation.session import ResearchSession, _main
from tests.ideation.test_session import ATOM, PROFILE, QUERIES, REVIEW, _idea, _session


def candidate(title):
    return {**_idea(title), "design": {
        "mechanism": title + " mechanism", "intervention": "Add a residual connection",
        "assumptions": ["Optimization is the limiting factor"],
        "minimal_test": "Compare a matched MLP without the connection",
        "kill_criterion": "No error reduction under matched compute",
        "confound_controls": ["Match parameter counts"], "resource_risk": "Low, CPU experiment",
        "novelty_queries": ["residual shallow multilayer perceptron"]}}


NOVELTY = {"status": "provisional_difference", "summary": "The shallow setting differs, provisionally.",
           "comparisons": [{"paper_id": "arXiv:1512.03385v1", "shared_mechanism": "Residual connections",
                             "difference": "The candidate concerns a shallow MLP",
                             "supporting_excerpt": "Residual connections ease optimization."}],
           "remaining_search": ["Inspect full text and search shallow-network work"]}


def test_deep_generation_is_diverse_auditable_and_resumable(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW,
                            *[{"ideas": [candidate(t)]} for t in ("Skip", "Transfer", "Counterexample")],
                            *[copy.deepcopy(NOVELTY) for _ in range(3)]], deep_ideation=True)
    s.review()
    ideas = s.ideate(3)
    assert {i.strategy for i in ideas} == {"mechanism", "transfer", "contradiction"}
    assert all(i.idea_id and i.design["kill_criterion"] for i in ideas)
    assert len(s.state.idea_graph["nodes"]) == 3
    assert len(json.loads((s.out / "idea_portfolio.json").read_text())["shortlist"]) == 3
    assert "abstracts_only" in (s.out / "ideas.md").read_text()
    # Three independent generation calls precede the novelty reviews, without peer candidates.
    for call in s.llm.calls[2:5]:
        assert "Archived decisions" in call["prompt"]
    resumed = ResearchSession(MockLLM(), PROFILE, s.out)
    assert resumed.state.deep_ideation
    handoff = resumed.select(1, actor="Researcher", reason="The falsifier is affordable")
    assert handoff["idea"]["design"] and handoff["selection_decision"]["kind"] == "human"
    assert "Kill criterion" in handoff["guidance"]


def test_graph_keeps_rejected_candidates_and_explicit_merge_lineage(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("A"), _idea("B")]}])
    s.review()
    ideas = s.ideate(2)
    ids = [i.idea_id for i in ideas]
    s.select(1)
    s.decide("reject", [ids[0]], actor="Researcher", reason="The control is confounded")
    assert not (s.out / "handoff.json").exists()
    assert list((s.out / "handoff_history").glob("*.json"))
    with pytest.raises(ValueError, match="rejected"):
        s.select(1)
    s.llm = MockLLM(jsons=[{"ideas": [candidate("Merged")]}, NOVELTY])
    merged = s.decide("merge", ids, actor="Researcher", reason="Combine the mechanism with the control", n=1)
    graph = s.state.idea_graph
    assert len(graph["nodes"]) == 3 and len(graph["rounds"]) == 2
    assert {e["from"] for e in graph["edges"] if e["to"] == merged[0].idea_id} == set(ids)
    assert all(e["relation"] == "merges" for e in graph["edges"])
    assert "The control is confounded" in s.llm.calls[0]["prompt"]


def test_rejection_cannot_disappear_when_generator_repeats_candidate(tmp_path):
    output = {"ideas": [_idea("Repeated")]}
    s = _session(tmp_path, [QUERIES, REVIEW, output, output])
    s.review()
    pid = s.ideate(1)[0].idea_id
    s.decide("reject", [pid], actor="Researcher", reason="Not useful")
    assert s.ideate(1)[0].status == "rejected"
    with pytest.raises(ValueError, match="rejected"):
        s.auto_select()


def test_novelty_requires_actual_abstract_support():
    paper = {"arxiv_id": "paper", "summary": "Residual connections ease optimization."}
    invalid = copy.deepcopy(NOVELTY)
    invalid["comparisons"][0]["paper_id"] = "paper"
    invalid["comparisons"][0]["supporting_excerpt"] = "This paper already tested our exact method."
    agent = NoveltyCritic(MockLLM(jsons=[invalid] * 3))
    with pytest.raises(AgentError):
        agent.run({"idea": {}, "papers": [paper]})
    uncertain = {**NOVELTY, "status": "uncertain", "comparisons": []}
    assert NoveltyCritic(MockLLM(jsons=[uncertain])).run({"idea": {}, "papers": []})["status"] == "uncertain"


def test_select_uses_current_context_and_automation_is_not_a_human(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("Candidate")]}])
    s.review()
    s.ideate(1)
    assert s.auto_select()["selection_decision"]["kind"] == "automated"
    s.profile = replace(PROFILE, cv_folds=5)
    with pytest.raises(ValueError, match="changed"):
        s.select(1)


def test_human_selection_cli_needs_no_api_key(tmp_path, monkeypatch):
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("Candidate")]}])
    s.review()
    s.ideate(1)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _main(["--profile", str(PROFILE.source_path), "--out", str(s.out), "--select", "1",
                  "--actor", "Researcher", "--reason", "Concrete falsifier"]) == 0


def test_failed_deep_round_preserves_previous_candidates(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("Old")]}])
    s.review()
    old = s.ideate(1)
    s.state.deep_ideation = True
    s.llm = MockLLM(jsons=[{"ideas": [candidate("New")]}])  # novelty response unavailable
    with pytest.raises(AssertionError):
        s.ideate(1)
    assert [i.idea_id for i in s.ideas] == [i.idea_id for i in old]
    assert len(s.state.idea_graph["rounds"]) == 1


def test_brief_updates_invalidate_handoff_and_reach_next_generation(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("Old")]}, {"ideas": [_idea("New")]}])
    s.review()
    s.ideate(1)
    s.select(1)
    brief = {"question": "Which mechanism survives the control?", "success_criteria": ["Discriminating evidence"],
             "constraints": ["CPU only"], "non_goals": ["Leaderboard records"],
             "resources": "One laptop", "literature_scope": "Shallow neural networks"}
    s.set_brief(brief, actor="Researcher")
    assert s.stage == "reviewed" and not s.ideas and not (s.out / "handoff.json").exists()
    s.ideate(1)
    assert "CPU only" in s.llm.calls[-1]["prompt"]
    assert s.select(1)["research_brief"] == brief
    assert len(s.state.idea_graph["rounds"]) == 2


def test_human_injection_preserves_authorship_without_model_calls(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW])
    s.review()
    calls = len(s.llm.calls)
    raw = candidate("Human mechanism")
    idea = s.inject(raw, actor="Researcher", reason="I want to test this explanation")
    assert len(s.llm.calls) == calls
    assert idea.strategy == "human" and idea.description == raw["description"]
    assert s.state.idea_graph["decisions"][-1]["action"] == "inject"
    assert s.select(1)["idea"]["idea_id"] == idea.idea_id


def test_ungrounded_human_seed_is_saved_but_cannot_be_selected(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW])
    s.review()
    raw = _idea("Unverified human mechanism", sources=[])
    idea = s.inject(raw, actor="Researcher", reason="Explore this intuition")
    assert idea.status == "needs_grounding" and idea.idea_id in s.state.idea_graph["nodes"]
    with pytest.raises(ValueError, match="grounding"):
        s.select(1)
    with pytest.raises(ValueError):
        s.auto_select()
    assert portfolio(s.ideas) == []


def test_unknown_citation_in_human_idea_is_not_silently_accepted(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW])
    s.review()
    with pytest.raises(ValueError, match="unknown paper"):
        s.inject(_idea("A", sources=["made-up"]), actor="Researcher", reason="Test")
    assert not s.ideas


def test_interactive_reject_is_recorded_and_cannot_be_selected(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("Candidate")]}])
    s.review()
    s.ideate(1)
    answers = iter(["x 1 Missing a discriminating control", "s 1", "q"])
    output = []
    assert s.interactive(input_fn=lambda prompt: next(answers), print_fn=output.append) is None
    assert s.state.idea_graph["decisions"][-1]["action"] == "reject"
    assert any("rejected" in text for text in output)


def test_failed_literature_update_cannot_reuse_old_selection(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("Old")]}, {}, {}, {}])
    s.review()
    s.ideate(1)
    s.select(1)
    with pytest.raises(AgentError):
        s.review(feedback="Check the contradictory literature first")
    assert s.state.review_pending and not (s.out / "handoff.json").exists()
    resumed = ResearchSession(MockLLM(), PROFILE, s.out)
    with pytest.raises(ValueError, match="pending"):
        resumed.select(1)
    with pytest.raises(RuntimeError, match="incomplete"):
        resumed.ideate(1)


def test_injection_does_not_revalidate_peers_from_stale_profile(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("Old setting")]}])
    s.review()
    old = s.ideate(1)[0].idea_id
    s.profile = replace(PROFILE, cv_folds=5)
    s.inject(_idea("New setting"), actor="Researcher", reason="New evaluation protocol")
    assert [i.title for i in s.ideas] == ["New setting"]
    assert old in s.state.idea_graph["nodes"]  # archived, never silently revalidated
