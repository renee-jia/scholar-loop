import copy

from scholarloop.core.llm import MockLLM
from scholarloop.ideation.tournament import Judge, Tournament
from scholarloop.ideation.litscout import Paper, merge_papers
from tests.ideation.test_tournament import _idea, PROFILE


def test_actual_novelty_evidence_changes_pairwise_prompt():
    a, b = _idea("A"), _idea("B")
    judge = Judge(None, PROFILE, "Innovator")
    before = judge.build_prompt({"topic": "fixture", "a": a, "b": b})
    a.novelty_review = {"papers": [{"arxiv_id": "prior:1", "title": "Prior mechanism",
                                   "summary": "The same mechanism was tested under matched controls."}],
                        "remaining_search": ["Check full-text experimental scope."]}
    after = judge.build_prompt({"topic": "fixture", "a": a, "b": b})
    assert before != after
    assert "same mechanism was tested" in after and "prior:1" in after
    assert "Check full-text" in after


def test_brief_dissent_and_evidence_follow_position_swaps():
    a, b = _idea("A"), _idea("B")
    evidence = {"brief": {"question": "mechanism over leaderboard", "constraints": ["single GPU"]},
                "dissent": [{"text": "The tuning confound remains unresolved"}],
                "papers": [{"arxiv_id": "arXiv:1", "title": "Reviewed paper", "summary": "Counterexample evidence"}]}
    llm = MockLLM(jsons=[{"winner": "A", "rationale": "r"}, {"winner": "B", "rationale": "r"}])
    result = Tournament(llm, PROFILE).run("fixture", [a, b], evidence=evidence)
    for call in llm.calls:
        assert all(text in call["prompt"] for text in ("single GPU", "tuning confound", "Counterexample evidence"))
    assert result.matches[0]["evidence_hash"]
    assert "Idea A:\nTitle: A" in llm.calls[0]["prompt"]
    assert "Idea A:\nTitle: B" in llm.calls[1]["prompt"]


def test_paper_merge_keeps_abstract_and_does_not_mutate_inputs():
    a = Paper("Same paper", "Useful abstract", "arXiv:1", "url", citations=1, code_url="code")
    b = Paper("Same paper", "", "arXiv:1", "url", citations=100)
    before = copy.deepcopy((a, b))
    for groups in ([[a], [b]], [[b], [a]]):
        merged = merge_papers(groups, 1)[0]
        assert merged.summary == "Useful abstract" and merged.citations == 100 and merged.code_url == "code"
        assert (a, b) == before


def test_ranking_bounds_abstract_payload_and_marks_omissions():
    a, b = _idea("A"), _idea("B")
    a.novelty_review = {"papers": [{"arxiv_id": str(i), "summary": "x" * 5000} for i in range(10)]}
    prompt = Judge(None, PROFILE, "Innovator").build_prompt({"topic": "fixture", "a": a, "b": b})
    assert '"omitted_papers": 4' in prompt and '"abstract_truncated": true' in prompt
    assert "x" * 2501 not in prompt
