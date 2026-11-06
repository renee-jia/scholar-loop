"""Ideation stage: grounded idea generation, the deterministic novelty check, ranking."""

from pathlib import Path

import pytest

from scholarloop.agent import AgentError
from scholarloop.ideation import Idea, Ideator, novelty_check, render_markdown, tried_claims
from scholarloop.ledger import Hypothesis, LedgerEntry
from scholarloop.litreview import LitReview
from scholarloop.litscout import Paper
from scholarloop.llm import MockLLM
from scholarloop.profile import load_profile

ROOT = Path(__file__).resolve().parent.parent
PROFILE = load_profile(ROOT / "profiles" / "image-classification.yaml")

PAPERS = [Paper("Deep Residual Learning", "residual connections ease optimization of deep networks",
                "arXiv:1512.03385", "u", citations=180000),
          Paper("SGDR warm restarts", "cosine annealing learning rate schedule with warm restarts",
                "arXiv:1608.03983", "u", citations=9000)]
REVIEW = LitReview(topic="mlp recipes", queries=["q"], papers=PAPERS, overview="o",
                   themes=[{"name": "t", "summary": "s", "papers": ["arXiv:1512.03385"]}],
                   paper_notes=[{"id": "arXiv:1512.03385", "contribution": "skip connections", "limitation": "deep only"}],
                   gaps=[{"gap": "residual + cosine never combined on shallow MLPs", "why_open": "w",
                          "sources": ["arXiv:1512.03385", "arXiv:1608.03983"]}], trends=[])


def _idea(title="Residual cosine MLP", sources=("arXiv:1512.03385",), feas=4, impact=3,
          description="Add skip connections to a shallow MLP trained with a cosine schedule."):
    return {"title": title, "description": description, "gap": "residual + cosine on shallow MLPs",
            "sources": list(sources), "hypothesis": {"claim": "skips + cosine lower error",
                                                     "predicted_effect": "-0.5 err"},
            "experiment_plan": "set depth=20, warmup=5, compare to baseline", "novelty_rationale": "n",
            "feasibility": feas, "impact": impact}


def test_ideator_grounds_citations_and_validates_scores():
    bad = {"ideas": [_idea(sources=("arXiv:0000.00000",), feas=9)]}
    good = {"ideas": [_idea(), _idea(title="Warmup sweep", sources=("arXiv:1608.03983",))]}
    llm = MockLLM(jsons=[bad, good])
    ideas = Ideator(llm, PROFILE).ideate(REVIEW, n=2, tried=["old claim"], feedback="be bolder")
    assert [i.title for i in ideas] == ["Residual cosine MLP", "Warmup sweep"]
    retry = llm.calls[1]["prompt"]
    assert "unknown paper ids" in retry and "feasibility must be a number in 1-5" in retry
    first = llm.calls[0]["prompt"]
    assert "old claim" in first and "be bolder" in first            # ledger + human feedback reach the agent
    assert "must-beat baseline 4.9" in first                        # the setting the idea must be testable in
    assert "residual + cosine never combined" in first              # the review's gaps are the raw material


def test_ideator_rejects_duplicate_titles():
    dup = {"ideas": [_idea(), _idea()]}
    with pytest.raises(AgentError):
        Ideator(MockLLM(jsons=[dup] * 3), PROFILE).ideate(REVIEW)


def test_novelty_check_flags_paper_lookalike_and_tested_claims():
    lookalike = Idea.from_dict({**_idea(title="Deep residual learning",
                                         description="residual connections ease optimization of deep networks"),
                                "novelty": {}, "selected": False})
    rep = novelty_check(lookalike, REVIEW)
    assert rep["closest_paper"] == "arXiv:1512.03385" and rep["flag"] and "reads close" in rep["reason"]

    fresh = Idea.from_dict({**_idea(description="Mixup between digit images under label smoothing."),
                            "novelty": {}, "selected": False})
    rep = novelty_check(fresh, REVIEW)
    assert not rep["flag"] and rep["paper_overlap"] < 0.6

    rep = novelty_check(fresh, REVIEW, tried=["skips + cosine lower error on digits"])
    assert rep["flag"] and "already-tested" in rep["reason"]


def test_tried_claims_dedups_ledger():
    es = [LedgerEntry(id="a", domain="d", hypothesis=Hypothesis(claim="Cosine helps", source="s"), metric_name="m"),
          LedgerEntry(id="b", domain="d", hypothesis=Hypothesis(claim="cosine  helps", source="s"), metric_name="m"),
          LedgerEntry(id="c", domain="d", hypothesis=Hypothesis(claim="Wider is better", source="s"), metric_name="m")]
    assert tried_claims(es) == ["Cosine helps", "Wider is better"]


def test_rank_key_and_markdown():
    a = Idea.from_dict({**_idea(title="A", feas=5, impact=5), "novelty": {"flag": True, "reason": "copy"}, "selected": False})
    b = Idea.from_dict({**_idea(title="B", feas=3, impact=3), "novelty": {"flag": False, "closest_paper": "x", "paper_overlap": 0.2}, "selected": True})
    assert b.rank_key() > a.rank_key()                              # a flagged idea ranks below a clean one
    md = render_markdown("topic", [a, b], feedback=["bolder"])
    assert "⚠️ copy" in md and "✅ selected" in md and "Feedback applied: bolder" in md
