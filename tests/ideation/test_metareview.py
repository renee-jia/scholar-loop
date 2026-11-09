"""Meta-review: recurring tournament critiques -> at most 5 evidenced recommendations -> the next
round's Ideator prompt. The Ideator also reads the experiment loop's skill library."""

import json
import time
from pathlib import Path

import pytest

from scholarloop.core.agent import AgentError
from scholarloop.core.llm import MockLLM
from scholarloop.core.profile import load_profile
from scholarloop.experiment.skills import Skill, SkillLibrary
from scholarloop.ideation.litscout import ArxivClient
from scholarloop.ideation.metareview import MetaReviewer, match_lines
from scholarloop.ideation.session import ResearchSession
from scholarloop.ideation.tournament import Tournament

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_profile(ROOT / "profiles" / "image-classification.yaml")
MATCHES = [{"a": 1, "b": 2, "judge": "Contrarian", "result": "b",
            "rationale_a_first": "A's effect is below seed noise.", "rationale_b_first": "same"},
           {"a": 1, "b": 3, "judge": "Pragmatist", "result": "draw",
            "rationale_a_first": "Neither sizes its effect.", "rationale_b_first": "tie"}]
TITLES = ["Alpha", "Beta", "Gamma"]


def test_match_lines_are_citeable():
    lines = match_lines(MATCHES, TITLES)
    assert lines[0].startswith("m1: Contrarian judged 'Alpha' vs 'Beta', preferred 'Beta'.")
    assert "split (a draw)" in lines[1]


def test_new_recommendation_must_recur_and_cite_real_matches():
    bad = {"recommendations": [{"text": "size effects vs noise", "support": ["m1"]}]}       # one match only
    worse = {"recommendations": [{"text": "x", "support": ["m1", "m9"]}]}                  # m9 doesn't exist
    good = {"recommendations": [{"text": "size effects vs noise", "support": ["m2", "m1"]}]}
    llm = MockLLM(jsons=[bad, worse, good])
    recs = MetaReviewer(llm).update("t", [], MATCHES, TITLES)
    assert recs == [{"text": "size effects vs noise", "support": ["m1", "m2"]}]
    assert "not recurring yet" in llm.calls[1]["prompt"] and "unknown support ids ['m9']" in llm.calls[2]["prompt"]


def test_cap_on_recommendations_is_enforced():
    many = {"recommendations": [{"text": f"r{i}", "support": ["m1", "m2"]} for i in range(6)]}
    with pytest.raises(AgentError):
        MetaReviewer(MockLLM(jsons=[many] * 3)).update("t", [], MATCHES, TITLES)


def test_standing_recommendation_is_kept_by_citing_prev():
    llm = MockLLM(jsons=[{"recommendations": [{"text": "size effects vs noise", "support": ["prev:1"]}]}])
    recs = MetaReviewer(llm).update("t", ["size effects vs noise"], MATCHES, TITLES)
    assert recs[0]["support"] == ["prev:1"] and "prev:1: size effects vs noise" in llm.calls[0]["prompt"]


def test_no_matches_carries_the_standing_list_without_a_call():
    llm = MockLLM()
    assert MetaReviewer(llm).update("t", ["keep me"], [], []) == [{"text": "keep me", "support": ["prev:1"]}]
    assert llm.calls == []


# ---- session: round 1's critiques reach round 2's Ideator ----------------------------------------
ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/1512.03385v1</id><title>Deep Residual Learning</title>
    <summary>Residual connections ease optimization.</summary><published>2015-12-10T00:00:00Z</published></entry>
</feed>"""
SRC = "arXiv:1512.03385v1"
REVIEW = {"overview": "o", "themes": [{"name": "Opt", "summary": "s", "papers": [SRC]}],
          "paper_notes": [{"id": SRC, "contribution": "skips", "limitation": "deep"}],
          "gaps": [{"gap": "g", "why_open": "w", "sources": [SRC]}], "trends": []}


def _ideas(*titles):
    return {"ideas": [{"title": t, "description": f"{t} on digits.", "gap": "g", "sources": [SRC],
                       "hypothesis": {"claim": f"{t} helps", "predicted_effect": "-0.3"},
                       "experiment_plan": "p", "novelty_rationale": "n", "feasibility": 3, "impact": 3}
                      for t in titles]}


def test_session_feeds_meta_review_and_skills_into_the_next_round(tmp_path):
    rec = "Size every predicted effect against seed-to-seed noise."
    scripted = iter([{"queries": ["mlp"], "rationale": "r"}, REVIEW,
                     _ideas("Alpha", "Beta"), {"recommendations": [{"text": rec, "support": ["m1", "m1"]}]},
                     _ideas("Gamma", "Delta"), {"recommendations": [{"text": rec, "support": ["prev:1"]}]}])

    def route(prompt):
        if "Idea A:" in prompt:
            return {"winner": "A", "rationale": "A's effect is below seed noise."}
        return next(scripted)

    skills = SkillLibrary(tmp_path / "skills")
    skills.add(Skill.make("training", 0.9, "mlp: under 100 epochs never converges", "exp_0003", time.time()))
    llm = MockLLM(router=route)
    s = ResearchSession(llm, PROFILE, tmp_path / "s", topic="mlp recipes",
                        sources=[ArxivClient(fetcher=lambda q, n: ATOM)], skills=skills,
                        tournament=Tournament(llm, PROFILE), meta_reviewer=MetaReviewer(llm, min_support=1))
    s.review()
    s.ideate(2)
    first_ideator = next(c["prompt"] for c in llm.calls if "Propose 2 distinct ideas" in c["prompt"])
    assert "recurring critiques" not in first_ideator             # nothing distilled yet
    assert "under 100 epochs never converges" in first_ideator    # the loop's lessons reach ideation
    assert s.state.meta_review == [{"text": rec, "support": ["m1"]}]
    assert "## What the idea committee kept saying" in (s.out / "ideas.md").read_text()

    s.ideate(2, feedback="bolder")
    second_ideator = [c["prompt"] for c in llm.calls if "Propose 2 distinct ideas" in c["prompt"]][1]
    assert rec in second_ideator                                  # round 1's critique shapes round 2
    saved = json.loads((s.out / "session.json").read_text())
    assert saved["meta_review"][0]["support"] == ["prev:1"]       # persisted, revised incrementally
