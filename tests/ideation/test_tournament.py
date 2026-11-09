"""Idea tournament: pairwise, position-swapped judgement + Elo + proximity, all deterministic code
around a Judge agent."""

import json
import re
from pathlib import Path

from scholarloop.core.llm import MockLLM
from scholarloop.core.profile import load_profile
from scholarloop.ideation.ideator import Idea
from scholarloop.ideation.litscout import ArxivClient
from scholarloop.ideation.session import ResearchSession
from scholarloop.ideation.tournament import (ELO_START, Tournament, elo_update, expected,
                                             proximity_clusters)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_profile(ROOT / "profiles" / "image-classification.yaml")


def _idea(title, claim=None, feas=3, impact=3):
    return Idea(title=title, description=f"{title}, tried on digits.", gap="g", sources=["arXiv:1"],
                hypothesis={"claim": claim or f"{title} lowers error", "predicted_effect": "-0.3"},
                experiment_plan="p", novelty_rationale="n", feasibility=feas, impact=impact)


def _shown(prompt, slot):
    return re.search(rf"Idea {slot}:\nTitle: (.+)", prompt).group(1)


def strength_judge(strength: dict):
    """A judge that always prefers the stronger title, whichever slot it is in."""
    def route(prompt):
        a, b = _shown(prompt, "A"), _shown(prompt, "B")
        return {"winner": "A" if strength[a] >= strength[b] else "B", "rationale": "r"}
    return route


def test_elo_math():
    assert expected(1200, 1200) == 0.5
    ra, rb = elo_update(1200, 1200, 1.0)
    assert ra == 1216 and rb == 1184                     # K=32, even match
    ra, rb = elo_update(1200, 1200, 0.5)
    assert ra == rb == 1200                              # a draw between equals changes nothing


def test_round_robin_recovers_the_true_order():
    ideas = [_idea("weak"), _idea("strong"), _idea("medium")]
    llm = MockLLM(router=strength_judge({"weak": 1, "medium": 2, "strong": 3}))
    res = Tournament(llm, PROFILE).run("t", ideas)
    assert [ideas[i].title for i in res.standings()] == ["strong", "medium", "weak"]
    assert len(res.matches) == 3 and len(llm.calls) == 6          # every pair, judged twice
    assert res.wins == [0, 2, 1] and res.losses == [2, 0, 1]
    assert {m["judge"] for m in res.matches} == set(PROFILE.debate_roles)   # personas rotate


def test_position_bias_becomes_a_draw():
    # a judge that always says "A" has no preference between ideas: every match must be a draw
    ideas = [_idea("x"), _idea("y"), _idea("z")]
    res = Tournament(MockLLM(router=lambda p: {"winner": "A", "rationale": "r"}), PROFILE).run("t", ideas)
    assert res.draws == [2, 2, 2] and res.wins == [0, 0, 0]
    assert res.ratings == [ELO_START] * 3


def test_judge_never_sees_self_scores_or_novelty():
    ideas = [_idea("x", feas=5, impact=5), _idea("y", feas=1, impact=1)]
    ideas[0].novelty = {"flag": True, "reason": "reads close to arXiv:9"}
    llm = MockLLM(router=lambda p: {"winner": "A", "rationale": "r"})
    Tournament(llm, PROFILE).run("t", ideas)
    for c in llm.calls:
        assert "feasibility" not in c["prompt"].lower() and "impact 5" not in c["prompt"]
        assert "arXiv:9" not in c["prompt"]
    assert "Title: y" in llm.calls[0]["prompt"] and "Title: x" in llm.calls[0]["prompt"]


def test_swiss_rounds_cap_the_cost_and_skip_repeats():
    ideas = [_idea(f"idea {c}", claim=f"claim {c}") for c in "abcdefgh"]   # 28 pairs > max_pairs
    llm = MockLLM(router=strength_judge({f"idea {c}": i for i, c in enumerate("abcdefgh")}))
    res = Tournament(llm, PROFILE, max_pairs=10, swiss_rounds=3).run("t", ideas)
    pairs = [frozenset((m["a"], m["b"])) for m in res.matches]
    assert len(pairs) == 12 and len(set(pairs)) == 12              # 3 rounds x 4 pairs, no rematch
    assert res.standings()[0] == 7                                 # the strongest idea leads


def test_proximity_groups_near_duplicates():
    ideas = [_idea("cosine warm restarts", claim="cosine warm restarts lower error"),
             _idea("mixup on digits", claim="mixup lowers error"),
             _idea("cosine warm restarts schedule", claim="cosine warm restarts lower error")]
    assert proximity_clusters(ideas) == [0, 1, 0]


# ---- session integration ------------------------------------------------------------------------
ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/1512.03385v1</id><title>Deep Residual Learning</title>
    <summary>Residual connections ease optimization.</summary><published>2015-12-10T00:00:00Z</published></entry>
</feed>"""
SRC = "arXiv:1512.03385v1"
QUERIES = {"queries": ["mlp"], "rationale": "r"}
REVIEW = {"overview": "o", "themes": [{"name": "Opt", "summary": "s", "papers": [SRC]}],
          "paper_notes": [{"id": SRC, "contribution": "skips", "limitation": "deep"}],
          "gaps": [{"gap": "g", "why_open": "w", "sources": [SRC]}], "trends": []}


def _raw(title, feas, impact):
    return {"title": title, "description": f"{title} on digits.", "gap": "g", "sources": [SRC],
            "hypothesis": {"claim": f"{title} helps", "predicted_effect": "-0.3"},
            "experiment_plan": "p", "novelty_rationale": "n", "feasibility": feas, "impact": impact}


def test_session_orders_ideas_by_tournament_and_auto_selects_its_winner(tmp_path):
    # the Ideator rates "Alpha" highest, but the judges prefer "Gamma": the tournament decides
    ideas = {"ideas": [_raw("Alpha", 5, 5), _raw("Beta", 3, 3), _raw("Gamma", 1, 1)]}
    judge = strength_judge({"Alpha": 1, "Beta": 2, "Gamma": 3})
    scripted = iter([QUERIES, REVIEW, ideas])          # planner, reviewer, ideator; judges are routed
    llm = MockLLM(router=lambda p: judge(p) if "Idea A:" in p else next(scripted))
    s = ResearchSession(llm, PROFILE, tmp_path / "s", topic="t",
                        sources=[ArxivClient(fetcher=lambda q, n: ATOM)], tournament=Tournament(llm, PROFILE))
    s.review()
    ranked = s.ideate(3)
    assert [i.title for i in ranked] == ["Gamma", "Beta", "Alpha"]
    assert ranked[0].tournament["rank"] == 1 and ranked[0].tournament["wins"] == 2
    md = (s.out / "ideas.md").read_text()
    assert "pairwise Elo tournament" in md and "| 1 | **Gamma**" in md
    (rnd,) = json.loads((s.out / "tournament.json").read_text())["rounds"]
    assert rnd["round"] == 1 and rnd["ideas"] == ["Gamma", "Beta", "Alpha"] and len(rnd["matches"]) == 3
    assert all(1 <= m["a"] <= 3 and 1 <= m["b"] <= 3 for m in rnd["matches"])   # renumbered to ideas.md
    assert s.auto_select()["topic"] == "Gamma"

    resumed = ResearchSession(MockLLM(), PROFILE, tmp_path / "s")              # persisted standings
    assert resumed.ideas[0].tournament["elo"] == ranked[0].tournament["elo"]


def test_stored_rank_is_the_displayed_position_even_for_a_flagged_winner(tmp_path):
    scripted = iter([QUERIES, REVIEW, {"ideas": [_raw("Alpha", 3, 3), _raw("Beta", 3, 3)]}])
    judge = strength_judge({"Alpha": 2, "Beta": 1})                    # Alpha wins on Elo ...
    llm = MockLLM(router=lambda p: judge(p) if "Idea A:" in p else next(scripted))
    s = ResearchSession(llm, PROFILE, tmp_path / "s", topic="t", tournament=Tournament(llm, PROFILE),
                        sources=[ArxivClient(fetcher=lambda q, n: ATOM)])
    s.review()
    import scholarloop.ideation.session as sess
    real = sess.novelty_check
    sess.novelty_check = lambda idea, *a, **k: {**real(idea, *a, **k), "flag": idea.title == "Alpha"}
    try:
        ranked = s.ideate(2)                                           # ... but Alpha is flagged
    finally:
        sess.novelty_check = real
    assert [i.title for i in ranked] == ["Beta", "Alpha"]
    assert [i.tournament["rank"] for i in ranked] == [1, 2]


def test_a_failing_judge_voids_its_match_not_the_round():
    def route(prompt):
        if "Title: broken" in prompt:
            return {"winner": "C", "rationale": "r"}                   # never schema-valid
        return strength_judge({"x": 2, "y": 1})(prompt)
    ideas = [_idea("x"), _idea("y"), _idea("broken")]
    res = Tournament(MockLLM(router=route), PROFILE).run("t", ideas)
    results = {(m["a"], m["b"]): m["result"] for m in res.matches}
    assert results == {(1, 2): "a", (1, 3): "void", (2, 3): "void"}
    assert res.ratings[2] == ELO_START                                 # a void match moves no rating


def test_flagged_idea_sinks_below_unflagged_even_with_top_elo():
    top, other = _idea("top"), _idea("other")
    top.tournament, other.tournament = {"elo": 1300.0}, {"elo": 1100.0}
    top.novelty = {"flag": True}
    assert other.rank_key() > top.rank_key()


def test_old_session_json_without_tournament_field_still_loads():
    d = _idea("x").to_dict()
    del d["tournament"]
    assert Idea.from_dict(d).tournament == {}
