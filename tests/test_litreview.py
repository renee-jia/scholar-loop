"""Literature review stage: query planning, multi-query retrieval, citation-grounded review."""

import json

import pytest

from scholarloop.agent import AgentError
from scholarloop.litreview import LitReview, LitReviewer, QueryPlanner, render_markdown
from scholarloop.litscout import ArxivClient, OpenAlexClient, Paper, retrieve
from scholarloop.llm import MockLLM

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/2401.00001v1</id><title>Fresh preprint on MLP mixing</title>
    <summary>A brand new token-mixing MLP.</summary><published>2024-01-02T00:00:00Z</published></entry>
  <entry><id>http://arxiv.org/abs/1608.03983v1</id><title>SGDR: Warm Restarts</title>
    <summary>Cosine annealing with warm restarts.</summary><published>2016-08-13T00:00:00Z</published></entry>
</feed>"""

OPENALEX = json.dumps({"results": [
    {"title": "Deep Residual Learning for Image Recognition", "publication_year": 2016,
     "abstract_inverted_index": {"Residual": [0], "connections": [1]}, "cited_by_count": 180000,
     "ids": {"openalex": "https://openalex.org/W1"},
     "locations": [{"landing_page_url": "https://arxiv.org/abs/1512.03385"}]},
    {"title": "SGDR: Warm Restarts", "publication_year": 2017,
     "abstract_inverted_index": {"Cosine": [0], "annealing": [1]}, "cited_by_count": 9000,
     "ids": {"openalex": "https://openalex.org/W2"},
     "locations": [{"landing_page_url": "https://arxiv.org/abs/1608.03983"}]},
    {"title": "Dropout", "publication_year": 2014,
     "abstract_inverted_index": {"Randomly": [0], "drop": [1], "units": [2]}, "cited_by_count": 40000,
     "ids": {"openalex": "https://openalex.org/W3", "doi": "https://doi.org/10.1/drop"}, "locations": []},
]})

PAPERS = [Paper("ResNet", "residual connections", "arXiv:1512.03385", "u", citations=180000, year=2016),
          Paper("SGDR", "cosine annealing", "arXiv:1608.03983", "u", citations=9000, year=2017)]


def _sources():
    return [OpenAlexClient(fetcher=lambda q, n: OPENALEX), ArxivClient(fetcher=lambda q, n: ATOM)]


def test_sources_parse_year():
    assert [p.year for p in ArxivClient(fetcher=lambda q, n: ATOM).search("x")] == [2024, 2016]
    assert OpenAlexClient(fetcher=lambda q, n: OPENALEX).search("x")[0].year == 2016


def test_retrieve_merges_queries_dedups_and_reserves_recent_slots():
    calls = []

    def oa(q, n):
        calls.append(q)
        return OPENALEX

    papers = retrieve([OpenAlexClient(fetcher=oa), ArxivClient(fetcher=lambda q, n: ATOM)],
                      ["mlp regularization", "cosine schedule"], max_total=4, recent_share=0.25)
    assert calls == ["mlp regularization", "cosine schedule"]          # every query hits every source
    ids = [p.arxiv_id for p in papers]
    assert len(ids) == len(set(ids)) == 4
    assert "arXiv:2401.00001v1" in ids                                 # an uncited preprint kept its slot
    assert ids[0] == "arXiv:1512.03385"                                # highest-cited first
    sgdr = next(p for p in papers if "1608.03983" in p.arxiv_id)
    assert sgdr.citations == 9000 and sgdr.year is not None           # merged across sources
    few = retrieve([OpenAlexClient(fetcher=oa)], ["q"], max_total=2)
    assert [p.arxiv_id for p in few] == ["arXiv:1512.03385", "doi:10.1/drop"]   # no uncited pool: pure impact order


def test_query_planner_dedups_and_drops_previous():
    planner = QueryPlanner(MockLLM(jsons=[{"queries": ["mlp dropout", "MLP Dropout", "cosine lr", ""],
                                           "rationale": "cover regularization + schedules"}]))
    qs = planner.plan("mlp regularization", previous=["cosine lr"])
    assert qs == ["mlp dropout"]


def test_reviewer_rejects_hallucinated_citations_then_accepts():
    bad = {"overview": "o", "themes": [{"name": "t", "summary": "s", "papers": ["arXiv:9999.99999"]}],
           "paper_notes": [{"id": "arXiv:1512.03385", "contribution": "c", "limitation": "l"}],
           "gaps": [{"gap": "g", "why_open": "w", "sources": ["arXiv:1512.03385"]}], "trends": []}
    good = {**bad, "themes": [{"name": "t", "summary": "s", "papers": ["arXiv:1512.03385"]}]}
    llm = MockLLM(jsons=[bad, good])
    rev = LitReviewer(llm).review("topic", PAPERS, queries=["q"])
    assert isinstance(rev, LitReview) and rev.themes[0]["papers"] == ["arXiv:1512.03385"]
    assert "unknown paper ids" in llm.calls[1]["prompt"]              # the retry carried the reason
    assert "[arXiv:1512.03385 · 2016, 180000 citations]" in llm.calls[0]["prompt"]


def test_reviewer_requires_a_gap():
    nogap = {"overview": "o", "themes": [{"name": "t", "summary": "s", "papers": []}],
             "paper_notes": [{"id": "arXiv:1512.03385", "contribution": "c", "limitation": "l"}],
             "gaps": [], "trends": []}
    with pytest.raises(AgentError):
        LitReviewer(MockLLM(jsons=[nogap] * 3)).review("topic", PAPERS)


def test_review_roundtrips_and_renders():
    rev = LitReview(topic="t", queries=["q1"], papers=PAPERS, overview="big picture",
                    themes=[{"name": "Schedules", "summary": "s", "papers": ["arXiv:1608.03983"]}],
                    paper_notes=[{"id": "arXiv:1608.03983", "contribution": "c", "limitation": "l"}],
                    gaps=[{"gap": "nobody combined A and B", "why_open": "w", "sources": ["arXiv:1512.03385"]}],
                    trends=["bigger"])
    again = LitReview.from_dict(json.loads(json.dumps(rev.to_dict())))
    assert again == rev
    md = render_markdown(rev)
    assert "# Literature review — t" in md and "nobody combined A and B" in md
    assert "*SGDR*" in md and "180000 cites" in md                   # citations resolve to titles + metadata
    assert "Schedules" in rev.to_prompt_block() and "Limitation: l" in rev.to_prompt_block()
