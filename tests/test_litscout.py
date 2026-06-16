"""Lit Scout: deterministic retrieval parsing (fake fetchers) + LLM extraction (MockLLM)."""

import json

from scholarloop.litscout import (
    ArxivClient, LitScout, OpenAlexClient, Paper, merge_papers,
)
from scholarloop.llm import MockLLM

ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1608.03983v1</id>
    <title>SGDR: Stochastic Gradient Descent with Warm Restarts</title>
    <summary>We propose a cosine annealing schedule with warm restarts.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1512.03385v1</id>
    <title>Deep Residual Learning for Image Recognition</title>
    <summary>Residual connections ease optimization of deep networks.</summary>
  </entry>
</feed>"""

EMPTY_XML = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def test_arxiv_parsing_from_canned_feed():
    arxiv = ArxivClient(fetcher=lambda q, n: ATOM_XML)
    papers = arxiv.search("image classification")
    assert [p.arxiv_id for p in papers] == ["1608.03983v1", "1512.03385v1"]
    assert papers[0].title.startswith("SGDR")
    assert "cosine" in papers[0].summary


def test_scout_extracts_findings_and_priors():
    arxiv = ArxivClient(fetcher=lambda q, n: ATOM_XML)
    llm = MockLLM(jsons=[{"findings": [
        {"technique": "cosine schedule", "source": "arXiv:1608.03983",
         "predicted_effect": "lower val error", "rationale": "warm restarts escape minima"},
    ]}])
    lit_context, lit_priors = LitScout(llm, arxiv).scout("image classification")
    assert "cosine schedule" in lit_context
    assert lit_priors == ["cosine schedule (arXiv:1608.03983)"]


def test_scout_handles_no_papers():
    arxiv = ArxivClient(fetcher=lambda q, n: EMPTY_XML)
    # LLM should never be called when there are no papers
    assert LitScout(MockLLM(), arxiv).scout("obscure topic") == ("", [])


# ---- OpenAlex (impact-ranked source) ----
OPENALEX_JSON = json.dumps({"results": [
    {"title": "Deep Residual Learning for Image Recognition",
     "abstract_inverted_index": {"Residual": [0], "connections": [1], "ease": [2], "optimization.": [3]},
     "cited_by_count": 180000,
     "ids": {"openalex": "https://openalex.org/W2194775991"},
     "locations": [{"landing_page_url": "https://arxiv.org/abs/1512.03385"}]},
    {"title": "Batch Normalization",
     "abstract_inverted_index": {"Normalize": [0], "activations.": [1]},
     "cited_by_count": 50000,
     "ids": {"openalex": "https://openalex.org/W1", "doi": "https://doi.org/10.5555/abc"},
     "locations": []},
]})


def test_openalex_parsing_reconstructs_abstract_and_citations():
    oa = OpenAlexClient(fetcher=lambda q, n: OPENALEX_JSON)
    papers = oa.search("image classification")
    assert papers[0].arxiv_id == "arXiv:1512.03385"           # extracted from the arXiv landing page
    assert papers[0].summary == "Residual connections ease optimization."   # inverted index rebuilt
    assert papers[0].citations == 180000
    assert papers[1].arxiv_id == "doi:10.5555/abc"            # falls back to DOI when no arXiv id


def test_merge_dedups_across_sources_and_ranks_by_citations():
    # same paper from two sources (arXiv id vs version-suffixed) collapses; higher-cited wins the row
    a = Paper("ResNet", "s", "arXiv:1512.03385v1", "u", citations=None)
    b = Paper("ResNet", "s", "arXiv:1512.03385", "u", citations=180000)
    c = Paper("Some niche paper", "s", "arXiv:9999.0001", "u", citations=12)
    merged = merge_papers([[a, c], [b]], max_results=5)
    assert [p.arxiv_id for p in merged] == ["arXiv:1512.03385", "arXiv:9999.0001"]  # deduped, ranked
    assert merged[0].citations == 180000                     # kept the record that carried citations


def test_scout_multisource_surfaces_citations_in_context():
    arxiv = ArxivClient(fetcher=lambda q, n: ATOM_XML)
    oa = OpenAlexClient(fetcher=lambda q, n: OPENALEX_JSON)
    captured = {}

    class Spy(MockLLM):
        def complete_json(self, prompt, schema, system=None):
            captured["prompt"] = prompt
            return {"findings": [{"technique": "residual connections", "source": "arXiv:1512.03385",
                                  "predicted_effect": "eases optimization"}]}

    lit_context, priors = LitScout(Spy(), sources=[arxiv, oa]).scout("image classification")
    assert "180000 citations" in captured["prompt"]          # impact signal reached the agent
    assert priors == ["residual connections (arXiv:1512.03385)"]
    assert "residual connections" in lit_context
