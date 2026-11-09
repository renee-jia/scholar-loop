"""Ideation, the human-in-the-loop front half: decide *what* is worth running.

  litscout   deterministic multi-source retrieval (arXiv, OpenAlex, Semantic Scholar) + LitScout agent
  litreview  QueryPlanner + LitReviewer: a citation-grounded literature review
  ideator    Ideator agent + deterministic novelty check against papers and tested claims
  tournament pairwise, position-swapped Judge matches -> Elo ranking + near-duplicate groups
  metareview recurring judge critiques -> <= 5 evidenced recommendations for the next round
  session    ResearchSession: review -> ideas -> human choice -> handoff.json (CLI: python -m scholarloop.ideation)
"""
