"""scholarloop: an auto-research framework that replicates a PhD research workflow.

Two halves, joined by a handoff:

  ideation (human-in-the-loop)         experiment loop (autonomous)
    scholarloop.session   review → ideas → choice    scholarloop.orchestrator  funnel / population / governor
    scholarloop.litreview QueryPlanner + LitReviewer  scholarloop.runner        frozen-scored real runs
    scholarloop.ideation  Ideator + novelty check     scholarloop.paper         grounded write-up + review

Foundational contracts: scholarloop.profile (pluggable domains), scholarloop.ledger (cross-run
memory), scholarloop.registry (VerifiedRegistry number grounding). Import modules directly, e.g.:

    from scholarloop.session import ResearchSession
    from scholarloop.orchestrator import Orchestrator
    from scholarloop.ledger import Ledger, LedgerEntry

Submodules are intentionally not re-exported here so that `python -m scholarloop.<mod>`
CLIs run without an import-ordering warning. Each module's docstring documents its role.
"""

__version__ = "0.0.1"
