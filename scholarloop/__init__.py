"""scholarloop: an auto-research framework that replicates a PhD research workflow.

The package is split by stage. Three halves run in order, joined by artifacts on disk; `core`
holds the contracts all three share:

  scholarloop.core         contracts: Agent harness, LLM clients, Profile, Ledger, VerifiedRegistry
  scholarloop.ideation     human-in-the-loop front half: literature review -> ideas -> choice -> handoff
  scholarloop.experiment   autonomous loop: Orchestrator, Runner, Reasoner, Debate, Reflector, Governor
  scholarloop.writing      author-approved back half: outline -> key points -> draft -> reviewer panel

Import modules directly, e.g.:

    from scholarloop.ideation.session import ResearchSession
    from scholarloop.experiment.orchestrator import Orchestrator
    from scholarloop.core.ledger import Ledger, LedgerEntry

CLIs: `python -m scholarloop.ideation`, `python -m scholarloop.writing`,
`python -m scholarloop.experiment.runner`, and `python -m scholarloop.core.{profile,ledger,registry}`.
Submodules are intentionally not re-exported here (nor from the subpackages) so those CLIs run
without an import-ordering warning. Each module's docstring documents its role.
"""

__version__ = "0.0.1"
