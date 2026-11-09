"""The autonomous experiment loop: decide *whether* an idea works.

  orchestrator  deterministic loop: step / funnel_step / population_step, adopt(handoff)
  runner        budgeted subprocess run + two-phase frozen scoring + registry/ledger capture
  reasoner      Reasoner agent: constraints + literature + skills -> one structured proposal
  reasoning     deterministic search-space analysis, dedup, confidence bounds
  debate        Critic personas + DebatePanel: a vote that gates GPU spend
  advisor       post-run PROCEED / REFINE / PIVOT
  director      campaign direction + topic, re-grounding on a PIVOT
  reflector     run outcome -> one lesson for the skill library
  skills        SkillLibrary: content-hash dedup, time decay, relevance-ranked rendering
  calibration   universal predict-then-verify: per-agent track records
  governor      budget / round-cap / convergence stop conditions
"""
