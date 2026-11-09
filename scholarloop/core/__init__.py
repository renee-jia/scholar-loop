"""Shared contracts every stage builds on.

  agent      Agent base class: typed JSON-schema I/O, validate -> retry-with-feedback, AgentTrace
  llm        LLMClient interface, MockLLM (tests / demos) and AnthropicLLM (live runs)
  profile    Domain Profile: the YAML that makes a domain pluggable (+ profile.schema.yaml)
  ledger     Experiment Ledger: append-only jsonl, the cross-run memory
  registry   VerifiedRegistry: every reported number must trace to a measurement
  text       small lexical helpers (terms, Jaccard, containment) shared by novelty checks
"""
