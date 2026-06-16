# Sample run — a complete flow with a real LLM

A **verbatim capture** of one autonomous ScholarLoop run: the real `digits-mlp` torch engine,
with **Claude Opus 4.8** as every agent, taken all the way from *"pick a direction"* to a
written-up, peer-reviewed paper. Produced by [`../run_to_paper.py`](../run_to_paper.py).

> **11 experiments · 45 LLM calls · ≈ $0.60.** The loop read the literature, proposed
> literature-grounded configs, debated them, and ran real PyTorch experiments through the
> multi-fidelity funnel. One idea climbed the full ladder — smoke → verify → full — to a confirmed
> **2.67% val error, less than half the 5.0 baseline**, while the Reasoner's over-optimistic gain
> predictions were caught by predict-then-verify and distilled into reusable lessons. It wrote the
> result up — then its own reviewer rejected the paper 3/10 as too marginal. (It's not wrong.)

| file | what it is |
|---|---|
| **[`run.md`](run.md)** | the autonomous run log — director's direction, the real (arXiv + OpenAlex, citation-ranked) findings, every experiment (funnel tiers + verdicts + predict-vs-measured), and the lessons the Reflector distilled |
| **[`paper.md`](paper.md)** | the paper the L5 Writer produced + the Reviewer's assessment. Every number is checked against the experiment registry (grounding ✅) |
| **[`experiments.jsonl`](experiments.jsonl)** | the raw ledger — one real torch training run per record |

Reproduce (non-deterministic — a fresh run yields different experiments and a different paper):

```bash
ANTHROPIC_API_KEY=sk-ant-... python examples/run_to_paper.py
```
