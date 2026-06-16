# Sample run — a complete flow with a real LLM

A **verbatim capture** of one autonomous ScholarLoop run: the real `digits-mlp` torch engine,
with **Claude Opus 4.8** as every agent, taken all the way from *"pick a direction"* to a
written-up, peer-reviewed paper. Produced by [`../run_to_paper.py`](../run_to_paper.py).

> **7 experiments · 24 LLM calls · ≈ $0.45.** A **governed population funnel**: each round the loop
> proposed several literature-grounded ideas and **smoke-screened them in parallel**; the weak one
> died at smoke, and the survivors climbed **smoke → verify → full** to a confirmed **3.82% val error,
> below the 5.0 baseline**. The **governor** then stopped the campaign on its own once two rounds
> passed with no further improvement (*loop-until-dry*). Along the way the **debate panel was
> calibrated** against ground truth (67% of its go/no-go calls were right), and the reviewer rejected
> the write-up 2/10 as too marginal. (It's not wrong.)
>
> This run exercises the loop-engineering stack — **parallel fan-out · self-stopping governor ·
> per-agent calibration** — on real torch. See [`../sample_run_diabetes/`](../sample_run_diabetes/)
> for the second domain.

| file | what it is |
|---|---|
| **[`run.md`](run.md)** | the autonomous run log — direction, the real (arXiv + OpenAlex, citation-ranked) findings, every experiment (funnel tiers + verdicts), the distilled lessons, and the **agent calibration** block |
| **[`paper.md`](paper.md)** | the paper the L5 Writer produced + the Reviewer's assessment. Every number is checked against the experiment registry (grounding ✅) |
| **[`experiments.jsonl`](experiments.jsonl)** | the raw ledger — one real torch training run per record |

Reproduce (non-deterministic — a fresh run yields different experiments and a different paper):

```bash
ANTHROPIC_API_KEY=sk-ant-... \
  SCHOLARLOOP_POPULATION=3 SCHOLARLOOP_MAX_WORKERS=3 \
  SCHOLARLOOP_DRY_PATIENCE=2 SCHOLARLOOP_STEPS=4 \
  python examples/run_to_paper.py
```
