# Sample run (second domain) — regression on diabetes

A second **verbatim capture**, on a different domain and a different metric: the real
`diabetes-mlp` torch engine (RMSE regression on sklearn diabetes), with **Claude Opus 4.8** as
every agent. Same orchestrator, funnel, and guards as the [digits run](../sample_run/) — only the
profile + engine pair are new, which is the whole point of pluggable domains. Produced by
[`../run_to_paper.py`](../run_to_paper.py) with `SCHOLARLOOP_PROFILE=diabetes-mlp`.

> **18 experiments · 57 LLM calls · ≈ $0.77.** A **governed population funnel** over four rounds.
> The must-beat bar is the **linear-model reference (56.5 RMSE)** — OLS/Ridge both land there on this
> split, and an untuned MLP only ties it. Fanning out across rounds, the loop found that a *shallow,
> well-regularized, long-trained* MLP (depth 1, width 64, lr 0.01, **weight_decay 5e-3, 400 epochs**)
> edges past linear regression to a best of **55.24 RMSE** (3-seed verify), confirmed below baseline
> at the full tier (55.27). It learned the honest negatives too — **widening to 128 units** (55.59) and
> **training longer, 500–600 epochs** (55.71 / 55.44) did *not* help. The **governor** ran the campaign
> to its round cap, and the reviewer rejected the write-up 3/10 as marginal.
>
> This run also exercises the loop-engineering stack — **parallel fan-out · governed loop ·
> per-agent calibration**. See [`../sample_run/`](../sample_run/) for the classification domain.

**Self-calibration and anti-hallucination, both firing live:**

- **Universal predict-then-verify** — over the run the **debate panel** was right on **100% of its 7**
  go/no-go calls, while the **Reasoner** only predicted the right direction **40% of 5** times (mean
  magnitude error ~0.95). Near an already-flat frontier its "finer-convergence" gains mostly didn't
  materialize — and the loop *records* that, surfacing it back into later prompts rather than hiding it.
- **Number-grounding flagged `442`** — the Writer mentioned the dataset's 442 samples, a real fact but
  not a recorded measurement, so the grounding audit marked the draft ⚠️. The gate is deliberately
  strict: only numbers that trace to the experiment registry pass unflagged (contrast the digits
  paper's ✅). Strictness over precision is the right trade for an anti-fabrication guard.

| file | what it is |
|---|---|
| **[`run.md`](run.md)** | the autonomous run log — direction, the real (arXiv + OpenAlex, citation-ranked) findings, every experiment (funnel tiers + verdicts), the distilled lessons, and the **agent calibration** block |
| **[`paper.md`](paper.md)** | the paper the L5 Writer produced + the Reviewer's assessment, with the grounding audit's ⚠️ on the un-measured `442` |
| **[`experiments.jsonl`](experiments.jsonl)** | the raw ledger — one real torch training run per record |

Reproduce (non-deterministic — a fresh run yields different experiments and a different paper):

```bash
ANTHROPIC_API_KEY=sk-ant-... \
  SCHOLARLOOP_PROFILE=diabetes-mlp \
  SCHOLARLOOP_OUT=examples/sample_run_diabetes \
  SCHOLARLOOP_POPULATION=3 SCHOLARLOOP_MAX_WORKERS=3 \
  SCHOLARLOOP_DRY_PATIENCE=2 SCHOLARLOOP_STEPS=4 \
  python examples/run_to_paper.py
```
