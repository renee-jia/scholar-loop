# Sample run: the whole flow with a real LLM

A **verbatim capture** of one ScholarLoop run on the real `digits-mlp` torch engine, with
**claude-opus-5-5** as every agent. It covers literature review, ideas, a tournament pick, a governed
campaign, a staged write-up and a panel review. Captured 2026-11-09.

> **5 ideas → 1 picked · 18 experiments · 3 at the full tier · best 3.27% val / 3.33% held-out
> test (baseline logreg 3.8) · panel review: reject, 2.2/10.**
> Spend: campaign ≈ $0.94 (56 calls), write-up ≈ $5.53 (42 calls). The ideation session is not
> metered.

What it shows, good and bad:

- **Ideation.** The tournament picked *sub-pixel augmentation for 8x8 digits*, 4-0-0 over the
  other four ideas. See [`lit_review.md`](lit_review.md) and [`ideas.md`](ideas.md).
- **Campaign.** It beat the bar: 5-fold CV val 3.27, held-out test 3.33. The bar is logreg@3.8.
- **The gap between idea and campaign.** The chosen idea needs a change to `train.py`. The
  Reasoner never used the edit channel. It tuned epochs, width and weight decay instead, so the
  hypothesis was never tested.
- **The tiers are not on the same folds.** All three configs promoted from 3 seeds rose at 5
  seeds, for example 2.90 → 3.27. The paper reads this as regression to the mean. The ledger
  shows something else. Seed s validates on fold s % 5, so verify sees folds 0-2 and full sees
  all five, and fold 4 is much harder: every full run scores 4.5-5.2 there and about 3 on the
  others. Training is deterministic per seed and fold, so each full run's folds 0-2 reproduce
  its verify mean exactly (exp_0003 on folds 0-2 is 3.2444, the same as exp_0002). All of the
  rise is folds 3-4. None of it is selection. Promotion now compares tiers on shared folds.
- **The reviewer caught it.** The panel rejected the paper because "the campaign's actual
  question is untested". The panel also flagged stage-dependent shifts and missing setup detail
  (optimizer, split sizes). The Writer was never given that detail. This is a real gap in the
  writing context.
- **Calibration.** The debate panel made 8 go/no-go calls and 50% were right. The Reasoner got
  the direction of the effect right 28% of the time.

| file | what it is |
|---|---|
| [`lit_review.md`](lit_review.md) · [`ideas.md`](ideas.md) | the research session: themes, gaps, and five ideas ranked by a pairwise Elo tournament |
| [`run.md`](run.md) | the campaign log: chosen idea, every experiment with tier, verdict and held-out score, distilled skills, agent calibration |
| [`outline.md`](outline.md) · [`key_points.md`](key_points.md) | the write-up's first two stages, each debated by a panel |
| [`paper.md`](paper.md) · [`review.md`](review.md) | the paper, with every number checked against the registry, and the three-reviewer assessment |
| [`experiments.jsonl`](experiments.jsonl) | the raw ledger, one real torch training run per record |

Reproduce. The run is not deterministic, so a fresh run gives different ideas, experiments and
paper:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m scholarloop.ideation --profile profiles/digits-mlp.yaml --out runs/d \
  --topic "small MLP image classification regularization" --auto
SCHOLARLOOP_HANDOFF=runs/d/handoff.json SCHOLARLOOP_OUT=runs/d \
  SCHOLARLOOP_POPULATION=3 SCHOLARLOOP_MAX_WORKERS=3 SCHOLARLOOP_DRY_PATIENCE=2 \
  SCHOLARLOOP_STEPS=4 SCHOLARLOOP_MAX_COST=8 python examples/run_to_paper.py < /dev/null
```
