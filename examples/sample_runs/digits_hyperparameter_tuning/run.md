# ScholarLoop — autonomous run log

A complete run on the real **digits-mlp** torch engine, driven by **claude-opus-5-5**. Every experiment below is a real PyTorch training run; every decision is a real LLM call.

- **18** experiments · **56** LLM calls · 83321+30095 tokens · ≈ **$0.935**
- baseline to beat: 3.8% val_top1_err

## 💡 Chosen idea (human-selected in a research session)
> **Sub-pixel augmentation for 8x8 digits: fractional shifts, rotations, noise** — Integer 1-pixel shifts are a large perturbation at 8x8 resolution. Implement on-the-fly augmentation using bilinear fractional translations (±0.5 px), small rotations (±10°), elastic-lite affine jitter, and Gaussian pixel noise. Ablate each one and their combination, sweeping magnitude, to map which label-preserving transforms help an MLP and by how much.  
> *hypothesis:* Small sub-pixel geometric augmentation gives the largest gain for an MLP lacking spatial priors. Integer 1-px shifts and strong noise give smaller gains or hurt.  
> *builds on:* doi:10.1145/3065386, arXiv:2010.11929, arXiv:2103.14030


## 🪜 Experiments (real torch · multi-fidelity funnel)

| id | tier | val_top1_err | held-out test | verdict | predicted→measured | grounded source |
|---|---|---|---|---|---|---|
| exp_0001 | smoke | 3.8194% | — | discarded | — | Krizhevsky et al. 2012, doi:10.1145/3065386 (regularization  |
| exp_0002 | verify | 3.2444% | — | kept | — | Krizhevsky et al. 2012, doi:10.1145/3065386 (regularization  |
| exp_0003 | full | 3.5494% | 3.7222% | kept | — | Krizhevsky et al. 2012, doi:10.1145/3065386 (regularization  |
| exp_0004 | smoke | 4.1667% | — | discarded | -0.4→0.9223 | doi:10.1145/3065386 (Krizhevsky et al.: regularization of la |
| exp_0006 | smoke | 3.8194% | — | discarded | -0.3→0.575 | Skill library lesson ablation_attribution/training_budget (c |
| exp_0007 | smoke | 3.125% | — | kept | -0.45→-0.1194 | Krizhevsky et al., ImageNet classification with deep CNNs (d |
| exp_0005 | smoke | 3.4722% | — | kept | -0.1→0.2278 | Krizhevsky et al., ImageNet classification with deep convolu |
| exp_0008 | verify | 3.1286% | — | kept | — | Krizhevsky et al., ImageNet classification with deep CNNs (d |
| exp_0009 | full | 3.5497% | 3.5% | kept | — | Krizhevsky et al., ImageNet classification with deep CNNs (d |
| exp_0010 | verify | 3.2444% | — | kept | — | Krizhevsky et al., ImageNet classification with deep convolu |
| exp_0011 | verify | 3.3601% | — | kept | — | Skill library lesson ablation_attribution/training_budget (c |
| exp_0012 | smoke | 3.8194% | — | discarded | -0.08→0.6944 | Krizhevsky et al., ImageNet classification with deep convolu |
| exp_0013 | smoke | 3.125% | — | kept | -0.07→0.0 | arXiv:2010.11929 (ViT: weak inductive bias models are data/r |
| exp_0014 | smoke | 3.4722% | — | kept | -0.07→0.3472 | Krizhevsky et al., ImageNet classification with deep convolu |
| exp_0015 | verify | 2.8968% | — | kept | — | arXiv:2010.11929 (ViT: weak inductive bias models are data/r |
| exp_0016 | full | 3.2712% | 3.3333% | kept | — | arXiv:2010.11929 (ViT: weak inductive bias models are data/r |
| exp_0017 | verify | 3.1282% | — | kept | — | Krizhevsky et al., ImageNet classification with deep convolu |
| exp_0018 | verify | 3.244% | — | kept | — | Krizhevsky et al., ImageNet classification with deep convolu |

_Held-out test: scored once on full-tier runs, on data the loop never selected on. The gap to the val column is the optimism that repeated selection on val adds._

## 🧠 Accumulated skills (self-improvement)

- [ablation_attribution / training_budget, w=0.70] On small-data MLPs (digits), don't credit width or weight-decay gains without checking the training budget. Cutting epochs from 240 to 120 erased the parent's gain: error rose from 3.24 to 4.17, worse than the 3.8 baseline. That points to the epoch count, likely interacting with lr and decay, as the main driver. Before attributing an improvement to one factor, run a one-factor-at-a-time ablation of epochs alone. When the budget is cut, also adjust lr or wd and compare against a baseline trained for the same number of epochs. Treat any prediction that depends on 'regularization beats longer training' as low confidence (expect at least ±1.0 error) until it has been ablated.
- [effect_size_calibration/baseline_anchoring, w=0.50] Size the predicted delta against the parent's score, not against a degraded ablation or an older baseline. Here the -0.45 prediction was anchored to the 3.8 baseline or the 4.17 regression, but the parent was already at 3.24, so the realistic headroom was about 0.1–0.2. That is roughly what was measured (-0.12). When an experiment restores a factor that an ablation removed (for example the training budget), expect it to recover to about the parent's level plus a small marginal gain. Cap the predicted improvement at a fraction of the parent-to-frontier gap. Note too that extending epochs past the point that recovers the parent's result gives diminishing returns on small datasets like digits.
- [evaluation_noise_and_calibration, w=0.45] On small validation sets like 8x8 digits (a few hundred samples), a gain of about 0.2 points, such as 3.125 to 2.897, is roughly one example. That is likely inside seed noise. Before keeping a budget or epoch change, or fixing it as the budget for downstream arms, do two things. First, state a numeric predicted delta. Second, run 3 or more seeds of both the parent and the child and require the gain to exceed the seed std. Without a predicted effect, calibration_error stays None and the 'plateau' claim cannot be tested.
- [experimental_design/baseline_control, w=0.45] When a no-aug control changes several knobs at once (width, weight decay, epochs) and moves the metric noticeably (3.80 to 3.24 here), treat it as the new baseline. Run every augmentation arm with the identical control config and compare against the control, not the original baseline. Also record a numeric predicted effect before each run so calibration error can be computed. If the gain from the control matters, follow up with single-knob ablations to attribute it.

## 🎯 Agent calibration (predict-then-verify)

- debate: 50% of 8 go/no-go calls were correct.
- reasoner: predicted the right direction 28% of 7 times; mean magnitude error 0.588.

## 🛑 Governor

The campaign stopped itself: round cap reached (4 rounds).

## 🔁 Agent trace (every call, auditable)

`advisor:4 · critic:Contrarian:12 · critic:Innovator:12 · critic:Pragmatist:12 · reasoner:12 · reflector:4`

See [`paper.md`](paper.md) for the write-up this run produced, and [`experiments.jsonl`](experiments.jsonl) for the raw ledger.
