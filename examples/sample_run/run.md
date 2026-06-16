# ScholarLoop — autonomous run log

A complete run on the real **digits-mlp** torch engine, driven by **claude-opus-4-8**. Every experiment below is a real PyTorch training run; every decision is a real LLM call.

- **11** experiments · **45** LLM calls · 56437+12781 tokens · ≈ **$0.602**
- baseline to beat: 5.0% val_top1_err

## 🎯 Director — direction
> Establish a strong MLP baseline on digits classification by tuning regularization and normalization techniques, then iterate on architectural and training improvements to beat 5.0 val_top1_err.  
> *topic for the Lit Scout:* MLP architecture regularization dropout batch normalization image classification MNIST

## 🔭 Lit Scout — grounded findings (real literature: arXiv + OpenAlex, citation-ranked)
- **Dropout regularization in fully-connected/MLP layers** (doi:10.1186/s40537-021-00444-8) — Reduces overfitting and improves test accuracy on MNIST by randomly deactivating units during training
- **Batch normalization between layers** (doi:10.1186/s40537-021-00444-8) — Stabilizes and accelerates training, allowing higher learning rates and improving convergence
- **ReLU activations with deeper hierarchical layers** (doi:10.1109/access.2019.2912200) — Mitigates vanishing gradients and improves feature abstraction over shallow networks
- **Explicit regularization (weight decay, data augmentation) acknowledging it is not the sole driver of generalization** (doi:10.1145/3446776) — Modest improvement in generalization gap, but SGD's implicit regularization also matters
- **Rely on SGD as implicit regularizer rather than over-tuning explicit regularization** (doi:10.1145/3446776) — Good test performance achievable with SGD even when explicit regularization is limited

## 🪜 Experiments (real torch · multi-fidelity funnel)

| id | tier | val_top1_err | verdict | predicted→measured | grounded source |
|---|---|---|---|---|---|
| exp_0001 | smoke | 2.6667% | kept | — | ReLU activations with deeper hierarchical layers (doi:10.110 |
| exp_0002 | verify | 2.6667% | kept | — | ReLU activations with deeper hierarchical layers (doi:10.110 |
| exp_0003 | full | 2.6667% | kept | — | ReLU activations with deeper hierarchical layers (doi:10.110 |
| exp_0004 | smoke | 4.2222% | kept | -2.5→1.5555 | ReLU activations with deeper hierarchical layers (doi:10.110 |
| exp_0005 | smoke | 2.6667% | kept | -0.4→0.0 | Heavy regularization strategy for data-efficient MLP trainin |
| exp_0006 | verify | 2.8148% | kept | — | Heavy regularization strategy for data-efficient MLP trainin |
| exp_0007 | smoke | 4.2222% | kept | -0.2→1.5555 | arXiv:2105.03404 (ResMLP) — well-designed MLPs with moderate |
| exp_0008 | smoke | 2.6667% | kept | -0.2→0.0 | arXiv:2201.03299 (regularization survey: weight decay/L2 as  |
| exp_0009 | verify | 2.6667% | kept | — | arXiv:2201.03299 (regularization survey: weight decay/L2 as  |
| exp_0010 | smoke | 3.3333% | kept | -0.3→0.6666 | arXiv:2201.03299 (regularization survey: stronger/combined r |
| exp_0011 | verify | 3.6296% | kept | — | arXiv:2201.03299 (regularization survey: stronger/combined r |

## 🧠 Accumulated skills (self-improvement)

- [regularization, w=0.60] When already at a good plateau, avoid stepping weight_decay by a full order of magnitude in one jump; a 10x increase to 1e-2 on the depth-2/width-256 MLP worsened val_top1_err from 2.6667 to 3.63 (underfitting/over-regularization). Tune weight_decay in smaller multiplicative steps (e.g., 2-3x) and bracket around the parent's working value rather than extrapolating from survey claims of 'stronger regularization helps.'
- [hyperparameter-tuning, w=0.60] Don't assume lowering lr from 0.1 to 0.05 monotonically improves an MLP; on digits-mlp it regressed val error from the 2.67 frontier to ~4.2. Sweep lr around the working baseline (e.g. 0.1, 0.07, 0.05) jointly with epochs rather than predicting a small gain from a single lower value, and verify convergence isn't being slowed by too-low lr at fixed epoch budget.
- [capacity-scaling, w=0.60] Don't assume adding depth/width to an MLP yields large monotonic error reductions on low-dimensional tabular data like digits; predicted gains were overestimated by ~4 points. Anchor capacity-increase predictions to small empirical deltas observed in the parent run rather than to deep-architecture literature, and validate with a quick depth-sweep before committing to large predicted improvements.
- [model_capacity, w=0.60] For small structured datasets like digits, a modest depth-2 MLP with width 256, moderate lr (0.1) and weight_decay (5e-4) over many epochs (120) reliably beats shallow baselines; start capacity scaling from this config rather than overly deep nets which add little benefit on low-dimensional inputs.
- [hyperparameter_tuning, w=0.50] On small datasets like digits, simultaneously raising lr to 0.1 and weight_decay to 1e-3 on a depth-2/width-256 MLP did not beat the lighter-regularized baseline (2.81 vs 2.67). Change one regularization knob at a time and validate; combining aggressive lr with stronger weight decay tends to over-regularize/destabilize rather than improve generalization.

## 🔁 Agent trace (every call, auditable)

`advisor:6 · critic:Contrarian:6 · critic:Innovator:6 · critic:Pragmatist:6 · director:4 · lit_scout:3 · reasoner:6 · reflector:6`

See [`paper.md`](paper.md) for the write-up this run produced, and [`experiments.jsonl`](experiments.jsonl) for the raw ledger.
