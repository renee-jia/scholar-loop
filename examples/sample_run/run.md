# ScholarLoop — autonomous run log

A complete run on the real **digits-mlp** torch engine, driven by **claude-opus-4-8**. Every experiment below is a real PyTorch training run; every decision is a real LLM call.

- **7** experiments · **24** LLM calls · 42428+9384 tokens · ≈ **$0.447**
- baseline to beat: 5.0% val_top1_err

## 🎯 Director — direction
> Establish a strong baseline MLP on digit classification with modern training tricks (proper normalization, regularization, and optimizer schedules) before exploring architectural novelty.  
> *topic for the Lit Scout:* multilayer perceptron MNIST digit classification regularization batch normalization data augmentation

## 🔭 Lit Scout — grounded findings (real literature: arXiv + OpenAlex, citation-ranked)
- **Geometric data augmentation (rotation, translation, shifts, flipping) of input images** (doi:10.1186/s40537-019-0197-0) — Reduced overfitting and improved test accuracy by enlarging effective training set
- **Random erasing / cutout style occlusion augmentation** (doi:10.1186/s40537-019-0197-0) — Improved robustness and generalization to partial-digit inputs
- **Elastic/random noise injection augmentation on pixels** (doi:10.1186/s40537-019-0197-0) — Lower variance and better generalization on MNIST
- **GAN-based synthetic sample generation to augment training data** (doi:10.1186/s40537-019-0197-0) — Increased effective dataset size and potentially higher accuracy
- **Dropout regularization in fully connected layers** (doi:10.1186/s40537-021-00444-8) — Reduced overfitting and improved test accuracy in the MLP
- **Batch normalization between MLP layers** (doi:10.1186/s40537-021-00444-8) — Faster convergence and more stable training, enabling higher learning rates
- **ReLU activations to mitigate vanishing gradients** (doi:10.1186/s40537-021-00444-8) — Improved trainability of deeper MLPs
- **L2 weight decay regularization** (doi:10.1007/s10462-023-10466-8) — Reduced overfitting via penalized large weights
- **Class re-weighting / resampling for imbalanced digit classes** (doi:10.1186/s40537-019-0192-5) — Improved minority-class recall if MNIST subset is imbalanced
- **Cost-sensitive loss functions for imbalanced data** (doi:10.1186/s40537-019-0192-5) — Better balanced accuracy across digit classes under imbalance

## 🪜 Experiments (real torch · multi-fidelity funnel)

| id | tier | val_top1_err | verdict | predicted→measured | grounded source |
|---|---|---|---|---|---|
| exp_0002 | smoke | 7.3333% | discarded | — | L2 weight decay regularization (doi:10.1007/s10462-023-10466 |
| exp_0003 | smoke | 4.4444% | kept | — | doi:10.1186/s40537-021-00444-8 (deeper ReLU MLPs improve tra |
| exp_0001 | smoke | 4.2222% | kept | — | L2 weight decay regularization (doi:10.1007/s10462-023-10466 |
| exp_0004 | verify | 3.9259% | kept | — | L2 weight decay regularization (doi:10.1007/s10462-023-10466 |
| exp_0005 | full | 3.8222% | kept | — | L2 weight decay regularization (doi:10.1007/s10462-023-10466 |
| exp_0006 | verify | 4.2222% | kept | — | doi:10.1186/s40537-021-00444-8 (deeper ReLU MLPs improve tra |
| exp_0007 | full | 4.1778% | kept | — | doi:10.1186/s40537-021-00444-8 (deeper ReLU MLPs improve tra |

## 🧠 Accumulated skills (self-improvement)

- [architecture_and_regularization, w=0.60] For small MLPs on digits-style tasks, combining a wider/deeper hidden stack (e.g. 256 units, depth 2) with moderate L2 weight decay (~5e-4) and a moderate LR (~0.05) over ~120 epochs reliably beats a shallow baseline; adopt this as a starting config and tune one factor at a time rather than changing capacity and regularization together so effects remain attributable.

## 🎯 Agent calibration (predict-then-verify)

- debate: 67% of 3 go/no-go calls were correct.

## 🔁 Agent trace (every call, auditable)

`advisor:1 · critic:Contrarian:3 · critic:Innovator:3 · critic:Pragmatist:3 · director:1 · lit_scout:1 · reasoner:9 · reflector:1`

See [`paper.md`](paper.md) for the write-up this run produced, and [`experiments.jsonl`](experiments.jsonl) for the raw ledger.
