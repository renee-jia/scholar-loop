# ScholarLoop — autonomous run log

A complete run on the real **diabetes-mlp** torch engine, driven by **claude-opus-4-8**. Every experiment below is a real PyTorch training run; every decision is a real LLM call.

- **18** experiments · **57** LLM calls · 77543+15251 tokens · ≈ **$0.769**
- baseline to beat: 56.5 val_rmse

## 🎯 Director — direction
> Establish a strong tabular MLP baseline with proper preprocessing (feature normalization, target scaling) and regularization (dropout, weight decay) before exploring architectural tricks. Focus on training stability and hyperparameter tuning given the small diabetes dataset.  
> *topic for the Lit Scout:* MLP regularization techniques for small tabular regression datasets

## 🔭 Lit Scout — grounded findings (real literature: arXiv + OpenAlex, citation-ranked)
- **Dropout regularization on hidden layers** (doi:10.1007/s42979-021-00815-1) — Reduces overfitting on small datasets by randomly deactivating neurons, improving generalization in tabular regression
- **Batch normalization between dense layers** (doi:10.1007/s42979-021-00815-1) — Stabilizes and accelerates training, providing mild regularization that can improve regression accuracy
- **L1/L2 weight regularization (weight decay)** (doi:10.1007/s42979-021-00592-x) — Penalizes large weights to curb overfitting on limited tabular samples
- **Early stopping based on validation loss** (doi:10.1007/s42979-021-00815-1) — Halts training before overfitting, improving validation RMSE on small datasets
- **Shallow architectures / limiting network depth** (doi:10.1016/j.apenergy.2020.114683) — Shallow models can match or beat deep nets on small tabular data while reducing overfitting
- **Cross-validation for hyperparameter selection** (doi:10.1007/s42979-021-00592-x) — More reliable tuning of regularization strength on small datasets, reducing variance in performance estimates
- **Feature scaling/normalization of inputs** (doi:10.1016/j.apenergy.2020.114683) — Improves convergence and accuracy of MLP regression on heterogeneous tabular features
- **Ensemble methods (bagging shallow learners)** (doi:10.1007/s42979-021-00592-x) — Combining models reduces variance and improves robustness on small tabular regression tasks

## 🪜 Experiments (real torch · multi-fidelity funnel)

| id | tier | val_rmse | verdict | predicted→measured | grounded source |
|---|---|---|---|---|---|
| exp_0001 | smoke | 55.3809 | kept | — | L1/L2 weight regularization (weight decay) (doi:10.1007/s429 |
| exp_0002 | smoke | 55.3893 | kept | — | L1/L2 weight regularization (weight decay) (doi:10.1007/s429 |
| exp_0003 | verify | 55.3248 | kept | — | L1/L2 weight regularization (weight decay) (doi:10.1007/s429 |
| exp_0004 | full | 55.3749 | kept | — | L1/L2 weight regularization (weight decay) (doi:10.1007/s429 |
| exp_0005 | verify | 55.4045 | kept | — | L1/L2 weight regularization (weight decay) (doi:10.1007/s429 |
| exp_0006 | full | 55.4262 | kept | — | L1/L2 weight regularization (weight decay) (doi:10.1007/s429 |
| exp_0007 | smoke | 55.587 | kept | -1.8→0.2622 | Shallow architectures / limiting network depth (doi:10.1016/ |
| exp_0008 | verify | 55.5907 | kept | — | Shallow architectures / limiting network depth (doi:10.1016/ |
| exp_0010 | smoke | 55.2871 | kept | -0.6→-0.0377 | Semi-supervised learning leveraging unlabelled tabular sampl |
| exp_0009 | smoke | 55.2911 | kept | -0.6→-0.0337 | Modular ensemble of multiple DNNs with varying depth and str |
| exp_0011 | verify | 55.2724 | kept | — | Semi-supervised learning leveraging unlabelled tabular sampl |
| exp_0012 | full | 55.2877 | kept | — | Semi-supervised learning leveraging unlabelled tabular sampl |
| exp_0013 | verify | 55.238 | kept | — | Modular ensemble of multiple DNNs with varying depth and str |
| exp_0014 | full | 55.2687 | kept | — | Modular ensemble of multiple DNNs with varying depth and str |
| exp_0015 | smoke | 55.4445 | kept | -0.5→0.2065 | Modular ensemble of multiple DNNs with varying depth and str |
| exp_0016 | smoke | 55.7126 | kept | -0.4→0.4746 | hyperparameter-tuning skill (shallow wide MLP, moderate lr,  |
| exp_0017 | verify | 55.449 | kept | — | Modular ensemble of multiple DNNs with varying depth and str |
| exp_0018 | verify | 55.7355 | kept | — | hyperparameter-tuning skill (shallow wide MLP, moderate lr,  |

## 🧠 Accumulated skills (self-improvement)

- [hyperparameter-tuning, w=0.55] On small tabular MLPs near a tuned frontier, lowering LR and increasing epochs does not reliably yield finer optima; predicted gains from 'finer convergence' are unreliable. Validate such claims with a quick LR/epoch sweep before assuming improvement, and treat sub-RMSE differences (<1 unit) as likely noise rather than real gains.
- [model_capacity, w=0.55] On small tabular datasets, widening a shallow MLP's hidden layer (e.g., to 128) yields negligible val_rmse improvement and can slightly hurt; capacity is rarely the bottleneck. Before scaling width, verify via a learning-curve or capacity sweep that the model is underfitting, and prefer regularization/feature tuning over wider layers when predicting gains.
- [hyperparameter-tuning, w=0.50] For small tabular regression tasks like diabetes-mlp, a shallow wide MLP (depth=1, hidden=64) with moderate lr (0.01), mild weight decay (1e-3), and extended epochs (400) reliably beats the baseline. Use this as a starting configuration and tune one factor at a time rather than combining several changes, so individual effects can be attributed.
- [regularization, w=0.40] On small tabular regression (e.g. diabetes, ~442 samples) with a shallow wide MLP, increasing weight_decay to ~5e-3 yields only marginal RMSE gains over the baseline (55.24 vs 55.32). Treat L2 strength as a low-impact lever here; prioritize other changes (feature scaling, capacity, learning-rate schedule) before sweeping weight_decay, and validate predicted vs measured improvement before assuming regularization is the dominant lever.

## 🎯 Agent calibration (predict-then-verify)

- debate: 100% of 7 go/no-go calls were correct.
- reasoner: predicted the right direction 40% of 5 times; mean magnitude error 0.954.

## 🔁 Agent trace (every call, auditable)

`advisor:4 · critic:Contrarian:10 · critic:Innovator:10 · critic:Pragmatist:10 · director:3 · lit_scout:2 · reasoner:12 · reflector:4`

See [`paper.md`](paper.md) for the write-up this run produced, and [`experiments.jsonl`](experiments.jsonl) for the raw ledger.
