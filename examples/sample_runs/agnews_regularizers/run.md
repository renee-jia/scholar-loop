# ScholarLoop — autonomous run log

A complete run on the real **agnews-bow** torch engine, driven by **claude-opus-5-5**. Every experiment below is a real PyTorch training run; every decision is a real LLM call.

- **30** experiments · **56** LLM calls · 178909+54505 tokens · ≈ **$1.806**
- baseline to beat: 13.3% val_top1_err

## 💡 Chosen idea (human-selected in a research session)
> **XGBoost-style column subsampling and L1 shrinkage transferred to neural BoW** — XGBoost regularizes sparse inputs with per-tree column subsampling, L1/L2 penalties on leaf weights, and shrinkage. This idea transfers each mechanism to the BoW MLP. Column subsampling becomes word-level input dropout: random vocabulary columns are zeroed per batch, with a fresh vocabulary mask per step. Leaf L1 becomes an L1 penalty on embedding rows, which sparsifies the word features. Shrinkage becomes a low-lr, many-epoch schedule with Polyak/EMA weight averaging.  
> *hypothesis:* Vocabulary-column dropout plus row-wise L1 on the embedding lowers error more than standard hidden dropout plus L2. Most of the gain comes from column subsampling.  
> *builds on:* arXiv:1603.02754


## 🪜 Experiments (real torch · multi-fidelity funnel)

| id | tier | val_top1_err | held-out test | verdict | idea test | predicted→measured | grounded source |
|---|---|---|---|---|---|---|---|
| exp_0001 | smoke | 13.0625% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0002 | smoke | 16.1875% | — | discarded | treatment · edit | — | Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, a |
| exp_0003 | smoke | 13.9375% | — | discarded | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: column subsampli |
| exp_0004 | verify | 12.3542% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0005 | full | 12.6875% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0006 | verify | 13.3125% | — | discarded | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: column subsampli |
| exp_0007 | verify | 14.5417% | — | discarded | treatment · edit | — | Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, a |
| exp_0008 | smoke | 13.9375% | — | discarded | control | 0.55→0.875 | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0010 | smoke | 12.1875% | — | kept | treatment · edit | -0.15→-0.875 | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0009 | smoke | 12.1875% | — | kept | treatment · edit | -0.2→-0.875 | Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, a |
| exp_0011 | verify | 12.1042% | — | kept | treatment · edit | — | Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, a |
| exp_0012 | full | 12.55% | — | kept | treatment · edit | — | Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, a |
| exp_0013 | verify | 12.0833% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0014 | full | 12.55% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0015 | verify | 13.7708% | — | discarded | control | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0016 | smoke | 11.75% | — | kept | treatment · edit | -0.2→-0.4375 | arXiv:1603.02754 (Chen & Guestrin, XGBoost: shrinkage as a r |
| exp_0017 | smoke | 11.9375% | — | kept | treatment · edit | -0.25→-0.25 | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0018 | smoke | 11.9375% | — | kept | treatment · edit | -0.2→-0.25 | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0019 | verify | 11.5833% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: shrinkage as a r |
| exp_0020 | full | 12.025% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: shrinkage as a r |
| exp_0021 | verify | 11.8542% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0022 | full | 11.8875% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0023 | verify | 11.8542% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0024 | full | 11.8875% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0025 | smoke | 12.25% | — | kept | control · edit | 0.4→0.3125 | arXiv:1603.02754 (Chen & Guestrin, XGBoost: shrinkage as a r |
| exp_0027 | smoke | 12.1875% | — | kept | treatment · edit | -0.1→0.25 | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0026 | smoke | 12.0% | — | kept | treatment · edit | -0.15→0.0625 | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0028 | verify | 11.875% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0029 | verify | 11.9375% | — | kept | treatment · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree  |
| exp_0030 | verify | 12.125% | — | kept | control · edit | — | arXiv:1603.02754 (Chen & Guestrin, XGBoost: shrinkage as a r |

## 🧠 Accumulated skills (self-improvement)

- [experimental_design/confounded_bundle, w=0.45] The 'shrinkage' change bundled three edits: halved lr, doubled epochs (twice the compute) and EMA at 0.999. The gain (12.55 to 11.89, about 0.66 pt) cannot be attributed to the variance-reduction mechanism. Before adopting it as a frontier recipe, run two cheap ablations against the parent: (a) the same epoch budget with EMA only, and (b) lr/2 with 2x epochs and no EMA. Compare at equal training compute. Also record a numeric predicted effect so calibration can be tracked.
- [experimental_design/confounded_ablation, w=0.35] To claim a regularizer is the 'main source of gain', change one knob per experiment: vary column dropout with group-L1 held fixed, then vary L1 alone, so the effect can be attributed. Record both knobs explicitly in the logged config, which here showed neither. State a numeric predicted delta before running. Treat sub-0.2pt changes in val error, like 12.69 to 12.55 here, as noise unless replicated across seeds. Keeping this run is fine, but it supports no conclusion about XGBoost-style column subsampling.
- [regularization_redundancy_and_noise, w=0.30] Group-L1 at 1e-5 on embedding rows, added on top of col-drop 0.4, EMA and weight decay, moved val_top1_err by +0.02pp (11.875 vs 11.854). That is within single-seed noise and slightly worse, so there is no evidence of benefit. Before stacking another shrinkage regularizer on an already-regularized frontier, log the penalty term's magnitude relative to the CE loss and the change in row norms after one epoch. If the penalty is below about 1% of the loss, sweep the coefficient upward by 10x-100x, or skip the idea. Do not keep a candidate as an improvement unless it beats the parent by more than seed variance; use at least 2-3 seeds when the delta is under about 0.1pp.
- [experimental_protocol, w=0.30] Before keeping a regularization change (e.g., column dropout plus group-L1 on embeddings), run a matched ablation without it: same seed(s), config and epoch budget. Also log a numeric predicted delta so calibration error can be computed. A single absolute val error (12.69) with no baseline delta cannot support any claim that the XGBoost-style regularizers transfer, so treat this result as unverified rather than as a confirmed gain.

## 🎯 Agent calibration (predict-then-verify)

- debate: 75% of 12 go/no-go calls were correct.
- reasoner: predicted the right direction 78% of 9 times; mean magnitude error 0.296.

## 🛑 Governor

The campaign stopped itself: round cap reached (4 rounds).

## 🔁 Agent trace (every call, auditable)

`advisor:4 · critic:Contrarian:12 · critic:Innovator:12 · critic:Pragmatist:12 · reasoner:12 · reflector:4`

See [`paper.md`](paper.md) for the write-up this run produced, and [`experiments.jsonl`](experiments.jsonl) for the raw ledger.
