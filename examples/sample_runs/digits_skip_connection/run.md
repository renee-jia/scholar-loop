# ScholarLoop — autonomous run log

A complete run on the real **digits-mlp** torch engine, driven by **claude-opus-5-5**. Every experiment below is a real PyTorch training run; every decision is a real LLM call.

- **24** experiments · **61** LLM calls · 238590+51410 tokens · ≈ **$1.983**
- baseline to beat: 3.8% val_top1_err

## 💡 Chosen idea (human-selected in a research session)
> **Pre-registered paired equivalence test: zero-parameter identity skip vs. weight-identical plain MLP on sklearn-digits** — This is the grounded version of the author's two-arm idea. Both arms share one architecture: an input projection 64→W (W = hidden = 128), then two hidden blocks f_l(h) = ReLU(Linear_{W→W}(h)), then a 10-way head. The residual arm computes h_{l+1} = h_l + f_l(h_l). The plain arm computes h_{l+1} = f_l(h_l). The identity skip has no weights, so the two arms have exactly the same parameter tensors by construction and need no width compensation. To isolate the path, each seed gives both arms bit-identical initial weights, the same minibatch order and the same number of optimizer steps. The only difference left is the '+ h_l' term. The primary contrast is pre-registered as a TOST equivalence test on the paired difference in final val_top1_err, with margin ±0.5 pp. That makes a null decisive rather than inconclusive. The 'trains differently' confound is handled in two ways. Each arm gets its own lr, picked on an inner split carved from the training data only. As a robustness check, both arms also run at a shared lr grid. Step-resolved validation curves separate speed from final error.  
> *hypothesis:* At two hidden blocks on sklearn-digits, the identity skip makes no practically meaningful difference to final validation error once parameters, initial weights, data order, step count and per-arm lr are matched (H1). Any effect is confined to early-training speed, consistent with the optimization-landscape account in [arXiv:1909.04653] and the depth-scaled shallow-path account in [openalex:W3007689282], which predicts a factor near 1 at this depth.  
> *builds on:* arXiv:1909.04653, openalex:W3007689282, arXiv:1605.07648, arXiv:1608.02908


## 🪜 Experiments (real torch · multi-fidelity funnel)

| id | tier | val_top1_err | held-out test | verdict | idea test | predicted→measured | grounded source |
|---|---|---|---|---|---|---|---|
| exp_0001 | smoke | 3.4722% | — | kept | treatment · edit | — | arXiv:1909.04653 (shortcuts change the optimization landscap |
| exp_0003 | smoke | 4.5139% | — | discarded | control | — | arXiv:1909.04653; openalex:W3007689282 |
| exp_0002 | smoke | 5.2083% | — | discarded | control | — | arXiv:1909.04653; openalex:W3007689282 |
| exp_0004 | verify | 3.592% | — | kept | treatment · edit | — | arXiv:1909.04653 (shortcuts change the optimization landscap |
| exp_0005 | full | 3.758% | — | kept | treatment · edit | — | arXiv:1909.04653 (shortcuts change the optimization landscap |
| exp_0006 | verify | 3.8235% | — | discarded | control | — | arXiv:1909.04653; openalex:W3007689282 |
| exp_0007 | smoke | 4.5139% | — | discarded | treatment · edit | -0.1→1.0417 | arXiv:1909.04653; openalex:W3007689282 |
| exp_0008 | smoke | 4.5139% | — | discarded | treatment | -0.05→1.0417 | arXiv:1909.04653 (shortcuts alter the landscape, not express |
| exp_0011 | smoke | 2.0833% | — | kept | — · edit | -0.85→-1.3889 | arXiv:2605.01172 (minibatch SGD beats full-batch GD on small |
| exp_0010 | smoke | 2.4306% | — | kept | — · edit | -0.9→-1.0416 | arXiv:2605.01172 (minibatch SGD vs full-batch GD on small ML |
| exp_0009 | smoke | 2.0833% | — | kept | — · edit | -0.8→-1.3889 | arXiv:2605.01172 (minibatch SGD vs full-batch GD on small ML |
| exp_0012 | verify | 2.5495% | — | kept | — · edit | — | arXiv:2605.01172 (minibatch SGD vs full-batch GD on small ML |
| exp_0013 | full | 2.7841% | — | kept | — · edit | — | arXiv:2605.01172 (minibatch SGD vs full-batch GD on small ML |
| exp_0014 | verify | 2.5495% | — | kept | — · edit | — | arXiv:2605.01172 (minibatch SGD beats full-batch GD on small |
| exp_0015 | full | 2.7841% | — | kept | — · edit | — | arXiv:2605.01172 (minibatch SGD beats full-batch GD on small |
| exp_0016 | verify | 2.5491% | — | kept | — · edit | — | arXiv:2605.01172 (minibatch SGD vs full-batch GD on small ML |
| exp_0017 | full | 2.7838% | — | kept | — · edit | — | arXiv:2605.01172 (minibatch SGD vs full-batch GD on small ML |
| exp_0020 | smoke | 2.0833% | — | kept | — | -0.2→-0.3473 | Krogh & Hertz 1992, 'A Simple Weight Decay Can Improve Gener |
| exp_0019 | smoke | 1.7361% | — | kept | — | -0.23→-0.6945 | arXiv:2605.01172 (minibatch SGD slows memorization relative  |
| exp_0018 | smoke | 1.7361% | — | kept | — | -0.33→-0.6945 | arXiv:1810.12281 (Zhang et al., Three Mechanisms of Weight D |
| exp_0021 | verify | 2.3176% | — | kept | — | — | arXiv:1810.12281 (Zhang et al., Three Mechanisms of Weight D |
| exp_0022 | verify | 2.3176% | — | kept | — | — | arXiv:2605.01172 (minibatch SGD slows memorization relative  |
| exp_0023 | verify | 2.3172% | — | kept | — | — | Krogh & Hertz 1992, 'A Simple Weight Decay Can Improve Gener |
| exp_0024 | full | 2.575% | — | kept | — | — | Krogh & Hertz 1992, 'A Simple Weight Decay Can Improve Gener |

## 🧠 Accumulated skills (self-improvement)

- [optimization_budget_confound, w=0.60] Switching from 120 full-batch GD steps to minibatch SGD (batch 64, ~2.7k steps, warmup plus cosine decay) cut val_top1_err from 3.76 to 2.78 on digits-mlp. That is about a 26% relative drop, from one run with no seed variation. This change confounds step count with gradient noise, so it does not show that minibatch noise or flatter minima caused the gain. Before tuning architecture on small datasets, check that the baseline is not under-optimized: train loss should have plateaued, or more steps should no longer help. To separate the two causes, run full-batch GD for the same ~2.7k steps with the same schedule. Compare frontier configs only at matched update budgets, and state a numeric predicted effect so calibration can be measured.
- [architecture_change_calibration, w=0.50] Do not assume that adding an identity skip path, even at depth 2, is trajectory-neutral at a fixed lr. Here the prediction was about -0.1 pp and the measured change was +1.04 pp (worse). Before testing skip vs. plain, keep the per-layer signal scale matched. Either scale the residual sum (e.g., multiply it by 1/sqrt(2)) or zero-initialize the residual branch's final layer. Alternatively, run a small lr sweep for each arm (e.g., 0.03, 0.05, 0.1) rather than sharing lr=0.1. Use at least 3 seeds per arm before concluding anything, and widen prediction intervals for structural changes to at least ±1 pp.
- [regularization_tuning, w=0.40] Once the small MLP is optimization-saturated (batch 64, about 2.7k cosine steps, low train loss), try stronger regularization before adding capacity or steps. Raising weight decay from 1e-4 to 4e-4 lowered the 5-seed mean val error from 2.78 to 2.58. This is one comparison, so treat it as a lead rather than an established effect. Next, bracket weight decay at 8e-4 and 1.6e-3 with all else fixed. Report per-seed deltas to check whether the gain sits mainly on the high-error seeds (e.g., seed 4). Always state a numeric predicted delta so calibration can be tracked.

## 🎯 Agent calibration (predict-then-verify)

- debate: 64% of 11 go/no-go calls were correct.
- reasoner: predicted the right direction 75% of 8 times; mean magnitude error 0.56.

## 🛑 Governor

The campaign stopped itself: round cap reached (4 rounds).

## 🔁 Agent trace (every call, auditable)

`advisor:4 · critic:Contrarian:12 · critic:Innovator:12 · critic:Pragmatist:12 · director:1 · lit_scout:1 · reasoner:12 · reflector:4`

See [`paper.md`](paper.md) for the write-up this run produced, and [`experiments.jsonl`](experiments.jsonl) for the raw ledger.
