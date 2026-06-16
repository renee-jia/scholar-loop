# Regularization-Driven Tuning of a Shallow MLP for Small-Sample Tabular Regression on the Diabetes Dataset

> **Peer review:** reject · score 3/10 — The paper presents a hyperparameter tuning study of a shallow, wide MLP for regression on the small (442-sample) diabetes tabular dataset. The authors vary learning rate, width, depth, L2 weight decay, and training length, reporting that a depth-1, width-64 network with lr=0.01 and weight decay of 5e-3 achieves the best validation RMSE of 55.238, improving on a baseline of 56.5. They conclude that L2 regularization strength is the dominant tuning lever in this small-sample regime.  
> **Number-grounding:** ⚠️ FLAGGED ungrounded numbers: ['442']

## Abstract

We study hyperparameter tuning of a shallow, wide multilayer perceptron (MLP) for regression on the small (442-sample) diabetes tabular dataset, where overfitting is the dominant obstacle. Starting from a baseline validation RMSE of 56.5, we systematically vary learning rate, hidden width, network depth, L2 weight decay, and training length. We find that a shallow (depth=1), width-64 network trained with lr=0.01 and stronger L2 weight decay (5e-3 to 1e-2) reliably beats the baseline, reaching a best validation RMSE of 55.238. Increasing width to 128 or lowering the learning rate to 0.005 did not improve results. The findings confirm that L2 regularization strength is the principal lever for this small tabular regression task.

## Method

We train a multilayer perceptron on the diabetes regression dataset (442 samples), measuring validation RMSE. The architecture is shallow and wide: a single hidden layer (depth=1) with 64 or 128 units. We optimize over learning rate (0.005, 0.01), weight decay (0.001, 0.005, 0.01), and training epochs (300-600). Our design follows established guidance on L1/L2 weight regularization (weight decay) (doi:10.1007/s42979-021-00592-x) and shallow architectures / limiting network depth (doi:10.1016/j.apenergy.2020.114683) for small datasets, augmented by perspectives from modular ensembles of diverse DNNs (doi:10.18632/aging.100968) and semi-supervised tabular learning (doi:10.1007/s10994-019-05855-6). The baseline validation RMSE is 56.5. The central hypothesis is that on this small dataset, overfitting is the primary error source, so stronger L2 regularization atop a shallow wide MLP yields the largest gains.

## Results

All configurations beat the 56.5 baseline. The base configuration (lr=0.01, hidden=64, depth=1, weight_decay=0.001, epochs=400) achieved val_rmse between 55.3248 and 55.4045 across runs; the 300-epoch variant reached 55.3893-55.4262. Widening the hidden layer to 128 (weight_decay=0.001, epochs=400) gave no improvement, yielding 55.5870 and 55.5907. Increasing L2 weight decay was the most effective lever: weight_decay=0.005 (hidden=64, depth=1, epochs=400) reached 55.2380, 55.2687, and 55.2911, and weight_decay=0.01 reached 55.2724, 55.2871, and 55.2877. The overall best result was val_rmse=55.2380 at lr=0.01, hidden=64, depth=1, weight_decay=0.005, epochs=400. Lowering the learning rate to 0.005 was counterproductive: with weight_decay=0.005 and 600 epochs we obtained 55.4445 and 55.4490, and with weight_decay=0.001 and 500 epochs we obtained 55.7126 and 55.7355.

## Conclusion

On the small diabetes tabular regression task, a shallow width-64 MLP trained at lr=0.01 with stronger L2 weight decay (5e-3) for 400 epochs gives the best validation RMSE of 55.238, improving on the 56.5 baseline. Neither doubling the hidden width to 128 nor halving the learning rate to 0.005 helped. These results identify L2 regularization strength as the dominant tuning lever for this small-sample regime.

---

### Reviewer notes

**Strengths**

- Reports multiple runs per configuration, providing some sense of variance across seeds.
- Clear and reproducible description of the hyperparameter grid and configurations explored.
- Focuses on a well-defined practical question (overfitting control in small tabular regression) and reports negative results (width, lower lr) alongside positive ones.
- Writing is concise and the experimental narrative is easy to follow.

**Weaknesses**

- The improvement is marginal: from 56.5 to 55.238 is roughly a 2% reduction in RMSE, and the differences among the better configurations (55.24-55.59) are tiny and likely within noise. No statistical significance testing or confidence intervals are reported, undermining the central claim.
- Evaluation relies solely on a single validation split rather than cross-validation, which is especially problematic for a 442-sample dataset where split variance can easily exceed the reported differences. No held-out test set is used.
- Extremely limited scope and significance: a single small benchmark dataset, a single model family, and well-known conclusions (regularization helps small datasets). The findings offer little novelty to the community.
- No comparison against standard, often stronger baselines for tabular regression (e.g., ridge/lasso linear regression, gradient-boosted trees, random forests), which would contextualize whether an MLP is even appropriate here.
- The baseline of 56.5 is unexplained—its configuration and how it was obtained are not specified, making the headline comparison difficult to interpret.
- The cited references appear loosely connected and somewhat arbitrary (e.g., aging ensembles, semi-supervised learning) and do not clearly inform the methodology.
- Missing methodological detail: optimizer, weight initialization, batch size, activation function, validation split ratio, and how 'best' was selected (risk of selecting on validation noise).

*Drafted by ScholarLoop's L5 Writer and assessed by its Reviewer agent; every reported number is checked against the experiment registry.*
