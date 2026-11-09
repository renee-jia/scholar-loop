# Wider, Deeper ReLU MLPs with L2 Weight Decay Reduce Validation Error Below a 5.0 Baseline

> **Peer review:** reject · score 2/10 — The paper reports six experiments training a 256-unit, depth-2 ReLU MLP with L2 weight decay (0.0005) under two learning-rate/epoch regimes, claiming all runs fall below a 5.0 val_top1_err baseline, with a best result of 3.8222 (lr=0.05, 120 epochs). It concludes that combining wider/deeper ReLU MLPs with L2 weight decay improves generalization.  
> **Number-grounding:** ✅ every number traces to a recorded measurement

## Abstract

We investigate whether a wider, two-hidden-layer ReLU multilayer perceptron (MLP) combined with stronger L2 weight decay and extended training can improve generalization over a 5.0 val_top1_err baseline. Across six experiments using a 256-unit, depth-2 architecture with weight decay of 0.0005, all runs fell below the baseline, achieving val_top1_err between 3.8222 and 4.4444. The best configuration (lr=0.05, 120 epochs) reached 3.8222. These results confirm that combining deeper ReLU MLPs (doi:10.1186/s40537-021-00444-8) with L2 regularization (doi:10.1007/s10462-023-10466-8) lowers validation error.

## Method

We trained MLPs with two hidden layers (depth=2) and 256 hidden units, using ReLU activations to mitigate vanishing gradients (doi:10.1186/s40537-021-00444-8) and L2 weight decay of 0.0005 to reduce overfitting (doi:10.1007/s10462-023-10466-8). Two training regimes were evaluated: (i) lr=0.05 with 120 epochs, and (ii) lr=0.03 with 150 epochs. The target was to reduce val_top1_err below the 5.0 baseline.

## Results

All six runs improved on the 5.0 baseline. With lr=0.05 and 120 epochs (hidden=256, depth=2, weight_decay=0.0005), val_top1_err was 4.2222 (exp_0001), 3.9259 (exp_0004), and 3.8222 (exp_0005), the best result observed. With lr=0.03 and 150 epochs (same architecture and weight decay), val_top1_err was 4.4444 (exp_0003), 4.2222 (exp_0006), and 4.1778 (exp_0007). The lr=0.05 / 120-epoch regime produced the lowest error (3.8222).

## Conclusion

A wider, depth-2 ReLU MLP with L2 weight decay of 0.0005 consistently reduced val_top1_err below the 5.0 baseline, with the best configuration (lr=0.05, 120 epochs) reaching 3.8222. These findings support combining deeper ReLU MLPs (doi:10.1186/s40537-021-00444-8) with L2 weight decay (doi:10.1007/s10462-023-10466-8) for improved generalization.

---

### Reviewer notes

**Strengths**

- The experimental setup and hyperparameters are reported clearly enough to be reproducible at a basic level.
- All reported runs are below the stated baseline, providing internally consistent evidence for the narrow claim.
- Results are tabulated per-experiment, allowing the reader to see variance across runs.

**Weaknesses**

- No description of the dataset, task, or even what val_top1_err refers to; the units and scale (3.8-4.4) are ambiguous and never defined (percentage? out of 100? something else?).
- No baseline architecture or training details are given, so the comparison to '5.0' is unanchored—we cannot tell if the baseline used the same data, splits, or budget.
- The experimental design conflates multiple changes (width, depth, weight decay, learning rate, epochs) without ablations, so no causal claim about any single factor (e.g., 'deeper' or 'L2') is supported. The title and conclusions overstate what was tested.
- Only two hyperparameter configurations with three runs each; no statistical analysis, no standard deviations, no significance testing despite drawing conclusions about which regime is 'best'.
- The cited DOIs are used as if they validate the specific empirical claims, but the work itself does not establish the general conclusions it attributes to them.
- Title says 'Wider, Deeper' but the network is fixed at depth 2 and width 256 with no variation, so the comparative adjectives are unjustified—there is no width/depth sweep.
- No test set, no comparison to alternative regularization or architectures, and no discussion of overfitting beyond asserting weight decay 'reduces' it.
- Trivial significance: the contribution amounts to a small hyperparameter tuning exercise on an undescribed task with no generalizable insight.

*Drafted by ScholarLoop's L5 Writer and assessed by its Reviewer agent; every reported number is checked against the experiment registry.*
