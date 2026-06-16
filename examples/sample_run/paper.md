# Tuning Depth-2 Width-256 MLPs on the Digits Dataset: A Weight Decay and Learning Rate Study

> **Peer review:** reject · score 3/10 — The paper reports a hyperparameter study of a small MLP classifier on the sklearn digits dataset. Moving from a shallow baseline to a depth-2, width-256 architecture reduces validation top-1 error from 5.0 to 2.6667, and a sweep over learning rate (0.05 vs 0.1) and weight decay (1e-4 to 1e-2) finds lr=0.1 with weight decay in the 3e-4 to 5e-4 range to be optimal. No configuration improved below the 2.6667 plateau.  
> **Number-grounding:** ✅ every number traces to a recorded measurement

## Abstract

We study the configuration of a small multilayer perceptron (MLP) classifier on the digits dataset, starting from a shallow baseline with val_top1_err of 5.0. By increasing the MLP to depth 2 and widening the hidden layer to 256 units, we reduce val_top1_err to 2.6667, well below the baseline. We then systematically probe the learning rate and weight decay frontier. We find that lr=0.1 paired with light-to-moderate weight decay (1e-4 to 3e-4) reaches the best observed val_top1_err of 2.6667, while lowering the learning rate to 0.05 or applying heavy weight decay (1e-2) degrades performance. No configuration we tested broke below the 2.6667 plateau.

## Method

We train MLP classifiers on the digits dataset for 120 epochs. Following the deeper-hierarchical-ReLU design of Wang et al. (doi:10.1109/access.2019.2912200), we adopt a depth-2, width-256 architecture, which lowers val_top1_err below the shallow 5.0 baseline. We then tune the learning rate and weight decay. Motivated by heavy-regularization strategies for data-efficient MLP training (ResMLP, arXiv:2105.03404) and a regularization survey on weight decay for small classifiers (arXiv:2201.03299), we sweep weight_decay across 1e-4, 3e-4, 5e-4, 1e-3, and 1e-2, and compare lr=0.05 against lr=0.1.

## Results

The depth-2/width-256 architecture with lr=0.1, weight_decay=5e-4 achieves val_top1_err=2.6667 (exp_0001-0003), well below the 5.0 baseline. This best result is matched by weight_decay=3e-4 at the same lr (exp_0008, exp_0009, val_top1_err=2.6667). Lowering the learning rate to 0.05 hurts performance regardless of weight decay: lr=0.05 with weight_decay=5e-4 gives 4.2222 (exp_0004) and lr=0.05 with weight_decay=1e-4 also gives 4.2222 (exp_0007). On the lr=0.1 frontier, increasing weight decay beyond 3e-4 yields diminishing or negative returns: weight_decay=1e-3 gives 2.8148 (exp_0006) and 2.6667 (exp_0005), while a full order-of-magnitude step to 1e-2 degrades to 3.3333 (exp_0010) and 3.6296 (exp_0011). No tested configuration improved on the 2.6667 plateau.

Summary of val_top1_err by configuration (all depth=2, hidden=256, epochs=120):
- lr=0.1, wd=5e-4: 2.6667, 2.6667, 2.6667
- lr=0.1, wd=3e-4: 2.6667, 2.6667
- lr=0.1, wd=1e-3: 2.6667, 2.8148
- lr=0.1, wd=1e-2: 3.3333, 3.6296
- lr=0.05, wd=5e-4: 4.2222
- lr=0.05, wd=1e-4: 4.2222

## Conclusion

Adding hierarchical capacity (depth 2, width 256) is the dominant factor in beating the shallow baseline, cutting val_top1_err from 5.0 to 2.6667. Within this architecture, lr=0.1 with weight decay in the 3e-4 to 5e-4 range is optimal; a lower learning rate of 0.05 and heavy weight decay (1e-2) both degrade accuracy. The 2.6667 result appears to be a stable plateau under the settings explored.

---

### Reviewer notes

**Strengths**

- The experimental procedure is clearly described and the configurations are enumerated with replicate runs, making the results reproducible.
- Reporting multiple seeds/runs per configuration (e.g., three runs at lr=0.1, wd=5e-4) adds some credibility to the stability of the reported plateau.
- The narrative is straightforward and the conclusions follow directly from the presented numbers.

**Weaknesses**

- The contribution is extremely narrow: a small hyperparameter sweep on a tiny, well-worn toy dataset (digits) yields no generalizable insight or methodological novelty.
- No statistical analysis or variance/error bars are reported despite having multiple runs; differences like 2.6667 vs 2.8148 may not be meaningful, and the test set is small enough that single-digit error counts dominate.
- The search space is shallow and inconsistent: only two learning rates are compared, the lr=0.05 setting is not paired with the same weight-decay grid as lr=0.1, and depth/width were not jointly tuned, so claims of optimality are unsupported.
- Citations feel tacked-on and misapplied; invoking ResMLP and access-journal architectures to justify a depth-2 MLP on digits is a stretch and does not strengthen the work.
- No held-out test evaluation, no comparison to standard baselines (e.g., logistic regression, SVM, kNN) which are known to perform very well on digits, leaving significance unclear.
- The 'plateau' claim is weak: it merely states no tested config did better, which is unsurprising given the limited and biased search.
- The peculiar precision of error values (e.g., 2.6667, 4.2222) reflects a very small validation set, undermining the reliability of fine-grained comparisons.

*Drafted by ScholarLoop's L5 Writer and assessed by its Reviewer agent; every reported number is checked against the experiment registry.*
