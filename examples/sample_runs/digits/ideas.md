# Research ideas — small MLP image classification regularization

_5 candidates · ranked by a pairwise Elo tournament (each pair judged twice, order swapped) · novelty check is lexical and advisory_

| # | idea | gap | Elo (W-D-L) | feasibility | impact | novelty check |
|---|---|---|---|---|---|---|
| 1 | ✅ **Sub-pixel augmentation for 8x8 digits: fractional shifts, rotations, noise** | Gap 3: Which label-preserving augmentations work on 8x8 digit images, and how much do they help an MLP? | 1261 (4-0-0) | 4 | 4 | ok (closest arXiv:2005.14165, 10%) |
| 2 | **Pixel-graph smoothness penalty on first-layer weights as a learned-free spatial prior** | Gap 3 (closing the gap between MLP and models with spatial bias on small data via regularization rather than architecture), also extending Gap 1's decay comparison. | 1229.3 (3-0-1) | 5 | 4 | ok (closest arXiv:2611.11496v1, 15%) |
| 3 | **2x2 factorial of dropout and small-batch BatchNorm (with GhostBN control)** | Gap 2: Do dropout and BatchNorm benefits add up, cancel, or interfere in a small MLP with small batches? | 1200.5 (2-0-2) | 5 | 3 | ok (closest arXiv:1502.03167, 9%) |
| 4 | **Depth-width sweep with residual MLP blocks under a fixed parameter budget** | Gap 4: Is there a width/depth sweet spot for regularized MLPs on digits, and do residual/dense skips help at depth 3-6? | 1168.9 (1-0-3) | 5 | 3 | ok (closest arXiv:2611.11474v1, 11%) |
| 5 | **Joint LR x decay grid: AdamW vs Adam+L2 on a tiny MLP** | Gap 1: Does decoupled weight decay (AdamW) beat Adam with an L2 penalty on val_top1_err for a small MLP on sklearn-digits after joint tuning across seeds? | 1140.3 (0-0-4) | 5 | 3 | ok (closest arXiv:1502.03167, 17%) |

## 1. Sub-pixel augmentation for 8x8 digits: fractional shifts, rotations, noise  ✅ selected

Integer 1-pixel shifts are a large perturbation at 8x8 resolution. Implement on-the-fly augmentation using bilinear fractional translations (±0.5 px), small rotations (±10°), elastic-lite affine jitter, and Gaussian pixel noise. Ablate each one and their combination, sweeping magnitude, to map which label-preserving transforms help an MLP and by how much.

- **Gap:** Gap 3: Which label-preserving augmentations work on 8x8 digit images, and how much do they help an MLP?
- **Hypothesis:** Small sub-pixel geometric augmentation gives the largest gain for an MLP lacking spatial priors. Integer 1-px shifts and strong noise give smaller gains or hurt. → *Fractional shift + ±10° rotation reduces val_top1_err by about 1 point relative to no augmentation, reaching about 2.0-2.5%. Integer ±1 px shifts gain under 0.4 points. Noise with sigma>0.15 (on [0,1] scale) increases error.*
- **Experiment:** Regularization edit: add an augmentation function in the training loop that reshapes to 8x8, applies an affine warp with bilinear sampling (pure numpy/torch grid_sample), then flattens. Arms: none; integer shift ±1; fractional shift ±{0.25, 0.5, 0.75}; rotation ±{5, 10, 15}°; noise sigma {0.05, 0.1, 0.2}; best shift+rotation; all combined. Fixed model: depth=1, hidden=256, AdamW lr=1e-3, wd=1e-3, epochs=240 (augmentation needs more epochs; also run 120 for parity). Run 5 seeds each. Compare to the no-aug arm and the 3.8 baseline. Report mean±std.
- **Builds on:** `doi:10.1145/3065386`, `arXiv:2010.11929`, `arXiv:2103.14030`
- **Why novel (agent):** Augmentation evidence in the review is for high-res natural images. ViT/Swin argue inductive bias versus data but never test MLPs at 8x8 or sub-pixel transforms.
- **Novelty check (lexical):** closest paper `arXiv:2005.14165` (10% term overlap)

## 2. Pixel-graph smoothness penalty on first-layer weights as a learned-free spatial prior

The MLP lacks a spatial prior. Add a regularizer on first-layer weights that penalizes differences between weights of 4-neighbor pixels on the 8x8 grid: the graph Laplacian quadratic form sum_h w_h^T L w_h. This injects a locality bias into a fully connected net and could substitute for convolution. Combine it with decoupled weight decay and compare it to plain decay of equal tuned strength.

- **Gap:** Gap 3 (closing the gap between MLP and models with spatial bias on small data via regularization rather than architecture), also extending Gap 1's decay comparison.
- **Hypothesis:** A Laplacian smoothness penalty on the first layer improves generalization beyond isotropic weight decay, because it encodes the locality prior that ViT/Swin show matters on small data. → *Best Laplacian+AdamW arm is 0.4-0.8 points below best AdamW-only, about 2.5-3.0% vs 3.0-3.4%. Gains are largest at hidden=512, where overparameterization is highest. The gains partially overlap with augmentation (sub-additive).*
- **Experiment:** Regularization edit: precompute the 64x64 grid Laplacian L. Add lambda*trace(W1 L W1^T) to the loss, where W1 is hidden×64. Sweep lambda in {0, 1e-4, 1e-3, 1e-2, 1e-1}, crossed with weight_decay in {1e-4, 1e-3, 1e-2} (AdamW, lr=1e-3), with hidden in {128, 512}, depth=1, epochs=120. Run 5 seeds. Secondary arm: best lambda + fractional-shift augmentation, to check additivity. Controls: AdamW-only best, and the 3.8 baseline. Visualize first-layer filters.
- **Builds on:** `arXiv:2611.11496v1`, `arXiv:1711.05101`, `arXiv:2010.11929`, `arXiv:2103.14030`
- **Why novel (agent):** Graph-smoothness penalties appear in the review only in an unrelated domain. The inductive-bias papers fix locality through architecture, never through a weight regularizer on an MLP, and AdamW does not consider structured decay.
- **Novelty check (lexical):** closest paper `arXiv:2611.11496v1` (15% term overlap)

## 3. 2x2 factorial of dropout and small-batch BatchNorm (with GhostBN control)

Run a factorial ablation of dropout (off/on) and BatchNorm (off/on) in a 2-layer MLP across batch sizes 16, 64 and 256. Add a fifth arm that replaces BN with LayerNorm, which keeps normalization but removes batch-statistic noise. This separates whether BN's benefit in this regime comes from normalization or from batch noise, and whether that noise substitutes for dropout.

- **Gap:** Gap 2: Do dropout and BatchNorm benefits add up, cancel, or interfere in a small MLP with small batches?
- **Hypothesis:** At small batch sizes, BN's batch noise and dropout are partially redundant regularizers. Their combined gain is sub-additive, with a negative interaction term, and the redundancy weakens at batch 256. → *At batch 16, interaction ≈ +0.3 to +0.6 error points (the combination is worse than additive). At batch 256 the interaction is near 0. LayerNorm+dropout matches or beats BN+dropout at batch 16. The best arm reaches about 2.5-3.0% vs the 3.8 baseline.*
- **Experiment:** Architecture edits: add an optional BN or LN layer after each linear layer, before ReLU, and optional dropout (p in {0.2, 0.4}, choosing the best p per arm on a separate fold). Use depth=2, hidden=256, AdamW lr=1e-3, wd=1e-3, epochs=120 with cosine schedule. Batch sizes {16, 64, 256}. Run 5 seeds × 5 CV folds per arm. Fit a two-way ANOVA (dropout × BN) per batch size and report the interaction coefficient with a CI. Use the same optimizer with no regularizers as reference, plus the 3.8 baseline.
- **Builds on:** `arXiv:1502.03167`, `doi:10.1145/3065386`
- **Why novel (agent):** BN and dropout were each introduced on large networks without a joint ablation. The small-batch, shallow-MLP regime, and an LN control that isolates batch noise, are absent from the reviewed work.
- **Novelty check (lexical):** closest paper `arXiv:1502.03167` (9% term overlap)

## 4. Depth-width sweep with residual MLP blocks under a fixed parameter budget

At matched parameter counts (~20k, 50k, 100k), compare plain MLPs and residual MLPs (pre-norm LayerNorm + skip) at depths 1-6. Also include a DenseNet-style variant that concatenates earlier features into the classifier. This locates the capacity/regularization sweet spot and tests whether skips rescue depth on ~1.8k samples.

- **Gap:** Gap 4: Is there a width/depth sweet spot for regularized MLPs on digits, and do residual/dense skips help at depth 3-6?
- **Hypothesis:** Without skips, error rises with depth beyond 2 at a fixed budget. Residual skips flatten this curve so that depth 3-4 matches or slightly beats the best shallow net. → *Plain depth-6 is about 1 point worse than plain depth-2. Residual depth-4 is within ±0.3 points of the best depth-1/2 model, around 2.7-3.2%. Dense concatenation gives no extra gain over residual.*
- **Experiment:** Architecture edits: plain MLP, ResMLP blocks (LN→Linear→ReLU→Linear + skip), and a dense variant (classifier sees the concat of all block outputs). Set the hidden knob per depth to match budgets of 20k, 50k and 100k params; depth in {1, 2, 3, 4, 6}. Use AdamW lr=1e-3, wd=1e-3, dropout 0.1, cosine schedule, epochs=120. Run 5 seeds. Plot error against depth per budget and architecture; compare to the 3.8 baseline.
- **Builds on:** `arXiv:1409.1556`, `arXiv:1608.06993`, `arXiv:1502.03167`
- **Why novel (agent):** Depth and connectivity results come only from CNNs at ImageNet scale. No reviewed paper tests skip connections in shallow MLPs under matched budgets with small data.
- **Novelty check (lexical):** closest paper `arXiv:2611.11474v1` (11% term overlap)

## 5. Joint LR x decay grid: AdamW vs Adam+L2 on a tiny MLP

Replace the default optimizer with Adam using either a coupled L2 penalty or decoupled weight decay (AdamW). Tune each over the same joint grid of learning rate and decay strength across 5 seeds. Compare the best validation error of each and the shape of its error surface, in particular whether the optimal decay drifts with learning rate. This tests whether the decoupling benefit reported for large CNNs survives at the 1-hidden-layer, ~1.8k-sample scale.

- **Gap:** Gap 1: Does decoupled weight decay (AdamW) beat Adam with an L2 penalty on val_top1_err for a small MLP on sklearn-digits after joint tuning across seeds?
- **Hypothesis:** With decoupled decay, the optimal decay strength stays nearly fixed as learning rate changes, and the best tuned AdamW error is at least as low as the best tuned Adam+L2 error. → *Best AdamW mean val_top1_err is about 0.3-0.6 points below Adam+L2, roughly 2.8-3.2% vs 3.3-3.6%, and both beat the 3.8 baseline. Under L2 the optimal decay shifts by more than 3x across the lr grid; under AdamW it shifts by less than 2x.*
- **Experiment:** Edit the optimizer in the train script to support adam_l2 and adamw. Grid: lr in {3e-4, 1e-3, 3e-3, 1e-2}; weight_decay in {0, 1e-4, 1e-3, 1e-2, 5e-2} for AdamW and the equivalent L2 coefficients for Adam. Hold hidden=128, depth=1, epochs=120 with cosine decay. Run 5 seeds per cell (stratified split held fixed). Report the best cell mean±std per method, test the gap with a paired t-test across seeds, and plot heatmaps of argmin decay against lr. Compare against the SGD baseline at lr=0.05, wd=1e-4 (3.8).
- **Builds on:** `arXiv:1412.6980`, `arXiv:1711.05101`
- **Why novel (agent):** The decoupling result was shown only on CIFAR/ImageNet CNNs. Adam's paper does not study regularization interplay. No paper in the review measures the surface at near-convex tiny-MLP scale.
- **Novelty check (lexical):** closest paper `arXiv:1502.03167` (17% term overlap)

## What the idea committee kept saying

_Recurring critiques from the tournament, distilled by the meta-reviewer; the next round of ideas is written against them. Support = the matches that show it._

1. Size the predicted effect against measurement resolution: on the ~360-image digits validation split, one image is ~0.3 points, so target effects of ≥1 point (several images). If a smaller effect is the point, state the seeds/folds/CI needed to resolve it.  _(support: r1:m2, r1:m3, r1:m5, r1:m6, r1:m8, r1:m9)_
2. Avoid equivalence or 'matches baseline' predictions and designs whose expected outcome is mostly null. Frame the hypothesis so that the predicted positive result is a clear, baseline-beating difference.  _(support: r1:m10, r1:m6, r1:m8)_
3. Don't re-test settled results (AdamW decoupling, residual/dense depth findings, dropout–BN disharmony) at small scale. Pitch a mechanism or claim that would actually shift beliefs, such as injecting an inductive bias through regularization instead of architecture.  _(support: r1:m1, r1:m10, r1:m3, r1:m4, r1:m7)_
4. Isolate the causal factor with a matched control, such as a LayerNorm arm to separate normalization from batch noise or from skip connections, or a no-aug or epoch-parity arm. Make sure the design's resolution can actually detect the claimed contrast; for example, a decay grid spaced 10x apart cannot detect a 2–3x shift. Check implementation details that could invert the prediction, such as PyTorch's AdamW scaling decay by lr.  _(support: r1:m1, r1:m3, r1:m6, r1:m9)_
5. Keep the run budget lean and within the fixed setup. Avoid sprawling grids and changes the fixed val_top1_err setup may not support, such as batch-size sweeps or CV resplits. Prefer a single cheap training-loop edit with a few arms × 5 seeds.  _(support: r1:m5, r1:m8)_

*Generated by ScholarLoop's Ideator from the literature review; pick one (or send feedback) to continue to experiments.*
