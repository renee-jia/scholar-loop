# Literature review — residual connections versus parameter count in shallow MLP image classifiers

_5 papers · 3 themes · 5 open gaps · queries: `residual connections shallow networks benefit` · `skip connections optimization loss landscape` · `parameter-matched comparison residual MLP` · `residual networks behave like ensembles shallow` · `identity shortcut gradient flow initialization` · `MLP baselines small tabular image datasets`_

## Overview

The five papers mostly study residual connections in deep networks (tens to thousands of layers), where the skip path is credited with easing optimization. None tests a shallow MLP on a tiny dataset like sklearn-digits (1,797 8x8 images). None runs the control the author wants: a no-skip arm matched on parameter count and training steps. The literature offers three mechanistic accounts of why skips help:
(a) the skip keeps the early-training function near identity or shallow, aided by normalization/initialization [openalex:W3007689282];
(b) the skip plus a suitable initialization steers gradient descent away from spurious optima, even in a two-layer net [arXiv:1909.04653];
(c) the benefit comes from having paths of different lengths, not from identity pass-through itself [arXiv:1605.07648].
[arXiv:1608.02908] and [arXiv:2104.07393] are mostly architecture extensions that add shortcuts; they do not isolate the skip. Two points are relevant to H1. First, (b) is the only result at shallow depth, and it ties the benefit to initialization and normalization, not capacity. Second, (a) predicts that at depth 2–3 the depth-dependent downscaling effect is negligible. Both lean toward H1 but neither tests it. With logistic regression already at ~3.8% error, the headroom is small. Any effect will need many seeds and confidence intervals to separate from noise.

## Themes

### Why skips help optimization: mechanism at initialization

These papers attribute residual benefits to optimization dynamics, not capacity. In normalized deep ResNets, the residual branch is downscaled by about 1/sqrt(depth), which keeps blocks near identity early in training [openalex:W3007689282]. In a two-layer convolutional ResNet, a properly initialized shortcut lets gradient descent avoid a spurious local optimum [arXiv:1909.04653]. Both predict effects that depend on depth and initialization. For a shallow MLP, that suggests any skip benefit should show up as a training-speed or initialization-sensitivity effect, not a capacity effect.

- [openalex:W3007689282](https://openalex.org/W3007689282) *Batch Normalization Biases Deep Residual Networks Towards Shallow Paths* (2020, 18 cites)
- [arXiv:1909.04653](http://arxiv.org/abs/1909.04653) *Towards Understanding the Importance of Shortcut Connections in Residual Networks* (2019, 39 cites)

### Is the identity path necessary? Alternatives and controls

FractalNet matches ResNet accuracy with no identity pass-through, relying on multiple path lengths [arXiv:1605.07648]. This is the closest thing in the set to a control showing the residual form is not fundamental. However, the comparison is not parameter-matched in the sense the author wants, and it is far deeper than an MLP.

- [arXiv:1605.07648](http://arxiv.org/abs/1605.07648) *FractalNet: Ultra-Deep Neural Networks without Residuals* (2016, 955 cites)

### Adding more shortcuts to deep architectures

These papers add skips to architectures and report gains: multilevel shortcuts in RoR [arXiv:1608.02908] and residual connections in capsule networks [arXiv:2104.07393]. Such additions typically confound the skip with depth, extra parameters (projection shortcuts), and the training schedule. That is exactly the confound the author wants to remove.

- [arXiv:1608.02908](http://arxiv.org/abs/1608.02908) *Residual Networks of Residual Networks: Multilevel Residual Networks* (2017, 415 cites)
- [arXiv:2104.07393](http://arxiv.org/abs/2104.07393) *Training Deep Capsule Networks with Residual Connections* (2021, 16 cites)

## Open gaps

1. **Two-arm test on sklearn-digits: a 2–3-layer MLP with a residual path versus a no-skip MLP matched on parameter count and optimizer steps, reporting val_top1_err over 20 or more seeds with confidence intervals. Use identity skips with equal widths, so the skip adds zero parameters, and compare against logistic regression at ~3.8%.**  
   Every paper here evaluates deep nets or stylized theory. None isolates the skip from capacity and training budget at shallow depth, which is exactly H1. (sources: `arXiv:1608.02908`, `arXiv:1605.07648`, `openalex:W3007689282`)

2. **Does the residual benefit at shallow depth (if any) vanish when the plain net gets an identity-like initialization or a downscaled final layer? Compare three arms: residual, plain, and plain with a near-identity or zero-init last layer.**  
   Two papers locate the benefit in initialization and normalization effects [openalex:W3007689282][arXiv:1909.04653]. If so, a plain net with a matched init should close the gap, but nobody has checked this on a small MLP. (sources: `openalex:W3007689282`, `arXiv:1909.04653`)

3. **Separate 'trains faster' from 'generalizes better': track val_top1_err as a function of optimizer steps, comparing the two arms under equal step counts and under early stopping.**  
   The mechanisms proposed are about optimization speed and avoiding bad optima, so a difference at a fixed budget may disappear at convergence. None of the papers reports matched-step curves for shallow models. (sources: `arXiv:1909.04653`, `openalex:W3007689282`)

4. **Depth sweep (1 to 6 hidden blocks) at fixed total parameters, to find the depth at which the skip starts to matter on digits.**  
   The theory predicts a depth-dependent crossover (downscaling of about sqrt(depth)). Where that crossover sits for small MLPs on tiny data is unmeasured, and it would turn a single negative result into a mechanistic one. (sources: `openalex:W3007689282`, `arXiv:1605.07648`)

5. **Path-diversity control: a plain MLP with a parallel short and long path (fractal-style join, no identity) versus a residual MLP at equal parameters.**  
   [arXiv:1605.07648] claims multiple path lengths, not identity, drive the effect, but only at ultra-depth. No one has tested whether that claim holds at shallow depth. (sources: `arXiv:1605.07648`)

## Trends

- Explanations of residual benefits have shifted from 'easier residual mapping' toward initialization and normalization effects [openalex:W3007689282][arXiv:1909.04653].
- Skepticism that identity skips are fundamental, with path diversity offered as an alternative [arXiv:1605.07648].
- Architectural work keeps stacking more shortcuts without capacity-matched ablations [arXiv:1608.02908][arXiv:2104.07393]. This leaves a clean parameter-matched, shallow-regime negative or positive result unclaimed.
- The literature concentrates on deep convolutional models. Shallow MLPs on tiny datasets are essentially unstudied in this set.

## Paper notes

| paper | contribution | limitation |
|---|---|---|
| [openalex:W3007689282](https://openalex.org/W3007689282) *Batch Normalization Biases Deep Residual Networks Towards Shallow Paths* (2020, 18 cites) | Shows that batch norm's benefit in deep ResNets comes from downscaling the residual branch relative to the skip, by a factor of about sqrt(depth), which biases the network toward shallow, identity-like paths early on. Proposes an initialization (SkipInit-style) that recovers trainability without BN. | The mechanism scales with depth, so it says little about 2–3-block MLPs where the factor is about 1. It does not examine accuracy differences at small scale or under matched parameter counts. |
| [arXiv:1909.04653](http://arxiv.org/abs/1909.04653) *Towards Understanding the Importance of Shortcut Connections in Residual Networks* (2019, 39 cites) | The only shallow-depth result in the set. In a two-layer non-overlapping conv ResNet, gradient descent with normalization and a suitable initialization provably avoids a spurious local optimum and reaches the global one in polynomial time. This implies the shortcut changes the optimization landscape, not expressivity. | Relies on a stylized teacher model, a convolutional (non-MLP) structure, and specific initialization assumptions. Gives no empirical comparison against a capacity-matched plain network on real data. |
| [arXiv:1605.07648](http://arxiv.org/abs/1605.07648) *FractalNet: Ultra-Deep Neural Networks without Residuals* (2016, 955 cites) | Shows that deep networks without any identity/residual connections, but with self-similar multi-length paths, match ResNets on CIFAR and ImageNet. This argues that the residual form itself is not essential. | Ultra-deep conv nets only. Parameter counts and training budgets are not tightly matched to the ResNet baselines, and there is nothing on shallow or fully connected models. |
| [arXiv:1608.02908](http://arxiv.org/abs/1608.02908) *Residual Networks of Residual Networks: Multilevel Residual Networks* (2017, 415 cites) | Adds level-wise shortcuts on top of ResNets (RoR) and reports improved CIFAR/SVHN error across ResNet variants. | Gains are confounded with architecture and training changes. There is no parameter-matched no-shortcut ablation, and it is about very deep nets. |
| [arXiv:2104.07393](http://arxiv.org/abs/2104.07393) *Training Deep Capsule Networks with Residual Connections* (2021, 16 cites) | Applies residual connections to make deeper capsule networks trainable. | No abstract is available, so claims cannot be verified. It concerns capsule architectures and is likely not parameter-controlled; peripheral to the question. |

## Corpus

- [arXiv:1605.07648](http://arxiv.org/abs/1605.07648) *FractalNet: Ultra-Deep Neural Networks without Residuals* (2016, 955 cites)
- [arXiv:1608.02908](http://arxiv.org/abs/1608.02908) *Residual Networks of Residual Networks: Multilevel Residual Networks* (2017, 415 cites)
- [openalex:W3007689282](https://openalex.org/W3007689282) *Batch Normalization Biases Deep Residual Networks Towards Shallow Paths* (2020, 18 cites)
- [arXiv:2104.07393](http://arxiv.org/abs/2104.07393) *Training Deep Capsule Networks with Residual Connections* (2021, 16 cites)
- [arXiv:1909.04653](http://arxiv.org/abs/1909.04653) *Towards Understanding the Importance of Shortcut Connections in Residual Networks* (2019, 39 cites)

*Retrieved from arXiv / OpenAlex and reviewed by ScholarLoop's LitReviewer agent. Every cited id is checked against the retrieved corpus.*
