# Literature review — small MLP image classification regularization

_20 papers · 4 themes · 4 open gaps · queries: `dropout regularization multilayer perceptron` · `weight decay small neural networks generalization` · `data augmentation small image datasets` · `label smoothing mixup regularization` · `batch normalization MLP classification` · `hyperparameter optimization small MLP`_

## Overview

The supplied list is mostly high-citation deep-learning landmarks plus several unrelated 2026 arXiv papers (econometrics, general relativity, kinetic PDEs, neutrino physics, sampling theory, survival statistics). Only a handful bear directly on regularizing a small MLP on sklearn-digits (8x8 grayscale, ~1797 images, 10 classes, val_top1_err): Adam [arXiv:1412.6980], decoupled weight decay / AdamW [arXiv:1711.05101], batch normalization [arXiv:1502.03167], and the dropout and data-augmentation regularizers used in AlexNet [doi:10.1145/3065386]. CNN and transformer architecture papers [arXiv:1409.1556, arXiv:1608.06993, arXiv:2010.11929, arXiv:2103.14030] supply indirect evidence: the inductive bias and data scale behind their results do not transfer to a tiny fully connected network. None of the papers evaluates on sklearn-digits or at this scale. Every claim about which regularizer helps a small MLP is therefore an extrapolation that needs direct testing. On a validation set of roughly 360 images, one image is about 0.28% error. Differences below about 1% need multiple seeds or cross-validation folds to be trusted.

## Themes

### Optimizer and weight-decay coupling

Adam supplies adaptive per-parameter step sizes. AdamW shows that L2 regularization and weight decay are not equivalent under Adam, and that decoupling them separates the choice of learning rate from the choice of decay strength. This is the most directly actionable theme for a small MLP whose main failure mode is overfitting.

- [arXiv:1412.6980](http://arxiv.org/abs/1412.6980) *Adam: A Method for Stochastic Optimization* (2014, 101800 cites)
- [arXiv:1711.05101](http://arxiv.org/abs/1711.05101) *Decoupled Weight Decay Regularization* (2017, 27112 cites)

### Normalization and stochastic regularizers

BatchNorm stabilizes training at higher learning rates and has a mild regularizing effect. AlexNet popularized dropout in fully connected layers and label-preserving data augmentation. Both were validated on large networks and datasets, not on tiny MLPs with small batches, where BatchNorm's batch statistics are noisy.

- [arXiv:1502.03167](http://arxiv.org/abs/1502.03167) *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* (2015, 33820 cites)
- [doi:10.1145/3065386](https://doi.org/10.1145/3065386) *ImageNet classification with deep convolutional neural networks* (2017, 114518 cites)

### Architectural inductive bias and data scale

VGG and DenseNet attribute gains to depth and connectivity in convolutional networks. ViT shows that architectures with weak inductive bias need large-scale pretraining to beat CNNs, and Swin reintroduces locality. The implied lesson for an MLP on raw pixels, which has no spatial prior, is that regularization and augmentation must substitute for missing inductive bias on small data. None of these papers tests that directly.

- [arXiv:1409.1556](http://arxiv.org/abs/1409.1556) *Very Deep Convolutional Networks for Large-Scale Image Recognition* (2014, 114148 cites)
- [arXiv:1608.06993](http://arxiv.org/abs/1608.06993) *Densely Connected Convolutional Networks* (2017, 50679 cites)
- [arXiv:2010.11929](http://arxiv.org/abs/2010.11929) *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2020, 47472 cites)
- [arXiv:2103.14030](http://arxiv.org/abs/2103.14030) *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (2021, 38130 cites)
- [doi:10.1145/3065386](https://doi.org/10.1145/3065386) *ImageNet classification with deep convolutional neural networks* (2017, 114518 cites)

### Off-topic or tangential

These papers are generative models, NLP pretraining, detection, protein structure, or non-ML fields. They offer at most loose analogies: graph-smoothness penalties [arXiv:2611.11496v1] and the KL term as a regularizer in a VAE [arXiv:1312.6114]. They provide no evidence for the reader's setting.

- [arXiv:1312.6114](http://arxiv.org/abs/1312.6114) *Auto-Encoding Variational Bayes* (2013, 36797 cites)
- [arXiv:1810.04805](http://arxiv.org/abs/1810.04805) *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018, 73198 cites)
- [arXiv:2005.14165](http://arxiv.org/abs/2005.14165) *Language Models are Few-Shot Learners* (2020, 33014 cites)
- [arXiv:1506.02640](http://arxiv.org/abs/1506.02640) *You Only Look Once: Unified, Real-Time Object Detection* (2016, 56543 cites)
- [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2) *Highly accurate protein structure prediction with AlphaFold* (2021, 44796 cites)
- [arXiv:1703.10593](http://arxiv.org/abs/1703.10593) *Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks* (2017, 24425 cites)
- [arXiv:2611.11496v1](http://arxiv.org/abs/2611.11496v1) *Graph-regularized synthetic control for spatial panels with post-only treated locations* (2026)
- [arXiv:2611.11495v1](http://arxiv.org/abs/2611.11495v1) *Smooth Rigidity Near Extremal Horizons* (2026)
- [arXiv:2611.11492v1](http://arxiv.org/abs/2611.11492v1) *A Legendre-Galerkin Method for Kinetic Models of Opinion Dynamics with Quasi-Invariant Consistency* (2026)
- [arXiv:2611.11474v1](http://arxiv.org/abs/2611.11474v1) *Machine learning assisted $ν/\barν$ discrimination in atmospheric neutrino events for CPV sensitivity in a 20-kton liquid scintillator detector* (2026)
- [arXiv:2611.11461v1](http://arxiv.org/abs/2611.11461v1) *Sharp Rates for Optimal Recovery of Bandlimited Functions from Localized Oversampling* (2026)
- [arXiv:2611.11456v1](http://arxiv.org/abs/2611.11456v1) *Model-assisted semi-supervised estimation of the Mann-Whitney probability under right censoring* (2026)

## Open gaps

1. **Does decoupled weight decay (AdamW) beat Adam with an L2 penalty on val_top1_err for a small MLP on sklearn-digits, after tuning both over a joint grid of learning rate and decay strength across several seeds?**  
   The equivalence result [arXiv:1711.05101] is shown only for large CNNs. A tiny, near-convex MLP may make the difference negligible, and no paper in the list reports results at this scale. (sources: `arXiv:1711.05101`, `arXiv:1412.6980`)

2. **Do the benefits of dropout and BatchNorm add up, cancel, or interfere in a small MLP trained with small batches?**  
   Dropout comes from [doi:10.1145/3065386] and BatchNorm's regularizing claim from [arXiv:1502.03167]. Neither paper ablates the two together, and small batches make batch statistics noisy. A 2x2 factorial experiment over seeds would settle it. (sources: `arXiv:1502.03167`, `doi:10.1145/3065386`)

3. **Which label-preserving augmentations work on 8x8 digit images: small shifts, small rotations, or pixel noise? How much of the gap to a CNN do they close for an MLP?**  
   Augmentation was validated on high-resolution natural images [doi:10.1145/3065386]. The view that weak inductive bias requires more data [arXiv:2010.11929] suggests augmentation should help MLPs most, but this has not been measured at 8x8 resolution, where a 1-pixel shift is a large perturbation. (sources: `doi:10.1145/3065386`, `arXiv:2010.11929`, `arXiv:2103.14030`)

4. **Is there a sweet spot in width and depth for regularized MLPs on digits? Do residual or dense skip connections help at depth 3–6?**  
   Evidence that depth and connectivity help [arXiv:1409.1556, arXiv:1608.06993] comes only from CNNs on large data. For tiny MLPs the trade-off between capacity and regularization is untested here. (sources: `arXiv:1409.1556`, `arXiv:1608.06993`)

## Trends

- Inductive bias and data scale have been traded off against each other: CNN priors [doi:10.1145/3065386, arXiv:1409.1556] gave way to transformers that depend on scale [arXiv:2010.11929], and then to models that reintroduce locality [arXiv:2103.14030]. For small data, this argues for explicit regularization or augmentation in place of architectural priors.
- Optimization and regularization are increasingly treated as coupled design choices, as with AdamW [arXiv:1711.05101] and BatchNorm [arXiv:1502.03167], rather than as independent knobs.
- Highly cited regularization evidence is concentrated at ImageNet scale. The small-data, small-model regime the reader works in is almost undocumented in this list, so claims must be re-validated with multiple seeds.

## Paper notes

| paper | contribution | limitation |
|---|---|---|
| [arXiv:1412.6980](http://arxiv.org/abs/1412.6980) *Adam: A Method for Stochastic Optimization* (2014, 101800 cites) | Introduces Adam, an adaptive moment-based optimizer and a robust default for small MLPs that needs little learning-rate tuning. | Contains no study of how Adam interacts with regularization. Its generalization relative to SGD on tiny datasets is not established, and the default hyperparameters were not tuned for problems with about 1.8k samples. |
| [arXiv:1711.05101](http://arxiv.org/abs/1711.05101) *Decoupled Weight Decay Regularization* (2017, 27112 cites) | Shows that L2 regularization and weight decay differ under Adam. Proposes AdamW, which decouples the decay step and makes the optimal decay strength less dependent on the learning rate. | Evaluated on CIFAR/ImageNet-scale CNNs. It is unclear whether the gap between L2 and decoupled decay is material for a two-layer MLP trained to near-zero training loss on digits. |
| [arXiv:1502.03167](http://arxiv.org/abs/1502.03167) *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* (2015, 33820 cites) | Introduces batch normalization, which permits higher learning rates and gives some regularization through noise in the batch statistics. | The 'internal covariate shift' rationale is asserted rather than isolated. Behavior with small batches and very shallow MLPs is untested, and its benefit may overlap with, or conflict with, dropout and weight decay. |
| [doi:10.1145/3065386](https://doi.org/10.1145/3065386) *ImageNet classification with deep convolutional neural networks* (2017, 114518 cites) | Demonstrates dropout in fully connected layers, data augmentation, and ReLU as practical tools against overfitting in a large network. | Large CNN on 1.2M images. Gives no ablation guidance for low-resolution 8x8 inputs, where crops and flips may not preserve the digit's label. |
| [arXiv:2010.11929](http://arxiv.org/abs/2010.11929) *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2020, 47472 cites) | Provides evidence that models with weak inductive bias depend heavily on data scale; without pretraining, ViT underperforms CNNs on mid-sized data. | Studies transformers with large pretraining. It does not quantify how much explicit regularization can replace data for a model with weak inductive bias, such as an MLP. |
| [arXiv:1608.06993](http://arxiv.org/abs/1608.06993) *Densely Connected Convolutional Networks* (2017, 50679 cites) | Shows that dense connectivity improves parameter efficiency and adds an implicit deep-supervision regularization in CNNs. | Convolutional only. Whether dense or skip connections help an MLP that is only a few layers deep is unexamined. |
| [arXiv:1409.1556](http://arxiv.org/abs/1409.1556) *Very Deep Convolutional Networks for Large-Scale Image Recognition* (2014, 114148 cites) | Shows that depth with small filters improves large-scale recognition. | Its finding that depth helps may invert in a regime with about 1.8k samples, where depth mainly adds capacity to overfit. |

## Corpus

- [doi:10.1145/3065386](https://doi.org/10.1145/3065386) *ImageNet classification with deep convolutional neural networks* (2017, 114518 cites)
- [arXiv:1409.1556](http://arxiv.org/abs/1409.1556) *Very Deep Convolutional Networks for Large-Scale Image Recognition* (2014, 114148 cites)
- [arXiv:1412.6980](http://arxiv.org/abs/1412.6980) *Adam: A Method for Stochastic Optimization* (2014, 101800 cites)
- [arXiv:1810.04805](http://arxiv.org/abs/1810.04805) *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018, 73198 cites)
- [arXiv:1506.02640](http://arxiv.org/abs/1506.02640) *You Only Look Once: Unified, Real-Time Object Detection* (2016, 56543 cites)
- [arXiv:1608.06993](http://arxiv.org/abs/1608.06993) *Densely Connected Convolutional Networks* (2017, 50679 cites)
- [arXiv:2010.11929](http://arxiv.org/abs/2010.11929) *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2020, 47472 cites)
- [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2) *Highly accurate protein structure prediction with AlphaFold* (2021, 44796 cites)
- [arXiv:2103.14030](http://arxiv.org/abs/2103.14030) *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (2021, 38130 cites)
- [arXiv:1312.6114](http://arxiv.org/abs/1312.6114) *Auto-Encoding Variational Bayes* (2013, 36797 cites)
- [arXiv:1502.03167](http://arxiv.org/abs/1502.03167) *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* (2015, 33820 cites)
- [arXiv:2005.14165](http://arxiv.org/abs/2005.14165) *Language Models are Few-Shot Learners* (2020, 33014 cites)
- [arXiv:1711.05101](http://arxiv.org/abs/1711.05101) *Decoupled Weight Decay Regularization* (2017, 27112 cites)
- [arXiv:1703.10593](http://arxiv.org/abs/1703.10593) *Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks* (2017, 24425 cites)
- [arXiv:2611.11496v1](http://arxiv.org/abs/2611.11496v1) *Graph-regularized synthetic control for spatial panels with post-only treated locations* (2026)
- [arXiv:2611.11495v1](http://arxiv.org/abs/2611.11495v1) *Smooth Rigidity Near Extremal Horizons* (2026)
- [arXiv:2611.11492v1](http://arxiv.org/abs/2611.11492v1) *A Legendre-Galerkin Method for Kinetic Models of Opinion Dynamics with Quasi-Invariant Consistency* (2026)
- [arXiv:2611.11474v1](http://arxiv.org/abs/2611.11474v1) *Machine learning assisted $ν/\barν$ discrimination in atmospheric neutrino events for CPV sensitivity in a 20-kton liquid scintillator detector* (2026)
- [arXiv:2611.11461v1](http://arxiv.org/abs/2611.11461v1) *Sharp Rates for Optimal Recovery of Bandlimited Functions from Localized Oversampling* (2026)
- [arXiv:2611.11456v1](http://arxiv.org/abs/2611.11456v1) *Model-assisted semi-supervised estimation of the Mann-Whitney probability under right censoring* (2026)

*Retrieved from arXiv / OpenAlex and reviewed by ScholarLoop's LitReviewer agent. Every cited id is checked against the retrieved corpus.*
