# Literature review — regularization for small text classifiers trained on limited labelled data

_20 papers · 5 themes · 5 open gaps · queries: `regularization text classification limited labeled data` · `dropout bag-of-words text classifier` · `mixup data augmentation text classification` · `label smoothing small dataset classification` · `word dropout noise injection NLP` · `weight decay early stopping small data`_

## Overview

The supplied list is almost entirely off-topic for regularization of small bag-of-words text classifiers on a 10k AG News subset. None of the papers studies dropout, weight decay, label smoothing, early stopping, data augmentation, or similar regularizers for text classification under limited labels. Most entries are highly cited landmarks from other areas: vision detection and backbones, protein structure, foundation LLMs, and graph SSL. There are also 2026 preprints on unrelated subjects such as graph coloring, connectomes, and pose ranking. A few are only indirectly relevant: [arXiv:1603.02754] gives a strong regularized, sparsity-aware baseline for BoW features; [arXiv:1609.02907] offers a semi-supervised route for limited labels; [arXiv:2005.14165] is a few-shot alternative to training small models. Methodological cautions in [arXiv:2611.14508v1] and [arXiv:2611.14519v1] about under-tuned baselines are directly applicable to how regularization comparisons should be run. The colleague should treat this list as a weak evidence base. The gaps below are experiments the setting can answer, not conclusions the papers support.

## Themes

### Strong non-neural baselines for sparse BoW features

Gradient-boosted trees with explicit L1/L2 penalties, shrinkage and column subsampling, plus sparsity-aware split finding, are a natural, heavily regularized competitor to small neural BoW classifiers. The paper benchmarks mostly tabular tasks, not text.

- [arXiv:1603.02754](http://arxiv.org/abs/1603.02754) *XGBoost: A Scalable Tree Boosting System* (2016, 62908 cites)

### Learning from limited labels via structure or pretraining

Two routes are represented. Graph-based semi-supervised propagation [arXiv:1609.02907] leverages unlabelled neighbours. Large-scale pretraining, then transfer or few-shot use, appears in [arXiv:2005.14165], [arXiv:1311.2524], [arXiv:2010.11929], [arXiv:2302.13971], [arXiv:2407.21783] and [arXiv:2303.08774]. These treat pretraining as the main regularizer when labels are scarce, but none isolates it against classic regularizers for small models.

- [arXiv:1609.02907](http://arxiv.org/abs/1609.02907) *Semi-Supervised Classification with Graph Convolutional Networks* (2016, 36115 cites)
- [arXiv:2005.14165](http://arxiv.org/abs/2005.14165) *Language Models are Few-Shot Learners* (2020, 33726 cites)
- [arXiv:1311.2524](http://arxiv.org/abs/1311.2524) *Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation* (2014, 33736 cites)
- [arXiv:2010.11929](http://arxiv.org/abs/2010.11929) *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2020, 51264 cites)
- [arXiv:2302.13971](http://arxiv.org/abs/2302.13971) *LLaMA: Open and Efficient Foundation Language Models* (2023, 23735 cites)
- [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) *The Llama 3 Herd of Models* (2024, 23856 cites)
- [arXiv:2303.08774](http://arxiv.org/abs/2303.08774) *GPT-4 Technical Report* (2023, 38274 cites)

### Fair comparison and tuning-budget methodology

These papers warn that comparisons are biased when baselines are under-tuned or inherit hyperparameters tuned on the favoured model [arXiv:2611.14508v1]. [arXiv:2611.14519v1] makes the same point about single, unjustified task-specific baselines and about how much of a model to adapt (from bias-only tuning to full fine-tuning). Both apply directly to regularization ablations.

- [arXiv:2611.14508v1](http://arxiv.org/abs/2611.14508v1) *When Does the Real Wiring Win? Testing connectome structure-function claims against unmeasured physiology* (2026)
- [arXiv:2611.14519v1](http://arxiv.org/abs/2611.14519v1) *Medical Image Segmentation: Foundation Model Adaptation versus Task-Specific Training* (2026)

### Architecture and capacity control

Capacity reduction through architecture is a form of implicit regularization. Examples are width multipliers [arXiv:1704.04861] and convolutional weight sharing [doi:10.1109/5.726791]. All evidence is from vision.

- [arXiv:1704.04861](http://arxiv.org/abs/1704.04861) *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications* (2017, 28387 cites)
- [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791) *Gradient-based learning applied to document recognition* (1998, 61780 cites)

### Off-topic entries

These papers have no bearing on text classifier regularization.

- [doi:10.32614/r.manuals](https://doi.org/10.32614/r.manuals) *R: A Language and Environment for Statistical Computing* (2000, 355987 cites)
- [arXiv:1506.01497](http://arxiv.org/abs/1506.01497) *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks* (2016, 74087 cites)
- [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2) *Highly accurate protein structure prediction with AlphaFold* (2021, 45813 cites)
- [arXiv:2103.14030](http://arxiv.org/abs/2103.14030) *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (2021, 38615 cites)
- [arXiv:2611.14518v1](http://arxiv.org/abs/2611.14518v1) *Toward Foundation Detection Models via Example-Based Grounding in EM-DETR* (2026)
- [arXiv:2611.14514v1](http://arxiv.org/abs/2611.14514v1) *Contrastive Discovery: Open-Ended Scientific Discovery over Competing Explanations* (2026)
- [arXiv:2611.14513v1](http://arxiv.org/abs/2611.14513v1) *Online Graph Coloring: Competitive Ratios Separating Randomized and Deterministic Algorithms* (2026)
- [arXiv:2611.14511v1](http://arxiv.org/abs/2611.14511v1) *Pyrite: Building and Benchmarking Protein-Ligand Pose Ranking Agents* (2026)

## Open gaps

1. **Does a fully tuned regularized linear model (logistic regression with L2 or L1 on TF-IDF BoW) match or beat a small MLP on val_top1_err on the 10k AG News subset? The test is to give each model its own hyperparameter search with an equal trial budget.**  
   No listed paper evaluates this setting. [arXiv:2611.14508v1] and [arXiv:2611.14519v1] show that comparisons under unequal tuning are biased, so any claimed regularization gain needs per-model tuning (the P2 estimand). (sources: `arXiv:2611.14508v1`, `arXiv:2611.14519v1`)

2. **Gradient-boosted trees versus neural BoW classifiers at 10k labels, comparing XGBoost's L1/L2 and subsampling regularizers against dropout and weight decay.**  
   [arXiv:1603.02754] claims strength on sparse data but reports no text-classification benchmarks. (sources: `arXiv:1603.02754`)

3. **Whether a kNN document graph built over the full AG News training pool, with labels only on the 10k subset, lowers error relative to supervised-only regularization when used for graph propagation.**  
   [arXiv:1609.02907] needs a given graph. How much a constructed similarity graph helps over dropout or weight decay alone has not been measured for BoW text. (sources: `arXiv:1609.02907`)

4. **Interaction of explicit regularizers (dropout, weight decay, label smoothing, early stopping) with model width in small BoW MLPs. Do they substitute for each other or stack?**  
   Width-as-capacity is studied only in vision [arXiv:1704.04861]. No listed paper crosses width with explicit regularizers on text. (sources: `arXiv:1704.04861`)

5. **Cost-matched comparison: a frozen pretrained embedding or LLM few-shot baseline versus a from-scratch regularized BoW model at equal labelled data.**  
   [arXiv:2005.14165] and [arXiv:2302.13971] imply pretraining dominates, but they never compare against a properly regularized small model under compute or latency constraints. (sources: `arXiv:2005.14165`, `arXiv:2302.13971`)

## Trends

- Pretraining at scale is increasingly treated as the default answer to label scarcity, displacing explicit regularization research ([arXiv:2005.14165], [arXiv:2407.21783]).
- Growing scrutiny of baseline tuning and fair comparisons ([arXiv:2611.14508v1], [arXiv:2611.14519v1]).
- Partial or parameter-efficient adaptation (for example bias-only tuning) as an implicit regularizer ([arXiv:2611.14519v1]).
- The supplied corpus contains almost no direct work on small text-classifier regularization, so the relevant literature must be gathered elsewhere.

## Paper notes

| paper | contribution | limitation |
|---|---|---|
| [arXiv:1603.02754](http://arxiv.org/abs/1603.02754) *XGBoost: A Scalable Tree Boosting System* (2016, 62908 cites) | Regularized boosting objective (L1/L2 on leaf weights, shrinkage, subsampling) with sparsity-aware splits suited to high-dimensional sparse inputs like BoW. | No text-classification or low-label experiments; whether it beats a tuned linear or MLP BoW model on 10k AG News is untested here. |
| [arXiv:1609.02907](http://arxiv.org/abs/1609.02907) *Semi-Supervised Classification with Graph Convolutional Networks* (2016, 36115 cites) | Semi-supervised node classification by propagating features over a graph, effective with very few labels. | It requires a graph, such as a document kNN or word co-occurrence graph, which AG News does not provide. Results are on citation networks, not news topics. |
| [arXiv:2005.14165](http://arxiv.org/abs/2005.14165) *Language Models are Few-Shot Learners* (2020, 33726 cites) | Shows scale enables few-shot classification without gradient updates, an alternative to training regularized small models. | Compute cost is incomparable to a small BoW model. It gives no evidence about what regularizes small models trained from scratch. |
| [arXiv:2611.14508v1](http://arxiv.org/abs/2611.14508v1) *When Does the Real Wiring Win? Testing connectome structure-function claims against unmeasured physiology* (2026) | Formalizes three tuning estimands: physiology tuned on the real wiring and transferred to nulls (P1), each wiring tuned on its own (P2), and P2 constrained by independent data (P3). It shows how tuning on the favoured model biases comparisons. | Its domain is neuroscience, and it gives no empirical ML regularization results. The transfer to ML ablations is by analogy. |
| [arXiv:2611.14519v1](http://arxiv.org/abs/2611.14519v1) *Medical Image Segmentation: Foundation Model Adaptation versus Task-Specific Training* (2026) | Compares adaptation depths (bias-only through full fine-tuning) against from-scratch baselines with justified configurations. | It covers medical segmentation only, not text or BoW. |
| [arXiv:1704.04861](http://arxiv.org/abs/1704.04861) *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications* (2017, 28387 cites) | Global width and resolution multipliers trade capacity against accuracy. | The evidence is from vision. Width as a regularizer for small-data text MLPs is not studied. |
| [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791) *Gradient-based learning applied to document recognition* (1998, 61780 cites) | Shows architectural priors (weight sharing) reduce effective capacity and improve generalization. | Its BoW inputs discard word order, so convolutional priors do not carry over directly. |

## Corpus

- [doi:10.32614/r.manuals](https://doi.org/10.32614/r.manuals) *R: A Language and Environment for Statistical Computing* (2000, 355987 cites)
- [arXiv:1506.01497](http://arxiv.org/abs/1506.01497) *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks* (2016, 74087 cites)
- [arXiv:1603.02754](http://arxiv.org/abs/1603.02754) *XGBoost: A Scalable Tree Boosting System* (2016, 62908 cites)
- [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791) *Gradient-based learning applied to document recognition* (1998, 61780 cites)
- [arXiv:2010.11929](http://arxiv.org/abs/2010.11929) *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2020, 51264 cites)
- [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2) *Highly accurate protein structure prediction with AlphaFold* (2021, 45813 cites)
- [arXiv:2103.14030](http://arxiv.org/abs/2103.14030) *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (2021, 38615 cites)
- [arXiv:2303.08774](http://arxiv.org/abs/2303.08774) *GPT-4 Technical Report* (2023, 38274 cites)
- [arXiv:1609.02907](http://arxiv.org/abs/1609.02907) *Semi-Supervised Classification with Graph Convolutional Networks* (2016, 36115 cites)
- [arXiv:1311.2524](http://arxiv.org/abs/1311.2524) *Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation* (2014, 33736 cites)
- [arXiv:2005.14165](http://arxiv.org/abs/2005.14165) *Language Models are Few-Shot Learners* (2020, 33726 cites)
- [arXiv:1704.04861](http://arxiv.org/abs/1704.04861) *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications* (2017, 28387 cites)
- [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) *The Llama 3 Herd of Models* (2024, 23856 cites)
- [arXiv:2302.13971](http://arxiv.org/abs/2302.13971) *LLaMA: Open and Efficient Foundation Language Models* (2023, 23735 cites)
- [arXiv:2611.14519v1](http://arxiv.org/abs/2611.14519v1) *Medical Image Segmentation: Foundation Model Adaptation versus Task-Specific Training* (2026)
- [arXiv:2611.14518v1](http://arxiv.org/abs/2611.14518v1) *Toward Foundation Detection Models via Example-Based Grounding in EM-DETR* (2026)
- [arXiv:2611.14514v1](http://arxiv.org/abs/2611.14514v1) *Contrastive Discovery: Open-Ended Scientific Discovery over Competing Explanations* (2026)
- [arXiv:2611.14513v1](http://arxiv.org/abs/2611.14513v1) *Online Graph Coloring: Competitive Ratios Separating Randomized and Deterministic Algorithms* (2026)
- [arXiv:2611.14511v1](http://arxiv.org/abs/2611.14511v1) *Pyrite: Building and Benchmarking Protein-Ligand Pose Ranking Agents* (2026)
- [arXiv:2611.14508v1](http://arxiv.org/abs/2611.14508v1) *When Does the Real Wiring Win? Testing connectome structure-function claims against unmeasured physiology* (2026)

*Retrieved from arXiv / OpenAlex and reviewed by ScholarLoop's LitReviewer agent. Every cited id is checked against the retrieved corpus.*
