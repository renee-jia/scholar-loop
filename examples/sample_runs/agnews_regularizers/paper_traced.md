# Which XGBoost Regularizers Transfer to a Small Bag-of-Embeddings Text Classifier? In One Small AG News Setting, Column Dropout and EMA Lowered Validation Error; Group-L1 and a Long Schedule Showed No Consistent Effect

> **Panel review:** reject · mean score 2.7/10 (harsh-but-fair 3 · harsh-critical 2 · open-minded 3)

## Abstract

We ask which of three XGBoost-derived regularizers lowers validation top-1 error, relative to a matched ablation, for a small bag-of-embeddings classifier trained on a 10k-article AG News subset: per-step vocabulary-column dropout, group-L1 on embedding rows, or shrinkage implemented as a low-lr long schedule with EMA weight averaging [scope_question]. In this setting, vocabulary-column dropout lowered validation error against the stock control on all three evaluated folds [claim_overall]. Of shrinkage's two parts, EMA weight averaging lowered error against its no-EMA ablation on every evaluated fold, and the low-lr long schedule showed no consistent effect [claim_overall]. Group-L1 on embedding rows showed no consistent effect [claim_overall]. On seed-fold pairs zero to two, the column-drop-only arm scored 12.0833 validation error against 13.7708 for the unregularized control, and column dropout was lower on every one of the three folds [claim_coldrop_vs_control]. At the default lr and epoch budget with column dropout, saving EMA-averaged weights gave a five-seed error of 12.025 against 12.55 without averaging, and EMA was lower on all five seed-fold pairs [claim_ema_default]. The unregularized control ran only on seed-fold pairs zero to two, so the column-dropout effect rests on three folds and no five-seed control exists [limit_1].

## Introduction

We study topic classification on AG News with a small bag-of-embeddings classifier trained on a 10k-article subset, and we ask which of three XGBoost-derived regularizers lowers validation top-1 error (lower is better) relative to a matched ablation: per-step vocabulary-column dropout, group-L1 on embedding rows, or shrinkage implemented as a low-lr long schedule with EMA weight averaging [scope_question].

These regularizers come from tree boosting: XGBoost is a scalable tree-boosting system whose regularized objective uses leaf-count and L2 leaf-weight penalties, alongside shrinkage and column subsampling, and it reports no text benchmarks [related_xgboost]. This study transfers column subsampling and shrinkage to a neural bag-of-embeddings model, adds group-L1 on embedding rows as its own L1-style analogue, and ablates each one separately [related_xgboost].

In this setting, vocabulary-column dropout lowered validation error against the stock control on all three evaluated folds, and group-L1 on embedding rows showed no consistent effect [claim_overall]. Of shrinkage's two parts, EMA weight averaging lowered error against its no-EMA ablation on every evaluated fold, while the low-lr long schedule showed no consistent effect [claim_overall].

Our first finding is that column dropout lowers error against the unregularized control: on seed-fold pairs zero to two, the column-drop-only arm scored 12.0833 validation error against 13.7708 for the control at the same configuration [claim_coldrop_vs_control]. Column dropout was lower on every one of the three folds [claim_coldrop_vs_control].

Our second finding isolates EMA within shrinkage: at the default lr and epoch budget with column dropout, saving EMA-averaged weights gave a five-seed error of 12.025 against 12.55 without averaging [claim_ema_default]. EMA was lower on all five seed-fold pairs [claim_ema_default].

Our third finding is a null result for group-L1: at the default schedule, adding weak group-L1 to column dropout left the five-seed mean unchanged at 12.55 with the penalty and 12.55 without it [claim_group_l1_default_null]. Per-seed values differed only at seed one and seed four [claim_group_l1_default_null].

## Related Work

XGBoost is a scalable tree-boosting system whose regularized objective penalizes leaf count and L2 leaf weights, and it also uses shrinkage and column subsampling [related_xgboost] [arXiv:1603.02754]. The XGBoost paper reports no text benchmarks [related_xgboost]. This study transfers column subsampling and shrinkage to a neural bag-of-embeddings model [related_xgboost]. It also adds group-L1 on embedding rows as its own L1-style analogue and ablates each mechanism separately [related_xgboost].

Work on connectome-constrained circuit models shows that comparisons are biased when baselines are under-tuned or inherit parameters tuned for the favoured model, and it names the machine-learning analogues [related_tuning_bias] [arXiv:2611.14508v1]. By analogy, the same caution applies here, because our control shares an untuned default configuration with the treatments [related_tuning_bias]. A related study cautions against relying on a single, unjustified task-specific baseline [related_single_baseline] [arXiv:2611.14519v1]. Our control is likewise one default-configuration model, with no tuned alternative regularizer in its place [related_single_baseline].

MobileNets treats width multipliers as a capacity-accuracy trade-off in vision [related_width] [arXiv:1704.04861]. This study varied width only inside confounded high-capacity bundles, so it does not isolate capacity [related_width].

## Setup

A frozen scorer defines the data, the hashed-token mean-of-embeddings model and validation top-1 error, and only the training script is edited [scope_method]. Each arm adds transferred mechanisms to the stock script [scope_method]. Most comparisons pair an arm with an ablation that removes one mechanism, at the same configuration and seed-fold pairs [scope_method]. Two comparisons lack this pairing: column dropout plus group-L1 against the stock control, and two column-drop configurations that differ in both drop rate and group-L1 [scope_method]. Runs proceed through smoke, verify and full tiers, and later tiers reuse earlier seeds [scope_method]. We report two high-capacity bundles without matched controls as uninterpretable [scope_method]. No arm was scored on the held-out test split [scope_method].

The data are a class-stratified AG News subset built from the Hugging Face copy fancyzhx/ag_news with sampling seed 0 [setup_data]. Of these articles, 8,000 from the training split form the cross-validation pool, and 2,000 from the official test split form a held-out test set [setup_data]. Text is lower-cased, split on non-alphanumerics and truncated to 64 tokens [setup_tokens]. Each token is hashed with crc32 modulo 20000, plus 1, into 20,000 buckets, with id 0 reserved for padding [setup_tokens].

The frozen model averages the non-padding token embeddings, applies depth ReLU hidden layers of width hidden, and outputs 4 class logits [setup_model]. Every arm trains with cross-entropy loss on the class logits, and group-L1, where present, is added to this loss [setup_loss].

The pool is cut into 5 stratified folds by StratifiedKFold with shuffling and random_state 0 [setup_folds]. A run with seed i trains on the remaining folds and is scored on validation fold i, so seeds and folds are paired [setup_folds]. The metric is top-1 error in percent on the validation fold, and lower is better [setup_metric]. No run scored the held-out test split [setup_metric].

The default configuration is lr 0.003, embed_dim 64, hidden 64, depth 1, weight_decay 0.0001 and 8 epochs [setup_default]. Training uses Adam with coupled weight decay, batch 128 and normal initialisation with std 0.1 [setup_default]. Smoke runs use one seed with a 120-second budget, verify runs use three seeds with 300 seconds, and full runs use five seeds with 900 seconds [setup_tiers]. Later tiers reuse the seeds of earlier tiers [setup_tiers]. Each run trains on four of the five folds of the pool with batch 128 for 8 or 16 epochs, so the configuration fixes the optimizer step count [setup_steps].

Thirty registered runs over twelve candidate configurations all completed with a recorded validation score, and none failed or were excluded [setup_runs]. Of these runs, twelve are smoke runs, twelve are verify runs and six are full-tier runs [setup_runs].

The control arm runs the unmodified training script at the default configuration, with no column dropout, group-L1 or averaging [setup_control]. Column dropout draws one Bernoulli keep mask over the vocabulary per step, shared across the batch, and maps dropped ids to padding id 0 [setup_mask]. A row that would be fully masked keeps its original ids [setup_mask]. The two high-capacity bundles lack this fallback [setup_mask].

The column-drop-only arm uses the same code with drop fraction p=0.4 and group-L1 coefficient 0 [setup_coldrop_only]. The column-drop plus group-L1 arm uses drop rate 0.25 at each step and adds group-L1 at 1e-5 on the sum of embedding-row L2 norms, excluding the padding row [setup_coldrop_l1]. The column-drop plus weak group-L1 arm uses drop fraction p=0.4 with group-L1 at 1e-6 [setup_coldrop_weak_l1].

The EMA arm adds weight averaging to the column-drop-only arm with decay min(0.999, (1+t)/(10+t)) at step t, and it saves the EMA weights [setup_ema]. The long-schedule EMA arm uses lr 0.0015, 16 epochs, drop fraction p=0.4 and group-L1 0 [setup_long_ema]. Its two implementations use EMA decay min(cap, (1+t)/(10+t)), the same warmup form as the default-schedule EMA arm, with cap 0.999 in one and 0.995 in the other, and both save the EMA weights [setup_long_ema]. The long-schedule no-EMA arm uses lr 0.0015, 16 epochs, drop fraction p=0.4 and group-L1 0, and it saves the raw final weights [setup_long_noema].

Two group-L1 arms keep the long-schedule EMA recipe (lr 0.0015, 16 epochs, p=0.4, warmup EMA capped at 0.999) [setup_long_l1]. One adds dense group-L1 at 1e-5 on all non-padding embedding rows [setup_long_l1]. The other adds lazy group-L1 at 1e-4 only on rows active in each batch [setup_long_l1].

High-capacity bundle A uses embed_dim 128, hidden 128, depth 2, lr 0.0015, weight_decay 5e-05 and 16 epochs [setup_highcap_a]. It adds column dropout p=0.3, group-L1 1e-5 and EMA at decay 0.999 without warmup [setup_highcap_a]. High-capacity bundle B uses the same configuration with column dropout p=0.25, group-L1 1e-5 and EMA decay 0.999 with warmup [setup_highcap_b].

## Results: Column Dropout

On seed-fold pairs zero to two, the column-drop-only arm scored 12.0833 validation error, with per-fold values of 12.1875, 11.875 and 12.1875 [claim_coldrop_vs_control]. The unregularized control at the same configuration scored 13.7708 (13.9375, 13.5 and 13.875 per fold), and its single-seed smoke run gave 13.9375 [claim_coldrop_vs_control]. Column dropout was therefore lower than the control on every one of the three folds [claim_coldrop_vs_control].

The column-drop plus group-L1 arm was also lower than the control on every fold of the same three seed-fold pairs [claim_coldrop_l1_vs_control]. This arm scored 12.3542 (13.0625, 12.0 and 12.0 per fold), and its smoke run gave 13.0625 [claim_coldrop_l1_vs_control]. The control it was compared against scored 13.7708 (13.9375, 13.5 and 13.875 per fold) [claim_coldrop_l1_vs_control].

Over five seed-fold pairs, the first two column-drop configurations, which differ in both drop rate and group-L1, showed no consistent ordering across seeds [claim_full_tier_coldrop]. The column-drop plus group-L1 arm scored 12.6875 (std 0.6673), with per-seed values of 13.0625, 12.0, 12.0, 12.875 and 13.5 [claim_full_tier_coldrop]. The column-drop-only arm scored 12.55 (std 0.7634), with per-seed values of 12.1875, 11.875, 12.1875, 12.6875 and 13.8125 [claim_full_tier_coldrop]. The column-drop plus weak group-L1 arm also scored 12.55, with a std of 0.724 [claim_full_tier_coldrop].

## Results: Shrinkage (EMA and Long Schedule)

At the default lr and epoch budget with column dropout, saving EMA-averaged weights gave a five-seed validation error of 12.025 (11.75, 11.125, 11.875, 11.75, 13.625), against 12.55 (12.1875, 11.875, 12.1875, 12.6875, 13.8125) without averaging [claim_ema_default]. EMA was lower on all five seed-fold pairs [claim_ema_default]. For the EMA arm, the verify tier gave 11.5833 and the smoke run 11.75 [claim_ema_default].

At the halved lr and doubled epoch budget with column dropout, EMA weights scored 11.8542 (11.9375, 11.625, 12.0; smoke 11.9375) [claim_ema_long]. The same schedule without EMA scored 12.125 (12.25, 11.6875, 12.4375; smoke 12.25) [claim_ema_long]. EMA was lower on each of the three folds [claim_ema_long].

The low-lr long schedule gave no consistent gain over the default schedule [claim_schedule_null]. With EMA, the long schedule scored 11.8875 over five seeds (11.9375, 11.625, 12.0, 10.8125, 13.0625), against 12.025 (11.75, 11.125, 11.875, 11.75, 13.625) for the default schedule [claim_schedule_null]. The long schedule was higher on seeds zero to two and lower on seeds three and four [claim_schedule_null]. Without EMA, the long schedule scored 12.125 (12.25, 11.6875, 12.4375) against 12.0833 (12.1875, 11.875, 12.1875), a mixed result [claim_schedule_null].

Under the warmup EMA, the averaging window grows with step count, so the with-EMA schedule comparison changes the averaging window and doubles the steps along with lr and epochs [limit_10]. The no-EMA pair, which compares the long schedule against the column-drop-only arm, is therefore the schedule-only contrast [limit_10].

## Results: Group-L1 on Embedding Rows

At the default schedule, adding weak group-L1 to column dropout left the five-seed mean validation error unchanged, at 12.55 with the penalty and 12.55 without it [claim_group_l1_default_null]. Per-seed values differed only at seed one (11.9375 with the penalty vs 11.875 without) and at seed four (13.75 vs 13.8125) [claim_group_l1_default_null]. The verify tier gave 12.1042 with the penalty and 12.0833 without it [claim_group_l1_default_null]. The smoke runs of both arms gave the same error, 12.1875 [claim_group_l1_default_null].

On top of the long-schedule EMA recipe, which scored 11.8542 with per-fold values of 11.9375, 11.625 and 12.0, dense group-L1 scored 11.875 (12.0, 11.6875 and 11.9375; smoke 12.0) [claim_group_l1_long_null]. Lazy group-L1 on batch-active rows, run at a larger coefficient, scored 11.9375 (12.1875, 11.75 and 11.875; smoke 12.1875) [claim_group_l1_long_null]. Neither penalty lowered error consistently across the three folds [claim_group_l1_long_null].

The group-L1 null may reflect coefficients that were too small relative to the coupled Adam weight decay, or Adam rescaling the penalty [limit_12]. Neither possibility was checked, so the null holds only for the coefficients we tried [limit_12].

## Results: Variance, Fold Difficulty and Uninterpretable Bundles

In every five-seed arm, the seed-four/fold-four pair had the highest validation error, at 13.5, 13.8125, 13.75, 13.625 and 13.0625 [claim_fold_four]. Each five-seed std (0.6673, 0.7634, 0.724, 0.9412, 0.8094) exceeds the three-seed std of the same arm on seeds zero to two [claim_fold_four].

Two bundles that doubled width, used depth two and combined all three mechanisms scored above both default-capacity column-drop arms on every one of folds zero to two [claim_highcap_bundles]. Bundle A, which averaged without warmup, scored 14.5417 (16.1875, 13.3125, 14.125; smoke 16.1875) [claim_highcap_bundles]. Bundle B, which used warmup, scored 13.3125 (13.9375, 12.5625, 13.4375; smoke 13.9375) [claim_highcap_bundles]. On the same folds, the column-drop plus group-L1 arm scored 12.3542 (13.0625, 12.0, 12.0) [claim_highcap_bundles]. The column-drop-only arm scored 12.0833 (12.1875, 11.875, 12.1875) [claim_highcap_bundles].

The high-capacity bundles confound capacity, lr, epochs, drop rate and EMA variant, and no capacity-matched control ran, so neither result can be attributed to any mechanism [limit_13]. One bundle averaged from initialisation without warmup, which offers a competing explanation for its high error [limit_13].

## Discussion: What Governs the EMA Gain

Two implementations of the long-schedule EMA recipe produced identical per-seed errors, at 11.8542 on the verify seeds in both and 11.8875 on the five-seed mean in both [claim_ema_cap_identity]. The two implementations differ only in the EMA decay cap [claim_ema_cap_identity].

The decay cap never binds in these runs [claim_ema_warmup_mechanism]. The step count is fixed by the recorded pool size, the five-fold split, the batch size and the epoch budget [claim_ema_warmup_mechanism]. At that step count, the warmup term of the decay schedule stays below either cap for the whole run [claim_ema_warmup_mechanism]. The averaging that helped is therefore governed by the warmup schedule, whose effective window is shorter than the nominal cap implies [claim_ema_warmup_mechanism].

We compared EMA effects across separate runs, not as raw versus averaged weights from the same run [limit_9]. Step counts were not logged, so the claim that the decay cap never binds is derived from the recorded fold size, batch size and epochs, not measured [limit_9].

We took no mechanism measurements: no train/validation gap, no learning curves, no per-token reliance and no embedding-row norms [limit_11]. Two hypotheses therefore remain untested: that the gains are early stopping in disguise, and that dropout prevents reliance on topic tokens [limit_11].

## Limitations

The unregularized control ran only on seed-fold pairs zero to two [limit_1]. The column-dropout effect therefore rests on three folds, and no five-seed control exists [limit_1]. Seed index and validation fold are coupled, so seed variance mixes initialisation noise with fold difficulty [limit_2]. Fold four is consistently hardest [limit_2].

The verify and full tiers reuse the seeds of earlier tiers [limit_3]. These tiers are nested subsets, not independent replicates [limit_3]. Each comparison is a sign pattern over three or five folds [limit_4]. No significance test or confidence interval was computed [limit_4].

Per-fold errors move in steps of one validation article [limit_5]. One of the three long-schedule EMA margins, at fold one, is a single article, so that comparison is weak on its own [limit_5]. The five-fold default-schedule EMA comparison carries most of the weight [limit_5]. All arms were selected and compared on validation folds [limit_6]. The held-out test split was never scored, so selection bias toward validation noise is possible [limit_6].

The planned comparison against hidden dropout plus weight decay, tuned with an equal trial budget, never ran [limit_7]. Column dropout may therefore act as generic regularization rather than as a word-specific effect [limit_7]. The control used the default configuration and was not tuned for its own arm [limit_7]. The two default-schedule column-drop configurations change drop rate and group-L1 together [limit_8]. Dose-response in the drop rate was not measured [limit_8].

The logistic-regression baseline quoted in the profile was never run under this protocol and is not directly comparable with any arm [limit_14]. Results cover one dataset subset, one small hashed-vocabulary model and short training budgets [limit_15]. Historical code provenance relies on recorded candidate source versions, not execution snapshots [limit_15].

## Conclusion

We asked which of three XGBoost-derived regularizers lowers validation top-1 error for a small bag-of-embeddings classifier, and in this setting vocabulary-column dropout lowered error against the stock control on all three evaluated folds [claim_overall]. Of shrinkage's two parts, EMA weight averaging lowered error against its no-EMA ablation on every evaluated fold, while the low-lr long schedule and group-L1 on embedding rows showed no consistent effect [claim_overall].

The strongest evidence is the five-seed EMA comparison at the default lr and epoch budget with column dropout, where EMA-averaged weights gave an error of 12.025 against 12.55 without averaging [claim_ema_default]. EMA was lower on all five seed-fold pairs [claim_ema_default].

The column-dropout result carries a clear limit: the unregularized control ran only on seed-fold pairs zero to two, so the effect rests on three folds and no five-seed control exists [limit_1].

The next step is the planned comparison against hidden dropout plus weight decay, tuned with an equal trial budget, which never ran [limit_7]. That comparison would test whether column dropout acts as generic regularization rather than a word-specific effect, since our control used the default configuration and was not tuned for its own arm [limit_7].

---

### Drafting notes (for the author)

- **Abstract**: 1 sentence(s) over 40 words (first: "We ask which of three XGBoost-derived regularizers lowers va…")
- **Introduction**: 1 sentence(s) over 40 words (first: "We study topic classification on AG News with a small bag-of…")

*Written under a version-bound scientific review and author release. Automated checks and semantic review can still miss errors; inspect the [evidence bindings](evidence_map.md).*
