# Key points — Which XGBoost Regularizers Transfer to a Small Bag-of-Embeddings Text Classifier? In One Small AG News Setting, Column Dropout and EMA Lowered Validation Error; Group-L1 and a Long Schedule Showed No Consistent Effect

_Question: For a small bag-of-embeddings classifier trained on a 10k-article AG News subset, which of three XGBoost-derived regularizers lowers validation top-1 error relative to a matched ablation: per-step vocabulary-column dropout, group-L1 on embedding rows, or shrinkage implemented as a low-lr long schedule with EMA weight averaging?_

## Abstract

1. *context* — For a small bag-of-embeddings classifier trained on a 10k-article AG News subset, which of three XGBoost-derived regularizers lowers validation top-1 error relative to a matched ablation: per-step vocabulary-column dropout, group-L1 on embedding rows, or shrinkage implemented as a low-lr long schedule with EMA weight averaging?  
   evidence: `scope_question`
2. *result* — In this setting, vocabulary-column dropout lowered validation error against the stock control on all three evaluated folds. Of shrinkage's two parts, EMA weight averaging lowered error against its no-EMA ablation on every evaluated fold, while the low-lr long schedule showed no consistent effect. Group-L1 on embedding rows showed no consistent effect.  
   evidence: `claim_overall`
3. *result* — On seed-fold pairs zero to two, the column-drop-only arm scored 12.0833 validation error (per fold 12.1875, 11.875, 12.1875). The unregularized control at the same configuration scored 13.7708 (13.9375, 13.5, 13.875), and its single-seed smoke run gave 13.9375. Column dropout was lower on every one of the three folds.  
   evidence: `claim_coldrop_vs_control`
4. *result* — At the default lr and epoch budget with column dropout, saving EMA-averaged weights gave a five-seed error of 12.025 (11.75, 11.125, 11.875, 11.75, 13.625), against 12.55 (12.1875, 11.875, 12.1875, 12.6875, 13.8125) without averaging. EMA was lower on all five seed-fold pairs. The verify tier gave 11.5833 and the smoke run 11.75.  
   evidence: `claim_ema_default`
5. *limitation* — The unregularized control ran only on seed-fold pairs zero to two. The column-dropout effect therefore rests on three folds, and no five-seed control exists.  
   evidence: `limit_1`

## Introduction

1. *context* — For a small bag-of-embeddings classifier trained on a 10k-article AG News subset, which of three XGBoost-derived regularizers lowers validation top-1 error relative to a matched ablation: per-step vocabulary-column dropout, group-L1 on embedding rows, or shrinkage implemented as a low-lr long schedule with EMA weight averaging?  
   evidence: `scope_question`
2. *challenge* — XGBoost is a scalable tree-boosting system whose regularized objective uses leaf-count and L2 leaf-weight penalties, alongside shrinkage and column subsampling; it reports no text benchmarks. This study transfers column subsampling and shrinkage to a neural bag-of-embeddings model and adds group-L1 on embedding rows as its own L1-style analogue, ablating each separately.  
   evidence: `related_xgboost`
3. *result* — In this setting, vocabulary-column dropout lowered validation error against the stock control on all three evaluated folds. Of shrinkage's two parts, EMA weight averaging lowered error against its no-EMA ablation on every evaluated fold, while the low-lr long schedule showed no consistent effect. Group-L1 on embedding rows showed no consistent effect.  
   evidence: `claim_overall`
4. *result* — On seed-fold pairs zero to two, the column-drop-only arm scored 12.0833 validation error (per fold 12.1875, 11.875, 12.1875). The unregularized control at the same configuration scored 13.7708 (13.9375, 13.5, 13.875), and its single-seed smoke run gave 13.9375. Column dropout was lower on every one of the three folds.  
   evidence: `claim_coldrop_vs_control`
5. *result* — At the default lr and epoch budget with column dropout, saving EMA-averaged weights gave a five-seed error of 12.025 (11.75, 11.125, 11.875, 11.75, 13.625), against 12.55 (12.1875, 11.875, 12.1875, 12.6875, 13.8125) without averaging. EMA was lower on all five seed-fold pairs. The verify tier gave 11.5833 and the smoke run 11.75.  
   evidence: `claim_ema_default`
6. *result* — At the default schedule, adding weak group-L1 to column dropout left the five-seed mean unchanged: 12.55 with the penalty and 12.55 without it. Per-seed values differed only at seed one (11.9375 vs 11.875) and seed four (13.75 vs 13.8125). The verify tier gave 12.1042 with the penalty and 12.0833 without it. The smoke runs of both arms gave 12.1875.  
   evidence: `claim_group_l1_default_null`

## Related Work

1. *comparison* — XGBoost is a scalable tree-boosting system whose regularized objective uses leaf-count and L2 leaf-weight penalties, alongside shrinkage and column subsampling; it reports no text benchmarks. This study transfers column subsampling and shrinkage to a neural bag-of-embeddings model and adds group-L1 on embedding rows as its own L1-style analogue, ablating each separately.  
   evidence: `related_xgboost`
2. *comparison* — That work demonstrates, in connectome-constrained circuit models, that comparisons are biased when baselines are under-tuned or inherit parameters tuned for the favoured model, and names the machine-learning analogues. By analogy, the same caution applies here, because the control shares an untuned default configuration with the treatments.  
   evidence: `related_tuning_bias`
3. *comparison* — That study cautions against relying on a single, unjustified task-specific baseline. The control here is likewise one default-configuration model and not a tuned alternative regularizer.  
   evidence: `related_single_baseline`
4. *comparison* — That paper treats width multipliers as a capacity-accuracy trade-off in vision. This study varied width only inside confounded high-capacity bundles and does not isolate capacity.  
   evidence: `related_width`

## Setup

1. *method* — A frozen scorer defines the data, the hashed-token mean-of-embeddings model and validation top-1 error; only the training script is edited. Arms add transferred mechanisms to the stock script. Most comparisons pair an arm with an ablation that removes one mechanism, at the same configuration and seed-fold pairs. Two comparisons do not: column dropout plus group-L1 against the stock control, and two column-drop configurations that differ in both drop rate and group-L1. Runs proceed through smoke, verify and full tiers, and later tiers reuse earlier seeds. Two high-capacity bundles without matched controls are reported as uninterpretable. No arm was scored on the held-out test split.  
   evidence: `scope_method`
2. *method* — Data are a class-stratified AG News subset built from the Hugging Face copy fancyzhx/ag_news with sampling seed 0: 8,000 training-split articles form the cross-validation pool, and 2,000 official-test articles form a held-out test set.  
   evidence: `setup_data`
3. *method* — Text is lower-cased, split on non-alphanumerics and truncated to 64 tokens. Each token is hashed with crc32 modulo 20000, plus 1, into 20,000 buckets, with id 0 as padding.  
   evidence: `setup_tokens`
4. *method* — The frozen model averages non-padding token embeddings, applies depth ReLU hidden layers of width hidden, and outputs 4 class logits.  
   evidence: `setup_model`
5. *method* — Every arm trains with cross-entropy loss on the class logits; group-L1, where present, is added to this loss.  
   evidence: `setup_loss`
6. *method* — The pool is cut into 5 stratified folds by StratifiedKFold with shuffling and random_state 0. A run with seed i trains on the remaining folds and is scored on validation fold i, so seeds and folds are paired.  
   evidence: `setup_folds`
7. *method* — The metric is top-1 error in percent on the validation fold, and lower is better. No run scored the held-out test split.  
   evidence: `setup_metric`
8. *method* — The default configuration is lr 0.003, embed_dim 64, hidden 64, depth 1, weight_decay 0.0001 and 8 epochs. It uses Adam with coupled weight decay, batch 128 and normal initialisation with std 0.1.  
   evidence: `setup_default`
9. *method* — Smoke runs use one seed with a 120-second budget, verify runs three seeds with 300 seconds, and full runs five seeds with 900 seconds. Later tiers reuse earlier seeds.  
   evidence: `setup_tiers`
10. *method* — Each run trains on four of the five folds of the pool with batch 128 for 8 or 16 epochs, so the optimizer step count is fixed by the configuration.  
   evidence: `setup_steps`
11. *method* — Thirty registered runs over twelve candidate configurations all completed with a recorded validation score; none failed or were excluded. Twelve are smoke runs, twelve are verify runs and six are full-tier runs.  
   evidence: `setup_runs`
12. *method* — The control arm runs the unmodified training script at the default configuration, with no column dropout, group-L1 or averaging.  
   evidence: `setup_control`
13. *method* — Column dropout draws one Bernoulli keep mask over the vocabulary per step, shared across the batch, and maps dropped ids to padding id 0. A row that would be fully masked keeps its original ids. The two high-capacity bundles lack this fallback.  
   evidence: `setup_mask`
14. *method* — The column-drop-only arm uses the same code with drop fraction p=0.4 and group-L1 coefficient 0.  
   evidence: `setup_coldrop_only`
15. *method* — The column-drop plus group-L1 arm uses drop rate 0.25 at each step and adds group-L1 at 1e-5 on the sum of embedding-row L2 norms, excluding the padding row.  
   evidence: `setup_coldrop_l1`
16. *method* — The column-drop plus weak group-L1 arm uses drop fraction p=0.4 with group-L1 at 1e-6.  
   evidence: `setup_coldrop_weak_l1`
17. *method* — The EMA arm adds weight averaging to the column-drop-only arm with decay min(0.999, (1+t)/(10+t)) at step t and saves the EMA weights.  
   evidence: `setup_ema`
18. *method* — The long-schedule EMA arm uses lr 0.0015, 16 epochs, drop fraction p=0.4 and group-L1 0. Both implementations use EMA decay min(cap, (1+t)/(10+t)), the same warmup form as the default-schedule EMA arm, with cap 0.999 in one and 0.995 in the other, and save the EMA weights.  
   evidence: `setup_long_ema`
19. *method* — The long-schedule no-EMA arm uses lr 0.0015, 16 epochs, drop fraction p=0.4 and group-L1 0, and saves raw final weights.  
   evidence: `setup_long_noema`
20. *method* — Two group-L1 arms keep the long-schedule EMA recipe (lr 0.0015, 16 epochs, p=0.4, warmup EMA capped at 0.999). One adds dense group-L1 at 1e-5 on all non-padding embedding rows. The other adds lazy group-L1 at 1e-4 only on rows active in each batch.  
   evidence: `setup_long_l1`
21. *method* — High-capacity bundle A uses embed_dim 128, hidden 128, depth 2, lr 0.0015, weight_decay 5e-05 and 16 epochs. It adds column dropout p=0.3, group-L1 1e-5 and EMA at decay 0.999 without warmup.  
   evidence: `setup_highcap_a`
22. *method* — High-capacity bundle B uses the same configuration with column dropout p=0.25, group-L1 1e-5 and EMA decay 0.999 with warmup.  
   evidence: `setup_highcap_b`

## Results: Column Dropout

1. *result* — On seed-fold pairs zero to two, the column-drop-only arm scored 12.0833 validation error (per fold 12.1875, 11.875, 12.1875). The unregularized control at the same configuration scored 13.7708 (13.9375, 13.5, 13.875), and its single-seed smoke run gave 13.9375. Column dropout was lower on every one of the three folds.  
   evidence: `claim_coldrop_vs_control`
2. *comparison* — On the same three seed-fold pairs, the column-drop plus group-L1 arm scored 12.3542 (per fold 13.0625, 12.0, 12.0), and its smoke run gave 13.0625. The control scored 13.7708 (13.9375, 13.5, 13.875), so this arm was also lower on every fold.  
   evidence: `claim_coldrop_l1_vs_control`
3. *result* — Over five seed-fold pairs, the column-drop plus group-L1 arm scored 12.6875 (std 0.6673; per seed 13.0625, 12.0, 12.0, 12.875, 13.5). The column-drop-only arm scored 12.55 (std 0.7634; 12.1875, 11.875, 12.1875, 12.6875, 13.8125). The column-drop plus weak group-L1 arm scored 12.55 (std 0.724). The first two column-drop configurations, which differ in both drop rate and group-L1, showed no consistent ordering across seeds.  
   evidence: `claim_full_tier_coldrop`

## Results: Shrinkage (EMA and Long Schedule)

1. *result* — At the default lr and epoch budget with column dropout, saving EMA-averaged weights gave a five-seed error of 12.025 (11.75, 11.125, 11.875, 11.75, 13.625), against 12.55 (12.1875, 11.875, 12.1875, 12.6875, 13.8125) without averaging. EMA was lower on all five seed-fold pairs. The verify tier gave 11.5833 and the smoke run 11.75.  
   evidence: `claim_ema_default`
2. *result* — At the halved lr and doubled epoch budget with column dropout, EMA weights scored 11.8542 (11.9375, 11.625, 12.0; smoke 11.9375). The same schedule without EMA scored 12.125 (12.25, 11.6875, 12.4375; smoke 12.25). EMA was lower on each of the three folds.  
   evidence: `claim_ema_long`
3. *comparison* — The low-lr long schedule gave no consistent gain over the default schedule. With EMA, it scored 11.8875 over five seeds (11.9375, 11.625, 12.0, 10.8125, 13.0625) against 12.025 (11.75, 11.125, 11.875, 11.75, 13.625). It was higher on seeds zero to two and lower on seeds three and four. Without EMA, it scored 12.125 (12.25, 11.6875, 12.4375) against 12.0833 (12.1875, 11.875, 12.1875), which is mixed.  
   evidence: `claim_schedule_null`
4. *limitation* — Under the warmup EMA, the averaging window grows with step count, so the with-EMA schedule comparison changes the averaging window and doubles the steps along with lr and epochs. The no-EMA pair, long schedule against the column-drop-only arm, is the schedule-only contrast.  
   evidence: `limit_10`

## Results: Group-L1 on Embedding Rows

1. *result* — At the default schedule, adding weak group-L1 to column dropout left the five-seed mean unchanged: 12.55 with the penalty and 12.55 without it. Per-seed values differed only at seed one (11.9375 vs 11.875) and seed four (13.75 vs 13.8125). The verify tier gave 12.1042 with the penalty and 12.0833 without it. The smoke runs of both arms gave 12.1875.  
   evidence: `claim_group_l1_default_null`
2. *result* — On top of the long-schedule EMA recipe (11.8542; per fold 11.9375, 11.625, 12.0), dense group-L1 scored 11.875 (12.0, 11.6875, 11.9375; smoke 12.0). Lazy group-L1 on batch-active rows at a larger coefficient scored 11.9375 (12.1875, 11.75, 11.875; smoke 12.1875). Neither penalty lowered error consistently across the three folds.  
   evidence: `claim_group_l1_long_null`
3. *limitation* — The group-L1 null may reflect coefficients too small relative to the coupled Adam weight decay, or Adam rescaling the penalty. Neither possibility was checked.  
   evidence: `limit_12`

## Results: Variance, Fold Difficulty and Uninterpretable Bundles

1. *result* — In every five-seed arm, the seed-four/fold-four pair had the highest error: 13.5, 13.8125, 13.75, 13.625 and 13.0625. Each five-seed std (0.6673, 0.7634, 0.724, 0.9412, 0.8094) exceeds the three-seed std of the same arm on seeds zero to two.  
   evidence: `claim_fold_four`
2. *result* — Two bundles that doubled width, used depth two and combined all three mechanisms scored above both default-capacity column-drop arms on every one of folds zero to two. Bundle A, averaging without warmup, scored 14.5417 (16.1875, 13.3125, 14.125; smoke 16.1875). Bundle B, with warmup, scored 13.3125 (13.9375, 12.5625, 13.4375; smoke 13.9375). The column-drop plus group-L1 arm scored 12.3542 (13.0625, 12.0, 12.0) and the column-drop-only arm 12.0833 (12.1875, 11.875, 12.1875).  
   evidence: `claim_highcap_bundles`
3. *limitation* — The high-capacity bundles confound capacity, lr, epochs, drop rate and EMA variant, and no capacity-matched control ran, so neither result can be attributed to any mechanism. One averaged from initialisation without warmup, a competing explanation for its high error.  
   evidence: `limit_13`

## Discussion: What Governs the EMA Gain

1. *result* — Two implementations of the long-schedule EMA recipe produced identical per-seed errors: verify 11.8542 in both, and five-seed 11.8875 in both. They differ only in the EMA decay cap.  
   evidence: `claim_ema_cap_identity`
2. *result* — The cap never binds in these runs. The step count is fixed by the recorded pool size, the five-fold split, the batch size and the epoch budget, and at that step count the warmup term of the decay schedule stays below either cap for the whole run. The averaging that helped is therefore governed by the warmup schedule, whose effective window is shorter than the nominal cap implies.  
   evidence: `claim_ema_warmup_mechanism`
3. *limitation* — EMA effects are compared across separate runs, not as raw versus averaged weights from the same run. Step counts were not logged; the claim that the decay cap never binds is derived from the recorded fold size, batch size and epochs, not measured.  
   evidence: `limit_9`
4. *limitation* — No mechanism measurements were taken: no train/validation gap, learning curves, per-token reliance or embedding-row norms. Two hypotheses remain untested: that gains are early stopping in disguise, and that dropout prevents reliance on topic tokens.  
   evidence: `limit_11`

## Limitations

1. *limitation* — The unregularized control ran only on seed-fold pairs zero to two. The column-dropout effect therefore rests on three folds, and no five-seed control exists.  
   evidence: `limit_1`
2. *limitation* — Seed index and validation fold are coupled, so seed variance mixes initialisation noise with fold difficulty. Fold four is consistently hardest.  
   evidence: `limit_2`
3. *limitation* — Verify and full tiers reuse the seeds of earlier tiers. They are nested subsets, not independent replicates.  
   evidence: `limit_3`
4. *limitation* — Each comparison is a sign pattern over three or five folds. No significance test or confidence interval was computed.  
   evidence: `limit_4`
5. *limitation* — Per-fold errors move in steps of one validation article. One of the three long-schedule EMA margins, at fold one, is a single article, so that comparison is weak on its own; the five-fold default-schedule EMA comparison carries most of the weight.  
   evidence: `limit_5`
6. *limitation* — All arms were selected and compared on validation folds. The held-out test split was never scored, so selection bias toward validation noise is possible.  
   evidence: `limit_6`
7. *limitation* — The planned comparison against hidden dropout plus weight decay, tuned with an equal trial budget, never ran. Column dropout may act as generic regularization rather than a word-specific effect. The control used the default configuration and was not tuned for its own arm.  
   evidence: `limit_7`
8. *limitation* — The two default-schedule column-drop configurations change drop rate and group-L1 together. Dose-response in the drop rate was not measured.  
   evidence: `limit_8`
9. *limitation* — The logistic-regression baseline quoted in the profile was never run under this protocol and is not directly comparable with any arm.  
   evidence: `limit_14`
10. *limitation* — Results cover one dataset subset, one small hashed-vocabulary model and short training budgets. Historical code provenance relies on recorded candidate source versions, not execution snapshots.  
   evidence: `limit_15`

## Conclusion

1. *result* — In this setting, vocabulary-column dropout lowered validation error against the stock control on all three evaluated folds. Of shrinkage's two parts, EMA weight averaging lowered error against its no-EMA ablation on every evaluated fold, while the low-lr long schedule showed no consistent effect. Group-L1 on embedding rows showed no consistent effect.  
   evidence: `claim_overall`
2. *result* — At the default lr and epoch budget with column dropout, saving EMA-averaged weights gave a five-seed error of 12.025 (11.75, 11.125, 11.875, 11.75, 13.625), against 12.55 (12.1875, 11.875, 12.1875, 12.6875, 13.8125) without averaging. EMA was lower on all five seed-fold pairs. The verify tier gave 11.5833 and the smoke run 11.75.  
   evidence: `claim_ema_default`
3. *limitation* — The unregularized control ran only on seed-fold pairs zero to two. The column-dropout effect therefore rests on three folds, and no five-seed control exists.  
   evidence: `limit_1`
4. *future_work* — The planned comparison against hidden dropout plus weight decay, tuned with an equal trial budget, never ran. Column dropout may act as generic regularization rather than a word-specific effect. The control used the default configuration and was not tuned for its own arm.  
   evidence: `limit_7`

## Discussion

### Round 1 — ✅ no blocking issue left
- **Skeptic**: accept
  - (suggestion) [Results: Shrinkage — claim_ema_long (and claim_overall's 'every evaluated fold')] limit_5 says the fold-one long-schedule EMA margin is a single validation article. It also says the five-fold default-schedule comparison carries most of the weight. That caveat sits only in the general Limitations section, far from claim_ema_long. A reader of the Shrinkage results sees 'lower on each of the three folds' with no qualification. → *Place limit_5 directly after claim_ema_long in the Shrinkage results, or duplicate it there. Keep claim_ema_default as the primary EMA evidence, as limit_5 states.*
  - (suggestion) [Abstract — column-dropout result] The abstract gives the column-dropout gain against an untuned, default-configuration control, qualified only by limit_1. limit_7 is the caveat that bounds this claim most: the control was untuned, the tuned hidden-dropout plus weight-decay comparison never ran, and the effect may be generic regularization. It first appears in the Conclusion. The statements are scoped ('against the stock control'), so this is not overreach, but the abstract undersells the main threat. → *Swap limit_1 in the abstract for limit_7, or add limit_7 alongside it. Alternatively, add related_tuning_bias or related_single_baseline near the column-dropout result in the Introduction.*
  - (suggestion) [Results: Variance, Fold Difficulty — claim_fold_four] limit_2 (seed and fold are coupled, so seed variance mixes initialisation noise with fold difficulty) is the interpretive frame for claim_fold_four. It appears only in Limitations. → *Place limit_2 immediately after claim_fold_four. limit_3 (nested tiers, not independent replicates) could also go there.*
  - (suggestion) [Abstract / Introduction repetition] The Introduction repeats the Abstract verbatim: scope_question, claim_overall, claim_coldrop_vs_control and claim_ema_default all appear in both. claim_overall and claim_ema_default also reappear in the Conclusion. The five-sentence abstract also stacks three result statements that overlap in content. → *In the abstract, keep claim_overall plus one supporting number (claim_ema_default). In the Introduction, drop the duplicated numeric statements, or keep only the group-L1 null to motivate the sections. Let the results sections carry the numbers.*
  - (suggestion) [Results: Column Dropout — claim_coldrop_l1_vs_control] This comparison changes drop rate and adds group-L1 at once, so it is not a matched ablation (scope_method says so). It is placed without an adjacent caveat, and claim_full_tier_coldrop's confound is described only inside that statement. → *Add limit_8 (drop rate and group-L1 changed together; no dose-response) at the end of the Column Dropout results section.*
- **Editor**: accept
  - (suggestion) [Abstract: claim_coldrop_vs_control and claim_ema_default] The abstract repeats the full per-fold numbers of two results right after claim_overall states the nugget. That makes it long and hides the one-sentence answer. The same three statements then appear again in the Introduction. → *Keep claim_overall and limit_1 in the abstract and choose at most one numeric statement, preferably claim_ema_default, the five-fold comparison. Leave the per-fold detail to the Results.*
  - (suggestion) [Introduction: result bullets (claim_overall, claim_coldrop_vs_control, claim_ema_default, claim_group_l1_default_null)] The Introduction restates three numeric results already given in the Abstract and later in Results. It reads as a second results section, not a motivation leading to contributions. → *After scope_question and related_xgboost, keep only claim_overall as the contribution summary. Drop the numeric results from the Introduction.*
  - (suggestion) [Introduction / Related Work: related_xgboost] The same statement appears verbatim in both sections. → *Keep it in the Introduction as the motivation for the transfer. In Related Work, either drop it or open with a transition to the baseline-tuning cautions.*
  - (suggestion) [Conclusion: claim_ema_default, limit_1, limit_7] The Conclusion repeats statements that already appear twice or three times: claim_ema_default four times in total, limit_1 three times, and limit_7 twice. The label future_work also fits limit_7 poorly, since that statement describes an experiment that did not run. → *Close with claim_overall plus limit_7 only. Drop the repeated claim_ema_default and limit_1, or keep one of them.*
  - (suggestion) [Limitations: limit_5] The caveat that the fold-one long-schedule EMA margin is a single article sits far from claim_ema_long, the result it qualifies. → *Place limit_5 directly after claim_ema_long in the Shrinkage results section.*
  - (suggestion) [Limitations: limit_8 and limit_2] limit_8 qualifies claim_full_tier_coldrop (drop rate and group-L1 confounded), and limit_2 qualifies claim_fold_four (seed and fold coupling). Both appear only in the general Limitations list, away from the results they qualify. → *Move limit_8 after claim_full_tier_coldrop and limit_2 after claim_fold_four, or cross-reference them there.*
  - (suggestion) [Results: Variance, Fold Difficulty and Uninterpretable Bundles] This section joins two unrelated topics. The high-capacity bundles do not answer the question; they are reported as uninterpretable. → *Split fold difficulty into its own short subsection, or fold it into the setup and limitations material. Put the bundles (claim_highcap_bundles plus limit_13) last, or in an appendix-style subsection with a one-line transition saying why they are reported.*
  - (suggestion) [Setup: arm definitions (setup_control through setup_highcap_b)] Eleven per-arm setup bullets make it hard for a reader to see which pairs form the matched ablations that scope_method describes. → *Order the arm bullets by comparison: control and column dropout, then EMA and the long schedule, then group-L1, then the bundles. Plan a table that lists each comparison pair.*
  - (suggestion) [Results: Column Dropout, claim_coldrop_l1_vs_control] This comparison is not matched: it changes drop rate and adds group-L1 against the control, as scope_method notes. Nothing beside it flags that. → *Add a transition noting that this is one of the two unmatched comparisons named in scope_method.*
- **Methodologist**: accept
  - (suggestion) [Abstract, limitation slot] The abstract names only limit_1 as a caveat. The column-dropout headline is measured against one untuned default control, and no tuned alternative regularizer was run (limit_7, related_single_baseline). That is the main threat to the strongest result. → *Add limit_7 to the abstract next to limit_1, or replace limit_1 with it, so readers see that no tuned baseline exists before the Limitations section.*
  - (suggestion) [Introduction, scope_question and result repetition] The Introduction repeats the abstract almost exactly: scope_question, claim_overall, claim_coldrop_vs_control and claim_ema_default all appear again. claim_overall and claim_ema_default reappear a third time in the Conclusion. → *Keep claim_overall as the single summary in the Introduction. Leave the per-fold number statements to the Results sections.*
  - (suggestion) [Setup, setup_tiers vs setup_steps] Tiers have wall-clock budgets of 120, 300 and 900 seconds. setup_steps says the step count is fixed by the configuration. No statement says whether any run hit its time budget before finishing its epochs. A reader cannot tell whether the budget ever truncated training. → *Place setup_steps directly after setup_tiers so the fixed step count is read as overriding the time cap. If the author recorded that no run was truncated, add that as a statement.*
  - (suggestion) [Limitations, limit_14] limit_14 refers to 'the logistic-regression baseline quoted in the profile', but no plan statement ever introduces that baseline. The limitation will read as unanchored. → *Either omit limit_14 or place it next to related_single_baseline / setup_control, where the baseline question is raised.*
  - (suggestion) [Results: Column Dropout, claim_coldrop_l1_vs_control] This comparison changes both drop rate and group-L1 against the control. scope_method notes this, but the Results section does not repeat it, so it could be read as a clean column-dropout effect. → *Place limit_8 directly after claim_full_tier_coldrop in this section, rather than only in Limitations.*
  - (suggestion) [Conclusion, limit_7 labelled future_work] limit_7 states what did not run. Labelling it future_work is acceptable, but it duplicates the Limitations entry word for word. → *Keep it in only one place, or use it once in the Conclusion as the main open question.*

*Every evidence id is a real run, paper, or the baseline; every number was checked against the registry. Approve to draft the prose.*
