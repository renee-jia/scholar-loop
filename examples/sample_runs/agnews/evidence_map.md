# Approved statement bindings

Writing release: `3cbc399bf5f45c81abd8873787591abdd58b5d160c0b8b4f835d5be4684e14e1`

This snapshot records the approved scope. Revalidate the live review before further writing.

Raw sources, measurements and input hashes: [approved_evidence.json](approved_evidence.json).

## claim_coldrop_vs_control

On seed-fold pairs zero to two, the column-drop-only arm scored 12.0833 validation error (per fold 12.1875, 11.875, 12.1875). The unregularized control at the same configuration scored 13.7708 (13.9375, 13.5, 13.875), and its single-seed smoke run gave 13.9375. Column dropout was lower on every one of the three folds.

Evidence: exp_0013, exp_0015, exp_0008.

- `exp_0013::val_top1_err.verify`: value=12.0833, std=0.1804; owning evidence `exp_0013`.
- `exp_0015::val_top1_err.verify`: value=13.7708, std=0.2366; owning evidence `exp_0015`.
- `exp_0008::val_top1_err.smoke`: value=13.9375, std=0.0; owning evidence `exp_0008`.
## claim_coldrop_l1_vs_control

On the same three seed-fold pairs, the column-drop plus group-L1 arm scored 12.3542 (per fold 13.0625, 12.0, 12.0), and its smoke run gave 13.0625. The control scored 13.7708 (13.9375, 13.5, 13.875), so this arm was also lower on every fold.

Evidence: exp_0004, exp_0001, exp_0015.

- `exp_0004::val_top1_err.verify`: value=12.3542, std=0.6134; owning evidence `exp_0004`.
- `exp_0001::val_top1_err.smoke`: value=13.0625, std=0.0; owning evidence `exp_0001`.
- `exp_0015::val_top1_err.verify`: value=13.7708, std=0.2366; owning evidence `exp_0015`.
## claim_full_tier_coldrop

Over five seed-fold pairs, the column-drop plus group-L1 arm scored 12.6875 (std 0.6673; per seed 13.0625, 12.0, 12.0, 12.875, 13.5). The column-drop-only arm scored 12.55 (std 0.7634; 12.1875, 11.875, 12.1875, 12.6875, 13.8125). The column-drop plus weak group-L1 arm scored 12.55 (std 0.724). The first two column-drop configurations, which differ in both drop rate and group-L1, showed no consistent ordering across seeds.

Evidence: exp_0005, exp_0014, exp_0012.

- `exp_0005::val_top1_err.full`: value=12.6875, std=0.6673; owning evidence `exp_0005`.
- `exp_0014::val_top1_err.full`: value=12.55, std=0.7634; owning evidence `exp_0014`.
- `exp_0012::val_top1_err.full`: value=12.55, std=0.724; owning evidence `exp_0012`.
## claim_group_l1_default_null

At the default schedule, adding weak group-L1 to column dropout left the five-seed mean unchanged: 12.55 with the penalty and 12.55 without it. Per-seed values differed only at seed one (11.9375 vs 11.875) and seed four (13.75 vs 13.8125). The verify tier gave 12.1042 with the penalty and 12.0833 without it. The smoke runs of both arms gave 12.1875.

Evidence: exp_0012, exp_0014, exp_0011, exp_0013, exp_0009, exp_0010.

- `exp_0012::val_top1_err.full`: value=12.55, std=0.724; owning evidence `exp_0012`.
- `exp_0014::val_top1_err.full`: value=12.55, std=0.7634; owning evidence `exp_0014`.
- `exp_0011::val_top1_err.verify`: value=12.1042, std=0.1443; owning evidence `exp_0011`.
- `exp_0013::val_top1_err.verify`: value=12.0833, std=0.1804; owning evidence `exp_0013`.
- `exp_0009::val_top1_err.smoke`: value=12.1875, std=0.0; owning evidence `exp_0009`.
- `exp_0010::val_top1_err.smoke`: value=12.1875, std=0.0; owning evidence `exp_0010`.
## claim_group_l1_long_null

On top of the long-schedule EMA recipe (11.8542; per fold 11.9375, 11.625, 12.0), dense group-L1 scored 11.875 (12.0, 11.6875, 11.9375; smoke 12.0). Lazy group-L1 on batch-active rows at a larger coefficient scored 11.9375 (12.1875, 11.75, 11.875; smoke 12.1875). Neither penalty lowered error consistently across the three folds.

Evidence: exp_0028, exp_0029, exp_0021, exp_0026, exp_0027.

- `exp_0028::val_top1_err.verify`: value=11.875, std=0.1654; owning evidence `exp_0028`.
- `exp_0029::val_top1_err.verify`: value=11.9375, std=0.2253; owning evidence `exp_0029`.
- `exp_0021::val_top1_err.verify`: value=11.8542, std=0.2009; owning evidence `exp_0021`.
- `exp_0026::val_top1_err.smoke`: value=12.0, std=0.0; owning evidence `exp_0026`.
- `exp_0027::val_top1_err.smoke`: value=12.1875, std=0.0; owning evidence `exp_0027`.
## claim_ema_default

At the default lr and epoch budget with column dropout, saving EMA-averaged weights gave a five-seed error of 12.025 (11.75, 11.125, 11.875, 11.75, 13.625), against 12.55 (12.1875, 11.875, 12.1875, 12.6875, 13.8125) without averaging. EMA was lower on all five seed-fold pairs. The verify tier gave 11.5833 and the smoke run 11.75.

Evidence: exp_0020, exp_0014, exp_0019, exp_0016.

- `exp_0020::val_top1_err.full`: value=12.025, std=0.9412; owning evidence `exp_0020`.
- `exp_0014::val_top1_err.full`: value=12.55, std=0.7634; owning evidence `exp_0014`.
- `exp_0019::val_top1_err.verify`: value=11.5833, std=0.4018; owning evidence `exp_0019`.
- `exp_0016::val_top1_err.smoke`: value=11.75, std=0.0; owning evidence `exp_0016`.
## claim_ema_long

At the halved lr and doubled epoch budget with column dropout, EMA weights scored 11.8542 (11.9375, 11.625, 12.0; smoke 11.9375). The same schedule without EMA scored 12.125 (12.25, 11.6875, 12.4375; smoke 12.25). EMA was lower on each of the three folds.

Evidence: exp_0021, exp_0030, exp_0017, exp_0025.

- `exp_0021::val_top1_err.verify`: value=11.8542, std=0.2009; owning evidence `exp_0021`.
- `exp_0030::val_top1_err.verify`: value=12.125, std=0.3903; owning evidence `exp_0030`.
- `exp_0017::val_top1_err.smoke`: value=11.9375, std=0.0; owning evidence `exp_0017`.
- `exp_0025::val_top1_err.smoke`: value=12.25, std=0.0; owning evidence `exp_0025`.
## claim_schedule_null

The low-lr long schedule gave no consistent gain over the default schedule. With EMA, it scored 11.8875 over five seeds (11.9375, 11.625, 12.0, 10.8125, 13.0625) against 12.025 (11.75, 11.125, 11.875, 11.75, 13.625). It was higher on seeds zero to two and lower on seeds three and four. Without EMA, it scored 12.125 (12.25, 11.6875, 12.4375) against 12.0833 (12.1875, 11.875, 12.1875), which is mixed.

Evidence: exp_0022, exp_0020, exp_0030, exp_0013.

- `exp_0022::val_top1_err.full`: value=11.8875, std=0.8094; owning evidence `exp_0022`.
- `exp_0020::val_top1_err.full`: value=12.025, std=0.9412; owning evidence `exp_0020`.
- `exp_0030::val_top1_err.verify`: value=12.125, std=0.3903; owning evidence `exp_0030`.
- `exp_0013::val_top1_err.verify`: value=12.0833, std=0.1804; owning evidence `exp_0013`.
## claim_ema_cap_identity

Two implementations of the long-schedule EMA recipe produced identical per-seed errors: verify 11.8542 in both, and five-seed 11.8875 in both. They differ only in the EMA decay cap.

Evidence: exp_0021, exp_0023, exp_0022, exp_0024, 97908ac0a654136db7de257a8bc70c2af303f0403c583b6ad68842fee775a3d9, 0c6bf992b1149fd2c3c085bbf9e823b850f05575113379ff88dd7055c447d375.

- `exp_0021::val_top1_err.verify`: value=11.8542, std=0.2009; owning evidence `exp_0021`.
- `exp_0023::val_top1_err.verify`: value=11.8542, std=0.2009; owning evidence `exp_0023`.
- `exp_0022::val_top1_err.full`: value=11.8875, std=0.8094; owning evidence `exp_0022`.
- `exp_0024::val_top1_err.full`: value=11.8875, std=0.8094; owning evidence `exp_0024`.
## claim_ema_warmup_mechanism

The cap never binds in these runs. The step count is fixed by the recorded pool size, the five-fold split, the batch size and the epoch budget, and at that step count the warmup term of the decay schedule stays below either cap for the whole run. The averaging that helped is therefore governed by the warmup schedule, whose effective window is shorter than the nominal cap implies.

Evidence: exp_0023, exp_0021, exp_0019, 0c6bf992b1149fd2c3c085bbf9e823b850f05575113379ff88dd7055c447d375, 97908ac0a654136db7de257a8bc70c2af303f0403c583b6ad68842fee775a3d9, scorer, train.

## claim_fold_four

In every five-seed arm, the seed-four/fold-four pair had the highest error: 13.5, 13.8125, 13.75, 13.625 and 13.0625. Each five-seed std (0.6673, 0.7634, 0.724, 0.9412, 0.8094) exceeds the three-seed std of the same arm on seeds zero to two.

Evidence: exp_0005, exp_0014, exp_0012, exp_0020, exp_0022, exp_0004, exp_0013, exp_0011, exp_0019, exp_0021.

- `exp_0005::val_top1_err.full`: value=12.6875, std=0.6673; owning evidence `exp_0005`.
- `exp_0014::val_top1_err.full`: value=12.55, std=0.7634; owning evidence `exp_0014`.
- `exp_0012::val_top1_err.full`: value=12.55, std=0.724; owning evidence `exp_0012`.
- `exp_0020::val_top1_err.full`: value=12.025, std=0.9412; owning evidence `exp_0020`.
- `exp_0022::val_top1_err.full`: value=11.8875, std=0.8094; owning evidence `exp_0022`.
- `exp_0004::val_top1_err.verify`: value=12.3542, std=0.6134; owning evidence `exp_0004`.
- `exp_0013::val_top1_err.verify`: value=12.0833, std=0.1804; owning evidence `exp_0013`.
- `exp_0011::val_top1_err.verify`: value=12.1042, std=0.1443; owning evidence `exp_0011`.
- `exp_0019::val_top1_err.verify`: value=11.5833, std=0.4018; owning evidence `exp_0019`.
- `exp_0021::val_top1_err.verify`: value=11.8542, std=0.2009; owning evidence `exp_0021`.
## claim_highcap_bundles

Two bundles that doubled width, used depth two and combined all three mechanisms scored above both default-capacity column-drop arms on every one of folds zero to two. Bundle A, averaging without warmup, scored 14.5417 (16.1875, 13.3125, 14.125; smoke 16.1875). Bundle B, with warmup, scored 13.3125 (13.9375, 12.5625, 13.4375; smoke 13.9375). The column-drop plus group-L1 arm scored 12.3542 (13.0625, 12.0, 12.0) and the column-drop-only arm 12.0833 (12.1875, 11.875, 12.1875).

Evidence: exp_0007, exp_0006, exp_0002, exp_0003, exp_0004, exp_0013.

- `exp_0007::val_top1_err.verify`: value=14.5417, std=1.4821; owning evidence `exp_0007`.
- `exp_0006::val_top1_err.verify`: value=13.3125, std=0.696; owning evidence `exp_0006`.
- `exp_0002::val_top1_err.smoke`: value=16.1875, std=0.0; owning evidence `exp_0002`.
- `exp_0003::val_top1_err.smoke`: value=13.9375, std=0.0; owning evidence `exp_0003`.
- `exp_0004::val_top1_err.verify`: value=12.3542, std=0.6134; owning evidence `exp_0004`.
- `exp_0013::val_top1_err.verify`: value=12.0833, std=0.1804; owning evidence `exp_0013`.
## claim_overall

In this setting, vocabulary-column dropout lowered validation error against the stock control on all three evaluated folds. Of shrinkage's two parts, EMA weight averaging lowered error against its no-EMA ablation on every evaluated fold, while the low-lr long schedule showed no consistent effect. Group-L1 on embedding rows showed no consistent effect.

Evidence: exp_0013, exp_0015, exp_0020, exp_0014, exp_0021, exp_0030, exp_0012, exp_0028, exp_0029, exp_0022.

## setup_data

Data are a class-stratified AG News subset built from the Hugging Face copy fancyzhx/ag_news with sampling seed 0: 8,000 training-split articles form the cross-validation pool, and 2,000 official-test articles form a held-out test set.

Evidence: scorer.

## setup_tokens

Text is lower-cased, split on non-alphanumerics and truncated to 64 tokens. Each token is hashed with crc32 modulo 20000, plus 1, into 20,000 buckets, with id 0 as padding.

Evidence: scorer.

## setup_model

The frozen model averages non-padding token embeddings, applies depth ReLU hidden layers of width hidden, and outputs 4 class logits.

Evidence: scorer.

## setup_folds

The pool is cut into 5 stratified folds by StratifiedKFold with shuffling and random_state 0. A run with seed i trains on the remaining folds and is scored on validation fold i, so seeds and folds are paired.

Evidence: scorer, exp_0005.

## setup_metric

The metric is top-1 error in percent on the validation fold, and lower is better. No run scored the held-out test split.

Evidence: scorer, profile, exp_0005.

## setup_tiers

Smoke runs use one seed with a 120-second budget, verify runs three seeds with 300 seconds, and full runs five seeds with 900 seconds. Later tiers reuse earlier seeds.

Evidence: profile, exp_0004, exp_0005.

## setup_default

The default configuration is lr 0.003, embed_dim 64, hidden 64, depth 1, weight_decay 0.0001 and 8 epochs. It uses Adam with coupled weight decay, batch 128 and normal initialisation with std 0.1.

Evidence: train, profile.

## setup_loss

Every arm trains with cross-entropy loss on the class logits; group-L1, where present, is added to this loss.

Evidence: train, f07c8e86dc90fa4fd07dc8ea0d42770c53ffd7ab5a408b387d05027b11bf1dbd.

## setup_steps

Each run trains on four of the five folds of the pool with batch 128 for 8 or 16 epochs, so the optimizer step count is fixed by the configuration.

Evidence: scorer, train, 0c6bf992b1149fd2c3c085bbf9e823b850f05575113379ff88dd7055c447d375.

## setup_mask

Column dropout draws one Bernoulli keep mask over the vocabulary per step, shared across the batch, and maps dropped ids to padding id 0. A row that would be fully masked keeps its original ids. The two high-capacity bundles lack this fallback.

Evidence: f07c8e86dc90fa4fd07dc8ea0d42770c53ffd7ab5a408b387d05027b11bf1dbd, a8276a911760d2787452432f590718947d828985c203339096fc767f8a3ce15d, c5a5c60f852c746d232a1532c64cac3a528be9696437a39f03253af15b6ad864, b61cfee4b2c9d4df9e8b80060f0a7fdea48531cdcc988f9ba3b56d4ea2029df3.

## setup_control

The control arm runs the unmodified training script at the default configuration, with no column dropout, group-L1 or averaging.

Evidence: 2a8bff2e12176fac283375e32ed76c004d77bb5de63fd39af9ef3cf0e8d32d2d, exp_0015.

## setup_coldrop_l1

The column-drop plus group-L1 arm uses drop rate 0.25 at each step and adds group-L1 at 1e-5 on the sum of embedding-row L2 norms, excluding the padding row.

Evidence: exp_0001, f07c8e86dc90fa4fd07dc8ea0d42770c53ffd7ab5a408b387d05027b11bf1dbd.

## setup_coldrop_only

The column-drop-only arm uses the same code with drop fraction p=0.4 and group-L1 coefficient 0.

Evidence: exp_0013, a8276a911760d2787452432f590718947d828985c203339096fc767f8a3ce15d.

## setup_coldrop_weak_l1

The column-drop plus weak group-L1 arm uses drop fraction p=0.4 with group-L1 at 1e-6.

Evidence: exp_0011, 7977b4ce677a2728596331e84d0e5be0cacf1871899dea2cdb352432deb7393b.

## setup_ema

The EMA arm adds weight averaging to the column-drop-only arm with decay min(0.999, (1+t)/(10+t)) at step t and saves the EMA weights.

Evidence: exp_0019, 9a434a3d44458554aeb8cd9995701d2b5d4369200caefe848ce449671ce641a2.

## setup_long_ema

The long-schedule EMA arm uses lr 0.0015, 16 epochs, drop fraction p=0.4 and group-L1 0. Both implementations use EMA decay min(cap, (1+t)/(10+t)), the same warmup form as the default-schedule EMA arm, with cap 0.999 in one and 0.995 in the other, and save the EMA weights.

Evidence: exp_0021, exp_0023, 0c6bf992b1149fd2c3c085bbf9e823b850f05575113379ff88dd7055c447d375, 97908ac0a654136db7de257a8bc70c2af303f0403c583b6ad68842fee775a3d9.

## setup_long_noema

The long-schedule no-EMA arm uses lr 0.0015, 16 epochs, drop fraction p=0.4 and group-L1 0, and saves raw final weights.

Evidence: exp_0030, 5795924c692ac96d52747e3188d9a763effbf23763d90e9ef2c1b08bdbc269b5.

## setup_long_l1

Two group-L1 arms keep the long-schedule EMA recipe (lr 0.0015, 16 epochs, p=0.4, warmup EMA capped at 0.999). One adds dense group-L1 at 1e-5 on all non-padding embedding rows. The other adds lazy group-L1 at 1e-4 only on rows active in each batch.

Evidence: exp_0028, exp_0029, 9f89c30dc29456181a948b1806cb5352c853d60a5b477240cf83ecf192cf402b, 15e6062617e1ddc7657c30315c13f75d1e167b4dcb48475a0793aa8012bc2c69.

## setup_highcap_a

High-capacity bundle A uses embed_dim 128, hidden 128, depth 2, lr 0.0015, weight_decay 5e-05 and 16 epochs. It adds column dropout p=0.3, group-L1 1e-5 and EMA at decay 0.999 without warmup.

Evidence: exp_0007, c5a5c60f852c746d232a1532c64cac3a528be9696437a39f03253af15b6ad864.

## setup_highcap_b

High-capacity bundle B uses the same configuration with column dropout p=0.25, group-L1 1e-5 and EMA decay 0.999 with warmup.

Evidence: exp_0006, b61cfee4b2c9d4df9e8b80060f0a7fdea48531cdcc988f9ba3b56d4ea2029df3.

## setup_runs

Thirty registered runs over twelve candidate configurations all completed with a recorded validation score; none failed or were excluded. Twelve are smoke runs, twelve are verify runs and six are full-tier runs.

Evidence: exp_0001, exp_0002, exp_0003, exp_0004, exp_0005, exp_0006, exp_0007, exp_0008, exp_0009, exp_0010, exp_0011, exp_0012, exp_0013, exp_0014, exp_0015, exp_0016, exp_0017, exp_0018, exp_0019, exp_0020, exp_0021, exp_0022, exp_0023, exp_0024, exp_0025, exp_0026, exp_0027, exp_0028, exp_0029, exp_0030.

## related_xgboost

XGBoost is a scalable tree-boosting system whose regularized objective uses leaf-count and L2 leaf-weight penalties, alongside shrinkage and column subsampling; it reports no text benchmarks. This study transfers column subsampling and shrinkage to a neural bag-of-embeddings model and adds group-L1 on embedding rows as its own L1-style analogue, ablating each separately.

Evidence: arXiv:1603.02754.

## related_tuning_bias

That work demonstrates, in connectome-constrained circuit models, that comparisons are biased when baselines are under-tuned or inherit parameters tuned for the favoured model, and names the machine-learning analogues. By analogy, the same caution applies here, because the control shares an untuned default configuration with the treatments.

Evidence: arXiv:2611.14508v1.

## related_single_baseline

That study cautions against relying on a single, unjustified task-specific baseline. The control here is likewise one default-configuration model and not a tuned alternative regularizer.

Evidence: arXiv:2611.14519v1.

## related_width

That paper treats width multipliers as a capacity-accuracy trade-off in vision. This study varied width only inside confounded high-capacity bundles and does not isolate capacity.

Evidence: arXiv:1704.04861.

## scope_question

For a small bag-of-embeddings classifier trained on a 10k-article AG News subset, which of three XGBoost-derived regularizers lowers validation top-1 error relative to a matched ablation: per-step vocabulary-column dropout, group-L1 on embedding rows, or shrinkage implemented as a low-lr long schedule with EMA weight averaging?

Evidence: profile.

## scope_method

A frozen scorer defines the data, the hashed-token mean-of-embeddings model and validation top-1 error; only the training script is edited. Arms add transferred mechanisms to the stock script. Most comparisons pair an arm with an ablation that removes one mechanism, at the same configuration and seed-fold pairs. Two comparisons do not: column dropout plus group-L1 against the stock control, and two column-drop configurations that differ in both drop rate and group-L1. Runs proceed through smoke, verify and full tiers, and later tiers reuse earlier seeds. Two high-capacity bundles without matched controls are reported as uninterpretable. No arm was scored on the held-out test split.

Evidence: profile.

## limit_1

The unregularized control ran only on seed-fold pairs zero to two. The column-dropout effect therefore rests on three folds, and no five-seed control exists.

Evidence: scope limitation.

## limit_2

Seed index and validation fold are coupled, so seed variance mixes initialisation noise with fold difficulty. Fold four is consistently hardest.

Evidence: scope limitation.

## limit_3

Verify and full tiers reuse the seeds of earlier tiers. They are nested subsets, not independent replicates.

Evidence: scope limitation.

## limit_4

Each comparison is a sign pattern over three or five folds. No significance test or confidence interval was computed.

Evidence: scope limitation.

## limit_5

Per-fold errors move in steps of one validation article. One of the three long-schedule EMA margins, at fold one, is a single article, so that comparison is weak on its own; the five-fold default-schedule EMA comparison carries most of the weight.

Evidence: scope limitation.

## limit_6

All arms were selected and compared on validation folds. The held-out test split was never scored, so selection bias toward validation noise is possible.

Evidence: scope limitation.

## limit_7

The planned comparison against hidden dropout plus weight decay, tuned with an equal trial budget, never ran. Column dropout may act as generic regularization rather than a word-specific effect. The control used the default configuration and was not tuned for its own arm.

Evidence: scope limitation.

## limit_8

The two default-schedule column-drop configurations change drop rate and group-L1 together. Dose-response in the drop rate was not measured.

Evidence: scope limitation.

## limit_9

EMA effects are compared across separate runs, not as raw versus averaged weights from the same run. Step counts were not logged; the claim that the decay cap never binds is derived from the recorded fold size, batch size and epochs, not measured.

Evidence: scope limitation.

## limit_10

Under the warmup EMA, the averaging window grows with step count, so the with-EMA schedule comparison changes the averaging window and doubles the steps along with lr and epochs. The no-EMA pair, long schedule against the column-drop-only arm, is the schedule-only contrast.

Evidence: scope limitation.

## limit_11

No mechanism measurements were taken: no train/validation gap, learning curves, per-token reliance or embedding-row norms. Two hypotheses remain untested: that gains are early stopping in disguise, and that dropout prevents reliance on topic tokens.

Evidence: scope limitation.

## limit_12

The group-L1 null may reflect coefficients too small relative to the coupled Adam weight decay, or Adam rescaling the penalty. Neither possibility was checked.

Evidence: scope limitation.

## limit_13

The high-capacity bundles confound capacity, lr, epochs, drop rate and EMA variant, and no capacity-matched control ran, so neither result can be attributed to any mechanism. One averaged from initialisation without warmup, a competing explanation for its high error.

Evidence: scope limitation.

## limit_14

The logistic-regression baseline quoted in the profile was never run under this protocol and is not directly comparable with any arm.

Evidence: scope limitation.

## limit_15

Results cover one dataset subset, one small hashed-vocabulary model and short training budgets. Historical code provenance relies on recorded candidate source versions, not execution snapshots.

Evidence: scope limitation.

