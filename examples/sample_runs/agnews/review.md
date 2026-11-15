# Review — Which XGBoost Regularizers Transfer to a Small Bag-of-Embeddings Text Classifier? In One Small AG News Setting, Column Dropout and EMA Lowered Validation Error; Group-L1 and a Long Schedule Showed No Consistent Effect

**Recommendation:** reject · **mean score:** 2.7/10

| reviewer | score | recommendation |
|---|---|---|
| harsh-but-fair | 3 | reject |
| harsh-critical | 2 | reject |
| open-minded | 3 | reject |

## Consensus weaknesses

- Results: Column Dropout, 'The unregularized control at the same configuration scored 13.7708' (also the headline abstract claim). The only evidence for column dropout is 3 folds against a default, untuned control. There is no five-seed control, although five-seed runs exist for nearly every treatment arm. With fold four consistently hardest, the 3-fold mean also omits the most informative fold. Fix: run the control on seeds 3–4 (two runs) plus a tuned hidden-dropout + weight-decay control with an equal trial budget. Without these, 'column dropout lowers error' cannot be separated from 'any regularization at p≈0.4 helps an under-regularized default'.
- Abstract/Introduction framing as 'XGBoost-derived regularizers'. The mapping is loose and the originality claim is weak:
- Vocabulary-column dropout with id→padding is word dropout, as in Iyyer et al. 2015 (DAN) and others.
- EMA weight averaging is Polyak/SWA-style averaging, not XGBoost shrinkage. XGBoost shrinkage scales each new tree's contribution.
- Related Work cites neither word dropout nor Polyak averaging/SWA/EMA literature.
The gains found are exactly the predictable ones. Fix: cite and position against word dropout and weight-averaging work, and state what the XGBoost lens predicts that those literatures do not.
- Results: Group-L1, 'adding weak group-L1 to column dropout left the five-seed mean unchanged … Per-seed values differed only at seed one and seed four'. At 1e-6 the penalty is ~0.016 total against a CE loss of order 0.3–0.4, and trajectories are nearly identical. This is evidence the treatment was not applied at a meaningful dose, not a null result. The abstract and conclusion still say group-L1 'showed no consistent effect' without the qualifier. Fix: either sweep the coefficient until embedding-row norms or sparsity measurably change (log row norms), or restate the claim as 'no effect at coefficients 1e-6–1e-4 that did not measurably change the embeddings'.
- Discussion: 'The averaging that helped is therefore governed by the warmup schedule, whose effective window is shorter than the nominal cap implies.' Two implementations that are numerically identical in this step range give identical errors. That is a code-equivalence check, not evidence about what governs the EMA gain. No warmup-vs-fixed-decay ablation at default capacity was run, and Bundle A's no-warmup variant is confounded. Fix: run EMA with fixed decays (e.g. 0.99, 0.995, 0.999 with bias correction) at default capacity, or drop 'governs'.
- Results: Column Dropout, paragraph 2, 'The column-drop plus group-L1 arm was also lower than the control on every fold'. This arm differs from the control in both p=0.25 dropout and group-L1, so it supports nothing beyond the first comparison. It is also presented with the same weight as a matched ablation. Fix: move it to a clearly labelled unmatched-comparison note, or cut it.
- Shrinkage section, 'The low-lr long schedule gave no consistent gain'. Two problems:
- The with-EMA contrast gives a lower five-seed mean (11.89 vs 12.03). Signs are mixed (3 worse, 2 better), and the paper itself notes the averaging window also changes.
- The no-EMA contrast is only 3 folds (12.125 vs 12.083), with the long schedule nominally worse.
Calling 'shrinkage' half-transferred rests on one 3-fold comparison. Fix: run the no-EMA long schedule on seeds 3–4 and report paired differences with a sign test or bootstrap CI.
- Quality/statistics (Limitations: 'No significance test or confidence interval was computed'). Thirty runs with five-seed stds of 0.67–0.94 and effects of 0.5 points demand at least paired tests. The best result, 5/5 EMA wins, has a one-sided sign-test p≈0.03 on nested, fold-coupled replicates. Fix: report paired differences with CIs, and decouple seed from fold (multiple seeds per fold) so initialisation noise and fold difficulty can be separated.
- Setup: 'No arm was scored on the held-out test split.' A 2,000-article test set exists and is never used, even for the final arms, so every claim is selected and evaluated on the same validation folds. This is cheap to fix: score the control, column-drop-only, EMA and long-EMA arms once on the held-out test set.
- Weak absolute context. Validation error of ~12% on a 4-class AG News subset sits next to no baseline beyond an untuned default. The Limitations section mentions a 'logistic-regression baseline quoted in the profile' that never ran, which is an unexplained reference in a paper. Fix: run TF-IDF logistic regression (or fastText) under the same folds so readers can tell whether 1.7 points is meaningful.
- Setup: 'Smoke runs use one seed with a 120-second budget … so the configuration fixes the optimizer step count.' Wall-clock budgets differ by tier, yet the step count is claimed to be configuration-fixed and the smoke and verify seed-0 scores coincide. The paper should state explicitly that no run hit its time budget. Otherwise the step-count derivation underlying the EMA-cap claim is not guaranteed.
- Clarity. The column-dropout and EMA results are each restated almost verbatim 4–5 times across Abstract, Introduction, Results and Conclusion. Drafting notes and tooling boilerplate are left in the manuscript. The Related Work analogy to connectome-constrained circuit models is decoration that adds no technical content. Fix: state each finding once with its table reference, and add a single results table with all arms × folds.
- Significance/insight. No mechanism measurements were taken (Discussion: 'no train/validation gap, no learning curves, no per-token reliance and no embedding-row norms'). The paper therefore cannot say whether column dropout acts on topic-token reliance or EMA acts as implicit early stopping. A practitioner learns only that word dropout and EMA help an untuned small model, which they would already expect.
- Originality (Introduction/Related Work, 'This study transfers column subsampling and shrinkage to a neural bag-of-embeddings model'): vocabulary-column dropout is word dropout as used in Deep Averaging Networks (Iyyer et al., 2015) and related token-dropout work. EMA is Polyak–Ruppert / Polyak averaging, standard in deep learning. Neither is cited. Mapping XGBoost shrinkage (a per-tree learning rate) onto EMA weight averaging is not a transfer of the same mechanism. Fix: cite and compare against word dropout and SWA/EMA literature, and state what, if anything, is new beyond the XGBoost framing.
- Quality/Abstract ('column dropout lowered validation error against the stock control on all three evaluated folds'): 3/3 positive signs gives a one-sided sign-test p = 0.125. The 5/5 EMA result gives p = 0.031 one-sided, 0.0625 two-sided. Seeds and folds are coupled and tiers are nested, so these are the only replicates. Fix: run the control on all five folds, decouple init seed from fold (several seeds per fold), and report paired CIs or a paired test.
- Quality/Limitations ('The control used the default configuration and was not tuned for its own arm'): the only comparator for column dropout is an untuned default with no dropout of any kind. Meanwhile p=0.4 was selected, after a p=0.25 variant, on the same validation folds used for the comparison. The 1.7-point gain is therefore exactly the predictable effect of adding any regularizer to an unregularized small model. Fix: run the planned hidden-dropout + weight-decay baseline with an equal tuning budget, and score the final choice on the held-out 2k test set.
- Significance/Setup (no reference point): absolute error around 12% on AG News with 6.4k training articles is far from standard results. No linear or fastText baseline was run under this protocol, and the logistic-regression number is explicitly not comparable (Limitations). Fix: report TF-IDF+LR and fastText under the identical folds, so readers can judge whether any arm is competitive at all.
- Overreach in the title and Abstract ('Group-L1 ... showed no consistent effect'): group-L1 was only tried at 1e-6, 1e-5 and lazy 1e-4. At 1e-6 it changed predictions at two folds (one-article differences that cancel exactly). This tests whether a negligible penalty does anything, not whether group-L1 transfers. Fix: sweep the coefficient until embedding-row norms or sparsity measurably change, log those norms, then report the effect on error.
- Clarity/Setup ('Smoke runs use one seed with a 120-second budget ... full runs five seeds with 900 seconds' vs 'the configuration fixes the optimizer step count'): it is unclear what the wall-clock budget does if step count is fixed. If the budget can truncate training, the step-count derivation underlying the EMA mechanism claim is unsafe. Fix: state whether any run hit its time budget, and log actual step counts.
- Results: Shrinkage ('The low-lr long schedule gave no consistent gain'): the with-EMA comparison flips sign by seed group (worse on seeds 0–2, better on 3–4). The no-EMA comparison has only three folds and differs by under one article per fold on average. Calling this a 'null' and folding it into a headline conclusion is not supported in either direction; it is underpowered. Fix: either run five seeds of the no-EMA long schedule, or describe it as inconclusive rather than as no effect.
- Clarity/presentation: repeatedly quoting the smoke result (which is the seed-0 run of the verify tier) presents duplicate data as additional evidence. The Related Work cites connectome-circuit papers by analogy instead of the directly relevant NLP regularization literature. Drafting notes and boilerplate footer text remain in the submission, and the 40-word title is unwieldy. Fix: drop smoke values from results, replace analogical citations with domain citations, and remove the authoring artifacts.
- Originality (Introduction, Related Work, 'This study transfers column subsampling and shrinkage to a neural bag-of-embeddings model'):
- Vocabulary/word dropout for averaged-embedding classifiers is established (e.g., word dropout in Deep Averaging Networks, Iyyer et al. 2015).
- EMA/Polyak weight averaging is a standard, reliably small gain.
- The XGBoost framing adds little mechanism; the 'transfer' is a relabelling of known neural techniques.
- Fix: cite and compare against per-token word dropout. The batch-shared per-vocabulary mask is the only genuinely distinctive design choice; isolate it against per-token dropout.
- Weak evidence for the headline (Abstract, 'column dropout was lower on every one of the three folds'):
- n=3 sign agreement has one-sided sign-test p=0.125.
- The control is the untuned default with no competing regularizer.
- Run count is not the issue; the question 'is this word-specific?' was never fixed with the right baseline.
- Fix: run the control on folds 3–4, and run the planned tuned hidden-dropout + weight-decay baseline with an equal trial budget.
- Uninformative nulls (Results: Group-L1, 'adding weak group-L1 to column dropout left the five-seed mean unchanged'):
- At 1e-6 the penalty changed 2 of 8,000 validation decisions (one article each on two folds), so it was effectively zero.
- The abstract-level statement 'Group-L1 on embedding rows showed no consistent effect' generalizes from coefficients never shown to move the model.
- Fix: report embedding-row norms or sparsity under each coefficient, and sweep the coefficient until a measurable effect on norms appears.
- Small EMA margin (Results: Shrinkage, 'EMA was lower on all five seed-fold pairs'):
- With 1,600-article folds, the per-fold margins are 7, 12, 5, 15 and 3 articles; the mean gain is ~8 articles.
- The 5/5 sign pattern (p≈0.03 one-sided) is the paper's best evidence, but the margins are small.
- EMA is compared across separate runs rather than raw vs averaged weights from the same run; the latter would remove seed noise for free.
- Fix: evaluate raw and EMA weights from the same trajectory, and add an early-stopping or best-checkpoint baseline. This would address the 'early stopping in disguise' hypothesis the Discussion leaves untested.
- Confound in the column-drop-only vs control comparison (Setup, Results: Column Dropout): the only drop-rate evidence besides p=0.4 is p=0.25, which is confounded with group-L1 1e-5. Dose-response is absent, so the gain cannot be distinguished from generic noise injection. Fix: a drop-rate sweep (e.g., 0.1/0.25/0.4/0.6) at L1=0.
- Setup clarity (Setup, 'Smoke runs use one seed with a 120-second budget ... full runs use five seeds with 900 seconds'):
- It is unclear whether these wall-clock budgets ever truncated training.
- The text says the step count is fixed by the configuration, yet step counts were not logged (Discussion). This also leaves the cap-never-binds argument unverified.
- Fix: state that all runs completed their epochs and log step counts.
- Title/scope (Title, Abstract, '10k-article AG News subset'): only 8k articles enter any experiment, and each model trains on 6,400; the 2k test set is never scored. Fix: describe the setting as an 8k CV pool and report one held-out test evaluation of the selected arms to bound validation-selection bias.
- Related Work cites two connectome-constrained circuit-modelling papers as analogies for baseline tuning bias. These are tangential, while directly relevant text-classification regularization work (word dropout, fastText, DAN, SWA/EMA literature) is missing.

## Strengths

- Paired seed-fold design with per-fold values reported for every arm, so readers can check sign patterns themselves rather than trusting means. For example, the default-schedule EMA contrast gives per-seed differences of 0.44, 0.75, 0.31, 0.94 and 0.19, all favouring EMA.
- The Setup section openly separates matched ablations from the two unmatched comparisons, and labels the high-capacity bundles uninterpretable instead of mining them for a claim.
- The derived claim that the EMA cap never binds checks out arithmetically:
- 6,400 training articles / batch 128 = 50 steps/epoch, so 400–800 steps.
- (1+t)/(10+t) reaches 0.995 only at t≈1,790 and 0.999 only at t≈8,990.
- The identical-error result for caps 0.995 and 0.999 is therefore internally consistent, and the paper flags that step counts were not logged.
- Errors are reported at single-article granularity (0.0625% steps on 1,600-article folds), and the Limitations section correctly notes that one long-schedule EMA margin is a single article.
- The Limitations section is comprehensive and matches the evidence. Examples: nested tiers rather than independent replicates, seed–fold coupling, the never-scored test split, and the planned tuned hidden-dropout+WD baseline that never ran.
- Numerical self-consistency is good. Every reported mean reproduces from its per-fold values (e.g., default EMA 60.125/5 = 12.025; weak-L1 62.75/5 = 12.55). The fold-four-hardest claim holds in all five five-seed arms.
- The derived claim that the EMA cap never binds is checkable and correct. With 6,400 training articles and batch 128 there are 50 steps/epoch, so 400 or 800 steps, and (1+t)/(10+t) peaks at about 0.978 or 0.989, below both caps. This explains the identical cap-0.999 and cap-0.995 runs.
- Most comparisons are matched one-mechanism ablations on identical seed-fold pairs, and the paper explicitly flags the two that are not: coldrop+L1 vs control, and the p=0.25/L1 vs p=0.4 pair.
- The limitations section names its own weak points: the single-article margin at fold one, the absent five-seed control, the never-run tuned hidden-dropout baseline, and the untested mechanism hypotheses.
- Uninterpretable high-capacity bundles are reported and explicitly not attributed to any mechanism, rather than being quietly dropped.
- Careful scoping throughout. Every conclusive sentence is tied to 'this setting', a fold count and a matched ablation. The Limitations section pre-empts most attacks: no five-seed control, coupled seed/fold, nested tiers, no significance test, validation-only selection.
- Most contrasts are true one-mechanism ablations at matched configuration and seed-fold pairs. The draft explicitly flags the two that are not: column-drop+L1 vs control, and p=0.25+L1e-5 vs p=0.4.
- The numbers are internally consistent and reproducible from the text. I checked:
- 12.025 = mean(11.75, 11.125, 11.875, 11.75, 13.625);
- 12.55 for both the column-drop-only and weak-L1 arms;
- 11.8875 for the long EMA arm;
- 13.7708 for the control.
- The EMA-cap identity (Discussion) is the most non-obvious observation. Identical per-seed errors at caps 0.999 and 0.995 serve as an implicit determinism check. They also show that the effective averaging window here is the warmup window, roughly t/9 steps, not the nominal 1/(1-cap). The derivation from 50 steps/epoch × 8 or 16 epochs is correct.
- The schedule contrast is correctly decomposed: the with-EMA schedule comparison confounds the averaging window, so the no-EMA pair is identified as the clean contrast.
- The high-capacity bundles are reported but explicitly declared uninterpretable rather than spun.

## Questions for the authors

- Can you run the unregularized control on seeds 3 and 4? Without it, the headline column-dropout claim has no five-seed support while every treatment does.
- Was a drop-rate sweep (e.g. p ∈ {0.1, 0.25, 0.4, 0.6}) with group-L1 fixed at 0 ever run? Without it, why p=0.4?
- Did any run reach its wall-clock budget before completing its epochs? If not, please state it, since the EMA-cap derivation depends on full step counts.
- Did embedding-row norms or the fraction of near-zero rows change at all under group-L1 1e-5/1e-4? If not, how should the group-L1 null be interpreted?
- Is the EMA gain present when evaluating raw vs averaged weights from the same run? That would remove run-to-run noise from the comparison.
- How does the default control compare with a tuned control (best of hidden dropout / weight decay over the same trial budget) and with a TF-IDF logistic-regression baseline on the same folds?
- What concrete prediction does the XGBoost framing make that the word-dropout and weight-averaging literatures do not?
- Was p=0.4 chosen after observing validation error of other drop rates on folds 0–2? If so, how do you separate selection from effect for the column-dropout claim?
- Why was the held-out 2,000-article test split never scored, even for the final EMA+column-dropout recipe? What are the test errors?
- Did any run terminate on its wall-clock budget rather than its epoch budget? What step counts were actually executed?
- How does column dropout compare to standard per-token (non-shared) word dropout, and to dropout on the hidden layer, at a matched tuning budget?
- Does EMA help the unregularized control as well? Without that, the EMA result is conditioned on column dropout being present.
- What group-L1 coefficient first produces a measurable change in embedding-row norms or the number of near-zero rows?
- Does the batch-shared per-vocabulary mask behave differently from standard per-token word dropout at the same rate? This is the one design choice that could make the column-subsampling transfer non-trivial.
- Did any smoke or verify run hit its 120 s / 300 s wall-clock budget before completing its 8 or 16 epochs? If so, are the tier results directly comparable?
- If raw and EMA weights are evaluated from the same training run, does the 5/5 sign pattern hold? Does a best-validation-checkpoint baseline match the EMA gain?
- Did group-L1 at 1e-5, 1e-4 (lazy) or 1e-6 measurably change embedding-row norms relative to the coupled weight decay of 1e-4? If not, can the null be interpreted at all?
- What is the control's error on folds 3 and 4? Given that fold 4 is consistently hardest, would the column-dropout margin survive there?
- Since the effective EMA window is ~t/9 under the warmup schedule, did you try a constant decay with an equivalent window? This would test whether the gain is specific to the growing-window form.

## Reviewer summaries

- **harsh-but-fair**: The paper asks which of three XGBoost-inspired regularizers lowers validation top-1 error for a small hashed bag-of-embeddings classifier on an 8,000-article AG News CV pool, out of a 10k subset. The three are per-step vocabulary-column (token) dropout, group-L1 on embedding rows, and 'shrinkage', operationalised as a low-lr/long schedule plus EMA weight averaging. It reports four findings:
- Column dropout (p=0.4) beat an untuned default control on 3 folds (12.08 vs 13.77).
- EMA beat its no-EMA ablation on 5/5 folds at the default schedule (12.025 vs 12.55) and on 3/3 at the long schedule.
- The long schedule and group-L1 showed no consistent effect.
- The EMA cap is irrelevant because the warmup term never reaches it.

The write-up is unusually honest about its limitations, and most numbers trace to listed runs. The empirical content is thin, though. There are 30 runs and no five-seed control. The control is untuned, and no strong alternative regularizer was run. The null results come from coefficients that barely perturb training. The two positive findings are well-known techniques (word dropout, Polyak/EMA averaging) whose gains are predictable. Nothing was measured that would explain why anything works.
- **harsh-critical**: The paper transfers three regularizers loosely derived from XGBoost to a hashed bag-of-embeddings classifier on an 8k-article AG News cross-validation pool: (i) vocabulary-column dropout, (ii) group-L1 on embedding rows, and (iii) 'shrinkage', split into a low-lr long schedule and warmup EMA weight averaging. All results are validation top-1 error on paired seed/folds; the held-out test split was never scored.

Reported effects:
- Column dropout (p=0.4) beats an untuned default control on 3 folds: 12.08 vs 13.77.
- EMA beats no-EMA on 5/5 folds at the default schedule (12.025 vs 12.55) and on 3/3 folds at the long schedule.
- Group-L1 and the long schedule show no consistent effect.

I checked the arithmetic and it is consistent: per-seed means reproduce, the fold-four ordering holds, and the derived step counts are consistent with the cap-never-binds claim, giving 400 and 800 steps and decays of about 0.978 and 0.989. The limitations section is unusually candid.

The core problems are elsewhere:
- The contribution is word dropout plus Polyak/EMA averaging, both long-established in NLP and deep learning.
- It is evaluated with 3–5 folds, no significance test, no tuned baseline and no test-set confirmation.
- The 'XGBoost transfer' framing is largely cosmetic.

A practitioner would learn nothing they would not already assume.
- **open-minded**: The draft tests whether three regularizers borrowed from XGBoost help a small hashed bag-of-embeddings classifier on an 8k-article AG News cross-validation pool. The three are per-step vocabulary-column dropout, group-L1 on embedding rows, and shrinkage, which is split into a low-lr long schedule and warmup EMA weight averaging. Results are reported as sign patterns over seed-fold pairs.

- **Column dropout** (p=0.4) beat the untuned default control on all 3 folds: 12.08 vs 13.77.
- **EMA averaging** beat its no-EMA ablation on 5/5 folds at the default schedule (12.025 vs 12.55) and on 3/3 folds at the long schedule.
- **Long schedule and group-L1** showed no consistent effect.
- **Cap identity:** two EMA implementations that differ only in decay cap gave bit-identical errors. The authors attribute this to the cap never binding under the (1+t)/(10+t) warmup.

I re-derived the reported means and they are internally consistent. The step-count argument also checks out: 6,400 training articles / 128 = 50 steps/epoch, so 400 or 800 steps, and 801/810 ≈ 0.989 is below both caps.

The write-up is unusually honest about its limits. However:
- The contribution is thin: the two positive findings are predictable from well-known word dropout and Polyak/EMA averaging.
- The headline column-dropout result rests on 3 folds against an untuned control.
- The null results use penalties whose effect on the model was never verified.
