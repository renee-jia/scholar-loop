# Review — Training Budget Sets the No-Augmentation Control for an MLP on 8x8 Digits, and 3-Seed Estimates Overstate It

**Recommendation:** reject · **mean score:** 2.2/10

| reviewer | score | recommendation |
|---|---|---|
| harsh-but-fair | 2 | reject |
| harsh-critical | 2 | reject |
| open-minded | 2.5 | reject |

## Consensus weaknesses

- Significance and originality: the campaign's actual question (which sub-pixel transforms help an MLP) is untested. What remains is one unresolved hyperparameter point.
- The control's absolute error (~3.3%) is weak for sklearn digits, where simple SVM/kNN/logistic baselines are commonly reported at roughly 1–3% error. The paper includes no such reference point.
- The 'resolution' insight in the Conclusion ('a treatment effect smaller than these verify-to-full shifts cannot be resolved at 5 seeds') essentially restates that SE ≈ 0.32–0.42 at n=5.
- Fix: run at least one augmentation arm against exp_0016 at matched seeds, and add a standard non-neural reference on the same splits.
- Title overreach: 'Training Budget Sets the No-Augmentation Control' is not supported by the paper's own text. The Training budget section states 'No step above 240 epochs is resolved beyond seed spread', and 240 vs 360 at 5 seeds is 0.28 apart against an SE of the difference ≈ 0.49.
- The only budget evidence with a visible effect is the single-seed 120-epoch smoke (4.1667, exp_0004), which has no spread.
- '3-Seed Estimates Overstate It' generalizes from three configs whose seed overlap is unknown.
- Fix: retitle to match the scope, or add multi-seed 120-epoch and >360-epoch runs.
- Inconsistent separation criterion. Setup says 'For 5-seed runs we add the standard error of the mean, which we use to size gaps between configurations.' Yet Training budget ('Both standard deviations are larger than the gap') and Width and weight decay ('within the larger of the two stds') judge separation by per-seed std.
- Std is the wrong yardstick for a difference of means.
- Fix: report the SE of each difference, or Welch intervals, for every pairwise comparison.
- The promotion rule is unstated, which weakens the regression-to-the-mean story. The Abstract and Results describe the full-stage configs as 'promoted for winning at 3 seeds'.
- But exp_0017 (3.1282) has a lower verify mean than exp_0008 (3.1286) and exp_0002 (3.2444), and exp_0018 (3.244) ties exp_0002; none of exp_0017, exp_0018 or exp_0010 went to full.
- So exp_0002 was not a 3-seed winner by mean.
- Fix: state the promotion criterion explicitly, and rewrite the selection argument to match it.
- The smoke-to-verify evidence points at a stage-dependent artifact rather than selection, and the paper underweights this.
- Every smoke fell at verify, kept and discarded alike (the one exception, exp_0007→exp_0008, rose by only 0.004). Every verify run that went to full then rose.
- A uniform drop followed by a uniform rise is exactly what a split or seed set that changes by stage would produce. Combined with the impossible grid values, the cleaner explanation is a pipeline difference between stages.
- Fix: log seeds and split indices per stage, then re-run exp_0016's config at 3 and 5 seeds on one fixed split with disjoint seeds.
- Internal contradiction on seed overlap. Setup says the per-stage seed overlap is 'unconfirmed' and the logs are 'silent'. Limitations then asserts 'Seeds overlap between stages, and the smoke runs may share seeds, so the 3-seed and 5-seed estimates are not independent samples [exp_0008, exp_0009]'.
- Fix: make Limitations conditional, matching Setup.
- Miscitation in Limitations: 'the claim that halving the budget erases the gain has no spread behind it [exp_0009]'. exp_0009 is the hidden=128, 300-epoch full run and has nothing to do with the 120-epoch claim.
- Fix: cite exp_0004 and exp_0002.
- Reproducibility: Setup repeatedly defers to a 'Setup table (Table 1)' and an 'appendix run ledger', neither of which appears in the draft.
- The optimizer, batch size, activation, schedule, split sizes and test-score aggregation (one seed vs a mean over seeds) are all unknown.
- Related Work discusses AdamW-style decoupled decay, but the optimizer, and hence what wd means, is unstated.
- Fix: include the table, and at least fill in the optimizer and the split sizes.
- Baseline weakness: the 3.8 baseline has an inferred config, unknown seeds, no spread and no test score.
- Every 'beats baseline' statement is therefore a comparison against an unverified point estimate.
- Fix: re-run the baseline config at 5 seeds on the same splits and score it on test.
- Discussion overreach and vagueness:
- 'consistent with reports that regularized fully connected layers tolerate long training schedules [doi:10.1145/3065386]' stretches an ImageNet CNN paper onto a 64-pixel MLP. Cut it or soften it.
- The sizing estimate says 'several more seeds than the 5 we ran' without giving the number. With σ=0.7247 and target SE=0.25, n≈9 for a one-sample comparison and ≈17 per arm for an unpaired one. State these numbers.
- The weight-decay ablation is too narrow to support the null-regularization reading in Width and weight decay ('this null result does not support the hypothesis that regularization strength limits error').
- It compares a single 2x step (5e-4 vs 1e-3) at one budget.
- Fix: sweep decay over orders of magnitude, or soften the sentence.
- Leftover drafting notes ('Drafting notes (for the author)') remain in the submission. The Conclusion's 'not significant' carries no number; add '≈1.6 SE'.
- The core contribution is missing. The campaign idea is sub-pixel augmentation versus integer shifts and noise, and Limitations states that 'no augmentation arm was run'. The paper therefore tests none of its motivating hypothesis. What remains is a hyperparameter sweep on a 64-pixel toy dataset. Fix: run the augmentation ablations at 360 epochs against exp_0016 with matched seeds; without them there is no paper.
- The title, 'Training Budget Sets the No-Augmentation Control', is contradicted by the paper's own evidence. Budget section: 'No step above 240 epochs is resolved beyond seed spread'. Discussion: the 360-epoch choice 'rests on that direction alone'. The only budget evidence below 240 epochs is a single seed (exp_0004, 4.1667). Fix: retitle, or run 120 vs 240 vs 360 at ≥5 seeds with reported SE.
- The regression-to-the-mean audit is internally inconsistent with Results: 'How far do 1-seed and 3-seed estimates move?'.
- Every config fell from smoke to verify, including kept smokes selected for being low (3.125→2.8968, 3.4722→3.1282). The authors concede this 'run[s] against that mechanism'.
- Every promoted config rose from verify to full, and its std inflated by up to 150× (0.0063→0.9374).
- A uniform drop-then-rise across all configs points to a stage-dependent seed set or split, not selection.
- No unselected config was ever run at 5 seeds, so selection cannot be separated from a stage effect.
The title claim '3-Seed Estimates Overstate It' is thus unsupported as a statement about 3-seed estimation in general. Fix: rerun some non-promoted configs at 5 seeds, and rerun the promoted ones on disjoint seeds with a fixed split.
- Setup and Limitations contradict each other on seed reuse. Setup says per-stage seed overlap 'is one of [the entries] marked unconfirmed'. Limitations asserts 'Seeds overlap between stages... so the 3-seed and 5-seed estimates are not independent samples'. Fix: state one position, backed by the actual seed lists.
- The 'insight' in the Conclusion is not novel. 'A treatment effect smaller than these verify-to-full shifts cannot be resolved at 5 seeds or fewer' follows directly from σ≈0.72–0.94 and SE=σ/√n. It is a power statement, not a finding about the setting. Fix: drop the novelty framing, or show something non-obvious, e.g. that 3-seed std systematically underestimates 5-seed std across many unselected configs.
- The Discussion sizing paragraph never gives a number. It says 'needs several more seeds than the 5 we ran' and 'roughly twice that count per arm'. With σ=0.7247, a 0.5-point gain and a 2-SE threshold, the one-sample n is ≈(2·0.7247/0.5)² ≈ 8.4, so 9 seeds. The unpaired n is ≈17 per arm. Fix: state these numbers explicitly.
- The baseline is unverifiable. The 3.8 figure has no run in the ledger, no spread and no test score. Its config is inferred from exp_0001's hypothesis text: narrower, wd ~1e-4, ~120 epochs. Every 'beats baseline' statement is therefore against a point estimate of unknown provenance.
- The obvious strong baselines are absent: sklearn MLPClassifier, an SVM-RBF and logistic regression.
- SVM-type models are commonly reported to reach lower errors than this control on this dataset.
The significance of a ~3.3% MLP control is therefore unclear. Fix: rerun the baseline config at 5 seeds, and add standard classical baselines.
- The pipeline integrity issue undermines all spreads. Limitations argues that std 0.0063 'cannot arise when each seed moves in whole-image steps'. If the authors cannot explain how errors are computed (split size, re-sampling, aggregation), no std or SE in the paper can be trusted. Fix: report split sizes, the per-seed raw errors, and the split policy.
- Reproducibility is weak, and there are referencing errors.
- The 'Setup table (Table 1)' and the 'appendix run ledger' are referenced but absent.
- Optimizer, batch size, activation, input scaling, schedule and initialization are all declared possibly unrecorded.
- Limitations cites [exp_0009] for the 120-epoch single-seed claim; exp_0009 is the 128-unit, 300-epoch full run and is irrelevant.
Fix: include the table, the ledger and the full config, and correct the citation.
- Related Work and Discussion overreach with citations.
- It is asserted that decay strength 'has not been measured' in small MLPs, with no support.
- AlexNet [doi:10.1145/3065386] is cited for 'regularized fully connected layers tolerate long training schedules', which is a stretch.
- Swin is cited without relevance to the question.
Fix: remove or justify these citations.
- The weight-decay ablation covers a single pair of values at one budget (exp_0010 vs exp_0018). The two means are equal (3.2444 vs 3.244), and the 1-seed smoke of the 5e-4 config was discarded. This cannot support any statement about regularization limiting error. The Width section's conclusion that the null 'does not support the hypothesis that regularization strength limits error' is fair, but it is the paper's only regularization evidence.
- Leftover drafting notes ('Drafting notes (for the author)', lint warnings) remain in the submission.
- Significance/originality: the paper never runs the idea it was built for (sub-pixel augmentation, its ablations and magnitude sweep). What remains is a null sweep of three knobs on an 8x8 toy dataset plus the observation that 3-seed winners shrink at 5 seeds, which is predictable winner's-curse behaviour, here with n=3. Fix: run at least the ±0.5 px bilinear-shift arm and the integer-shift arm against exp_0016 at matched seeds.
- Title vs evidence: "Training Budget Sets the No-Augmentation Control" overreaches. Results: Training budget concludes "No step above 240 epochs is resolved beyond seed spread," the Discussion says the choice "rests on that direction alone," and the Width/decay section says the control was chosen "by default." Fix: retitle, e.g. "A Seed-Calibrated No-Augmentation Control…", or obtain enough seeds to resolve 240 vs 360.
- Self-contradiction on seed reuse. Setup says per-stage seed overlap is "unconfirmed… where the logs are silent," but Limitations asserts "Seeds overlap between stages, and the smoke runs may share seeds, so the 3-seed and 5-seed estimates are not independent." The Conclusion then says "we do not know how far the seeds of the two stages overlap." Fix: state one position. The identical 3.8194 across three distinct smoke configs is evidence for a shared smoke seed, so say that and stop there.
- Missing artefacts. Setup repeatedly defers to "The Setup table (Table 1)" for split sizes, optimizer, batch size, activation, scaling, schedule, init and test aggregation. Setup also cites "the appendix run ledger." Neither is included. As written, the paper is not reproducible, and the reader cannot check which entries are "unrecorded." Fix: include both.
- Inferable details left unstated. Every smoke value is an integer multiple of 0.3472 = 100/288: 3.125=9, 3.4722=10, 3.8194=11, 4.1667=12 errors. So the smoke validation split is almost certainly 288 images. Test errors 3.3333, 3.5 and 3.7222 are integer counts over 1800 = 5×360 (60, 63, 67), consistent with a 5-seed mean on a 360-image test split. Fix: Setup and Limitations should state these inferences rather than treating split sizes and test aggregation as wholly unknown.
- Mis-citation in Limitations. "the claim that halving the budget erases the gain has no spread behind it [exp_0009]": exp_0009 is the hidden=128, 300-epoch full run and says nothing about 120 epochs. Fix: cite exp_0004 only.
- Results: Width and weight decay: "this null result does not support the hypothesis that regularization strength limits error" draws a conclusion from a single 2x change in decay at 3 seeds, with stds of 0.53 and 0.20. A null at this power is uninformative rather than evidence against. Fix: say the comparison is underpowered, or sweep decay over an order of magnitude (e.g. 1e-4, the inferred baseline value, to 1e-2).
- The Limitations sentence "Two outcomes would overturn the main finding. If more seeds placed the mean of exp_0016 at or above 3.8…" implies the main finding is that the control beats 3.8. The abstract and conclusion already concede that margin is not significant. Fix: restate the main finding as the measured control plus its spread.
- Discussion, sizing paragraph: "needs several more seeds than the 5 we ran" is vague when the number is trivially computable. With σ=0.7247, a 0.5 gain and a 2-SE criterion, the one-sample case needs n≈9; the unpaired two-arm case needs ≈17 per arm. Fix: state these numbers explicitly; the Conclusion's "well over 5 per arm" should cite them.
- Baseline provenance. The 3.8 target's config is inferred solely from the hypothesis text of exp_0001, and its seeds and spread are unknown. Every "vs baseline" statement in the Abstract and Conclusion therefore compares against an unverified point. Fix: re-run the inferred baseline config (hidden<256, wd=1e-4, 120 epochs) at 5 seeds under the same scorer.

## Strengths

- Every validation number traces to a run id, and spread is reported (mean ± std over seeds) at both the 3- and 5-seed stages.
- Each 5-seed full run is scored once on a held-out test split that is explicitly not used for selection, and the paper reports those test numbers alongside the validation means.
- The verify-to-full audit is a concrete, useful observation for this protocol: all three promoted configs rose (2.8968→3.2712, 3.1286→3.5497, 3.2444→3.5494) and their stds grew. In particular, the 3-seed std of 0.0063 for exp_0008 became 0.9374 at 5 seeds (exp_0009).
- SE arithmetic is correct where it is stated. For example, (3.8−3.2712)/(0.7247/√5) ≈ 1.6 SE, and the other two full runs sit below 1 SE from the baseline.
- The authors spotted the grid inconsistency themselves and say so. Smoke errors of 3.8194, 4.1667, 3.125 and 3.4722 fit an n≈288 validation set, but the 3-seed mean of 3.1286 and std of 0.0063 cannot arise from such a set. Flagging this is good scientific hygiene.
- The Limitations section names two concrete falsifiers for the main finding: more seeds pushing exp_0016 to 3.8 or above, and the 5-seed rise vanishing once full-stage seeds are disjoint from verify-stage seeds.
- Every selling-point number in the abstract traces to a run id, and I could verify the arithmetic. Examples: (3.8−3.2712)/(0.7247/√5) ≈ 1.63 SE; the 320→360 verify step of 0.2314 lies within 0.3424 but exceeds 0.1977; the other two full runs sit ≈0.6–0.7 SE below 3.8.
- Each full run is scored once on a held-out test split that is not used for selection. The paper also admits that the selected validation mean (3.2712) is optimistic because it was picked from three full runs.
- The authors flag their own data-integrity anomaly. A 3-seed std of 0.0063 (exp_0008) and non-grid means (3.1286, 3.5494) are incompatible with whole-image errors on a fixed small split, and this is surfaced in Setup and Limitations rather than hidden.
- Each overclaim-prone sentence is hedged ('consistent with... without establishing it', 'not significant', 'unresolved'). The Limitations section names concrete falsifiers, such as disjoint-seed reruns and more seeds pushing exp_0016 to ≥3.8.
- Arithmetic checks out wherever I recomputed it: (3.8−3.2712)/(0.7247/√5)≈1.63 SE ("about one and a half"); exp_0003 and exp_0009 both sit ≈0.6–0.7 SE below 3.8; each verify-to-full rise is ≈0.8–1.2 five-seed SEs, matching "about one"; and every adjacent verify-stage budget gap does lie within the larger std.
- The held-out test discipline is explicit and respected throughout. Test scores for exp_0003/0009/0016 are reported once and never used to rank configs, and the validation mean of exp_0016 is correctly flagged as mildly optimistic because it was selected on validation.
- Results: 1-seed vs 3-seed honestly reports evidence against its own preferred mechanism. Kept smokes fell at verify (e.g. 3.125→2.8968), which selection-induced regression would not predict, and the selection argument is restricted to verify-to-full shifts.
- Limitations makes a genuinely useful granularity argument: means such as 3.1286 and a 3-seed std of 0.0063 cannot arise from whole-image errors on one fixed small split, implying per-seed resampling or an inconsistent scorer. This is the most non-obvious observation in the paper.
- The overturning conditions are stated concretely (mean of exp_0016 ≥3.8 with more seeds; the rise vanishing with disjoint seeds). The Discussion correctly argues that augmentation arms must be scored against the matched control, not against the fixed 3.8.

## Questions for the authors

- What exactly was the rule for promoting a config from verify to full, given that exp_0017 (3.1282) was not promoted while exp_0002 (3.2444) was?
- How can a 3-seed mean of 3.1286 and a std of 0.0063 (exp_0008) arise if smoke errors are multiples of 1/288? Is the validation split redrawn, or resized, per seed or per stage?
- Were full-stage seeds a superset of verify-stage seeds? If so, what were the two extra seeds' errors?
- Is the reported held-out test error from one seed's model or the mean over 5 seeds, and how large is the test split?
- Which optimizer was used, and is weight decay coupled (L2) or decoupled?
- What do standard non-neural baselines (e.g., an RBF SVM or logistic regression) score on the same splits?
- Why was no augmentation arm run, given that the campaign was scoped around it? Even a single fractional-shift arm at 5 matched seeds would make the paper about its stated subject.
- What are the train/validation/test split sizes? Is the split redrawn per seed? How can a 3-seed std of 0.0063 arise under your scoring?
- Were the 5 full-stage seeds a superset of the 3 verify seeds? If so, what are the means on the 2 new seeds alone?
- Why did every config, including those kept for scoring low, drop from smoke to verify? Does the smoke stage use a different split or seed than verify?
- Was the held-out test number computed from a single seed's model or as a 5-seed mean? What is its spread?
- Where does the 3.8 baseline come from? Can you rerun its exact config at 5 seeds?
- How does the control compare to sklearn's MLPClassifier or an SVM on the same splits?
- Why was no augmentation arm run, given that it was the stated campaign goal?
- Can you confirm whether the validation split is resampled per seed? The smoke values being multiples of 100/288 while 3-seed means are not suggests split variance is mixed into every std you report.
- Is the held-out test error a 5-seed mean over 360 images (the values match counts over 1800), or one seed's model?
- Were the full-stage seeds a superset of the verify-stage seeds? If so, the verify-to-full rise for exp_0015→exp_0016 is driven by the two new seeds alone. What were their individual values?
- Why was the 360-epoch config not run at hidden=128, and why was no budget beyond 360 tried, given that every stage's mean was still decreasing at 360?
- Why does the paper exist without a single augmentation arm when the control and seed protocol were apparently ready? Was compute the constraint?

## Reviewer summaries

- **harsh-but-fair**: The draft is a control-only study. It does not test the sub-pixel augmentation hypothesis that motivated the campaign; no augmentation arm was run. It reports a no-augmentation control for a one-hidden-layer MLP on sklearn 8x8 digits, staged as 1-seed smoke, 3-seed verify and 5-seed full runs. Three findings are claimed:
- The three configs that went to the 5-seed stage all rose in validation error relative to their 3-seed estimates (by roughly 0.3–0.42 points), which the authors read as consistent with regression to the mean.
- No budget (above 240 epochs), width or weight-decay difference is resolved.
- The chosen control (hidden=256, wd=1e-3, 360 epochs) reaches 3.27 ± 0.72 validation error (3.33 held-out test), about 1.6 SE below a 3.8 baseline. The paper itself calls this not significant.

The write-up is unusually candid about its limits. However, the net contribution is a single unresolved hyperparameter point on a toy dataset with weak absolute accuracy. Several internal inconsistencies also remain: the title, the promotion rule, the seed-overlap claims, the separation criterion and one miscitation. In addition, a data-pipeline anomaly that the authors themselves flag undermines trust in every number reported.
- **harsh-critical**: The draft sets out to fix a no-augmentation MLP control on sklearn 8x8 digits for a planned sub-pixel augmentation study. The study itself (fractional shifts, rotations, noise, magnitude sweeps) was never run. What remains:
- A small sweep of epochs (120–360), width (128/256) and weight decay (5e-4/1e-3) at fixed lr=0.05 and depth 1.
- A smoke/verify/full staging protocol.
- An 'audit' observing that all three configs promoted at 3 seeds had higher mean error at 5 seeds.

The selected control (exp_0016: hidden 256, wd 1e-3, 360 epochs) reaches 3.2712 ± 0.7247 validation error. That is about 1.6 SE below an unverified 3.8 baseline, which the authors concede is not significant. Its held-out test error is 3.3333. No knob is resolved beyond seed spread.

The write-up is unusually candid about these limits. But the contribution is a non-significant tuning result on a toy dataset. Its main 'insight' — that effects smaller than seed noise cannot be resolved at ≤5 seeds — restates the standard error. The headline 'regression to the mean' finding is confounded by a stage-dependent pattern the paper itself documents.
- **open-minded**: The draft reports a no-augmentation MLP control on sklearn 8x8 digits, intended as the reference for sub-pixel augmentation arms that were never run. It makes three contributions.

1. A stage-shift audit: all three configs promoted at 3 seeds rose in validation error at 5 seeds (2.8968→3.2712, 3.1286→3.5497, 3.2444→3.5494).
2. A budget/width/decay sweep (120–360 epochs, hidden 128/256, wd 5e-4/1e-3) in which no difference is resolved beyond seed spread.
3. A selected control (hidden=256, wd=1e-3, 360 epochs) at 3.2712±0.7247 validation error over 5 seeds and 3.3333 held-out test error, about 1.6 SE below the 3.8 baseline (not significant).

The write-up is unusually candid about its limits. It hedges the regression-to-the-mean reading, scopes it to three configs, flags the suspicious 0.0063 std, and notes that the test aggregation, split policy and seed overlap are unrecorded. However, the campaign's stated idea (sub-pixel augmentation) is entirely untested. The surviving content is a null hyperparameter sweep on a toy dataset plus an n=3 observation about shrinkage from 3 to 5 seeds, which is a textbook expectation. The paper also references a Setup table and an appendix ledger that do not appear, and it contradicts itself on seed overlap.
