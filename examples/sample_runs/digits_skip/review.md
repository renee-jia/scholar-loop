# Review — Does a Zero-Parameter Identity Skip Change Final Error in a Shallow MLP? A Matched-Arm Test on sklearn-digits

**Recommendation:** reject · **mean score:** 2.7/10

| reviewer | score | recommendation |
|---|---|---|
| harsh-but-fair | 3 | reject |
| harsh-critical | 2 | reject |
| open-minded | 3 | reject |

## Consensus weaknesses

- Results: Matched Protocol and Abstract, on "the engine recorded no interval, so equivalence is not established" and "the registered one-rate attribution cannot be applied". The interval is derivable from numbers the paper itself reports.
- lr 0.05: with mean 0.1214, SD 0.5325 and n = 20, a paired-t 90% interval is roughly [-0.09, +0.33], inside the ±0.5 margin.
- lr 0.02: with mean 0.7304 and SD 0.5625, it is roughly [+0.51, +0.95], just outside the margin.
Under the registered rule (modulo the seed/fold dependence caveat), this reads as equivalence at 0.05 and an effect at 0.02 only, which the rule attributes to LR sensitivity. The headline 'neither supports nor falsifies' is therefore an artifact of not running a one-line computation. Fix: report the paired-t interval and a fold-blocked bootstrap or permutation interval, then apply the rule as registered.
- Results and Discussion, on the 0.73 pp gap at lr 0.02. The most parsimonious reading is that the plain arm is simply under-trained at lr 0.02 under 120 epochs at constant LR.
- plain_lr002 scores 3.64%, essentially the logistic-regression baseline of 3.8%.
- The plain arm improves by 0.94 pp when the LR is raised.
The paper names this competing account but supplies no evidence against it. Fix: log training loss and validation error per epoch; or train to convergence (or to a matched training-loss threshold); or add a cosine schedule so that endpoint error is not a proxy for speed.
- Setup, on the shared-LR design and the dropped per-arm tuning. The registered plan tuned LR per arm from {0.01, …, 0.2} precisely to avoid comparing one arm at a suboptimal LR. With only two values, and the plain arm still improving at the top of the grid, the sentence "lowered error more in the plain arm" cannot be read as a sensitivity property. It may only mean that 0.05 is closer to the plain arm's optimum. Fix: sweep at least {0.02, 0.05, 0.1, 0.2} for both arms and compare at each arm's best LR and across the curve.
- Abstract and Introduction, on "All four protocol arms scored below the depth-one full-batch reference arm at 4.2454". The reference differs in depth, optimiser regime (full-batch, one step per epoch) and step count. The comparison carries no information about the skip. The claim that it 'anchors the paired differences' is never explained, since nothing is computed relative to it. Fix: cut the sentence from the Abstract, or define what the anchor is for.
- Abstract, on "weight-identical plain MLP ... once ... initial weights ... are matched". The paper later concedes (Limitations) that the residual arm computes a different function from step zero. This means 'matched' means matched parameters, not matched initial behaviour. Fix: state this in the Abstract, and consider a control that equalises initial output scale, e.g. a scaled skip or a zero-initialised branch.
- Introduction, fourth contribution, and the whole Results: Exploration section. The minibatch-versus-full-batch, weight-decay and width runs do not bear on the skip question. They use a different identity mechanism (reparametrisation inside the ReLU) and are all inconclusive by the authors' own account. Listing them as a contribution inflates the paper. Fix: move them to an appendix, or drop them.
- Limitations, on seeds coupled to folds. With 20 seeds cycling through 5 folds, there are effectively 4 replicates per fold, and fold-level variance enters every paired difference. The analysis treats the 20 differences as independent. Fix: report per-fold mean differences, use a fold-stratified test, or cross seeds fully with folds.
- Results, on reporting precision. Differences are reported as 0.73037, 0.121405, -0.93936 and so on, while one image equals 0.35 pp and half the lr 0.05 pairs are exact ties. Fix: round to about 0.01 pp. Also report a continuous metric such as validation log-loss, so the paired test is not dominated by discretisation.
- Sign convention. The registered prediction is stated as resid − plain, while all results are reported as plain − resid. This invites sign errors when readers apply the rule. Fix: use one convention throughout.
- Related Work and Introduction. The Introduction repeats the Related Work paragraphs nearly verbatim. Related work also omits the most relevant prior evidence on why skips help optimisation even at modest depth, for example:
- shattered gradients (Balduzzi et al.);
- He et al. 2016's plain-versus-residual comparisons;
- Fixup and zero-init residual branches;
- Veit et al.'s ensemble view.
Fix: deduplicate, and position against these works.
- Significance and originality. That an identity skip eases optimisation and interacts with learning rate is well known. Showing this on one 8x8 dataset, at one depth and one width, without convergence diagnostics does not change practitioner behaviour. The non-obvious finding would be equivalence at convergence, but that was not tested.
- **Results: Matched Protocol / Abstract — the verdict is avoidable.**
- The claim: "The registered one-rate attribution cannot be applied, because it presupposes an interval that was not computed."
- The interval follows trivially from numbers already in the paper.
  - lr 0.02: mean 0.730, SD 0.563, n=20 gives a paired-t 90% CI of about [0.51, 0.95], which is entirely outside ±0.5 pp.
  - lr 0.05: mean 0.121, SD 0.533 gives about [-0.09, 0.33], which is entirely inside the margin.
- Applied as registered, the rule therefore attributes the effect to learning-rate sensitivity rather than leaving the question open.
- The paper withholds a conclusion its own data almost decide.
- Fix: compute the paired t/TOST intervals, plus a fold-clustered bootstrap given the seed–fold coupling, and state the rule's outcome.
- **Setup / Limitations — fold reuse undercuts n=20.**
- The text says "Folds cycle through 1, 2, 3, 4, 0". So the 20 'independent' pairs use only 5 distinct validation sets of ~288 images each.
- The effective sample for between-fold variation is 5, not 20, and the SD-based reasoning above is optimistic.
- Fix: analyse with fold as a blocking or cluster factor, or decouple seeds from folds.
- **Setup — held-out test split is never used.**
- The setup says "A fixed stratified 20% test split is held out", yet every number is validation-fold error.
- The exploration choices (minibatch, schedule, lr range) were made on these same folds.
- Fix: report test-split error for the four protocol arms.
- **Discussion / Limitations — the positive finding is confounded with convergence.**
- The paper offers: "the identity path speeds optimisation under the fixed step budget".
- A competing account fits equally well: at lr 0.02 with a constant rate, 120 epochs and no schedule, the plain arm is simply under-trained.
  - The plain arm at lr 0.02 (3.64%) is barely better than the logistic-regression baseline (3.8%).
  - The plain arm drops 0.94 pp when lr rises to 0.05.
- With no training-loss or validation curves, the only non-null result plausibly reflects the training budget rather than the architecture. It would also be unsurprising.
- Fix: log per-epoch train loss and validation error, and extend training or tune lr per arm as originally registered, to check whether the lr-0.02 gap closes at convergence.
- **Abstract / Introduction — 'weight-identical' overstates the control.**
- The abstract says the skip is compared against "a weight-identical plain MLP".
- Limitations concedes the arms compute different functions from step zero.
- The residual arm is also scored through an exported width-256 stack, not the trained network.
- Fix: rephrase as 'parameter-count and initialisation matched', and report the export max_abs_diff values.
- **Setup — the reference arm is mis-described.**
- The setup states the reference arm "anchors the paired differences". But no paired difference uses it.
- It is a depth-one, full-batch arm with a different optimiser regime, so "All four protocol arms scored below the depth-one full-batch reference arm at 4.2454" is uninformative about skips.
- Fix: either drop the reference arm or state what inference it supports. The logistic-regression baseline is the more meaningful anchor.
- **Sign convention.**
- The registered prediction is stated as (resid − plain), while all results are reported as (plain − resid).
- This invites misreading against the registered margin.
- Fix: use one convention throughout.
- **Exploration / Introduction (fourth contribution) — exploration is presented as a contribution.**
- Exploration runs are counted as a contribution even though they use a different architecture (depth 2 vs depth 3), a different skip (a reparametrisation inside the ReLU) and a different optimiser (warmup + cosine schedule).
- They are single- or few-seed, and the text itself concludes that none of them can be attributed to anything.
- They add length without evidence on the question.
- Fix: move them to an appendix and remove them from the contributions list.
- **Results: Exploration (width) — possible configuration error.**
- The text reports "identical per-seed scores of 1.7361, 2.7778 and 2.439" at weight decay 0.0003 and 0.0004.
- Identical scores across three seeds strongly suggest weight decay was not actually varied.
- exp_0023 and exp_0024 are also listed with identical configurations without explanation.
- Fix: verify the saved configs or drop the claim.
- **Significance / originality.**
- The setting is a single toy dataset, one depth and one width family, with a 2-layer-block MLP where skip effects are known to be minor.
- The question as posed (does a skip matter at shallow depth?) has a widely expected answer: little, and lr-dependent.
- Nothing here would change practice.
- Fix: requires multiple datasets/depths and properly computed equivalence tests, at minimum.
- **Related Work — misrepresentation and padding.**
- The claim that RoR "reports accuracy gains without a matched no-shortcut ablation" needs a citation-level check.
- The capsule-network reference is self-admittedly peripheral and carries no abstract. It is decoration and should be cut.
- Self-consistency (Results: Matched Protocol, last paragraph; Abstract, final paragraph): the claim that 'the registered one-rate attribution cannot be applied, because it presupposes an interval that was not computed' does not hold. The interval follows directly from the reported summary statistics.
- lr 0.02: mean 0.730, SD 0.563, n=20 gives SE ≈ 0.126. The 90% t-interval is ≈ [0.51, 0.95], entirely outside ±0.5.
- lr 0.05: mean 0.121, SD 0.533 gives an interval ≈ [−0.09, 0.33], inside the margin.
- Under the registered rule, this is an effect at one learning rate only, attributed to learning-rate sensitivity, i.e. H1 is not falsified.
Fix: compute the paired t-interval or TOST from the stored per-seed values. If seed–fold dependence is the concern, also report a fold-clustered or permutation interval. Then state the verdict.
- Overreach in framing (Abstract, Discussion): the lower-rate 'effect' coincides with an under-trained plain arm. Plain at lr 0.02 scores 3.64, barely below the 3.8 logistic-regression baseline in Setup, and well above plain at lr 0.05 (2.70).
- The most parsimonious reading is that the skip compensates for a sub-optimal learning rate under a fixed step budget, not that it changes reachable error.
- The registered plan's per-arm learning-rate tuning (dropped in deviations) was exactly the control for this.
Fix: run per-arm learning-rate selection on the inner split as registered, or at least add lr 0.1. Log training loss and per-epoch validation error so speed can be separated from endpoint error.
- Unclear role of the reference arm (Setup: 'The reference arm anchors the paired differences'; Abstract: 'All four protocol arms scored below the depth-one full-batch reference arm at 4.2454').
- The reference arm (depth 1, full-batch, train.py v2) enters no paired difference.
- It differs from the protocol arms in depth, optimiser regime and step count, so the comparison is uninformative about the skip.
Fix: either state what the anchor is used for, or drop the sentence from the Abstract and contributions.
- Sign-convention inconsistency (Setup vs Results): the registered prediction is stated as resid − plain, but all results are reported as plain − identity. Readers must flip signs to check the rule.
Fix: report in the registered direction, or state the flip explicitly next to the rule.
- Significance and originality (Introduction, contributions 1–4): a skip helping a shallow unnormalised MLP at a low learning rate and washing out at a better one is the predictable outcome. There is no mechanism measurement and no second dataset or depth. Contribution 3 ('the rule neither supports nor falsifies') is an absence of analysis rather than a contribution. Contribution 4 is admittedly unattributable.
Fix: one of the following would add a non-obvious element:
- a depth sweep showing where the skip starts to matter under per-arm tuning;
- loss curves showing whether the gap is a convergence-speed effect;
- a second dataset.
- Scope of the exploration section (Exploration Runs; Results: Exploration): roughly a third of the paper covers full-batch vs minibatch training, weight decay and width. These are single- to five-seed runs, use a different skip (the W = I + W' reparametrisation inside the ReLU, per Limitations) and a different schedule (warmup + cosine). They do not bear on the protocol question. The width result (2.3176 vs 2.3172 on 3 seeds) and the identical per-seed scores at weight decay 0.0003 vs 0.0004 are noise-level.
Fix: move this material to an appendix and remove contribution 4 from the Introduction.
- Statistical dependence (Limitations: 'Seeds and folds are coupled'): 20 seeds over 5 folds means 4 seeds per fold, so per-seed differences are not independent. Fix: report per-fold mean differences, a mixed-effects estimate, or decouple seeds from folds.
- Clarity and presentation:
- Numbers are given to 5–6 significant digits (0.73037, −0.330395) on a metric quantised at 0.35 pp.
- Related Work repeats the Introduction nearly verbatim.
- The capsule-network citation is described as peripheral and carries no content; cut it.
- The 'export max_abs_diff < 1e-3' assertion is unverifiable because the DIAG logs are missing (Limitations). That missing evidence covers init and data-order hashes, step counts and skip activation, which the matched-arm claim relies on.
Fix: report numbers to two decimals, deduplicate the Related Work, and include the DIAG logs.

## Strengths

- The paired design is genuinely matched: identical initial weights, identical per-epoch permutations from a generator seeded with 1000003 plus the seed, identical fold per seed index, and a code diff between train.py v7 and v8 limited to the SKIP flag. This is the right way to isolate a zero-parameter architectural change.
- Per-seed sign counts are reported alongside means, medians and seed SDs (e.g. 17/3/0 at lr 0.02 and 7/10/3 at lr 0.05). This gives the reader more than a single mean, and the ties expose the coarse metric.
- The draft is honest about its deviations from the registered plan: 20 instead of 30 seeds, no per-arm LR tuning, a two-value LR grid, and no TOST. The deviations are listed explicitly in Setup rather than hidden.
- The Limitations section correctly identifies most of the threats I would raise:
- seed and fold coupling;
- the export equivalence check of the residual network to a width-256 stack;
- the missing DIAG logs;
- the exploration identity being a W = I + W' reparametrisation inside the ReLU rather than the protocol's additive skip;
- the missing training curves.
- The authors explicitly refuse to over-interpret the exploration runs. For example, they note that exp_0007 and exp_0008 agree because of deterministic identical code rather than replication, and that the minibatch gain bundles step count, noise and schedule.
- The plain and skip arms share initial weights, minibatch permutation (a Generator seeded with 1000003+seed) and step count. This is a cleaner control than most skip ablations.
- Per-seed sign counts are reported alongside means, medians and seed SDs (17/3/0 at lr 0.02; 7/10/3 at lr 0.05). These let a reader judge the discreteness of the effect against the ~0.35 pp single-image resolution.
- Deviations from the pre-registration are stated explicitly in Setup: 20 rather than 30 seeds, no per-arm lr tuning, a two-value grid, and no TOST.
- The Limitations section is unusually candid about several problems: seed–fold coupling, missing DIAG logs, the export-based scoring of residual arms, and the non-equivalence of the exploration reparametrisation (W=I+W') to the protocol's additive skip.
- The matched-arm design isolates the skip path itself: identical initial weights, shared per-epoch permutation generator (1000003 + seed), identical step count and the same fold per seed index across arms (Setup and Protocol).
- Effect direction is reported per seed (17/3/0 at lr 0.02; 7/10/3 at lr 0.05) alongside mean, median and SD. With a 0.35 pp metric quantum this is more informative than a mean alone.
- Deviations from the pre-registration are disclosed explicitly: 20 instead of 30 seeds, no per-arm learning-rate tuning, two learning rates, descriptive statistics instead of TOST.
- The paper disciplines its own exploration results. It notes that the two seed-zero identity runs at lr 0.1 agreeing at 4.5139 reflects deterministic code, not replication. It also notes that warmup-only changes match the weight-decay changes in size, so the weight-decay drop is not attributed.
- The learning-rate × skip interaction (plain arm improves by 0.94 pp from lr 0.02 to 0.05, skip arm by 0.33 pp; 18/2/0 vs 14/3/3) is a concrete observation. It reframes the question from 'does the skip matter' to 'does the skip matter at a well-tuned learning rate'.

## Questions for the authors

- Can you report the paired-t 90% intervals and a fold-blocked bootstrap interval computed from the existing per-seed differences, and state the registered-rule verdict that follows from them?
- Do the training loss of plain_lr002 and its validation error still decrease at epoch 120? If you train both arms to convergence, or add a cosine schedule, does the lr 0.02 gap persist?
- What is each arm's best validation error over a wider shared-LR grid (e.g. up to 0.2)? Does the gap at each arm's optimum fall inside ±0.5 pp?
- What exactly does the depth-one reference arm 'anchor', given that no reported quantity is computed relative to it?
- Were the DIAG checks (init hashes, data-order hashes, step counts, export max_abs_diff) actually run and passed? If so, can the logs be included so the void conditions can be verified?
- Why are 20 seeds mapped onto 5 folds in a fixed cycle rather than fully crossed? What is the per-fold breakdown of the lr 0.02 effect?
- Is the 20% held-out test split ever evaluated? If not, why reserve it?
- Why were paired t intervals or TOST not computed from the 20 per-seed differences, when the reported means and SDs already imply that lr 0.02 lies outside the margin and lr 0.05 inside it?
- With only 5 distinct validation folds, how do the conclusions change under a fold-clustered analysis?
- What is the training loss of plain_lr002 at epoch 120? Does the lr-0.02 gap persist when both arms are trained to convergence?
- Why is the held-out test split never reported?
- In the width runs, were exp_0021 and exp_0022 actually run with different weight decay, given their identical per-seed scores?
- What inference is the depth-one full-batch reference arm meant to support?
- Given the per-seed differences you already have, what are the paired 90% intervals at each learning rate? Does the verdict change to 'effect at one learning rate only, attributed to learning-rate sensitivity' as the summary statistics imply?
- Why was per-arm learning-rate tuning dropped? Can you report which learning rate each arm would select on the registered inner split?
- Do the init and data-order hashes from DIAG confirm identical initial weights and batch order across paired arms? Did the width-256 export assertion pass on all 40 skip runs?
- Does the lr-0.02 gap persist if both arms are trained longer (e.g. 2–4× epochs), or does it close as the plain arm converges?
- What is the per-fold breakdown of the lr-0.02 difference? Is it driven by one or two folds?
- Why was the held-out 20% test split never evaluated, even for the final protocol arms?

## Reviewer summaries

- **harsh-but-fair**: The paper asks whether a zero-parameter identity skip changes final validation error in a two-block, width-128 ReLU MLP on sklearn-digits. The control is a plain MLP with the same parameters, initial weights, minibatch order, step count and a shared learning rate. Twenty paired seeds were run at two shared learning rates (0.02 and 0.05) with constant-LR minibatch SGD.

The main results:
- At lr 0.02 the identity arm is lower by a mean of 0.73 pp (17 of 20 seeds lower, none higher).
- At lr 0.05 the gap shrinks to 0.12 pp (median 0, 10 ties).
- Raising the LR helps the plain arm more than the identity arm.

The authors decline to give a verdict on H1 because no interval was computed. A long exploration section on full-batch versus minibatch training, weight decay and width is appended; it is mostly unrelated to the question.

The limitations section is unusually candid. However, the study does not deliver the answer its own data could give. The main effect is also plausibly an under-training artifact of a constant-LR, fixed-epoch recipe. The question and technique are well explored, and the setting is a single tiny dataset.
- **harsh-critical**: The paper asks whether a zero-parameter identity skip changes final validation top-1 error in a two-block, width-128 ReLU MLP on sklearn-digits. Each plain arm is paired with a skip arm on 20 seeds, with matched initial weights, minibatch order and step count, at two shared learning rates (0.02 and 0.05).

At lr 0.02 the skip arm is lower by a mean of 0.73 pp: 17 seeds lower, 3 tied, none higher. At lr 0.05 the mean gap is 0.12 pp, with a median of 0 and 10 ties. The plain arm benefits more from the higher learning rate.

The authors declare H1 neither supported nor falsified, because no interval was computed. They also append single-seed and few-seed exploration runs on minibatch versus full-batch training, weight decay and width.

The write-up is careful about its own limitations. However:
- The verdict is left open by choice rather than necessity. The intervals follow directly from the reported means and SDs.
- The setting is a toy at the smallest possible scale.
- The one positive finding is most plausibly an under-convergence artefact of a fixed 120-epoch constant-lr budget, which the paper itself cannot rule out.
- Roughly half the paper is exploration material that does not bear on the question.

The result would not change what any practitioner does.
- **open-minded**: The paper tests whether a zero-parameter identity skip changes final validation error in a two-block, width-128 ReLU MLP on sklearn-digits. The design pairs plain and skip arms on 20 seeds with matched initial weights, minibatch order and step count, at two shared learning rates (0.02 and 0.05).

- At lr 0.02 the skip arm is better by a mean of 0.73 pp (SD 0.56; lower on 17 seeds, tied on 3, higher on none).
- At lr 0.05 the gap shrinks to 0.12 pp (median 0, 10 ties).
- The plain arm gains more from the higher learning rate, so skip and learning rate interact.

The authors declare the registered rule inapplicable because no interval was computed, and give no verdict. A long exploration section on full-batch vs minibatch training, weight decay and width is reported as non-confirmatory.

The write-up is careful and candid about its limits. However, the contribution is small: one dataset, one depth, two learning rates. The headline pattern is that a skip helps where the plain network is under-trained and washes out at a better learning rate, which is what one would predict. The non-verdict is also self-imposed, since the reported mean, SD and n are enough to compute the registered interval.
