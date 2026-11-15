# Research directions — regularization for small text classifiers trained on limited labelled data

## Your steer

> What the author wants to find out: I want to know which cheap regularizers actually help a small bag-of-embeddings text classifier when labels are scarce, and why, rather than collecting another leaderboard point.
> The author's hunches:
>   H1. Word dropout helps mostly because it stops the model from leaning on a few topic-specific tokens, not because it acts like generic noise.
>   H2. Part of any reported gain is early stopping in disguise: a regularized model simply peaks later within the same epoch budget.
> Stay away from: Pretrained encoders or external data; Pure hyperparameter sweeps with no mechanism behind them
> Kind of result the author values: A mechanism claim with a matched control; a clean negative result is welcome.
> The author already knows or has tried: A logistic-regression baseline reaches about 13.3% error on this split

## Direction map

_Directions are lines of inquiry, not experiments. Choose one or more; ideas are generated only inside the directions you choose._

| # | direction | stance toward your hunches | builds on |
|---|---|---|---|
| 1 | **Targeted versus random word dropout** | develops H1 | `arXiv:1704.04861`, `arXiv:2611.14508v1` |
| 2 | **Is it just later peaking? Best-epoch-matched comparison** | challenges H2 | `arXiv:2611.14508v1`, `arXiv:2611.14519v1` |
| 3 | **Token-structured versus matched generic noise** | challenges H1 | `arXiv:1704.04861`, `arXiv:2611.14508v1` |
| 4 | **Width versus explicit regularization: substitutes or stack?** | outside | `arXiv:1704.04861`, `doi:10.1109/5.726791`, `arXiv:2611.14508v1` |
| 5 | ✅ **Boosting-style regularizers transferred to a neural bag-of-words...** | yours | — |

### 1. Targeted versus random word dropout

If word dropout works by breaking reliance on a few topic-specific tokens, then dropping only the most class-informative tokens should capture most of the gain. Dropping the same number of uninformative tokens should capture little of it.

- **How it relates to your steer:** Tests H1 directly. The dropout rate is held fixed, and only which tokens are eligible to be dropped changes. A gain that tracks token informativeness supports H1's 'leaning on a few tokens' story over 'generic noise'.
- **Gap:** No reviewed paper studies dropout for BoW text (overview). Gap 4 lists dropout among the untested explicit regularizers for small BoW MLPs.
- **Open question:** Does the val error reduction from word dropout depend on which tokens are dropped at a matched expected drop count? Does it shrink the share of logit mass carried by the top-k tokens per document?
- **First experiment:** 1. Rank vocabulary by chi-square or mutual information with the label on the training split. 2. Train the default MLP under four conditions: no dropout; uniform word dropout at p=0.1; dropout restricted to the top-5% informative tokens, rate set to match the expected tokens dropped per doc; dropout restricted to the bottom-50% tokens, also count-matched. 3. Use 3 seeds and the 8-epoch budget, giving roughly 12 short runs. 4. Report val_top1_err. 5. Report a concentration statistic: the fraction of the summed |gradient×input| attribution from each doc's top 3 tokens.
- **How it could turn out dull:** At 10k docs with an 8-epoch budget, all dropout variants may land within seed noise of each other (±0.3%), and none may beat 13.3. The result would then say nothing about mechanism.
- **ID:** `dir_ecf3244566`

### 2. Is it just later peaking? Best-epoch-matched comparison

Regularizer gains under a fixed 8-epoch budget may disappear once each model is evaluated at its own best epoch with its own tuned learning rate and epoch count. Under this reading, the regularizer only shifts when the model peaks.

- **How it relates to your steer:** Argues with H2 by stating what would make it wrong. H2 is false if the regularized model's best-epoch error is still lower than the unregularized model's best-epoch error after both get an equal tuning budget over lr and epochs. H2 is true if the gap closes.
- **Gap:** Gap 1 and the tuning-estimand methodology: comparisons where only one arm is tuned are biased. Here that means comparing against a baseline whose stopping point was not tuned.
- **Open question:** After matching per-model tuning of lr and stopping epoch, how much of the dropout, weight-decay or label-smoothing gain remains?
- **First experiment:** 1. Train each arm with per-epoch val logging and epochs extended to 30: baseline (weight decay 1e-4, no dropout), word dropout, and label smoothing 0.1. 2. Give each arm the same 6-point lr grid. 3. Record the oracle best-epoch error and the epoch at which it occurs. 4. Compare the fixed-8-epoch gain with the best-epoch-matched gain. 5. Use 3 seeds.
- **How it could turn out dull:** Oracle best-epoch selection on val is optimistically biased for every arm. Small residual gaps (<0.3%) may then be uninterpretable without a held-out split taken from the 10k.
- **ID:** `dir_bacd44bd5e`

### 3. Token-structured versus matched generic noise

H1 is wrong if structure-free noise reproduces word dropout's gain when matched for perturbation magnitude. Two such controls are Gaussian noise added to the pooled embedding and dropout on embedding dimensions.

- **How it relates to your steer:** This is the matched control H1 itself implies. H1 claims the gain is 'not because it acts like generic noise', so the test equalizes noise magnitude and checks whether token structure matters at all.
- **Gap:** Gap 4: there is no evidence on whether explicit regularizers on text substitute for each other. Matching the perturbation magnitude makes it possible to tell the mechanisms apart.
- **Open question:** Measured at the pooled representation, does word dropout beat dimension dropout and Gaussian noise when all three induce the same variance?
- **First experiment:** 1. Measure the per-example variance of the pooled embedding under word dropout at p=0.1. 2. Calibrate Gaussian sigma and embedding-dimension dropout rate so each produces the same variance. 3. Train all three arms plus a no-noise arm, using 3 seeds and default knobs. 4. Compare val_top1_err.
- **How it could turn out dull:** Variance matching may be too crude, because word dropout also changes the mean through its interaction with document length. Equal results could then reflect miscalibration rather than a real equivalence.
- **ID:** `dir_ba83c32427`

### 4. Width versus explicit regularization: substitutes or stack?

In a bag-of-embeddings MLP, embed_dim and hidden width may act as the dominant regularizer. Once capacity is tuned, explicit regularizers may add nothing, and a properly tuned linear model may remain the one to beat.

- **How it relates to your steer:** Comes from the literature rather than the hunches. The capacity-control papers and Gap 1 suggest the 13.3 linear baseline may be hard to beat at all. If that holds, the question 'which cheap regularizer helps' reduces to capacity.
- **Gap:** Gaps 1 and 4: capacity and explicit regularizers have never been crossed for text, and per-model tuning is required.
- **Open question:** Across embed_dim in {8, 16, 64, 128}, does the best achievable error with dropout plus weight decay fall below the best achievable error without them?
- **First experiment:** 1. Cross 4 embed_dim values with 2 regularizer settings (none vs word dropout 0.1 + weight decay 1e-3). 2. Hold hidden=embed_dim and depth=1. 3. Give each cell a 3-point lr grid and use 2 seeds. 4. Plot the error-versus-width curves.
- **How it could turn out dull:** Every cell may sit at or above 13.3. That clean negative would show small MLPs cannot beat linear BoW here, but it would not identify any mechanism.
- **ID:** `dir_7cae76a9fa`

### 5. Boosting-style regularizers transferred to a neural bag-of-words...  ✅ chosen

Boosting-style regularizers transferred to a neural bag-of-words classifier: per-batch vocabulary-column subsampling (word-level input dropout) and L1 shrinkage on embedding rows, each ablated against hidden dropout plus weight decay

- Written by you. Not yet tied to reviewed papers.
- **ID:** `dir_f558f68fb9`

> Your note on the choice: The author picked this idea from an earlier idea list of this session (XGBoost-style column subsampling and L1 shrinkage); it also bears on hunch H1.
