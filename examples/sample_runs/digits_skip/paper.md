# Does a Zero-Parameter Identity Skip Change Final Error in a Shallow MLP? A Matched-Arm Test on sklearn-digits

> **Panel review:** reject · mean score 2.7/10 (harsh-but-fair 3 · harsh-critical 2 · open-minded 3)

## Abstract

We ask whether, in a shallow ReLU MLP on sklearn-digits, a zero-parameter identity skip path changes final validation error relative to a weight-identical plain MLP once parameters, initial weights, minibatch order, step count and a shared learning rate are matched. A frozen protocol trained a two-block, width-128 MLP with and without the identity skip, pairing the two arms on twenty seeds with identical initial weights, minibatch order and step count at each of two shared learning rates. A depth-one default recipe served as the anchor arm. The engine reports per-seed paired differences in validation top-1 error (lower error is better), and we judge these descriptively against the registered ±0.5 pp prediction. Earlier single-seed full-batch contrasts, a switch to minibatch SGD, and weight-decay and width changes are reported as exploration only, and no new experiments were run.

At the lower shared learning rate, the mean paired difference (plain minus identity) is 0.73037, with the identity arm lower on seventeen seeds, tied on three and higher on none. At the higher shared learning rate, the mean paired difference is 0.121405, with the identity arm lower on seven seeds, tied on ten and higher on three. Moving from the lower to the higher shared learning rate lowered error more in the plain arm, with a mean paired change of -0.93936, than in the identity arm, with a mean paired change of -0.330395. All four protocol arms scored below the depth-one full-batch reference arm at 4.2454.

Judged by the registered rule, the protocol neither supports nor falsifies H1. The lower-rate mean already exceeds the margin, and the higher-rate mean lies inside it with no recorded interval, so equivalence is not established and the registered one-rate attribution cannot be applied.

## Introduction

We ask whether, in a shallow ReLU MLP on sklearn-digits, a zero-parameter identity skip path changes final validation error relative to a weight-identical plain MLP once parameters, initial weights, minibatch order, step count and a shared learning rate are matched.

FractalNet matches ResNets without identity connections in ultra-deep convolutional networks, and we ask the related question of whether the identity path matters at shallow MLP depth with a parameter-identical control. A separate account attributes batch normalisation's benefit in deep ResNets to downscaling the residual branch by a factor of about sqrt(depth), whereas we work at a depth where that factor is near one and use no normalisation.

RoR adds multilevel shortcuts to deep ResNets and reports accuracy gains without a matched no-shortcut ablation, so we isolate a single zero-parameter skip instead. Hourglass MLPs compare skip placement under parameter-matched designs on generative tasks, and we compare skip against no skip on a classification task.

A frozen protocol trained a two-block, width-128 MLP with and without a zero-parameter identity skip, paired on twenty seeds with identical initial weights, minibatch order and step count at each of two shared learning rates, with a depth-one default recipe as the anchor arm. Per-seed paired differences in validation top-1 error (lower is better) are judged descriptively against the registered ±0.5 pp prediction, and earlier exploration runs are reported as exploration only, with no new experiments run.

As our first contribution, we report that at the lower shared learning rate the paired difference (plain minus identity) has mean 0.73037 over twenty seeds, with the identity arm lower on seventeen seeds and higher on none. At the higher shared learning rate the same paired difference has mean 0.121405 and median 0.0, with the identity arm lower on seven seeds, tied on ten and higher on three. As our second contribution, we report that moving from the lower to the higher shared learning rate lowered error more in the plain arm (mean paired change -0.93936) than in the identity arm (mean paired change -0.330395). All four protocol arms scored below the depth-one full-batch reference arm at 4.2454.

As our third contribution, we show that the registered rule neither supports nor falsifies H1, because the engine recorded no interval at the higher rate and the mean paired difference already exceeds the margin at the lower rate. As our fourth contribution, exploration runs show that reshuffled minibatch SGD with warmup and cosine decay lowered error from 3.758 under full-batch training to five-seed means of 2.7841 and 2.7838, a change that bundles step count, gradient noise and learning-rate schedule and so cannot be attributed to any one of them.

## Related Work

The shallow-paths account attributes the benefit of batch normalisation in deep ResNets to downscaling the residual branch by a factor of about sqrt(depth). Our study works at a depth where that factor is near one, and it uses no normalisation. Shortcut theory proves that, in a stylised two-layer convolutional ResNet, gradient descent with proper normalisation and suitable initialisation avoids a spurious optimum when shortcuts are present. We measure only whether a matched zero-parameter skip changes final error in an unnormalised shallow MLP, so we do not test that optimisation mechanism.

FractalNet matches ResNets without identity connections in ultra-deep convolutional networks. We ask the related question of whether the identity path matters, but at shallow MLP depth and with a parameter-identical control.

RoR adds multilevel shortcuts to deep ResNets and reports accuracy gains without a matched no-shortcut ablation. Our study instead isolates a single zero-parameter skip. Hourglass MLPs compare skip placement under parameter-matched designs on generative tasks. We compare skip against no skip on a classification task. Residual connections have also been applied to train deeper capsule networks. That work is peripheral to our question, and its record carries no abstract.

## Setup and Protocol

Data are sklearn-digits 8x8 images scaled by 1/16. A fixed stratified 20% test split is held out, and the other 80% is cut into 5 stratified folds. The assigned fold serves as the validation split, and the remaining folds form the training data. Each validation fold has about 288 images, so one misclassified image is about 0.35 pp. The metric is val_top1_err, the top-1 error in percent on the validation fold, where lower is better. The frozen scorer computes it by rebuilding a Linear/ReLU stack from the saved state_dict. For context, the profile records a logistic-regression baseline of 3.8 percent validation error.

The protocol architecture is an input projection 64->128 with ReLU, then two blocks h=ReLU(W h+b), then a 10-way head. This network equals build_model with depth=3 and hidden=128. Arm plain_lr005 runs train.py v7 with SKIP=False at lr 0.05, which gives the plain block h=ReLU(W h+b). Arm plain_lr002 also runs train.py v7 with SKIP=False and differs from plain_lr005 only in using lr 0.02.

Arm resid_lr005 runs train.py v8, which differs from v7 only in SKIP=True, so that h_{l+1} = h_l + ReLU(W_l h_l + b_l), at lr 0.05. Its weights are exported as an equivalent width-256 stack, and the run asserts max_abs_diff < 1e-3. Arm resid_lr002 runs train.py v8 with SKIP=True and differs from resid_lr005 only in using lr 0.02. The reference arm runs the unmodified train.py v2 at depth 1 and hidden 128, full-batch, with lr 0.05 for 120 epochs. The reference arm anchors the paired differences and is not part of the skip contrast.

Protocol training uses minibatch SGD with batch 64, momentum 0.9, weight decay 0.0001 and cross-entropy loss for 120 epochs, at a constant learning rate with no schedule. Torch and numpy are seeded with the seed index, and weights use default PyTorch Linear initialisation via build_model. Per-epoch permutations come from a torch Generator seeded with 1000003 plus the seed, so they are identical across paired arms. Every protocol arm ran seeds 1 to 20 at verify fidelity with a 180-second budget per run. Folds cycle through 1, 2, 3, 4, 0, and each seed index uses the same fold in every arm.

The selected idea pre-registered a paired TOST at α = 0.05 with a 0.5 pp margin on final error over 30 paired seeds. Under that plan, per-arm lr was to be chosen from {0.01, 0.02, 0.05, 0.1, 0.2} on a 20% inner split, with a shared-lr robustness sweep. The registered prediction states that at lr=0.05 the paired mean difference (resid_lr005 - plain_lr005) in final val_top1_err has magnitude under 0.5 pp and its 90% interval lies inside [-0.5, +0.5] pp. The prediction states the same for the lr=0.02 pair and expects both arms to reach roughly 2-4% error.

The registered outcome rule falsifies H1 only if the paired 90% interval lies entirely outside [-0.5, +0.5] pp at both lr=0.05 and lr=0.02. Under the same rule, an effect at one lr only is attributed to lr sensitivity. An interval straddling a margin boundary counts as inconclusive.

The run protocol deviated from this plan by dropping per-arm lr tuning and reducing the shared-lr grid to two values, 0.05 and 0.02. It also used 20 seeds instead of 30 and computed descriptive paired mean differences instead of TOST.

## Exploration Runs (Earlier in Time)

The exploration full-batch runs train depth 2 at hidden 128 with full-batch SGD at momentum 0.9 and weight decay 0.0001 for 120 epochs, taking one step per epoch. In these runs the identity path reparametrises each square hidden Linear as W_eff = I + W'. The reparametrisation adds no parameters, and it is folded back into plain weights after training so that the frozen scorer evaluates the trained function.

The full-batch exploration runs used two learning rates: exp_0001, exp_0002, exp_0004 and exp_0005 used lr 0.05, and exp_0003, exp_0006, exp_0007 and exp_0008 used lr 0.1. Two of the lr 0.1 runs come from different script versions: exp_0007 runs train.py v3 with SKIP hard-coded on, and exp_0008 runs v1 with a skip flag defaulting to 1. Both run at lr 0.1 and otherwise perform the same computation.

Each exploration run seeds torch and numpy with the run seed and uses default PyTorch Linear initialisation via build_model. Exploration runs come at three fidelities: smoke uses seed 0 on fold 0, verify uses seeds 0-2 on folds 0-2, and full uses seeds 0-4 on folds 0-4.

The exploration minibatch runs use SGD with momentum 0.9 and cross-entropy loss at batch 64. A torch Generator seeded with the run seed reshuffles the batches each epoch, under a linear warmup then cosine decay with warmup_frac 0.03 or 0.05. Minibatch runs exp_0012, exp_0013, exp_0016 and exp_0017 used lr 0.05, depth 2, hidden 128, weight_decay 0.0001 and 120 epochs with the identity path on. The weight-decay and width arms exp_0021 to exp_0024 also used lr 0.05, and weight_decay 0.0004 was the highest value run.

All weight-decay and width arms use the warmup-0.05 minibatch script: exp_0024 at hidden 128 with weight_decay 0.0004, exp_0023 at hidden 128 with 0.0004, exp_0021 at hidden 256 with 0.0004, and exp_0022 at hidden 256 with 0.0003. These arms, exp_0021 to exp_0024, used depth 2 and 120 epochs. Runs exp_0014 and exp_0015 used lr 0.05, depth 2, hidden 128, weight_decay 0.0001, 120 epochs and warmup_frac 0.03 with the identity path on.

## Results: Matched Protocol

At the lower shared learning rate, over twenty paired seeds, the plain arm scored 3.6363 and the identity-skip arm scored 2.906. The paired difference (plain minus identity) has mean 0.73037, median 0.6945 and seed SD 0.562533. The identity arm was lower on seventeen seeds, tied on three and higher on none.

At the higher shared learning rate, over twenty paired seeds, the plain arm scored 2.697 and the identity-skip arm scored 2.5756. The paired difference (plain minus identity) has mean 0.121405, median 0.0 and seed SD 0.532522. The identity arm was lower on seven seeds, tied on ten and higher on three.

Moving from the lower to the higher shared learning rate lowered error more in the plain arm than in the identity arm. The plain-arm paired difference (higher minus lower rate) has mean -0.93936, with eighteen seeds lower, two tied and none higher. The identity-arm paired difference has mean -0.330395, with fourteen seeds lower, three tied and three higher. All four protocol arms scored below the depth-one full-batch reference arm, which scored 4.2454.

Judged by the registered rule, the protocol neither supports nor falsifies H1. At the higher learning rate the mean paired difference lies inside the registered margin, but the engine recorded no interval, so equivalence is not established. At the lower learning rate the mean paired difference already exceeds the margin, so the registered prediction fails there. The registered one-rate attribution cannot be applied, because it presupposes an interval that was not computed.

## Results: Exploration

The single-seed full-batch exploration contrasts point the same way or not at all. At the lower exploration learning rate, the identity-path arm scored 3.4722 against 5.2083 for the plain arm. At the higher exploration learning rate, both arms scored 4.5139. That pair gives one seed-zero contrast favouring the identity path and one exact tie, and neither is replicated, so these runs leave the skip effect unresolved.

At the higher full-batch exploration learning rate on seed zero, the plain arm and two identity-path runs all scored 4.5139. The two identity-path runs use scripts that perform the same computation and differ only in how the skip is switched on, so their agreement shows deterministic code, not independent replication. The plain arm's three-seed mean at this rate was 3.8235. No identity-path run with more than one seed exists at this rate.

With the identity path held fixed, reshuffled minibatch SGD with warmup and cosine decay gave five-seed means of 2.7841 and 2.7838 for two warmup fractions, against 3.758 for full-batch training. Every seed was lower in both minibatch runs. Three-seed runs of the two minibatch implementations scored 2.5495 and 2.5491. The change bundles step count, gradient noise and learning-rate schedule, so it cannot be attributed to any one of them.

Raising weight decay to the top of its range in the minibatch regime gave a five-seed mean of 2.575 against 2.7838. Seeds zero, two and four fell (2.4306 to 2.0833, 2.439 to 2.0906, 4.5296 to 4.1812), and seeds one and three were unchanged. Changing only the warmup fraction moves seed zero between 2.4306 and 2.0833 and seed two between 2.439 and 2.7875. Those warmup-driven changes are of the same size as the weight-decay changes, so the drop cannot be attributed to weight decay.

At matched weight decay on three seeds, the wider network scored 2.3176 against 2.3172 for the narrower one. The wider network was lower on seed zero (1.7361 versus 2.0833), equal on seed one (2.7778) and higher on seed two (2.439 versus 2.0906). The width contrast is therefore mixed, with no net change. A wider run that also lowered weight decay gave identical per-seed scores of 1.7361, 2.7778 and 2.439.

## Discussion

One explanation fits the pattern: the identity path speeds optimisation under the fixed step budget, so its advantage is largest where the plain arm trains more slowly, at the lower learning rate. The same account would explain why the identity arm is less sensitive to learning rate. This explanation is untested. No validation curves or training-loss logs are in the record, and a slower-converging plain arm and a genuine endpoint effect would produce the same final-error data.

The two arms differ in learning-rate sensitivity, so the path and the learning rate interact. The registered rule, which attributes an effect at one learning rate to learning-rate sensitivity, cannot cleanly separate the two.

## Limitations

The engine computed only mean paired differences, and no TOST, no 90% interval and no bootstrap were computed. The protocol also used twenty seeds rather than the registered thirty. The verdict is therefore descriptive and cannot declare equivalence at either learning rate. The validation fold has about 288 images, so one image moves error by about 0.35 pp. Many paired differences are exact ties or single-image changes, which is close to the registered margin.

Residual arms are scored through an exported width-256 Linear/ReLU stack that re-expresses the trained width-128 residual network. Equivalence is asserted in-script, but float summation order may flip rare ties. The DIAG logs are not in the dossier: init and data-order hashes, step counts, skip activation, export max_abs_diff and activation norms. The protocol's void conditions therefore cannot be checked from the record.

No per-epoch validation curves exist, so speed cannot be separated from endpoint error. The optimisation-speed explanation therefore remains a hypothesis. A competing explanation is that the plain arm is under-converged at the lower learning rate. Seeds and folds are coupled, because each seed is paired with a fold in a fixed cycle. As a result, seed and fold effects cannot be separated, and the twenty observations may be dependent.

Within the protocol, the residual arm differs from the plain arm in its initial function, because the skip is active from step zero, and not only in its trajectory. Learning rate was shared rather than tuned per arm. In the exploration runs, the identity path sits inside the ReLU as a weight reparametrisation (W = I + W'), which is not the protocol's additive skip. In that reparametrisation, weight decay pulls the weights toward identity rather than zero.

The two seed-zero identity-path runs at the higher full-batch learning rate agree because the code is identical and deterministic. Neither script logs parametrisation registration, so that it took effect is inferred from the code and from the lower-rate difference.

The minibatch-versus-full-batch gain is descriptive across five paired seed/fold pairs with no significance test. No full-batch control with a matched step count separates gradient noise from step count. The weight-decay per-seed changes match warmup-only changes in size. No training error was logged, so the overfitting premise behind the weight-decay change is untested.

Width is isolated only at weight decay 0.0004, on three seeds (exp_0021 versus exp_0023). exp_0022 changes width and weight decay jointly, and no five-seed width result exists. Identical per-seed scores at weight decay 0.0003 and 0.0004 may be coincidences of the coarse metric. No saved artifact config was checked to confirm which weight decay was applied.

Results come from one dataset, one depth and one width family. Nothing here generalises to deeper networks, other datasets or normalised residual networks. Current engine source is not proof of historical execution, and legacy exploration records may lack code, data or environment snapshots. The record does not count runs that failed, were killed under the time budget, timed out or were voided. It lists each recorded run as kept or discarded and gives the training attempts of each protocol arm, and nothing further can be recovered.

## Conclusion

We asked whether, in a shallow ReLU MLP on sklearn-digits, a zero-parameter identity skip path changes final validation error relative to a weight-identical plain MLP once parameters, initial weights, minibatch order, step count and a shared learning rate are matched.

At the lower shared learning rate, the identity-skip arm scored below the plain arm over twenty paired seeds, with a mean paired difference (plain minus identity) of 0.73037. The identity arm was lower on seventeen seeds and higher on none. At the higher shared learning rate, the mean paired difference shrank to 0.121405, with a median of 0.0 and ten tied seeds.

Raising the shared learning rate lowered error more in the plain arm (mean paired difference -0.93936) than in the identity arm (mean paired difference -0.330395). Judged by the registered rule, the protocol neither supports nor falsifies H1, because no interval was recorded at the higher rate and the mean paired difference already exceeds the margin at the lower rate.

No per-epoch validation curves exist, so the optimisation-speed explanation remains a hypothesis against the competing account that the plain arm is under-converged at the lower learning rate. The engine computed no TOST, no 90% interval and no bootstrap, and the protocol used twenty seeds rather than the registered thirty, so the verdict is descriptive and cannot declare equivalence at either learning rate.

---

### Drafting notes (for the author)

- **Abstract**: 1 sentence(s) over 40 words (first: "We ask whether, in a shallow ReLU MLP on sklearn-digits, a z…")
- **Introduction**: 3 sentence(s) over 40 words (first: "We ask whether, in a shallow ReLU MLP on sklearn-digits, a z…")
- **Conclusion**: 1 sentence(s) over 40 words (first: "We asked whether, in a shallow ReLU MLP on sklearn-digits, a…")

*Written under a version-bound scientific review and author release. Automated checks and semantic review can still miss errors; inspect the [evidence bindings](evidence_map.md) and the [sentence-by-sentence trace](paper_traced.md).*
