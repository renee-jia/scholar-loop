# Approved statement bindings

Writing release: `fbc601c8cf23c9750e76fb8c1073a8c890d6b9548296800b2ce195a950ddc2e6`

This snapshot records the approved scope. Revalidate the live review before further writing.

Raw sources, measurements and input hashes: [approved_evidence.json](approved_evidence.json).

## claim_protocol_lr005

At the higher shared learning rate, over twenty paired seeds, the plain arm scored 2.697 and the identity-skip arm 2.5756. The paired difference (plain minus identity) has mean 0.121405, median 0.0 and seed SD 0.532522. The identity arm was lower on seven seeds, tied on ten and higher on three.

Evidence: protocol_a74a42fceb8b6deef18f8722_plain_lr005, protocol_a74a42fceb8b6deef18f8722_resid_lr005, protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9.

- `protocol_a74a42fceb8b6deef18f8722_plain_lr005::val_top1_err.verify`: value=2.697, std=1.1468; owning evidence `protocol_a74a42fceb8b6deef18f8722_plain_lr005`.
- `protocol_a74a42fceb8b6deef18f8722_resid_lr005::val_top1_err.verify`: value=2.5756, std=1.1394; owning evidence `protocol_a74a42fceb8b6deef18f8722_resid_lr005`.
- `protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9::val_top1_err.difference.plain_lr005.minus.resid_lr005`: value=0.121405, std=None; owning evidence `protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9`.
## claim_protocol_lr002

At the lower shared learning rate, over twenty paired seeds, the plain arm scored 3.6363 and the identity-skip arm 2.906. The paired difference (plain minus identity) has mean 0.73037, median 0.6945 and seed SD 0.562533. The identity arm was lower on seventeen seeds, tied on three and higher on none.

Evidence: protocol_a74a42fceb8b6deef18f8722_plain_lr002, protocol_a74a42fceb8b6deef18f8722_resid_lr002, protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9.

- `protocol_a74a42fceb8b6deef18f8722_plain_lr002::val_top1_err.verify`: value=3.6363, std=0.9821; owning evidence `protocol_a74a42fceb8b6deef18f8722_plain_lr002`.
- `protocol_a74a42fceb8b6deef18f8722_resid_lr002::val_top1_err.verify`: value=2.906, std=1.0328; owning evidence `protocol_a74a42fceb8b6deef18f8722_resid_lr002`.
- `protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9::val_top1_err.difference.plain_lr002.minus.resid_lr002`: value=0.73037, std=None; owning evidence `protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9`.
## claim_protocol_verdict

Judged by the registered rule, the protocol neither supports nor falsifies H1. At the higher learning rate the mean paired difference lies inside the registered margin, but the engine recorded no interval, so equivalence is not established. At the lower learning rate the mean paired difference already exceeds the margin, so the registered prediction fails there. The registered one-rate attribution cannot be applied, because it presupposes an interval that was not computed.

Evidence: protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9.

- `protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9::val_top1_err.difference.plain_lr005.minus.resid_lr005`: value=0.121405, std=None; owning evidence `protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9`.
- `protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9::val_top1_err.difference.plain_lr002.minus.resid_lr002`: value=0.73037, std=None; owning evidence `protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9`.
## claim_lr_sensitivity

Moving from the lower to the higher shared learning rate lowered error more in the plain arm than in the identity arm. The plain-arm paired difference (higher minus lower rate) has mean -0.93936, with eighteen seeds lower, two tied and none higher. The identity-arm difference has mean -0.330395, with fourteen lower, three tied and three higher. All four protocol arms scored below the depth-one full-batch reference arm at 4.2454.

Evidence: protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9, protocol_a74a42fceb8b6deef18f8722_reference.

- `protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9::val_top1_err.difference.plain_lr005.minus.plain_lr002`: value=-0.93936, std=None; owning evidence `protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9`.
- `protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9::val_top1_err.difference.resid_lr005.minus.resid_lr002`: value=-0.330395, std=None; owning evidence `protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9`.
- `protocol_a74a42fceb8b6deef18f8722_reference::val_top1_err.verify`: value=4.2454, std=0.8742; owning evidence `protocol_a74a42fceb8b6deef18f8722_reference`.
## claim_mechanism_speed

One explanation fits the pattern: the identity path speeds optimisation under the fixed step budget, so its advantage is largest where the plain arm trains more slowly, at the lower learning rate. The same account would explain why the identity arm is less sensitive to learning rate. This explanation is untested. No validation curves or training-loss logs are in the record, and a slower-converging plain arm and a genuine endpoint effect would produce the same final-error data.

Evidence: protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9, arXiv:1909.04653, openalex:W3007689282.

## claim_skip_effect_unresolved

The single-seed full-batch exploration contrasts point the same way or not at all. At the lower exploration learning rate, the identity-path arm scored 3.4722 against 5.2083 for the plain arm. At the higher exploration learning rate, both arms scored 4.5139. That is one seed-zero contrast favouring the identity path and one exact tie, and neither is replicated, so these runs leave the skip effect unresolved.

Evidence: exp_0001, exp_0002, exp_0003, exp_0008.

- `exp_0001::val_top1_err.smoke`: value=3.4722, std=0.0; owning evidence `exp_0001`.
- `exp_0002::val_top1_err.smoke`: value=5.2083, std=0.0; owning evidence `exp_0002`.
- `exp_0003::val_top1_err.smoke`: value=4.5139, std=0.0; owning evidence `exp_0003`.
- `exp_0008::val_top1_err.smoke`: value=4.5139, std=0.0; owning evidence `exp_0008`.
## claim_fullbatch_lr01_seed0

At the higher full-batch exploration learning rate on seed zero, the plain arm and two identity-path runs all scored 4.5139. The two identity-path runs use scripts that perform the same computation and differ only in how the skip is switched on, so their agreement shows deterministic code, not independent replication. The plain arm's three-seed mean at this rate was 3.8235. No identity-path run with more than one seed exists at this rate.

Evidence: exp_0003, exp_0006, exp_0007, exp_0008.

- `exp_0003::val_top1_err.smoke`: value=4.5139, std=0.0; owning evidence `exp_0003`.
- `exp_0006::val_top1_err.verify`: value=3.8235, std=0.598; owning evidence `exp_0006`.
- `exp_0007::val_top1_err.smoke`: value=4.5139, std=0.0; owning evidence `exp_0007`.
- `exp_0008::val_top1_err.smoke`: value=4.5139, std=0.0; owning evidence `exp_0008`.
## claim_minibatch_vs_fullbatch

With the identity path held fixed, reshuffled minibatch SGD with warmup and cosine decay gave five-seed means of 2.7841 and 2.7838 for two warmup fractions, against 3.758 for full-batch training. Every seed was lower in both minibatch runs. Three-seed runs of the two minibatch implementations scored 2.5495 and 2.5491. The change bundles step count, gradient noise and learning-rate schedule, so it cannot be attributed to any one of them.

Evidence: exp_0005, exp_0012, exp_0013, exp_0014, exp_0015, exp_0016, exp_0017.

- `exp_0005::val_top1_err.full`: value=3.758, std=0.9026; owning evidence `exp_0005`.
- `exp_0013::val_top1_err.full`: value=2.7841, std=1.0751; owning evidence `exp_0013`.
- `exp_0017::val_top1_err.full`: value=2.7838, std=1.046; owning evidence `exp_0017`.
- `exp_0012::val_top1_err.verify`: value=2.5495, std=0.4038; owning evidence `exp_0012`.
- `exp_0016::val_top1_err.verify`: value=2.5491, std=0.1981; owning evidence `exp_0016`.
## claim_weight_decay

Raising weight decay to the top of its range in the minibatch regime gave a five-seed mean of 2.575 against 2.7838. Seeds zero, two and four fell (2.4306 to 2.0833, 2.439 to 2.0906, 4.5296 to 4.1812); seeds one and three were unchanged. Changing only the warmup fraction moves seed zero between 2.4306 and 2.0833 and seed two between 2.439 and 2.7875. Those are changes of the same size, so the drop cannot be attributed to weight decay.

Evidence: exp_0024, exp_0017, exp_0013.

- `exp_0024::val_top1_err.full`: value=2.575, std=0.9735; owning evidence `exp_0024`.
- `exp_0017::val_top1_err.full`: value=2.7838, std=1.046; owning evidence `exp_0017`.
- `exp_0013::val_top1_err.full`: value=2.7841, std=1.0751; owning evidence `exp_0013`.
## claim_width

At matched weight decay on three seeds, the wider network scored 2.3176 against 2.3172 for the narrower one. It was lower on seed zero (1.7361 versus 2.0833), equal on seed one (2.7778) and higher on seed two (2.439 versus 2.0906). The width contrast is mixed with no net change. A wider run that also lowered weight decay gave identical per-seed scores of 1.7361, 2.7778 and 2.439.

Evidence: exp_0021, exp_0022, exp_0023.

- `exp_0021::val_top1_err.verify`: value=2.3176, std=0.5313; owning evidence `exp_0021`.
- `exp_0023::val_top1_err.verify`: value=2.3172, std=0.3989; owning evidence `exp_0023`.
- `exp_0022::val_top1_err.verify`: value=2.3176, std=0.5313; owning evidence `exp_0022`.
## setup_data

Data are sklearn-digits 8x8 images scaled by 1/16. A fixed stratified 20% test split is held out, and the other 80% is cut into 5 stratified folds; the assigned fold is the validation split and the rest is training data.

Evidence: scorer, profile.

## setup_metric

The metric is val_top1_err: top-1 error in percent on the validation fold. It is computed by the frozen scorer, which rebuilds a Linear/ReLU stack from the saved state_dict.

Evidence: scorer, profile.

## setup_baseline

The profile records a logistic-regression baseline of 3.8 percent validation error.

Evidence: profile.

## setup_protocol_prediction

Registered prediction: at lr=0.05, the paired mean difference (resid_lr005 - plain_lr005) in final val_top1_err has magnitude under 0.5 pp and its 90% interval lies inside [-0.5, +0.5] pp. The same holds for the lr=0.02 pair, and both arms reach roughly 2-4% error.

Evidence: protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9.

## setup_protocol_rule

Registered outcome rule: H1 is falsified only if the paired 90% interval lies entirely outside [-0.5, +0.5] pp at both lr=0.05 and lr=0.02. An effect at one lr only is attributed to lr sensitivity. An interval straddling a margin boundary is inconclusive.

Evidence: protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9.

## setup_preregistered_plan

The selected idea pre-registered a paired TOST at α = 0.05 with a 0.5 pp margin on final error over 30 paired seeds. Per-arm lr was to be chosen from {0.01, 0.02, 0.05, 0.1, 0.2} on a 20% inner split, with a shared-lr robustness sweep.

Evidence: protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9.

## setup_protocol_deviations

The run protocol dropped per-arm lr tuning and reduced the shared-lr grid to two values, 0.05 and 0.02. It used 20 seeds instead of 30 and computed descriptive paired mean differences instead of TOST.

Evidence: protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9.

## setup_protocol_seeds

Every protocol arm ran seeds 1 to 20 at verify fidelity with a 180-second budget per run. Folds cycle through 1, 2, 3, 4, 0, and each seed index uses the same fold in every arm.

Evidence: protocol_a74a42fceb8b6deef18f8722_plain_lr005, protocol_a74a42fceb8b6deef18f8722_resid_lr005, protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9.

## setup_protocol_arch

Protocol architecture: input projection 64->128 with ReLU, then two blocks h=ReLU(W h+b), then a 10-way head. This equals build_model with depth=3 and hidden=128.

Evidence: protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9, 74bc1cc036682bba09f68b01c40ce3fcb5b77d14d66d32145e4eb51f5f919657.

## setup_protocol_optim

Protocol training: minibatch SGD with batch 64, momentum 0.9, weight decay 0.0001 and cross-entropy loss for 120 epochs, at a constant learning rate with no schedule.

Evidence: protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9, 74bc1cc036682bba09f68b01c40ce3fcb5b77d14d66d32145e4eb51f5f919657.

## setup_protocol_seeding

Protocol seeding: torch and numpy are seeded with the seed index, and weights use default PyTorch Linear initialisation via build_model. Per-epoch permutations come from a torch Generator seeded with 1000003 plus the seed, identical across paired arms.

Evidence: 74bc1cc036682bba09f68b01c40ce3fcb5b77d14d66d32145e4eb51f5f919657, f1e5b8fc27d64deb2364f4eb72f53c26e3c3f4625507b9b112a4800000e4cf15.

## setup_arm_plain_lr005

Arm plain_lr005 runs train.py v7 with SKIP=False at lr 0.05: the plain block h=ReLU(W h+b).

Evidence: protocol_a74a42fceb8b6deef18f8722_plain_lr005, 74bc1cc036682bba09f68b01c40ce3fcb5b77d14d66d32145e4eb51f5f919657.

## setup_arm_resid_lr005

Arm resid_lr005 runs train.py v8, which differs from v7 only in SKIP=True, so h_{l+1} = h_l + ReLU(W_l h_l + b_l), at lr 0.05. Weights are exported as an equivalent width-256 stack, and the run asserts max_abs_diff < 1e-3.

Evidence: protocol_a74a42fceb8b6deef18f8722_resid_lr005, f1e5b8fc27d64deb2364f4eb72f53c26e3c3f4625507b9b112a4800000e4cf15.

## setup_arm_plain_lr002

Arm plain_lr002 runs train.py v7 with SKIP=False and differs from plain_lr005 only in using lr 0.02.

Evidence: protocol_a74a42fceb8b6deef18f8722_plain_lr002, e03d7217ca1fcc72387d45c0c4e380d87b6bdf7432756f238a693053f243a35f.

## setup_arm_resid_lr002

Arm resid_lr002 runs train.py v8 with SKIP=True and differs from resid_lr005 only in using lr 0.02.

Evidence: protocol_a74a42fceb8b6deef18f8722_resid_lr002, 2f6f50ed4642f0314637475c5a662bc0589f13e818497eb3b7d1fd93bd28e406.

## setup_arm_reference

The reference arm runs the unmodified train.py v2 at depth 1 and hidden 128, full-batch, with lr 0.05 for 120 epochs. It anchors the paired differences and is not part of the skip contrast.

Evidence: protocol_a74a42fceb8b6deef18f8722_reference, 67918bb7fb30b8ecc4c0c5bfd40cf8c71602e58a97a5dcfab3646a7f772f0af4.

## setup_val_size

Each validation fold has about 288 images, so one misclassified image is about 0.35 pp.

Evidence: protocol:a74a42fceb8b6deef18f872206ae101d679e3707eeb54b8953c098d19085b6d9.

## setup_explore_fullbatch

Exploration full-batch runs train depth 2 at hidden 128 with full-batch SGD at momentum 0.9 and weight decay 0.0001 for 120 epochs, one step per epoch.

Evidence: exp_0001, exp_0002, c63fac48ea43687b487e0195ddb27c90212e634b23318d6eed2ce207176f3ec9.

## setup_explore_identity

The exploration identity path reparametrises each square hidden Linear as W_eff = I + W'. This adds no parameters, and the reparametrisation is folded back into plain weights after training so the frozen scorer evaluates the trained function.

Evidence: exp_0001, c63fac48ea43687b487e0195ddb27c90212e634b23318d6eed2ce207176f3ec9.

## setup_explore_lr01_scripts

exp_0007 runs train.py v3 with SKIP hard-coded on, and exp_0008 runs v1 with a skip flag defaulting to 1. Both run at lr 0.1 and otherwise perform the same computation.

Evidence: exp_0007, exp_0008, 378eed0d7be2b0f7ec5b01f39ead6ba5e443f83f06b1acb0350476a085de2d84, 2d3ad342a13786a9dfd5d6806b428378cbbf96f1332df3eb99d08403e21d1126.

## setup_minibatch

Exploration minibatch runs use SGD with momentum 0.9 and cross-entropy loss at batch 64. Batches are reshuffled each epoch by a torch Generator seeded with the run seed, under a linear warmup then cosine decay, with warmup_frac 0.03 or 0.05.

Evidence: exp_0013, exp_0017, ef2138a3788b76f5816d39fb57bb38cd9adc88856c150fcafa41c76e60f3b7f0, 6e425c80444bddd8de86612fae0098041f15681f62d8a827133022da325d9fa2.

## setup_explore_seeding

Exploration runs seed torch and numpy with the run seed and use default PyTorch Linear initialisation via build_model.

Evidence: c63fac48ea43687b487e0195ddb27c90212e634b23318d6eed2ce207176f3ec9, 6e425c80444bddd8de86612fae0098041f15681f62d8a827133022da325d9fa2.

## setup_explore_fidelity

Exploration fidelities: smoke uses seed 0 on fold 0, verify uses seeds 0-2 on folds 0-2, and full uses seeds 0-4 on folds 0-4.

Evidence: exp_0001, exp_0004, exp_0005.

## setup_wd_width_arms

The weight-decay and width arms all use the warmup-0.05 minibatch script: exp_0024 at hidden 128 with weight_decay 0.0004, exp_0023 at hidden 128 with 0.0004, exp_0021 at hidden 256 with 0.0004, and exp_0022 at hidden 256 with 0.0003.

Evidence: exp_0024, exp_0023, exp_0021, exp_0022.

## setup_explore_lr

Full-batch exploration runs exp_0001, exp_0002, exp_0004 and exp_0005 used lr 0.05. Runs exp_0003, exp_0006, exp_0007 and exp_0008 used lr 0.1.

Evidence: exp_0001, exp_0002, exp_0004, exp_0005, exp_0003, exp_0006, exp_0007, exp_0008.

## setup_minibatch_config

Minibatch runs exp_0012, exp_0013, exp_0016 and exp_0017 used lr 0.05, depth 2, hidden 128, weight_decay 0.0001 and 120 epochs with the identity path on. The weight-decay and width arms exp_0021 to exp_0024 also used lr 0.05; weight_decay 0.0004 was the highest value run.

Evidence: exp_0012, exp_0013, exp_0016, exp_0017, exp_0021, exp_0022, exp_0023, exp_0024.

## setup_wd_width_config

The weight-decay and width arms exp_0021 to exp_0024 used depth 2 and 120 epochs. Runs exp_0014 and exp_0015 used lr 0.05, depth 2, hidden 128, weight_decay 0.0001, 120 epochs and warmup_frac 0.03 with the identity path on.

Evidence: exp_0021, exp_0022, exp_0023, exp_0024, exp_0014, exp_0015.

## related_shortcut_theory

This paper proves that, in a stylised two-layer convolutional ResNet, gradient descent with proper normalisation and suitable initialisation avoids a spurious optimum when shortcuts are present. This study measures only whether a matched zero-parameter skip changes final error in an unnormalised shallow MLP; it does not test that optimisation mechanism.

Evidence: arXiv:1909.04653.

## related_shallow_paths

This paper attributes batch normalisation's benefit in deep ResNets to downscaling the residual branch by a factor of about sqrt(depth). This study works at a depth where that factor is near one and uses no normalisation.

Evidence: openalex:W3007689282.

## related_fractalnet

FractalNet matches ResNets without identity connections in ultra-deep convolutional networks. This study asks the related question of whether the identity path matters, but at shallow MLP depth and with a parameter-identical control.

Evidence: arXiv:1605.07648.

## related_ror

RoR adds multilevel shortcuts to deep ResNets and reports accuracy gains without a matched no-shortcut ablation. This study isolates a single zero-parameter skip instead.

Evidence: arXiv:1608.02908.

## related_capsule

This paper applies residual connections to train deeper capsule networks. It is peripheral here, and its record carries no abstract.

Evidence: arXiv:2104.07393.

## related_hourglass

Hourglass MLPs compare skip placement under parameter-matched designs on generative tasks. This study compares skip against no skip on a classification task.

Evidence: arXiv:2510.01796.

## scope_question

In a shallow ReLU MLP on sklearn-digits, does a zero-parameter identity skip path change final validation error relative to a weight-identical plain MLP, once parameters, initial weights, minibatch order, step count and a shared learning rate are matched?

Evidence: profile.

## scope_method

A frozen protocol trained a two-block, width-128 MLP with and without a zero-parameter identity skip. The two arms were paired on twenty seeds, with identical initial weights, minibatch order and step count, at each of two shared learning rates. A depth-one default recipe served as the anchor arm. The engine reports per-seed paired differences in validation top-1 error. These are judged descriptively against the registered ±0.5 pp prediction. Earlier exploration runs come first in time and are reported as exploration only: single-seed full-batch skip-versus-plain contrasts, a switch to minibatch SGD, and weight-decay and width changes. No new experiments were run.

Evidence: profile.

## limit_1

The engine computed only mean paired differences. No TOST, no 90% interval and no bootstrap were computed, and the protocol used twenty seeds rather than the registered thirty. The verdict is therefore descriptive and cannot declare equivalence at either learning rate.

Evidence: scope limitation.

## limit_2

The validation fold has about 288 images, so one image moves error by about 0.35 pp. Many paired differences are exact ties or single-image changes, which is close to the registered margin.

Evidence: scope limitation.

## limit_3

Residual arms are scored through an exported width-256 Linear/ReLU stack that re-expresses the trained width-128 residual network. Equivalence is asserted in-script, but float summation order may flip rare ties.

Evidence: scope limitation.

## limit_4

The DIAG logs are not in the dossier: init and data-order hashes, step counts, skip activation, export max_abs_diff and activation norms. The protocol's void conditions therefore cannot be checked from the record.

Evidence: scope limitation.

## limit_5

No per-epoch validation curves exist, so speed cannot be separated from endpoint error. The optimisation-speed explanation remains a hypothesis. A competing explanation is that the plain arm is under-converged at the lower learning rate.

Evidence: scope limitation.

## limit_6

Seeds and folds are coupled: each seed is paired with a fold in a fixed cycle, so seed and fold effects cannot be separated and the twenty observations may be dependent.

Evidence: scope limitation.

## limit_7

Within the protocol, the residual arm differs from the plain arm in its initial function, because the skip is active from step zero, and not only in its trajectory. Learning rate was shared rather than tuned per arm.

Evidence: scope limitation.

## limit_8

The exploration identity path sits inside the ReLU as a weight reparametrisation (W = I + W'). It is not the protocol's additive skip, and weight decay pulls its weights toward identity rather than zero.

Evidence: scope limitation.

## limit_9

The two seed-zero identity-path runs at the higher full-batch learning rate agree because the code is identical and deterministic. Neither script logs parametrisation registration; that it took effect is inferred from the code and from the lower-rate difference.

Evidence: scope limitation.

## limit_10

The minibatch-versus-full-batch gain is descriptive across five paired seed/fold pairs with no significance test. No full-batch control with a matched step count separates gradient noise from step count.

Evidence: scope limitation.

## limit_11

The weight-decay per-seed changes match warmup-only changes in size. No training error was logged, so the overfitting premise behind the weight-decay change is untested.

Evidence: scope limitation.

## limit_12

Width is isolated only at weight decay 0.0004, on three seeds (exp_0021 versus exp_0023). exp_0022 changes width and weight decay jointly, and no five-seed width result exists.

Evidence: scope limitation.

## limit_13

Identical per-seed scores at weight decay 0.0003 and 0.0004 may be coincidences of the coarse metric. No saved artifact config was checked to confirm which weight decay was applied.

Evidence: scope limitation.

## limit_14

Results come from one dataset, one depth and one width family. Nothing here generalises to deeper networks, other datasets or normalised residual networks.

Evidence: scope limitation.

## limit_15

Current engine source is not proof of historical execution. Legacy exploration records may lack code, data or environment snapshots.

Evidence: scope limitation.

## limit_16

The record does not count runs that failed, were killed under the time budget, timed out or were voided. It lists each recorded run as kept or discarded and gives the training attempts of each protocol arm; nothing further can be recovered.

Evidence: scope limitation.

## limit_17

The two arms differ in learning-rate sensitivity, so the path and the learning rate interact. The registered rule, which attributes an effect at one learning rate to learning-rate sensitivity, cannot cleanly separate the two.

Evidence: scope limitation.

