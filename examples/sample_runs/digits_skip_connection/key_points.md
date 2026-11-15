# Key points — Does a Zero-Parameter Identity Skip Change Final Error in a Shallow MLP? A Matched-Arm Test on sklearn-digits

_Question: In a shallow ReLU MLP on sklearn-digits, does a zero-parameter identity skip path change final validation error relative to a weight-identical plain MLP, once parameters, initial weights, minibatch order, step count and a shared learning rate are matched?_

## Abstract

1. *context* — In a shallow ReLU MLP on sklearn-digits, does a zero-parameter identity skip path change final validation error relative to a weight-identical plain MLP, once parameters, initial weights, minibatch order, step count and a shared learning rate are matched?  
   evidence: `scope_question`
2. *method* — A frozen protocol trained a two-block, width-128 MLP with and without a zero-parameter identity skip. The two arms were paired on twenty seeds, with identical initial weights, minibatch order and step count, at each of two shared learning rates. A depth-one default recipe served as the anchor arm. The engine reports per-seed paired differences in validation top-1 error. These are judged descriptively against the registered ±0.5 pp prediction. Earlier exploration runs come first in time and are reported as exploration only: single-seed full-batch skip-versus-plain contrasts, a switch to minibatch SGD, and weight-decay and width changes. No new experiments were run.  
   evidence: `scope_method`
3. *result* — At the lower shared learning rate, over twenty paired seeds, the plain arm scored 3.6363 and the identity-skip arm 2.906. The paired difference (plain minus identity) has mean 0.73037, median 0.6945 and seed SD 0.562533. The identity arm was lower on seventeen seeds, tied on three and higher on none.  
   evidence: `claim_protocol_lr002`
4. *result* — At the higher shared learning rate, over twenty paired seeds, the plain arm scored 2.697 and the identity-skip arm 2.5756. The paired difference (plain minus identity) has mean 0.121405, median 0.0 and seed SD 0.532522. The identity arm was lower on seven seeds, tied on ten and higher on three.  
   evidence: `claim_protocol_lr005`
5. *comparison* — Moving from the lower to the higher shared learning rate lowered error more in the plain arm than in the identity arm. The plain-arm paired difference (higher minus lower rate) has mean -0.93936, with eighteen seeds lower, two tied and none higher. The identity-arm difference has mean -0.330395, with fourteen lower, three tied and three higher. All four protocol arms scored below the depth-one full-batch reference arm at 4.2454.  
   evidence: `claim_lr_sensitivity`
6. *result* — Judged by the registered rule, the protocol neither supports nor falsifies H1. At the higher learning rate the mean paired difference lies inside the registered margin, but the engine recorded no interval, so equivalence is not established. At the lower learning rate the mean paired difference already exceeds the margin, so the registered prediction fails there. The registered one-rate attribution cannot be applied, because it presupposes an interval that was not computed.  
   evidence: `claim_protocol_verdict`

## Introduction

1. *context* — In a shallow ReLU MLP on sklearn-digits, does a zero-parameter identity skip path change final validation error relative to a weight-identical plain MLP, once parameters, initial weights, minibatch order, step count and a shared learning rate are matched?  
   evidence: `scope_question`
2. *challenge* — FractalNet matches ResNets without identity connections in ultra-deep convolutional networks. This study asks the related question of whether the identity path matters, but at shallow MLP depth and with a parameter-identical control.  
   evidence: `related_fractalnet`
3. *challenge* — This paper attributes batch normalisation's benefit in deep ResNets to downscaling the residual branch by a factor of about sqrt(depth). This study works at a depth where that factor is near one and uses no normalisation.  
   evidence: `related_shallow_paths`
4. *challenge* — RoR adds multilevel shortcuts to deep ResNets and reports accuracy gains without a matched no-shortcut ablation. This study isolates a single zero-parameter skip instead.  
   evidence: `related_ror`
5. *challenge* — Hourglass MLPs compare skip placement under parameter-matched designs on generative tasks. This study compares skip against no skip on a classification task.  
   evidence: `related_hourglass`
6. *method* — A frozen protocol trained a two-block, width-128 MLP with and without a zero-parameter identity skip. The two arms were paired on twenty seeds, with identical initial weights, minibatch order and step count, at each of two shared learning rates. A depth-one default recipe served as the anchor arm. The engine reports per-seed paired differences in validation top-1 error. These are judged descriptively against the registered ±0.5 pp prediction. Earlier exploration runs come first in time and are reported as exploration only: single-seed full-batch skip-versus-plain contrasts, a switch to minibatch SGD, and weight-decay and width changes. No new experiments were run.  
   evidence: `scope_method`
7. *result* — At the lower shared learning rate, over twenty paired seeds, the plain arm scored 3.6363 and the identity-skip arm 2.906. The paired difference (plain minus identity) has mean 0.73037, median 0.6945 and seed SD 0.562533. The identity arm was lower on seventeen seeds, tied on three and higher on none.  
   evidence: `claim_protocol_lr002`
8. *result* — At the higher shared learning rate, over twenty paired seeds, the plain arm scored 2.697 and the identity-skip arm 2.5756. The paired difference (plain minus identity) has mean 0.121405, median 0.0 and seed SD 0.532522. The identity arm was lower on seven seeds, tied on ten and higher on three.  
   evidence: `claim_protocol_lr005`
9. *comparison* — Moving from the lower to the higher shared learning rate lowered error more in the plain arm than in the identity arm. The plain-arm paired difference (higher minus lower rate) has mean -0.93936, with eighteen seeds lower, two tied and none higher. The identity-arm difference has mean -0.330395, with fourteen lower, three tied and three higher. All four protocol arms scored below the depth-one full-batch reference arm at 4.2454.  
   evidence: `claim_lr_sensitivity`
10. *result* — Judged by the registered rule, the protocol neither supports nor falsifies H1. At the higher learning rate the mean paired difference lies inside the registered margin, but the engine recorded no interval, so equivalence is not established. At the lower learning rate the mean paired difference already exceeds the margin, so the registered prediction fails there. The registered one-rate attribution cannot be applied, because it presupposes an interval that was not computed.  
   evidence: `claim_protocol_verdict`
11. *result* — With the identity path held fixed, reshuffled minibatch SGD with warmup and cosine decay gave five-seed means of 2.7841 and 2.7838 for two warmup fractions, against 3.758 for full-batch training. Every seed was lower in both minibatch runs. Three-seed runs of the two minibatch implementations scored 2.5495 and 2.5491. The change bundles step count, gradient noise and learning-rate schedule, so it cannot be attributed to any one of them.  
   evidence: `claim_minibatch_vs_fullbatch`

## Related Work

1. *context* — This paper attributes batch normalisation's benefit in deep ResNets to downscaling the residual branch by a factor of about sqrt(depth). This study works at a depth where that factor is near one and uses no normalisation.  
   evidence: `related_shallow_paths`
2. *context* — This paper proves that, in a stylised two-layer convolutional ResNet, gradient descent with proper normalisation and suitable initialisation avoids a spurious optimum when shortcuts are present. This study measures only whether a matched zero-parameter skip changes final error in an unnormalised shallow MLP; it does not test that optimisation mechanism.  
   evidence: `related_shortcut_theory`
3. *comparison* — FractalNet matches ResNets without identity connections in ultra-deep convolutional networks. This study asks the related question of whether the identity path matters, but at shallow MLP depth and with a parameter-identical control.  
   evidence: `related_fractalnet`
4. *comparison* — RoR adds multilevel shortcuts to deep ResNets and reports accuracy gains without a matched no-shortcut ablation. This study isolates a single zero-parameter skip instead.  
   evidence: `related_ror`
5. *comparison* — Hourglass MLPs compare skip placement under parameter-matched designs on generative tasks. This study compares skip against no skip on a classification task.  
   evidence: `related_hourglass`
6. *context* — This paper applies residual connections to train deeper capsule networks. It is peripheral here, and its record carries no abstract.  
   evidence: `related_capsule`

## Setup and Protocol

1. *method* — Data are sklearn-digits 8x8 images scaled by 1/16. A fixed stratified 20% test split is held out, and the other 80% is cut into 5 stratified folds; the assigned fold is the validation split and the rest is training data.  
   evidence: `setup_data`
2. *method* — Each validation fold has about 288 images, so one misclassified image is about 0.35 pp.  
   evidence: `setup_val_size`
3. *method* — The metric is val_top1_err: top-1 error in percent on the validation fold. It is computed by the frozen scorer, which rebuilds a Linear/ReLU stack from the saved state_dict.  
   evidence: `setup_metric`
4. *context* — The profile records a logistic-regression baseline of 3.8 percent validation error.  
   evidence: `setup_baseline`
5. *method* — Protocol architecture: input projection 64->128 with ReLU, then two blocks h=ReLU(W h+b), then a 10-way head. This equals build_model with depth=3 and hidden=128.  
   evidence: `setup_protocol_arch`
6. *method* — Arm plain_lr005 runs train.py v7 with SKIP=False at lr 0.05: the plain block h=ReLU(W h+b).  
   evidence: `setup_arm_plain_lr005`
7. *method* — Arm plain_lr002 runs train.py v7 with SKIP=False and differs from plain_lr005 only in using lr 0.02.  
   evidence: `setup_arm_plain_lr002`
8. *method* — Arm resid_lr005 runs train.py v8, which differs from v7 only in SKIP=True, so h_{l+1} = h_l + ReLU(W_l h_l + b_l), at lr 0.05. Weights are exported as an equivalent width-256 stack, and the run asserts max_abs_diff < 1e-3.  
   evidence: `setup_arm_resid_lr005`
9. *method* — Arm resid_lr002 runs train.py v8 with SKIP=True and differs from resid_lr005 only in using lr 0.02.  
   evidence: `setup_arm_resid_lr002`
10. *method* — The reference arm runs the unmodified train.py v2 at depth 1 and hidden 128, full-batch, with lr 0.05 for 120 epochs. It anchors the paired differences and is not part of the skip contrast.  
   evidence: `setup_arm_reference`
11. *method* — Protocol training: minibatch SGD with batch 64, momentum 0.9, weight decay 0.0001 and cross-entropy loss for 120 epochs, at a constant learning rate with no schedule.  
   evidence: `setup_protocol_optim`
12. *method* — Protocol seeding: torch and numpy are seeded with the seed index, and weights use default PyTorch Linear initialisation via build_model. Per-epoch permutations come from a torch Generator seeded with 1000003 plus the seed, identical across paired arms.  
   evidence: `setup_protocol_seeding`
13. *method* — Every protocol arm ran seeds 1 to 20 at verify fidelity with a 180-second budget per run. Folds cycle through 1, 2, 3, 4, 0, and each seed index uses the same fold in every arm.  
   evidence: `setup_protocol_seeds`
14. *method* — The selected idea pre-registered a paired TOST at α = 0.05 with a 0.5 pp margin on final error over 30 paired seeds. Per-arm lr was to be chosen from {0.01, 0.02, 0.05, 0.1, 0.2} on a 20% inner split, with a shared-lr robustness sweep.  
   evidence: `setup_preregistered_plan`
15. *method* — Registered prediction: at lr=0.05, the paired mean difference (resid_lr005 - plain_lr005) in final val_top1_err has magnitude under 0.5 pp and its 90% interval lies inside [-0.5, +0.5] pp. The same holds for the lr=0.02 pair, and both arms reach roughly 2-4% error.  
   evidence: `setup_protocol_prediction`
16. *method* — Registered outcome rule: H1 is falsified only if the paired 90% interval lies entirely outside [-0.5, +0.5] pp at both lr=0.05 and lr=0.02. An effect at one lr only is attributed to lr sensitivity. An interval straddling a margin boundary is inconclusive.  
   evidence: `setup_protocol_rule`
17. *limitation* — The run protocol dropped per-arm lr tuning and reduced the shared-lr grid to two values, 0.05 and 0.02. It used 20 seeds instead of 30 and computed descriptive paired mean differences instead of TOST.  
   evidence: `setup_protocol_deviations`

## Exploration Runs (Earlier in Time)

1. *method* — Exploration full-batch runs train depth 2 at hidden 128 with full-batch SGD at momentum 0.9 and weight decay 0.0001 for 120 epochs, one step per epoch.  
   evidence: `setup_explore_fullbatch`
2. *method* — The exploration identity path reparametrises each square hidden Linear as W_eff = I + W'. This adds no parameters, and the reparametrisation is folded back into plain weights after training so the frozen scorer evaluates the trained function.  
   evidence: `setup_explore_identity`
3. *method* — Full-batch exploration runs exp_0001, exp_0002, exp_0004 and exp_0005 used lr 0.05. Runs exp_0003, exp_0006, exp_0007 and exp_0008 used lr 0.1.  
   evidence: `setup_explore_lr`
4. *method* — exp_0007 runs train.py v3 with SKIP hard-coded on, and exp_0008 runs v1 with a skip flag defaulting to 1. Both run at lr 0.1 and otherwise perform the same computation.  
   evidence: `setup_explore_lr01_scripts`
5. *method* — Exploration runs seed torch and numpy with the run seed and use default PyTorch Linear initialisation via build_model.  
   evidence: `setup_explore_seeding`
6. *method* — Exploration fidelities: smoke uses seed 0 on fold 0, verify uses seeds 0-2 on folds 0-2, and full uses seeds 0-4 on folds 0-4.  
   evidence: `setup_explore_fidelity`
7. *method* — Exploration minibatch runs use SGD with momentum 0.9 and cross-entropy loss at batch 64. Batches are reshuffled each epoch by a torch Generator seeded with the run seed, under a linear warmup then cosine decay, with warmup_frac 0.03 or 0.05.  
   evidence: `setup_minibatch`
8. *method* — Minibatch runs exp_0012, exp_0013, exp_0016 and exp_0017 used lr 0.05, depth 2, hidden 128, weight_decay 0.0001 and 120 epochs with the identity path on. The weight-decay and width arms exp_0021 to exp_0024 also used lr 0.05; weight_decay 0.0004 was the highest value run.  
   evidence: `setup_minibatch_config`
9. *method* — The weight-decay and width arms all use the warmup-0.05 minibatch script: exp_0024 at hidden 128 with weight_decay 0.0004, exp_0023 at hidden 128 with 0.0004, exp_0021 at hidden 256 with 0.0004, and exp_0022 at hidden 256 with 0.0003.  
   evidence: `setup_wd_width_arms`
10. *method* — The weight-decay and width arms exp_0021 to exp_0024 used depth 2 and 120 epochs. Runs exp_0014 and exp_0015 used lr 0.05, depth 2, hidden 128, weight_decay 0.0001, 120 epochs and warmup_frac 0.03 with the identity path on.  
   evidence: `setup_wd_width_config`

## Results: Matched Protocol

1. *result* — At the lower shared learning rate, over twenty paired seeds, the plain arm scored 3.6363 and the identity-skip arm 2.906. The paired difference (plain minus identity) has mean 0.73037, median 0.6945 and seed SD 0.562533. The identity arm was lower on seventeen seeds, tied on three and higher on none.  
   evidence: `claim_protocol_lr002`
2. *result* — At the higher shared learning rate, over twenty paired seeds, the plain arm scored 2.697 and the identity-skip arm 2.5756. The paired difference (plain minus identity) has mean 0.121405, median 0.0 and seed SD 0.532522. The identity arm was lower on seven seeds, tied on ten and higher on three.  
   evidence: `claim_protocol_lr005`
3. *comparison* — Moving from the lower to the higher shared learning rate lowered error more in the plain arm than in the identity arm. The plain-arm paired difference (higher minus lower rate) has mean -0.93936, with eighteen seeds lower, two tied and none higher. The identity-arm difference has mean -0.330395, with fourteen lower, three tied and three higher. All four protocol arms scored below the depth-one full-batch reference arm at 4.2454.  
   evidence: `claim_lr_sensitivity`
4. *result* — Judged by the registered rule, the protocol neither supports nor falsifies H1. At the higher learning rate the mean paired difference lies inside the registered margin, but the engine recorded no interval, so equivalence is not established. At the lower learning rate the mean paired difference already exceeds the margin, so the registered prediction fails there. The registered one-rate attribution cannot be applied, because it presupposes an interval that was not computed.  
   evidence: `claim_protocol_verdict`

## Results: Exploration

1. *result* — The single-seed full-batch exploration contrasts point the same way or not at all. At the lower exploration learning rate, the identity-path arm scored 3.4722 against 5.2083 for the plain arm. At the higher exploration learning rate, both arms scored 4.5139. That is one seed-zero contrast favouring the identity path and one exact tie, and neither is replicated, so these runs leave the skip effect unresolved.  
   evidence: `claim_skip_effect_unresolved`
2. *result* — At the higher full-batch exploration learning rate on seed zero, the plain arm and two identity-path runs all scored 4.5139. The two identity-path runs use scripts that perform the same computation and differ only in how the skip is switched on, so their agreement shows deterministic code, not independent replication. The plain arm's three-seed mean at this rate was 3.8235. No identity-path run with more than one seed exists at this rate.  
   evidence: `claim_fullbatch_lr01_seed0`
3. *comparison* — With the identity path held fixed, reshuffled minibatch SGD with warmup and cosine decay gave five-seed means of 2.7841 and 2.7838 for two warmup fractions, against 3.758 for full-batch training. Every seed was lower in both minibatch runs. Three-seed runs of the two minibatch implementations scored 2.5495 and 2.5491. The change bundles step count, gradient noise and learning-rate schedule, so it cannot be attributed to any one of them.  
   evidence: `claim_minibatch_vs_fullbatch`
4. *result* — Raising weight decay to the top of its range in the minibatch regime gave a five-seed mean of 2.575 against 2.7838. Seeds zero, two and four fell (2.4306 to 2.0833, 2.439 to 2.0906, 4.5296 to 4.1812); seeds one and three were unchanged. Changing only the warmup fraction moves seed zero between 2.4306 and 2.0833 and seed two between 2.439 and 2.7875. Those are changes of the same size, so the drop cannot be attributed to weight decay.  
   evidence: `claim_weight_decay`
5. *result* — At matched weight decay on three seeds, the wider network scored 2.3176 against 2.3172 for the narrower one. It was lower on seed zero (1.7361 versus 2.0833), equal on seed one (2.7778) and higher on seed two (2.439 versus 2.0906). The width contrast is mixed with no net change. A wider run that also lowered weight decay gave identical per-seed scores of 1.7361, 2.7778 and 2.439.  
   evidence: `claim_width`

## Discussion

1. *context* — One explanation fits the pattern: the identity path speeds optimisation under the fixed step budget, so its advantage is largest where the plain arm trains more slowly, at the lower learning rate. The same account would explain why the identity arm is less sensitive to learning rate. This explanation is untested. No validation curves or training-loss logs are in the record, and a slower-converging plain arm and a genuine endpoint effect would produce the same final-error data.  
   evidence: `claim_mechanism_speed`
2. *limitation* — The two arms differ in learning-rate sensitivity, so the path and the learning rate interact. The registered rule, which attributes an effect at one learning rate to learning-rate sensitivity, cannot cleanly separate the two.  
   evidence: `limit_17`

## Limitations

1. *limitation* — The engine computed only mean paired differences. No TOST, no 90% interval and no bootstrap were computed, and the protocol used twenty seeds rather than the registered thirty. The verdict is therefore descriptive and cannot declare equivalence at either learning rate.  
   evidence: `limit_1`
2. *limitation* — The validation fold has about 288 images, so one image moves error by about 0.35 pp. Many paired differences are exact ties or single-image changes, which is close to the registered margin.  
   evidence: `limit_2`
3. *limitation* — Residual arms are scored through an exported width-256 Linear/ReLU stack that re-expresses the trained width-128 residual network. Equivalence is asserted in-script, but float summation order may flip rare ties.  
   evidence: `limit_3`
4. *limitation* — The DIAG logs are not in the dossier: init and data-order hashes, step counts, skip activation, export max_abs_diff and activation norms. The protocol's void conditions therefore cannot be checked from the record.  
   evidence: `limit_4`
5. *limitation* — No per-epoch validation curves exist, so speed cannot be separated from endpoint error. The optimisation-speed explanation remains a hypothesis. A competing explanation is that the plain arm is under-converged at the lower learning rate.  
   evidence: `limit_5`
6. *limitation* — Seeds and folds are coupled: each seed is paired with a fold in a fixed cycle, so seed and fold effects cannot be separated and the twenty observations may be dependent.  
   evidence: `limit_6`
7. *limitation* — Within the protocol, the residual arm differs from the plain arm in its initial function, because the skip is active from step zero, and not only in its trajectory. Learning rate was shared rather than tuned per arm.  
   evidence: `limit_7`
8. *limitation* — The exploration identity path sits inside the ReLU as a weight reparametrisation (W = I + W'). It is not the protocol's additive skip, and weight decay pulls its weights toward identity rather than zero.  
   evidence: `limit_8`
9. *limitation* — The two seed-zero identity-path runs at the higher full-batch learning rate agree because the code is identical and deterministic. Neither script logs parametrisation registration; that it took effect is inferred from the code and from the lower-rate difference.  
   evidence: `limit_9`
10. *limitation* — The minibatch-versus-full-batch gain is descriptive across five paired seed/fold pairs with no significance test. No full-batch control with a matched step count separates gradient noise from step count.  
   evidence: `limit_10`
11. *limitation* — The weight-decay per-seed changes match warmup-only changes in size. No training error was logged, so the overfitting premise behind the weight-decay change is untested.  
   evidence: `limit_11`
12. *limitation* — Width is isolated only at weight decay 0.0004, on three seeds (exp_0021 versus exp_0023). exp_0022 changes width and weight decay jointly, and no five-seed width result exists.  
   evidence: `limit_12`
13. *limitation* — Identical per-seed scores at weight decay 0.0003 and 0.0004 may be coincidences of the coarse metric. No saved artifact config was checked to confirm which weight decay was applied.  
   evidence: `limit_13`
14. *limitation* — Results come from one dataset, one depth and one width family. Nothing here generalises to deeper networks, other datasets or normalised residual networks.  
   evidence: `limit_14`
15. *limitation* — Current engine source is not proof of historical execution. Legacy exploration records may lack code, data or environment snapshots.  
   evidence: `limit_15`
16. *limitation* — The record does not count runs that failed, were killed under the time budget, timed out or were voided. It lists each recorded run as kept or discarded and gives the training attempts of each protocol arm; nothing further can be recovered.  
   evidence: `limit_16`

## Conclusion

1. *context* — In a shallow ReLU MLP on sklearn-digits, does a zero-parameter identity skip path change final validation error relative to a weight-identical plain MLP, once parameters, initial weights, minibatch order, step count and a shared learning rate are matched?  
   evidence: `scope_question`
2. *result* — At the lower shared learning rate, over twenty paired seeds, the plain arm scored 3.6363 and the identity-skip arm 2.906. The paired difference (plain minus identity) has mean 0.73037, median 0.6945 and seed SD 0.562533. The identity arm was lower on seventeen seeds, tied on three and higher on none.  
   evidence: `claim_protocol_lr002`
3. *result* — At the higher shared learning rate, over twenty paired seeds, the plain arm scored 2.697 and the identity-skip arm 2.5756. The paired difference (plain minus identity) has mean 0.121405, median 0.0 and seed SD 0.532522. The identity arm was lower on seven seeds, tied on ten and higher on three.  
   evidence: `claim_protocol_lr005`
4. *comparison* — Moving from the lower to the higher shared learning rate lowered error more in the plain arm than in the identity arm. The plain-arm paired difference (higher minus lower rate) has mean -0.93936, with eighteen seeds lower, two tied and none higher. The identity-arm difference has mean -0.330395, with fourteen lower, three tied and three higher. All four protocol arms scored below the depth-one full-batch reference arm at 4.2454.  
   evidence: `claim_lr_sensitivity`
5. *result* — Judged by the registered rule, the protocol neither supports nor falsifies H1. At the higher learning rate the mean paired difference lies inside the registered margin, but the engine recorded no interval, so equivalence is not established. At the lower learning rate the mean paired difference already exceeds the margin, so the registered prediction fails there. The registered one-rate attribution cannot be applied, because it presupposes an interval that was not computed.  
   evidence: `claim_protocol_verdict`
6. *limitation* — No per-epoch validation curves exist, so speed cannot be separated from endpoint error. The optimisation-speed explanation remains a hypothesis. A competing explanation is that the plain arm is under-converged at the lower learning rate.  
   evidence: `limit_5`
7. *future_work* — The engine computed only mean paired differences. No TOST, no 90% interval and no bootstrap were computed, and the protocol used twenty seeds rather than the registered thirty. The verdict is therefore descriptive and cannot declare equivalence at either learning rate.  
   evidence: `limit_1`

## Discussion

### Round 1 — ✅ no blocking issue left
- **Skeptic**: accept
  - (suggestion) [Conclusion – limit_1 tagged (future_work)] limit_1 records missing TOST, missing intervals and 20 rather than 30 seeds. It is a limitation, not a future-work item. Labelling it future_work implies the missing analysis is a planned extension rather than a gap in the current evidence. → *Re-tag limit_1 in the Conclusion as (limitation), or drop it there, since claim_protocol_verdict already states the descriptive-only scope.*
  - (suggestion) [Results: Exploration – claim_skip_effect_unresolved / claim_fullbatch_lr01_seed0] The exploration identity path is a weight reparametrisation inside the ReLU (W = I + W'), not the protocol's additive skip. limit_8 says so, but it appears only in the Limitations list. A reader of the exploration results may treat them as the same intervention as the protocol arms. → *Place limit_8 (and limit_9 for the lr 0.1 duplicate runs) directly after the exploration skip results, or at the start of the Exploration Runs section.*
  - (suggestion) [Introduction – claim_minibatch_vs_fullbatch] An exploration-only result is placed in the Introduction among the headline protocol results. It concerns the optimiser regime, not the skip question. This risks presenting it as a contribution on par with the matched protocol. → *Drop it from the Introduction, or move it after a sentence that marks it as exploration context; keep it in Results: Exploration.*
  - (suggestion) [Abstract – claim_lr_sensitivity] The learning-rate interaction appears in the Abstract without the caveat that it undermines the registered one-rate attribution. That caveat is limit_17, which appears only in the Discussion. → *Consider pairing claim_lr_sensitivity with limit_17 wherever both are used, or drop claim_lr_sensitivity from the Abstract for brevity.*
  - (suggestion) [Sign convention – claim_protocol_lr002/lr005 vs setup_protocol_prediction] The results report plain minus identity. The registered prediction is defined as resid minus plain. The verdict uses magnitude, so nothing is wrong, but readers may stumble over the flip. → *Author note: consider stating the sign convention once in Setup (e.g. alongside setup_protocol_prediction).*
  - (suggestion) [Whole plan – repetition] claim_protocol_lr002, claim_protocol_lr005, claim_lr_sensitivity and claim_protocol_verdict each appear verbatim four times: Abstract, Introduction, Results and Conclusion. scope_question and scope_method also repeat. → *Keep the full results in Results and the Abstract. Trim the Introduction to the verdict plus a pointer, and keep the Conclusion to the verdict and the key limitation.*
- **Editor**: accept
  - (suggestion) [Abstract / overall plan (nugget)] The plan has one clear question (scope_question) but never marks a single nugget. The abstract gives four results of equal weight: lr002, lr005, the lr-sensitivity comparison and the verdict. A reader cannot tell which is the takeaway. The evidence supports this one: the skip's advantage is clear at the lower shared lr (17/20 lower, none higher), shrinks to mostly ties at the higher lr, and the registered rule cannot settle H1. → *Name the nugget explicitly, built from claim_protocol_lr002 + claim_lr_sensitivity with claim_protocol_verdict as the qualifier. Order the abstract as question → method → lr002 → lr005 → verdict, and let claim_lr_sensitivity frame the two results rather than stand as a fourth item.*
  - (suggestion) [Abstract, Introduction, Results: Matched Protocol, Conclusion] The same four statements are repeated verbatim in all four sections: claim_protocol_lr002, claim_protocol_lr005, claim_lr_sensitivity and claim_protocol_verdict. scope_question also appears three times and scope_method twice. Read in order, the paper says everything four times. → *Keep the full numeric statements in Results. In the Introduction, keep only claim_protocol_lr002 and claim_protocol_verdict as a preview. In the Conclusion, keep claim_protocol_verdict and limit_5, and drop the repeated numbers.*
  - (suggestion) [Introduction — claim_minibatch_vs_fullbatch] An exploration-only result about the optimiser change, made with the skip held fixed, appears in the Introduction. It does not answer the skip question and pulls attention away from the nugget. → *Remove it from the Introduction. It already appears in Results: Exploration.*
  - (suggestion) [Introduction challenge bullets vs Related Work] related_fractalnet, related_shallow_paths, related_ror and related_hourglass each appear in both the Introduction and Related Work. → *Keep one or two in the Introduction to motivate the matched control (fractalnet, ror). Leave the full set to Related Work.*
  - (suggestion) [Related Work — related_capsule] The statement itself says the work is peripheral and has no abstract. It does not serve the question. → *Drop it.*
  - (suggestion) [Section order: Exploration Runs (Earlier in Time) placed before Results: Matched Protocol] Exploration setup sits between the protocol setup and the protocol results. The reader has to hold the protocol details through ten unrelated setup bullets before seeing the main result. Chronological order is not the reader's order. → *Move the exploration setup into, or just before, Results: Exploration, after the matched-protocol results. Add a transition saying exploration predates the protocol and uses a different skip.*
  - (suggestion) [Results: Exploration — claim_skip_effect_unresolved and claim_fullbatch_lr01_seed0] Both statements report the same 4.5139 tie at the higher full-batch lr. → *Keep claim_skip_effect_unresolved. Either drop claim_fullbatch_lr01_seed0 or place it straight after as the explanation of why the two identity runs agree, so the tie is not stated twice in a row.*
  - (suggestion) [Results: Exploration — placement of limit_8] The exploration identity path (W = I + W', inside the ReLU) is a different intervention from the protocol's additive skip. That caveat appears only in the long Limitations list, so readers may compare exploration skip contrasts directly with protocol results. → *Place limit_8, or reference it, at the start of Results: Exploration.*
  - (suggestion) [Results: Matched Protocol — claim_protocol_lr005 (median 0.0, ten ties)] The tie-heavy lr005 result is hard to read without the granularity caveat. setup_val_size is far back in Setup. → *Place limit_2 right after claim_protocol_lr005, or after claim_protocol_verdict.*
  - (suggestion) [Conclusion — limit_1 tagged future_work] limit_1 is a limitation statement, not future work. Labelling it future_work suggests a plan the text does not state. → *Tag it as a limitation, or omit it from the Conclusion since it already appears in Limitations.*
- **Methodologist**: accept
  - (suggestion) [Abstract / Introduction / Results: Matched Protocol / Conclusion (claim_protocol_lr002, claim_protocol_lr005, claim_lr_sensitivity, claim_protocol_verdict)] The same four approved statements appear verbatim in the Abstract, Introduction, Results and Conclusion, and scope_question and scope_method are also repeated. A reader gets the same numbers four times, and the Introduction loses room to motivate the matched design. → *Keep the full statements in Results. In the Abstract keep claim_protocol_lr002, claim_protocol_lr005 and claim_protocol_verdict. In the Introduction and Conclusion use only claim_protocol_verdict, plus claim_lr_sensitivity in the Conclusion, so that each number has one home.*
  - (suggestion) [Abstract] The abstract reports the lower-rate gap (mean 0.73 pp, 17/0/3) with no limitation beside it. The approved limitations that bound that number (limit_1: twenty seeds and no interval; limit_2: one image is about 0.35 pp) appear only later. A reader skimming the abstract could take the lower-rate result as an established effect. → *Add limit_1, or limit_2, to the Abstract after claim_protocol_verdict so that the descriptive status and the metric resolution are visible next to the headline numbers.*
  - (suggestion) [Introduction: claim_minibatch_vs_fullbatch] An exploration-only result, measured with the in-ReLU W = I + W' identity path and a warmup/cosine schedule, sits in the Introduction next to the protocol results. Its caveats are absent there: limit_8 (a different skip form from the protocol's), limit_10 (descriptive, five seeds) and the exploration-only label. Readers may conflate it with the matched protocol. → *Move claim_minibatch_vs_fullbatch out of the Introduction and keep it only in Results: Exploration, where limit_8 and limit_10 can sit nearby. Alternatively, place limit_8 directly after it.*
  - (suggestion) [Setup and Protocol vs Exploration Runs (setup_protocol_arch vs setup_explore_fullbatch / setup_minibatch_config)] The protocol network is described as two blocks, 'build_model with depth=3', while exploration runs are 'depth 2 at hidden 128'. The plan does not reconcile the two depth conventions, so a reader cannot tell whether the exploration networks have one hidden block or two. That matters when exploration contrasts are compared with protocol contrasts. → *Place setup_protocol_arch's build_model mapping next to the first exploration depth statement. Ask the author whether an approved statement can say how exploration 'depth 2' maps to blocks; if none exists, record the mapping as unrecorded.*
  - (suggestion) [Exploration Runs (Earlier in Time)] A per-run wall-clock or time budget is given for protocol arms (180 s) but not for exploration runs. No approved statement records one, so exploration compute is not fully reproducible. → *Exploration-run time budget: if it is unrecorded, say so, citing limit_15 (legacy records may lack snapshots).*
  - (suggestion) [Conclusion: limit_1 labelled future_work] limit_1 states a limitation; it does not propose future work. The label misdescribes its role, although the text itself is accurate. → *Relabel it as limitation.*
  - (suggestion) [Results: Exploration] limit_8, limit_9 and limit_13 bear directly on interpreting the exploration contrasts but appear only in the general Limitations list. limit_8 covers the different skip form, limit_9 covers registration not being logged at lr 0.1, and limit_13 covers the identical scores at weight decay 0.0003 and 0.0004. → *Add a cross-reference: place limit_8 at the start of Results: Exploration, and limit_9 after claim_fullbatch_lr01_seed0, or note that these caveats appear in Limitations.*
  - (suggestion) [Author: setup_arm_reference / claim_lr_sensitivity] The reference arm is said to 'anchor the paired differences', but it is full-batch at depth 1 while the protocol arms are minibatch at two blocks. How it anchors a paired difference is not explained. The comparison 'all four arms below 4.2454' therefore mixes a full-batch versus minibatch change with the depth and skip factors. → *Author: consider clarifying in an approved statement that the reference arm is a sanity anchor only, not a comparator for the skip effect.*

*Every evidence id is a real run, paper, or the baseline; every number was checked against the registry. Approve to draft the prose.*
