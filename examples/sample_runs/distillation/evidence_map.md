# Approved statement bindings

Writing release: `428e387c04b02e2675d835002dd934314ba29d4517988e118838f8274114682a`

This snapshot records the approved scope. Revalidate the live review before further writing.

Raw sources, measurements and input hashes: [approved_evidence.json](approved_evidence.json).

## claim_all_arms_observation

Mean validation invalid rate (%) over five seeds, with seed SD, was:
- off_policy_ref: 14.0 (3.234192)
- pure_on_policy: 9.04 (8.478679)
- fixed_switch_1500: 15.0 (3.896152)
- adaptive_tau25: 12.88 (2.229798)
- yoked_replay: 11.0 (8.866792)
- corrupt_prefix: 12.76 (6.582401)

Evidence: protocol_8198ab82d8c25b04b09e5215_off_policy_ref, protocol_8198ab82d8c25b04b09e5215_pure_on_policy, protocol_8198ab82d8c25b04b09e5215_fixed_switch_1500, protocol_8198ab82d8c25b04b09e5215_adaptive_tau25, protocol_8198ab82d8c25b04b09e5215_yoked_replay, protocol_8198ab82d8c25b04b09e5215_corrupt_prefix.

- `protocol_8198ab82d8c25b04b09e5215_off_policy_ref::val_invalid_rate.full`: value=14.0, std=3.2342; owning evidence `protocol_8198ab82d8c25b04b09e5215_off_policy_ref`.
- `protocol_8198ab82d8c25b04b09e5215_pure_on_policy::val_invalid_rate.full`: value=9.04, std=8.4787; owning evidence `protocol_8198ab82d8c25b04b09e5215_pure_on_policy`.
- `protocol_8198ab82d8c25b04b09e5215_fixed_switch_1500::val_invalid_rate.full`: value=15.0, std=3.8962; owning evidence `protocol_8198ab82d8c25b04b09e5215_fixed_switch_1500`.
- `protocol_8198ab82d8c25b04b09e5215_adaptive_tau25::val_invalid_rate.full`: value=12.88, std=2.2298; owning evidence `protocol_8198ab82d8c25b04b09e5215_adaptive_tau25`.
- `protocol_8198ab82d8c25b04b09e5215_yoked_replay::val_invalid_rate.full`: value=11.0, std=8.8668; owning evidence `protocol_8198ab82d8c25b04b09e5215_yoked_replay`.
- `protocol_8198ab82d8c25b04b09e5215_corrupt_prefix::val_invalid_rate.full`: value=12.76, std=6.5824; owning evidence `protocol_8198ab82d8c25b04b09e5215_corrupt_prefix`.
## claim_adaptive_vs_offpolicy

The adaptive-switch arm had a lower mean val_invalid_rate (12.88) than off_policy_ref (14.0), a paired mean difference of -1.12. The per-seed paired differences were small and mixed in sign: four of the five seed–fold pairs were within about one point of zero, and one of those four was positive. Most of the mean improvement comes from the single pair where off_policy_ref had its highest value (18.4). The improvement is small relative to the seed–fold spread and is not robust across pairs.

Evidence: protocol_8198ab82d8c25b04b09e5215_adaptive_tau25, protocol_8198ab82d8c25b04b09e5215_off_policy_ref, protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

- `protocol_8198ab82d8c25b04b09e5215_adaptive_tau25::val_invalid_rate.full`: value=12.88, std=2.2298; owning evidence `protocol_8198ab82d8c25b04b09e5215_adaptive_tau25`.
- `protocol_8198ab82d8c25b04b09e5215_off_policy_ref::val_invalid_rate.full`: value=14.0, std=3.2342; owning evidence `protocol_8198ab82d8c25b04b09e5215_off_policy_ref`.
- `protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6::val_invalid_rate.difference.adaptive_tau25`: value=-1.12, std=None; owning evidence `protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6`.
## claim_adaptive_not_below_pure_on

The adaptive-switch arm did not beat pure_on_policy on mean val_invalid_rate: pure_on_policy averaged 9.04 and the adaptive arm averaged 12.88. The pre-registered prediction that the adaptive switch would be below both pure regimes by a margin therefore fails for the on-policy endpoint. The adaptive arm had the narrower spread across seed–fold pairs.

Evidence: protocol_8198ab82d8c25b04b09e5215_adaptive_tau25, protocol_8198ab82d8c25b04b09e5215_pure_on_policy.

- `protocol_8198ab82d8c25b04b09e5215_adaptive_tau25::val_invalid_rate.full`: value=12.88, std=2.2298; owning evidence `protocol_8198ab82d8c25b04b09e5215_adaptive_tau25`.
- `protocol_8198ab82d8c25b04b09e5215_pure_on_policy::val_invalid_rate.full`: value=9.04, std=8.4787; owning evidence `protocol_8198ab82d8c25b04b09e5215_pure_on_policy`.
## claim_pure_on_high_variance

pure_on_policy had a paired mean difference of -4.96 against off_policy_ref, but the per-pair pattern was heterogeneous. On three of five seed–fold pairs it was much better than off-policy, reaching values of 4.6, 2.6 and 4.2. On one pair it was about equal. On the remaining pair it was clearly worse (23.2 against the reference's 15.8). The direction is not uniform and the spread is large, so the sign of the mean difference is not established at this sample size.

Evidence: protocol_8198ab82d8c25b04b09e5215_pure_on_policy, protocol_8198ab82d8c25b04b09e5215_off_policy_ref, protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

- `protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6::val_invalid_rate.difference.pure_on_policy`: value=-4.96, std=None; owning evidence `protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6`.
- `protocol_8198ab82d8c25b04b09e5215_pure_on_policy::val_invalid_rate.full`: value=9.04, std=8.4787; owning evidence `protocol_8198ab82d8c25b04b09e5215_pure_on_policy`.
- `protocol_8198ab82d8c25b04b09e5215_off_policy_ref::val_invalid_rate.full`: value=14.0, std=3.2342; owning evidence `protocol_8198ab82d8c25b04b09e5215_off_policy_ref`.
## claim_fixed_switch_worse

The fixed-step switch arm had the highest mean val_invalid_rate of all arms (15.0), a paired mean difference of 1.0 against off_policy_ref. The paired differences were mixed in sign: three of the five seed–fold pairs were worse than the reference and two were better, and the mean difference is within the seed–fold spread. The direction of the fixed-step arm relative to pure off-policy is therefore not established. The adaptive arm (12.88) had a lower mean than the fixed-step arm, which formally satisfies the prediction that adaptive is no worse than the fixed switch. This contrast confounds trigger responsiveness with switch timing: according to self-reported diagnostics, the adaptive switches fired earlier than the fixed step on every seed, so the adaptive arm spent more of training on-policy.

Evidence: protocol_8198ab82d8c25b04b09e5215_fixed_switch_1500, protocol_8198ab82d8c25b04b09e5215_adaptive_tau25, protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

- `protocol_8198ab82d8c25b04b09e5215_fixed_switch_1500::val_invalid_rate.full`: value=15.0, std=3.8962; owning evidence `protocol_8198ab82d8c25b04b09e5215_fixed_switch_1500`.
- `protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6::val_invalid_rate.difference.fixed_switch_1500`: value=1.0, std=None; owning evidence `protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6`.
- `protocol_8198ab82d8c25b04b09e5215_adaptive_tau25::val_invalid_rate.full`: value=12.88, std=2.2298; owning evidence `protocol_8198ab82d8c25b04b09e5215_adaptive_tau25`.
## claim_trigger_exercised

According to self-reported diagnostics, the adaptive trigger fired before the forced-switch cap on all five seed–fold pairs, and no switch was forced. On-policy batches occurred only after the reported switch step. The adaptive mechanism was therefore exercised in this run, rather than every switch coming from the cap.

Evidence: protocol_8198ab82d8c25b04b09e5215_adaptive_tau25.

## claim_hypothesis_not_supported

The pre-registered falsifying outcome was met. The adaptive-switch arm did not beat the better pure regime: pure_on_policy had the lower mean (9.04 against 12.88), so the adaptive switch was not better than the better pure regime by more than the seed–fold spread. The prediction that the adaptive arm would beat both pure regimes is therefore not supported in this configuration. The two controls meant to test why a trigger might help (yoked replay and corrupted prefixes) did not implement their intended contrasts and support no conclusion either way (see limitations). The conservative reading is that this experiment does not distinguish the switching rules.

Evidence: protocol_8198ab82d8c25b04b09e5215_off_policy_ref, protocol_8198ab82d8c25b04b09e5215_pure_on_policy, protocol_8198ab82d8c25b04b09e5215_fixed_switch_1500, protocol_8198ab82d8c25b04b09e5215_adaptive_tau25, protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

- `protocol_8198ab82d8c25b04b09e5215_pure_on_policy::val_invalid_rate.full`: value=9.04, std=8.4787; owning evidence `protocol_8198ab82d8c25b04b09e5215_pure_on_policy`.
- `protocol_8198ab82d8c25b04b09e5215_adaptive_tau25::val_invalid_rate.full`: value=12.88, std=2.2298; owning evidence `protocol_8198ab82d8c25b04b09e5215_adaptive_tau25`.
## setup_task

Task multimodal-transduction-L6-A8: each input is L=6 symbols from an alphabet of 8. Three outputs are valid: the copy, the reversal, and the reversal incremented mod 8. A mixture of modes counts as invalid.

Evidence: scorer, profile.

## setup_metric

The metric val_invalid_rate is the percentage of greedy student decodes on a validation fold that match none of the valid outputs. Lower is better. Each validation fold holds 500 prompts.

Evidence: scorer, profile.

## setup_models

The student is a frozen-architecture pre-norm causal transformer with d_model 32, 2 layers and 2 heads. The frozen teacher has d_model 96 and 2 layers, trained for 8000 steps on inputs disjoint from the fold pool and the test set.

Evidence: scorer.

## setup_training

Every arm trains for 3000 steps with batch 64 using AdamW at lr 0.003. The loss is forward KL between teacher and student next-token distributions at temperature 1.0 over output positions.

Evidence: train, profile, protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

## setup_seeds_folds

Each arm ran five seeds (zero through four) over five cross-validation folds at full fidelity. Run records list fold indices alongside seed indices, so seed and fold vary together.

Evidence: protocol_8198ab82d8c25b04b09e5215_off_policy_ref, protocol_8198ab82d8c25b04b09e5215_adaptive_tau25, protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

## setup_budget

Each full-fidelity run had a time budget of 1200 seconds.

Evidence: profile, protocol_8198ab82d8c25b04b09e5215_off_policy_ref.

## setup_arm_offpolicy

The off_policy_ref arm runs the unmodified script with on_policy_frac 0, so the teacher samples every training prefix.

Evidence: a325c9cc091fd2dcaa9aa478f9c79066b5e4faa9fde234d50ae0f40093e974b9, protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

## setup_arm_pureon

The pure_on_policy arm sets on_policy_frac to 1 throughout, so the student samples every prefix and the teacher labels it.

Evidence: 15dfd3b761058e74d05b785ec68e0a769a9df2bd0b25b43bcecbe045d88fcea3, protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

## setup_arm_fixed

The fixed_switch_1500 arm uses on_policy_frac 0 before step 1500 and 1.0 afterwards. A probe runs every 100 steps for logging only.

Evidence: 62e3500a89598b3557f866cd0d4f28ef1b31a40d263cbc05ac6fd37e7578c2d7, protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

## setup_arm_adaptive

In the adaptive_tau25 arm, a probe greedily decodes 256 training prompts every 100 steps. When the probe invalid rate falls below 25%, on_policy_frac switches from 0 to 1.0. The switch is forced at step 1500 if the trigger has not fired.

Evidence: b51aec68765d0ffe52f1a6cbcbd02e7a310307b18c0bd0858d8bf0ed9ba56c00, protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

## setup_arm_yoked

The yoked_replay arm first replays the adaptive run for the seed to find its firing step. It then trains a fresh student from training seed seed+1000 and switches to on_policy_frac 1.0 at exactly that step, ignoring the probe.

Evidence: 6fb44c755d1d5a0b0b65e8d6e2f5edaaf5f5acabb790be48e82806e75b88f916, protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

## setup_arm_corrupt

The corrupt_prefix arm uses the adaptive trigger and cap. After the switch, it samples student rollouts to measure per-token error p, then trains on ground-truth outputs (mode chosen uniformly) with each symbol substituted at rate p.

Evidence: 888f737d59a8422b59f6defa9a2831a89f9d83bb631891f6a5fa6880b12ea894, protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

## setup_prediction

Registered prediction: adaptive_tau25 is at least 1.0 point below both pure regimes and below the 8.8 hard-label baseline, and no worse than fixed_switch_1500. yoked_replay is within noise of adaptive or worse. corrupt_prefix recovers less than half of the adaptive gain over off_policy_ref.

Evidence: protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

## setup_analysis

Analysis: for each arm, per-seed paired mean differences against off_policy_ref, read descriptively beside the seed spread, with no inferential test.

Evidence: protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

## setup_baseline

The profile declares a val_invalid_rate of 8.8 for a hard-label training baseline. That baseline was not re-run in this protocol.

Evidence: profile.

## setup_falsification

Registered falsification: the hypothesis fails if adaptive_tau25 does not beat the better of off_policy_ref and pure_on_policy by more than the seed standard deviation, or if both yoked_replay and corrupt_prefix match it. A trigger that fires only through the cap on every seed leaves responsiveness untested.

Evidence: protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

## setup_folds

Every possible input is shuffled once. Five folds of 500 prompts each are cut from the start of that order, the next 2000 prompts form the held-out test set, and the teacher trains on the rest. A run on one fold validates on that fold and trains on the prompts of the other four.

Evidence: scorer.

## setup_prefix_sampling

Training prefixes are sampled, not decoded greedily: each batch draws its output sequences from either the teacher or the student at sampling temperature 1.0, and on_policy_frac is the probability that a batch uses the student.

Evidence: train, scorer.

## setup_attempts

Each arm's run record lists 5 training attempts for its five seeds.

Evidence: protocol_8198ab82d8c25b04b09e5215_off_policy_ref, protocol_8198ab82d8c25b04b09e5215_pure_on_policy, protocol_8198ab82d8c25b04b09e5215_fixed_switch_1500, protocol_8198ab82d8c25b04b09e5215_adaptive_tau25, protocol_8198ab82d8c25b04b09e5215_yoked_replay, protocol_8198ab82d8c25b04b09e5215_corrupt_prefix.

## setup_teacher_training

The teacher is trained once with a fixed seed for 8000 steps of AdamW at lr 1e-3 and batch 128. Each training target is one valid output chosen uniformly at random from the three modes, and the loss is cross-entropy over output positions.

Evidence: scorer.

## setup_environment

The frozen protocol records the software environment: Python 3.12.9, torch 2.11.0 and numpy 2.2.6.

Evidence: protocol:8198ab82d8c25b04b09e52158075ce48f69b4306524857080af1a75da198b2b6.

## setup_probe

The probe batch is a fixed subset of 256 training prompts, drawn once per run from a generator seeded with 10000 plus the run seed. The probe and the evaluation decode greedily.

Evidence: b51aec68765d0ffe52f1a6cbcbd02e7a310307b18c0bd0858d8bf0ed9ba56c00, scorer.

## setup_token_error

Per-token error p is the smallest number of positions at which a sampled student rollout differs from any valid output, divided by the output length. It is computed on each post-switch batch. Corruption draws use a generator seeded with 20000 plus the run seed.

Evidence: 888f737d59a8422b59f6defa9a2831a89f9d83bb631891f6a5fa6880b12ea894.

## related_gkd

GKD trains students on a mixture of fixed and self-generated sequences with a static student-data fraction. This study keeps that fraction knob but switches it once during training, conditioned on the student's probe invalid rate.

Evidence: arXiv:2306.13649.

## related_imitation

Imitation-learning distillation treats the teacher as an oracle on student-visited states to address exposure bias. This study designed a corrupted-prefix control to test that account. The control's corruption rate tracked its own student's rollout error, not the adaptive student's, so it yields no conclusion and the account remains untested here.

Evidence: arXiv:2009.07253, doi:10.1007/s00521-025-11162-0.

## related_seqkd

Sequence-level KD trains on teacher-generated outputs and is the canonical off-policy baseline. This study's reference arm is off-policy word-level distillation on teacher-sampled prefixes.

Evidence: arXiv:1606.07947.

## related_minillm

MiniLLM changes both the divergence (reverse KL) and the sampling distribution. This study holds the forward-KL loss fixed and varies only the prefix source and its schedule.

Evidence: arXiv:2306.08543.

## related_adaptive_triggers

DASH-OPD and DTDD switch adaptively between teacher and student within rollouts, using discrepancy signals. This study instead makes a single, one-way, training-schedule switch gated by a probe invalid rate. Their abstracts do not describe yoked or corrupted-prefix controls.

Evidence: arXiv:2607.29078, arXiv:2610.19302.

## related_state_distribution

A controlled study on GSM8K argues that the state distribution supervised matters alongside the loss. This study varies only the state source, on a non-language transduction task.

Evidence: arXiv:2605.22731v1.

## related_mask_forcing

Mask Forcing perturbs student self-rollouts in video diffusion distillation. This study's corrupted-prefix arm perturbs ground-truth discrete prefixes as a control, not as a method.

Evidence: arXiv:2609.09123.

## related_curriculum

Instance-level curriculum distillation orders training instances using an earlier student snapshot. This study schedules the policy that generates prefixes rather than the order of instances.

Evidence: arXiv:2106.10885v1.

## related_capacity

Prior work reports that student-generated outputs are costly and hard to optimise when the student is much smaller than the teacher. This study uses one fixed small student and does not vary the capacity gap.

Evidence: arXiv:2409.12512.

## related_entropy

Entropy-aware on-policy distillation reports instability of reverse-KL signals under high teacher entropy. This study's teacher models a three-mode distribution, but teacher entropy was not measured or varied.

Evidence: arXiv:2603.07079.

## scope_question

On the multimodal-transduction-L6-A8 seq-distill task, does switching from off-policy to on-policy training when the student's own probe invalid rate falls below a threshold lower val_invalid_rate compared with pure off-policy, pure on-policy and a fixed-step switch? If it does, does the benefit depend on the trigger responding to the student's state, and on the student's own prefixes?

Evidence: profile.

## scope_method

One pre-registered falsification protocol was run with six arms, each at full fidelity over five seed–fold pairs. Seed i was trained on the training prompts of cross-validation fold i and evaluated on that fold's val prompts, so seed and fold are confounded and no arm was evaluated on a single fold. Learning rate, temperature, forward-KL loss and step budget were the same in every arm, and the student architecture and teacher were frozen. The arms were: off_policy_ref (reference); pure_on_policy; a fixed switch to fully on-policy at a mid-run step; an adaptive arm that switches to fully on-policy once the greedy invalid rate on a training-prompt probe batch drops below threshold, with a forced-switch cap; a yoked replay that switches at the adaptive arm's firing step but trains on a different seed; and a corrupted-prefix arm that uses the same trigger but replaces post-switch prefixes with ground truth under token substitution. The trusted metric is val_invalid_rate from the frozen scorer. The analysis reports descriptive paired mean differences against off_policy_ref, with no inferential test. Probe trajectories, switch steps and on-policy batch counts are self-reported by train.py and are not verified by the scorer. The author decided to run no further experiments. This revision therefore narrows the argument: two control comparisons and one mechanism hypothesis are withdrawn as claims and moved to the limitations.

Evidence: profile.

## limit_1

Evaluation scope (corrected from the previous version): seed i was trained and evaluated on cross-validation fold i, so training prompts, probe prompts and the val set all change with the seed. The spread across seeds is seed-plus-fold variation. Nothing was evaluated on a single fold. Paired differences against the reference pair on both seed and fold.

Evidence: scope limitation.

## limit_2

Withdrawn claim (claim_corrupt_recovers_gain): the corrupted-prefix control set its substitution rate from its own student's sampled rollout error, not the adaptive student's. Because that student drifted to higher rollout error than the adaptive student on every seed (self-reported student_rollout_tokerr_mean), the control did not implement the specified matched noise level. Its post-switch prefixes were also exact ground truth rather than teacher samples. Its mean paired gain is also dominated by the same outlying reference pair as the adaptive gain. It is no longer treated as a test of whether the benefit needs the student's own prefixes; that question is unanswered.

Evidence: scope limitation.

## limit_3

Withdrawn claim (claim_yoked_matches_or_beats): the yoked replay arm trained on different seeds from the reference and adaptive arms. Its recorded per-seed differences are therefore not paired observations, and only a difference of means can be reported. That difference does not separate yoked from adaptive given the yoked arm's large spread, and the yoked students switched in different training states from their adaptive sources. Whether online responsiveness adds value beyond switch timing is untested.

Evidence: scope limitation.

## limit_4

Withdrawn claim (claim_variance_mechanism_hypothesis): the previous version proposed that more training on student-generated or noisy prefixes yields more seed-dependent outcomes. The data contradict this premise. The adaptive and yoked arms had identical on-policy batch counts yet very different spreads, and with five values per arm the spread estimates cannot rank variance. No mode-commitment diagnostic was recorded, and no mechanism for the dispersion is claimed.

Evidence: scope limitation.

## limit_5

Correction to claim_pure_on_high_variance: the earlier wording said the mean advantage rested on a 'minority' of seeds. In fact three of five pairs were much better, one about equal and one worse. The point that the sign of the mean difference is not established is kept.

Evidence: scope limitation.

## limit_6

Only five seed–fold pairs per arm, with no inferential test and no interval on any paired difference. Each value comes from a single evaluation on a few hundred val prompts, and the binomial evaluation noise is of the same order as the adaptive-versus-reference mean difference. 'Within noise' was never quantified.

Evidence: scope limitation.

## limit_7

Only one adaptive setting (one probe threshold, full on-policy target) and one fixed switch step were run. The threshold sweep, partial targets, the static mixed-fraction arm and the other fixed steps were omitted.

Evidence: scope limitation.

## limit_8

The adaptive versus fixed-switch contrast is confounded with total on-policy duration.

Evidence: scope limitation.

## limit_9

The yoked and corrupt arms do not match the other arms in compute: the yoked arm spends replay compute and the corrupt arm discards student rollouts. No compute-matched total was recorded.

Evidence: scope limitation.

## limit_10

Probe trajectories, switch steps, on-policy batch counts and per-token error rates are self-reported by train.py and not verified by the scorer. The probe uses training prompts, not val.

Evidence: scope limitation.

## limit_11

Single synthetic task, single tiny fixed student, single divergence. The results do not generalise beyond this setting.

Evidence: scope limitation.

## limit_12

No arm's mean reached the profile's declared hard-label baseline. That baseline was not re-measured in this study, so the comparison is indicative only.

Evidence: scope limitation.

## limit_13

Run verdicts are recorded as 'discarded' and the protocol outcome is 'unreviewed'. The author has decided not to run further experiments, so the requested reruns (cross-yoked seeds, replayed-rate corruption, mode-share diagnostics, more seeds) were not done.

Evidence: scope limitation.

## limit_14

Competing explanation, untested: the pure-on-policy mean advantage could come from some seeds committing to a single consistent output mode, which greedy decoding rewards, rather than from correcting exposure bias.

Evidence: scope limitation.

## limit_15

Attempts made outside the registered protocol, if any, are not reported. Hardware is not recorded, so the time budget is not tied to a machine. Setup details absent from the setup statements are not part of the recorded evidence.

Evidence: scope limitation.

## limit_16

Implementation details not restated in the setup statements are fixed in the frozen scorer and the versioned training scripts, and are not described further here.

Evidence: scope limitation.

