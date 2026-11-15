# Key points — Does an Adaptive Off-to-On-Policy Switch Help Distillation? A Pre-Registered Negative Result on a Multimodal Transduction Task

_Question: On the multimodal-transduction-L6-A8 seq-distill task, does switching from off-policy to on-policy training when the student's own probe invalid rate falls below a threshold lower val_invalid_rate compared with pure off-policy, pure on-policy and a fixed-step switch? If it does, does the benefit depend on the trigger responding to the student's state, and on the student's own prefixes?_

## Abstract

1. *context* — On the multimodal-transduction-L6-A8 seq-distill task, does switching from off-policy to on-policy training when the student's own probe invalid rate falls below a threshold lower val_invalid_rate compared with pure off-policy, pure on-policy and a fixed-step switch? If it does, does the benefit depend on the trigger responding to the student's state, and on the student's own prefixes?  
   evidence: `scope_question`
2. *result* — The pre-registered falsifying outcome was met. The adaptive-switch arm did not beat the better pure regime: pure_on_policy had the lower mean (9.04 against 12.88), so the adaptive switch was not better than the better pure regime by more than the seed–fold spread. The prediction that the adaptive arm would beat both pure regimes is therefore not supported in this configuration. The two controls meant to test why a trigger might help (yoked replay and corrupted prefixes) did not implement their intended contrasts and support no conclusion either way (see limitations). The conservative reading is that this experiment does not distinguish the switching rules.  
   evidence: `claim_hypothesis_not_supported`
3. *method* — One pre-registered falsification protocol was run with six arms, each at full fidelity over five seed–fold pairs. Seed i was trained on the training prompts of cross-validation fold i and evaluated on that fold's val prompts, so seed and fold are confounded and no arm was evaluated on a single fold. Learning rate, temperature, forward-KL loss and step budget were the same in every arm, and the student architecture and teacher were frozen. The arms were: off_policy_ref (reference); pure_on_policy; a fixed switch to fully on-policy at a mid-run step; an adaptive arm that switches to fully on-policy once the greedy invalid rate on a training-prompt probe batch drops below threshold, with a forced-switch cap; a yoked replay that switches at the adaptive arm's firing step but trains on a different seed; and a corrupted-prefix arm that uses the same trigger but replaces post-switch prefixes with ground truth under token substitution. The trusted metric is val_invalid_rate from the frozen scorer. The analysis reports descriptive paired mean differences against off_policy_ref, with no inferential test. Probe trajectories, switch steps and on-policy batch counts are self-reported by train.py and are not verified by the scorer. The author decided to run no further experiments. This revision therefore narrows the argument: two control comparisons and one mechanism hypothesis are withdrawn as claims and moved to the limitations.  
   evidence: `scope_method`
4. *result* — According to self-reported diagnostics, the adaptive trigger fired before the forced-switch cap on all five seed–fold pairs, and no switch was forced. On-policy batches occurred only after the reported switch step. The adaptive mechanism was therefore exercised in this run, rather than every switch coming from the cap.  
   evidence: `claim_trigger_exercised`
5. *result* — The adaptive-switch arm did not beat pure_on_policy on mean val_invalid_rate: pure_on_policy averaged 9.04 and the adaptive arm averaged 12.88. The pre-registered prediction that the adaptive switch would be below both pure regimes by a margin therefore fails for the on-policy endpoint. The adaptive arm had the narrower spread across seed–fold pairs.  
   evidence: `claim_adaptive_not_below_pure_on`

## Introduction

1. *context* — Task multimodal-transduction-L6-A8: each input is L=6 symbols from an alphabet of 8. Three outputs are valid: the copy, the reversal, and the reversal incremented mod 8. A mixture of modes counts as invalid.  
   evidence: `setup_task`
2. *context* — GKD trains students on a mixture of fixed and self-generated sequences with a static student-data fraction. This study keeps that fraction knob but switches it once during training, conditioned on the student's probe invalid rate.  
   evidence: `related_gkd`
3. *context* — DASH-OPD and DTDD switch adaptively between teacher and student within rollouts, using discrepancy signals. This study instead makes a single, one-way, training-schedule switch gated by a probe invalid rate. Their abstracts do not describe yoked or corrupted-prefix controls.  
   evidence: `related_adaptive_triggers`
4. *challenge* — Instance-level curriculum distillation orders training instances using an earlier student snapshot. This study schedules the policy that generates prefixes rather than the order of instances.  
   evidence: `related_curriculum`
5. *challenge* — On the multimodal-transduction-L6-A8 seq-distill task, does switching from off-policy to on-policy training when the student's own probe invalid rate falls below a threshold lower val_invalid_rate compared with pure off-policy, pure on-policy and a fixed-step switch? If it does, does the benefit depend on the trigger responding to the student's state, and on the student's own prefixes?  
   evidence: `scope_question`
6. *method* — Registered prediction: adaptive_tau25 is at least 1.0 point below both pure regimes and below the 8.8 hard-label baseline, and no worse than fixed_switch_1500. yoked_replay is within noise of adaptive or worse. corrupt_prefix recovers less than half of the adaptive gain over off_policy_ref.  
   evidence: `setup_prediction`
7. *method* — Registered falsification: the hypothesis fails if adaptive_tau25 does not beat the better of off_policy_ref and pure_on_policy by more than the seed standard deviation, or if both yoked_replay and corrupt_prefix match it. A trigger that fires only through the cap on every seed leaves responsiveness untested.  
   evidence: `setup_falsification`
8. *result* — The pre-registered falsifying outcome was met. The adaptive-switch arm did not beat the better pure regime: pure_on_policy had the lower mean (9.04 against 12.88), so the adaptive switch was not better than the better pure regime by more than the seed–fold spread. The prediction that the adaptive arm would beat both pure regimes is therefore not supported in this configuration. The two controls meant to test why a trigger might help (yoked replay and corrupted prefixes) did not implement their intended contrasts and support no conclusion either way (see limitations). The conservative reading is that this experiment does not distinguish the switching rules.  
   evidence: `claim_hypothesis_not_supported`
9. *result* — The adaptive-switch arm did not beat pure_on_policy on mean val_invalid_rate: pure_on_policy averaged 9.04 and the adaptive arm averaged 12.88. The pre-registered prediction that the adaptive switch would be below both pure regimes by a margin therefore fails for the on-policy endpoint. The adaptive arm had the narrower spread across seed–fold pairs.  
   evidence: `claim_adaptive_not_below_pure_on`

## Related Work

1. *context* — Sequence-level KD trains on teacher-generated outputs and is the canonical off-policy baseline. This study's reference arm is off-policy word-level distillation on teacher-sampled prefixes.  
   evidence: `related_seqkd`
2. *comparison* — Imitation-learning distillation treats the teacher as an oracle on student-visited states to address exposure bias. This study designed a corrupted-prefix control to test that account. The control's corruption rate tracked its own student's rollout error, not the adaptive student's, so it yields no conclusion and the account remains untested here.  
   evidence: `related_imitation`
3. *comparison* — GKD trains students on a mixture of fixed and self-generated sequences with a static student-data fraction. This study keeps that fraction knob but switches it once during training, conditioned on the student's probe invalid rate.  
   evidence: `related_gkd`
4. *comparison* — DASH-OPD and DTDD switch adaptively between teacher and student within rollouts, using discrepancy signals. This study instead makes a single, one-way, training-schedule switch gated by a probe invalid rate. Their abstracts do not describe yoked or corrupted-prefix controls.  
   evidence: `related_adaptive_triggers`
5. *comparison* — Instance-level curriculum distillation orders training instances using an earlier student snapshot. This study schedules the policy that generates prefixes rather than the order of instances.  
   evidence: `related_curriculum`
6. *comparison* — MiniLLM changes both the divergence (reverse KL) and the sampling distribution. This study holds the forward-KL loss fixed and varies only the prefix source and its schedule.  
   evidence: `related_minillm`
7. *comparison* — A controlled study on GSM8K argues that the state distribution supervised matters alongside the loss. This study varies only the state source, on a non-language transduction task.  
   evidence: `related_state_distribution`
8. *limitation* — Prior work reports that student-generated outputs are costly and hard to optimise when the student is much smaller than the teacher. This study uses one fixed small student and does not vary the capacity gap.  
   evidence: `related_capacity`
9. *limitation* — Entropy-aware on-policy distillation reports instability of reverse-KL signals under high teacher entropy. This study's teacher models a three-mode distribution, but teacher entropy was not measured or varied.  
   evidence: `related_entropy`
10. *comparison* — Mask Forcing perturbs student self-rollouts in video diffusion distillation. This study's corrupted-prefix arm perturbs ground-truth discrete prefixes as a control, not as a method.  
   evidence: `related_mask_forcing`

## Setup and Pre-Registered Protocol

1. *method* — One pre-registered falsification protocol was run with six arms, each at full fidelity over five seed–fold pairs. Seed i was trained on the training prompts of cross-validation fold i and evaluated on that fold's val prompts, so seed and fold are confounded and no arm was evaluated on a single fold. Learning rate, temperature, forward-KL loss and step budget were the same in every arm, and the student architecture and teacher were frozen. The arms were: off_policy_ref (reference); pure_on_policy; a fixed switch to fully on-policy at a mid-run step; an adaptive arm that switches to fully on-policy once the greedy invalid rate on a training-prompt probe batch drops below threshold, with a forced-switch cap; a yoked replay that switches at the adaptive arm's firing step but trains on a different seed; and a corrupted-prefix arm that uses the same trigger but replaces post-switch prefixes with ground truth under token substitution. The trusted metric is val_invalid_rate from the frozen scorer. The analysis reports descriptive paired mean differences against off_policy_ref, with no inferential test. Probe trajectories, switch steps and on-policy batch counts are self-reported by train.py and are not verified by the scorer. The author decided to run no further experiments. This revision therefore narrows the argument: two control comparisons and one mechanism hypothesis are withdrawn as claims and moved to the limitations.  
   evidence: `scope_method`
2. *method* — Task multimodal-transduction-L6-A8: each input is L=6 symbols from an alphabet of 8. Three outputs are valid: the copy, the reversal, and the reversal incremented mod 8. A mixture of modes counts as invalid.  
   evidence: `setup_task`
3. *method* — Every possible input is shuffled once. Five folds of 500 prompts each are cut from the start of that order, the next 2000 prompts form the held-out test set, and the teacher trains on the rest. A run on one fold validates on that fold and trains on the prompts of the other four.  
   evidence: `setup_folds`
4. *method* — The metric val_invalid_rate is the percentage of greedy student decodes on a validation fold that match none of the valid outputs. Lower is better. Each validation fold holds 500 prompts.  
   evidence: `setup_metric`
5. *method* — The student is a frozen-architecture pre-norm causal transformer with d_model 32, 2 layers and 2 heads. The frozen teacher has d_model 96 and 2 layers, trained for 8000 steps on inputs disjoint from the fold pool and the test set.  
   evidence: `setup_models`
6. *method* — The teacher is trained once with a fixed seed for 8000 steps of AdamW at lr 1e-3 and batch 128. Each training target is one valid output chosen uniformly at random from the three modes, and the loss is cross-entropy over output positions.  
   evidence: `setup_teacher_training`
7. *method* — Every arm trains for 3000 steps with batch 64 using AdamW at lr 0.003. The loss is forward KL between teacher and student next-token distributions at temperature 1.0 over output positions.  
   evidence: `setup_training`
8. *method* — Training prefixes are sampled, not decoded greedily: each batch draws its output sequences from either the teacher or the student at sampling temperature 1.0, and on_policy_frac is the probability that a batch uses the student.  
   evidence: `setup_prefix_sampling`
9. *method* — The probe batch is a fixed subset of 256 training prompts, drawn once per run from a generator seeded with 10000 plus the run seed. The probe and the evaluation decode greedily.  
   evidence: `setup_probe`
10. *method* — The off_policy_ref arm runs the unmodified script with on_policy_frac 0, so the teacher samples every training prefix.  
   evidence: `setup_arm_offpolicy`
11. *method* — The pure_on_policy arm sets on_policy_frac to 1 throughout, so the student samples every prefix and the teacher labels it.  
   evidence: `setup_arm_pureon`
12. *method* — The fixed_switch_1500 arm uses on_policy_frac 0 before step 1500 and 1.0 afterwards. A probe runs every 100 steps for logging only.  
   evidence: `setup_arm_fixed`
13. *method* — In the adaptive_tau25 arm, a probe greedily decodes 256 training prompts every 100 steps. When the probe invalid rate falls below 25%, on_policy_frac switches from 0 to 1.0. The switch is forced at step 1500 if the trigger has not fired.  
   evidence: `setup_arm_adaptive`
14. *method* — The yoked_replay arm first replays the adaptive run for the seed to find its firing step. It then trains a fresh student from training seed seed+1000 and switches to on_policy_frac 1.0 at exactly that step, ignoring the probe.  
   evidence: `setup_arm_yoked`
15. *method* — The corrupt_prefix arm uses the adaptive trigger and cap. After the switch, it samples student rollouts to measure per-token error p, then trains on ground-truth outputs (mode chosen uniformly) with each symbol substituted at rate p.  
   evidence: `setup_arm_corrupt`
16. *method* — Per-token error p is the smallest number of positions at which a sampled student rollout differs from any valid output, divided by the output length. It is computed on each post-switch batch. Corruption draws use a generator seeded with 20000 plus the run seed.  
   evidence: `setup_token_error`
17. *method* — Each arm ran five seeds (zero through four) over five cross-validation folds at full fidelity. Run records list fold indices alongside seed indices, so seed and fold vary together.  
   evidence: `setup_seeds_folds`
18. *method* — Each arm's run record lists 5 training attempts for its five seeds.  
   evidence: `setup_attempts`
19. *method* — Each full-fidelity run had a time budget of 1200 seconds.  
   evidence: `setup_budget`
20. *method* — The frozen protocol records the software environment: Python 3.12.9, torch 2.11.0 and numpy 2.2.6.  
   evidence: `setup_environment`
21. *method* — Analysis: for each arm, per-seed paired mean differences against off_policy_ref, read descriptively beside the seed spread, with no inferential test.  
   evidence: `setup_analysis`
22. *method* — The profile declares a val_invalid_rate of 8.8 for a hard-label training baseline. That baseline was not re-run in this protocol.  
   evidence: `setup_baseline`
23. *method* — Registered prediction: adaptive_tau25 is at least 1.0 point below both pure regimes and below the 8.8 hard-label baseline, and no worse than fixed_switch_1500. yoked_replay is within noise of adaptive or worse. corrupt_prefix recovers less than half of the adaptive gain over off_policy_ref.  
   evidence: `setup_prediction`
24. *method* — Registered falsification: the hypothesis fails if adaptive_tau25 does not beat the better of off_policy_ref and pure_on_policy by more than the seed standard deviation, or if both yoked_replay and corrupt_prefix match it. A trigger that fires only through the cap on every seed leaves responsiveness untested.  
   evidence: `setup_falsification`
25. *limitation* — Attempts made outside the registered protocol, if any, are not reported. Hardware is not recorded, so the time budget is not tied to a machine. Setup details absent from the setup statements are not part of the recorded evidence.  
   evidence: `limit_15`
26. *limitation* — Implementation details not restated in the setup statements are fixed in the frozen scorer and the versioned training scripts, and are not described further here.  
   evidence: `limit_16`

## Results

1. *result* — Mean validation invalid rate (%) over five seeds, with seed SD, was:
- off_policy_ref: 14.0 (3.234192)
- pure_on_policy: 9.04 (8.478679)
- fixed_switch_1500: 15.0 (3.896152)
- adaptive_tau25: 12.88 (2.229798)
- yoked_replay: 11.0 (8.866792)
- corrupt_prefix: 12.76 (6.582401)  
   evidence: `claim_all_arms_observation`
2. *result* — According to self-reported diagnostics, the adaptive trigger fired before the forced-switch cap on all five seed–fold pairs, and no switch was forced. On-policy batches occurred only after the reported switch step. The adaptive mechanism was therefore exercised in this run, rather than every switch coming from the cap.  
   evidence: `claim_trigger_exercised`
3. *comparison* — The adaptive-switch arm did not beat pure_on_policy on mean val_invalid_rate: pure_on_policy averaged 9.04 and the adaptive arm averaged 12.88. The pre-registered prediction that the adaptive switch would be below both pure regimes by a margin therefore fails for the on-policy endpoint. The adaptive arm had the narrower spread across seed–fold pairs.  
   evidence: `claim_adaptive_not_below_pure_on`
4. *comparison* — The adaptive-switch arm had a lower mean val_invalid_rate (12.88) than off_policy_ref (14.0), a paired mean difference of -1.12. The per-seed paired differences were small and mixed in sign: four of the five seed–fold pairs were within about one point of zero, and one of those four was positive. Most of the mean improvement comes from the single pair where off_policy_ref had its highest value (18.4). The improvement is small relative to the seed–fold spread and is not robust across pairs.  
   evidence: `claim_adaptive_vs_offpolicy`
5. *comparison* — pure_on_policy had a paired mean difference of -4.96 against off_policy_ref, but the per-pair pattern was heterogeneous. On three of five seed–fold pairs it was much better than off-policy, reaching values of 4.6, 2.6 and 4.2. On one pair it was about equal. On the remaining pair it was clearly worse (23.2 against the reference's 15.8). The direction is not uniform and the spread is large, so the sign of the mean difference is not established at this sample size.  
   evidence: `claim_pure_on_high_variance`
6. *limitation* — Correction to claim_pure_on_high_variance: the earlier wording said the mean advantage rested on a 'minority' of seeds. In fact three of five pairs were much better, one about equal and one worse. The point that the sign of the mean difference is not established is kept.  
   evidence: `limit_5`
7. *comparison* — No arm's mean reached the profile's declared hard-label baseline. That baseline was not re-measured in this study, so the comparison is indicative only.  
   evidence: `limit_12`
8. *comparison* — The fixed-step switch arm had the highest mean val_invalid_rate of all arms (15.0), a paired mean difference of 1.0 against off_policy_ref. The paired differences were mixed in sign: three of the five seed–fold pairs were worse than the reference and two were better, and the mean difference is within the seed–fold spread. The direction of the fixed-step arm relative to pure off-policy is therefore not established. The adaptive arm (12.88) had a lower mean than the fixed-step arm, which formally satisfies the prediction that adaptive is no worse than the fixed switch. This contrast confounds trigger responsiveness with switch timing: according to self-reported diagnostics, the adaptive switches fired earlier than the fixed step on every seed, so the adaptive arm spent more of training on-policy.  
   evidence: `claim_fixed_switch_worse`

## Discussion

1. *result* — The pre-registered falsifying outcome was met. The adaptive-switch arm did not beat the better pure regime: pure_on_policy had the lower mean (9.04 against 12.88), so the adaptive switch was not better than the better pure regime by more than the seed–fold spread. The prediction that the adaptive arm would beat both pure regimes is therefore not supported in this configuration. The two controls meant to test why a trigger might help (yoked replay and corrupted prefixes) did not implement their intended contrasts and support no conclusion either way (see limitations). The conservative reading is that this experiment does not distinguish the switching rules.  
   evidence: `claim_hypothesis_not_supported`
2. *limitation* — The adaptive versus fixed-switch contrast is confounded with total on-policy duration.  
   evidence: `limit_8`
3. *limitation* — Competing explanation, untested: the pure-on-policy mean advantage could come from some seeds committing to a single consistent output mode, which greedy decoding rewards, rather than from correcting exposure bias.  
   evidence: `limit_14`

## Limitations

1. *limitation* — Evaluation scope (corrected from the previous version): seed i was trained and evaluated on cross-validation fold i, so training prompts, probe prompts and the val set all change with the seed. The spread across seeds is seed-plus-fold variation. Nothing was evaluated on a single fold. Paired differences against the reference pair on both seed and fold.  
   evidence: `limit_1`
2. *limitation* — Only five seed–fold pairs per arm, with no inferential test and no interval on any paired difference. Each value comes from a single evaluation on a few hundred val prompts, and the binomial evaluation noise is of the same order as the adaptive-versus-reference mean difference. 'Within noise' was never quantified.  
   evidence: `limit_6`
3. *limitation* — Only one adaptive setting (one probe threshold, full on-policy target) and one fixed switch step were run. The threshold sweep, partial targets, the static mixed-fraction arm and the other fixed steps were omitted.  
   evidence: `limit_7`
4. *limitation* — The yoked and corrupt arms do not match the other arms in compute: the yoked arm spends replay compute and the corrupt arm discards student rollouts. No compute-matched total was recorded.  
   evidence: `limit_9`
5. *limitation* — Probe trajectories, switch steps, on-policy batch counts and per-token error rates are self-reported by train.py and not verified by the scorer. The probe uses training prompts, not val.  
   evidence: `limit_10`
6. *limitation* — Single synthetic task, single tiny fixed student, single divergence. The results do not generalise beyond this setting.  
   evidence: `limit_11`
7. *limitation* — Run verdicts are recorded as 'discarded' and the protocol outcome is 'unreviewed'. The author has decided not to run further experiments, so the requested reruns (cross-yoked seeds, replayed-rate corruption, mode-share diagnostics, more seeds) were not done.  
   evidence: `limit_13`
8. *limitation* — Withdrawn claim (claim_corrupt_recovers_gain): the corrupted-prefix control set its substitution rate from its own student's sampled rollout error, not the adaptive student's. Because that student drifted to higher rollout error than the adaptive student on every seed (self-reported student_rollout_tokerr_mean), the control did not implement the specified matched noise level. Its post-switch prefixes were also exact ground truth rather than teacher samples. Its mean paired gain is also dominated by the same outlying reference pair as the adaptive gain. It is no longer treated as a test of whether the benefit needs the student's own prefixes; that question is unanswered.  
   evidence: `limit_2`
9. *limitation* — Withdrawn claim (claim_yoked_matches_or_beats): the yoked replay arm trained on different seeds from the reference and adaptive arms. Its recorded per-seed differences are therefore not paired observations, and only a difference of means can be reported. That difference does not separate yoked from adaptive given the yoked arm's large spread, and the yoked students switched in different training states from their adaptive sources. Whether online responsiveness adds value beyond switch timing is untested.  
   evidence: `limit_3`
10. *limitation* — Withdrawn claim (claim_variance_mechanism_hypothesis): the previous version proposed that more training on student-generated or noisy prefixes yields more seed-dependent outcomes. The data contradict this premise. The adaptive and yoked arms had identical on-policy batch counts yet very different spreads, and with five values per arm the spread estimates cannot rank variance. No mode-commitment diagnostic was recorded, and no mechanism for the dispersion is claimed.  
   evidence: `limit_4`

## Conclusion

1. *result* — The pre-registered falsifying outcome was met. The adaptive-switch arm did not beat the better pure regime: pure_on_policy had the lower mean (9.04 against 12.88), so the adaptive switch was not better than the better pure regime by more than the seed–fold spread. The prediction that the adaptive arm would beat both pure regimes is therefore not supported in this configuration. The two controls meant to test why a trigger might help (yoked replay and corrupted prefixes) did not implement their intended contrasts and support no conclusion either way (see limitations). The conservative reading is that this experiment does not distinguish the switching rules.  
   evidence: `claim_hypothesis_not_supported`
2. *result* — The adaptive-switch arm did not beat pure_on_policy on mean val_invalid_rate: pure_on_policy averaged 9.04 and the adaptive arm averaged 12.88. The pre-registered prediction that the adaptive switch would be below both pure regimes by a margin therefore fails for the on-policy endpoint. The adaptive arm had the narrower spread across seed–fold pairs.  
   evidence: `claim_adaptive_not_below_pure_on`
3. *limitation* — Single synthetic task, single tiny fixed student, single divergence. The results do not generalise beyond this setting.  
   evidence: `limit_11`
4. *future_work* — Run verdicts are recorded as 'discarded' and the protocol outcome is 'unreviewed'. The author has decided not to run further experiments, so the requested reruns (cross-yoked seeds, replayed-rate corruption, mode-share diagnostics, more seeds) were not done.  
   evidence: `limit_13`

## Discussion

### Round 1 — ✅ no blocking issue left
- **Skeptic**: accept
  - (suggestion) [Abstract and Conclusion: claim_adaptive_not_below_pure_on (final sentence 'The adaptive arm had the narrower spread across seed–fold pairs')] The abstract and conclusion both carry the 'narrower spread' observation. A reader may take it as a selling point for the adaptive arm. limit_4 states that 'with five values per arm the spread estimates cannot rank variance', so headline placement over-weights a descriptive SD comparison the paper itself disclaims. → *Keep claim_adaptive_not_below_pure_on in Results only, beside limit_4 or limit_6. In the abstract and conclusion, claim_hypothesis_not_supported already carries the 9.04 vs 12.88 comparison. Author: consider whether this sentence should be qualified in the approved text.*
  - (suggestion) [Abstract: 'pure_on_policy had the lower mean (9.04 against 12.88)'] The abstract reports pure_on_policy's lower mean without the caveat from claim_pure_on_high_variance that the sign of its difference from off_policy_ref is not established (SD 8.48, one pair at 23.2). This invites a reading that pure on-policy is the winning regime. → *Add a short pointer to claim_pure_on_high_variance or limit_6 in the abstract, or drop claim_adaptive_not_below_pure_on from the abstract so the conservative closing sentence of claim_hypothesis_not_supported is the last word.*
  - (suggestion) [Discussion: limit_14 ('the pure-on-policy mean advantage could come from...')] The phrase 'mean advantage' presupposes an advantage. The Results statement claim_pure_on_high_variance says the sign is not established at this sample size. → *Author: consider rewording limit_4 to 'the lower pure-on-policy mean'. In the plan, place limit_14 immediately after claim_pure_on_high_variance, or right after the reminder that the sign is not established.*
  - (suggestion) [Results: claim_all_arms_observation table (yoked_replay 11.0, corrupt_prefix 12.76)] The yoked and corrupt means are both at or below the adaptive mean, and they appear in Results with no adjacent caveat. The statements that these controls are invalid (limit_2, limit_3) appear only in the Limitations section, several sections later. → *Place a pointer next to the table, for example by moving limit_2 and limit_3, or the related_imitation note, into Results. Then nobody reads the control means as evidence.*
  - (suggestion) [Whole plan: repeated statements] The plan repeats statements heavily. claim_hypothesis_not_supported appears 4 times and claim_adaptive_not_below_pure_on 4 times. scope_method, scope_question, setup_task, setup_prediction, setup_falsification, related_gkd, related_adaptive_triggers and related_curriculum each appear twice. This dilutes each section's message. → *Use each statement once where it does its main job: the claims in Results, with abstract and conclusion placement only for the headline claim. Move prediction and falsification to Setup and the related-work contrasts to Related Work. Keep the Introduction to the task, the question and one positioning statement.*
  - (suggestion) [Results: limit_5; Abstract/Setup: scope_method final sentences] Revision-history wording ('Correction to claim_pure_on_high_variance', 'This revision therefore narrows the argument', 'Withdrawn claim') addresses earlier drafts, not the paper's reader. limit_5 adds no information beyond claim_pure_on_high_variance. → *Omit limit_5. Author: consider approved variants of scope_method, limit_2, limit_3 and limit_4 without the revision framing.*
  - (suggestion) [Conclusion: limit_13 labelled (future_work)] limit_13 records that further runs will not be done. Labelling it future_work implies they are planned. → *Relabel it as a limitation, or omit it from the Conclusion since it already appears in Limitations.*
  - (suggestion) [Abstract: ordering and length] The full scope_method paragraph sits between two results in the abstract. This makes the abstract long and hard to follow. → *Order the abstract as question → method → falsification result → trigger-exercised. Consider leaving the full scope_method text to the Setup section only.*
- **Editor**: accept
  - (suggestion) [Abstract: order of bullets] The verdict [claim_hypothesis_not_supported] comes before the method [scope_method]. The reader meets 'falsifying outcome', 'yoked replay' and 'corrupted prefixes' before any arm is defined. [claim_adaptive_not_below_pure_on] then repeats the 9.04 vs 12.88 numbers the verdict has already given. → *Reorder to: scope_question, then scope_method, then claim_trigger_exercised (the mechanism fired), then claim_hypothesis_not_supported. Drop claim_adaptive_not_below_pure_on from the abstract. Its only new content is the narrower spread, which can wait for Results.*
  - (suggestion) [Abstract: pure_on_policy framing] The abstract names pure_on_policy as 'the better pure regime' (9.04) without saying that its advantage over off-policy is not established in sign. A reader may take away that pure on-policy wins. → *Consider adding claim_pure_on_high_variance, or at least placing it right after the verdict in Results and the Discussion. That way the 'experiment does not distinguish the switching rules' reading is visibly backed.*
  - (suggestion) [Introduction: duplicated statements] The Introduction restates scope_question, which is also in the Abstract. It also repeats related_gkd, related_adaptive_triggers and related_curriculum verbatim in Related Work, and setup_prediction and setup_falsification verbatim in Setup. It then repeats both headline results, which reappear in Results, Discussion and Conclusion. claim_hypothesis_not_supported appears four times in total. → *In the Introduction, keep setup_task, one positioning statement (related_gkd), scope_question, setup_falsification and claim_hypothesis_not_supported. Move the other related-work items and setup_prediction to their own sections only. In the Discussion, do not restate claim_hypothesis_not_supported; open with limit_8 and limit_14 as the interpretation of the result. Keep the Conclusion's restatement.*
  - (suggestion) [Introduction: related_curriculum tagged 'challenge'] A related-work contrast about instance ordering is not the challenge or gap that motivates the question. The Introduction has no stated 'why this matters' beat before scope_question. → *Drop related_curriculum from the Introduction, keeping it in Related Work. Let related_gkd (a static fraction versus a state-conditioned switch) serve as the motivating gap just before scope_question.*
  - (suggestion) [Setup: scope_method duplicated with arm-level setup] scope_method appears in full in both the Abstract and Setup, and it overlaps setup_seeds_folds, setup_analysis and the per-arm statements. setup_attempts and setup_seeds_folds also overlap. → *In Setup, either open with scope_method as the overview and drop setup_seeds_folds and setup_attempts, or omit scope_method from Setup and rely on the detailed statements. setup_prediction and setup_falsification should appear in only one section.*
  - (suggestion) [Results: placement of limit_5 and fixed-switch claim] limit_5 is a revision-history correction ('the earlier wording said'). Readers of the paper have no earlier version, so placing it in Results distracts. claim_fixed_switch_worse appears after limit_12, which breaks the flow of arm-versus-arm comparisons. → *Move limit_5 to Limitations or omit it, since claim_pure_on_high_variance already states the corrected pattern. Order Results as: all-arms table, trigger exercised, adaptive vs pure_on, pure_on vs off-policy, adaptive vs off-policy, fixed switch, then limit_12.*
  - (suggestion) [Author: revision-history wording in scope_method, limit_1, limit_5 and the withdrawn-claim limits] Several approved statements are worded as a response to a previous version ('This revision', 'corrected from the previous version', 'Withdrawn claim'). This reads oddly in a standalone paper. → *The author may consider rewording these statements so they describe the current scope directly.*
  - (suggestion) [Conclusion: limit_13 tagged future_work] limit_13 records discarded verdicts and reruns that were not done. It is not a forward-looking statement. → *Relabel it as a limitation. Alternatively, keep it in Limitations only and let the Conclusion end on limit_11.*
- **Methodologist**: accept
  - (suggestion) [Results: claim_all_arms_observation (table of means with 'seed SD')] The table labels its spread as 'seed SD', but each seed also changes the fold. That is only explained in scope_method and in limit_1, which sits far away in Limitations. A reader of the table alone could take the SD for pure seed variance. The table also omits that yoked_replay used different training seeds (seed+1000), which limit_3 explains. → *Place limit_1 (or a pointer to it) directly after the table in Results. Also mention limit_3 there, so readers see that the yoked arm's row is not paired with the others.*
  - (suggestion) [Abstract and Introduction: repeated statements] claim_hypothesis_not_supported appears four times: Abstract, Introduction, Discussion and Conclusion. claim_adaptive_not_below_pure_on also appears four times. scope_question appears twice and scope_method twice. The full scope_method paragraph in the Abstract makes it very long and buries the result. → *Keep scope_method only in Setup, and keep at most one of the two headline claims in the Abstract. Drop claim_adaptive_not_below_pure_on from the Introduction, since claim_hypothesis_not_supported already states the numbers.*
  - (suggestion) [Introduction: related_gkd, related_adaptive_triggers, related_curriculum] These three statements appear in both the Introduction and Related Work, word for word. → *Keep them in Related Work only. If the Introduction needs a motivation hook, use one of them at most.*
  - (suggestion) [Conclusion: limit_13 tagged as future_work] limit_13 reports discarded verdicts and reruns that were not done. It describes a limitation of this study, not planned future work, so the tag misrepresents it. → *Re-tag it as a limitation, or keep it only in Limitations. Its main role is to disclose that the runs were discarded and the protocol is unreviewed.*
  - (suggestion) [Author: limit_6 vs setup_metric] limit_6 says each value comes from 'a few hundred val prompts'. setup_metric states that each validation fold holds 500 prompts. The vague wording sits next to an exact count and weakens the binomial-noise argument. → *Author: consider rewording limit_6 to say 500 val prompts, matching setup_metric.*
  - (suggestion) [Author: claim_all_arms_observation precision] The SDs are given to six decimals (e.g. 3.234192), while the means have one or two. With n=5 this suggests more precision than the data have. → *Author: consider rounding the SDs to one or two decimals.*
  - (suggestion) [Discussion: limit_8 and claim_fixed_switch_worse] limit_8 repeats the confound that claim_fixed_switch_worse already states in Results. → *Either drop limit_8 from the Discussion or move it into Limitations, to avoid saying the same thing twice.*

*Every evidence id is a real run, paper, or the baseline; every number was checked against the registry. Approve to draft the prose.*
