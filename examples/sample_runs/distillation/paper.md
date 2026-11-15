# Does an Adaptive Off-to-On-Policy Switch Help Distillation? A Pre-Registered Negative Result on a Multimodal Transduction Task

> **Panel review:** reject · mean score 2.2/10 (harsh-but-fair 2 · harsh-critical 2 · open-minded 2.5)

## Abstract

On the multimodal-transduction-L6-A8 seq-distill task, we asked whether switching from off-policy to on-policy training, once the student's own probe invalid rate falls below a threshold, lowers val_invalid_rate compared with pure off-policy, pure on-policy and a fixed-step switch. The pre-registered falsifying outcome was met: the adaptive-switch arm did not beat the better pure regime, so the experiment does not distinguish the switching rules. We ran one pre-registered protocol with six arms, each over five seed–fold pairs, with the same learning rate, temperature, forward-KL loss and step budget in every arm and val_invalid_rate (lower is better) computed by a frozen scorer. According to self-reported diagnostics, the adaptive trigger fired before the forced-switch cap on all five seed–fold pairs, so the adaptive mechanism was exercised in this run. The adaptive arm still did not beat pure_on_policy on mean val_invalid_rate, which averaged 12.88 for the adaptive arm against 9.04 for pure_on_policy.

## Introduction

The task, multimodal-transduction-L6-A8, gives the student an input of L=6 symbols from an alphabet of 8, and three outputs are valid: the copy, the reversal, and the reversal incremented mod 8. An output that mixes modes counts as invalid.

GKD trains students on a mixture of fixed and self-generated sequences with a static student-data fraction. We keep that fraction knob and switch it once during training, conditioned on the student's probe invalid rate. DASH-OPD and DTDD switch adaptively between teacher and student within rollouts using discrepancy signals, and their abstracts do not describe yoked or corrupted-prefix controls. Our study instead makes a single, one-way, training-schedule switch gated by a probe invalid rate. Instance-level curriculum distillation orders training instances using an earlier student snapshot, whereas we schedule the policy that generates prefixes rather than the order of instances.

We ask whether, on the multimodal-transduction-L6-A8 seq-distill task, switching from off-policy to on-policy training when the student's own probe invalid rate falls below a threshold lowers val_invalid_rate compared with pure off-policy, pure on-policy and a fixed-step switch. If it does, we ask whether the benefit depends on the trigger responding to the student's state, and on the student's own prefixes.

The registered prediction was that adaptive_tau25 is at least 1.0 point below both pure regimes and below the 8.8 hard-label baseline, and no worse than fixed_switch_1500. The prediction further held that yoked_replay is within noise of adaptive or worse, and that corrupt_prefix recovers less than half of the adaptive gain over off_policy_ref. The registered falsification states that the hypothesis fails if adaptive_tau25 does not beat the better of off_policy_ref and pure_on_policy by more than the seed standard deviation, or if both yoked_replay and corrupt_prefix match it. A trigger that fires only through the cap on every seed would leave responsiveness untested.

The pre-registered falsifying outcome was met: pure_on_policy had the lower mean (9.04 against 12.88), so the adaptive switch did not beat the better pure regime by more than the seed–fold spread. The yoked-replay and corrupted-prefix controls did not implement their intended contrasts, so the conservative reading is that this experiment does not distinguish the switching rules. The prediction that the adaptive switch would sit below both pure regimes by a margin therefore fails for the on-policy endpoint of val_invalid_rate. The adaptive arm did have the narrower spread across seed–fold pairs.

## Related Work

Sequence-level knowledge distillation [arXiv:1606.07947] trains the student on teacher-generated outputs and is the canonical off-policy baseline. Our reference arm follows this off-policy setting and applies word-level distillation on teacher-sampled prefixes.

Imitation-learning distillation [arXiv:2009.07253] [doi:10.1007/s00521-025-11162-0] treats the teacher as an oracle on student-visited states, which addresses exposure bias. We designed a corrupted-prefix control to test this exposure-bias account. The control's corruption rate tracked its own student's rollout error instead of the adaptive student's, so the control yields no conclusion and the account remains untested here.

GKD [arXiv:2306.13649] trains students on a mixture of fixed and self-generated sequences, with a static student-data fraction. We keep that fraction knob and switch it once during training, conditioned on the student's probe invalid rate. DASH-OPD [arXiv:2607.29078] and DTDD [arXiv:2610.19302] instead switch adaptively between teacher and student within rollouts, using discrepancy signals. Our switch is a single, one-way change in the training schedule, gated by a probe invalid rate. The abstracts of these two methods do not describe yoked or corrupted-prefix controls.

Instance-level curriculum distillation [arXiv:2106.10885v1] orders training instances using an earlier student snapshot. Our study schedules the policy that generates prefixes and leaves the order of instances unchanged.

MiniLLM [arXiv:2306.08543] changes two factors at once: the divergence, which becomes reverse KL, and the sampling distribution. We hold the forward-KL loss fixed and vary only the prefix source and its schedule. A controlled study on GSM8K [arXiv:2605.22731v1] argues that the state distribution under supervision matters alongside the loss. Our study varies only the state source, and does so on a non-language transduction task.

Prior work [arXiv:2409.12512] reports that student-generated outputs are costly and hard to optimise when the student is much smaller than the teacher. We use one fixed small student and do not vary the capacity gap. Entropy-aware on-policy distillation [arXiv:2603.07079] reports that reverse-KL signals become unstable under high teacher entropy. Our teacher models a three-mode distribution, but we did not measure or vary teacher entropy.

Mask Forcing [arXiv:2609.09123] perturbs student self-rollouts in video diffusion distillation. Our corrupted-prefix arm perturbs ground-truth discrete prefixes and serves only as a control, with no claim as a method.

## Setup and Pre-Registered Protocol

We ran one pre-registered falsification protocol with six arms, each at full fidelity over five seed–fold pairs. Seed i was trained on the training prompts of cross-validation fold i and evaluated on that fold's val prompts, so seed and fold are confounded and no arm was evaluated on a single fold. Learning rate, temperature, forward-KL loss and step budget were the same in every arm, and the student architecture and teacher were frozen. The arms were off_policy_ref (the reference), pure_on_policy, a fixed switch to fully on-policy at a mid-run step, and an adaptive arm that switches to fully on-policy once the greedy invalid rate on a training-prompt probe batch drops below threshold, with a forced-switch cap. Two further arms completed the set: a yoked replay that switches at the adaptive arm's firing step but trains on a different seed, and a corrupted-prefix arm that uses the same trigger but replaces post-switch prefixes with ground truth under token substitution. The trusted metric is val_invalid_rate from the frozen scorer, and the analysis reports descriptive paired mean differences against off_policy_ref, with no inferential test. Probe trajectories, switch steps and on-policy batch counts are self-reported by train.py and are not verified by the scorer. The author decided to run no further experiments, so this revision narrows the argument: two control comparisons and one mechanism hypothesis are withdrawn as claims and moved to the limitations.

In the task multimodal-transduction-L6-A8, each input is L=6 symbols from an alphabet of 8. Three outputs are valid: the copy, the reversal, and the reversal incremented mod 8. A mixture of modes counts as invalid. Every possible input is shuffled once, and five folds of 500 prompts each are cut from the start of that order. The next 2000 prompts form the held-out test set, and the teacher trains on the rest. A run on one fold validates on that fold and trains on the prompts of the other four. The metric val_invalid_rate is the percentage of greedy student decodes on a validation fold that match none of the valid outputs, and lower is better. Each validation fold holds 500 prompts.

The student is a frozen-architecture pre-norm causal transformer with d_model 32, 2 layers and 2 heads. The frozen teacher has d_model 96 and 2 layers, trained for 8000 steps on inputs disjoint from the fold pool and the test set. The teacher is trained once with a fixed seed for 8000 steps of AdamW at lr 1e-3 and batch 128. Each teacher training target is one valid output chosen uniformly at random from the three modes, and the loss is cross-entropy over output positions.

Every arm trains for 3000 steps with batch 64 using AdamW at lr 0.003. The loss is forward KL between teacher and student next-token distributions at temperature 1.0 over output positions. Training prefixes are sampled rather than decoded greedily: each batch draws its output sequences from either the teacher or the student at sampling temperature 1.0. The quantity on_policy_frac is the probability that a batch uses the student.

The probe batch is a fixed subset of 256 training prompts, drawn once per run from a generator seeded with 10000 plus the run seed. The probe and the evaluation decode greedily. The off_policy_ref arm runs the unmodified script with on_policy_frac 0, so the teacher samples every training prefix. The pure_on_policy arm sets on_policy_frac to 1 throughout, so the student samples every prefix and the teacher labels it. The fixed_switch_1500 arm uses on_policy_frac 0 before step 1500 and 1.0 afterwards, with a probe every 100 steps for logging only.

In the adaptive_tau25 arm, a probe greedily decodes 256 training prompts every 100 steps. When the probe invalid rate falls below 25%, on_policy_frac switches from 0 to 1.0. The switch is forced at step 1500 if the trigger has not fired. The yoked_replay arm first replays the adaptive run for the seed to find its firing step. It then trains a fresh student from training seed seed+1000 and switches to on_policy_frac 1.0 at exactly that step, ignoring the probe.

The corrupt_prefix arm uses the adaptive trigger and cap. After the switch, it samples student rollouts to measure per-token error p, then trains on ground-truth outputs (mode chosen uniformly) with each symbol substituted at rate p. Per-token error p is the smallest number of positions at which a sampled student rollout differs from any valid output, divided by the output length. It is computed on each post-switch batch, and corruption draws use a generator seeded with 20000 plus the run seed.

Each arm ran five seeds (zero through four) over five cross-validation folds at full fidelity. Run records list fold indices alongside seed indices, so seed and fold vary together. Each arm's run record lists 5 training attempts for its five seeds. Each full-fidelity run had a time budget of 1200 seconds. The frozen protocol records the software environment: Python 3.12.9, torch 2.11.0 and numpy 2.2.6.

The analysis computes, for each arm, per-seed paired mean differences against off_policy_ref, read descriptively beside the seed spread, with no inferential test. The profile declares a val_invalid_rate of 8.8 for a hard-label training baseline. That baseline was not re-run in this protocol.

The registered prediction was that adaptive_tau25 is at least 1.0 point below both pure regimes and below the 8.8 hard-label baseline, and no worse than fixed_switch_1500. It further predicted that yoked_replay is within noise of adaptive or worse, and that corrupt_prefix recovers less than half of the adaptive gain over off_policy_ref. Under the registered falsification, the hypothesis fails if adaptive_tau25 does not beat the better of off_policy_ref and pure_on_policy by more than the seed standard deviation, or if both yoked_replay and corrupt_prefix match it. A trigger that fires only through the cap on every seed leaves responsiveness untested.

Attempts made outside the registered protocol, if any, are not reported. Hardware is not recorded, so the time budget is not tied to a machine. Setup details absent from the setup statements are not part of the recorded evidence. Implementation details not restated in the setup statements are fixed in the frozen scorer and the versioned training scripts, and are not described further here.

## Results

Mean validation invalid rate (%, lower is better) over five seeds, with seed SD in parentheses, was 14.0 (3.234192) for off_policy_ref, 9.04 (8.478679) for pure_on_policy, 15.0 (3.896152) for fixed_switch_1500, 12.88 (2.229798) for adaptive_tau25, 11.0 (8.866792) for yoked_replay and 12.76 (6.582401) for corrupt_prefix.

The precondition for testing trigger responsiveness held: according to self-reported diagnostics, the adaptive trigger fired before the forced-switch cap on all five seed–fold pairs, and no switch was forced. On-policy batches occurred only after the reported switch step. The adaptive mechanism was therefore exercised in this run, rather than every switch coming from the cap.

The pre-registered prediction that the adaptive switch would sit below both pure regimes by a margin fails for the on-policy endpoint, because pure_on_policy averaged 9.04 and the adaptive arm averaged 12.88. The adaptive arm had the narrower spread across seed–fold pairs.

Against off_policy_ref, the same prediction meets only a small gain: the adaptive arm had a lower mean val_invalid_rate (12.88 against 14.0), a paired mean difference of -1.12. The per-seed paired differences were small and mixed in sign, with four of the five seed–fold pairs within about one point of zero and one of those four positive. Most of the mean improvement comes from the single pair where off_policy_ref had its highest value (18.4). The improvement is therefore small relative to the seed–fold spread and is not robust across pairs.

The pure on-policy endpoint had a paired mean difference of -4.96 against off_policy_ref, but the per-pair pattern was heterogeneous. On three of five seed–fold pairs it was much better than off-policy, reaching values of 4.6, 2.6 and 4.2. On one pair it was about equal, and on the remaining pair it was clearly worse (23.2 against the reference's 15.8). This per-pair count corrects an earlier wording that placed the mean advantage on a 'minority' of seeds, since three of five pairs were much better, one about equal and one worse. Because the direction is not uniform and the spread is large, the sign of the mean difference is not established at this sample size.

The prediction that the adaptive arm would fall below the profile's declared hard-label baseline is not met, since no arm's mean reached that baseline. That baseline was not re-measured in this study, so the comparison is indicative only.

For the prediction that the adaptive arm would be no worse than the fixed switch, the reference point is weak: the fixed-step switch arm had the highest mean val_invalid_rate of all arms (15.0), a paired mean difference of 1.0 against off_policy_ref. Its paired differences were mixed in sign, with three of the five seed–fold pairs worse than the reference and two better, and the mean difference lies within the seed–fold spread. The direction of the fixed-step arm relative to pure off-policy is therefore not established. The adaptive arm (12.88) had a lower mean than the fixed-step arm, which formally satisfies the prediction that adaptive is no worse than the fixed switch. This contrast confounds trigger responsiveness with switch timing, because self-reported diagnostics show the adaptive switches fired earlier than the fixed step on every seed, so the adaptive arm spent more of training on-policy.

## Discussion

The pre-registered falsifying outcome was met. The adaptive-switch arm did not beat the better pure regime: pure_on_policy had the lower mean (9.04 against 12.88), so the adaptive switch was not better than the better pure regime by more than the seed–fold spread. The prediction that the adaptive arm would beat both pure regimes is therefore not supported in this configuration. The two controls meant to test why a trigger might help (yoked replay and corrupted prefixes) did not implement their intended contrasts and support no conclusion either way, as the limitations explain. The conservative reading is that this experiment does not distinguish the switching rules.

The adaptive versus fixed-switch contrast is confounded with total on-policy duration. A competing explanation remains untested: the pure-on-policy mean advantage could come from some seeds committing to a single consistent output mode, which greedy decoding rewards, rather than from correcting exposure bias.

## Limitations

Seed i was trained and evaluated on cross-validation fold i, so training prompts, probe prompts and the val set all change with the seed. The spread across seeds is therefore seed-plus-fold variation, and nothing was evaluated on a single fold. Paired differences are paired against the reference on both seed and fold.

Each arm has only five seed–fold pairs, and we ran no inferential test and computed no interval on any paired difference. Each value comes from a single evaluation on a few hundred val prompts, and the binomial evaluation noise is of the same order as the adaptive-versus-reference mean difference. 'Within noise' was never quantified.

We ran only one adaptive setting (one probe threshold, full on-policy target) and one fixed switch step. The threshold sweep, partial targets, the static mixed-fraction arm and the other fixed steps were omitted. The yoked and corrupt arms also do not match the other arms in compute: the yoked arm spends replay compute and the corrupt arm discards student rollouts. No compute-matched total was recorded.

Probe trajectories, switch steps, on-policy batch counts and per-token error rates are self-reported by train.py and are not verified by the scorer. The probe uses training prompts, not val. The study covers a single synthetic task, a single tiny fixed student and a single divergence, and the results do not generalise beyond this setting.

Run verdicts are recorded as 'discarded' and the protocol outcome as 'unreviewed'. The author has decided not to run further experiments, so the requested reruns (cross-yoked seeds, replayed-rate corruption, mode-share diagnostics, more seeds) were not done.

We withdraw the claim that the corrupted-prefix control recovers the adaptive gain (claim_corrupt_recovers_gain), because that control set its substitution rate from its own student's sampled rollout error instead of the adaptive student's. That student drifted to higher rollout error than the adaptive student on every seed (self-reported student_rollout_tokerr_mean), so the control did not implement the specified matched noise level. Its post-switch prefixes were also exact ground truth rather than teacher samples, and its mean paired gain is dominated by the same outlying reference pair as the adaptive gain. The control is no longer treated as a test of whether the benefit needs the student's own prefixes, and that question is unanswered.

We withdraw the claim that the yoked replay arm matches or beats the adaptive arm (claim_yoked_matches_or_beats), because the yoked arm trained on different seeds from the reference and adaptive arms. Its recorded per-seed differences are therefore not paired observations, and only a difference of means can be reported. That difference does not separate yoked from adaptive given the yoked arm's large spread, and the yoked students switched in different training states from their adaptive sources. Whether online responsiveness adds value beyond switch timing is untested.

We also withdraw the variance-mechanism hypothesis (claim_variance_mechanism_hypothesis), which in the previous version proposed that more training on student-generated or noisy prefixes yields more seed-dependent outcomes. The data contradict this premise: the adaptive and yoked arms had identical on-policy batch counts yet very different spreads. With five values per arm, the spread estimates cannot rank variance. No mode-commitment diagnostic was recorded, and we claim no mechanism for the dispersion.

## Conclusion

We asked whether a probe-gated switch from off-policy to on-policy training lowers val_invalid_rate on this task, and the pre-registered falsifying outcome was met: the adaptive-switch arm did not beat the better pure regime, so this experiment does not distinguish the switching rules. Pure_on_policy averaged 9.04 and the adaptive arm averaged 12.88 (lower is better), although the adaptive arm had the narrower spread across seed–fold pairs.

The scope is one synthetic task with one tiny fixed student and one divergence, so the results do not generalise beyond this setting. The author has decided not to run further experiments, so the requested reruns (cross-yoked seeds, replayed-rate corruption, mode-share diagnostics, more seeds) were not done and remain the next steps.

---

### Drafting notes (for the author)

- **Setup and Pre-Registered Protocol**: 2 sentence(s) over 40 words (first: "The arms were off_policy_ref (the reference), pure_on_policy…")
- **Results**: 2 sentence(s) over 40 words (first: "Mean validation invalid rate (%, lower is better) over five …")
- **Conclusion**: 1 sentence(s) over 40 words (first: "We asked whether a probe-gated switch from off-policy to on-…")

*Written under a version-bound scientific review and author release. Automated checks and semantic review can still miss errors; inspect the [evidence bindings](evidence_map.md) and the [sentence-by-sentence trace](paper_traced.md).*
