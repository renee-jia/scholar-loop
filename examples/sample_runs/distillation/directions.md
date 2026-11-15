# Research directions — on-policy versus off-policy knowledge distillation for autoregressive sequence models

## Your steer

> What the author wants to find out: I want to study on-policy versus off-policy distillation: how the two compare, and when each one works better.

## Direction map

_Directions are lines of inquiry, not experiments. Choose one or more; ideas are generated only inside the directions you choose._

| # | direction | stance toward your hunches | builds on |
|---|---|---|---|
| 1 | **Capacity-gap crossover map for on- vs off-policy** | develops | `arXiv:2306.13649`, `arXiv:2409.12512`, `arXiv:2009.07253`, `arXiv:1606.07947` |
| 2 | **Factorial separation of state distribution and divergence** | develops | `arXiv:2306.08543`, `arXiv:2307.15190`, `arXiv:2605.22731v1`, `arXiv:2306.13649` |
| 3 | **Validity gains are sharpening, not exposure-bias correction** | challenges | `arXiv:1606.07947`, `arXiv:2603.07079`, `arXiv:2609.09123`, `arXiv:2409.12512` |
| 4 | ✅ **Off-to-on data-source curriculum** | develops | `arXiv:2009.07253`, `arXiv:2306.13649`, `arXiv:2106.10885v1`, `doi:10.1007/s00521-025-11162-0` |
| 5 | **Corrupted-prefix rollouts as a discrete analogue of noise-masked distillation** | outside | `arXiv:2609.09123`, `arXiv:2306.13649`, `arXiv:2605.22731v1` |

### 1. Capacity-gap crossover map for on- vs off-policy

Whether on-policy distillation beats off-policy on val_invalid_rate depends on student capacity relative to the teacher. There should be a crossover: small students are hurt by noisy self-rollouts, while mid-size students gain from correcting their own errors.

- **How it relates to your steer:** The researcher asks how the two regimes compare and when each works better. This direction answers 'when' along the axis the literature flags as decisive but never sweeps: student size. It does so with the knobs the setting exposes, d_model, n_layers and on_policy_frac.
- **Gap:** Gap 1: no study produces a crossover curve of on-policy benefit against student capacity. GKD tunes the mixing ratio per task, and the on-policy degradation reported for large capacity gaps is not quantified.
- **Open question:** Is there a student size below which raising on_policy_frac increases val_invalid_rate? Is that threshold sharp or gradual?
- **First experiment:** Run a 3×4 grid: d_model in {16, 32, 64} with n_layers=2, crossed with on_policy_frac in {0, 0.25, 0.5, 1.0}. Keep lr=0.003, steps=3000 and temperature=1.0, and use 2 seeds each. Plot val_invalid_rate against on_policy_frac for each size and compare to the 8.8 baseline.
- **How it could turn out dull:** Every student size is far below the teacher, or the task is easy enough that all sizes saturate. The curves then come out parallel with no crossover, and the only conclusion is that a bigger student is better.
- **ID:** `dir_7766dae9e3`

### 2. Factorial separation of state distribution and divergence

The apparent on-policy gain on structural validity may come mostly from the divergence, with reverse KL's mode-seeking suppressing malformed tokens, rather than from training on student-visited prefixes. A crossed design can attribute the effect to one or the other.

- **How it relates to your steer:** Comparing 'on-policy vs off-policy' is ill-posed while the loss changes along with the sampling. This direction sharpens the researcher's comparison so that 'on-policy wins' means the prefixes, not the KL direction.
- **Gap:** Gap 2: sampling distribution and divergence are confounded. MiniLLM changes both at once, f-DISTILL varies only the loss, and the state-distribution study varies only states on a single text task.
- **Open question:** On a multimodal transduction task scored by invalid rate, does the main effect sit in the sampling (on_policy_frac) or in the divergence (FKL/RKL/JSD)? Do the two interact?
- **First experiment:** Edit the loss to support FKL, RKL and JSD(0.5). Cross these with on_policy_frac in {0, 1} at the default d_model=32 and n_layers=2, giving 6 cells × 2 seeds. Compute main effects and the interaction on val_invalid_rate.
- **How it could turn out dull:** Both factors produce small effects within seed noise at 3000 steps, or only one divergence trains stably on-policy. Either way the factorial collapses to a one-factor result.
- **ID:** `dir_5c4f76dd2e`

### 3. Validity gains are sharpening, not exposure-bias correction

Challenge to the framing that on-policy wins because it corrects exposure bias. On a validity metric, a lower invalid rate may come simply from concentrating the student on the teacher's modes. If so, off-policy SeqKD on low-temperature or greedy teacher outputs would match or beat on-policy at a fraction of the cost.

- **How it relates to your steer:** The researcher implicitly expects on-policy and off-policy to differ for regime-specific reasons. This direction argues the difference may be an artefact of output entropy. For it to be wrong, on-policy must still win after the off-policy student's entropy is matched via temperature. Finding out requires an entropy-matched comparison.
- **Gap:** Gaps 3 and 4: invalid rate has never been the primary metric. Mode-seeking may trade coverage for validity. SeqKD's simplification argument has never been tested per teacher-entropy bucket against on-policy training.
- **Open question:** Once student output entropy is matched, does on-policy still reduce val_invalid_rate relative to off-policy? Is the residual effect concentrated on high teacher-entropy examples?
- **First experiment:** Run on_policy_frac=0 with temperature in {0.3, 0.6, 1.0} against on_policy_frac=1 at temperature=1.0, with 2 seeds each. Log mean student token entropy alongside val_invalid_rate. Then check whether invalid rate is predicted by entropy alone, regardless of regime.
- **How it could turn out dull:** Lowering temperature both sharpens the targets and reduces information about the teacher. A null result would then be ambiguous between 'sharpening explains it' and 'temperature hurt learning'.
- **ID:** `dir_0d563b1657`

### 4. Off-to-on data-source curriculum  ✅ chosen

A schedule that starts off-policy, while the student's rollouts are mostly garbage, and switches to on-policy once rollouts are near-valid should beat both pure regimes. The best switch point should be tied to the student's own invalid rate rather than to a fixed step.

- **How it relates to your steer:** This turns the researcher's 'when does each work better' into a within-run question. Each regime may be best at a different training phase, so the answer could be 'both, in sequence'.
- **Gap:** Gap 6: DAgger mixing and GKD's λ are static or simply annealed. Curricula order instances, never the policy that generates the training prefixes.
- **Open question:** Is there an optimal switch point? Does an adaptive trigger, such as switching when the student's batch invalid rate falls below a threshold, beat a fixed step?
- **First experiment:** Edit the schedule so that on_policy_frac is 0 until step S and then 1. Sweep S over {0, 750, 1500, 2250, 3000} at the default size with 2 seeds, and compare val_invalid_rate against the two pure endpoints.
- **How it could turn out dull:** The curve over S is monotone, with one pure regime best throughout. This reduces to the plain comparison and yields only a minor tuning trick.
- **ID:** `dir_591a5b4f38`

### 5. Corrupted-prefix rollouts as a discrete analogue of noise-masked distillation

Clean student rollouts give the teacher only the states the student already reaches. Randomly corrupting tokens in student or teacher prefixes before teacher scoring may widen the state coverage cheaply. It could reduce invalid outputs as much as true on-policy training, without the instability that small students show on-policy.

- **How it relates to your steer:** The literature, not the steer, points here. Rollout perturbation reduced mode collapse in video distillation but is untested for discrete transduction. It sits between the researcher's two regimes, as a third way to shape the prefix distribution.
- **Gap:** Gap 7: noisy or masked self-rollouts are demonstrated only for diffusion video. Their discrete analogue has never been compared with clean on-policy or off-policy training. Gap 8 adds that on-policy evidence is thin for non-text multimodal transduction.
- **Open question:** Does corrupting prefixes at rate p recover most of the on-policy benefit on val_invalid_rate when on_policy_frac=0? Does it help further when added on top of on_policy_frac>0?
- **First experiment:** Add an augmentation that replaces a fraction p in {0, 0.05, 0.15} of prefix tokens with random tokens before the teacher provides targets. Run it at on_policy_frac in {0, 0.5} at the default size, 6 cells × 2 seeds.
- **How it could turn out dull:** Random corruption produces states unlike the student's real errors. The result is then just generic token-noise regularization, which is unrelated to the on/off-policy question.
- **ID:** `dir_c9ddc0ae81`
