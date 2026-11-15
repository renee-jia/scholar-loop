# Literature review — on-policy versus off-policy knowledge distillation for autoregressive sequence models

_20 papers · 6 themes · 8 open gaps · queries: `on-policy distillation language models` · `generalized knowledge distillation student-generated sequences` · `sequence-level knowledge distillation` · `exposure bias imitation learning distillation` · `reverse KL distillation autoregressive models` · `DAgger scheduled sampling sequence generation`_

## Overview

The papers split along one axis: which state (prefix) distribution supervision is applied on. Off-policy KD trains on fixed prefixes: ground truth for word-level KD, or teacher-generated sequences for SeqKD [arXiv:1606.07947] and its speech variants [arXiv:1811.04531][doi:10.1109/icassp.2019.8682671][doi:10.1016/j.neucom.2020.11.025]. On-policy KD trains on prefixes sampled from the student, with teacher feedback; it began as imitation-learning/DAgger-style distillation [arXiv:2009.07253][doi:10.1007/s00521-025-11162-0][arXiv:2406.09393] and was popularized for LLMs [arXiv:2306.13649][arXiv:2306.08543]. A second, often confounded axis is the divergence used: forward KL, reverse KL, JSD or TVD [arXiv:2307.15190][arXiv:2603.07079][arXiv:2409.12512]. Recent work tries to separate the two axes explicitly [arXiv:2605.22731v1] and survey it [arXiv:2604.00626].

Three points matter for the reader's setting: a small multimodal transduction task (L6-A8) scored by val_invalid_rate. First, almost all evidence comes from NLP/LLM benchmarks scored by ROUGE, BLEU or accuracy, not by structural validity of outputs. Second, the claimed on-policy advantage (exposure-bias correction) depends on student capacity, teacher–student gap and teacher entropy. These conditions are rarely varied systematically, and [arXiv:2409.12512] reports that on-policy training gets harder when the student is much smaller. Third, the reverse-KL mode-seeking that on-policy methods typically pair with is itself contested [arXiv:2603.07079][arXiv:2609.09123]. A 'validity' metric may reward mode-seeking (fewer malformed outputs) even where coverage suffers. That makes the reader's metric a useful, under-studied lens.

## Themes

### Off-policy sequence-level distillation (teacher-generated or fixed data)

The student is trained on teacher beam outputs or ground-truth prefixes with word- or sequence-level losses. SeqKD showed that training on teacher modes simplifies the target distribution and works well for NMT and ASR. It is cheap because teacher outputs can be precomputed. Its weakness is that the student never sees its own errors at training time.

- [arXiv:1606.07947](http://arxiv.org/abs/1606.07947) *Sequence-Level Knowledge Distillation* (2016, 1192 cites)
- [arXiv:1811.04531](http://arxiv.org/abs/1811.04531) *Sequence-level Knowledge Distillation for Model Compression of Attention-based Sequence-to-sequence Speech Recognition* (2019, 28 cites)
- [doi:10.1109/icassp.2019.8682671](https://doi.org/10.1109/icassp.2019.8682671) *Investigation of Sequence-level Knowledge Distillation Methods for CTC Acoustic Models* (2019, 27 cites)
- [doi:10.1016/j.neucom.2020.11.025](https://doi.org/10.1016/j.neucom.2020.11.025) *Mutual-learning sequence-level knowledge distillation for automatic speech recognition* (2020, 28 cites)
- [arXiv:2004.03846](http://arxiv.org/abs/2004.03846) *Structure-Level Knowledge Distillation For Multilingual Sequence Labeling* (2020, 35 cites)

### Imitation-learning view: exposure bias and DAgger-style distillation

Distillation is framed as imitation learning, with the teacher acting as an oracle queried on student-visited states. Mixing student rollouts with teacher labels reduces distribution drift in free-running decoding. Dynamic oracles extend DAgger to sequence metrics.

- [arXiv:2009.07253](http://arxiv.org/abs/2009.07253) *Autoregressive Knowledge Distillation through Imitation Learning* (2020, 70 cites)
- [doi:10.1007/s00521-025-11162-0](https://doi.org/10.1007/s00521-025-11162-0) *Mitigating exposure bias in large language model distillation: an imitation learning approach* (2025, 33 cites)
- [arXiv:2406.09393](http://arxiv.org/abs/2406.09393) *Improving Autoregressive Training with Dynamic Oracles* (2024, 0 cites)

### On-policy distillation for LLMs (GKD, MiniLLM and successors)

The student samples its own outputs and receives token-level teacher feedback, usually under reverse KL or generalized JSD. GKD interpolates between fixed and student-generated data through a mixing fraction, which is directly a knob for on- vs off-policy. MiniLLM uses policy-gradient optimization of reverse KL. Later work extends the idea to context distillation.

- [arXiv:2306.13649](http://arxiv.org/abs/2306.13649) *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (2023, 272 cites)
- [arXiv:2306.08543](http://arxiv.org/abs/2306.08543) *MiniLLM: On-Policy Distillation of Large Language Models* (2023, 474 cites)
- [arXiv:2602.12275](http://arxiv.org/abs/2602.12275) *On-Policy Context Distillation for Language Models* (2026, 212 cites)
- [arXiv:2604.00626](https://arxiv.org/abs/2604.00626) *A Survey of On-Policy Distillation for Large Language Models* (2026, 193 cites)

### Divergence choice and its interaction with sampling

Forward KL (mode-covering) and reverse KL (mode-seeking), along with f-divergence generalizations, yield different students. Several papers argue that reverse KL fails to capture most teacher behavior or collapses diversity when teacher entropy is high. They propose hybrids such as entropy-gated FKL+RKL or noise-perturbed rollouts.

- [arXiv:2307.15190](http://arxiv.org/abs/2307.15190) *f-Divergence Minimization for Sequence-Level Knowledge Distillation* (2023, 90 cites)
- [arXiv:2409.12512](http://arxiv.org/abs/2409.12512) *Exploring and enhancing the transfer of distribution in knowledge distillation for autoregressive language models* (2026, 3 cites)
- [arXiv:2603.07079](https://arxiv.org/abs/2603.07079) *Entropy-Aware On-Policy Distillation of Language Models* (2026, 153 cites)
- [arXiv:2609.09123](https://arxiv.org/abs/2609.09123) *Mask Forcing: Improving Autoregressive Video Diffusion Distillation via Dual-Noise Masking Rollout* (2026, 5 cites)

### Disentangling state distribution from objective

These works argue that the key variable is the distribution of prefixes on which supervision is applied, not the loss. They run controlled small-scale comparisons of SFT, RL and on-policy distillation.

- [arXiv:2605.22731v1](http://arxiv.org/abs/2605.22731v1) *Post-Training is About States, Not Tokens: A State Distribution View of SFT, RL, and On-Policy Distillation* (2026)
- [arXiv:2604.00626](https://arxiv.org/abs/2604.00626) *A Survey of On-Policy Distillation for Large Language Models* (2026, 193 cites)

### What transfers through distillation channels

These works study which teacher properties (knowledge, backdoors) survive different distillation channels, and how synthetic data can stand in for real data in autoregressive training. They are relevant to whether off-policy teacher data carries the target property at all.

- [arXiv:2610.26311v1](http://arxiv.org/abs/2610.26311v1) *Selective Transfer through Sequence Distillation: From Benchmark Knowledge to Backdoor Behaviour* (2026)
- [arXiv:2310.09983](http://arxiv.org/abs/2310.09983) *Farzi Data: Autoregressive Data Distillation* (2023, 11 cites)
- [arXiv:2106.10885v1](http://arxiv.org/abs/2106.10885v1) *Knowledge Distillation via Instance-level Sequence Learning* (2021)

## Open gaps

1. **No study maps when on-policy beats off-policy as a function of student capacity relative to the teacher. A grid over student size × student-data fraction (GKD λ ∈ {0, 0.25, 0.5, 1}) on a fixed transduction task, measuring val_invalid_rate, would directly answer the reader's question.**  
   GKD tunes the mixing ratio per task, and [arXiv:2409.12512] reports that on-policy degrades with large capacity gaps. Neither produces a crossover curve. (sources: `arXiv:2306.13649`, `arXiv:2409.12512`, `arXiv:2009.07253`)

2. **The sampling distribution (on vs off) and the divergence (FKL vs RKL vs JSD) are confounded in most comparisons. A full 2×3 factorial at matched compute would show whether 'on-policy helps' is actually 'RKL helps' or vice versa.**  
   MiniLLM changes both at once. f-DISTILL varies only the divergence, and the state-distribution study varies only states on a single LM task. (sources: `arXiv:2306.08543`, `arXiv:2307.15190`, `arXiv:2605.22731v1`, `arXiv:2306.13649`)

3. **Effect on structural validity/invalid-output rate rather than quality scores. Mode-seeking objectives and on-policy correction of self-generated errors both plausibly reduce invalid outputs. However, no paper uses invalid rate as the primary metric or checks whether it tracks BLEU/accuracy.**  
   The evaluated metrics are BLEU, ROUGE, WER and GSM8K accuracy. The diversity-collapse critiques imply validity and coverage may trade off, but this is unmeasured. (sources: `arXiv:2603.07079`, `arXiv:2609.09123`, `arXiv:1606.07947`, `arXiv:2306.08543`)

4. **Teacher entropy as a predictor of which regime wins. Measuring per-example teacher entropy on the multimodal task, and checking whether off-policy SeqKD (teacher mode) or on-policy FKL/RKL gives a lower invalid rate per entropy bucket, would give a 'when' rule.**  
   Entropy-aware OPD gates the loss by entropy but does not compare against off-policy distillation per entropy bucket. SeqKD's simplification argument is likewise untested by entropy. (sources: `arXiv:2603.07079`, `arXiv:1606.07947`, `arXiv:2409.12512`)

5. **Compute-matched comparison: on-policy requires student sampling plus teacher scoring at every step, while SeqKD precomputes the teacher outputs once. No paper reports invalid rate vs total FLOPs or wall-clock for both.**  
   Papers report final-quality numbers at different budgets. [arXiv:2409.12512] notes the cost of student-generated outputs but gives no Pareto curve. (sources: `arXiv:2409.12512`, `arXiv:2306.13649`, `arXiv:1606.07947`)

6. **Schedules from off- to on-policy: whether warm-starting with SeqKD and then switching to on-policy beats either pure regime, and when to switch. Instance-ordering curricula exist, but not data-source curricula.**  
   DAgger-style mixing and GKD's fixed λ are static or follow a simple anneal. The curriculum work orders instances, not the policy that generates them. (sources: `arXiv:2009.07253`, `arXiv:2306.13649`, `arXiv:2106.10885v1`, `doi:10.1007/s00521-025-11162-0`)

7. **Rollout perturbation for discrete on-policy distillation. Mask Forcing's noisy self-rollouts reduce mode collapse in video, but noise or masking on student token prefixes is untested for text or multimodal transduction.**  
   The method is demonstrated only for diffusion video, and the discrete analogue (e.g., random token corruption of student prefixes) has not been compared with clean on-policy training. (sources: `arXiv:2609.09123`, `arXiv:2306.13649`)

8. **Whether on-policy advantages appear for non-language or multimodal transduction at small scale. Speech work stayed off-policy, and on-policy work is almost all text LLMs.**  
   The ASR/CTC KD papers never tried student-sampled training. The only controlled state-distribution study is GSM8K text. (sources: `arXiv:1811.04531`, `doi:10.1016/j.neucom.2020.11.025`, `doi:10.1109/icassp.2019.8682671`, `arXiv:2605.22731v1`)

## Trends

- The field shifted from off-policy SeqKD (2016–2020, NMT/ASR) to on-policy distillation as the default for LLMs (2023 onward), reframed through imitation learning and exposure bias.
- Reverse KL became the default objective paired with on-policy sampling, and recent work questions its mode-seeking side effects, moving toward hybrid FKL/RKL or entropy-adaptive objectives.
- There is growing effort to disentangle the state distribution from the loss, with controlled small-scale studies and unifying surveys replacing method-versus-method leaderboard comparisons.
- The on-policy idea is spreading to new settings (context distillation, agents, autoregressive video diffusion), while evidence in non-text discrete transduction remains thin.
- Cost and capacity-gap limits of on-policy distillation are increasingly acknowledged but not yet quantified as decision rules.

## Paper notes

| paper | contribution | limitation |
|---|---|---|
| [arXiv:1606.07947](http://arxiv.org/abs/1606.07947) *Sequence-Level Knowledge Distillation* (2016, 1192 cites) | Introduces word-level and sequence-level KD (training on teacher beam outputs) for NMT. Shows that SeqKD lets a small student match much of the teacher and can even remove the need for beam search. This is the canonical off-policy baseline. | The student is trained only on teacher prefixes, so exposure bias is untreated. It offers no analysis of when training on the teacher's mode helps or hurts, beyond NMT/BLEU. |
| [arXiv:2306.13649](http://arxiv.org/abs/2306.13649) *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (2023, 272 cites) | GKD trains on a mixture of fixed and student-generated sequences with flexible divergences (FKL, RKL, generalized JSD). It shows on-policy data generally helps on summarization, translation and arithmetic reasoning. Its student-data fraction is the most direct experimental knob for on- vs off-policy. | Optimal divergence and mixing ratio are task-dependent and chosen empirically. There is no predictive account of when on-policy wins, and experiments use T5-scale NLP tasks with quality metrics rather than validity/structural-error metrics. |
| [arXiv:2306.08543](http://arxiv.org/abs/2306.08543) *MiniLLM: On-Policy Distillation of Large Language Models* (2023, 474 cites) | MiniLLM argues for reverse KL for generative LLM distillation and optimizes it with policy gradient on student samples, including variance-reduction tricks. It reports better calibration and fewer degenerate long generations. | The gains mix two factors, RKL and on-policy sampling, which are not cleanly ablated. Policy-gradient training adds variance and cost, and the evaluation focuses on instruction following. |
| [arXiv:2009.07253](http://arxiv.org/abs/2009.07253) *Autoregressive Knowledge Distillation through Imitation Learning* (2020, 70 cites) | Casts autoregressive KD as imitation learning with the teacher as oracle on student-visited states. It outperforms SeqKD on translation and summarization. | It predates LLM-scale on-policy work, and the comparison covers a limited range of student/teacher size gaps. It is not analyzed under divergence choice. |
| [arXiv:2307.15190](http://arxiv.org/abs/2307.15190) *f-Divergence Minimization for Sequence-Level Knowledge Distillation* (2023, 90 cites) | The f-DISTILL framework unifies SeqKD and ENGINE as approximations of f-divergence minimization. It derives step-wise decompositions and compares KL, RKL, JS and TVD variants. | It focuses on the objective rather than the sampling distribution. Whether a given divergence's benefit persists under on- vs off-policy sampling is not systematically crossed. |
| [arXiv:2409.12512](http://arxiv.org/abs/2409.12512) *Exploring and enhancing the transfer of distribution in knowledge distillation for autoregressive language models* (2026, 3 cites) | Argues that RKL mimics only some teacher features. It finds that student-generated outputs are costly and hard to optimize when the student is much smaller than the teacher, and proposes remedies. | The evidence is LLM-specific. The capacity-gap threshold at which on-policy stops helping is not quantified as a curve. |
| [arXiv:2603.07079](https://arxiv.org/abs/2603.07079) *Entropy-Aware On-Policy Distillation of Language Models* (2026, 153 cites) | Shows that RKL-based on-policy distillation reduces diversity and gives unstable signals when teacher entropy is high. Adds forward KL selectively on high-entropy tokens. | The entropy gate is heuristic. Results concern LM tasks where diversity matters, and the effect on validity-type metrics is unexplored. |
| [arXiv:2605.22731v1](http://arxiv.org/abs/2605.22731v1) *Post-Training is About States, Not Tokens: A State Distribution View of SFT, RL, and On-Policy Distillation* (2026) | Controlled study (Qwen3-0.6B, GSM8K) that formalizes post-training as state-distribution shaping. It separates the effect of which prefixes are supervised from the loss used. | Single small model and a single task. Whether its conclusions hold for non-language transduction or for different teacher-student gaps is untested. |
| [arXiv:2604.00626](https://arxiv.org/abs/2604.00626) *A Survey of On-Policy Distillation for Large Language Models* (2026, 193 cites) | Survey that unifies on-policy distillation methods around student states, teacher information and policy updates. Useful for situating method variants. | A survey adds no new controlled evidence, and its comparisons inherit the heterogeneity of the underlying papers. |
| [arXiv:2602.12275](http://arxiv.org/abs/2602.12275) *On-Policy Context Distillation for Language Models* (2026, 212 cites) | Extends on-policy RKL distillation to context distillation, where the teacher is the same model conditioned on extra context. | Different setting: the teacher and student share capacity, so lessons about capacity gaps do not transfer. |
| [doi:10.1007/s00521-025-11162-0](https://doi.org/10.1007/s00521-025-11162-0) *Mitigating exposure bias in large language model distillation: an imitation learning approach* (2025, 33 cites) | Imitation-learning approach to mitigating exposure bias in LLM distillation. | Abstract-level evidence only. Its relation to GKD/MiniLLM and the controls used are unclear. |
| [arXiv:2406.09393](http://arxiv.org/abs/2406.09393) *Improving Autoregressive Training with Dynamic Oracles* (2024, 0 cites) | Builds dynamic oracles for DAgger on metrics such as span F1, ROUGE and BLEU, preserving no-regret guarantees for decomposable metrics. | Not distillation per se: the oracle is metric-derived rather than a teacher model. Guarantees are limited to decomposable metrics. |
| [arXiv:1811.04531](http://arxiv.org/abs/1811.04531) *Sequence-level Knowledge Distillation for Model Compression of Attention-based Sequence-to-sequence Speech Recognition* (2019, 28 cites) | Applies SeqKD with multiple beam hypotheses to attention-based ASR, achieving 9.8x compression. | Off-policy only, with a WER cost of up to 7%. There is no comparison with student-sampled training. |
| [doi:10.1109/icassp.2019.8682671](https://doi.org/10.1109/icassp.2019.8682671) *Investigation of Sequence-level Knowledge Distillation Methods for CTC Acoustic Models* (2019, 27 cites) | Lattice-based sequence-level KD for CTC acoustic models. Shows that frame-level KD fails for CTC while sequence-level KD works. | Specific to CTC (non-autoregressive alignment), so it is only indirectly informative for autoregressive on-policy questions. |
| [arXiv:2609.09123](https://arxiv.org/abs/2609.09123) *Mask Forcing: Improving Autoregressive Video Diffusion Distillation via Dual-Noise Masking Rollout* (2026, 5 cites) | Identifies RKL mode-seeking as the cause of over-saturation in autoregressive video diffusion distillation. Perturbs student self-rollouts with dual-noise masking. | Diffusion/video domain. Its rollout perturbation idea has not been tested for discrete token transduction. |
| [arXiv:2610.26311v1](http://arxiv.org/abs/2610.26311v1) *Selective Transfer through Sequence Distillation: From Benchmark Knowledge to Backdoor Behaviour* (2026) | Shows that different teacher properties transfer very differently through the same distillation channels, with benchmark knowledge surviving even via number sequences. | About what transfers, not about on- vs off-policy efficiency. The channels are not mapped onto student-sampled training. |

## Corpus

- [arXiv:2307.15190](http://arxiv.org/abs/2307.15190) *f-Divergence Minimization for Sequence-Level Knowledge Distillation* (2023, 90 cites)
- [arXiv:2306.13649](http://arxiv.org/abs/2306.13649) *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (2023, 272 cites)
- [arXiv:1606.07947](http://arxiv.org/abs/1606.07947) *Sequence-Level Knowledge Distillation* (2016, 1192 cites)
- [arXiv:2009.07253](http://arxiv.org/abs/2009.07253) *Autoregressive Knowledge Distillation through Imitation Learning* (2020, 70 cites)
- [arXiv:2409.12512](http://arxiv.org/abs/2409.12512) *Exploring and enhancing the transfer of distribution in knowledge distillation for autoregressive language models* (2026, 3 cites)
- [arXiv:1811.04531](http://arxiv.org/abs/1811.04531) *Sequence-level Knowledge Distillation for Model Compression of Attention-based Sequence-to-sequence Speech Recognition* (2019, 28 cites)
- [arXiv:2106.10885v1](http://arxiv.org/abs/2106.10885v1) *Knowledge Distillation via Instance-level Sequence Learning* (2021)
- [arXiv:2406.09393](http://arxiv.org/abs/2406.09393) *Improving Autoregressive Training with Dynamic Oracles* (2024, 0 cites)
- [arXiv:2609.09123](https://arxiv.org/abs/2609.09123) *Mask Forcing: Improving Autoregressive Video Diffusion Distillation via Dual-Noise Masking Rollout* (2026, 5 cites)
- [arXiv:2306.08543](http://arxiv.org/abs/2306.08543) *MiniLLM: On-Policy Distillation of Large Language Models* (2023, 474 cites)
- [arXiv:2004.03846](http://arxiv.org/abs/2004.03846) *Structure-Level Knowledge Distillation For Multilingual Sequence Labeling* (2020, 35 cites)
- [arXiv:2602.12275](http://arxiv.org/abs/2602.12275) *On-Policy Context Distillation for Language Models* (2026, 212 cites)
- [doi:10.1007/s00521-025-11162-0](https://doi.org/10.1007/s00521-025-11162-0) *Mitigating exposure bias in large language model distillation: an imitation learning approach* (2025, 33 cites)
- [arXiv:2604.00626](https://arxiv.org/abs/2604.00626) *A Survey of On-Policy Distillation for Large Language Models* (2026, 193 cites)
- [arXiv:2610.26311v1](http://arxiv.org/abs/2610.26311v1) *Selective Transfer through Sequence Distillation: From Benchmark Knowledge to Backdoor Behaviour* (2026)
- [doi:10.1016/j.neucom.2020.11.025](https://doi.org/10.1016/j.neucom.2020.11.025) *Mutual-learning sequence-level knowledge distillation for automatic speech recognition* (2020, 28 cites)
- [arXiv:2310.09983](http://arxiv.org/abs/2310.09983) *Farzi Data: Autoregressive Data Distillation* (2023, 11 cites)
- [arXiv:2605.22731v1](http://arxiv.org/abs/2605.22731v1) *Post-Training is About States, Not Tokens: A State Distribution View of SFT, RL, and On-Policy Distillation* (2026)
- [arXiv:2603.07079](https://arxiv.org/abs/2603.07079) *Entropy-Aware On-Policy Distillation of Language Models* (2026, 153 cites)
- [doi:10.1109/icassp.2019.8682671](https://doi.org/10.1109/icassp.2019.8682671) *Investigation of Sequence-level Knowledge Distillation Methods for CTC Acoustic Models* (2019, 27 cites)

*Retrieved from arXiv / OpenAlex and reviewed by ScholarLoop's LitReviewer agent. Every cited id is checked against the retrieved corpus.*
