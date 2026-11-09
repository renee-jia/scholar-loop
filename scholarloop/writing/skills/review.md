# Reviewing a draft (persona reviewers + the self-consistency audit)

## Dimensions (NeurIPS/ICML style)
- **Quality**: are the claims technically sound and supported by the reported runs? Seeds, spread,
  baseline, direction of the metric.
- **Clarity**: could an expert reproduce the setup from the text? Is terminology stable? Does each
  paragraph carry one message with a stated first sentence?
- **Significance**: does the result change what a practitioner in this setting would do?
- **Originality**: is there a non-obvious insight, or only a predictable gain from known knobs?

## Self-consistency audit (most rejections come from the paper contradicting itself)
1. Abstract vs experiments: put each selling-point sentence of the abstract next to the setup
   detail that could contradict it; both must be able to be true.
2. Conclusions vs evidence scope: every conclusive sentence names the exact scope it was measured
   in (dataset, metric, seeds, budget). Flag any sentence that reaches beyond.
3. Sentence-level traceability: every conclusive sentence points to a table, number, run id or
   citation. A sentence with no source is either derived (show it) or decoration (cut it).
4. Volume is not direction: more runs do not answer a question that was never fixed.

## Common rejection signals to look for
- Insufficient contribution: the technique is well explored and the gain is predictable.
- Unclear writing: a design choice with no stated motivation; missing details for reproduction.
- Weak effect: a marginal improvement, or strong relative gain with weak absolute numbers.
- Incomplete evaluation: no spread reported, no strongest baseline, a single setting.
- Overreach: claims beyond the dataset/budget/seed scope that was run.

## Output discipline
Each weakness names the section and the sentence it concerns and what evidence would fix it.
Scores are honest: a marginal, single-dataset result with a clean write-up is a 3-4, not a 7.
Strengths are specific ("the 5-seed confirmation with reported spread"), not generic ("well written").
