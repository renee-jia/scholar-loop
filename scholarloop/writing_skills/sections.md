# Section guides (what each section owes the reader)

Every section and subsection runs Goal → Problem → Solution: say what you are trying to achieve and
what makes it hard before saying how. State the section's point in its first one or two sentences.

## Abstract
Five sentences (see narrative.md). Numbers in the abstract must be the same numbers as in the
results, at the same precision, from the same setting.

## Introduction
Task in one sentence (what output from what input). The challenge prior work leaves, stated as a
limitation plus its technical reason. Our move, and why it should work. The evidence, in one
sentence with the headline number. Contribution bullets that each name a result. No related-work
survey here; one or two anchoring citations only.

## Related work
Group by mechanism, not by paper or year: 2-4 topics, each with a topic sentence, one compact
summary of the representative methods, the limitation relevant to our challenge, and the exact
difference of our approach in technical terms. Cite the one or two works that matter and say why;
never a bracketed list. Do not hide the strongest baseline.

## Method / setup
Enough to reimplement: the data and split, the model shape, every hyperparameter that was set, the
seeds, the budget per run, how the metric is computed and who computes it (the frozen scorer). For
each design choice: what it is, why it is needed, why it should help. Present the final design;
ablations belong in experiments.

## Experiments / results
Before each result, say what claim it tests. For each comparison, state the setting, the number of
seeds, and the spread (std or SEM, and which). Tell the reader what to look at: "the 5-seed mean
drops from 4.22 to 3.82 while the spread stays under 0.2". Report failures honestly: ideas that died
at the smoke screen, seeds that crashed, deltas that went the wrong way. Keep decimal precision
consistent within a metric. Direction of the metric is stated once and never left ambiguous.

## Discussion
Analytical prose, not a list: claim → evidence → mechanism → principle. Separate what was measured
(assertive) from why it happened (hedged: "we hypothesize", "this is consistent with").

## Limitations (required)
Tie each limitation to a scope boundary: data regime, budget, single dataset, seed count, the
reconstructable validation split. Say why it does not undermine the core claim, or what would.
Honesty here is rewarded by reviewers; omission is not.

## Conclusion
Restate the question and the answer, the strongest evidence in one sentence, the insight, one
limitation, one concrete next step. Do not re-run the abstract with cosmetic changes.
