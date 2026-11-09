# Narrative: the paper is one argument, not a list of experiments

## The one-sentence question test (gates everything)
Before any outline, write: "The question this paper answers is ______." It passes only if
1. it is a complete, specific question (not a topic, not "we apply X to Y"),
2. a reader can see from the sentence alone that the question is currently open,
3. a reader knows what evidence would settle it.
If the sentence fails, fix it before anything else. Every experiment is evidence only relative to a
fixed question; more results do not grow a question.

## The nugget
State the one insight a reader should take away. It is not the method; it is why the method works
or the finding that changes how the reader thinks. Every section, table and figure serves it. If you
cannot state it in one sentence, the paper's focus is unclear.

## What / Why / So-what (must be clear by the end of the introduction)
- **What**: 1-3 specific claims within one theme ("X improves Y by Z under condition W").
- **Why believe it**: the evidence: which runs, how many seeds, against which baseline.
- **So what**: why a reader who does not work on this exact setup should care.

## Contributions
Each contribution names a specific result with its number and comparison, never a restatement of
the abstract. Bad: "(1) a novel method; (2) extensive experiments." Good: "(1) a 5-seed confirmed
drop from 5.0% to 3.82% error from warmup + weight decay alone; (2) evidence that depth beyond 2
layers hurts on this data."

## Negative results are results
A method that fails with a clean, explainable failure mode is a legitimate finding. What is fatal is
a method that works with no account of why. Report killed and discarded runs when they bound the
claim ("depth 4 diverged at every learning rate tried").

## Scope discipline
Every conclusive sentence sits inside the evidence that supports it: name the dataset, the metric,
the seeds, the setting. "On digits under a 60-second budget" is not weakness; it is what makes the
remaining claims credible. One claim beyond the evidence makes a reviewer re-doubt every other one.

## Abstract (Farquhar's five sentences)
1. What you achieved ("We show…", "We find…"). 2. Why it is hard or matters. 3. How (with the
specialist keywords). 4. What evidence you have. 5. The single most remarkable number.
Delete any opening sentence that could be prepended to any paper in the field.

## Introduction logic (backward first, then forward)
Answer backward: what problem, why no established answer, what is the contribution, why does it
work, what is the new insight. Then write forward: task → the challenge prior methods leave → our
move → why it works → the evidence → 2-4 contribution bullets. Methods must start by page 2-3.
