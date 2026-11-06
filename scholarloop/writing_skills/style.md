# Prose style (enforced in part by `lint_prose`)

## Sentence mechanics (Gopen & Swan)
- Keep subject and verb close; put the action in the verb ("we analyzed", not "we performed an
  analysis").
- Old information first, new information last; the end of the sentence is the stress position, so
  put the number or the point there.
- One paragraph, one message, stated in its first sentence; every sentence links to the previous
  one by cause, contrast, consequence or refinement; close the paragraph with a consequence, not a
  trailing citation.
- One idea per sentence. If a sentence needs "while", "unlike" or a semicolon to hold two claims,
  make it two sentences. Sentences past ~30 words are suspect.

## Word choice (Lipton, Perez, Steinhardt)
- Be specific: "error", "RMSE", "wall-clock", never "performance".
- Pick one term per concept and keep it for the whole paper.
- Delete intensifiers and fillers: very, really, quite, actually, basically, essentially,
  extensive, comprehensive, numerous, various, "in order to", "it is worth noting that".
- Minimize bare pronouns: "this result shows", not "this shows".
- Prefer short words: use, show, help.

## Calibrated confidence
Assert measured facts ("drops to 3.82%"). Hedge mechanisms ("we hypothesize", "is consistent
with"). Never hedge a number ("may improve by 1.2") and never assert a cause you did not test.
Empirical work *shows* and *provides evidence*; it does not *prove* or *demonstrate* universal
truths. "Significantly" only with a test or a stated spread.

## Not allowed (reads as machine-written; the linter rejects these)
- Vocabulary: delve, leverage (as a verb), landscape (abstract sense), tapestry, testament,
  underscore(s), pivotal, showcase, foster, realm, seamless, multifaceted, paradigm shift,
  groundbreaking, intricate, crucial/vital (as hollow intensifiers).
- Em-dashes. Recast with a comma, colon, parentheses or a new sentence.
- "not X, but Y" / "not only … but also" contrast frames. State Y.
- Consecutive sentences opening with Moreover / Furthermore / Additionally / Notably / Importantly.
- Formulaic openers: "In recent years", "With the rapid development of", "Despite recent advances".
- Significance hype: paves the way, sheds light on, bridges the gap, opens new avenues, a crucial
  step toward, has the potential to revolutionize.
- Novelty padding: "novel" more than once per section, "to the best of our knowledge",
  "for the first time".
- Contribution clichés: "extensive experiments", "strong results", "a novel framework".

## Numbers
Every number in the text is a recorded measurement, a logged configuration value, or the stated
baseline, written at the recorded precision or rounded consistently. Do not derive new numbers in
prose (differences, percentages of improvement) unless the key points list them as derived with
their operands. Give the direction of every metric once.

## Citations
Cite only ids from the reviewed corpus or the ledger. If a claim needs a citation you do not have,
write "[citation needed]" rather than inventing one.
