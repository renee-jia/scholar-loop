# AG News: a partial result with a missing comparator

One complete run with `claude-opus-5-5` on the `agnews-bow` engine, a bag-of-embeddings news
classifier. The author picked a transfer of three boosting regularizers: vocabulary-column
dropout, group-L1 on embedding rows, and shrinkage as a long low-rate schedule with weight
averaging.

Column dropout lowered validation error against the stock control on all three evaluated folds
(12.08 against 13.77) and weight averaging on all five (12.03 against 12.55). Group-L1 and the
long schedule showed no consistent effect. The argument records that the comparator the idea
named, tuned hidden dropout with weight decay, never ran.

Scientific review took two rounds and one author approval. The simulated reviewers gave the
paper 2.7 out of 10. Model spend through the reviewed paper: $13.39.

Open it in the web UI: start `python3 -m scholarloop.ui` from the repository root and click
**Explore a real run**. The run is read-only there.

| File | What it is |
|---|---|
| `steer.json`, `lit_review.md`, `directions.md`, `ideas.md` | the author's steer, the literature review, the direction map and the ideas with their tournament |
| `experiments.jsonl`, `registry/`, `experiment_agenda/` | the ledger of every training run, the frozen protocols and the code version of each arm |
| `scientific_review.md` | the argument, the four critics' reports and the author's release |
| `outline.md`, `key_points.md`, `paper.md`, `review.md` | the staged write-up and the simulated reviewers' reports |
| `paper_traced.md`, `evidence_map.md` | the approved statement behind every sentence, and the runs behind every statement |
| `reliability/` | fault injection, trace statistics and the claim audit for this run |

This is a trimmed copy of the run directory: agent traces, superseded review rounds and logs are
left out, and absolute paths are rewritten relative to the run. The approval was bound to the
interpreter, platform and package versions of the machine that ran it, so it is a record and
does not validate again elsewhere.
