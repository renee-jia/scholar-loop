# Digits: a skip connection against a parameter-matched control

One complete run with `claude-opus-5-5` on the `digits-mlp` engine. The author supplied the
idea: separate the effect of a residual path from the effect of added parameters in a shallow
MLP, with a parameter-matched control under the same training resources.

The first experiments ran as an open campaign, which compared the two arms on one seed and spent
the rest of its budget on batching, weight decay and width. Review called the question
unresolved, and the author requested a frozen protocol: both arms on twenty paired seeds at two
shared learning rates, predicting that the paired difference stays within half a point. At the
higher rate it did (2.58 against 2.70); at the lower rate it did not (2.91 against 3.64). The
approved argument says the protocol neither supports nor falsifies the prediction.

Scientific review took seven rounds and two author approvals. The simulated reviewers gave the
paper 2.7 out of 10. Model spend through the reviewed paper: $25.66.

[`../digits_hyperparameter_tuning`](../digits_hyperparameter_tuning/) is a different, earlier run on the
same engine, in which the picked idea was never tested.

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
