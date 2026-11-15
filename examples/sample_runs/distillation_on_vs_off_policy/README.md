# Language-model distillation: a falsified prediction

One complete run with `claude-opus-5-5` on the `seq-distill` engine, which distils a two-layer
student from a frozen teacher on a transduction task. The steer was one sentence: how on-policy
and off-policy distillation compare, and when each works better.

The author picked the idea of switching from teacher-sampled to student-sampled prefixes once the
student's invalid rate on a probe falls below a threshold. The protocol froze six arms over five
seeds and registered that the adaptive arm would beat both pure regimes by a point. It did not:
12.88% invalid decodes against 9.04% for pure on-policy and 14.0% for pure off-policy.

Scientific review took ten rounds and three author approvals. The critics forced three readings
to be withdrawn, and the approved argument states the falsification with sixteen limitations.
The simulated reviewers gave the paper 2.2 out of 10. Model spend through the reviewed paper:
$34.40.

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
