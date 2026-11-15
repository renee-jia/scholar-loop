# Run process report

Cost and convergence from the run's own records. Dollar figures are estimates from token counts; the spend file is the total the cap was enforced on.

| trace | agent calls | attempts | first attempt rejected | never accepted | model seconds | input tokens | output tokens | USD |
|---|---|---|---|---|---|---|---|---|
| experiments.jsonl.agents.jsonl | 63 | 68 | 5 | 0 | 1292 | 1119167 | 127291 | 4.87 |
| ideation.agents.jsonl | 18 | 23 | 5 | 0 | 580 | 118489 | 56363 | 1.60 |
| paper2/writing.agents.jsonl | 29 | 33 | 4 | 0 | 543 | 368158 | 59138 | 2.66 |
| review.agents.jsonl | 19 | 25 | 4 | 2 | 3622 | 3467622 | 316989 | 12.46 |
| writing.agents.jsonl | 8 | 9 | 1 | 0 | 141 | 124661 | 15108 | 0.80 |

Traced: $22.39. Spend file: $27.17 over 202 requests. The difference is requests that left no reading in a trace.

## Scientific review

| round | status | critics | new blocker | new major | new minor | open | approvals so far |
|---|---|---|---|---|---|---|---|
| 1 | NEEDS_REFRAMING | 4 | 0 | 7 | 8 | 15 | 0 |
| 2 | REVIEW_FAILED | 0 | 0 | 0 | 0 | 15 | 0 |
| 3 | REVIEW_FAILED | 0 | 0 | 0 | 0 | 15 | 0 |
| 4 | READY_FOR_AUTHOR | 4 | 0 | 0 | 3 | 0 | 0 |
| 4 | READY_TO_WRITE | 4 | 0 | 0 | 3 | 0 | 1 |
| 5 | NEEDS_REFRAMING | 4 | 0 | 3 | 8 | 3 | 1 |
| 6 | NEEDS_REFRAMING | 4 | 0 | 1 | 3 | 2 | 1 |
| 7 | READY_FOR_AUTHOR | 4 | 0 | 0 | 1 | 0 | 1 |
| 7 | READY_TO_WRITE | 4 | 0 | 0 | 1 | 0 | 2 |

## Writing sessions

| directory | stage reached | outline rounds | key-point rounds | stopped for missing evidence | sections | reviewer scores |
|---|---|---|---|---|---|---|
| paper2 | reviewed | 2 | 1 | no | 10 | [3, 2, 3] |
| . | outlined | 2 | 0 | yes | 0 |  |
