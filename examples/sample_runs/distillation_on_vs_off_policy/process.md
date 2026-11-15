# Run process report

Cost and convergence from the run's own records. Dollar figures are estimates from token counts; the spend file is the total the cap was enforced on.

| trace | agent calls | attempts | first attempt rejected | never accepted | model seconds | input tokens | output tokens | USD |
|---|---|---|---|---|---|---|---|---|
| ideation.agents.jsonl | 12 | 15 | 4 | 1 | 456 | 98184 | 45539 | 1.30 |
| paper/writing.agents.jsonl | 8 | 8 | 0 | 0 | 150 | 110224 | 15587 | 0.75 |
| paper2/writing.agents.jsonl | 30 | 32 | 2 | 0 | 552 | 414551 | 60589 | 2.87 |
| reliability/claims.agents.jsonl | 6 | 7 | 2 | 1 | 665 | 508661 | 71566 | 2.04 |
| review.agents.jsonl | 35 | 47 | 8 | 5 | 3869 | 3532180 | 412449 | 16.60 |
| writing.agents.jsonl | 42 | 54 | 8 | 1 | 1459 | 692990 | 177025 | 6.31 |

Traced: $29.88. Spend file: $38.37 over 260 requests. The difference is requests that left no reading in a trace.

## Scientific review

| round | status | critics | new blocker | new major | new minor | open | approvals so far |
|---|---|---|---|---|---|---|---|
| 1 | NEEDS_EXPERIMENTS | 4 | 0 | 12 | 7 | 19 | 0 |
| 2 | REVIEW_FAILED | 0 | 0 | 0 | 0 | 19 | 0 |
| 3 | NEEDS_REFRAMING | 4 | 0 | 0 | 1 | 1 | 0 |
| 4 | READY_FOR_AUTHOR | 4 | 0 | 0 | 0 | 0 | 0 |
| 4 | READY_TO_WRITE | 4 | 0 | 0 | 0 | 0 | 1 |
| 5 | REVIEW_FAILED | 0 | 0 | 0 | 0 | 0 | 1 |
| 6 | REVIEW_FAILED | 0 | 0 | 0 | 0 | 0 | 1 |
| 7 | NEEDS_REFRAMING | 4 | 0 | 2 | 6 | 2 | 1 |
| 8 | READY_FOR_AUTHOR | 4 | 0 | 0 | 5 | 0 | 1 |
| 8 | READY_TO_WRITE | 4 | 0 | 0 | 5 | 0 | 2 |
| 9 | NEEDS_REFRAMING | 4 | 0 | 1 | 5 | 1 | 2 |
| 10 | READY_FOR_AUTHOR | 4 | 0 | 0 | 3 | 0 | 2 |
| 10 | READY_TO_WRITE | 4 | 0 | 0 | 3 | 0 | 3 |

## Writing sessions

| directory | stage reached | outline rounds | key-point rounds | stopped for missing evidence | sections | reviewer scores |
|---|---|---|---|---|---|---|
| paper | outlined | 2 | 0 | yes | 0 |  |
| paper2 | reviewed | 1 | 1 | no | 8 | [2, 2, 2.5] |
| . | keypointed | 2 | 3 | no | 0 |  |
