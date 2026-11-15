# Run process report

Cost and convergence from the run's own records. Dollar figures are estimates from token counts; the spend file is the total the cap was enforced on.

| trace | agent calls | attempts | first attempt rejected | never accepted | model seconds | input tokens | output tokens | USD |
|---|---|---|---|---|---|---|---|---|
| experiments.jsonl.agents.jsonl | 61 | 61 | 0 | 0 | 1303 | 905288 | 129478 | 4.82 |
| ideation.agents.jsonl | 12 | 14 | 2 | 0 | 460 | 69142 | 43225 | 1.14 |
| reliability/claims.agents.jsonl | 3 | 3 | 0 | 0 | 205 | 432381 | 23651 | 1.29 |
| review.agents.jsonl | 5 | 6 | 1 | 0 | 645 | 915754 | 68545 | 3.10 |
| writing.agents.jsonl | 30 | 30 | 0 | 0 | 479 | 317769 | 51059 | 2.29 |

Traced: $12.65. Spend file: $17.00 over 160 requests. The difference is requests that left no reading in a trace.

## Scientific review

| round | status | critics | new blocker | new major | new minor | open | approvals so far |
|---|---|---|---|---|---|---|---|
| 1 | NEEDS_REFRAMING | 4 | 0 | 5 | 21 | 26 | 0 |
| 2 | READY_FOR_AUTHOR | 4 | 0 | 0 | 0 | 0 | 0 |
| 2 | READY_TO_WRITE | 4 | 0 | 0 | 0 | 0 | 1 |

## Writing sessions

| directory | stage reached | outline rounds | key-point rounds | stopped for missing evidence | sections | reviewer scores |
|---|---|---|---|---|---|---|
| . | reviewed | 2 | 1 | no | 11 | [3, 2, 3] |
