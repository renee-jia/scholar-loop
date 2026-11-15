# Guard reliability: fault injection

Detection of single-site injected faults by the production postchecks. Fabrication faults refer to something absent from the evidence; misattribution faults use real evidence in the wrong place. In legacy mode a lexical check cannot see misattribution. In reviewed mode numbers are bound to the statements a sentence cites, so part of it is visible.

Guard mode: reviewed. Seed 0, at most 8 sites per fault and artifact. Unmodified controls: 0 of 12 accepted artifacts rejected.

## Detection by fault

| tier | fault | detected |
|---|---|---|
| fabrication | dropped_limitations | 1/1 (100%, 95% CI 21-100) |
| fabrication | dropped_section | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_citation | 53/53 (100%, 95% CI 93-100) |
| fabrication | fabricated_number | 33/33 (100%, 95% CI 90-100) |
| fabrication | fabricated_statement_id | 72/72 (100%, 95% CI 95-100) |
| fabrication | near_miss_citation | 21/21 (100%, 95% CI 85-100) |
| fabrication | uncited_claim | 13/13 (100%, 95% CI 77-100) |
| fabrication | uncited_sentence | 56/56 (100%, 95% CI 94-100) |
| misattribution | direction_flip | 16/53 (30%, 95% CI 20-44) |
| misattribution | misattributed_number | 33/33 (100%, 95% CI 90-100) |
| misattribution | miscitation | 8/21 (38%, 95% CI 21-59) |
| misattribution | wrong_statement_id | 20/72 (28%, 95% CI 19-39) |

| tier | detected |
|---|---|
| fabrication | 257/257 (100%, 95% CI 99-100) |
| misattribution | 77/179 (43%, 95% CI 36-50) |

## Where faults pass, by pipeline stage

| tier | fault | stage | detected |
|---|---|---|---|
| fabrication | dropped_limitations | outline | 1/1 (100%, 95% CI 21-100) |
| fabrication | dropped_section | key_points | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_citation | ideas | 5/5 (100%, 95% CI 57-100) |
| fabrication | fabricated_citation | key_points | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_citation | lit_review | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_citation | outline | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_citation | section | 24/24 (100%, 95% CI 86-100) |
| fabrication | fabricated_number | key_points | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_number | outline | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_number | section | 17/17 (100%, 95% CI 82-100) |
| fabrication | fabricated_statement_id | key_points | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_statement_id | outline | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_statement_id | section | 56/56 (100%, 95% CI 94-100) |
| fabrication | near_miss_citation | ideas | 5/5 (100%, 95% CI 57-100) |
| fabrication | near_miss_citation | lit_review | 8/8 (100%, 95% CI 68-100) |
| fabrication | near_miss_citation | section | 8/8 (100%, 95% CI 68-100) |
| fabrication | uncited_claim | ideas | 5/5 (100%, 95% CI 57-100) |
| fabrication | uncited_claim | key_points | 8/8 (100%, 95% CI 68-100) |
| fabrication | uncited_sentence | section | 56/56 (100%, 95% CI 94-100) |
| misattribution | direction_flip | key_points | 8/8 (100%, 95% CI 68-100) |
| misattribution | direction_flip | outline | 8/8 (100%, 95% CI 68-100) |
| misattribution | direction_flip | section | 0/37 (0%, 95% CI 0-9) |
| misattribution | misattributed_number | key_points | 8/8 (100%, 95% CI 68-100) |
| misattribution | misattributed_number | outline | 8/8 (100%, 95% CI 68-100) |
| misattribution | misattributed_number | section | 17/17 (100%, 95% CI 82-100) |
| misattribution | miscitation | ideas | 0/5 (0%, 95% CI 0-43) |
| misattribution | miscitation | lit_review | 0/8 (0%, 95% CI 0-32) |
| misattribution | miscitation | section | 8/8 (100%, 95% CI 68-100) |
| misattribution | wrong_statement_id | key_points | 8/8 (100%, 95% CI 68-100) |
| misattribution | wrong_statement_id | outline | 0/8 (0%, 95% CI 0-32) |
| misattribution | wrong_statement_id | section | 12/56 (21%, 95% CI 13-34) |

## Ablation (one guard disabled at a time)

Only guards whose removal changes the count are listed.

| tier | configuration | detected |
|---|---|---|
| fabrication | all_guards | 257/257 (100%, 95% CI 99-100) |
| fabrication | without_citation | 221/257 (86%, 95% CI 81-90) |
| fabrication | without_structure | 36/257 (14%, 95% CI 10-19) |
| misattribution | all_guards | 77/179 (43%, 95% CI 36-50) |
| misattribution | without_structure | 0/179 (0%, 95% CI 0-2) |
