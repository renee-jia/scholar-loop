# Guard reliability: fault injection

Detection of single-site injected faults by the production postchecks. Fabrication faults refer to something absent from the evidence; misattribution faults use real evidence in the wrong place. In legacy mode a lexical check cannot see misattribution. In reviewed mode numbers are bound to the statements a sentence cites, so part of it is visible.

Guard mode: reviewed. Seed 0, at most 8 sites per fault and artifact. Unmodified controls: 0 of 15 accepted artifacts rejected.

## Detection by fault

| tier | fault | detected |
|---|---|---|
| fabrication | dropped_limitations | 1/1 (100%, 95% CI 21-100) |
| fabrication | dropped_section | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_citation | 63/63 (100%, 95% CI 94-100) |
| fabrication | fabricated_number | 70/70 (100%, 95% CI 95-100) |
| fabrication | fabricated_statement_id | 102/102 (100%, 95% CI 96-100) |
| fabrication | near_miss_citation | 16/16 (100%, 95% CI 81-100) |
| fabrication | uncited_claim | 14/14 (100%, 95% CI 78-100) |
| fabrication | uncited_sentence | 86/86 (100%, 95% CI 96-100) |
| misattribution | direction_flip | 14/33 (42%, 95% CI 27-59) |
| misattribution | misattributed_number | 65/70 (93%, 95% CI 84-97) |
| misattribution | miscitation | 2/16 (12%, 95% CI 4-36) |
| misattribution | wrong_statement_id | 42/102 (41%, 95% CI 32-51) |

| tier | detected |
|---|---|
| fabrication | 360/360 (100%, 95% CI 99-100) |
| misattribution | 123/221 (56%, 95% CI 49-62) |

## Where faults pass, by pipeline stage

| tier | fault | stage | detected |
|---|---|---|---|
| fabrication | dropped_limitations | outline | 1/1 (100%, 95% CI 21-100) |
| fabrication | dropped_section | key_points | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_citation | ideas | 6/6 (100%, 95% CI 61-100) |
| fabrication | fabricated_citation | key_points | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_citation | lit_review | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_citation | outline | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_citation | section | 33/33 (100%, 95% CI 90-100) |
| fabrication | fabricated_number | key_points | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_number | outline | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_number | section | 54/54 (100%, 95% CI 93-100) |
| fabrication | fabricated_statement_id | key_points | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_statement_id | outline | 8/8 (100%, 95% CI 68-100) |
| fabrication | fabricated_statement_id | section | 86/86 (100%, 95% CI 96-100) |
| fabrication | near_miss_citation | ideas | 6/6 (100%, 95% CI 61-100) |
| fabrication | near_miss_citation | lit_review | 8/8 (100%, 95% CI 68-100) |
| fabrication | near_miss_citation | section | 2/2 (100%, 95% CI 34-100) |
| fabrication | uncited_claim | ideas | 6/6 (100%, 95% CI 61-100) |
| fabrication | uncited_claim | key_points | 8/8 (100%, 95% CI 68-100) |
| fabrication | uncited_sentence | section | 86/86 (100%, 95% CI 96-100) |
| misattribution | direction_flip | key_points | 8/8 (100%, 95% CI 68-100) |
| misattribution | direction_flip | outline | 6/6 (100%, 95% CI 61-100) |
| misattribution | direction_flip | section | 0/19 (0%, 95% CI 0-17) |
| misattribution | misattributed_number | key_points | 8/8 (100%, 95% CI 68-100) |
| misattribution | misattributed_number | outline | 8/8 (100%, 95% CI 68-100) |
| misattribution | misattributed_number | section | 49/54 (91%, 95% CI 80-96) |
| misattribution | miscitation | ideas | 0/6 (0%, 95% CI 0-39) |
| misattribution | miscitation | lit_review | 0/8 (0%, 95% CI 0-32) |
| misattribution | miscitation | section | 2/2 (100%, 95% CI 34-100) |
| misattribution | wrong_statement_id | key_points | 8/8 (100%, 95% CI 68-100) |
| misattribution | wrong_statement_id | outline | 0/8 (0%, 95% CI 0-32) |
| misattribution | wrong_statement_id | section | 34/86 (40%, 95% CI 30-50) |

## Ablation (one guard disabled at a time)

Only guards whose removal changes the count are listed.

| tier | configuration | detected |
|---|---|---|
| fabrication | all_guards | 360/360 (100%, 95% CI 99-100) |
| fabrication | without_citation | 320/360 (89%, 95% CI 85-92) |
| fabrication | without_structure | 40/360 (11%, 95% CI 8-15) |
| misattribution | all_guards | 123/221 (56%, 95% CI 49-62) |
| misattribution | without_structure | 0/221 (0%, 95% CI 0-2) |
