# Guard reliability: natural fault rate

A rejected first attempt would have been used by a harness without checks. Rates describe these runs and this model only.

| agent | calls | first attempt rejected | by guard | recovered | unrecovered | mean attempts |
|---|---|---|---|---|---|---|
| ALL | 127 | 22 (17%) | citation 3, invalid_json 1, structure 18 | 15 | 7 | 1.228 |
| argument_builder | 4 | 4 (100%) | structure 4 | 1 | 3 | 2.5 |
| claim_fidelity_reviewer | 2 | 0 (0%) | none | 0 | 0 | 1.0 |
| direction_mapper | 1 | 1 (100%) | citation 1 | 1 | 0 | 2.0 |
| idea_study_planner | 3 | 3 (100%) | invalid_json 1, structure 2 | 1 | 2 | 2.333 |
| ideator:contradiction | 1 | 1 (100%) | citation 1 | 1 | 0 | 2.0 |
| ideator:mechanism | 2 | 1 (50%) | structure 1 | 0 | 1 | 1.0 |
| ideator:transfer | 1 | 1 (100%) | citation 1 | 1 | 0 | 2.0 |
| key_pointer | 4 | 3 (75%) | structure 3 | 3 | 0 | 2.25 |
| lit_reviewer | 1 | 0 (0%) | none | 0 | 0 | 1.0 |
| novelty_critic | 5 | 0 (0%) | none | 0 | 0 | 1.0 |
| outliner | 13 | 3 (23%) | structure 3 | 3 | 0 | 1.231 |
| panel:Editor | 16 | 1 (6%) | structure 1 | 1 | 0 | 1.125 |
| panel:Methodologist | 16 | 0 (0%) | none | 0 | 0 | 1.0 |
| panel:Skeptic | 17 | 1 (6%) | structure 1 | 0 | 1 | 1.118 |
| query_planner | 1 | 0 (0%) | none | 0 | 0 | 1.0 |
| reviewer:harsh-but-fair | 1 | 0 (0%) | none | 0 | 0 | 1.0 |
| reviewer:harsh-critical | 1 | 0 (0%) | none | 0 | 0 | 1.0 |
| reviewer:open-minded | 1 | 0 (0%) | none | 0 | 0 | 1.0 |
| scientific_critic:argument | 7 | 0 (0%) | none | 0 | 0 | 1.0 |
| scientific_critic:experimental | 7 | 0 (0%) | none | 0 | 0 | 1.0 |
| scientific_critic:falsification | 7 | 0 (0%) | none | 0 | 0 | 1.0 |
| scientific_critic:implementation | 7 | 1 (14%) | structure 1 | 1 | 0 | 1.286 |
| section_drafter | 9 | 2 (22%) | structure 2 | 2 | 0 | 1.222 |
