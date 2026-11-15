# ScholarLoop reference

The details behind the [README](README.md): command-line modes, review and release rules,
final-test controls, comparison harnesses and environment variables.

## Driving a run from the command line

The web UI runs these same commands as background jobs.

```bash
# 1 · ideation: steer, literature review, direction map, ideas, tournament, your pick
python -m scholarloop.ideation --profile profiles/digits-mlp.yaml \
    --topic "regularization recipes for small MLPs" --out runs/s1

# 2 · experiments: the campaign adopts the idea you picked
SCHOLARLOOP_HANDOFF=runs/s1/handoff.json SCHOLARLOOP_OUT=runs/s1 \
    python examples/run_to_paper.py
```

Scientific review, your approval, then writing:

```bash
python -m scholarloop.review review  --profile profiles/digits-mlp.yaml \
    --ledger runs/s1/experiments.jsonl --registry runs/s1/registry --research runs/s1 --out runs/s1
python -m scholarloop.review approve --profile profiles/digits-mlp.yaml \
    --ledger runs/s1/experiments.jsonl --registry runs/s1/registry --research runs/s1 --out runs/s1 \
    --author Researcher --review-hash EXACT_REVIEW_HASH
python -m scholarloop.writing --profile profiles/digits-mlp.yaml \
    --ledger runs/s1/experiments.jsonl --registry runs/s1/registry --research runs/s1 --out runs/s1
```

### Scripted and deeper ideation modes

`--steer examples/steer.json` (or `--interest "…" --hunch "…" --avoid "…"`) gives the steer
without a terminal. The run then stops at the direction map; `--directions 1,3`,
`--own-direction "…"` and `--direction-note "…"` record your choice and continue.
`--stop-after review|directions|ideas` writes that page and exits for a human. `--feedback "…"`
and `--select 2` resume from another process. `--auto` chooses every mapped direction and the
tournament's top idea; `--unsteered` waives the steer. Both are logged as not being your call.
`--deep --brief examples/research_brief.json` generates ideas by independent strategies
(mechanism, transfer, contradiction) with retrieval per idea.
`--decision reject|refine|merge|revive --idea-ids …` records your decisions against stable
idea IDs. `python -m scholarloop.ideation.pilot` runs a small validation-only pilot across
several ideas before you commit to one.

A meta-reviewer turns the judges' recurring complaints into at most five recommendations. Each
must cite the matches that show it, and the next round of ideas is written against them.

## Details by topic

<details>
<summary><b>Candidates, snapshots and what the Reasoner sees</b></summary>

The Reasoner and Director receive a bounded validation evidence view: configurations, scores,
seed/fold identities, failures and the frontier. It excludes held-out scores and scientific
control runs. Observed losing regions are soft warnings, not causal exclusions; repeated
fidelity tiers count once per candidate when deriving these associations.

Each new run records a `candidate_id` and a separate `run_spec`. Candidate snapshots in
`registry/candidates/` bind the requested configuration, local engine Python sources, frozen
scorer and declared execution context. Different implementations remain distinct even at
identical knobs; changing seed or fidelity creates a different run of the same candidate. The
Reasoner sees the frontier's source and inherits its code and requested knobs by default,
including after restarting a campaign. `base: "root"` explicitly starts from repository code and
profile defaults. Changing scorer or dependencies requires a new root candidate. Legacy runs
without snapshots remain readable but cannot supply inherited implementations.

Scientific review includes these snapshots and rejects missing or modified candidate files. They
document execution inputs; they do not certify undeclared data, external dependencies or
scientific validity. Frontier selection and convergence tracking use a common fidelity and
seed/fold scope, and prediction calibration uses matched reference observations when available.
The funnel's SEM margin is a resource-allocation heuristic, not a significance test.

Loop components: `Orchestrator.population_step(k, max_workers)` (parallel population funnel),
`Governor(max_cost, max_rounds, dry_patience)` (self-stopping), `CalibrationLog`
(predict-then-verify), `SkillLibrary.render(query=…)` (relevance-ranked, time-decaying lessons).

</details>

<details>
<summary><b>Scientific review, releases and the experiment agenda</b></summary>

Optional `--critic-model ROLE=MODEL` mappings choose different reviewer models; every critic
must answer evidence questions for each claim. `--max-llm-calls` caps cumulative CLI requests,
including retries, rather than dollars. Old v1 releases require re-review under the v2 contract.
Changed evidence, code, declared data or research scope invalidates a release. An existing
writing session bound to an older release needs a new output directory and `--review-dir`.

For `NEEDS_EXPERIMENTS`, use `python -m scholarloop.experiment.agenda plan` with the same
profile, ledger, registry, research and out arguments. Inspect the frozen protocol, then use
`run` with `--protocol-hash HASH --author Researcher` and re-run scientific review.

Use `SCHOLARLOOP_RESUME=1` to continue a campaign containing frozen protocols; fresh mode
refuses to erase that evidence. Protocol hashes detect changed inputs, not hostile execution.
`run_to_paper.py` pauses for the author's decision, including in unattended mode;
`SCHOLARLOOP_WRITING=oneshot` keeps the single-call writer behind the same gate.

In reviewed writing, claim dependencies and evidence IDs are validated, and a separate semantic
reviewer checks attribution, scope and mandatory limitations. A drafted section that drops one
of its agreed key points is flagged in the draft's notes.

</details>

<details>
<summary><b>Final-test controls and attempt logs</b></summary>

Final-test access closes that campaign to new exploration and experiment plans, including when
test scoring fails. A previously frozen confirmation protocol may finish its planned conditions;
review and writing may still inspect its results. This is a ledger-level guard, not OS isolation
or proof that a reused dataset is statistically independent. Historical campaigns containing
test results are also closed to further exploration.

Before a confirmation run starts, the runner persists and fsyncs a test-access receipt beside
its ledger in `<ledger>.test-access/`. Keep this directory with the ledger when moving a
campaign. An incomplete receipt or a receipt without its matching ledger result closes execution
and blocks scientific review; an interrupted run is never automatically replayed. Completed
receipts are part of the review snapshot and permit the original protocol to continue. Recovery
requires inspecting the run and reconciling its real evidence; deleting a receipt or restarting
with the same test data does not restore statistical independence.

Confirmation continuation is validated at the public runner boundary: a claimed protocol must
exist under its content hash and match the frozen context, condition, effective code and config,
seeds, fidelity and test-access setting. Each condition has one canonical run ID. Reusing a
protocol string cannot authorize another candidate or a repeat; exploration artifact caches
cannot be supplied to protocol confirmation runs. A standalone explicit confirmation without a
protocol remains a single access that closes its campaign.

Agent attempts are persisted as `*.agents.jsonl` by experiment, ideation, review and writing
sessions. Each attempt has a durable start event, input and schema fingerprints, model identity,
and a finish event containing the structured output, validation errors, status and elapsed time.
Raw prompts are not stored. An unmatched start denotes an unfinished attempt; provider errors
and invalid JSON remain visible. Client usage snapshots are cumulative and must not be treated
as per-attempt costs when requests share a client. `python -m scholarloop.evaluation` reports
these logs, including corrupt tails.

</details>

<details>
<summary><b>Search comparisons and multi-task suites</b></summary>

Search comparisons freeze declared profile and engine inputs and settings in
`comparison_plan.json`; resuming with different settings is rejected. Interrupted campaign
directories are preserved, failed campaigns are recorded, and budget-skipped repeats remain in
the reported denominator. Score intervals describe observed outcomes only. Run-matched campaigns
reserve each training attempt before dispatch and never exceed their configured count. Failed
and interrupted attempts remain charged; cached models do not consume a new reservation.
Accounting is persisted beside the ledger as `*.training_budget.json`. This caps training
attempts, not GPU seconds or financial cost. `compare_search.py --max-calls` caps model requests
persistently across retries and repeats (default 100); `--max-cost` is an estimated stop
threshold between campaigns. Unknown prices block cost-limited campaigns.

For multiple tasks, use `python -m scholarloop.experiment.suite --manifest suite.json --out runs/suite`.
The manifest has `version: 1` and a `tasks` array. Every task supplies `id`, `split`
(`development` or `evaluation`), `dataset_group`, `profile`, `seeds` and `train_budget`. Profile
paths are relative to the manifest. Both partitions are required; shared dataset groups or
profile dataset names across partitions are rejected. The suite freezes its declared inputs,
retains every planned repeat, and keeps per-task metrics separate. The CLI runs the
random-search baseline without API calls. Programmatic model arms must provide an orchestrator
with a persistent `CallLimitedLLM` budget. Dataset declarations and disjoint labels do not
establish real-world independence.

</details>

<details>
<summary><b>Research iterations after preliminary results</b></summary>

`ResearchCycle` separates scientific progress from parameter-search progress. Each assessment
reads the complete current dossier (including failures and negative results) and the latest five
iteration records, with older records retained on disk and an explicit omission count. It asks
whether the evidence is insufficient, promising within scope, or contradictory within scope;
these are provisional model assessments, not verified scientific conclusions.

A stable issue register is reconstructed from all rounds and supplied independently of the
five-round history window. Every assessment must update every tracked concern exactly once.
Proposing `addressed_pending_review` requires an existing dossier evidence ID absent when the
issue opened, plus a rationale. This checks provenance, not whether evidence logically resolves
the concern: addressed proposals remain visible and still require scientific review. Reworded
concerns remain separate rather than silently merging away uncertainty.

The next-step recommendation must name the primary and competing explanations, a discriminating
comparison, contrasting predictions, a falsifying outcome, claim limits and a training budget.
When issues already exist, an experimental recommendation must reference at least one tracked
issue ID and select a named competing explanation. It must state prospective decisions for
primary-supporting, alternative-supporting and inconclusive outcomes, along with what the
experiment cannot resolve. These fields accompany the recommendation into critic review and the
experiment planner; the review hash and frozen protocol include the original recommendation.
Critic requests take precedence over the provisional recommendation.

Iteration-derived protocols additionally require a `discrimination` mapping: reference and
treatment names, the reviewed competing explanation, why the contrast is informative, controlled
factors, residual confounds, inconclusive outcomes and departures from the recommendation.
Identical executable configurations for an ablation, falsification or boundary test are
rejected. This is a structural check, not a causal certificate.

Each reassessment must appraise every protocol in the current dossier, citing its protocol ID
and contrasting observations with the frozen predictions. Missing derived analysis requires
`execution_incomplete`; incomplete comparisons cannot favor either explanation. Actions include
debugging, replication, ablation, falsification, boundary tests, pivoting and stopping.
Recommendations cannot execute experiments, issue author approval or open the final test.

```bash
python -m scholarloop.experiment.research_cycle \
  --profile profiles/digits-mlp.yaml --ledger runs/study/experiments.jsonl \
  --registry runs/study/registry --out runs/study \
  --question "Does the mechanism explain the validation improvement?" \
  --max-training-attempts 12

python -m scholarloop.review review \
  --profile profiles/digits-mlp.yaml --ledger runs/study/experiments.jsonl \
  --registry runs/study/registry --out runs/study \
  --iteration runs/study/research_iterations.json
```

Review rejects a recommendation if its evidence snapshot is stale. Accepted experimental
requests proceed through the experiment agenda: freeze the protocol, execute its conditions,
reassess the new evidence, and review again. This closes a research iteration path; it does not
establish that the policy improves itself.

Pairwise ideation ranking receives bounded retrieved abstracts, provisional novelty comparisons,
the research brief and prior dissent. Position-swapped judgments and omission markers remain
explicit, and match records bind the evidence context by hash.

</details>

<details>
<summary><b>Controlled research-decision development harness</b></summary>

A free synthetic harness (no model, no training) for developing evaluations of research
decisions:

```bash
python -m scholarloop.experiment.decision_bench \
  --out runs/decision-development --repeats 3 --seed 0 --budget 3
```

Four single-cause task families share similar preliminary gains: leakage, extra compute,
population-specific effects, and gains that survive the three available controls. Policies
receive a public task plus their observed intervention results. The evaluator holds the truth
and scores decisions without a model as judge. Budget units are synthetic intervention costs.
The runner compares accept-all, abstain-all, fixed-order and random-order controls on the same
tasks and budget. `plan.json` freezes task generation, actions, budget, arms and implementation
hash; `report.json` retains every episode, failure, observation and decision. Failed and
exhausted episodes remain in the accuracy denominator.

This is an evaluator development fixture, not evidence that ScholarLoop improves research. The
full `ResearchCycle` is not yet a benchmark arm. Trusted in-process policies have only an
API-level information boundary. Reports contain truth labels and must not be fed back into an
agent evaluated on the same tasks.

**Model-policy ablation (uses the API, not part of tests):**

```bash
python -m scholarloop.experiment.decision_policy \
  --out runs/decision-model-development --model YOUR_MODEL_ID \
  --repeats 3 --budget 3 --max-calls 8
```

`plain` and `structured` policies use the same model, response schema, budget and per-episode
request cap; only their system instructions differ. `--max-calls` applies separately to each
task and arm, including schema-repair retries: three repeats, four families and two arms reserve
at most 192 requests. Prompts, schema, model, implementation and evaluator hashes are frozen with
the plan. Completed episodes are reused; an episode started without a saved result is retained
as interrupted and never silently rerun.

Add `--arms plain,structured,iteration` to include a stateful component arm that reuses the
production `IterationReviewer` validation and cross-round issue replay on explicitly marked
synthetic evidence. It evaluates the reviewer and state component, not the entire research
cycle, critic panel, frozen-protocol execution or manuscript workflow. With three arms the
maximum is 288 requests. No paid model comparison has been performed.

</details>

<details>
<summary><b>Environment variables for <code>examples/run_to_paper.py</code></b></summary>

| Variable | Meaning |
|---|---|
| `SCHOLARLOOP_OUT` | output directory (default `runs/live`) |
| `SCHOLARLOOP_PROFILE` | `profiles/<name>.yaml` (default `digits-mlp`) |
| `SCHOLARLOOP_MODEL` | model id (default `claude-opus-5-5`) |
| `SCHOLARLOOP_HANDOFF` | a research session's `handoff.json`; the campaign adopts its idea |
| `SCHOLARLOOP_STEPS`, `SCHOLARLOOP_POPULATION`, `SCHOLARLOOP_MAX_WORKERS` | rounds, ideas per round, parallel smoke runs |
| `SCHOLARLOOP_DRY_PATIENCE` | stop after this many rounds without a frontier improvement |
| `SCHOLARLOOP_MAX_COST` | estimated dollar threshold, checked between rounds and stages |
| `SCHOLARLOOP_WRITING` | `session` (interactive on a terminal), `auto`, or `oneshot` |
| `SCHOLARLOOP_PAPER_ONLY` | skip the campaign and write up the saved ledger |
| `SCHOLARLOOP_RESUME` | continue the saved campaign |
| `SCHOLARLOOP_PERSIST_SKILLS` | keep lessons in `~/.scholarloop/skills/<domain>` across campaigns |

</details>
