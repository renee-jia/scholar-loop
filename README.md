<div align="center">

# 🔬 Scholar Loop

### An auditable research loop: literature review, ideas, real experiments, scientific review, and a paper whose every number traces to a measurement.

[![tests](https://github.com/renee-jia/scholar-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/renee-jia/scholar-loop/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](#-quickstart)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![status](https://img.shields.io/badge/status-research%20preview-orange)](#-status-and-limits)

</div>

---

ScholarLoop runs the loop a PhD student runs. It reads the literature and writes a review,
proposes ideas and ranks them in a tournament, lets **you pick one**, runs real ML experiments
that must test that idea against a matched control, has critics review the evidence before a
word of the paper is written, and then writes the paper in stages that **you approve**.

The language model does the open-ended reasoning. Everything that can be checked is checked in
code: which papers may be cited, which numbers may be reported, which files an experiment may
edit, when the held-out test set may be opened, and whether your approval still matches the
evidence. A model output that fails a check is rejected and retried with the errors as feedback.

What this project claims, and what it does not:

- **Claimed:** a faithful, inspectable research workflow. Each claim in the paper is bound to
  runs in an append-only ledger, each number to a frozen-scorer measurement, each citation to a
  retrieved paper.
- **Not claimed:** that the agent is a better optimizer than simple baselines. On a five-knob
  search it does not beat random search ([measured below](#-evidence-so-far)). Passing the
  checks means an output is traceable, not that it is scientifically correct.

```text
 IDEATION (you choose)              EXPERIMENTS (autonomous)            REVIEW + WRITING (you approve)

 topic ─▶ literature review ─▶ ideas ─▶ pick ─▶ control vs treatment ─▶ scientific review ─▶ release ─▶ outline ─▶ key points ─▶ draft ─▶ reviewers
            ▲ your feedback      ▲ tournament     smoke → verify → full      │        ▲          ▲ you       ▲ you        ▲ you
                                                   ledger + registry          └ experiments or narrower claims
```

## 🚀 Quickstart

No GPU and no API key are needed for the demos. They use a scripted model and local engines.

```bash
pip install -e ".[dev]"             # pyyaml + jsonschema + pytest
python examples/quickstart.py       # one idea through the smoke → verify → full funnel
python examples/ideation_demo.py    # review → feedback → ideas → pick → the loop runs it
python examples/writing_demo.py     # outline → panel debate → approve → key points → prose → review

pip install -e ".[dev,engines]"     # adds torch + scikit-learn for the real engines
python examples/campaign_demo.py    # the whole agent chain on real torch training, scripted model
python examples/governed_campaign.py   # parallel population funnel under a self-stopping governor
pytest -q
```

`quickstart.py` prints:

```text
baseline to beat: 4.9% val_top1_err

  smoke  val_top1_err=3.7644%  [kept]
  verify val_top1_err=3.8004%  [kept]
  full   val_top1_err=3.8208%  [kept]
```

Live runs need `pip install -e ".[llm]"` and `ANTHROPIC_API_KEY`. The AG News engine also needs
`".[nlp]"` for a one-time dataset download.

## 🧭 The workflow, stage by stage

| Stage | What happens | You | Artifacts |
|---|---|---|---|
| **Literature review** | plans queries, retrieves papers from arXiv and OpenAlex (citation-ranked, with slots reserved for recent preprints), writes themes, per-paper notes and open gaps | approve, or ask for more coverage | `lit_review.md` |
| **Ideas** | proposes ideas tied to the gaps, your engine's metric and knobs, and what the ledger already tried; ranks them by a pairwise tournament | pick one, push back, inject your own | `ideas.md`, `tournament.json`, `handoff.json` |
| **Experiments** | proposes runs, a debate panel votes on whether each deserves compute, a three-tier funnel runs them, a frozen scorer measures them | none (unattended) | `experiments.jsonl`, `registry/`, `run.md` |
| **Scientific review** | four role-specific critics answer evidence questions for every claim; open concerns require experiments or narrower claims | approve the exact review hash | `scientific_review.md` |
| **Writing** | outline, then key points, then prose; each stage argued by a Skeptic, an Editor and a Methodologist | approve the outline and the key points | `outline.md`, `key_points.md`, `paper.md` |
| **Reviewer panel** | three persona reviewers score the draft against a checklist | read it | `review.md` |

### 1. Ideation: a review you can read, ideas you can choose

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m scholarloop.ideation --profile profiles/digits-mlp.yaml \
    --topic "regularization recipes for small MLPs" --out runs/s1
#   (a)pprove · (r)efine <feedback> · (q)uit  >  r also cover data-side regularization such as mixup
#   (s)elect <n> · (r)efine <feedback> · (q)uit >  s 2
```

Ideas are not ranked by the model grading its own work. A rotating panel compares every pair
twice with the order swapped; a win in only one order counts as a draw, which cancels position
bias. Elo is computed in code and the match log is saved. A meta-reviewer turns the judges'
recurring complaints into at most five recommendations, each citing the matches behind it, and
the next round of ideas is written against them.

Scripted use: `--stop-after review` writes the report and exits for a human; `--feedback "…"`
and `--select 2` resume from another process; `--auto` picks the tournament's top idea. Add
`--deep --brief examples/research_brief.json` for independent mechanism, transfer and
contradiction strategies with per-idea retrieval. `--inject-idea` accepts a human-authored
candidate, and `--decision reject|refine|merge|revive` records your decisions on stable idea IDs.

### 2. Experiments: the chosen idea must actually be tested

```bash
SCHOLARLOOP_HANDOFF=runs/s1/handoff.json SCHOLARLOOP_OUT=runs/s1 \
    python examples/run_to_paper.py       # campaign, scientific review, write-up
```

One round of the loop:

```text
  Reasoner proposes N ideas ─▶ debate panel votes ─▶ parallel smoke screen (1 seed, cheap)
                                                              │ survivors only
                                                              ▼
                            verify (3 seeds + noise margin) ─▶ full (5 seeds)
                                                              │
       calibrate each agent's prediction against the result · distill one lesson into the skill library
                                                              │
                              governor: budget? round cap? converged? ─▶ stop, or next round
```

- **Treatment versus control.** When a campaign adopts your idea, every proposal declares
  whether it applies the idea (`treatment`) or is the matched run without it (`control`). The
  harness allows one control at most before the first treatment, and the loop cannot pivot away
  before a treatment has run. An idea that no hyperparameter expresses is implemented by
  replacing the training script through the edit channel, which runs in an isolated copy.
- **Frozen scoring.** The training script only trains and saves weights. A separate frozen
  scorer, run from the pristine repository, produces the number. The agent cannot print a metric.
- **Budget-aware funnel.** Most ideas die after one cheap run. Only survivors get more seeds.
- **Predict, then verify.** Each proposal predicts its effect before it runs. Predictions are
  scored against the result, and each agent's track record is shown to it in later rounds.
- **Held-out test.** Exploration never scores the test split. Opening it needs an explicit
  confirmation request and closes the campaign to further exploration.

### 3. Scientific review, then writing

```bash
python -m scholarloop.review review  --profile profiles/digits-mlp.yaml \
    --ledger runs/s1/experiments.jsonl --registry runs/s1/registry --research runs/s1 --out runs/s1
# Read scientific_review.md. Approve only a READY_FOR_AUTHOR review, by its exact hash:
python -m scholarloop.review approve --profile profiles/digits-mlp.yaml \
    --ledger runs/s1/experiments.jsonl --registry runs/s1/registry --research runs/s1 --out runs/s1 \
    --author Researcher --review-hash EXACT_REVIEW_HASH
python -m scholarloop.writing --profile profiles/digits-mlp.yaml \
    --ledger runs/s1/experiments.jsonl --registry runs/s1/registry --research runs/s1 --out runs/s1
```

Writing does not start until the review is approved. Your approval is bound to a hash of the
evidence, the argument and the tracked execution inputs; if any of them changes, the release is
void and writing stops. When the critics ask for controls or ablations
(`NEEDS_EXPERIMENTS`), `python -m scholarloop.experiment.agenda` freezes them as a protocol,
runs every condition, keeps negative results and failed seeds, and returns the evidence for
re-review. Completing experiments never approves writing.

A one-shot "write the paper" call gives fluent prose over a weak argument. The writing session
builds the argument first: the outline is debated and approved, then the key points, and only
then is each section drafted from the agreed points. The writing guidance ships in the repo
(`scholarloop/writing/skills/`), not in a chat window.

## 🛡️ What is enforced in code

| Check | What it rejects |
|---|---|
| **Citation grounding** | a paper id that is not in the retrieved corpus, in the review, an idea, the outline, a key point or the prose |
| **Number grounding** | a number that no measurement supports. In reviewed writing each sentence must cite an approved statement, and its numbers are checked against the statements cited in that sentence |
| **Evidence ids** | a run id or statement id that does not exist; a result claim with no evidence |
| **Frozen scoring** | a metric produced by anything other than the frozen scorer; edits outside the training entrypoint; a scorer file that changed during the run |
| **Idea testing** | a second control before the first treatment of an adopted idea |
| **Test-set boundary** | scoring the held-out split during exploration; continuing to explore after it was opened |
| **Release binding** | writing against a review whose evidence, argument or code has since changed |
| **Tournament and meta-review** | a win that holds in only one presentation order; a recommendation that no recorded match supports |
| **Prose lint** | em-dashes, "not X, but Y", stacked connectives, hype phrases and common machine vocabulary |
| **Budgets** | requests past a persistent call cap; training attempts past a reservation; spend past a shared dollar cap |

Every agent attempt is logged to `*.agents.jsonl` with its inputs' fingerprints, the structured
output, the validation errors and the outcome, so a rejected attempt stays visible.

These are deterministic checks, not OS isolation. They defeat the realistic threat, a language
model proposing edits, and not a hostile process with filesystem access.

## 📊 Evidence so far

### The full flow, live, on digit classification

One complete run with `claude-opus-5-5` as every agent is captured verbatim in
[`examples/sample_runs/digits`](examples/sample_runs/digits/): literature review, five ideas, a
tournament pick, 18 real training runs, a staged write-up and a three-reviewer panel.

| | |
|---|---|
| Result | 3.27% validation error, 3.33% on the held-out test, against a logistic-regression baseline of 3.8% |
| Cost | about $0.94 for the campaign and $5.53 for the write-up |
| Reviewer panel | **reject, 2.2/10** |

The panel rejected the paper for the right reason. The tournament picked an augmentation idea,
and the campaign only tuned hyperparameters, so the hypothesis was never tested. The same run
exposed that the fidelity tiers were compared on different validation folds. Both are fixed:
adopted ideas are now tested as treatment versus control, and tiers are compared on shared
folds. The capture's [README](examples/sample_runs/digits/README.md) lists everything the run
exposed.

### Agent versus random search, at a matched budget

`examples/compare_search.py --arms agent,random --repeats 3 --rounds 6 --match runs`, on digits:

| Agent version | Held-out error | Random search | Outcome |
|---|---|---|---|
| first version | 4.22 (1 of 3 campaigns reached the full tier) | 3.13 ± 0.08 (3 of 3) | random wins |
| given knob ranges and an explore phase | 3.72 ± 0.24 (3 of 3) | 3.13 | random wins all 9 pairs |
| warm-started on random draws | 3.14 | 3.13 | tie, and every agent best was one of the random draws |

On a five-knob search, a language model proposing configurations does not beat random search.
Its advantage, if it has one, is in structural changes to the training script, which is what the
treatment-versus-control path now exercises. These results are reported as they came out.

### How reliable are the guards?

`scholarloop.reliability` measures the checks themselves, without a model as judge. It changes
one site at a time in artifacts the guards had accepted and passes each variant through the
production check. On the run behind the digits capture (617 injected faults, 14 unmodified
controls, none falsely rejected):

| Fault tier | Example | Detected |
|---|---|---|
| Fabrication | a citation, run id or number that does not exist; a required section removed | 276 / 284 (97%) |
| Misattribution | a real citation, run id or measurement used in the wrong place; a flipped direction word | 0 / 333 (0%) |

Three things follow. The study found a real gap: identifiers written inline in outline and
key-point text were not checked. Closing it moved fabrication detection from 88% to 97%. The
remaining misses are numbers in the outline, which may state derived quantities such as
differences and cannot be looked up. And a lexical check cannot see misattribution at all. That
capture predates reviewed writing, where numbers are bound to the statements a sentence cites;
the same study runs in reviewed mode on newer runs.

```bash
python -m scholarloop.reliability inject   --profile profiles/digits-mlp.yaml --run runs/s1 --out runs/s1/reliability
python -m scholarloop.reliability traces   --traces runs/s1 --out runs/s1/reliability      # natural fault rate
python -m scholarloop.reliability baseline --profile profiles/digits-mlp.yaml --run runs/s1 \
    --out runs/s1/reliability --repeats 3                                                  # USES THE API
```

`traces` reads the attempt logs: a rejected first attempt is the output a harness without checks
would have used. `baseline` asks one model call to write the whole paper with the grounding
rules stated only in the prompt, then audits it with the same checkers.

## 🧪 Engines and domains

A domain is a YAML profile plus an engine: a frozen `prepare.py` (data, model construction,
scoring) and an editable `train.py`. Adding one needs no change to the orchestrator.

| Profile | Task | Notes |
|---|---|---|
| `digits-mlp` | 8x8 digit classification with an MLP | no download; about 1-2 s per run on CPU; 5-fold validation and a held-out test |
| `agnews-bow` | AG News topic classification, bag-of-embeddings | fixed 10,000-article subset; about 1 s per run on CPU; one-time download |
| `image-classification` | deterministic stub | used by the free demos and tests |
| `cheater` | adversarial fixture | a training script that tries to fabricate its metric, for the scoring tests |

### NLP case study under one spend cap

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install -e ".[dev,engines,llm,nlp]"
bash examples/case_study_agnews.sh        # USES THE API; CAP=10 (USD) by default
```

The script runs ideation, the campaign, scientific review, the staged write-up and the
reliability evaluation in one directory, and resumes each stage from saved state. Run it in a
terminal: scientific review and writing wait for your decision. On this task a method change
has a measurable effect (word dropout lowers error by about a point at the default training
length) and a competing explanation (stopping earlier does about as well), which is what the
control, the treatment and the critics are for. A captured run is not yet published.

## 💸 Cost and budget controls

| Control | Scope |
|---|---|
| `SCHOLARLOOP_SPEND_CAP` + `SCHOLARLOOP_SPEND_FILE` | a hard dollar cap shared by every command pointed at the same file; checked before each request, so it can be overshot by at most the request in flight |
| `--max-llm-calls`, `--max-calls` | a persistent cap on model requests, retries included |
| `SCHOLARLOOP_MAX_COST`, `Governor(max_cost=…)` | an estimated threshold checked between rounds and stages |
| training reservations | a cap on training attempts; failed attempts stay charged |

Spend is estimated from token counts and a price table. A model without a price is refused while
a dollar cap is set.

## 🗂️ Repository layout

```text
scholarloop/
├── core/            agent harness, LLM clients, profile (+ schema), ledger, registry, budgets
├── ideation/        literature retrieval and review, ideator, tournament, meta-review, session
├── experiment/      orchestrator, runner, reasoner, debate, advisor, director, reflector,
│                    skills, calibration, governor, experiment agenda, research cycle, comparisons
├── review/          evidence dossier, critics, issue history, author release
├── writing/         staged writing session, one-shot fallback, skills/*.md
├── reliability.py   fault injection, attempt-log analysis, single-call baseline
└── evaluation.py    read-only workflow diagnostics
engines/             a frozen prepare.py and an editable train.py per domain
profiles/            domain profiles (YAML)
examples/            runnable demos, the live flow, the case-study script, sample_runs/digits
tests/               mirrors the package; runs offline with a scripted model
```

Command-line entry points: `python -m scholarloop.ideation`, `.review`, `.writing`,
`.reliability`, `.evaluation`, `.experiment.agenda`, `.experiment.research_cycle`,
`.experiment.suite`, `.experiment.decision_bench`, `.experiment.decision_policy`,
`.ideation.pilot`. Each takes `--help`.

## 📌 Status and limits

Research preview. The whole workflow runs end to end on real experiments and has been run live
against the Anthropic API. What is not there yet, or not shown:

- **No evidence of better research.** The mechanisms are tested as software. Their effect on
  research quality needs matched-budget evaluation with expert judgment, which has not been done.
- **No OS isolation.** The frozen-scorer hash detects tampering but does not prevent filesystem
  writes, and deterministic splits over public data can be reconstructed. Container isolation and
  a runner-provisioned private test set are not implemented.
- **Semantic errors pass lexical checks.** A real citation attached to the wrong claim is caught,
  if at all, by the critics and the semantic reviewer, which are model judgments.
- **Roles are declared, order is enforced.** Whether a run labelled `treatment` really implements
  the idea is the agent's claim. The role, the edit and the candidate's source are recorded for
  review.
- **Novelty is provisional.** Novelty checks compare against retrieved abstracts and do not
  certify originality.
- **One model provider.** Live runs use the Anthropic SDK through an `LLMClient` interface; other
  providers need a client implementation.
- **Small tasks.** The shipped domains train in seconds on CPU. Nothing here has been run at scale.

## 📖 Reference

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

## License

[MIT](LICENSE).
