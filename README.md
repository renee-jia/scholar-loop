<div align="center">

# 🔬 Scholar Loop

### Autonomous, multi-agent AI research — a PhD's workflow on a single-GPU budget.

**read papers → find a gap → run real experiments → reflect → write & self-review**

[![tests](https://github.com/renee-jia/scholar-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/renee-jia/scholar-loop/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](#-quickstart)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![status](https://img.shields.io/badge/status-research%20preview-orange)](#-status)

</div>

---

ScholarLoop runs the loop a PhD actually runs: it reads the literature and **writes you a review**,
proposes **novelty-checked ideas** and lets **you pick**, **runs real ML experiments** on the chosen
one, scores them against a frozen ground-truth metric, learns from its failures, and then writes it
up **the slow way** — outline, key points, prose — with AI panels arguing each step and **you signing
off** before the next. A deterministic harness keeps the agents honest and **impossible to reward-hack**.

| Stage | What it does |
|---|---|
| 📚 **Lit Review** | plans queries → pulls real papers (arXiv + OpenAlex) → themes · per-paper notes · **open gaps** → `lit_review.md` |
| 💡 **Ideator** | gaps + your engine's setting + what the ledger tried → N distinct, cited ideas, **novelty-checked** → `ideas.md` |
| 🧑‍🔬 **You** | read the review, push back; read the ideas, pick one (or push back) → `handoff.json` |
| 🎯 **Director** | *(unattended runs)* reads ledger + literature trends → sets the next direction & topic |
| 🔭 **Lit Scout** | pulls real papers from arXiv + OpenAlex, citation-ranked → structured, *cited* findings |
| 💡 **Reasoner** | constraints + literature + past lessons → the next experiment |
| 🗳️ **Debate Panel** | three personas vote — *is this worth a GPU?* |
| 🪜 **Funnel** | smoke → verify → full · a cheap screen kills most ideas |
| ⚙️ **Runner** | runs a **real** torch experiment, scored by a **frozen** metric |
| 🪞 **Reflector** | turns the outcome into a lesson in a decaying skill library |
| 🚦 **Advisor** | **PROCEED · REFINE · PIVOT** — steers the loop |
| ✍️ **Writing session** | outline → key points → prose, each argued by a Skeptic · Editor · Methodologist panel and **approved by you**; every number, citation and sentence checked in code |
| 🧑‍⚖️ **Reviewer panel** | three persona reviewers score the draft against a NeurIPS-style checklist → `review.md` |
| 🗄️ **Ledger + Skills** | the durable memory every step reads from |

## 🧑‍🔬 Ideation first — a review you can read, ideas you can choose

The expensive part of research isn't running experiments, it's deciding **which** experiment
deserves them. So the loop has a front half that stops and asks:

```text
  topic ──▶ [1] literature review ──▶ lit_review.md   ◀── you: approve, or "also cover mixup"
                   │                                       (new queries · corpus grows · review rewritten)
                   ▼
            [2] candidate ideas ──▶ ideas.md          ◀── you: pick #2, or "be bolder"
                   │                                       (ranked by a pairwise Elo tournament ·
                   │                                        re-ideate on the same corpus)
                   ▼
            [3] handoff.json ──▶ Orchestrator.adopt() ──▶ smoke → verify → full …
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m scholarloop.ideation --profile profiles/digits-mlp.yaml \
    --topic "regularization recipes for small MLPs" --out runs/s1       # interactive
#   (a)pprove · (r)efine <feedback> · (q)uit  >  r also cover data-side regularization such as mixup
#   (s)elect <n> · (r)efine <feedback> · (q)uit >  s 2
SCHOLARLOOP_HANDOFF=runs/s1/handoff.json SCHOLARLOOP_OUT=runs/s1 \
    python examples/run_to_paper.py                                      # run the chosen idea, then write it up
```

Non-interactive too: `--stop-after review` writes the report and exits for a human; `--feedback "…"`
and `--select 2` resume a session from another process (all state lives in `session.json`);
`--auto` picks the tournament's top idea for unattended use (`--no-tournament` falls back to the
Ideator's self-scores). See it end to end, free and deterministic, in
[`examples/ideation_demo.py`](examples/ideation_demo.py).

What keeps it honest, in deterministic code rather than prompts:

| | |
|---|---|
| 📎 **Citation grounding** | every paper id the review or an idea cites must be in the retrieved corpus — a hallucinated reference is rejected and the agent retried (the literature analogue of number grounding) |
| 🏆 **Idea tournament** | ideas are ranked by pairwise matches, not by the Ideator grading its own work: a rotating persona panel compares every pair **twice, order swapped** (a win in only one order is a draw, which cancels position bias), Elo is computed in code, near-duplicates are grouped, and the match log lands in `tournament.json` |
| 🧾 **Meta-review** | a meta-reviewer turns the judges' *recurring* complaints into at most 5 recommendations, each citing the matches that show it (a one-off remark is rejected in code); the list is revised each round, not rewritten, and the next round's Ideator writes against it, alongside the lessons the experiment loop has learned (`--skills`) |
| 🔍 **Novelty check** | each idea is lexically compared to every reviewed paper and every hypothesis already in the ledger; the closest match is shown and flagged past a threshold — advisory, so *you* judge |
| ⚙️ **Testable by construction** | the Ideator sees the engine's metric, baseline, knobs and editable surfaces, so each idea comes with an experiment plan the loop can actually run |
| 🧭 **Recency × impact** | retrieval fills the corpus with the highest-cited papers but reserves slots for fresh, uncited preprints |

## ✍️ Writing, step by step — argue the outline before anyone writes a sentence

A one-shot "write the paper" call gives fluent prose over a weak argument. ScholarLoop builds the
argument first, lets different agents fight over it, and asks you before moving on:

```text
  evidence bank ──▶ [1] outline      Outliner proposes ─▶ Skeptic · Editor · Methodologist argue
  (runs · papers ·      │             ─▶ Outliner revises … until the panel accepts ──▶ outline.md
   baseline)            │                                                         ◀── you: approve / feedback
                        ▼
                   [2] key points   per section: claim + the evidence ids behind it, same debate
                        │                                                         ◀── you: approve / feedback
                        ▼
                   [3] draft        one call per section from the agreed points, under the bundled
                        │           style guide ──▶ paper.md
                        ▼
                   [4] review       three persona reviewers ──▶ review.md
```

```bash
# run_to_paper.py ends in this session (interactive on a terminal, unattended otherwise;
# SCHOLARLOOP_WRITING=oneshot keeps the single-call writer). The same directory resumes or revises later:
python -m scholarloop.writing --profile profiles/digits-mlp.yaml \
    --ledger runs/s1/experiments.jsonl --registry runs/s1/registry \
    --research runs/s1 --out runs/s1                 # interactive; --approve / --feedback "…" / --auto for scripts
python examples/writing_demo.py                     # the whole flow, free and deterministic
```

The writing discipline ships **in the repo**, not in a chat window: `scholarloop/writing/skills/`
holds the guidance the agents load as system prompts — the one-sentence question test and the
nugget, Farquhar's five-sentence abstract, Gopen & Swan sentence mechanics, Lipton's word choice,
claim-evidence calibration, the AI-tell catalogue, and a NeurIPS-style review checklist. What can be
checked is checked in code, and rejected + retried:

| | |
|---|---|
| 🔢 **Number grounding** | every number in a key point or a section traces to a run, a logged config value, the budget, or the baseline |
| 📎 **Evidence grounding** | every evidence id behind a claim is a real run id, a reviewed paper id, or `baseline`; result/method/comparison claims must cite one |
| 🧹 **Prose lint** | em-dashes, "not X, but Y", stacked *Moreover/Furthermore*, hype ("paves the way"), and the usual machine vocabulary are rejected before the text reaches you |
| 🗂️ **Coverage** | a drafted section that drops one of its agreed key points is flagged in the draft's notes |

## ✨ Highlights

| | |
|---|---|
| 🧪 **Real, pluggable experiments** | Drives real PyTorch runs (CPU-fast, no download). The shipped domain is **digit classification** (error %); a new one is just a YAML profile + an engine pair, **zero orchestrator changes**. |
| 🤖 **16 agents, one harness** | QueryPlanner · LitReviewer · Ideator · Director · Lit Scout · Reasoner · Debate · Reflector · Advisor · Outliner · Panelists · KeyPointer · SectionDrafter · Reviewer panel (+ the one-shot Writer/Reviewer) — typed JSON-schema I/O, validate→retry, one shared audit trace. |
| 🔭 **Literature-grounded** | The Lit Scout pulls real papers from arXiv + OpenAlex, ranks them by citation impact, and distills *cited* techniques — so ideas aren't blind hill-climbing. |
| 💸 **Budget-aware funnel** | One idea climbs **smoke → verify → full**, each tier gated. Bad ideas die after one cheap run; marginal ones never burn a full run. |
| ⚙️ **Engineered loop** | A **parallel population funnel** — propose N ideas, smoke-screen them all at once, climb only the survivors — under a **self-stopping governor** that halts on budget, round cap, or convergence (*loop-until-dry*). |
| 🧠 **Self-improving** | Predicts each idea's effect, scores the prediction against reality, and distills failures into a **relevance-ranked, time-decaying skill library** re-injected next round. |
| 🎯 **Calibrated agents** | **Universal predict-then-verify** — every agent's checkable claims (Reasoner deltas, Debate go/no-go) are scored against ground truth, so the loop learns *which of its own agents to trust*. |
| 🛡️ **Can't be reward-hacked** | Two-phase **frozen scoring** (`train.py` can't fake the metric or see the val set) + edit **allowlist** + **VerifiedRegistry** number-grounding — proven by a bundled `cheater` engine. A **held-out test split**, scored only on full-tier runs and never shown to an agent, gives the paper a number free of the bias that selecting on val every round adds; small datasets cross-validate (`cv_folds`), each seed on its own fold. |
| ✅ **Honest & testable** | 160 tests, **no API key or GPU needed** — the whole loop, the research and writing sessions included, runs against a deterministic `MockLLM`. |

> The LLM does only the open-ended reasoning. Everything checkable — search-space pruning, dedup,
> calibration, number-grounding, promotion gates — is deterministic, unit-tested code, and the
> metric is the **only** optimization target (no LLM-as-judge in the optimization loop). Multiple adversarial review passes
> found and fixed real bugs across the loop's correctness and reward-hacking boundaries.

## 🔁 Loop engineering — the part that isn't the prompt

The leverage isn't in any single agent call — it's in the **outer loop** around them: how it fans
out, what it spends compute on, what it remembers, and when it stops. ScholarLoop treats that loop
as the product. One **governed round** looks like this:

```text
  Director topic ─▶  Reasoner ✦ proposes N distinct ideas
                       │   fed: literature priors · relevance-ranked skills · each agent's track record
                       ▼
                  ╭──────────── parallel smoke screen · max_workers ────────────╮
                  │   idea₁ 4.2     idea₂ 7.3 ✗     idea₃ 4.4      …     ideaₙ   │   cheap · concurrent
                  ╰──────────────────────────────┬──────────────────────────────╯
                                 survivors only   │   (the clearly-worse die here)
                                                  ▼
                       verify · 3 seeds + significance  ─▶  full · 5 seeds        compute spent on the few
                                                  │
                                                  ▼
                       calibrate every agent's claim  ·  distill one lesson → skills
                                                  │
                                                  ▼
                       governor ▸  budget?   rounds?   converged?  ──▶  stop ◇ or next round ↺
```

Four pillars, each a few lines of deterministic code — and each pinned by tests so it can't silently rot:

| | pillar | what it buys you | in code |
|---|---|---|---|
| 🌐 | **Parallel population funnel** | explore wide, pay narrow — propose *N*, smoke-screen them **all at once**, climb only survivors | `Orchestrator.population_step(k, max_workers)` |
| 🛑 | **Self-stopping governor** | run unattended — halt on a **$ budget**, a round cap, or **loop-until-dry** convergence | `Governor(max_cost, max_rounds, dry_patience)` |
| 🎯 | **Universal predict-then-verify** | the loop learns *which of its own agents to trust* — Reasoner deltas & Debate go/no-go scored vs ground truth | `CalibrationLog` → next prompt |
| 🧭 | **Relevance-ranked context** | a growing skill library stays useful — surface the lessons that bear on *this* idea, not just the heaviest | `SkillLibrary.render(query=…)` |

> Net effect: a loop you can actually let run — it **fans out** to explore, **screens cheaply in
> parallel**, **concentrates compute** on what survives, **calibrates itself**, and **knows when to
> stop**. See it live in [`examples/governed_campaign.py`](examples/governed_campaign.py) (free,
> deterministic) or in the two captured Opus runs at the end.

## 🗂️ Repository layout

The package is split by stage; each subpackage's `__init__.py` lists what its modules do.

```
scholarloop/
├── core/          shared contracts: Agent harness, LLM clients, Profile (+ schema), Ledger, VerifiedRegistry
├── ideation/      front half, human-in-the-loop: litscout → litreview → ideator → tournament → session   (python -m scholarloop.ideation)
├── experiment/    autonomous loop: orchestrator, runner, reasoner, debate, reflector, advisor, director, governor, …
└── writing/       back half, author-approved: session, paper (one-shot fallback), skills/*.md   (python -m scholarloop.writing)
engines/           L1 experiment engines: a frozen prepare.py + an editable train.py per domain
profiles/          domain profiles (YAML) that bind an engine, a metric, budgets and baselines
examples/          runnable demos; sample_runs/digits is a verbatim live capture
tests/             mirrors the package: tests/{core,ideation,experiment,writing} + cross-cutting regressions
```

## 🚀 Quickstart

```bash
pip install -e ".[dev]"             # pyyaml + jsonschema + pytest
python examples/quickstart.py       # the whole loop in <1s — no GPU, no API key
python examples/ideation_demo.py    # review → feedback → ideas → you pick → the loop runs it (<1s, free)
python examples/writing_demo.py     # outline → panel debate → you approve → key points → prose → review (<1s, free)

pip install -e ".[dev,engines]"     # adds torch + scikit-learn for the real experiment engines
python examples/campaign_demo.py    # a full campaign on real torch, MockLLM-scripted
pytest -q
# ".[llm]" adds the Anthropic client, only needed for live API runs (e.g. examples/run_to_paper.py)
```

`quickstart.py` runs one idea through the funnel:

```
baseline to beat: 4.9% val_top1_err
  smoke  3.7644%  [kept]
  verify 3.8004%  [kept]
  full   3.8208%  [kept]    →  climbed 3 tiers, 3 kept
```

## 🎬 See it run — a real campaign

`campaign_demo.py` drives the **whole agent chain on real torch**, scripted by `MockLLM` so it's
deterministic and free (abridged):

```text
=== CAMPAIGN · digits-mlp (real torch) · baseline 5.0% ===

🎯 Director    scale width/depth and tune the optimizer
🔭 Lit Scout   wider/deeper layers (arXiv:1512.03385) · SGD momentum + cosine (arXiv:1608.03983)

idea 1   🗳️ run      🪜 smoke 4.67% → verify 4.96%      🚦 proceed
idea 2   🗳️ REJECT   → skipped, no GPU spent
idea 3   🪜 smoke 52.0% discarded     🚦 pivot     predicted −1.0, measured +47 → calib_err 48.3
```

A single run shows the system

- **ground its idea in literature** before proposing it,
- **save a GPU run** when the debate panel vetoes a redundant idea,
- **kill a bad idea cheaply** at the smoke tier — no full run,
- **catch its own wrong prediction** via predict-then-verify, then **pivot**.

## 🧪 Live, on claude-opus-5-5: the whole flow, and a fair test against random search

| run | what | result | cost |
|---|---|---|---|
| [**full flow**](examples/sample_runs/digits/) | lit review → 5 ideas → tournament pick → 18 torch runs → staged write-up → 3-reviewer panel | **3.27% val / 3.33% held-out test**, beats logreg@3.8; panel: reject 2.2/10 | ≈ $0.94 campaign + $5.53 write-up |
| agent vs random | `examples/compare_search.py --arms agent,random --repeats 3 --rounds 6 --match runs` | random search wins: held-out **3.13 ± 0.08** (3/3 reached full) vs agent 4.22 (1/3); after giving the agent the knob ranges and an explore phase (e32f237), agent 3.72 ± 0.24 (3/3), still 0 of 9 pairs; warm-started on the random arm's first draws (723516c), a tie (3.14 vs 3.13, P = 0.5), but every agent best is one of those random draws: its own ideas never won | ≈ $1.94 + $1.66 + $1.13 |

Both are reported as they came out. The full flow beats its baseline, and every number traces to
a frozen-scorer measurement. The panel still rejects the paper, for the right reason: the tournament
picked an augmentation idea, and the campaign only tuned hyperparameters. The comparison is the
uncomfortable one. Given the same number of training runs, plain random search over the profile's
knobs beats the agent on digits. The first agent stayed close to the defaults and moved one knob a little
at a time. The second explores, but it lowers lr for wider and deeper nets, a stability prior
that is wrong here: every random winner sits at lr 0.15-0.2 with one hidden layer. On a five-knob search, an LLM proposing configs does not beat random search. The
next step is the agent's real advantage: structural edits to the train script, with knob tuning left
to the search. Until it does, the claim on digits is *a faithful, auditable
research loop*, not *a better optimizer than random search*. The capture's
[README](examples/sample_runs/digits/README.md) lists what the run exposed.

## 📊 Status

**Research preview** — the full PhD-workflow skeleton runs end-to-end on real experiments, with the
anti-reward-hacking guards in place and adversarially reviewed, an ideation-first research
session (literature review → novelty-checked ideas → your choice) in front of the loop, and a
step-by-step writing session (outline → key points → prose, panel-argued and author-approved) after it. It has been **run live against the
real Anthropic API**, captured verbatim in [examples/sample_runs/digits](examples/sample_runs/digits/): a
real Opus campaign that beats its baseline and writes itself up. It has also been measured against
random search at a matched budget, and on digits it loses (above). Next: make the agent win that
comparison (bolder, range-aware proposals; fold-matched tier comparisons), then a per-idea second
retrieval pass, container sandboxing for the residual boundaries, and scale.

License: [MIT](LICENSE).
