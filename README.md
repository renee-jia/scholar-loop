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

ScholarLoop runs the loop a PhD actually runs: it reads the literature, forms a grounded
hypothesis, **runs real ML experiments**, scores them against a frozen ground-truth metric, learns
from its failures, and drafts a peer-reviewed write-up — autonomously, with a deterministic harness
that keeps the agents honest and **impossible to reward-hack**.

| Stage | What it does |
|---|---|
| 🎯 **Director** | reads ledger + literature trends → sets the next direction, topic & budget |
| 🔭 **Lit Scout** | pulls real papers from arXiv + OpenAlex, citation-ranked → structured, *cited* findings |
| 💡 **Reasoner** | constraints + literature + past lessons → the next experiment |
| 🗳️ **Debate Panel** | three personas vote — *is this worth a GPU?* |
| 🪜 **Funnel** | smoke → verify → full · a cheap screen kills most ideas |
| ⚙️ **Runner** | runs a **real** torch experiment, scored by a **frozen** metric |
| 🪞 **Reflector** | turns the outcome into a lesson in a decaying skill library |
| 🚦 **Advisor** | **PROCEED · REFINE · PIVOT** — steers the loop |
| ✍️ **Writer + Reviewer** | confirmed findings → number-grounded draft → peer review |
| 🗄️ **Ledger + Skills** | the durable memory every step reads from |

## ✨ Highlights

| | |
|---|---|
| 🧪 **Real, pluggable experiments** | Drives real PyTorch runs (CPU-fast, no download). Two domains ship today — **digit classification** (error %) and **diabetes regression** (RMSE) — and a new one is just a YAML profile + an engine pair, **zero orchestrator changes**. |
| 🤖 **8 agents, one harness** | Director · Lit Scout · Reasoner · Debate · Reflector · Advisor · Writer · Reviewer — typed JSON-schema I/O, validate→retry, one shared audit trace. |
| 🔭 **Literature-grounded** | The Lit Scout pulls real papers from arXiv + OpenAlex, ranks them by citation impact, and distills *cited* techniques — so ideas aren't blind hill-climbing. |
| 💸 **Budget-aware funnel** | One idea climbs **smoke → verify → full**, each tier gated. Bad ideas die after one cheap run; marginal ones never burn a full run. |
| ⚙️ **Engineered loop** | A **parallel population funnel** — propose N ideas, smoke-screen them all at once, climb only the survivors — under a **self-stopping governor** that halts on budget, round cap, or convergence (*loop-until-dry*). |
| 🧠 **Self-improving** | Predicts each idea's effect, scores the prediction against reality, and distills failures into a **relevance-ranked, time-decaying skill library** re-injected next round. |
| 🎯 **Calibrated agents** | **Universal predict-then-verify** — every agent's checkable claims (Reasoner deltas, Debate go/no-go) are scored against ground truth, so the loop learns *which of its own agents to trust*. |
| 🛡️ **Can't be reward-hacked** | Two-phase **frozen scoring** (`train.py` can't fake the metric or see the val set) + edit **allowlist** + **VerifiedRegistry** number-grounding — proven by a bundled `cheater` engine. |
| ✅ **Honest & testable** | 108 tests, **no API key or GPU needed** — the whole loop runs against a deterministic `MockLLM`. |

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

## 🚀 Quickstart

```bash
pip install -e ".[dev]"             # pyyaml + jsonschema + pytest
python examples/quickstart.py       # the whole loop in <1s — no GPU, no API key

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
  full   3.7644%  [kept]    →  climbed 3 tiers, 3 kept
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

## 🧪 Two live runs on Claude Opus 4.8 — real torch, real API, end to end

| domain | metric | baseline | best (confirmed) | the climb | cost | self-review |
|---|---|---|---|---|---|---|
| [**digits-mlp**](examples/sample_run/) | val error | 5.0% | **3.82%** | population → verify → full, governed | ≈ $0.45 | reject 2/10 |
| [**diabetes-mlp**](examples/sample_run_diabetes/) | val RMSE | 56.5 *(linear model)* | **55.24** | population → verify → full, governed | ≈ $0.77 | reject 3/10 |

Both are **governed population funnels** — each round fans out several ideas, smoke-screens them in
parallel, and climbs only the survivors, while the loop halts itself (on convergence or a round cap)
and scores each agent's predictions against ground truth. An idea beats the baseline in each, every
number traces to a frozen-metric measurement, and the system's *own* reviewer still rejects the
papers as too marginal. (It's not wrong.) Each link opens the captured **paper**, **run log**, and
raw **ledger** — every number reproducible from the jsonl.

## 📊 Status

**Research preview** — the full PhD-workflow skeleton runs end-to-end on real experiments, with the
anti-reward-hacking guards in place and adversarially reviewed. It has been **run live against the
real Anthropic API** across **two domains** — captured verbatim for both
[classification](examples/sample_run/) and [regression](examples/sample_run_diabetes/), each a
real Opus campaign that beats its baseline and writes itself up. Next: container sandboxing for the
residual boundaries and scale.

License: [MIT](LICENSE).
