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
| 🧠 **Self-improving** | Predicts each idea's effect, scores the prediction against reality, and distills failures into a **time-decaying skill library** re-injected next round. |
| 🛡️ **Can't be reward-hacked** | Two-phase **frozen scoring** (`train.py` can't fake the metric or see the val set) + edit **allowlist** + **VerifiedRegistry** number-grounding — proven by a bundled `cheater` engine. |
| ✅ **Honest & testable** | 76 tests, **no API key or GPU needed** — the whole loop runs against a deterministic `MockLLM`. |

> The LLM does only the open-ended reasoning. Everything checkable — search-space pruning, dedup,
> calibration, number-grounding, promotion gates — is deterministic, unit-tested code, and the
> metric is the **only** optimization target (no LLM-as-judge). Three adversarial review passes
> found and fixed real bugs; see [`DESIGN.md`](DESIGN.md) for the architecture and residual boundaries.

## 🚀 Quickstart

```bash
pip install -e ".[dev]"          # pyyaml + jsonschema + pytest   (".[llm]" adds the Anthropic client)

python examples/quickstart.py    # the whole loop in <1s — no GPU, no API key
python examples/campaign_demo.py # a full campaign on real torch, MockLLM-scripted
pytest -q
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

## 📊 Status

**Research preview** — the full PhD-workflow skeleton runs end-to-end on real experiments, with the
anti-reward-hacking guards in place and adversarially reviewed. It has been **run live against the
real Anthropic API** (see [`examples/sample_run/`](examples/sample_run/) for a captured Opus run
that beats a baseline and writes itself up) across **two domains** (classification + regression).
Next: container sandboxing for the residual boundaries, statistical-significance gating, and scale.

License: [MIT](LICENSE).
