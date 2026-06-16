# ScholarLoop — Design

> **ScholarLoop** is an auto-research framework that replicates a PhD's research workflow end-to-end. From "read literature / track new architectures" → "find a gap, generate an idea" → "design experiments" → "auto-edit code & run experiments" → "analyze" → "accumulate intuition across experiments" → "write paper / self-review."
>
> Scope: **broad ML research** (CV / NLP / RL, multiple datasets and metrics), on a **single-GPU to small-cluster (1–8 GPU)** budget.
>
> Kernel philosophy: every layer stays "single-file readable, diff-auditable, metric-comparable." Get the minimal loop working first, then add layers.

---

## 0. What we distilled from prior work

| | Minimal experiment kernel | AI-Researcher (HKUDS) | AutoResearchClaw |
|---|---|---|---|
| Strengths | Minimal experiment engine; fair 5-min comparison; single ground-truth metric | Paper+code dual grounding; smoke-then-scale; debiased dual review | Pluggable YAML domains; VerifiedRegistry anti-hallucination; failure→skill-library self-evolution |
| Weaknesses | No memory; no literature; local hill-climbing; single domain | No external memory; regresses to conservative baselines; LLM-judge as optimization target is dangerous | 23 stages too heavy; brittle AST dedup; over-HITL hurts |

**The 5 core patterns we take:**
1. **A minimal experiment kernel** as the L1 engine (fixed `prepare`, single editable `train`, fixed budget, single ground-truth metric).
2. **Pluggable domains = declarative YAML profiles** (AutoResearchClaw) → solves "broad ML." The orchestrator never understands CV/NLP/RL; all knowledge lives in the profile.
3. **VerifiedRegistry + post-hoc number grounding** (AutoResearchClaw) → the #1 lever against hallucination / reward-hacking.
4. **Smoke-gate + Advisor↔Coder loop + "kill after N tries"** (AI-Researcher) → saves GPU under a small budget.
5. **Failure → time-decaying prompt skill library** (AutoResearchClaw MetaClaw) → this is the PhD's "accumulated intuition," with no training, backbone-agnostic.

**What we explicitly avoid:** 23-stage pipeline (collapse to ~6); LLM-judge as an optimization target; HITL at every step; AST shape-based dedup (use result-level hash dedup instead).

---

## 1. Layered architecture

```
L5  Writer + Reviewer       write paper → simulated review → revise   (reuse ARS academic-paper / reviewer)
L4  Research Director        read ledger trends + literature hotspots → set direction, allocate budget, PROCEED/REFINE/PIVOT
L3  Idea Engine              Lit Scout → Novelty Check → Hypothesis Gen (every hypothesis must cite a source)
L2  Experiment Orchestrator  experiment matrix / multi-fidelity funnel / smoke-gate / VerifiedRegistry
L1  Experiment Engine        = the minimal kernel, but the train script & metric are injected by the domain profile
L0  Memory & Ledger          per-experiment: diff + hypothesis + citation + metric + verdict;  Skill Library: failure → decaying skill
```

Minimal runnable loop = **L0 + L1 + L3**: read paper → produce a cited idea → edit code → smoke-validate → record to ledger. Build this first; L2/L4/L5 are incremental.

---

## 2. Key data contracts (define schemas before writing code)

### 2.1 Domain Profile (`profiles/<domain>.yaml`) — solves "broad ML"
The orchestrator stays domain-agnostic; all domain knowledge is declared here:

```yaml
name: image-classification
paradigm: supervised
train_entrypoint: engines/vision/train.py    # the single file L1 may edit
fixed_module:      engines/vision/prepare.py  # data / evaluation, NOT editable
metric:
  name: val_top1_err      # lower is better; pick one ground-truth primary metric
  direction: minimize
  vocab_independent: false
budget:
  smoke_sec: 120          # coarse screen
  verify_sec: 1800        # recheck
  full_sec: 14400         # confirm (use sparingly on a small cluster)
datasets:   [cifar10, cifar100]
baselines:  [resnet18@5.5, vit-s@4.9]      # "must beat X" in the metric's units (here error %, lower=better)
allowed_edits: [architecture, optimizer, schedule, augmentation]
forbidden_edits: [prepare.py, eval path, dataset split]    # guards against reward hacking
debate_roles: [Innovator, Pragmatist, Contrarian]
```

Adding a domain = one profile + one `train.py`/`prepare.py` pair. **No orchestrator changes.**

### 2.2 Experiment Ledger (`ledger.jsonl`, one record per line) — solves "no memory"
One record per **run** = one (idea, fidelity) pair. The same idea climbing the funnel
produces several linked records (smoke → verify → full), chained via `parent`. Append-only;
no in-place updates. An idea-level view is reconstructed by the orchestrator from the chain.
```json
{
  "id": "exp_0427",
  "parent": "exp_0391",
  "domain": "image-classification",
  "hypothesis": {"claim": "...", "source": "arXiv:2401.xxxxx", "predicted_effect": "-0.4 top1_err"},
  "diff": "engines/vision/train.py @@ ...",
  "fidelity": ["verify"],
  "metric": {"name": "val_top1_err", "verify": 5.8, "seeds": [5.8, 5.9, 5.7]},
  "verdict": "kept",
  "registry_id": "exp_0427"
}
```
All meta-analysis, Director decisions, and novelty dedup rely on this. **Must exist on Day 1.**

### 2.3 VerifiedRegistry (`registry/<exp_id>.json`) — the #1 anti-hallucination lever
At runtime, **automatically** capture every condition's mean/std/seed and freeze it here. Any number in a report (draft / verdict) that is not in the registry → **reject that conclusion**. Prose can never override the measured direction.

### 2.4 Skill Library (`skills/arc-<id>.json`) — the PhD's "intuition"
Each entry = `{category, severity s∈(0,1], mitigation, source, ts}`. On the next run, the top
entries are injected into the Reasoner prompt with a time-decay weight `w = s·2^(-Δdays/T½)`
(== `s·exp(-ln2·Δt/T½)`, T½ ≈ 30 days). The Reflector mines these from each run's outcome
(a large `calibration_error` is a strong signal); dedup is by content hash so a rejected
lesson can't reappear as fresh. No training, backbone-agnostic. `SkillLibrary.for_domain(name)`
persists per-domain under `~/.scholarloop/skills/<domain>/`, so intuition **compounds across
campaigns** (not just within one run), kept isolated per domain. (Implemented in `skills.py`.)

---

## 3. Core loop (L2 orchestration, optimized for a small budget)

```
for each idea from L3 (already carries a literature citation + must-beat baseline):
  1. REASONER (§4.5): reason over ledger+lit+skills → diff + prediction + search-space constraints
  2. Guard check: did the diff touch forbidden_edits, or fall outside the constraints? → discard
  3. SMOKE  (smoke_sec, single seed)        → coarse screen: kill only ideas clearly worse than the gate (slack band)
  4. VERIFY (verify_sec, 3 seeds)           → smoke survivors enter; promote only if the pessimistic bound beats the gate
  5. FULL   (full_sec, 5 seeds, sparingly)  → only "stable winners" from verify; widest seed set = tightest final number
  6. Write every number into the VerifiedRegistry + ledger at each step
  7. Advisor analyzes → PROCEED / REFINE (back to step 1) / PIVOT (back to L3); kill after N REFINEs
  8. REFLECTOR (§4.5): score prediction vs measurement → distill one lesson into the Skill Library
```

The **multi-fidelity funnel** is the survival strategy on a small cluster: a 120s coarse screen kills 90%, and only winners burn a full run. On a single card, set `full_sec` very large and gate it behind a human go-ahead. **Implemented** in `Orchestrator.funnel_step` (`run(funnel=True)`): one idea climbs smoke→verify→full, each tier cleared by a deterministic `_promote` gate. The two stages are deliberately asymmetric — the standard high-recall-then-high-precision funnel shape:
- **Smoke** is a single noisy seed, so it screens with **slack** (`smoke_slack`, default 0.25): it drops only ideas *clearly* worse than the gate (more than 25% of the gate's magnitude past it), keeping borderline candidates that a multi-seed verify might confirm. This is what stops a good idea from dying to one unlucky smoke seed.
- **Verify** runs 3 seeds and enforces the strict bar: the **pessimistic confidence bound** (`mean ± promote_z·SEM`, not just the mean) must beat the gate, so a win has to be robust to seed noise.

The gate itself is the more stringent of the frontier and the per-domain `baseline`, which is calibrated to what the engine can actually reach (see the profile comments: digits ~3.1% floor → baseline 5.0; diabetes ties the linear model ~56.5). Tiers chain via `parent`; the Reasoner/debate/reflect/advisor wrap the idea once, not per tier.

### 3.1 Loop engineering — population fan-out + a governed loop

The funnel above runs one idea at a time. **`population_step(k, max_workers=N)`** turns it into a **parallel population funnel**: propose `k` mutually-distinct ideas, **smoke-screen them all at once** (concurrently — each run is a subprocess, so threads give real parallelism; the ledger append takes a process lock so concurrent writers can't interleave), then climb only the smoke survivors up verify→full, best-first. Expensive tiers are spent on a pre-screened few, not on every idea — the high-recall/high-precision shape applied to a *batch*. Experiment ids are assigned up front so parallel writers never collide.

The campaign loop itself is **governed** rather than fixed-length. `run(..., governor=Governor(...))` consults a small pure state machine at the top of each round that ends the campaign on any of three independent conditions:
- **budget** — a USD ceiling on cumulative LLM spend (`cost_of` prices the shared client's `usage`), with 50/80/100% alerts on the way up;
- **rounds** — a hard iteration cap;
- **convergence** — *loop-until-dry*: stop after `dry_patience` rounds with no frontier improvement (keep going while ideas still help, stop once the search has clearly plateaued).

`Governor` calls no LLM, so the stop logic is fully unit-tested on its own. Together these make an autonomous loop you can actually let run unattended: it fans out to explore, screens cheaply in parallel, concentrates compute on survivors, and halts itself when the money, the patience, or the headroom runs out.

### 3.2 Loop engineering — context assembly + universal predict-then-verify

Two more pieces govern *what the loop knows about itself* each round:

- **Context assembly (bounded, relevant).** The skill library grows every round, but only a few lessons can enter the prompt. Selection is now **relevance-aware**: `SkillLibrary.render(query=...)` ranks by decayed weight **boosted by lexical overlap** with the current direction (topic + Advisor/Director guidance + literature priors), so the lessons shown bear on *this* idea, not just the globally heaviest. With no query it reproduces pure-weight order (backward-compatible). The ledger is already compressed into search-space constraints, so the agent's window stays bounded and on-topic as the campaign scales.
- **Universal predict-then-verify (`calibration.py`).** Self-calibration is generalized from the Reasoner to *any* agent that commits to a checkable claim. A `CalibrationLog` scores two kinds of claim once ground truth lands — **delta** (the Reasoner's `predicted_delta` vs the measured change; a "hit" is getting the direction right) and **binary** (the Debate panel voting "run" = "this beats the gate", vs whether it actually did) — and renders a per-agent track record ("debate panel: 40% of approved ideas beat the gate") back into the next round's Reasoner prompt. The verifier loop, closed: the system learns not just *what* worked but *which of its own agents to trust*.

---

## 4. L3 Idea Engine (the key to escaping "local hill-climbing")

- **Lit Scout**: given a direction, search several sources and extract "what architecture change / optimizer / trick was proposed." **Implemented** as injectable per-source clients (`ArxivClient`, `OpenAlexClient`, `SemanticScholarClient`) that the scout queries in parallel, de-duplicates, and **ranks by citation count** so the highest-impact techniques surface first (impact grounding). Default sources are arXiv + OpenAlex (no key, carries citations); Semantic Scholar is opt-in (rate-limited without a key). Each source degrades independently — one being down doesn't kill the scout. Code-link grounding (paperswithcode-style paper↔repo pairs) is surfaced when a source provides it; broadening that coverage is future work.
- **Novelty Check**: dedup a new idea against the ledger + already-read literature at the **result level** (config+metric hash, not code shape).
- **Hypothesis Gen**: force the output `{claim, source_paper, predicted_effect, how_to_implement}`. A hypothesis with no source is not allowed into the queue → narrows the search prior to "directions others have already validated," fighting search-space explosion.
- **Must-beat baseline injection**: pull the SOTA baseline from the profile into the prompt, to prevent regression to conservative methods seen in training data.

---

## 4.5 Reasoning Layer (ReAct-light) — how each iteration gets smarter

A bare optimize-the-metric loop reasons implicitly and shallowly (guess → run → keep/discard), which yields
local hill-climbing and search-space explosion. ScholarLoop makes reasoning an **explicit,
auditable, measurable** artifact via three mechanisms layered on the loop. Structure:
**ReAct-light** — one reason→act→observe chain per iteration. Cheap (one LLM call before a
run, one after), and it drops straight into the existing loop.

### Components
- **Reasoner** — runs once per iteration, between L3 (a cited idea) and L2 (the run). Inputs:
  the relevant ledger slice, the idea's literature, the active skill library. Output: a
  `ReasoningTrace` that justifies the next experiment **and bounds its search space**.
- **Reflector** — runs once after the result is recorded. Scores prediction vs measurement,
  distills one lesson into L0's skill library, and updates search-space state.

```
L3 idea ─► Reasoner ─► (diff + prediction + constraints) ─► L2 run ─► result ─► Reflector ─► L0 skills
   ▲                                                                                  │
   └──────────────── PIVOT (constraints say the region is exhausted) ◄────────────────┘
```

**Two edit channels.** The Reasoner acts through whichever fits the move:
- *config override* — a structured hyperparameter dict the runner injects via env (no source
  mutation, parallel-safe). Covers the hparam search space; the effective config is logged.
- *source diff* — editing `train.py` for genuine architecture changes; needs per-run isolation
  (temp copy / worktree, Phase 2) and passes the `forbidden_edits` guard before running.
Both report the exact config that ran, so the ledger and search-space reasoning stay honest.

### Data added to the ledger entry
```jsonc
"reasoning": {
  "trace": "ledger: lr∈[0.2,0.4] all lost; cosine helped in arXiv:1608.03983 → try lr=0.1 + cosine",
  "act": "set HPARAMS.lr=0.1, schedule=cosine",            // the concrete edit
  "search_space_constraints": {
    "ruled_out": ["lr>0.3 (3 losses)", "depth>30 (diverged)"],
    "focus":     ["schedule", "warmup"],                    // high-leverage knobs (sensitivity)
    "priors":    ["lr>0.3 diverges (arXiv:1608.03983)"]
  }
},
"prediction": {"predicted": -0.4, "measured": -0.1, "calibration_error": 0.3}  // filled by Reflector
```

### The three mechanisms
1. **Predict-then-Verify (makes reasoning measurable).** Each iteration predicts a
   direction+magnitude *before* running, then scores it against the registry-captured result.
   A running calibration score per reasoning source (a knob, a paper, a skill) up-weights what
   predicts well and down-weights "reasoning theater." The quantified analogue of a PhD's intuition.
2. **Search-Space Reasoning (limits the search — the headline win).** The Reasoner derives
   `search_space_constraints` from the ledger: *region pruning* (cluster prior runs into
   exhausted / promising / unexplored, exclude exhausted), *sensitivity* (which knobs actually
   move the metric vs noise → focus edits there), *literature priors* (paper claims → hard bounds).
   Constraints are injected into the code-editing prompt **and enforced post-edit** — an edit
   outside them is discarded like a forbidden edit, so the limit actually binds.
3. **Reflection (compounds intuition).** After each result, distill one lesson
   `{category, severity, mitigation}` into the skill library (§2.4) with 30-day time-decay.
   Lessons are keyed so a judge-rejected direction does not silently reappear next round.

### Why it's safe / cheap
- One extra LLM call per iteration + one after — negligible vs a GPU run.
- The trace lives in the ledger → you can replay *why* the system tried each thing.
- Calibration demotes unhelpful reasoning by measurement, not faith (counters reasoning theater).
- Constraints are enforced, not merely suggested.

### Phase mapping
- Mechanisms 1 + 2 land in **Phase 1** (with the Idea Engine): the Reasoner consumes
  literature + ledger and emits constraints; predict-verify scores against the registry.
- Mechanism 3 (skill library + decay) matures in **Phase 4**, but the Reflector stub that
  *writes* lessons starts in Phase 1 so data accrues early.

---

## 4.6 Multi-agent harness — how the agents are driven

The system is multi-agent, but the *harness* matters more than the agent count. Rules:

1. **Deterministic control flow; model-driven steps.** The Orchestrator owns the loop,
   fan-out, gating, and budget (code). Agents only do the open-ended reasoning. An LLM never
   decides control flow.
2. **Typed I/O contract per agent.** Every agent declares a JSON schema; the harness
   validates the output at the boundary and **retries with feedback** on a structural
   failure, so downstream code always receives a well-formed object — never prose to parse.
3. **Checkable work stays in code.** Dedup, calibration, and constraint *enforcement* are
   deterministic (`reasoning.py`), not delegated to an agent. The agent proposes; the code
   verifies. The search-space limit binds because the Orchestrator gates on it, not because
   the agent was asked nicely.
4. **Tools are deterministic; extraction is an agent.** The Lit Scout's arXiv retrieval is
   plain HTTP (injectable, mockable); only turning papers into structured findings is an LLM
   step. Same split everywhere: fetch with code, judge with an agent.
5. **Every agent call is traced** (`AgentTrace`: agent name + attempts + output) for
   auditability — and that's where token/cost accounting plugs in once usage is surfaced.
6. **No agent holds durable state.** Agents are stateless functions of their context; all
   memory lives in the ledger. This keeps long runs from drifting (the AI-Researcher failure mode).

```
Orchestrator (deterministic loop)
  ├─► LitScout     arXiv + OpenAlex → impact-ranked cited findings     [built]
  ├─► Reasoner     constraints + lit + skills → structured proposal    [built]
  ├─► DebatePanel  Innovator/Pragmatist/Contrarian vote run/reject     [built]
  ├─► Reflector    run outcome (esp. calibration_error) → skill lesson [built]
  └─► Advisor      post-run PROCEED / REFINE / PIVOT (bounded refines)  [built]

L4  Director       ledger trend + lit hotspots → direction + topic + budget   [built]
L5  Writer+Reviewer findings → grounded draft → peer review                   [built]
  gates on: forbidden_edits, ruled-out regions, dedup, debate-majority  (code/panel, never an LLM-judge on the metric)
```

Base class `agent.py::Agent` owns validate → retry → trace; `Reasoner`, `LitScout`, the
`DebatePanel` critics, `Reflector`, `Advisor`, `Director`, and the L5 `Writer`/`Reviewer` all
run on it, logging to one shared `AgentTrace`. The debate gives diverse/adversarial *judgment
on whether to spend a GPU* — never an LLM-judge on the metric itself (the measured result
decides quality). The Advisor steers the loop (PROCEED/REFINE/PIVOT, with refines bounded so a
dead idea can't loop forever); the Director sets the campaign's direction and re-grounds on a PIVOT.

---

## 5. L5 Writer + Reviewer (`paper.py`, built)
- `gather_findings` collects `verdict=kept` ledger entries with a measured score.
- **`Writer`** drafts a short paper (title / abstract / sections) using only those numbers.
- **Number grounding gate** (`grounded_registry` + `audit_draft`): every number in the draft
  must trace to a recorded fact — a captured measurement or a logged config value — or it is
  flagged. Deterministic; the Writer proposes, the registry verifies. Prose can't invent results.
- **`Reviewer`** critiques (strengths / weaknesses / score 1-10 / recommendation). The LLM judge
  is used for the qualitative review only, **never as the optimization target** (the metric decides quality).
- Optional: hand a grounded, kept-findings draft to the ARS `academic-paper` / `-reviewer`
  skills for a heavier external pass.

---

## 6. Phased rollout

| Phase | Content | Output | Acceptance |
|---|---|---|---|
| **0** | Reuse the minimal experiment kernel + wrap with `ledger.jsonl` | An experiment engine that logs | Run one idea fully logged |
| **1** | L3 idea engine (reuse skills) + Reasoning Layer (§4.5): Reasoner emits search-space constraints, predict-verify scores vs registry, Reflector stub writes lessons | Cited ideas + a reasoning trace per run | 100% of ideas have a source; every run logs a prediction + calibration_error |
| **2** | L2 multi-fidelity funnel + VerifiedRegistry + forbidden_edits guard | Anti-self-deception orchestration | An injected cheating idea gets blocked |
| **3** | Profile-ize domains (one each for vision/nlp/rl) | Pluggable L1 | Add a new domain without touching orchestrator |
| **4** | L4 Director + Skill Library self-evolution | Sets direction, accumulates intuition | Retry rate drops on the 2nd round |
| **5** | L5 Writer + Reviewer | Submission-grade loop | Produce a draft with real numbers |

---

## 7. The hard problems (know them up front)

1. **Reward hacking is the #1 enemy**: give write-access to optimize a single metric and the agent *will* optimize the metric itself (edit eval / overfit val / shorten sequences). → `forbidden_edits` guard + frozen held-out + VerifiedRegistry, three lines of defense.
   - **§7.1 Two-phase frozen scoring (RESOLVED, hardened).** The editable `train.py` only trains and saves a model to `$SCHOLARLOOP_ARTIFACT`; it gets the training split *only* (`prepare.load_train()`), never the val set. The runner then runs the FROZEN `<fixed_module> score <artifact>` **from pristine ROOT** as a separate step and trusts only its number — train.py's stdout is ignored. Hardening from adversarial review:
     - *Fabricated metric* — blocked (the `cheater` engine + test: a faked `0.1` is overridden by the honest score).
     - *Path traversal / import-shadowing via the edit channel* — the source-diff channel may ONLY replace the train entrypoint (strict allowlist, normpath-checked), with a containment guard in `_engine_root`. A `../../scholarloop/x.py` or new-shadow-file edit is refused before any write.
     - *Runtime tamper of the scorer* — the scorer runs from pristine ROOT, never the edited engine copy, so train.py can't reach the scoring code through its working dir.
     - *Arbitrary-unpickle code-exec in the scorer* — the torch engine saves a `state_dict` and the frozen `build_model` reconstructs the architecture; the scorer loads with `weights_only=True` (no agent objects unpickled).
     - *Runtime tamper of the frozen scorer* — a frozen-module integrity hash is taken at the start
       of the run and re-checked before every score; if `train.py` overwrote the frozen `prepare.py`
       at runtime, the result is refused (the run is killed, so the agent gains nothing). Detection,
       not prevention.
     - **Residual boundary (needs sandboxing, documented honestly):** the integrity hash *detects*
       a tampered scorer but doesn't *prevent* arbitrary filesystem writes; and for a *public* dataset
       the val split is reconstructable (deterministic split over public data). True prevention needs
       container isolation of the train process + a runner-provisioned private holdout — an infra layer
       beyond these guards. The current guards defeat the realistic (LLM-proposing-edits) threat model,
       not a hostile process with unrestricted filesystem access.
2. **Statistical noise**: wins from a 5-min / 120s run are mostly noise. → multi-seed + a verify layer for confirmation; don't let a noisy winner pollute the ledger.
3. **Search-space explosion**: → literature grounding narrows the prior to validated directions, and the Reasoning Layer (§4.5) prunes exhausted regions + focuses high-leverage knobs each round.
4. **Cost**: 100 experiments/night × seeds × full-runs is a real bill. → Director budget allocation + auto-pause at 50/80/100% is survival, not a nice-to-have.
5. **Novel ≠ effective**: if no one in the literature did it, maybe it doesn't work. → novelty must pair with a sanity eval (must-beat baseline).
6. **Long-horizon memory decay** (AI-Researcher's own admission): detail gets compressed out of context over long runs. → keep all state in the ledger, never in the context window.

---

## 8. Wiring to existing skills (don't reinvent)
- Literature: `literature-survey` · `gaps` · `deep-research`
- Experiment design / debugging: `experiment-design` · `debug` · `launch` · `compare`
- Reproducing new architectures: `reproduce` (arXiv → smoke → replicate)
- Writing / review: `academic-paper` · `academic-paper-reviewer` · `xray` · `factcheck`

---

> Path: Phase 0→1 first gets the minimal loop working — "read paper → generate idea → edit code → validate → remember." Everything else layers on incrementally.
