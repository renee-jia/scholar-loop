"""Experiment runner: the Phase-0 L1/L2 plumbing that ties the three contracts together.

Given a domain Profile and a fidelity tier, the runner:
  1. derives the train module from `profile.train_entrypoint`,
  2. runs it once per seed as a subprocess under the tier's wall-clock budget,
  3. parses the `SCHOLARLOOP_RESULT` line each run emits (via the frozen `prepare` module),
  4. captures every measured number into a VerifiedRegistry,
  5. compares the aggregate against the profile's must-beat baseline to set a verdict,
  6. appends one record to the Experiment Ledger.

This is the minimal "read idea -> run -> measure -> remember" loop.
The agent that edits train.py and the L3 idea engine plug in above this; both are later phases.

CLI:
    python -m scholarloop.runner run \
        --profile profiles/image-classification.yaml --fidelity smoke \
        --id exp_0001 --claim "lower lr helps" --source arXiv:1609.04836
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean, stdev

from scholarloop.ledger import Hypothesis, Ledger, LedgerEntry
from scholarloop.profile import Profile, load_profile
from scholarloop.reasoning import prediction_from_scores
from scholarloop.registry import VerifiedRegistry

_RESULT_RE = re.compile(r"^SCHOLARLOOP_RESULT (\{.*\})\s*$", re.MULTILINE)

# Seeds per fidelity tier. Confidence grows monotonically up the funnel: smoke is a single-seed
# coarse screen, verify is a 3-seed check, and full is the widest 5-seed confirmation. Full must
# strictly extend verify's seed set (not just re-run smoke's seed 0, which is deterministic and
# would make the final tier a vacuous repeat) so the terminal number is the tightest estimate.
SEEDS_PER_FIDELITY = {"smoke": [0], "verify": [0, 1, 2], "full": [0, 1, 2, 3, 4]}

ROOT = Path(__file__).resolve().parent.parent


class RunError(RuntimeError):
    """A train run failed, timed out, or emitted no parseable result."""


def _module_from_entrypoint(entrypoint: str) -> str:
    """'engines/vision/train.py' -> 'engines.vision.train'."""
    return entrypoint.removesuffix(".py").replace("/", ".")


def _engine_root(train_entrypoint: str, edits: list[dict]) -> Path:
    """Copy the engine package into a throwaway temp root and apply the agent's edits there.

    This is the source-diff edit channel under per-run isolation: architecture
    edits run against a copy, so the real repo source is never mutated and concurrent runs can't
    clobber each other. Edits to frozen surfaces are rejected upstream, so the copied
    `prepare.py` stays original — the frozen scorer's integrity holds.
    """
    pkg_rel = Path(train_entrypoint).parent           # e.g. engines/torch_vision
    root = Path(tempfile.mkdtemp(prefix="scholarloop_engine_"))
    shutil.copytree(ROOT / pkg_rel, root / pkg_rel)
    anc = pkg_rel.parent                              # make ancestor packages importable
    while str(anc) not in (".", ""):
        init = root / anc / "__init__.py"
        init.parent.mkdir(parents=True, exist_ok=True)
        if not init.exists():
            init.write_text("")
        anc = anc.parent
    root_resolved = root.resolve()
    for e in edits:
        dst = (root / e["path"]).resolve()
        # containment guard: never let an edit path (e.g. "../../x") escape the temp copy
        if root_resolved not in dst.parents:
            raise ValueError(f"edit path escapes the engine root: {e['path']}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(e["content"])
    return root


def _run(module_args: list[str], cwd: Path, timeout_sec: int, extra_env: dict, seed: int, what: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(extra_env)
    try:
        return subprocess.run([sys.executable, "-m", *module_args], cwd=str(cwd), env=env,
                              capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        raise RunError(f"seed {seed}: {what} exceeded budget of {timeout_sec}s")


def _run_one_seed(train_module: str, score_module: str, seed: int, timeout_sec: int,
                  config_override: dict | None, engine_cwd: Path,
                  frozen_guard: tuple = (None, None)) -> dict:
    """Two-phase run for one seed — the reward-hacking guard.

    Phase 1 runs the EDITABLE train module, which must write a model artifact to
    $SCHOLARLOOP_ARTIFACT. Its stdout is intentionally ignored. Phase 2 runs the FROZEN
    `<fixed_module> score <artifact>` step; only ITS emitted number is trusted. So an edited
    train.py cannot fabricate the metric — it can only produce a model the frozen scorer judges.
    Both phases run from `engine_cwd` (the repo root, or an isolated engine copy when editing).
    """
    extra = {"SCHOLARLOOP_SEED": str(seed)}
    if config_override:
        extra["SCHOLARLOOP_CONFIG"] = json.dumps(config_override)
    fd, artifact = tempfile.mkstemp(prefix="scholarloop_art_")
    os.close(fd)
    extra["SCHOLARLOOP_ARTIFACT"] = artifact
    try:
        p1 = _run([train_module], engine_cwd, timeout_sec, extra, seed, "train")
        if p1.returncode != 0:
            raise RunError(f"seed {seed}: train exit {p1.returncode}\n{p1.stderr[-500:]}")
        if not os.path.exists(artifact) or os.path.getsize(artifact) == 0:
            raise RunError(f"seed {seed}: train produced no artifact")

        # Frozen scorer — the trusted metric — runs from PRISTINE ROOT, never the (possibly
        # edited / runtime-tampered) engine copy. Plus: verify the scorer file wasn't tampered
        # at runtime before we trust it.
        _verify_frozen(*frozen_guard)
        p2 = _run([score_module, "score", artifact], ROOT, max(120, timeout_sec), extra, seed, "score")
        if p2.returncode != 0:
            raise RunError(f"seed {seed}: score exit {p2.returncode}\n{p2.stderr[-500:]}")
        matches = _RESULT_RE.findall(p2.stdout)
        if not matches:
            raise RunError(f"seed {seed}: frozen scorer emitted no result")
        return json.loads(matches[-1])
    finally:
        try:
            os.remove(artifact)
        except OSError:
            pass


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_frozen(path: Path, expected: str | None) -> None:
    """Refuse to score if the frozen module changed since the run started — a defense against an
    adversarial train.py overwriting (or deleting) the trusted scorer at runtime.
    Detection, not prevention: a tampered run is killed, so the agent gains nothing. Full FS isolation
    needs a sandbox."""
    if expected is None:
        return
    if not path.exists() or _file_hash(path) != expected:   # gone OR modified — both are tampering
        raise RunError(f"frozen scorer {path.name} was modified or removed during training — "
                       f"refusing to trust the result")


def _run_seeds(seeds: list[int], run_one):
    """Run each seed, tolerating per-seed failures. Returns (results, failures). Raises RunError
    only if EVERY seed failed — so a single crashed/timed-out seed in a multi-seed verify run
    doesn't void the seeds that did succeed."""
    results, failures = [], []
    for s in seeds:
        try:
            results.append(run_one(s))
        except RunError as e:
            failures.append(f"seed {s}: {e}")
    if not results:
        raise RunError("; ".join(failures) or "no seeds ran")
    return results, failures


def run_experiment(
    profile: Profile,
    fidelity: str,
    exp_id: str,
    hypothesis: Hypothesis,
    *,
    config_override: dict | None = None,
    edits: list[dict] | None = None,
    diff: str = "",
    parent: str | None = None,
    reasoning: dict | None = None,
    predicted_delta: float | None = None,
    parent_score: float | None = None,
    ledger_path: str | Path = "ledger.jsonl",
    registry_dir: str | Path = "registry",
) -> LedgerEntry:
    """Run one experiment at one fidelity tier and record it. Returns the ledger entry.

    Two edit channels: `config_override` sweeps the hyperparameter space without
    mutating source; `edits` (list of {path, content}) is the source-diff channel, applied in an
    isolated engine copy. Edits to frozen surfaces are refused here (the `forbidden_edits` guard,
    defense-in-depth — the orchestrator also gates them before calling).
    """
    edits = edits or []
    # Strict allowlist: the source-diff channel may ONLY replace the train entrypoint. This
    # subsumes the forbidden_edits denylist and blocks path traversal, frozen-file edits, and
    # import-shadowing new files in one rule.
    allowed = os.path.normpath(profile.train_entrypoint)
    illegal = [e["path"] for e in edits if os.path.normpath(e["path"]) != allowed]
    if illegal:
        raise ValueError(f"edits may only replace {profile.train_entrypoint!r}; refused: {illegal}")

    train_module = _module_from_entrypoint(profile.train_entrypoint)
    score_module = _module_from_entrypoint(profile.fixed_module)   # frozen scorer owns the metric
    frozen_file = ROOT / profile.fixed_module
    frozen_hash = _file_hash(frozen_file) if frozen_file.exists() else None
    budget = profile.budget.seconds_for(fidelity)
    seeds = SEEDS_PER_FIDELITY[fidelity]
    engine_root = _engine_root(profile.train_entrypoint, edits) if edits else None
    engine_cwd = engine_root if engine_root is not None else ROOT

    config: dict = {}
    agg: float | None = None
    registry = VerifiedRegistry(Path(registry_dir) / f"{exp_id}.json", exp_id=exp_id)
    try:
        results, failures = _run_seeds(
            seeds, lambda s: _run_one_seed(train_module, score_module, s, budget,
                                           config_override, engine_cwd, (frozen_file, frozen_hash)))
        per_seed = [float(r["value"]) for r in results]
        config = results[0].get("config", {})   # hparams are fixed across seeds for one run
        agg = round(mean(per_seed), 4)
        std = round(stdev(per_seed), 4) if len(per_seed) > 1 else 0.0
        registry.capture(f"{profile.metric.name}.{fidelity}", agg, seeds=per_seed, std=std)
        registry.save()
        baseline = profile.best_baseline()
        if baseline is None:
            verdict = "kept"  # nothing to beat yet
        else:
            verdict = "kept" if profile.metric.is_better(agg, baseline) else "discarded"
        metric_blob = {"name": profile.metric.name, fidelity: agg, "seeds": per_seed}
        if failures:                            # partial run: some seeds crashed but enough survived
            metric_blob["seeds_failed"] = len(failures)
    except RunError as e:
        registry.status = "killed"   # mark, so an empty registry isn't mistaken for a clean run
        registry.save()
        verdict, metric_blob = "killed", {"name": profile.metric.name, "error": str(e)}
    finally:
        if engine_root is not None:
            shutil.rmtree(engine_root, ignore_errors=True)

    prediction = prediction_from_scores(predicted_delta, agg, parent_score)
    entry = LedgerEntry(
        id=exp_id, domain=profile.name, hypothesis=hypothesis,
        metric_name=profile.metric.name, parent=parent, diff=diff, config=config,
        fidelity=[fidelity], metric=metric_blob,
        reasoning=reasoning or {}, prediction=prediction, verdict=verdict,
        registry_id=exp_id,
    )
    Ledger(ledger_path).append(entry)
    return entry


def _main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="scholarloop.runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one experiment at one fidelity")
    r.add_argument("--profile", required=True)
    r.add_argument("--fidelity", choices=list(SEEDS_PER_FIDELITY), default="smoke")
    r.add_argument("--id", required=True)
    r.add_argument("--claim", required=True)
    r.add_argument("--source", required=True, help="literature source; ungrounded ideas are rejected")
    r.add_argument("--parent", default=None)
    r.add_argument("--ledger", default="ledger.jsonl")
    r.add_argument("--registry-dir", default="registry")
    a = ap.parse_args(argv)

    profile = load_profile(a.profile)
    entry = run_experiment(
        profile, a.fidelity, a.id,
        Hypothesis(claim=a.claim, source=a.source),
        parent=a.parent, ledger_path=a.ledger, registry_dir=a.registry_dir,
    )
    print(f"{entry.id}  verdict={entry.verdict}  {entry.metric_name}={entry.primary_score()}  "
          f"(baseline={profile.best_baseline()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
