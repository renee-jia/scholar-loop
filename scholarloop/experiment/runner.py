"""Experiment runner: the Phase-0 L1/L2 plumbing that ties the three contracts together.

Given a domain Profile and a fidelity tier, the runner:
  1. derives the train module from `profile.train_entrypoint`,
  2. runs it once per seed as a subprocess under the tier's wall-clock budget,
  3. parses the `SCHOLARLOOP_RESULT` line each run emits (via the frozen `prepare` module),
  4. captures every measured number into a VerifiedRegistry,
  5. compares the aggregate against the profile's must-beat baseline to set a verdict,
  6. appends one record to the Experiment Ledger.

This is the minimal "read idea -> run -> measure -> remember" loop.
The Reasoner's edit channel and the idea engine plug in above this (`orchestrator.py`, `session.py`).

CLI:
    python -m scholarloop.experiment.runner run \
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

from scholarloop.core.ledger import Hypothesis, Ledger, LedgerEntry
from scholarloop.core.profile import Profile, load_profile
from scholarloop.experiment.reasoning import prediction_from_scores
from scholarloop.core.registry import VerifiedRegistry

_RESULT_RE = re.compile(r"^SCHOLARLOOP_RESULT (\{.*\})\s*$", re.MULTILINE)

# Seeds per fidelity tier. Confidence grows monotonically up the funnel: smoke is a single-seed
# coarse screen, verify is a 3-seed check, and full is the widest 5-seed confirmation. Full must
# strictly extend verify's seed set (not just re-run smoke's seed 0, which is deterministic and
# would make the final tier a vacuous repeat) so the terminal number is the tightest estimate.
SEEDS_PER_FIDELITY = {"smoke": [0], "verify": [0, 1, 2], "full": [0, 1, 2, 3, 4]}


def fold_for_seed(seed: int, cv_folds: int) -> int | None:
    """The CV fold seed `seed` trains and validates on, or None without cross-validation.

    On a small dataset one fixed val split is too noisy to select on (a 442-sample dataset
    leaves ~70 per val fold). Pairing seed s with fold s % K reuses the seeds the funnel already runs: the full
    tier's five seeds cover all five folds, so the reported mean is a K-fold estimate and its std
    carries split noise as well as seed noise, at no extra compute. Every run at one tier sees the
    same folds, so runs stay comparable. The held-out test split is outside every fold."""
    return seed % cv_folds if cv_folds >= 2 else None

ROOT = Path(__file__).resolve().parents[2]  # the checkout root; real runs use profile.root (same thing here)


class RunError(RuntimeError):
    """A train run failed, timed out, or emitted no parseable result."""


def _module_from_entrypoint(entrypoint: str) -> str:
    """'engines/vision/train.py' -> 'engines.vision.train'."""
    return entrypoint.removesuffix(".py").replace("/", ".")


def _engine_root(train_entrypoint: str, edits: list[dict], src_root: Path = ROOT) -> Path:
    """Copy the engine package into a throwaway temp root and apply the agent's edits there.

    This is the source-diff edit channel under per-run isolation: architecture
    edits run against a copy, so the real repo source is never mutated and concurrent runs can't
    clobber each other. Edits to frozen surfaces are rejected upstream, so the copied
    `prepare.py` stays original — the frozen scorer's integrity holds.
    """
    pkg_rel = Path(train_entrypoint).parent           # e.g. engines/torch_vision
    root = Path(tempfile.mkdtemp(prefix="scholarloop_engine_"))
    shutil.copytree(src_root / pkg_rel, root / pkg_rel)
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
                  frozen_guard: tuple = (None, None), pristine: Path = ROOT,
                  holdout: bool = False, fold: int | None = None,
                  cached_artifact: Path | None = None) -> dict:
    """Two-phase run for one seed — the reward-hacking guard.

    Phase 1 runs the EDITABLE train module, which must write a model artifact to
    $SCHOLARLOOP_ARTIFACT. Its stdout is intentionally ignored. Phase 2 runs the FROZEN
    `<fixed_module> score <artifact>` step; only ITS emitted number is trusted. So an edited
    train.py cannot fabricate the metric — it can only produce a model the frozen scorer judges.
    Phase 1 runs from `engine_cwd` (the project root, or an isolated engine copy when editing);
    phase 2 always runs from `pristine` (the untouched project root).

    With `holdout`, the same artifact is scored a second time on the frozen held-out test split and
    returned as `holdout_value`. The caller stores it apart from the selection metric.

    With a CV `fold`, train.py gets $SCHOLARLOOP_FOLD (the engine's load_train() leaves that fold
    out) and the scorer is called as `score <artifact> val <fold>`. The fold the scorer evaluates
    comes from the runner's argv, never from the environment train.py ran in.

    With `cached_artifact`, the trained model is kept at that path instead of a temp file, and if
    it already exists training is skipped: the same model (same config, edits, train script, seed
    and fold; the caller keys the path on all of them) is only re-scored, so `"reused": True`.
    A higher tier then trains just its new seeds, and the full tier can still score the reused
    seeds on the held-out split.
    """
    extra = {"SCHOLARLOOP_SEED": str(seed)}
    if fold is not None:
        extra["SCHOLARLOOP_FOLD"] = str(fold)
    if config_override:
        extra["SCHOLARLOOP_CONFIG"] = json.dumps(config_override)
    reused = cached_artifact is not None and cached_artifact.exists() and cached_artifact.stat().st_size > 0
    if cached_artifact is not None:
        cached_artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact = str(cached_artifact)
    else:
        fd, artifact = tempfile.mkstemp(prefix="scholarloop_art_")
        os.close(fd)
    extra["SCHOLARLOOP_ARTIFACT"] = artifact
    ok = False
    try:
        if not reused:
            p1 = _run([train_module], engine_cwd, timeout_sec, extra, seed, "train")
            if p1.returncode != 0:
                raise RunError(f"seed {seed}: train exit {p1.returncode}\n{p1.stderr[-500:]}")
            if not os.path.exists(artifact) or os.path.getsize(artifact) == 0:
                raise RunError(f"seed {seed}: train produced no artifact")

        # Frozen scorer — the trusted metric — runs from PRISTINE ROOT, never the (possibly
        # edited / runtime-tampered) engine copy. Plus: verify the scorer file wasn't tampered
        # at runtime before we trust it.
        _verify_frozen(*frozen_guard)
        val_args = ["val", str(fold)] if fold is not None else []
        p2 = _run([score_module, "score", artifact, *val_args], pristine, max(120, timeout_sec), extra, seed, "score")
        if p2.returncode != 0:
            raise RunError(f"seed {seed}: score exit {p2.returncode}\n{p2.stderr[-500:]}")
        matches = _RESULT_RE.findall(p2.stdout)
        if not matches:
            raise RunError(f"seed {seed}: frozen scorer emitted no result")
        result = json.loads(matches[-1])
        if fold is not None:
            result["fold"] = fold
        if holdout:
            _verify_frozen(*frozen_guard)          # tampering still kills the seed, as for val
            result.update(_score_holdout(score_module, artifact, pristine, timeout_sec, extra, seed,
                                         fold))
        if reused:
            result["reused"] = True
        ok = True
        return result
    finally:
        if cached_artifact is None or not ok:      # keep a cached model only if it scored cleanly
            try:
                os.remove(artifact)
            except OSError:
                pass


def _score_holdout(score_module: str, artifact: str, pristine: Path, timeout_sec: int,
                   extra: dict, seed: int, fold: int | None = None) -> dict:
    """Score the artifact on the held-out test split: {"holdout_value": v} or {"holdout_error": msg}.
    Never raises on a scoring failure. The holdout is a reporting number, so a crash or a missing
    `test` split must not cost the seed its val score (which would change the selection metric)."""
    try:
        fold_arg = [str(fold)] if fold is not None else []
        p3 = _run([score_module, "score", artifact, "test", *fold_arg], pristine, max(120, timeout_sec),
                  extra, seed, "holdout score")
    except RunError as e:
        return {"holdout_error": str(e)}
    held = _RESULT_RE.findall(p3.stdout) if p3.returncode == 0 else []
    if not held:
        return {"holdout_error": f"seed {seed}: frozen scorer served no holdout result (exit "
                                 f"{p3.returncode}; does the engine support `score <artifact> test`?)"}
    return {"holdout_value": float(json.loads(held[-1])["value"])}


def _artifact_key(config_override: dict | None, edits: list[dict], train_file: Path) -> str:
    """What makes two seed runs the same training: config, edits and the train script itself."""
    h = hashlib.sha256(json.dumps({"config": config_override or {}, "edits": edits},
                                  sort_keys=True, default=str).encode())
    h.update(train_file.read_bytes() if train_file.exists() else b"")
    return h.hexdigest()[:24]


def _cache_path(cache: str | Path | None, key: str | None, seed: int) -> Path | None:
    return None if cache is None or key is None else Path(cache) / f"{key}_seed{seed}.artifact"


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


def _aggregate_holdout(results: list[dict], registry: VerifiedRegistry, metric: str, exp_id: str) -> dict:
    """The held-out blob over the seeds whose test scoring succeeded; errors are kept, and when no
    seed has a holdout score the blob carries only the error (warned on stderr: usually an engine
    without `test` support on a `holdout: true` profile, a configuration problem)."""
    held = [r["holdout_value"] for r in results if "holdout_value" in r]
    errors = [r["holdout_error"] for r in results if "holdout_error" in r]
    if not held:
        print(f"warning: {exp_id}: no held-out score ({errors[0]})", file=sys.stderr)
        return {"error": errors[0]}
    h_agg = round(mean(held), 4)
    h_std = round(stdev(held), 4) if len(held) > 1 else 0.0
    registry.capture(f"{metric}.holdout", h_agg, seeds=held, std=h_std)
    registry.save()
    blob = {"value": h_agg, "seeds": held, "std": h_std}
    if errors:
        blob["seeds_failed"] = len(errors)
    return blob


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
    artifact_cache: str | Path | None = None,
) -> LedgerEntry:
    """Run one experiment at one fidelity tier and record it. Returns the ledger entry.

    Two edit channels: `config_override` sweeps the hyperparameter space without
    mutating source; `edits` (list of {path, content}) is the source-diff channel, applied in an
    isolated engine copy. Edits to frozen surfaces are refused here (the `forbidden_edits` guard,
    defense-in-depth — the orchestrator also gates them before calling).

    `artifact_cache` (a directory) keeps each seed's trained model, keyed on the config, the edits,
    the train script's hash, the seed and the fold. A tier that needs a seed an earlier tier already
    trained re-scores that model instead of training it again (`metric["reused_seeds"]`).
    """
    edits = edits or []
    # Strict allowlist: the source-diff channel may ONLY replace the train entrypoint. This
    # subsumes the forbidden_edits denylist and blocks path traversal, frozen-file edits, and
    # import-shadowing new files in one rule.
    allowed = os.path.normpath(profile.train_entrypoint)
    illegal = [e["path"] for e in edits if os.path.normpath(e["path"]) != allowed]
    if illegal:
        raise ValueError(f"edits may only replace {profile.train_entrypoint!r}; refused: {illegal}")

    root = profile.root                                              # where the engines live
    train_module = _module_from_entrypoint(profile.train_entrypoint)
    score_module = _module_from_entrypoint(profile.fixed_module)   # frozen scorer owns the metric
    frozen_file = root / profile.fixed_module
    frozen_hash = _file_hash(frozen_file) if frozen_file.exists() else None
    budget = profile.budget.seconds_for(fidelity)
    seeds = SEEDS_PER_FIDELITY[fidelity]
    # Held-out test: scored on the terminal tier only, so the loop (which selects on val at every
    # tier) never sees it and it carries no selection bias. Stored under metric["holdout"], which
    # primary_score() and every agent prompt ignore; only the writing stage reads it.
    holdout = profile.holdout and fidelity == "full"
    engine_root = _engine_root(profile.train_entrypoint, edits, root) if edits else None
    engine_cwd = engine_root if engine_root is not None else root

    cache_key = (_artifact_key(config_override, edits, engine_cwd / profile.train_entrypoint)
                 if artifact_cache is not None else None)
    config: dict = {}
    agg: float | None = None
    registry = VerifiedRegistry(Path(registry_dir) / f"{exp_id}.json", exp_id=exp_id)
    try:
        results, failures = _run_seeds(
            seeds, lambda s: _run_one_seed(train_module, score_module, s, budget, config_override,
                                           engine_cwd, (frozen_file, frozen_hash), pristine=root,
                                           holdout=holdout, fold=fold_for_seed(s, profile.cv_folds),
                                           cached_artifact=_cache_path(artifact_cache, cache_key, s)))
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
        if profile.cv_folds >= 2:               # which fold each surviving seed was validated on
            metric_blob["folds"] = [r["fold"] for r in results]
        if holdout:
            metric_blob["holdout"] = _aggregate_holdout(results, registry, profile.metric.name, exp_id)
        reused = sum(1 for r in results if r.get("reused"))
        if reused:                              # models an earlier tier trained, re-scored here
            metric_blob["reused_seeds"] = reused
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
    ap = argparse.ArgumentParser(prog="scholarloop.experiment.runner")
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
