"""Content-addressed candidate implementations, separate from repeated runs.

Snapshots cover the local engine Python sources, scorer, requested configuration and
declared execution context. They do not certify undeclared data or dependencies.
"""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from pathlib import Path


class CandidateError(ValueError):
    pass


def content_id(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def execution_context(profile):
    versions = {}
    for name in ("numpy", "scikit-learn", "torch", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"domain": profile.name, "metric": profile.metric.name,
            "direction": profile.metric.direction, "datasets": list(profile.datasets),
            "cv_folds": profile.cv_folds, "python": platform.python_version(),
            "packages": versions}


def engine_sources(profile):
    root = profile.root.resolve()
    train, scorer = root / profile.train_entrypoint, root / profile.fixed_module
    paths = set(train.parent.rglob("*.py")) | {train, scorer}
    sources = {}
    for path in sorted(paths):
        if root not in path.resolve().parents:
            raise CandidateError("Engine sources must remain inside the profile root")
        # Profile entrypoints use '/' on every OS; snapshot keys must use the same spelling.
        sources[path.relative_to(root).as_posix()] = path.read_text() if path.is_file() else None
    return sources


def validate_snapshot(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
        raise CandidateError("Unsupported candidate snapshot")
    required = {"version", "id", "train_entrypoint", "fixed_module", "sources", "config", "context"}
    if set(snapshot) != required or snapshot["id"] != content_id({k: v for k, v in snapshot.items() if k != "id"}):
        raise CandidateError("Candidate snapshot was modified")
    if (not isinstance(snapshot["sources"], dict) or not isinstance(snapshot["config"], dict)
            or not isinstance(snapshot["context"], dict)):
        raise CandidateError("Invalid candidate sources or configuration")
    if any(not isinstance(v, (str, type(None))) for v in snapshot["sources"].values()):
        raise CandidateError("Candidate sources must be text or explicitly missing")
    for name in (snapshot["train_entrypoint"], snapshot["fixed_module"], *snapshot["sources"]):
        if not isinstance(name, str) or not name:
            raise CandidateError("Invalid candidate source path")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name not in snapshot["sources"]:
            raise CandidateError("Invalid candidate source path")
    return snapshot


def check_environment(snapshot, profile):
    validate_snapshot(snapshot)
    if (snapshot["train_entrypoint"] != profile.train_entrypoint or
            snapshot["fixed_module"] != profile.fixed_module or
            snapshot["context"] != execution_context(profile)):
        raise CandidateError("Candidate execution context changed; start a new root candidate")
    current = engine_sources(profile)
    # Only train.py is editable; inherited implementations require the same dependencies.
    old = {k: v for k, v in snapshot["sources"].items() if k != profile.train_entrypoint}
    now = {k: v for k, v in current.items() if k != profile.train_entrypoint}
    if old != now:
        raise CandidateError("Candidate engine dependencies changed; start a new root candidate")


def make_snapshot(profile, config=None, edits=(), *, base=None):
    if base is not None:
        check_environment(base, profile)
    sources = dict(base["sources"]) if base is not None else engine_sources(profile)
    seen = set()
    for edit in edits:
        if edit["path"] != profile.train_entrypoint or edit["path"] in seen:
            raise CandidateError("Candidate edits may only replace the train entrypoint once")
        if not isinstance(edit["content"], str):
            raise CandidateError("Candidate source must be text")
        seen.add(edit["path"])
        sources[edit["path"]] = edit["content"]
    value = {"version": 1, "train_entrypoint": profile.train_entrypoint,
             "fixed_module": profile.fixed_module, "sources": sources,
             "config": {**profile.knobs, **(base["config"] if base else {}), **(config or {})},
             "context": execution_context(profile)}
    if any(isinstance(v, float) and not math.isfinite(v) for v in value["config"].values()):
        raise CandidateError("Candidate configuration must be finite")
    # JSON distinguishes 18 from 18.0, although they request the same numeric knob.
    value["config"] = {k: int(v) if isinstance(v, float) and v.is_integer() else v
                       for k, v in value["config"].items()}
    return {**value, "id": content_id(value)}


class CandidateStore:
    def __init__(self, registry_dir):
        self.path = Path(registry_dir) / "candidates"

    def load(self, candidate_id):
        if not isinstance(candidate_id, str) or len(candidate_id) != 64 or any(c not in "0123456789abcdef" for c in candidate_id):
            raise CandidateError("Invalid candidate id")
        try:
            value = validate_snapshot(json.loads((self.path / f"{candidate_id}.json").read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            raise CandidateError(f"Missing or unreadable candidate {candidate_id}") from exc
        if value["id"] != candidate_id:
            raise CandidateError("Candidate file belongs to another implementation")
        return value

    def save(self, snapshot):
        validate_snapshot(snapshot)
        self.path.mkdir(parents=True, exist_ok=True)
        destination = self.path / f"{snapshot['id']}.json"
        if destination.exists():
            if self.load(snapshot["id"]) != snapshot:
                raise CandidateError("Candidate id collision")
            return
        fd, temporary = tempfile.mkstemp(dir=self.path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
