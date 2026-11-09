"""Domain Profile: the declarative contract that makes the framework domain-agnostic.

A profile is a YAML file (conventionally under `profiles/`) validated against the schema bundled
with the package (`scholarloop/core/profile.schema.yaml`). Engine paths in it are resolved from the
profile's own location (`Profile.root`), never from where scholarloop is installed.
The orchestrator reads a Profile to learn which file to edit, which metric is ground truth,
what the per-fidelity budgets are, and which edits are forbidden (the anti reward-hacking
boundary). It never hard-codes anything about the domain itself.

CLI:
    python -m scholarloop.core.profile validate profiles/image-classification.yaml
"""

from __future__ import annotations

import fnmatch
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from jsonschema import Draft7Validator

_SCHEMA_PATH = Path(__file__).resolve().parent / "profile.schema.yaml"   # ships inside the package


class ProfileError(ValueError):
    """Raised when a profile is missing or fails schema validation."""


@dataclass(frozen=True)
class Metric:
    name: str
    direction: str  # "minimize" | "maximize"
    vocab_independent: bool = False

    def is_better(self, candidate: float, reference: float) -> bool:
        """True if `candidate` beats `reference` under this metric's direction."""
        if self.direction == "minimize":
            return candidate < reference
        return candidate > reference


@dataclass(frozen=True)
class Budget:
    smoke_sec: int
    verify_sec: int
    full_sec: int

    def seconds_for(self, fidelity: str) -> int:
        try:
            return {"smoke": self.smoke_sec, "verify": self.verify_sec, "full": self.full_sec}[fidelity]
        except KeyError:
            raise ProfileError(f"unknown fidelity {fidelity!r}; expected smoke|verify|full")


@dataclass(frozen=True)
class Profile:
    name: str
    paradigm: str
    train_entrypoint: str
    fixed_module: str
    metric: Metric
    budget: Budget
    allowed_edits: tuple[str, ...]
    forbidden_edits: tuple[str, ...]
    datasets: tuple[str, ...] = ()
    baselines: tuple[str, ...] = ()
    knobs: dict = field(default_factory=dict)   # tunable hyperparameters (name → default value)
    debate_roles: tuple[str, ...] = ("Innovator", "Pragmatist", "Contrarian")
    holdout: bool = False   # the engine's frozen scorer also serves a held-out test split (see runner)
    cv_folds: int = 0       # >= 2: seed s is trained/validated on fold s % cv_folds (see runner)
    source_path: Path | None = field(default=None, compare=False)

    @property
    def root(self) -> Path:
        """The directory `train_entrypoint` / `fixed_module` are relative to. Resolved from the
        profile, not from the installed package, so a pip-installed scholarloop still finds a
        project's engines: the parent of a `profiles/` folder (the repo layout); else the nearest
        ancestor of the profile file that actually contains `train_entrypoint`; else the cwd."""
        if self.source_path is not None:
            here = Path(self.source_path).resolve().parent
            if here.name == "profiles":
                return here.parent
            for d in (here, *here.parents):
                if (d / self.train_entrypoint).exists():
                    return d
        return Path.cwd()

    def best_baseline(self) -> float | None:
        """The hardest must-beat baseline score (best under the metric direction), or None."""
        scores = []
        for b in self.baselines:
            try:
                scores.append(float(b.rsplit("@", 1)[1]))
            except (IndexError, ValueError):
                raise ProfileError(f"malformed baseline {b!r}; expected 'name@<float>'")
        if not scores:
            return None
        return min(scores) if self.metric.direction == "minimize" else max(scores)

    def touches_forbidden(self, paths: list[str]) -> list[str]:
        """Return the subset of `paths` that hit a forbidden edit surface.

        `forbidden_edits` entries are path globs (e.g. "engines/vision/prepare.py",
        "*/prepare.py", "prepare.py"). Matching is on whole path components / basenames,
        never raw substrings — so "models/retrieval.py" is NOT flagged by a "prepare.py"
        rule, and a spaced phrase can never silently fail to match a real path.
        """
        hits = []
        for p in paths:
            # normalize ('a/./b', 'a//b', '\\' separators) and case-fold for case-insensitive
            # filesystems, so trivial spelling variants can't evade the guard
            norm = os.path.normpath(p).replace(os.sep, "/").casefold()
            base = norm.rsplit("/", 1)[-1]
            for pat in self.forbidden_edits:
                pl = pat.casefold()
                if (fnmatch.fnmatch(norm, pl) or fnmatch.fnmatch(norm, f"*/{pl}")
                        or fnmatch.fnmatch(base, pl)):
                    hits.append(p)
                    break
        return hits


def _load_schema() -> dict:
    with _SCHEMA_PATH.open() as f:
        return yaml.safe_load(f)


def load_profile(path: str | Path) -> Profile:
    """Load and validate a domain profile. Raises ProfileError on any problem."""
    path = Path(path)
    if not path.exists():
        raise ProfileError(f"profile not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ProfileError(f"profile must be a mapping, got {type(raw).__name__}")

    validator = Draft7Validator(_load_schema())
    errors = sorted(validator.iter_errors(raw), key=lambda e: list(e.path))
    if errors:
        joined = "\n".join(f"  - {'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors)
        raise ProfileError(f"{path} failed schema validation:\n{joined}")

    m = raw["metric"]
    b = raw["budget"]
    return Profile(
        name=raw["name"],
        paradigm=raw["paradigm"],
        train_entrypoint=raw["train_entrypoint"],
        fixed_module=raw["fixed_module"],
        metric=Metric(m["name"], m["direction"], m.get("vocab_independent", False)),
        budget=Budget(b["smoke_sec"], b["verify_sec"], b["full_sec"]),
        allowed_edits=tuple(raw["allowed_edits"]),
        forbidden_edits=tuple(raw["forbidden_edits"]),
        datasets=tuple(raw.get("datasets", [])),
        baselines=tuple(raw.get("baselines", [])),
        knobs=dict(raw.get("knobs", {})),
        debate_roles=tuple(raw.get("debate_roles", ["Innovator", "Pragmatist", "Contrarian"])),
        holdout=bool(raw.get("holdout", False)),
        cv_folds=int(raw.get("cv_folds", 0)),
        source_path=path,
    )


def _main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "validate":
        print("usage: python -m scholarloop.core.profile validate <profile.yaml>", file=sys.stderr)
        return 2
    try:
        p = load_profile(argv[1])
    except ProfileError as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1
    print(f"OK: {p.name} ({p.paradigm}) metric={p.metric.name}/{p.metric.direction} "
          f"budget={p.budget.smoke_sec}/{p.budget.verify_sec}/{p.budget.full_sec}s "
          f"baseline={p.best_baseline()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
