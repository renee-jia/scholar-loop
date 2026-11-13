"""Small, versioned contracts. Scientific judgement belongs to reviewers, not hashes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

VERSION = 2
ROLES = ("implementation", "experimental", "falsification", "argument")


class ReviewBlocked(RuntimeError):
    """The current evidence/argument has no valid permission to enter writing."""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, default=str).encode()).hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number: {s}")))
    except (OSError, ValueError) as e:
        raise ReviewBlocked(f"Cannot read {path}: {e}") from e


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def obj(properties, required=None):
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": list(properties) if required is None else required}


TEXT = {"type": "string", "minLength": 1}
STRINGS = {"type": "array", "items": TEXT}
CLAIM_SCHEMA = obj({
    "id": {"type": "string", "pattern": "^claim_[a-zA-Z0-9_]+$"},
    "text": TEXT,
    "kind": {"type": "string", "enum": ["observation", "comparison", "mechanism", "generalization"]},
    "evidence": {"type": "array", "items": TEXT, "minItems": 1, "uniqueItems": True},
    "measurements": {"type": "array", "items": TEXT, "uniqueItems": True},
    "depends_on": {"type": "array", "items": TEXT, "uniqueItems": True},
    "scope": TEXT,
})
# A paper needs more than conclusions. `setup` holds the facts a reader needs to rerun the study
# (settings, sizes, seeds), each tied to the record that states it. `related_work` says how the
# study stands to a reviewed paper. Both are reviewed and approved like claims, so writing can
# use them; both are optional so earlier arguments stay valid.
SETUP_SCHEMA = obj({
    "id": {"type": "string", "pattern": "^setup_[a-zA-Z0-9_]+$"},
    "text": TEXT,
    "evidence": {"type": "array", "items": TEXT, "minItems": 1, "uniqueItems": True},
})
RELATED_SCHEMA = obj({
    "id": {"type": "string", "pattern": "^related_[a-zA-Z0-9_]+$"},
    "text": TEXT,
    "papers": {"type": "array", "items": TEXT, "minItems": 1, "uniqueItems": True},
})
STATEMENT_ID_PATTERN = (r"claim_[A-Za-z0-9_]+|setup_[A-Za-z0-9_]+|related_[A-Za-z0-9_]+"
                        r"|scope_question|scope_method|limit_\d+")
ARGUMENT_SCHEMA = obj({
    "question": TEXT, "method_summary": TEXT,
    "claims": {"type": "array", "items": CLAIM_SCHEMA, "minItems": 1},
    "limitations": {"type": "array", "items": TEXT, "minItems": 1},
    "setup": {"type": "array", "items": SETUP_SCHEMA},
    "related_work": {"type": "array", "items": RELATED_SCHEMA},
}, required=["question", "method_summary", "claims", "limitations"])
ISSUE_SCHEMA = obj({
    "target": TEXT,
    "severity": {"type": "string", "enum": ["blocker", "major", "minor"]},
    "observation": TEXT, "evidence": STRINGS, "required_action": TEXT,
    "resolution_criterion": TEXT,
    "action": {"type": "string", "enum": ["experiment", "reframe", "inspect", "human"]},
})
REVIEW_SCHEMA = obj({
    "verdict": {"type": "string", "enum": ["pass", "revise"]},
    "summary": TEXT, "issues": {"type": "array", "items": ISSUE_SCHEMA},
    "claim_checks": {"type": "array", "minItems": 1, "items": obj({
        "claim_id": TEXT, "question": TEXT, "answer": TEXT,
        "status": {"type": "string", "enum": ["supported_in_scope", "concern"]},
        "evidence": {**STRINGS, "minItems": 1, "uniqueItems": True},
    })},
    "dispositions": {"type": "array", "items": obj({
        "issue_id": TEXT,
        "status": {"type": "string", "enum": ["verified", "open", "out_of_scope"]},
        "reason": TEXT, "evidence": STRINGS,
    })},
})
