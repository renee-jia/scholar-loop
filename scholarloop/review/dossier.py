"""Strict snapshots of all runs, including failures; typed numeric references.

Unlike the search ledger reader this fails on corrupt/missing evidence. Current engine
code is explicitly distinguished from historical execution provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from jsonschema import Draft7Validator

from scholarloop.core.ledger import LedgerEntry
from scholarloop.core.registry import VerifiedRegistry, _number_tokens
from scholarloop.review.contracts import ARGUMENT_SCHEMA, ReviewBlocked, VERSION, digest, atomic_json


def snapshot(profile, ledger_path, registry_dir, research_dir=None) -> dict:
    ledger_path, registry_dir = Path(ledger_path).resolve(), Path(registry_dir).resolve()
    paths = {"ledger": ledger_path,
             "train": (profile.root / profile.train_entrypoint).resolve(),
             "scorer": (profile.root / profile.fixed_module).resolve()}
    if profile.source_path:
        paths["profile"] = Path(profile.source_path).resolve()
    if research_dir:
        for name in ("lit_review.json", "handoff.json", "research_brief.json"):
            paths[name] = Path(research_dir).resolve() / name
    for p in sorted(registry_dir.glob("*.json")):
        paths[f"registry/{p.name}"] = p.resolve()
    for p in sorted((registry_dir / "protocols").glob("*.json")):
        paths[f"protocols/{p.name}"] = p.resolve()
    files = {}
    for name, p in paths.items():
        raw = p.read_bytes() if p.is_file() else None
        if raw is None and name in {"ledger", "train", "scorer", "profile"}:
            raise ReviewBlocked(f"Required input is missing: {p}")
        files[name] = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
                       "text": raw.decode("utf-8") if raw is not None else None}
    contexts = {}
    for name, file in files.items():
        if name.startswith("protocols/") and not name.endswith(".analysis.json"):
            from scholarloop.experiment.spec import frozen_context
            try:
                envelope = _json(file["text"])
                contexts[name] = digest(frozen_context(profile, envelope["spec"]))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ReviewBlocked(f"Cannot snapshot protocol inputs: {name}: {exc}") from exc
    return {"profile": json.loads(json.dumps(asdict(profile), default=str)), "files": files,
            "protocol_contexts": contexts,
            "registry_dir": str(registry_dir),
            "research_dir": str(Path(research_dir).resolve()) if research_dir else None}


def _json(text):
    try:
        return json.loads(text, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (ValueError, TypeError) as exc:
        raise ReviewBlocked(f"Malformed evidence JSON: {exc}") from exc


def build_dossier(inputs: dict) -> dict:
    files, profile = inputs["files"], inputs["profile"]
    runs, measurements, checks = [], {}, []
    seen = set()
    for line in files["ledger"]["text"].splitlines():
        if not line.strip():
            continue
        try:
            raw = _json(line)
            e = LedgerEntry.from_dict(raw)
            e.validate()
        except (ValueError, KeyError, TypeError) as exc:
            raise ReviewBlocked(f"Invalid ledger record: {exc}") from exc
        if e.id in seen or not re.fullmatch(r"[A-Za-z0-9_-]+", e.id):
            raise ReviewBlocked(f"Duplicate/invalid run id: {e.id}")
        seen.add(e.id)
        if e.domain != profile["name"] or e.metric_name != profile["metric"]["name"]:
            raise ReviewBlocked(f"Run {e.id} does not match this profile/metric")
        rid = e.registry_id or e.id
        if not re.fullmatch(r"[A-Za-z0-9_-]+", rid):
            raise ReviewBlocked(f"Invalid registry id: {rid}")
        reg_file = files.get(f"registry/{rid}.json")
        if not reg_file:
            checks.append(f"{e.id}: missing registry")
            reg = {}
        else:
            reg = _json(reg_file["text"])
            if not isinstance(reg, dict) or not isinstance(reg.get("measurements"), dict):
                raise ReviewBlocked(f"{e.id}: registry has invalid structure")
            if reg.get("exp_id") != e.id:
                checks.append(f"{e.id}: registry belongs to another run")
            if e.primary_score() is not None and reg.get("status") != "ok":
                checks.append(f"{e.id}: scored run has non-ok registry")
            if reg.get("measurements") and (e.verdict not in ("kept", "discarded") or e.primary_score() is None):
                checks.append(f"{e.id}: measurements belong to an incomplete or killed run")
        for key, m in reg.get("measurements", {}).items():
            if not isinstance(m, dict):
                raise ReviewBlocked(f"{e.id}/{key}: invalid measurement structure")
            value = m.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ReviewBlocked(f"{e.id}/{key}: invalid measurement")
            seeds, std = m.get("seeds", []), m.get("std")
            scalars = [*seeds, *([] if std is None else [std])] if isinstance(seeds, list) else [None]
            if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in scalars):
                raise ReviewBlocked(f"{e.id}/{key}: invalid seed values or uncertainty")
            if std is not None and std < 0:
                raise ReviewBlocked(f"{e.id}/{key}: negative standard deviation")
            ref = f"{e.id}::{key}"
            measurements[ref] = {"run_id": e.id, "key": key, "value": value,
                                 "seeds": m.get("seeds", []), "std": m.get("std")}
        if e.primary_score() is not None:
            tier = next(f for f in ("full", "verify", "smoke") if f in e.metric)
            m = measurements.get(f"{e.id}::{e.metric_name}.{tier}")
            if not m or m["value"] != e.primary_score() or m["seeds"] != e.metric.get("seeds", []):
                checks.append(f"{e.id}: ledger and registry measurement disagree")
            elif m["seeds"] and abs(m["value"] - round(mean(m["seeds"]), 4)) > 1e-8:
                checks.append(f"{e.id}: aggregate disagrees with its recorded seed measurements")
        runs.append(raw)
    if not runs:
        checks.append("No experiments have been recorded")
    if not measurements:
        checks.append("No verified measurements are available")
    research = {k: _json(files[k]["text"]) for k in ("lit_review.json", "handoff.json", "research_brief.json")
                if k in files and files[k]["text"] is not None}
    papers = {p["arxiv_id"]: p for p in research.get("lit_review.json", {}).get("papers", [])}
    for p in research.get("handoff.json", {}).get("idea", {}).get("novelty_review", {}).get("papers", []):
        papers.setdefault(p["arxiv_id"], p)
    protocols = {}
    for name, file in files.items():
        if not name.startswith("protocols/") or name.endswith(".analysis.json"):
            continue
        envelope = _json(file["text"])
        pid = name.removeprefix("protocols/").removesuffix(".json")
        if digest(envelope) != pid:
            checks.append(f"Protocol {pid}: frozen specification was modified")
            continue
        ref = f"protocol:{pid}"
        protocols[ref] = {"specification": envelope, "analysis": None}
        related = [r for r in runs if r.get("protocol", {}).get("id") == pid]
        spec = envelope["spec"]
        conditions = {c["name"]: c for c in spec["conditions"]}
        names = [r["protocol"].get("condition") for r in related]
        if len(names) != len(set(names)) or set(names) - set(conditions):
            checks.append(f"Protocol {pid}: duplicate or unknown conditions")
            continue
        for r in related:
            if r.get("purpose") != spec["purpose"] or r["fidelity"] != [spec["fidelity"]] or \
                    r["metric"].get("requested_seeds") != spec["seeds"]:
                checks.append(f"{r['id']}: execution disagrees with the frozen protocol")
        analysis_file = files.get(f"protocols/{pid}.analysis.json")
        if analysis_file:
            from scholarloop.experiment.agenda import analyze
            if set(names) != set(conditions):
                checks.append(f"Protocol {pid}: analysis is missing planned conditions")
                continue
            expected = analyze({"protocol_id": pid, "results": [
                {"condition": r["protocol"]["condition"], "entry": r} for r in related]}, spec)
            analysis = _json(analysis_file["text"])
            if analysis != expected:
                checks.append(f"Protocol {pid}: derived analysis differs from recorded paired results")
                continue
            protocols[ref]["analysis"] = analysis
            for c in analysis["comparisons"]:
                if c["status"] == "complete":
                    key = f"{spec['metric']}.difference.{c['condition']}"
                    measurements[f"{ref}::{key}"] = {"run_id": ref, "key": key,
                        "value": c["mean_difference"], "seeds": c["deltas"], "std": None}
    for r in runs:
        pid = r.get("protocol", {}).get("id")
        if pid and f"protocol:{pid}" not in protocols:
            checks.append(f"{r['id']}: frozen experiment protocol is missing")
    return {"schema_version": VERSION, "input_hash": digest(inputs), "profile": profile,
            "input_manifest": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in files.items()},
            "runs": runs, "measurements": measurements, "papers": papers, "research": research,
            "protocols": protocols,
            "current_code": {k: files[k]["text"] for k in ("train", "scorer")},
            "checks": checks,
            "provenance_note": "Current engine source is not proof of historical execution. "
            "Legacy records may lack code/data/environment snapshots. Review the scope accordingly."}


def evidence_ids(dossier):
    return ({r["id"] for r in dossier["runs"]} | set(dossier["papers"]) |
            set(dossier.get("protocols", {})) | {"profile", "train", "scorer"})


def validate_argument(argument: dict, dossier: dict) -> list[str]:
    errors = [e.message for e in Draft7Validator(ARGUMENT_SCHEMA).iter_errors(argument)]
    if errors:
        return errors
    claims = argument["claims"]
    ids = [c["id"] for c in claims]
    if len(set(ids)) != len(ids):
        errors.append("Claim ids must be unique")
    known = evidence_ids(dossier)
    graph = {}
    for c in claims:
        graph[c["id"]] = c["depends_on"]
        if set(c["depends_on"]) - set(ids):
            errors.append(f"{c['id']}: unknown claim dependency")
        if set(c["evidence"]) - known:
            errors.append(f"{c['id']}: unknown evidence")
        reg = VerifiedRegistry(":claim:")
        for ref in c["measurements"]:
            m = dossier["measurements"].get(ref)
            if not m:
                errors.append(f"{c['id']}: unknown measurement {ref}")
            elif m["run_id"] not in c["evidence"]:
                errors.append(f"{c['id']}: measurement {ref} belongs to an uncited run")
            else:
                reg.capture(ref, m["value"])
                if m["std"] is not None:
                    reg.capture(ref + ".std", m["std"])
        # No global small-integer/config/year exceptions for scientific claims.
        bad = reg.audit_text(c["text"])
        if bad:
            errors.append(f"{c['id']}: numbers lack explicitly bound measurements: {bad}")
    visiting, done = set(), set()

    def visit(cid):
        if cid in visiting:
            return False
        if cid in done or cid not in graph:
            return True
        visiting.add(cid)
        ok = all(visit(x) for x in graph[cid])
        visiting.remove(cid)
        done.add(cid)
        return ok

    if not all(visit(cid) for cid in ids):
        errors.append("Claim dependencies must be acyclic")
    return errors


def approved_statements(dossier: dict) -> dict:
    a = dossier["argument"]
    result = {c["id"]: c for c in a["claims"]}
    for cid, text in [("scope_question", a["question"]), ("scope_method", a["method_summary"])]:
        result[cid] = {"id": cid, "text": text, "evidence": ["profile"], "measurements": []}
    for i, text in enumerate(a["limitations"], 1):
        cid = f"limit_{i}"
        result[cid] = {"id": cid, "text": text, "evidence": [], "measurements": []}
    return result


def write_evidence_map(directory, dossier, release):
    """Portable evidence receipt beside a paper; it is a snapshot, not a new approval."""
    directory = Path(directory)
    atomic_json(directory / "approved_evidence.json", {"dossier": dossier, "release": release})
    lines = ["# Approved statement bindings", "", f"Writing release: `{release['id']}`", "",
             "This snapshot records the approved scope. Revalidate the live review before further writing.", "",
             "Raw sources, measurements and input hashes: [approved_evidence.json](approved_evidence.json).", ""]
    for cid, item in approved_statements(dossier).items():
        lines += [f"## {cid}", "", item["text"], "",
                  f"Evidence: {', '.join(item['evidence']) or 'scope limitation'}.", ""]
        for ref in item["measurements"]:
            m = dossier["measurements"][ref]
            lines += [f"- `{ref}`: value={m['value']}, std={m['std']}; owning evidence `{m['run_id']}`."]
    (directory / "evidence_map.md").write_text("\n".join(lines) + "\n")


STATEMENT_ID = re.compile(r"\[(claim_[A-Za-z0-9_]+|scope_question|scope_method|limit_\d+)\]")


def audit_prose(text: str, dossier: dict) -> list[str]:
    """Check sentence-local statement citations and numbers, never a global numeric bag.

    Semantic entailment is separately reviewed by FidelityReviewer. This check cannot
    prove that a paraphrase preserves the meaning or units of an approved statement.
    """
    statements = approved_statements(dossier)
    errors = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", text.strip()):
        if not sentence.strip():
            continue
        ids = STATEMENT_ID.findall(sentence)
        if not ids or set(ids) - set(statements):
            errors.append("Every sentence must cite an approved [claim_*], [scope_*] or [limit_*]")
            continue
        allowed = [t for cid in ids for t in _number_tokens(statements[cid]["text"])]
        reg = VerifiedRegistry(":sentence:")
        for i, tok in enumerate(allowed):
            reg.capture(str(i), float(tok))
        bad = reg.audit_text(STATEMENT_ID.sub("", sentence))
        if bad:
            errors.append(f"Numbers not supported by this sentence's approved statements: {bad}")
    return errors
