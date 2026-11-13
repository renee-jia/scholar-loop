"""Strict snapshots of all runs, including failures; typed numeric references.

Unlike the search ledger reader this fails on corrupt/missing evidence. Current engine
code is explicitly distinguished from historical execution provenance.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, stdev

from jsonschema import Draft7Validator

from scholarloop.core.ledger import LedgerEntry
from scholarloop.core.candidates import CandidateStore, CandidateError, validate_snapshot, content_id
from scholarloop.core.registry import VerifiedRegistry, _number_tokens
from scholarloop.review.contracts import (ARGUMENT_SCHEMA, STATEMENT_ID_PATTERN, ReviewBlocked, VERSION,
                                           digest, atomic_json)


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
    # Include referenced candidates only; orphan proposals are not experimental evidence.
    if ledger_path.is_file():
        for line in ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            row = _json(line)
            if not isinstance(row, dict):
                raise ReviewBlocked("Invalid ledger record")
            cid = row.get("candidate_id")
            if cid:
                try:
                    CandidateStore(registry_dir).load(cid)
                except CandidateError as exc:
                    raise ReviewBlocked(str(exc)) from exc
                paths[f"candidates/{cid}.json"] = registry_dir / "candidates" / f"{cid}.json"
    from scholarloop.experiment.test_boundary import access_records
    try:
        for path, record in access_records(ledger_path):
            paths[f"test_access/{path.name}"] = path
    except ValueError as exc:
        raise ReviewBlocked(str(exc)) from exc
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
    candidates = {}
    for name, file in files.items():
        if name.startswith("candidates/"):
            try:
                candidate = validate_snapshot(_json(file["text"]))
            except (ValueError, KeyError, TypeError) as exc:
                raise ReviewBlocked(f"Invalid candidate snapshot: {name}: {exc}") from exc
            if name != f"candidates/{candidate['id']}.json":
                raise ReviewBlocked("Candidate snapshot filename disagrees with its identity")
            candidates[candidate["id"]] = candidate
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
        if e.candidate_id:
            candidate = candidates.get(e.candidate_id)
            if candidate is None:
                raise ReviewBlocked(f"{e.id}: candidate snapshot is missing")
            spec = e.run_spec
            if (not spec or spec.get("id") != content_id({k: v for k, v in spec.items() if k != "id"})
                    or spec.get("candidate_id") != e.candidate_id
                    or spec.get("implementation_id") != content_id(candidate["sources"])
                    or [spec.get("fidelity")] != e.fidelity or spec.get("purpose") != e.purpose
                    or spec.get("protocol") != e.protocol
                    or spec.get("seeds") != e.metric.get("requested_seeds")
                    or candidate["context"].get("domain") != e.domain
                    or candidate["context"].get("metric") != e.metric_name):
                checks.append(f"{e.id}: execution disagrees with its candidate/run specification")
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
                                 "seeds": m.get("seeds", []), "std": m.get("std"),
                                 **_seed_summary(m.get("seeds", []))}
        if e.primary_score() is not None:
            tier = next(f for f in ("full", "verify", "smoke") if f in e.metric)
            m = measurements.get(f"{e.id}::{e.metric_name}.{tier}")
            if not m or m["value"] != e.primary_score() or m["seeds"] != e.metric.get("seeds", []):
                checks.append(f"{e.id}: ledger and registry measurement disagree")
            elif m["seeds"] and abs(m["value"] - round(mean(m["seeds"]), 4)) > 1e-8:
                checks.append(f"{e.id}: aggregate disagrees with its recorded seed measurements")
        runs.append(raw)
    for name, file in files.items():
        if name.startswith("test_access/"):
            record = _json(file["text"])
            matching = [r for r in runs if r["id"] == record["run_id"]
                        and r.get("candidate_id") == record["candidate_id"]
                        and r.get("protocol") == record["protocol"] and r.get("run_spec", {}).get("holdout")]
            if len(matching) != 1:
                checks.append("Unresolved final test access: " + record["run_id"])
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
                        "value": c["mean_difference"], "seeds": c["deltas"], "std": None,
                        **_seed_summary(c["deltas"])}
    for r in runs:
        pid = r.get("protocol", {}).get("id")
        if pid and f"protocol:{pid}" not in protocols:
            checks.append(f"{r['id']}: frozen experiment protocol is missing")
    return {"schema_version": VERSION, "input_hash": digest(inputs), "profile": profile,
            "input_manifest": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in files.items()},
            "runs": runs, "measurements": measurements, "papers": papers, "research": research,
            "protocols": protocols, "candidates": candidates,
            "current_code": {k: files[k]["text"] for k in ("train", "scorer")},
            "checks": checks,
            "provenance_note": "Current engine source is not proof of historical execution. "
            "Legacy records may lack code/data/environment snapshots. Review the scope accordingly."}


class _Sources:
    """Each distinct file text once. A later version of a path is written as a unified diff
    against the closest earlier one when that is clearly shorter than the text itself."""

    def __init__(self):
        self.table, self._seen = {}, {}

    def ref(self, path: str, text: str) -> str:
        versions = self._seen.setdefault(path, [])
        for name, known in versions:
            if known == text:
                return name
        name = f"{path}@v{len(versions) + 1}"
        entry = text
        diffs = [(base, "".join(difflib.unified_diff(known.splitlines(True), text.splitlines(True),
                                                     fromfile=base, tofile=name, n=2)))
                 for base, known in versions]
        if diffs:
            base, diff = min(diffs, key=lambda item: len(item[1]))
            if len(diff) < 0.6 * len(text):
                entry = {"unified_diff_against": base, "diff": diff}
        versions.append((name, text))
        self.table[name] = entry
        return name


def prompt_view(dossier: dict) -> dict:
    """The dossier as a reviewer reads it: the same evidence with every repeated block stated
    once. Candidate snapshots, protocol envelopes and the current engine repeat whole source
    files; a protocol repeats the idea it came from; the literature review and the novelty
    check repeat paper records. Nothing is summarised or dropped except file hashes, and every
    id keeps its meaning. Validation and hashing always use the full dossier, never this view."""
    view = json.loads(json.dumps(dossier))
    profile, sources = view["profile"], _Sources()
    paths = {"train": profile.get("train_entrypoint"), "scorer": profile.get("fixed_module")}

    def refs(files):
        return {path: sources.ref(path, text) for path, text in files.items()}

    # Runs first, in ledger order, so `@v1` of a path is what the earliest run executed.
    ordered = [r.get("candidate_id") for r in view["runs"] if r.get("candidate_id")]
    for cid in [*dict.fromkeys(ordered), *view.get("candidates", {})]:
        candidate = view.get("candidates", {}).get(cid)
        if candidate and isinstance(candidate.get("sources"), dict) and "source_versions" not in candidate:
            candidate["source_versions"] = refs(candidate.pop("sources"))
    idea = view.get("research", {}).get("handoff.json", {}).get("idea")
    for protocol in view.get("protocols", {}).values():
        envelope = protocol["specification"]
        frozen = envelope.get("frozen")
        if isinstance(frozen, dict) and isinstance(frozen.get("sources"), dict):
            frozen["source_versions"] = refs(frozen.pop("sources"))
            if frozen.get("profile") == dossier["profile"]:
                frozen["profile"] = "identical to `profile`"
        if idea is not None and envelope.get("idea") == idea:
            envelope["idea"] = "identical to research['handoff.json'].idea"
        for condition in envelope.get("spec", {}).get("conditions", []):
            for edit in condition.get("edits", []):
                if isinstance(edit.get("content"), str) and edit.get("path"):
                    edit["content_version"] = sources.ref(edit["path"], edit.pop("content"))
    view["current_code"] = {name: sources.ref(paths[name] or name, text)
                            for name, text in view["current_code"].items()}
    view["source_files"] = sources.table
    known = set(view["papers"])

    def paper_ids(holder):
        if isinstance(holder, dict) and isinstance(holder.get("papers"), list):
            holder["papers"] = [p["arxiv_id"] if p.get("arxiv_id") in known and
                                p == dossier["papers"][p["arxiv_id"]] else p for p in holder["papers"]]

    research = view.get("research", {})
    paper_ids(research.get("lit_review.json"))
    if isinstance(idea, dict):
        paper_ids(idea.get("novelty_review"))
    view["input_manifest"] = sorted(view["input_manifest"])
    view["citable_ids"] = {"evidence": sorted(evidence_ids(dossier)),
                           "measurements": sorted(dossier["measurements"])}
    view["reading_note"] = (
        "File texts are in `source_files`, each distinct version once; a version given as a diff "
        "applies to the named earlier version. `source_versions`, `content_version` and "
        "`current_code` point into that table. A paper listed by id alone is in `papers`. "
        "`citable_ids` lists every id an evidence or measurement reference may use.")
    return view


def evidence_ids(dossier):
    return ({r["id"] for r in dossier["runs"]} | set(dossier["papers"]) |
            set(dossier.get("protocols", {})) | set(dossier.get("candidates", {})) |
            {"profile", "train", "scorer"})


def _seed_summary(seeds) -> dict:
    """The median and sample standard deviation of the per-seed values, computed here so that a
    claim can state them and a reviewer can ask for them: a number that no record holds cannot be
    written, and reviewers kept asking for these two."""
    if not isinstance(seeds, list) or len(seeds) < 2:
        return {}
    return {"median": round(median(seeds), 6), "seed_sd": round(stdev(seeds), 6)}


def explain_unknown(ids, known, dossier) -> str:
    """Say what is wrong with each id that is not citable evidence, and what to write instead.
    'unknown evidence' alone sent models round the same mistake three times."""
    notes = []
    for eid in sorted(set(ids) - set(known)):
        m = dossier["measurements"].get(eid)
        if m:
            notes.append(f"{eid!r} is a measurement reference, not evidence: cite its run {m['run_id']!r}")
            continue
        close = difflib.get_close_matches(eid, sorted(known), n=1, cutoff=0.6)
        notes.append(f"{eid!r} is not an id in the dossier" + (f"; did you mean {close[0]!r}?" if close else
                     "; use a run id, a paper id, a protocol or candidate id, profile, train or scorer"))
    return "; ".join(notes)


def _where_recorded(tokens, claim, dossier) -> str:
    """Tell the author of a rejected claim what to do about each number: bind the measurement
    that records it, or remove it."""
    hints = []
    for tok in dict.fromkeys(tokens):
        owners = [ref for ref, m in dossier["measurements"].items() if ref not in claim["measurements"]
                  and any(isinstance(v, (int, float)) and abs(v - float(tok)) < 1e-9
                          for v in (m["value"], m["std"], *m["seeds"]))]
        hints.append(f"{tok} is recorded by {owners[:3]}: add it to `measurements` and cite its run"
                     if owners else f"{tok} is in no measurement: remove it or use words")
    return " (" + "; ".join(hints) + ")"


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
            errors.append(f"{c['id']}: unknown evidence: " + explain_unknown(c["evidence"], known, dossier))
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
                for stat in ("median", "seed_sd"):
                    if m.get(stat) is not None:
                        reg.capture(f"{ref}.{stat}", abs(m[stat]))
                        reg.capture(f"{ref}.{stat}.signed", m[stat])
                # Per-seed values are recorded measurements too: a claim that binds the
                # aggregate may state the seeds it was computed from.
                for i, seed_value in enumerate(m["seeds"]):
                    reg.capture(f"{ref}.seed{i}", seed_value)
                # "lower by 4.96" states a recorded difference of -4.96: the size is the recorded
                # number and the direction is in the words, which reviewers read.
                for i, v in enumerate([m["value"], *m["seeds"]]):
                    if v < 0:
                        reg.capture(f"{ref}.magnitude{i}", -v)
        # What a cited run's training script reported about itself (switch steps, batch counts)
        # is recorded in the ledger. A claim may quote it; whether it says so is for reviewers.
        for run in dossier["runs"]:
            if run["id"] in c["evidence"]:
                for name, values in (run.get("metric", {}).get("self_reported") or {}).items():
                    for i, v in enumerate(values if isinstance(values, list) else [values]):
                        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                            reg.capture(f"{run['id']}.self_reported.{name}.{i}", v)
        # No global small-integer/config/year exceptions for scientific claims.
        bad = reg.audit_text(c["text"])
        if bad:
            errors.append(f"{c['id']}: numbers lack explicitly bound measurements: {bad}"
                          + _where_recorded(bad, c, dossier))
    errors += _validate_support(argument, dossier, ids)
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


_NUMERAL = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _recorded_numbers(dossier: dict, eid: str) -> set[float]:
    """Every numeral written in one piece of evidence: the profile, a source file, a protocol,
    a candidate snapshot, a run record or a paper record."""
    if eid == "profile":
        source = dossier["profile"]
    elif eid in ("train", "scorer"):
        source = dossier["current_code"][eid]
    else:
        source = (dossier.get("protocols", {}).get(eid) or dossier.get("candidates", {}).get(eid)
                  or dossier["papers"].get(eid)
                  or next((r for r in dossier["runs"] if r["id"] == eid), None))
    text = source if isinstance(source, str) else json.dumps(source, ensure_ascii=False, default=str)
    return {float(tok) for tok in _NUMERAL.findall(text.replace("\\n", " "))}


def _validate_support(argument: dict, dossier: dict, claim_ids: list[str]) -> list[str]:
    """Setup and related-work statements. A setup statement cites the records that state it, and
    every numeral in it must be written in one of them: a setting is copied, never computed.
    A related-work statement cites reviewed papers and may only use numerals from their records.
    Whether the sentence is a fair reading of its source is the reviewers' question."""
    errors, seen = [], set(claim_ids)
    known = evidence_ids(dossier)
    for kind, key, sources in (("setup", "evidence", known), ("related_work", "papers", set(dossier["papers"]))):
        for item in argument.get(kind, []):
            if item["id"] in seen:
                errors.append(f"{item['id']}: statement ids must be unique")
            seen.add(item["id"])
            unknown = sorted(set(item[key]) - sources)
            if unknown:
                errors.append(f"{item['id']}: unknown {key} {unknown}")
                continue
            recorded = set().union(*(_recorded_numbers(dossier, eid) for eid in item[key]))
            bad = [tok for tok in _number_tokens(item["text"]) if float(tok) not in recorded]
            if bad:
                errors.append(f"{item['id']}: numbers {bad} are not written in {item[key]}; "
                              "cite the record that states them or use words")
    return errors


def paper_readiness(argument: dict, dossier: dict) -> list[str]:
    """What a paper could not be written without. Writing may use approved statements only, so
    an argument of conclusions alone passes review and then stalls two stages into writing. This
    is checked when the argument is built, before any reviewer is called, and again before
    writing starts. It asks whether the material exists, not whether it is right."""
    gaps = []
    if not argument.get("setup"):
        gaps.append("no setup statements: state the settings a reader needs to rerun the study, "
                    "each citing the record that states it")
    if dossier["papers"] and not argument.get("related_work"):
        gaps.append("no related_work statements although papers were reviewed: say how the study "
                    "stands to the papers it builds on or differs from")
    bound = {dossier["measurements"][ref]["run_id"] for c in argument["claims"]
             for ref in c["measurements"] if ref in dossier["measurements"]}
    for ref, protocol in dossier.get("protocols", {}).items():
        if not protocol.get("analysis"):
            continue
        pid = ref.removeprefix("protocol:")
        missing = sorted(r["protocol"]["condition"] for r in dossier["runs"]
                         if r.get("protocol", {}).get("id") == pid and r["id"] not in bound
                         and any(m["run_id"] == r["id"] for m in dossier["measurements"].values()))
        if missing:
            gaps.append(f"arms {missing} of {ref} ran but no claim reports their recorded value: "
                        "report every arm in an observation claim, even one whose reading is withdrawn")
    return gaps


def approved_statements(dossier: dict) -> dict:
    a = dossier["argument"]
    result = {c["id"]: c for c in a["claims"]}
    for item in a.get("setup", []):
        result[item["id"]] = {**item, "measurements": []}
    for item in a.get("related_work", []):
        result[item["id"]] = {"id": item["id"], "text": item["text"], "evidence": item["papers"],
                              "measurements": []}
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


STATEMENT_ID = re.compile(r"\[(" + STATEMENT_ID_PATTERN + r")\]")


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
            errors.append("Every sentence must cite an approved [claim_*], [setup_*], [related_*], "
                          "[scope_*] or [limit_*]")
            continue
        allowed = [t for cid in ids for t in _number_tokens(statements[cid]["text"])]
        reg = VerifiedRegistry(":sentence:")
        for i, tok in enumerate(allowed):
            reg.capture(str(i), float(tok))
        bad = reg.audit_text(STATEMENT_ID.sub("", sentence))
        if bad:
            errors.append(f"Numbers not supported by this sentence's approved statements: {bad}")
    return errors
