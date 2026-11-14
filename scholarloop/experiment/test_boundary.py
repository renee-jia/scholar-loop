"""Prevent adaptive search after a campaign has consumed its final test split."""

from scholarloop.core.ledger import Ledger


def require_unopened_test(ledger_path, *, continuing_protocol=None):
    entries = list(Ledger(ledger_path).read_all())
    for _, record in access_records(ledger_path):
        matching = [e for e in entries if e.id == record["run_id"]
                    and e.candidate_id == record["candidate_id"]
                    and e.protocol == record["protocol"] and e.run_spec.get("holdout")]
        if len(matching) != 1:
            raise ValueError("Unresolved final test access; inspect the interrupted run before continuing")
    for entry in entries:
        # Failed test attempts also consume access; a missing score is not permission to retry search.
        if entry.run_spec.get("holdout") or "holdout" in entry.metric:
            if continuing_protocol and entry.protocol.get("id") == continuing_protocol:
                continue
            raise ValueError("Final test access has already occurred in this campaign; "
                             "new exploration and planning are closed. Complete the frozen "
                             "confirmation protocol or review the existing evidence.")


def access_directory(ledger_path):
    from pathlib import Path
    path = Path(ledger_path).resolve()
    return path.with_name(path.name + ".test-access")


def access_records(ledger_path):
    import json
    records = []
    directory = access_directory(ledger_path)
    if directory.exists() and not directory.is_dir():
        raise ValueError("Uncertain final test access directory")
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text())
            if (set(record) != {"version", "run_id", "candidate_id", "protocol"}
                    or record["version"] != 1 or not isinstance(record["protocol"], dict)
                    or not isinstance(record["run_id"], str) or not isinstance(record["candidate_id"], str)):
                raise ValueError("invalid record")
        except (ValueError, TypeError, OSError) as exc:
            raise ValueError(f"Uncertain final test access record: {path}") from exc
        records.append((path, record))
    return records


def reserve_test_access(ledger_path, run_id, candidate_id, protocol):
    """Persist intent before execution. A crash leaves an unresolved, closed campaign."""
    import hashlib
    import json
    import os
    from scholarloop.core.locking import exclusive_lock
    directory = access_directory(ledger_path)
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / ".lock"):
        require_unopened_test(ledger_path, continuing_protocol=protocol.get("id"))
        path = directory / (hashlib.sha256(run_id.encode()).hexdigest() + ".json")
        value = {"version": 1, "run_id": run_id, "candidate_id": candidate_id, "protocol": protocol}
        with path.open("x") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        # The receipt file is flushed on every platform. Windows cannot open a
        # directory through os.open; directory-entry fsync is a POSIX facility.
        if os.name != "nt":
            for target in (directory, directory.parent):
                fd = os.open(target, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)


def validate_confirmation_run(profile, registry_dir, ledger_path, protocol, candidate,
                              fidelity, seeds, holdout, exp_id, artifact_cache=None):
    """A continuation names a frozen condition, not an arbitrary reusable token."""
    import re
    from pathlib import Path
    from scholarloop.core.candidates import make_snapshot
    from scholarloop.experiment.spec import frozen_context, validate_spec
    from scholarloop.review.contracts import digest, read_json, ReviewBlocked
    pid = protocol.get("id")
    if not isinstance(pid, str) or not re.fullmatch(r"[0-9a-f]{64}", pid):
        raise ValueError("Confirmation requires a valid frozen protocol id")
    try:
        envelope = read_json(Path(registry_dir) / "protocols" / f"{pid}.json")
        if digest(envelope) != pid:
            raise ValueError("Frozen confirmation protocol was modified")
        spec = envelope["spec"]
        errors = validate_spec(spec, profile, envelope.get("issues", []),
                               iteration=envelope.get("research_iteration"))
        if errors or spec["purpose"] != "confirmation":
            raise ValueError(f"Invalid confirmation protocol: {errors}")
        if frozen_context(profile, spec) != envelope["frozen"]:
            raise ValueError("Frozen confirmation inputs changed")
        conditions = [c for c in spec["conditions"] if c["name"] == protocol.get("condition")]
        if len(conditions) != 1:
            raise ValueError("Unknown frozen confirmation condition")
        condition = conditions[0]
        expected = make_snapshot(profile, {k["name"]: k["value"] for k in condition["config"]}, condition["edits"])
        if (candidate["id"] != expected["id"] or fidelity != spec["fidelity"]
                or seeds != spec["seeds"] or holdout != spec["holdout"]):
            raise ValueError("Execution differs from frozen confirmation condition")
        if exp_id != f"protocol_{pid[:24]}_{condition['name']}":
            raise ValueError("Confirmation run id must identify its frozen condition")
        if artifact_cache is not None:
            raise ValueError("Confirmation protocols cannot reuse exploration artifacts")
        if any(e.protocol.get("id") == pid and e.protocol.get("condition") == condition["name"]
               for e in Ledger(ledger_path).read_all()):
            raise ValueError("Confirmation condition already executed")
    except (KeyError, TypeError, ReviewBlocked, OSError) as exc:
        raise ValueError(f"Cannot validate frozen confirmation protocol: {exc}") from exc
