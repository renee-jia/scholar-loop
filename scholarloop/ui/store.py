"""Read existing research artifacts without inventing another workflow state machine."""
from __future__ import annotations

import importlib.util
import io
import json
import math
import time
import uuid
import zipfile
from pathlib import Path
from threading import RLock

from scholarloop.core.llm import model_from_env
from scholarloop.core.profile import load_profile
from scholarloop.review.contracts import atomic_json
from scholarloop.ui.catalog import writing_directory

TEXT_TYPES = {".md", ".json", ".jsonl", ".txt", ".csv", ".yaml", ".yml", ".py", ".log"}


def read_json(path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def within(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Path must stay inside the selected research directory")
    return path


class Store:
    def __init__(self, workspace, runs_dir):
        self.workspace = Path(workspace).resolve()
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.runs_dir / ".ui"
        self.state_dir.mkdir(exist_ok=True)
        self.index_path = self.state_dir / "projects.json"
        self.lock = RLock()
        self.records = read_json(self.index_path)

    def profiles(self):
        profiles = []
        for path in sorted((self.workspace / "profiles").glob("*.yaml")):
            try:
                p = load_profile(path)
                profiles.append({"id": path.name, "name": p.name, "metric": p.metric.name,
                                 "direction": p.metric.direction, "baseline": p.best_baseline(),
                                 "datasets": list(p.datasets), "knobs": p.knobs})
            except (ValueError, OSError):
                continue
        return profiles

    def config(self, data, prior=None):
        prior = prior or {}
        title = data.get("title", prior.get("title", ""))
        if not isinstance(title, str) or not title.strip() or len(title) > 250:
            raise ValueError("Enter a research title (up to 250 characters)")
        mode = data.get("mode", prior.get("mode", "live"))
        if mode not in {"live", "demo"}:
            raise ValueError("Unknown research mode")
        profile = data.get("profile", prior.get("profile", "digits-mlp.yaml"))
        if mode == "demo":
            profile = "image-classification.yaml"
        if profile not in {p["id"] for p in self.profiles()}:
            raise ValueError("Unknown domain profile")
        model = data.get("model", prior.get("model", model_from_env()))
        if not isinstance(model, str) or not model.strip() or len(model) > 150:
            raise ValueError("Enter a model ID")
        try:
            calls = int(data.get("max_calls", prior.get("max_calls", 100)))
            cap = float(data.get("spend_cap", prior.get("spend_cap", 10)))
        except (ValueError, TypeError, OverflowError):
            raise ValueError("Invalid budget") from None
        if not 1 <= calls <= 100000 or not math.isfinite(cap) or not 0 < cap <= 100000:
            raise ValueError("The budget must be positive, and the request limit must be between 1 and 100000")
        return {"title": title.strip(), "mode": mode, "profile": profile, "model": model.strip(),
                "max_calls": calls, "spend_cap": cap}

    def _save(self):
        atomic_json(self.index_path, self.records)

    def create(self, data):
        config = self.config(data)
        ident = time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:8]
        path = self.runs_dir / ident
        path.mkdir()
        record = {**config, "id": ident, "path": str(path), "created": time.time(), "readonly": False}
        with self.lock:
            self.records[ident] = record
            self._save()
        return record

    def import_run(self, path, profile=None, readonly=False, *, discovery=False):
        path = Path(path)
        if not path.is_absolute():
            path = self.workspace / path
        path = path.resolve()
        if not any(path.is_relative_to(root) for root in (self.workspace, self.runs_dir)):
            raise ValueError("Research must be inside the workspace or runs directory")
        if not path.is_dir():
            raise ValueError("No ScholarLoop research records found in this directory")
        sample = self.workspace / "examples" / "sample_runs"
        readonly = readonly or path.is_relative_to(sample)
        with self.lock:
            for record in self.records.values():
                if Path(record["path"]) == path:
                    if record.get("deleted_at") and not discovery:
                        record.pop("deleted_at")
                        self._save()
                    return dict(record)
            if not any((path / f).is_file() for f in
                       ("session.json", "experiments.jsonl", "scientific_review.json", "writing.json")):
                raise ValueError("No ScholarLoop research records found in this directory")
            session = read_json(path / "session.json")
            domain = "digits-mlp"
            ledger = path / "experiments.jsonl"
            if ledger.exists():
                with ledger.open(encoding="utf-8") as stream:
                    first = next((line for line in stream if line.strip()), "{}")
                domain = json.loads(first).get("domain", domain)
            config = self.config({"title": session.get("topic") or path.name,
                                  "profile": profile or domain + ".yaml"})
            ident = "import-" + uuid.uuid4().hex[:8]
            record = {**config, "id": ident, "path": str(path), "created": path.stat().st_mtime,
                      "readonly": readonly}
            self.records[ident] = record
            self._save()
            return record

    def get(self, ident):
        with self.lock:
            if ident not in self.records or self.records[ident].get("deleted_at"):
                raise ValueError("Research not found")
            return dict(self.records[ident])

    def update(self, ident, data):
        with self.lock:
            prior = self.get(ident)
            if prior["readonly"]:
                raise ValueError("Archived samples are read-only")
            config = self.config(data, prior)
            if config["profile"] != prior["profile"] or config["mode"] != prior["mode"]:
                raise ValueError("The domain and mode cannot be changed after creation; start a new study")
            self.records[ident] = {**prior, **config}
            self._save()
            return dict(self.records[ident])

    def delete(self, ident):
        """Remove a project from the workspace, retaining its source files for reopening."""
        with self.lock:
            prior = self.get(ident)
            self.records[ident] = {**prior, "deleted_at": time.time()}
            try:
                self._save()
            except OSError:
                self.records[ident] = prior
                raise
            return {"id": ident, "deleted": True, "files_retained": True}

    def list_runs(self):
        # Discover CLI-created campaigns in the configured root, without moving their files.
        for path in self.runs_dir.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                try:
                    self.import_run(path, discovery=True)
                except (ValueError, OSError):
                    pass
        with self.lock:
            return sorted((dict(r) for r in self.records.values() if not r.get("deleted_at")),
                          key=lambda r: r["created"], reverse=True)

    def artifacts(self, ident):
        root = Path(self.get(ident)["path"])
        files = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in TEXT_TYPES and p.resolve().is_relative_to(root):
                files.append({"path": p.relative_to(root).as_posix(), "size": p.stat().st_size})
        return sorted(files, key=lambda f: ("/" in f["path"], f["path"]))

    def artifact(self, ident, name):
        root = Path(self.get(ident)["path"])
        p = within(root / name, root)
        if not p.is_file() or p.suffix not in TEXT_TYPES:
            raise ValueError("Artifact not found")
        if p.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("This file exceeds 4 MB; export the research records to view it")
        return p.read_text(encoding="utf-8", errors="replace")

    def export(self, ident):
        root = Path(self.get(ident)["path"])
        files = self.artifacts(ident)
        if sum(f["size"] for f in files) > 100 * 1024 * 1024:
            raise ValueError("Research records exceed 100 MB; copy them from the local research directory")
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            for f in files:
                archive.write(within(root / f["path"], root), f["path"])
        return out.getvalue()

    def detail(self, ident):
        record = self.get(ident)
        root = Path(record["path"])
        warnings = []

        def read(name, default=None):
            try:
                return read_json(root / name, default)
            except (ValueError, OSError) as exc:
                warnings.append(f"{name}: {exc}")
                return {} if default is None else default

        entries = []
        ledger = root / "experiments.jsonl"
        if ledger.exists():
            with ledger.open(encoding="utf-8", errors="replace") as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        e = json.loads(line)
                        score = next((e.get("metric", {}).get(f) for f in ("full", "verify", "smoke")
                                      if e.get("metric", {}).get(f) is not None), None)
                        entries.append({k: e.get(k) for k in ("id", "metric_name", "fidelity", "verdict", "config",
                                       "hypothesis", "prediction", "purpose", "candidate_id")}
                                       | {"score": score, "role": e.get("reasoning", {}).get("idea_role"),
                                          "seeds": e.get("metric", {}).get("seeds", [])})
                    except (ValueError, AttributeError, TypeError) as exc:
                        warnings.append(f"Cannot read experiment record on line {number}: {exc}")
        session = read("session.json")
        review = read("scientific_review.json")
        try:
            writing_root = writing_directory(root)
        except ValueError as exc:
            warnings.append(str(exc))
            writing_root = root
        writing = read((writing_root / "writing.json").relative_to(root))
        writing_artifacts = {name: (writing_root / name).relative_to(root).as_posix()
                             for name in ("writing.json", "outline.md", "key_points.md", "paper.md", "review.md",
                                          "evidence_map.md", "approved_evidence.json") if (writing_root / name).exists()}
        protocols = []
        for folder in ("experiment_agenda",):
            for file in (root / folder).glob("*.json"):
                state = read(file.relative_to(root))
                if state.get("protocol_id"):
                    envelope = read("registry/protocols/" + state["protocol_id"] + ".json")
                    protocols.append({"id": state["protocol_id"], "status": state.get("status"),
                                      "kind": "pilot" if envelope.get("source_kind") == "pilot" else "agenda",
                                      "path": file.with_suffix(".md").relative_to(root).as_posix()})
        release_error = ""
        if review.get("status") in {"READY_FOR_AUTHOR", "READY_TO_WRITE"}:
            try:
                from scholarloop.review.session import ScientificReviewSession
                reviewer = ScientificReviewSession(None, load_profile(self.workspace / "profiles" / record["profile"]),
                             root, ledger_path=ledger, registry_dir=root / "registry",
                             research_dir=root if session else None)
                reviewer._check_review(review)
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                release_error = str(exc)
        dossier = review.get("dossier") or {}
        activity = []
        trace_paths = list(root.glob("*.agents.jsonl"))
        if writing_root != root:
            trace_paths += list(writing_root.glob("*.agents.jsonl"))
        for path in trace_paths:
            # Read a bounded tail, not the full prompts/outputs of a long-running study.
            with path.open("rb") as stream:
                size = path.stat().st_size
                stream.seek(max(0, size - 128 * 1024))
                if size > 128 * 1024:
                    stream.readline()
                for line in stream:
                    try:
                        event = json.loads(line)
                        if (isinstance(event, dict) and event.get("event") in {"started", "finished"}
                                and isinstance(event.get("timestamp"), (int, float))):
                            activity.append({k: event.get(k) for k in ("agent", "event", "status", "attempt", "timestamp")})
                    except (ValueError, TypeError, UnicodeError):
                        continue  # A running writer may have an incomplete last line.
        activity.sort(key=lambda e: e.get("timestamp") or 0, reverse=True)
        return {"config": record, "session": {k: session.get(k) for k in (
                "topic", "stage", "steer", "direction_map", "chosen_directions", "direction_note", "ideas", "selected")},
                "review": {"status": review.get("status"), "hash": review.get("review_hash"),
                           "issues": review.get("issues", []), "reports": review.get("reports", []),
                           "argument": dossier.get("argument"), "release_error": release_error},
                "writing": {"stage": writing.get("stage", "new"), "demo": writing.get("demo", False),
                            "artifacts": writing_artifacts, "directory": str(writing_root)},
                "entries": entries, "protocols": protocols, "artifacts": self.artifacts(ident),
                "spend": read("ui_spend.json"), "warnings": warnings,
                "iteration": read("research_iterations.json"), "handoff": read("handoff.json"), "activity": activity[:12]}

    def environment(self):
        return {"python": __import__("sys").version.split()[0], "workspace": str(self.workspace),
                "runs_dir": str(self.runs_dir), "default_model": model_from_env(),
                "dependencies": {n: importlib.util.find_spec(n) is not None
                                 for n in ("anthropic", "torch", "sklearn", "datasets")}}
