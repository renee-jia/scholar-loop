"""Background processes with durable logs, explicit cancellation and no automatic replay."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from scholarloop.review.contracts import atomic_json
from scholarloop.ui.catalog import LABELS, build_command, needs_model, validate
from scholarloop.ui.store import read_json


class Jobs:
    def __init__(self, store):
        self.store = store
        self.root = store.state_dir / "jobs"
        self.root.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.processes = {}
        self.records = {}
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.closing = False
        for path in self.root.glob("*/job.json"):
            record = read_json(path)
            if record.get("status") == "running":
                outcome = read_json(path.parent / "outcome.json")
                record["status"] = self.status(outcome["exit_code"]) if "exit_code" in outcome else "interrupted"
                record["exit_code"] = outcome.get("exit_code")
                atomic_json(path, record)
            if record.get("action") in LABELS:
                record["label"] = LABELS[record["action"]][0]
            self.records[record["id"]] = record

    @staticmethod
    def status(code):
        return "succeeded" if code == 0 else "attention" if code == 2 else "failed"

    def list(self, run_id=None):
        with self.lock:
            return sorted((dict(r) for r in self.records.values() if run_id is None or r["run_id"] == run_id),
                          key=lambda r: r["started"], reverse=True)

    def active(self, run_id):
        return any(r["status"] == "running" for r in self.list(run_id))

    def start(self, run_id, action, params):
        params = validate(action, params)
        with self.lock:
            if self.closing:
                raise ValueError("Server is shutting down")
            config = self.store.get(run_id)
            if config["readonly"]:
                raise ValueError("Archived samples are read-only; create a new study to run actions")
            if self.active(run_id):
                raise ValueError("A task is already running for this study; wait for it to finish or stop it first")
            if config["mode"] == "demo" and action not in {"demo", "evaluate", "traces", "decision_bench"}:
                raise ValueError("Offline demos only support simulated experiments and diagnostics; create a live study for research")
            if needs_model(action, params) and not self.api_key:
                raise ValueError("Enter an Anthropic API key in Connections, or set ANTHROPIC_API_KEY before starting the server")
            if needs_model(action, params) and not self.store.environment()["dependencies"]["anthropic"]:
                raise ValueError('Install the model dependencies first: pip install -e ".[llm]"')
            ident = uuid.uuid4().hex
            folder = self.root / ident
            folder.mkdir()
            run = Path(config["path"])
            argv = build_command(action, params, config, run, self.store.workspace, folder)
            atomic_json(folder / "command.json", {"argv": argv, "run": str(run)})
            env = dict(os.environ)
            env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                       PYTHONPATH=str(self.store.workspace) + os.pathsep + env.get("PYTHONPATH", ""),
                       ANTHROPIC_API_KEY=self.api_key, SCHOLARLOOP_MODEL=config["model"],
                       SCHOLARLOOP_SPEND_CAP=str(config["spend_cap"]),
                       SCHOLARLOOP_SPEND_FILE=str(run / "ui_spend.json"))
            record = {"id": ident, "run_id": run_id, "action": action, "label": LABELS[action][0],
                      "status": "running", "started": time.time(), "finished": None, "exit_code": None}
            atomic_json(folder / "job.json", record)
            self.records[ident] = record
            try:
                with (folder / "output.log").open("wb") as output:
                    process = subprocess.Popen([sys.executable, "-u", "-m", "scholarloop.ui.process_runner",
                                                str(folder / "command.json")], cwd=self.store.workspace,
                                               env=env, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                               start_new_session=os.name != "nt",
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                self.processes[ident] = process
                threading.Thread(target=self._watch, args=(ident, process), daemon=True).start()
            except OSError as exc:
                record.update(status="failed", finished=time.time(), error=str(exc))
                atomic_json(folder / "job.json", record)
                raise ValueError(f"Cannot start task: {exc}") from exc
            return dict(record)

    def _watch(self, ident, process):
        code = process.wait()
        with self.lock:
            record = self.records[ident]
            if record["status"] != "cancelled":
                record["status"] = self.status(code)
            record.update(exit_code=code, finished=time.time())
            atomic_json(self.root / ident / "job.json", record)
            self.processes.pop(ident, None)

    def log(self, ident, offset=0):
        if ident not in self.records or offset < 0:
            raise ValueError("Job not found")
        path = self.root / ident / "output.log"
        if not path.exists():
            return {"text": "", "offset": 0}
        with path.open("rb") as stream:
            stream.seek(min(offset, path.stat().st_size))
            data = stream.read(128 * 1024)
            # Avoid splitting a UTF-8 character at the chunk boundary.
            while data and len(data) >= 128 * 1024:
                try:
                    data.decode("utf-8")
                    break
                except UnicodeDecodeError as exc:
                    if exc.end == len(data):
                        extra = stream.read(1)
                        if extra:
                            data += extra
                            continue
                    break
            text = data.decode("utf-8", errors="replace")
            if self.api_key:
                text = text.replace(self.api_key, "[redacted]")
            return {"text": text, "offset": stream.tell()}

    def cancel(self, ident):
        with self.lock:
            if ident not in self.records:
                raise ValueError("Job not found")
            process = self.processes.get(ident)
            if process is None or process.poll() is not None:
                return dict(self.records[ident])
            if os.name == "nt":
                result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                                        creationflags=subprocess.CREATE_NO_WINDOW, timeout=10)
                if result.returncode and process.poll() is None:
                    raise ValueError("Cannot stop the task process; it is still running: " + result.stderr.decode(errors="replace"))
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.records[ident]["status"] = "cancelled"
            atomic_json(self.root / ident / "job.json", self.records[ident])
            return dict(self.records[ident])

    def close(self):
        with self.lock:
            self.closing = True
            ids = list(self.processes)
        for ident in ids:
            self.cancel(ident)
