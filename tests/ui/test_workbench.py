"""UI integration: CLI contracts, real offline execution, author gates and local HTTP boundary."""
import argparse
import io
import json
import os
import runpy
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

from scholarloop.core.llm import MockLLM
from scholarloop.review.contracts import atomic_json
from scholarloop.ui.catalog import FIELDS, build_command, validate
from scholarloop.ui.jobs import Jobs
from scholarloop.ui.server import Application, make_server
from scholarloop.ui.store import Store

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    obj = Application(ROOT, tmp_path / "runs")
    yield obj
    obj.jobs.close()


def project(app, **kwargs):
    return app.store.create({"title": "测试研究 · provenance", "mode": "live",
                             "profile": "image-classification.yaml", **kwargs})


def wait_job(jobs, ident):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = next(j for j in jobs.list() if j["id"] == ident)
        if job["status"] != "running":
            return job
        time.sleep(.05)
    pytest.fail("Background job did not finish")


def test_real_offline_funnel_persists_and_resumes(app):
    r = project(app, mode="demo")
    job = app.jobs.start(r["id"], "demo", {})
    done = wait_job(app.jobs, job["id"])
    assert done["status"] == "succeeded", app.jobs.log(job["id"])["text"]
    detail = app.store.detail(r["id"])
    assert [e["fidelity"] for e in detail["entries"]] == [["smoke"], ["verify"], ["full"]]
    assert detail["entries"][-1]["score"] is not None
    assert any(e["status"] == "accepted" for e in detail["activity"])
    assert "OFFLINE DEMO" in app.jobs.log(job["id"])["text"]
    assert "secret" not in json.dumps(detail)
    restarted = Store(ROOT, app.store.runs_dir)
    assert restarted.get(r["id"])["title"] == r["title"]
    second = app.jobs.start(r["id"], "demo", {})
    assert wait_job(app.jobs, second["id"])["status"] == "succeeded"
    entries = app.store.detail(r["id"])["entries"]
    assert len(entries) > 3 and entries[:3] == detail["entries"]
    with zipfile.ZipFile(io.BytesIO(app.store.export(r["id"]))) as archive:
        assert "experiments.jsonl" in archive.namelist()
        assert any(n.startswith("registry/candidates/") for n in archive.namelist())
        assert "demo.md" in archive.namelist()


@pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
def test_catalog_arguments_are_accepted_by_real_cli_parsers(app, tmp_path, monkeypatch):
    class Parsed(Exception):
        pass
    original = argparse.ArgumentParser.parse_args
    def capture(parser, *args, **kwargs):
        original(parser, *args, **kwargs)
        raise Parsed()
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    r = project(app)
    run = Path(r["path"])
    atomic_json(run / "session.json", {"stage": "new"})
    for action, fields in FIELDS.items():
        if action in {"demo", "campaign", "literature", "directions", "ideas", "refine_ideas"}:
            continue
        values = {f["name"]: f["default"] for f in fields}
        for f in fields:
            if f.get("required") and not values[f["name"]]:
                values[f["name"]] = {} if f["type"] == "json" else "test"
        if action == "suite":
            values["manifest"] = {"version": 1, "tasks": []}
        params = validate(action, values)
        folder = tmp_path / action
        folder.mkdir()
        command = build_command(action, params, r, run, ROOT, folder)
        module_index = command.index("-m") + 1
        monkeypatch.setattr(sys, "argv", command[module_index:])
        with pytest.raises(Parsed):
            runpy.run_module(command[module_index], run_name="__main__")


def test_ui_review_approval_and_writing_keep_author_and_staleness_gates(tmp_path, monkeypatch):
    from tests.test_scientific_review import setup, PASS, ARGUMENT, writing_outputs, PANEL_PASS
    from scholarloop.review.session import main as review_main
    from scholarloop.writing.session import _main as writing_main
    from scholarloop.core.ledger import Ledger
    from scholarloop.core import llm
    profile, ledger, registry, unused = setup(tmp_path)
    store = Store(tmp_path, tmp_path / "runs")
    config = store.config({"title": "Review gate", "profile": "image-classification.yaml"})
    folder = tmp_path / "requests"
    folder.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    model = MockLLM(jsons=[PASS] * 4)
    monkeypatch.setattr(llm, "AnthropicLLM", lambda **kw: model)

    def execute(action, values):
        cmd = build_command(action, validate(action, values), config, tmp_path, tmp_path, folder)
        module = cmd.index("-m") + 1
        return (writing_main if cmd[module] == "scholarloop.writing" else review_main)(cmd[module + 1:])

    assert execute("write", {}) == 2
    assert model.calls == []
    assert execute("review", {"argument": ARGUMENT}) == 0
    saved = json.loads((tmp_path / "scientific_review.json").read_text())
    assert saved["status"] == "READY_FOR_AUTHOR" and not (tmp_path / "paper.md").exists()
    assert execute("approve", {"author": "UI author", "review_hash": "wrong"}) == 2
    assert execute("approve", {"author": "UI author", "review_hash": saved["review_hash"]}) == 0
    from scholarloop.ui.catalog import writing_directory
    manuscript = writing_directory(tmp_path)
    outline, kp, bodies = writing_outputs()
    model = MockLLM(jsons=[outline, *([PANEL_PASS] * 3), kp, *([PANEL_PASS] * 3),
                           *bodies, {"verdict": "pass", "issues": []}])
    assert execute("write", {}) == 0
    assert (manuscript / "outline.md").exists()
    assert execute("write", {}) == 0
    assert execute("write", {}) == 0
    assert (manuscript / "paper.md").exists()
    entry = next(Ledger(ledger).read_all())
    entry.id = "exp_0002"
    Ledger(ledger).append(entry)
    before = len(model.calls)
    assert execute("write", {}) == 2
    assert len(model.calls) == before


def test_input_validation_and_readonly_sample(app, tmp_path):
    with pytest.raises(ValueError):
        validate("shell", {"command": "anything"})
    for value in ("nan", float("inf"), -1, 0):
        with pytest.raises(ValueError):
            project(app, spend_cap=value)
    for value in (True, 1.5, -1):
        with pytest.raises(ValueError):
            validate("campaign", {"rounds": value})
    r = project(app)
    with pytest.raises(ValueError, match="API key"):
        app.jobs.start(r["id"], "review", {})
    with pytest.raises(ValueError):
        app.store.artifact(r["id"], "../../projects.json")
    with pytest.raises(ValueError):
        app.store.update(r["id"], {"profile": "digits-mlp.yaml"})
    sample = app.store.import_run(ROOT / "examples/sample_runs/digits")
    assert sample["readonly"]
    assert len(app.store.detail(sample["id"])["entries"]) == 18
    with pytest.raises(ValueError, match="read-only"):
        app.jobs.start(sample["id"], "evaluate", {})


def test_interrupted_work_is_retained_and_never_replayed(app):
    folder = app.jobs.root / "interrupted"
    folder.mkdir()
    atomic_json(folder / "job.json", {"id": "interrupted", "run_id": "r", "status": "running", "started": 1})
    jobs = Jobs(app.store)
    assert jobs.list()[0]["status"] == "interrupted"
    assert not jobs.processes
    atomic_json(folder / "job.json", {"id": "interrupted", "run_id": "r", "status": "running", "started": 1})
    atomic_json(folder / "outcome.json", {"exit_code": 0})
    assert Jobs(app.store).list()[0]["status"] == "succeeded"


def test_stage_specific_research_feedback_and_live_campaign_adapter(app, monkeypatch, tmp_path):
    from examples.ideation_demo import ATOM, MIXUP, REVIEW_2, DIRECTIONS, IDEAS, PROPOSAL
    from scholarloop.ideation import litscout, tournament, metareview
    from scholarloop.ideation import session as research_session
    from scholarloop.core import llm
    from scholarloop.ui.worker import main
    from scholarloop.ideation.session import _main as ideation_main
    from tests.test_end_to_end import SchemaRoutedLLM

    feed = ATOM.replace("</feed>", MIXUP.split(">", 1)[1])
    arxiv = litscout.ArxivClient
    offline = lambda *a, **kw: arxiv(fetcher=lambda q, n: feed)
    monkeypatch.setattr(litscout, "ArxivClient", offline)
    monkeypatch.setattr(litscout, "OpenAlexClient", offline)
    monkeypatch.setattr(research_session, "ArxivClient", offline)
    monkeypatch.setattr(research_session, "OpenAlexClient", offline)
    real_tournament, real_meta = tournament.Tournament, metareview.MetaReviewer
    monkeypatch.setattr(tournament, "Tournament", lambda client, profile: real_tournament(
        MockLLM(router=lambda p: {"winner": "A", "rationale": "Offline comparison"}), profile))
    monkeypatch.setattr(metareview, "MetaReviewer", lambda client: real_meta(MockLLM(jsons=[{"recommendations": []}])))
    client = MockLLM(jsons=[{"queries": ["warmup"], "rationale": "test"}, REVIEW_2, DIRECTIONS, IDEAS,
                           {"queries": ["mixup"], "rationale": "test"}, REVIEW_2])
    monkeypatch.setattr(llm, "AnthropicLLM", lambda **kw: client)
    r = project(app)
    run = Path(r["path"])

    def execute(action, values):
        return main({"action": action, "params": validate(action, values), "config": r,
                     "run": str(run), "workspace": str(ROOT)})

    steer = {"interest": "Understand small training recipes", "hunches": "Schedules matter most"}
    assert execute("literature", steer) == 0
    assert execute("directions", {}) == 0
    assert execute("ideas", {"directions": "1,2"}) == 0
    cmd = build_command("select", validate("select", {"select": 1}), r, run, ROOT, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    assert ideation_main(cmd[cmd.index("-m") + 2:]) == 0
    assert (run / "handoff.json").exists()
    # The production campaign adapter uses the real funnel, budgets and adopted handoff.
    old_client, client = client, SchemaRoutedLLM()
    assert execute("campaign", {"rounds": 1}) == 0
    assert (run / "experiments.jsonl.training_budget.json").exists()
    assert app.store.detail(r["id"])["entries"]
    client = old_client
    # Explicit literature feedback must revisit literature even after idea selection.
    assert execute("literature", {**steer, "feedback": "Revisit data-side regularization"}) == 0
    session = json.loads((run / "session.json").read_text(encoding="utf-8"))
    assert session["stage"] == "reviewed" and not session["handoff"] and not session["ideas"]
    assert not (run / "handoff.json").exists()


def test_new_release_retains_old_manuscript_and_pilot_kind(app):
    from scholarloop.ui.catalog import writing_directory
    r = project(app)
    run = Path(r["path"])
    atomic_json(run / "scientific_review.json", {"release": {"id": "a" * 64}})
    first = writing_directory(run)
    first.mkdir(parents=True)
    (first / "paper.md").write_text("old evidence")
    atomic_json(run / "scientific_review.json", {"release": {"id": "b" * 64}})
    second = writing_directory(run)
    assert second != first and (first / "paper.md").read_text() == "old evidence"
    assert "paper.md" not in app.store.detail(r["id"])["writing"]["artifacts"]
    pid = "c" * 64
    atomic_json(run / "experiment_agenda" / (pid + ".json"), {"protocol_id": pid, "status": "PLANNED"})
    atomic_json(run / "registry/protocols" / (pid + ".json"), {"source_kind": "pilot"})
    assert app.store.detail(r["id"])["protocols"][0]["kind"] == "pilot"


def test_cancel_process_and_reject_overlapping_work(app, monkeypatch):
    from scholarloop.ui import jobs as module
    r = project(app, mode="demo")
    monkeypatch.setattr(module, "build_command", lambda *args: [sys.executable, "-m", "timeit", "-n", "1", "-r", "1",
                                                               "-s", "import time", "time.sleep(30)"])
    job = app.jobs.start(r["id"], "demo", {})
    with pytest.raises(ValueError, match="running"):
        app.jobs.start(r["id"], "demo", {})
    proc = app.jobs.processes[job["id"]]
    app.jobs.cancel(job["id"])
    assert wait_job(app.jobs, job["id"])["status"] == "cancelled"
    proc.wait(timeout=10)


@pytest.fixture
def http(app):
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_address[1])
    def request(path, data=None, headers=None):
        h = {"Content-Type": "application/json", "X-ScholarLoop-Token": app.token}
        h.update(headers or {})
        req = urllib.request.Request(base + path, data=None if data is None else json.dumps(data).encode(), headers=h)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), exc.headers
    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_http_origin_token_files_and_secret_boundary(app, http):
    code, body, headers = http("/")
    assert code == 200 and app.token.encode() in body and "frame-ancestors" in headers["Content-Security-Policy"]
    assert http("/api/runs", {"title": "blocked"}, {"X-ScholarLoop-Token": "bad"})[0] == 403
    assert http("/api/bootstrap", headers={"Host": "attacker.test"})[0] == 403
    assert http("/api/runs", {"title": "blocked"}, {"Origin": "https://attacker.test"})[0] == 403
    assert http("/api/connection", {"api_key": "test-secret"})[0] == 200
    assert b"test-secret" not in http("/api/bootstrap")[1]
    assert "test-secret" not in app.store.index_path.read_text() if app.store.index_path.exists() else True
    code, body, _ = http("/api/runs", {"title": "中文测试", "mode": "demo"})
    assert code == 201
    ident = json.loads(body)["id"]
    code, body, _ = http(f"/api/runs/{ident}")
    assert code == 200 and json.loads(body)["config"]["title"] == "中文测试"
    assert http(f"/api/runs/{ident}/artifact?path=..%2F..%2Fpyproject.toml")[0] == 400
    assert http("/app.js")[0] == 200
    assert http("/style.css")[0] == 200
    assert http("/api/runs", ["invalid"])[0] == 400


def test_delete_requires_confirmation_and_blocks_active_jobs(app, http):
    r = project(app)
    endpoint = f"/api/runs/{r['id']}/delete"
    for payload in ({}, {"confirmed": False}, {"confirmed": "true"}, {"confirmed": 1}):
        assert http(endpoint, payload)[0] == 400
        assert app.store.get(r["id"])["title"] == r["title"]
    assert http(endpoint, {"confirmed": True}, {"X-ScholarLoop-Token": "bad"})[0] == 403
    assert http(endpoint, {"confirmed": True}, {"Origin": "https://attacker.test"})[0] == 403
    app.jobs.records["active"] = {"id": "active", "run_id": r["id"], "status": "running", "started": 1}
    assert http(endpoint, {"confirmed": True})[0] == 400
    assert app.store.get(r["id"])
    app.jobs.records["active"]["status"] = "succeeded"
    code, body, _ = http(endpoint, {"confirmed": True})
    assert code == 200 and json.loads(body)["files_retained"] is True
    assert http(f"/api/runs/{r['id']}")[0] == 400
    assert app.store.list_runs() == []
    with pytest.raises(ValueError, match="not found"):
        app.jobs.start(r["id"], "evaluate", {})


def test_deleted_research_stays_deleted_until_explicitly_reopened(app):
    r = project(app)
    root = Path(r["path"])
    artifact = root / "session.json"
    artifact.write_text('{"topic":"Saved research"}', encoding="utf-8")
    app.store.delete(r["id"])
    assert artifact.read_text(encoding="utf-8") == '{"topic":"Saved research"}'
    assert app.store.list_runs() == []  # Auto-discovery must not undo deletion.
    restarted = Store(app.store.workspace, app.store.runs_dir)
    assert restarted.list_runs() == []
    with pytest.raises(ValueError, match="not found"):
        restarted.get(r["id"])
    restored = restarted.import_run(root)
    assert restored["id"] == r["id"]
    assert len(restarted.list_runs()) == 1
    empty = project(app)
    app.store.delete(empty["id"])
    assert app.store.import_run(empty["path"])["id"] == empty["id"]


def test_delete_archive_removes_only_its_workspace_entry(app):
    sample = ROOT / "examples/sample_runs/digits"
    before = (sample / "experiments.jsonl").read_bytes()
    r = app.store.import_run(sample, readonly=True)
    app.store.delete(r["id"])
    assert app.store.list_runs() == []
    assert (sample / "experiments.jsonl").read_bytes() == before
    assert app.store.import_run(sample)["readonly"] is True
