"""Run the actual Python entry point and offline workers on each supported OS."""
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from scholarloop.core.locking import exclusive_lock

ROOT = Path(__file__).resolve().parents[2]


def test_real_launch_unicode_paths_worker_and_duplicate_start(tmp_path):
    runs = tmp_path / "research workspace 研究"
    output = tmp_path / "server.log"
    argv = [sys.executable, "-X", "utf8=0", "-m", "scholarloop.ui", "--no-browser",
            "--workspace", str(ROOT), "--runs-dir", str(runs), "--port", "0"]
    env = {**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": "utf-8", "ANTHROPIC_API_KEY": ""}
    with output.open("wb") as log:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt",
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            # A cold interpreter on a hosted macOS runner can take longer than
            # local startup; still fail if it exits or never becomes ready.
            deadline = time.monotonic() + 90
            while True:
                text = output.read_text(encoding="utf-8")
                match = re.search(r"ScholarLoop workspace: (http://127\.0\.0\.1:\d+)", text)
                if match:
                    break
                assert process.poll() is None, text
                assert time.monotonic() < deadline, text
                time.sleep(.05)
            assert process.poll() is None, "The launcher must stay in the foreground"
            url = match[1]

            def request(path, data=None, token=""):
                req = urllib.request.Request(url + path,
                    data=None if data is None else json.dumps(data).encode("utf-8"),
                    headers={"Content-Type": "application/json", "X-ScholarLoop-Token": token})
                with urllib.request.urlopen(req, timeout=5) as response:
                    return response.read().decode("utf-8"), response.headers

            html, _ = request("/")
            token = re.search(r'name="scholarloop-token" content="([^"]+)"', html)[1]
            assert '<html lang="en">' in html
            for path, content_type in (("/app.js", "javascript"), ("/style.css", "text/css"),
                                       ("/favicon.svg", "image/svg+xml")):
                body, headers = request(path)
                assert body and content_type in headers["Content-Type"]
            boot = json.loads(request("/api/bootstrap")[0])
            assert Path(boot["environment"]["runs_dir"]) == runs.resolve()
            project = json.loads(request("/api/runs", {"title": "研究 · café", "mode": "demo"}, token)[0])
            request(f"/api/runs/{project['id']}/actions", {"action": "demo", "params": {}}, token)
            deadline = time.monotonic() + 30
            while True:
                detail = json.loads(request(f"/api/runs/{project['id']}")[0])
                if detail["jobs"][0]["status"] != "running":
                    break
                assert time.monotonic() < deadline, detail["jobs"]
                time.sleep(.05)
            assert detail["jobs"][0]["status"] == "succeeded", detail["jobs"]
            assert detail["config"]["title"] == "研究 · café"
            assert len(detail["entries"]) == 3
            # A second launcher must neither detach nor alter the first server's state.
            duplicate = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=15)
            assert duplicate.returncode == 1
            assert "already running" in duplicate.stderr.decode("utf-8")
            assert process.poll() is None
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=5)
    with exclusive_lock(runs / ".ui-server.lock"):
        pass
