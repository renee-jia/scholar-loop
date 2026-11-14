"""Startup failures must distinguish a running workspace from an unavailable port."""
import errno
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scholarloop.core.locking import exclusive_lock
from scholarloop.ui import __main__ as cli
from scholarloop.ui import server

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def args(tmp_path, monkeypatch):
    # Exercise the server branch without restarting this test interpreter.
    class Utf8Sys:
        flags = SimpleNamespace(utf8_mode=1)
        def __getattr__(self, name):
            return getattr(sys, name)
    monkeypatch.setattr(cli, "sys", Utf8Sys())
    return ["--workspace", str(ROOT), "--runs-dir", str(tmp_path / "runs"), "--no-browser"]


def test_competing_lock_is_reported_as_already_running(args, tmp_path, capsys, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(server, "Application", lambda *a: pytest.fail("Must not initialize jobs behind another UI"))
    with exclusive_lock(runs / ".ui-server.lock"):
        assert cli.main(args) == 1
    error = capsys.readouterr().err
    assert "already running" in error and str(runs) in error
    assert "Ctrl+C" in error and "Permission denied" not in error


def test_port_failure_releases_directory_lock(args, tmp_path, capsys, monkeypatch):
    def occupied(*a):
        raise OSError(errno.EADDRINUSE, "Address already in use")
    monkeypatch.setattr(server, "make_server", occupied)
    assert cli.main(args) == 1
    error = capsys.readouterr().err
    assert "Cannot listen on 127.0.0.1:8765" in error and "--port 0" in error
    with exclusive_lock(tmp_path / "runs/.ui-server.lock"):
        pass


def test_file_access_failure_is_not_reported_as_running(args, capsys, monkeypatch):
    def denied(*a):
        raise PermissionError(errno.EACCES, "Access denied", "projects.json")
    monkeypatch.setattr(server, "Application", denied)
    assert cli.main(args) == 1
    error = capsys.readouterr().err
    assert "Cannot access the local UI files" in error
    assert "already running" not in error


def test_utf8_launcher_waits_for_child_cleanup_after_ctrl_c(monkeypatch):
    calls = []
    class Child:
        waits = 0
        def wait(self):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return 7
    child = Child()
    def launch(argv):
        calls.append(argv)
        return child
    monkeypatch.setattr(cli.subprocess, "Popen", launch)
    assert cli.run_utf8(["--no-browser"]) == 7
    assert child.waits == 2
    assert calls == [[sys.executable, "-X", "utf8", "-m", "scholarloop.ui", "--no-browser"]]
