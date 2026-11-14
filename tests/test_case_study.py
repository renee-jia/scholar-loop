"""The portable case-study runner must preserve its human gates and shared cap."""
import json
import subprocess
import sys

import pytest

from examples import case_study_agnews as case


@pytest.fixture
def commands(tmp_path, monkeypatch):
    for key in ("OUT", "CAP", "MODEL", "TOPIC", "STEER", "DIRECTIONS", "SELECT", "SCHOLARLOOP_PAPER_ONLY"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(case.subprocess, "run", run)
    return calls


def test_case_study_waits_for_both_human_selections(tmp_path, commands, capsys):
    out = tmp_path / "study with spaces 研究"
    args = ["--out", str(out), "--cap", "7"]
    assert case.main(args) == 0
    assert len(commands) == 1 and "--stop-after" in commands[0][0]
    assert commands[0][0][-1] == "directions"
    assert "--directions" in capsys.readouterr().out
    (out / "directions.md").write_text("Directions", encoding="utf-8")
    commands.clear()
    assert case.main([*args, "--directions", "1,3"]) == 0
    assert len(commands) == 1 and commands[0][0][-1] == "ideas"
    assert "--select" in capsys.readouterr().out


def test_case_study_reuses_saved_campaign_and_one_budget(tmp_path, commands, monkeypatch):
    out = tmp_path / "study with spaces 研究"
    out.mkdir()
    for name in ("handoff.json", "writing.json", "experiments.jsonl"):
        (out / name).write_text("{}", encoding="utf-8")
    (out / "spend.json").write_text(json.dumps({"usd": 1.25}), encoding="utf-8")
    monkeypatch.setenv("CAP", "7")  # Retain compatibility with the Bash entry point.
    assert case.main(["--out", str(out)]) == 0
    assert len(commands) == 4  # Campaign, trace analysis, injection, baseline; no new ideation.
    for argv, kwargs in commands:
        assert argv[0] == sys.executable
        assert kwargs["env"]["SCHOLARLOOP_SPEND_FILE"] == str(out / "spend.json")
        assert kwargs["env"]["SCHOLARLOOP_SPEND_CAP"] == "7.0"
        assert kwargs["env"]["PYTHONUTF8"] == "1"
        assert not kwargs.get("shell")
    assert commands[0][1]["env"]["SCHOLARLOOP_PAPER_ONLY"] == "1"
    assert commands[0][1]["env"]["SCHOLARLOOP_HANDOFF"] == str(out / "handoff.json")
    assert commands[-1][0][-2:] == ["--repeats", "3"]


def test_case_study_stops_on_failed_ideation(tmp_path, monkeypatch):
    monkeypatch.setattr(case.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 2))
    with pytest.raises(SystemExit) as error:
        case.main(["--out", str(tmp_path), "--directions", "1", "--select", "1"])
    assert error.value.code == 2
