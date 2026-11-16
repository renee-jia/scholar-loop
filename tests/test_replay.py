"""The recorded-run replay: no model, no training; the shipped sample must stay readable."""
import json

import pytest

from scholarloop import replay

pytestmark = pytest.mark.skipif(not (replay.SAMPLES / replay.DEFAULT_SAMPLE).is_dir(),
                                reason="no recorded sample in this copy")


def test_shipped_sample_replays_traces_and_checks():
    record = replay.load()
    headings = [h for h, _ in replay.stages(record)]
    assert headings[0].startswith("1. Steer") and any("author approves" in h for h in headings)
    labelled = replay.variants(record)
    assert replay.check(record, labelled[0][1]) == []                    # the paper's own sentence
    assert all(replay.check(record, sentence) for _, sentence in labelled[1:])   # each break is refused
    out = replay.trace(record, " ".join(labelled[0][1].split()[:8]))
    assert "approved statement [claim_" in out and "measurement exp_" in out and "run exp_" in out
    assert replay.trace(record, "zebra quantum lighthouse").startswith("No sentence")
    # every sentence of the recorded paper still passes the check it was written under
    for section in record["paper"]["sections"]:
        assert replay.check(record, section["body"]) == [], section["heading"]


def test_command_line_runs_without_a_terminal(capsys):
    assert replay.main(["replay", "--no-pause"]) == 0
    assert "8. The harness at work" in capsys.readouterr().out
    assert replay.main(["check", "--examples"]) == 0
    out = capsys.readouterr().out
    assert out.count("REFUSED") == 4 and out.count("ACCEPTED") == 1
    assert replay.main(["check", "The error is 1.23 percent."]) == 0 and "REFUSED" in capsys.readouterr().out
    assert replay.main(["statements"]) == 0 and "[scope_question]" in capsys.readouterr().out
    assert replay.main(["replay", "--run", "does/not/exist"]) == 1


def test_capture_writes_a_record_a_copy_can_be_replayed_from(tmp_path):
    record = replay.load()
    (tmp_path / "copy").mkdir()
    (tmp_path / "copy" / "record.json").write_text(json.dumps(record), encoding="utf-8")
    assert replay.load(tmp_path / "copy")["paper"]["title"] == record["paper"]["title"]
    assert replay.load(tmp_path / "copy" / "record.json")["name"] == record["name"]
