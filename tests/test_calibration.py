"""Universal predict-then-verify: delta/binary scoring + the per-agent CalibrationLog (pure)."""

from scholarloop.calibration import CalibrationLog, score_binary, score_delta


def test_score_delta_rewards_correct_direction():
    hit = score_delta("reasoner", -2.0, -1.5)            # predicted improvement, got improvement
    assert hit.hit is True and hit.error == 0.5
    miss = score_delta("reasoner", -2.0, 1.0)            # predicted improvement, got a regression
    assert miss.hit is False and miss.error == 3.0
    assert score_delta("reasoner", None, -1.0) is None   # not checkable -> no verdict
    assert score_delta("reasoner", -1.0, None) is None


def test_score_binary_go_no_go():
    assert score_binary("debate", True, True).hit is True
    bad = score_binary("debate", True, False)
    assert bad.hit is False and bad.error == 1.0


def test_calibration_log_summarizes_per_agent_and_renders():
    log = CalibrationLog()
    assert log.render() == ""                            # nothing scored yet
    log.record(score_delta("reasoner", -2.0, -1.0))      # hit (dir), err 1.0
    log.record(score_delta("reasoner", -1.0, 2.0))       # miss, err 3.0
    log.record(score_binary("debate", True, True))       # hit
    log.record(score_binary("debate", True, False))      # miss
    log.record(None)                                     # ignored

    stats = log.by_agent()
    assert stats["reasoner"]["n"] == 2 and stats["reasoner"]["hit_rate"] == 0.5
    assert stats["reasoner"]["mean_error"] == 2.0        # (1.0 + 3.0) / 2
    assert stats["debate"]["n"] == 2 and stats["debate"]["hit_rate"] == 0.5

    block = log.render()
    assert "reasoner" in block and "debate" in block and "50%" in block
