"""Tests for the deterministic search-space reasoning (mechanism 1 & 2)."""

from pathlib import Path

from scholarloop.ledger import Hypothesis, LedgerEntry
from scholarloop.profile import load_profile
from scholarloop.reasoning import (
    analyze_search_space,
    calibration_error,
    confidence_bound,
    is_duplicate_config,
    score_prediction,
)

ROOT = Path(__file__).resolve().parent.parent
PROFILE = load_profile(ROOT / "profiles" / "image-classification.yaml")  # minimize, baseline 4.9


def _entry(eid, config, score, verdict="discarded", parent=None):
    return LedgerEntry(
        id=eid, domain="image-classification",
        hypothesis=Hypothesis(claim="c", source="arXiv:1"),
        metric_name="val_top1_err", parent=parent, config=config,
        fidelity=["verify"], metric={"verify": score, "seeds": [score]}, verdict=verdict,
    )


def test_too_few_entries_yields_empty_constraints():
    c = analyze_search_space([_entry("e1", {"lr": 0.1}, 4.0)], PROFILE)
    assert c.ruled_out == [] and c.focus == [] and len(c.tried) == 1


def test_region_pruning_rules_out_losing_side():
    # winners have low lr; every loser has lr strictly above the winners -> rule out high lr
    entries = [
        _entry("w1", {"lr": 0.1, "wd": 1e-4}, 4.5, verdict="kept"),   # beats 4.9
        _entry("w2", {"lr": 0.15, "wd": 1e-4}, 4.7, verdict="kept"),
        _entry("l1", {"lr": 0.4, "wd": 1e-4}, 6.0),
        _entry("l2", {"lr": 0.5, "wd": 1e-4}, 6.3),
    ]
    c = analyze_search_space(entries, PROFILE)
    assert any(r.startswith("lr>") for r in c.ruled_out), c.ruled_out
    # wd never varies and never separates win/lose -> not ruled out
    assert not any(r.startswith("wd") for r in c.ruled_out)


def test_focus_picks_high_leverage_knob():
    # lr swings the metric a lot; wd barely moves it -> lr should be in focus, wd should not
    entries = [
        _entry("a", {"lr": 0.1, "wd": 1e-4}, 4.0, verdict="kept"),
        _entry("b", {"lr": 0.1, "wd": 5e-4}, 4.1, verdict="kept"),
        _entry("c", {"lr": 0.9, "wd": 1e-4}, 8.0),
        _entry("d", {"lr": 0.9, "wd": 5e-4}, 8.1),
    ]
    c = analyze_search_space(entries, PROFILE)
    assert "lr" in c.focus
    assert "wd" not in c.focus


def test_duplicate_config_detection():
    entries = [_entry("a", {"lr": 0.1, "depth": 18}, 4.0)]
    assert is_duplicate_config({"lr": 0.1, "depth": 18}, entries) is True
    assert is_duplicate_config({"lr": 0.2, "depth": 18}, entries) is False


def test_prediction_calibration():
    assert calibration_error(-0.4, -0.1) == 0.3
    parent = _entry("p", {"lr": 0.2}, 5.0, verdict="discarded")
    child = _entry("c", {"lr": 0.1}, 4.7, verdict="kept", parent="p")
    pred = score_prediction(child, parent, predicted_delta=-0.5)
    assert pred["measured"] == -0.3          # 4.7 - 5.0
    assert pred["calibration_error"] == 0.2  # |−0.5 − (−0.3)|


def test_confidence_bound_is_pessimistic_on_the_metric_side():
    assert confidence_bound([4.0], "minimize") == 4.0          # single value -> the mean
    assert confidence_bound([3.8, 4.0, 4.2], "minimize", z=0) == 4.0   # z=0 -> the mean
    # minimize: the bound sits ABOVE the mean (the worse side that must still clear a gate)
    assert confidence_bound([3.8, 4.0, 4.2], "minimize", z=1.0) > 4.0
    # maximize: the bound sits BELOW the mean
    assert confidence_bound([3.8, 4.0, 4.2], "maximize", z=1.0) < 4.0


def test_prediction_without_parent_is_uncalibrated():
    child = _entry("c", {"lr": 0.1}, 4.7)
    pred = score_prediction(child, None, predicted_delta=-0.5)
    assert pred["measured"] is None and pred["calibration_error"] is None
