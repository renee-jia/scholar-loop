"""The loop Governor: budget / round-cap / convergence stop conditions (pure, no LLM)."""

import pytest

from scholarloop.governor import Governor, cost_of


def test_governor_rejects_unbounded_and_degenerate_configs():
    with pytest.raises(ValueError):
        Governor()                       # no stop condition at all -> would loop forever
    with pytest.raises(ValueError):
        Governor(dry_patience=0)         # 0 would stop before any round runs


def test_cost_of_known_and_unknown_model():
    assert cost_of({"input_tokens": 1_000_000, "output_tokens": 1_000_000}, "claude-opus-4-8") == 30.0
    assert cost_of({"input_tokens": 1000, "output_tokens": 0}, "no-such-model") is None   # unmetered
    assert cost_of({}, "claude-haiku-4-5") == 0.0


def test_budget_stop_and_threshold_alerts_fire_once():
    g = Governor(max_cost=10.0)
    assert g.should_stop(4.0)[0] is False
    assert g.alerts(4.0) == []                              # below 50%
    assert any("50%" in a for a in g.alerts(5.0))           # crosses 50%
    assert g.alerts(5.5) == []                              # 50% already announced, 80% not yet
    assert any("80%" in a for a in g.alerts(8.0))           # crosses 80%
    stop, why = g.should_stop(10.0)
    assert stop and "exhausted" in why


def test_round_cap():
    g = Governor(max_rounds=2)
    assert g.should_stop(None)[0] is False
    g.record_round(False)
    assert g.should_stop(None)[0] is False
    g.record_round(False)
    stop, why = g.should_stop(None)
    assert stop and "round cap" in why


def test_dry_convergence_resets_on_improvement():
    g = Governor(dry_patience=2)
    assert g.update_frontier(5.0, "minimize") is True       # first result always improves
    g.record_round(True)
    assert g.should_stop(None)[0] is False
    assert g.update_frontier(6.0, "minimize") is False      # worse -> no improvement
    g.record_round(False)                                   # dry streak = 1
    assert g.should_stop(None)[0] is False
    assert g.update_frontier(5.5, "minimize") is False      # still not below the 5.0 frontier
    g.record_round(False)                                   # dry streak = 2 -> converged
    stop, why = g.should_stop(None)
    assert stop and "no frontier improvement" in why


def test_improvement_resets_the_dry_streak():
    g = Governor(dry_patience=2)
    g.update_frontier(5.0, "maximize"); g.record_round(True)
    g.update_frontier(4.0, "maximize"); g.record_round(False)   # worse for maximize -> dry 1
    assert g.update_frontier(6.0, "maximize") is True            # a new best resets the streak
    g.record_round(True)                                         # dry back to 0
    assert g.should_stop(None)[0] is False
