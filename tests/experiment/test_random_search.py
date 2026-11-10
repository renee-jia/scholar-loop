"""Random-search control arm and the matched-budget comparison around it."""

import random
from pathlib import Path

import pytest

from scholarloop.core.ledger import Ledger
from scholarloop.core.llm import MockLLM
from scholarloop.core.profile import load_profile
from scholarloop.experiment.compare import (arm_stats, campaign_summary, render, run_campaign,
                                            t95, train_runs, win_rate)
from scholarloop.experiment.orchestrator import Orchestrator
from scholarloop.experiment.random_search import SOURCE, RandomProposer, knob_range, sample

ROOT = Path(__file__).resolve().parents[2]
STUB = load_profile(ROOT / "profiles" / "image-classification.yaml")
DIGITS = load_profile(ROOT / "profiles" / "digits-mlp.yaml")


def test_knob_ranges_bracket_the_default_and_respect_types():
    rng = random.Random(0)
    for name, default in DIGITS.knobs.items():
        for _ in range(200):
            v = sample(*knob_range(default), rng)
            assert type(v) is type(default), name
            if default:
                assert default / 4 * 0.99 <= v <= default * 4 * 1.01, (name, v)
    assert knob_range(0) == ("int", 0, 10) and knob_range(0.0)[0] == "zero_or_log"
    assert all(sample("logint", 1, 4, rng) >= 1 for _ in range(50))


def test_proposer_is_seeded_and_never_repeats_a_tried_config(tmp_path):
    a = RandomProposer(STUB, seed=3).propose([])
    b = RandomProposer(STUB, seed=3).propose([])
    assert a.config == b.config and a.admissible and a.hypothesis.source == SOURCE
    assert a.predicted_delta is None                       # random search predicts nothing

    tiny = RandomProposer(STUB, seed=0, space={k: ("choice", 1, 2) for k in STUB.knobs})
    orch = Orchestrator(None, STUB, ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "r")
    orch.reasoner = tiny
    run_campaign(orch, rounds=40)                          # 2^4 = 16 configs exist
    configs = [tuple(sorted(e.config.items())) for e in Ledger(tmp_path / "l.jsonl").read_all()
               if e.fidelity == ["smoke"]]
    assert len(configs) == len(set(configs)) == 16         # each once, then the space is exhausted


def test_random_arm_runs_through_the_same_funnel_with_no_llm_call(tmp_path):
    llm = MockLLM()                                        # any call would raise (empty queue)
    orch = Orchestrator(llm, STUB, ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "r")
    orch.reasoner = RandomProposer(STUB, seed=1, space={"lr": ("choice", 0.1, 0.1),
                                                       "depth": ("int", 19, 21),
                                                       "weight_decay": ("choice", 5e-4, 5e-4),
                                                       "warmup": ("int", 4, 6)})
    entries = run_campaign(orch, rounds=3)                 # near the stub's optimum: ideas climb
    assert llm.calls == []
    s = campaign_summary(entries, STUB)
    assert s["ideas"] == 3 and s["best_full"] is not None
    assert s["train_runs"] == train_runs(entries) == sum(
        {"smoke": 1, "verify": 3, "full": 5}[e.fidelity[0]] for e in entries)


def test_campaign_bounded_by_training_runs(tmp_path):
    orch = Orchestrator(None, STUB, ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "r")
    orch.reasoner = RandomProposer(STUB, seed=0)
    entries = run_campaign(orch, max_train_runs=5)
    assert 5 <= train_runs(entries) < 5 + 9                # stops after the round that crosses it
    with pytest.raises(ValueError):
        run_campaign(orch)


def _summary(val, held, runs=10):
    return {"ideas": 3, "train_runs": runs, "tiers": {"smoke": 3, "verify": 1, "full": 1}, "killed": 0,
            "best_full": None if val is None else {"id": "exp_0003", "val": val, "holdout": held,
                                                   "config": {}}}


def test_arm_stats_intervals_and_win_rate():
    s = arm_stats([_summary(3.0, 3.2), _summary(4.0, 4.4), _summary(None, None)])
    assert s["campaigns"] == 3 and s["reached_full"] == 2
    assert s["val"]["mean"] == 3.5 and s["val"]["ci95"][0] < 3.0 < 4.0 < s["val"]["ci95"][1]
    assert t95(2) == 4.30 and t95(12) == 2.23 and t95(500) == 2.04
    assert win_rate([1.0, 2.0], [3.0, 4.0], "minimize") == 1.0
    assert win_rate([1.0], [1.0], "minimize") == 0.5
    assert win_rate([], [1.0], "minimize") is None


def test_report_compares_two_arms_on_the_held_out_score():
    md = render({"agent": [_summary(3.0, 3.1), _summary(3.2, 3.3)],
                 "random": [_summary(3.5, 3.6), _summary(3.4, 3.2)]}, DIGITS, match="runs")
    assert "Budget matched by **runs**" in md and "| agent | 2 | 2 |" in md
    assert "P(agent reports a better held-out score than random) = **0.75**" in md


def test_compare_script_runs_random_offline_and_resumes(tmp_path, monkeypatch):
    from examples import compare_search
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args = ["--profile", "image-classification", "--arms", "random", "--repeats", "2",
            "--rounds", "2", "--out", str(tmp_path)]
    assert compare_search.main(args) == 0
    assert "| random | 2 |" in (tmp_path / "compare.md").read_text()
    first = (tmp_path / "random" / "rep1" / "summary.json").read_text()
    assert compare_search.main(args) == 0                  # finished campaigns are kept, not rerun
    assert (tmp_path / "random" / "rep1" / "summary.json").read_text() == first
    assert compare_search.main(args[:3] + ["agent"] + args[4:]) == 1   # the agent arm needs a key


def test_reasoner_searches_the_random_arms_space_and_explores_first():
    from scholarloop.experiment.reasoner import Reasoner
    from scholarloop.experiment.reasoning import analyze_search_space
    r = Reasoner(None, DIGITS, explore_ideas=2)
    prompt = lambda n: r.build_prompt({"constraints": analyze_search_space([], DIGITS), "n_ideas": n})  # noqa: E731
    lo, hi = knob_range(DIGITS.knobs["lr"])[1:]
    assert f"lr {lo:.3g}..{hi:.3g} (log scale)" in prompt(0)     # the same range random search samples
    assert "EXPLORE (idea 2 of the first 2)" in prompt(1)
    assert "REFINE" in prompt(2) and "EXPLORE" not in prompt(2)
