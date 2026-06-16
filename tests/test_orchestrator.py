"""The autonomous loop end-to-end on MockLLM: Reasoner proposes → enforce → run → reflect."""

from pathlib import Path

from scholarloop.ledger import Hypothesis, Ledger, LedgerEntry
from scholarloop.llm import MockLLM
from scholarloop.orchestrator import Orchestrator
from scholarloop.profile import load_profile
from scholarloop.reasoner import Reasoner

ROOT = Path(__file__).resolve().parent.parent
PROFILE = load_profile(ROOT / "profiles" / "image-classification.yaml")  # minimize, baseline 4.9


def _proposal_json(config: dict, *, source="arXiv:1608.03983", predicted_delta=-0.2):
    return {
        "reasoning_trace": "ledger shows headroom; trying a cosine+SGD recipe",
        "config": [{"name": k, "value": v} for k, v in config.items()],
        "hypothesis": {"claim": "tuned recipe helps", "source": source,
                       "predicted_effect": "lower val error"},
        "predicted_delta": predicted_delta,
    }


def _entry(eid, config, score, verdict="discarded"):
    return LedgerEntry(
        id=eid, domain="image-classification",
        hypothesis=Hypothesis(claim="c", source="arXiv:1"),
        metric_name="val_top1_err", config=config,
        fidelity=["verify"], metric={"verify": score, "seeds": [score]}, verdict=verdict,
    )


def test_reasoner_enforces_ruled_out_region():
    # ledger rules out high lr (winners low, all losers high) -> a high-lr proposal must violate
    entries = [
        _entry("w1", {"lr": 0.1}, 4.5, verdict="kept"),
        _entry("w2", {"lr": 0.15}, 4.7, verdict="kept"),
        _entry("l1", {"lr": 0.4}, 6.0),
        _entry("l2", {"lr": 0.5}, 6.3),
    ]
    llm = MockLLM(jsons=[_proposal_json({"lr": 0.9})])
    proposal = Reasoner(llm, PROFILE).propose(entries)
    assert not proposal.admissible
    assert any("lr>" in v for v in proposal.violations)


def test_reasoner_admits_in_bounds_config():
    entries = [
        _entry("w1", {"lr": 0.1}, 4.5, verdict="kept"),
        _entry("w2", {"lr": 0.15}, 4.7, verdict="kept"),
        _entry("l1", {"lr": 0.4}, 6.0),
        _entry("l2", {"lr": 0.5}, 6.3),
    ]
    llm = MockLLM(jsons=[_proposal_json({"lr": 0.12})])   # inside the winning range
    proposal = Reasoner(llm, PROFILE).propose(entries)
    assert proposal.admissible
    assert proposal.config == {"lr": 0.12}


def test_full_autonomous_loop_runs_and_reflects(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    # two scripted proposals, both near the stub's hidden optimum (lr.1/depth20/wd5e-4/warmup5)
    llm = MockLLM(jsons=[
        _proposal_json({"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}),
        _proposal_json({"lr": 0.1, "depth": 22, "weight_decay": 5e-4, "warmup": 5},
                       predicted_delta=-0.1),
    ])
    orch = Orchestrator(llm, PROFILE, ledger_path=ledger_path, registry_dir=tmp_path / "registry")
    entries = orch.run(2, fidelity="smoke")

    assert len(entries) == 2
    e1, e2 = entries
    # the proposed config was actually run (captured from the engine)
    assert e1.config["lr"] == 0.1 and e1.config["depth"] == 20
    # reasoning trace + search-space constraints were recorded
    assert "trace" in e1.reasoning and "search_space_constraints" in e1.reasoning
    # predict-then-verify: e1 has no parent (no prior kept) → measured None; e2 calibrates vs e1
    assert e1.prediction["predicted"] == -0.2
    assert e2.prediction["measured"] is not None
    assert e2.prediction["calibration_error"] is not None
    # both kept (near-optimal beats baseline 4.9) and persisted
    assert {e.verdict for e in entries} == {"kept"}
    assert len(list(Ledger(ledger_path).read_all())) == 2


def test_loop_grounds_on_lit_scout_and_traces_both_agents(tmp_path):
    from scholarloop.litscout import ArxivClient, LitScout
    from tests.test_litscout import ATOM_XML

    arxiv = ArxivClient(fetcher=lambda q, n: ATOM_XML)
    scout = LitScout(MockLLM(jsons=[{"findings": [
        {"technique": "cosine schedule", "source": "arXiv:1608.03983",
         "predicted_effect": "lower err", "rationale": "warm restarts"}]}]), arxiv)
    reasoner_llm = MockLLM(jsons=[_proposal_json(
        {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5})])

    orch = Orchestrator(reasoner_llm, PROFILE, lit_scout=scout, topic="image classification",
                        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")
    entry = orch.step()

    assert entry is not None
    # the Lit Scout's finding flowed into the Reasoner's search-space priors, then onto the ledger
    priors = entry.reasoning["search_space_constraints"]["priors"]
    assert any("cosine schedule" in p for p in priors)
    # both agents logged to the one shared trace
    assert {c.agent for c in orch.trace.calls} == {"lit_scout", "reasoner"}


def test_full_loop_debate_runs_then_reflector_writes_skill(tmp_path):
    from scholarloop.debate import DebatePanel
    from scholarloop.reflector import Reflector
    from scholarloop.skills import SkillLibrary

    reasoner_llm = MockLLM(jsons=[_proposal_json(
        {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5})])
    panel = DebatePanel(MockLLM(jsons=[{"verdict": "run", "concern": ""}] * 3), PROFILE.debate_roles)
    reflector = Reflector(MockLLM(jsons=[{"worth_recording": True, "category": "optimizer",
                                          "severity": 0.7, "mitigation": "cosine schedule helps"}]))
    lib = SkillLibrary(tmp_path / "skills")

    orch = Orchestrator(reasoner_llm, PROFILE, debate_panel=panel, reflector=reflector,
                        skill_library=lib, ledger_path=tmp_path / "ledger.jsonl",
                        registry_dir=tmp_path / "registry")
    entry = orch.step()

    assert entry is not None
    assert entry.reasoning["debate"]["decision"] == "run"          # panel verdict recorded
    skills = lib.all()
    assert len(skills) == 1 and skills[0].source == entry.id        # reflection persisted, tied to the run


def test_full_loop_debate_reject_skips_the_run(tmp_path):
    from scholarloop.debate import DebatePanel

    reasoner_llm = MockLLM(jsons=[_proposal_json(
        {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5})])
    panel = DebatePanel(MockLLM(jsons=[{"verdict": "reject", "concern": "a"},
                                       {"verdict": "reject", "concern": "b"},
                                       {"verdict": "run", "concern": ""}]), PROFILE.debate_roles)
    orch = Orchestrator(reasoner_llm, PROFILE, debate_panel=panel,
                        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")
    assert orch.step() is None
    assert list(Ledger(tmp_path / "ledger.jsonl").read_all()) == []  # nothing ran


def test_funnel_winner_climbs_smoke_verify_full(tmp_path):
    cfg = {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}   # near-ideal -> beats baseline
    orch = Orchestrator(MockLLM(jsons=[_proposal_json(cfg)]), PROFILE,
                        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")
    produced = orch.funnel_step()
    assert [e.fidelity[0] for e in produced] == ["smoke", "verify", "full"]
    assert all(e.verdict == "kept" for e in produced)
    # tiers chain via parent
    assert produced[1].parent == produced[0].id and produced[2].parent == produced[1].id


def test_funnel_loser_dies_at_smoke(tmp_path):
    cfg = {"lr": 0.05, "depth": 18, "weight_decay": 1e-4, "warmup": 0}   # far from ideal -> ~7.9 > 4.9
    orch = Orchestrator(MockLLM(jsons=[_proposal_json(cfg)]), PROFILE,
                        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")
    produced = orch.funnel_step()
    assert len(produced) == 1
    assert produced[0].fidelity == ["smoke"] and produced[0].verdict == "discarded"


def test_population_funnel_smokes_all_then_climbs_only_survivors(tmp_path):
    # propose 3 ideas in one round: two beat the baseline (climb), one is far off (dies at smoke)
    good1 = {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}    # ~3.76
    good2 = {"lr": 0.1, "depth": 22, "weight_decay": 5e-4, "warmup": 5}    # ~3.92
    bad = {"lr": 0.05, "depth": 18, "weight_decay": 1e-4, "warmup": 0}     # ~7.9 -> dies at smoke
    llm = MockLLM(jsons=[_proposal_json(good1), _proposal_json(good2), _proposal_json(bad)])
    orch = Orchestrator(llm, PROFILE, ledger_path=tmp_path / "ledger.jsonl",
                        registry_dir=tmp_path / "registry")
    produced = orch.population_step(k=3)

    smokes = [e for e in produced if e.fidelity == ["smoke"]]
    assert len(smokes) == 3                                  # every idea got a smoke screen
    assert len({e.id for e in produced}) == len(produced)    # ids never collide across the batch
    # the clearly-bad idea never leaves smoke
    bad_rows = [e for e in produced if e.config == bad]
    assert len(bad_rows) == 1 and bad_rows[0].verdict == "discarded"
    # at least one good idea climbed past smoke into verify/full
    assert any(e.fidelity[0] in ("verify", "full") for e in produced)


def test_run_with_governor_stops_on_round_cap(tmp_path):
    from scholarloop.governor import Governor

    cfg = {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}
    gov = Governor(max_rounds=2)
    orch = Orchestrator(MockLLM(jsons=[_proposal_json(cfg)] * 8), PROFILE,
                        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")
    orch.run(governor=gov, funnel=True)                     # no n_steps: the governor bounds the loop
    assert gov.rounds == 2                                   # stopped exactly at the cap


def test_best_of_returns_none_for_an_all_killed_batch(tmp_path):
    orch = Orchestrator(MockLLM(), PROFILE, ledger_path=tmp_path / "ledger.jsonl",
                        registry_dir=tmp_path / "registry")
    killed = LedgerEntry(id="exp_k", domain="image-classification",
                         hypothesis=Hypothesis("c", "arXiv:1"), metric_name="val_top1_err",
                         fidelity=["smoke"], metric={}, verdict="killed")   # no score
    assert orch._best_of([killed]) is None                  # nothing to reflect on, not a killed entry


def test_calibrate_records_reasoner_and_debate_verdicts(tmp_path):
    orch = Orchestrator(MockLLM(), PROFILE, ledger_path=tmp_path / "ledger.jsonl",
                        registry_dir=tmp_path / "registry")
    # an entry that carries both a scored prediction and a debate "run" decision that paid off
    entry = LedgerEntry(
        id="e1", domain="image-classification", hypothesis=Hypothesis("c", "arXiv:1"),
        metric_name="val_top1_err", fidelity=["verify"], metric={"verify": 4.0, "seeds": [4.0]},
        prediction={"predicted": -1.0, "measured": -0.8, "calibration_error": 0.2},
        reasoning={"debate": {"decision": "run"}}, verdict="kept")
    orch._calibrate(entry)
    stats = orch.calibration.by_agent()
    assert stats["reasoner"]["n"] == 1 and stats["reasoner"]["hit_rate"] == 1.0   # right direction
    assert stats["debate"]["n"] == 1 and stats["debate"]["hit_rate"] == 1.0        # approved -> kept

    # a rejected/uncalibratable entry adds nothing
    bare = LedgerEntry(id="e2", domain="image-classification", hypothesis=Hypothesis("c", "arXiv:1"),
                       metric_name="val_top1_err", fidelity=["smoke"], metric={"smoke": 9.0},
                       prediction={"predicted": -1.0, "measured": None}, verdict="discarded")
    orch._calibrate(bare)
    assert orch.calibration.by_agent()["reasoner"]["n"] == 1   # measured None -> not scored


def test_promote_gate_uses_statistical_significance(tmp_path):
    orch = Orchestrator(MockLLM(), PROFILE,            # promote_z=1.0 default
                        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")
    # mean (4.8) is below the gate 4.9, but the seed spread makes it NOT significant -> no promotion
    noisy = LedgerEntry(id="e", domain="image-classification",
                        hypothesis=Hypothesis("c", "arXiv:1"), metric_name="val_top1_err",
                        fidelity=["verify"], metric={"verify": 4.8, "seeds": [4.4, 4.8, 5.2]},
                        verdict="kept")
    assert orch._promote(noisy, "verify", 4.9) is False
    # a tight result clearly below the gate -> the pessimistic bound clears it -> promote
    tight = LedgerEntry(id="e2", domain="image-classification",
                        hypothesis=Hypothesis("c", "arXiv:1"), metric_name="val_top1_err",
                        fidelity=["verify"], metric={"verify": 4.0, "seeds": [3.8, 4.0, 4.2]},
                        verdict="kept")
    assert orch._promote(tight, "verify", 4.9) is True
    assert orch._promote(tight, "full", 4.9) is False        # full is terminal
    # a degraded verify (only 1 seed survived the crash) is NOT the robustness check full deserves
    degraded = LedgerEntry(id="e3", domain="image-classification",
                           hypothesis=Hypothesis("c", "arXiv:1"), metric_name="val_top1_err",
                           fidelity=["verify"], metric={"verify": 4.0, "seeds": [4.0]}, verdict="kept")
    assert orch._promote(degraded, "verify", 4.9) is False
    # an unrecognized tier never climbs on the bare mean (default-deny)
    assert orch._promote(tight, "rerun", 4.9) is False


def test_smoke_screen_keeps_borderline_but_drops_clearly_worse(tmp_path):
    # Smoke is a single noisy seed: it should keep a borderline idea (worse than the gate but within
    # the slack band) for a fair multi-seed verify, while still discarding the clearly-worse ones.
    orch = Orchestrator(MockLLM(), PROFILE,                   # smoke_slack=0.25 default; gate 4.9 -> band 6.125
                        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")

    def smoke(score):
        return LedgerEntry(id="s", domain="image-classification",
                           hypothesis=Hypothesis("c", "arXiv:1"), metric_name="val_top1_err",
                           fidelity=["smoke"], metric={"smoke": score, "seeds": [score]}, verdict="kept")

    assert orch._promote(smoke(5.5), "smoke", 4.9) is True    # worse than gate but within slack -> promote
    assert orch._promote(smoke(7.0), "smoke", 4.9) is False   # clearly worse than the band -> drop cheaply
    # the slack is smoke-only: at verify the same borderline number must clear the strict gate
    assert orch._promote(smoke(5.5), "verify", 4.9) is False


def test_orchestrator_skips_inadmissible_proposal(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    # seed a ledger that rules out high lr
    led = Ledger(ledger_path)
    for e in [_entry("w1", {"lr": 0.1}, 4.5, "kept"), _entry("w2", {"lr": 0.15}, 4.7, "kept"),
              _entry("l1", {"lr": 0.4}, 6.0), _entry("l2", {"lr": 0.5}, 6.3)]:
        led.append(e)
    llm = MockLLM(jsons=[_proposal_json({"lr": 0.95})])   # ruled out
    orch = Orchestrator(llm, PROFILE, ledger_path=ledger_path, registry_dir=tmp_path / "registry")
    assert orch.step() is None
    assert len(list(led.read_all())) == 4   # nothing new appended
