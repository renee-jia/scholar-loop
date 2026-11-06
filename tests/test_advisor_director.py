"""Advisor + Director agents and the orchestrator's PROCEED/REFINE/PIVOT loop control."""

from pathlib import Path

from scholarloop.advisor import Advisor
from scholarloop.director import Director
from scholarloop.ledger import Hypothesis, LedgerEntry
from scholarloop.llm import MockLLM
from scholarloop.orchestrator import Orchestrator
from scholarloop.profile import load_profile

ROOT = Path(__file__).resolve().parent.parent
PROFILE = load_profile(ROOT / "profiles" / "image-classification.yaml")


def _proposal_json(config, *, source="arXiv:1608.03983", predicted_delta=-0.2):
    return {"reasoning_trace": "t", "config": [{"name": k, "value": v} for k, v in config.items()],
            "hypothesis": {"claim": "c", "source": source, "predicted_effect": "lower err"},
            "predicted_delta": predicted_delta}


def _entry(eid, score, verdict="kept"):
    return LedgerEntry(id=eid, domain="image-classification",
                       hypothesis=Hypothesis("c", "arXiv:1"), metric_name="val_top1_err",
                       config={"lr": 0.1}, fidelity=["smoke"], metric={"smoke": score},
                       verdict=verdict, ts=1.0)


def test_advisor_returns_decision():
    out = Advisor(MockLLM(jsons=[{"decision": "pivot", "rationale": "exhausted lr sweep"}])) \
        .advise(_entry("e2", 5.0), _entry("e1", 4.0))
    assert out["decision"] == "pivot"


def test_director_picks_topic_and_direction():
    out = Director(MockLLM(jsons=[{"direction": "try attention", "topic": "vision transformer cifar",
                                   "rationale": "headroom"}]), PROFILE) \
        .direct([_entry("e1", 4.5)])
    assert out["topic"] == "vision transformer cifar"


def test_pivot_clears_lit_cache_and_calls_director(tmp_path):
    cfg = {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}
    advisor = Advisor(MockLLM(jsons=[{"decision": "pivot", "rationale": "switch direction"}]))
    director = Director(MockLLM(jsons=[{"direction": "explore augmentation",
                                        "topic": "data augmentation", "rationale": "new angle"}]),
                        PROFILE)
    from scholarloop.litscout import ArxivClient, LitScout
    scout = LitScout(MockLLM(), ArxivClient(fetcher=lambda q, n: "<feed/>"))   # never called here
    orch = Orchestrator(MockLLM(jsons=[_proposal_json(cfg)]), PROFILE,
                        advisor=advisor, director=director, lit_scout=scout,
                        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")
    orch._lit = ("stale lit", ["x"])      # pretend literature was cached
    orch.step()
    assert orch._lit is None              # pivot dropped the cache so the scout re-grounds
    assert orch.topic == "data augmentation"       # Director chose the new topic
    assert orch.guidance == "explore augmentation"


def test_pivot_without_a_scout_keeps_adopted_literature(tmp_path):
    # a human-chosen idea's literature (ResearchSession handoff) is the only grounding the loop has
    # when no Lit Scout is configured -- a pivot must not throw it away
    cfg = {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}
    advisor = Advisor(MockLLM(jsons=[{"decision": "pivot", "rationale": "switch direction"}]))
    orch = Orchestrator(MockLLM(jsons=[_proposal_json(cfg)]), PROFILE, advisor=advisor,
                        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")
    orch.adopt({"topic": "chosen idea", "guidance": "test it", "lit_context": "ctx", "lit_priors": ["p"]})
    orch.step()
    assert orch._lit == ("ctx", ["p"])
    assert "change approach" in orch.guidance       # the pivot still redirects the Reasoner


def test_refine_is_bounded_then_forces_pivot(tmp_path):
    cfg1 = {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}
    cfg2 = {"lr": 0.1, "depth": 22, "weight_decay": 5e-4, "warmup": 5}   # distinct -> not deduped
    # advisor says refine twice; with max_refines=2 the 2nd refine should force a pivot (count resets)
    advisor = Advisor(MockLLM(jsons=[{"decision": "refine", "rationale": "tweak lr"},
                                     {"decision": "refine", "rationale": "tweak wd"}]))
    orch = Orchestrator(MockLLM(jsons=[_proposal_json(cfg1), _proposal_json(cfg2)]), PROFILE,
                        advisor=advisor, max_refines=2,
                        ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry")
    orch.step()
    assert orch._refine_count == 1        # first refine counted
    orch.step()
    assert orch._refine_count == 0        # second hit the bound -> forced pivot reset the counter
