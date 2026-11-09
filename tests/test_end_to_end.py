"""End to end, offline: an ideation session picks an idea, examples/run_to_paper.py runs the
campaign on it and writes the paper, then the campaign is resumed and finally restarted fresh.

Every LLM call goes to a fake routed by the requested schema, so this exercises the real wiring
(handoff, orchestrator, runner, frozen scorer, held-out test, CV folds, skills, calibration,
writing) without an API key or network access. The stub engine keeps it to a few seconds."""

import dataclasses
import itertools
import json
from pathlib import Path

import pytest

from scholarloop.core.ledger import Ledger
from scholarloop.core.llm import MockLLM
from scholarloop.core.profile import load_profile
from scholarloop.ideation.litscout import ArxivClient
from scholarloop.ideation.metareview import MetaReviewer
from scholarloop.ideation.session import ResearchSession
from scholarloop.ideation.tournament import Tournament

ROOT = Path(__file__).resolve().parents[1]
ATOM = """<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/1608.03983v1</id><title>SGDR</title>
<summary>Cosine annealing with warm restarts.</summary><published>2016-08-13T00:00:00Z</published></entry>
</feed>"""
SRC = "arXiv:1608.03983v1"
# stub-engine configs: the first is the stub's hidden optimum, so the funnel climbs to the full tier
CONFIGS = [{"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5},
           {"lr": 0.08, "depth": 20, "weight_decay": 5e-4, "warmup": 4},
           {"lr": 0.3, "depth": 8, "weight_decay": 1e-2, "warmup": 0}]


class SchemaRoutedLLM(MockLLM):
    """Answers each agent by the keys of the schema it asks for."""

    def __init__(self):
        super().__init__()
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.configs = itertools.cycle(CONFIGS)

    def complete_json(self, prompt, schema, *, system=None, max_tokens=16000):
        self.calls.append({"prompt": prompt, "system": system})
        self.usage["calls"] += 1
        self.usage["input_tokens"] += len(prompt) // 4
        self.usage["output_tokens"] += 50
        p = set(schema.get("properties", {}))
        if "queries" in p:
            return {"queries": ["cosine schedules"], "rationale": "r"}
        if "themes" in p:
            return {"overview": "o", "themes": [{"name": "Schedules", "summary": "s", "papers": [SRC]}],
                    "paper_notes": [{"id": SRC, "contribution": "restarts", "limitation": "big nets"}],
                    "gaps": [{"gap": "cosine on small nets", "why_open": "untested", "sources": [SRC]}],
                    "trends": []}
        if "ideas" in p:
            return {"ideas": [{"title": t, "description": f"{t}.", "gap": "cosine on small nets",
                               "sources": [SRC], "experiment_plan": "tune the schedule",
                               "hypothesis": {"claim": f"{t} lowers error", "predicted_effect": "-1"},
                               "novelty_rationale": "n", "feasibility": f, "impact": 3}
                              for t, f in [("Warmup plus decay", 3), ("Constant rate", 5)]]}
        if p == {"winner", "rationale"}:
            first = prompt.split("Idea B:")[0]
            return {"winner": "A" if "Title: Warmup" in first else "B", "rationale": "bigger effect"}
        if "recommendations" in p:
            return {"recommendations": []}
        if "reasoning_trace" in p:
            cfg = next(self.configs)
            return {"reasoning_trace": "t", "config": [{"name": k, "value": v} for k, v in cfg.items()],
                    "hypothesis": {"claim": "warmup helps", "source": SRC, "predicted_effect": "-1"},
                    "predicted_delta": -1.0}
        if "verdict" in p and "concern" in p:
            return {"verdict": "run", "concern": "none"}
        if "worth_recording" in p:
            return {"worth_recording": True, "category": "schedule", "severity": 0.5,
                    "mitigation": "keep warmup"}
        if "decision" in p:
            return {"decision": "proceed", "rationale": "improving"}
        if "direction" in p:
            return {"direction": "d", "topic": "t", "rationale": "r"}
        if "findings" in p:
            return {"findings": []}
        if "abstract" in p:
            return {"title": "T", "abstract": "A.", "sections": [{"heading": "Results", "body": "It held."}]}
        if "recommendation" in p and "score" in p:
            return {"summary": "s", "strengths": ["a"], "weaknesses": ["b"], "score": 5,
                    "recommendation": "weak_accept"}
        raise AssertionError(f"unrouted schema {sorted(p)}")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """run_to_paper wired to the fake LLM, offline sources and the stub engine (holdout + 5 folds)."""
    import examples.run_to_paper as rtp

    llm = SchemaRoutedLLM()
    stub = dataclasses.replace(load_profile(ROOT / "profiles" / "image-classification.yaml"),
                               holdout=True, cv_folds=5)
    offline = lambda *a, **k: ArxivClient(fetcher=lambda q, n: ATOM)     # noqa: E731
    monkeypatch.setattr(rtp, "AnthropicLLM", lambda model=None: llm)
    monkeypatch.setattr(rtp, "load_profile", lambda path: stub)
    monkeypatch.setattr(rtp, "OpenAlexClient", offline)
    monkeypatch.setattr(rtp, "ArxivClient", offline)
    for var in ("SCHOLARLOOP_RESUME", "SCHOLARLOOP_PAPER_ONLY", "SCHOLARLOOP_PERSIST_SKILLS",
                "SCHOLARLOOP_DRY_PATIENCE", "SCHOLARLOOP_MAX_COST", "SCHOLARLOOP_POPULATION",
                "SCHOLARLOOP_MODEL", "SCHOLARLOOP_TOPIC", "SCHOLARLOOP_PROFILE"):
        monkeypatch.delenv(var, raising=False)

    session = ResearchSession(llm, stub, tmp_path / "research", topic="small-net schedules",
                              sources=[offline()], tournament=Tournament(llm, stub),
                              meta_reviewer=MetaReviewer(llm))
    session.review()
    session.ideate(2)
    session.auto_select()

    out = tmp_path / "campaign"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline")
    monkeypatch.setenv("SCHOLARLOOP_HANDOFF", str(tmp_path / "research" / "handoff.json"))
    monkeypatch.setenv("SCHOLARLOOP_OUT", str(out))
    monkeypatch.setenv("SCHOLARLOOP_WRITING", "oneshot")

    def run(steps: int, *, resume: bool = False) -> list:
        monkeypatch.setenv("SCHOLARLOOP_STEPS", str(steps))
        if resume:
            monkeypatch.setenv("SCHOLARLOOP_RESUME", "1")
        else:
            monkeypatch.delenv("SCHOLARLOOP_RESUME", raising=False)
        assert rtp.main() == 0
        return list(Ledger(out / "experiments.jsonl").read_all())

    return run, out, session


def test_ideation_to_paper_then_resume_then_fresh(campaign):
    run, out, session = campaign
    assert session.handoff["topic"] == "Warmup plus decay"        # the tournament's pick, not self-scores

    first = run(2)
    ids = [e.id for e in first]
    assert ids == [f"exp_{i:04d}" for i in range(1, len(ids) + 1)]
    full = [e for e in first if e.fidelity == ["full"]]
    assert full, "the stub's optimum should climb to the full tier"
    assert full[0].metric["folds"] == [0, 1, 2, 3, 4] and full[0].holdout_score() is not None
    for name in ("paper.md", "run.md", "usage.json", "calibration.json"):
        assert (out / name).exists(), name
    assert "held-out test" in (out / "run.md").read_text()
    usage1 = json.loads((out / "usage.json").read_text())
    calib1 = json.loads((out / "calibration.json").read_text())
    skills1 = sorted(p.name for p in (out / "skills").rglob("*") if p.is_file())
    assert calib1 and skills1

    resumed = run(1, resume=True)                                     # one more round on top
    assert [e.id for e in resumed[:len(first)]] == ids                # nothing rewritten ...
    assert len(resumed) > len(first)                                  # ... and the ids continue
    assert len({e.id for e in resumed}) == len(resumed)
    usage2 = json.loads((out / "usage.json").read_text())
    assert usage2["calls"] > usage1["calls"]                          # spend accumulates
    assert len(json.loads((out / "calibration.json").read_text())) >= len(calib1)
    assert set(skills1) <= {p.name for p in (out / "skills").rglob("*") if p.is_file()}
    assert (out / "paper.md").exists()                                # rewritten on the larger ledger

    fresh = run(1)                                                    # no resume: a clean start
    assert fresh[0].id == "exp_0001" and len(fresh) < len(resumed)
    assert json.loads((out / "usage.json").read_text())["calls"] < usage2["calls"]
