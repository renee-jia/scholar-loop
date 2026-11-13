"""ResearchSession: the review → ideas → human choice → handoff state machine, persisted and resumable."""

import json
from pathlib import Path

from scholarloop.core.ledger import Hypothesis, Ledger, LedgerEntry
from scholarloop.ideation.litscout import ArxivClient
from scholarloop.core.llm import MockLLM
from scholarloop.experiment.orchestrator import Orchestrator
from scholarloop.core.profile import load_profile
from scholarloop.ideation.session import ResearchSession, build_handoff

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_profile(ROOT / "profiles" / "image-classification.yaml")

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/1512.03385v1</id><title>Deep Residual Learning</title>
    <summary>Residual connections ease optimization.</summary><published>2015-12-10T00:00:00Z</published></entry>
  <entry><id>http://arxiv.org/abs/1608.03983v1</id><title>SGDR</title>
    <summary>Cosine annealing with warm restarts.</summary><published>2016-08-13T00:00:00Z</published></entry>
</feed>"""
ATOM2 = ATOM.replace("1512.03385v1", "1710.09412v1").replace("Deep Residual Learning", "Mixup") \
            .replace("Residual connections ease optimization.", "Convex combinations of training pairs.")

QUERIES = {"queries": ["mlp regularization", "cosine schedule"], "rationale": "r"}
REVIEW = {"overview": "o",
          "themes": [{"name": "Optimization", "summary": "s", "papers": ["arXiv:1512.03385v1"]}],
          "paper_notes": [{"id": "arXiv:1512.03385v1", "contribution": "skips", "limitation": "deep nets"}],
          "gaps": [{"gap": "skips on shallow MLPs", "why_open": "untested", "sources": ["arXiv:1512.03385v1"]}],
          "trends": ["deeper"]}


def _idea(title, sources=("arXiv:1512.03385v1",), feas=4, impact=4, description=None):
    return {"title": title, "description": description or f"{title}: try it on digits.", "gap": "skips on shallow MLPs",
            "sources": list(sources), "hypothesis": {"claim": f"{title} lowers error", "predicted_effect": "-0.3"},
            "experiment_plan": "set depth=20 warmup=5 lr=0.1 weight_decay=5e-4", "novelty_rationale": "n",
            "feasibility": feas, "impact": impact}


def _session(tmp_path, jsons, fetcher=lambda q, n: ATOM, **kw):
    return ResearchSession(MockLLM(jsons=jsons), PROFILE, tmp_path / "s", topic="mlp recipes",
                           sources=[ArxivClient(fetcher=fetcher)], **kw)


def test_review_then_ideas_then_select_writes_artifacts_and_handoff(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("Skip MLP"), _idea("Warm restarts", feas=2)]}])
    rev = s.review()
    assert s.stage == "reviewed" and rev.queries == QUERIES["queries"]
    # deduped across queries; SGDR's abstract matches "cosine schedule", so it ranks first
    assert [p.arxiv_id for p in rev.papers] == ["arXiv:1608.03983v1", "arXiv:1512.03385v1"]
    assert (s.out / "lit_review.md").read_text().startswith("# Literature review — mlp recipes")

    ideas = s.ideate(2)
    assert s.stage == "ideated" and [i.title for i in ideas] == ["Skip MLP", "Warm restarts"]
    assert all(i.novelty for i in ideas)                        # every idea was novelty-checked
    assert "| 1 | **Skip MLP**" in (s.out / "ideas.md").read_text()

    h = s.select(1)
    assert s.stage == "selected" and h["topic"] == "Skip MLP"
    assert h["lit_priors"] == ["Skip MLP (arXiv:1512.03385v1)"]
    assert "Deep Residual Learning: skips" in h["lit_context"]  # the idea's sources, with the review's notes
    assert "set depth=20" in h["guidance"]
    assert json.loads((s.out / "handoff.json").read_text()) == h
    assert "✅ selected" in (s.out / "ideas.md").read_text()


def test_session_persists_and_resumes_across_instances(tmp_path):
    s1 = _session(tmp_path, [QUERIES, REVIEW])
    s1.review()
    # a fresh process: same out dir, no topic needed, state + review come back
    s2 = ResearchSession(MockLLM(jsons=[{"ideas": [_idea("Skip MLP")]}]), PROFILE, tmp_path / "s",
                         sources=[ArxivClient(fetcher=lambda q, n: ATOM)])
    assert s2.stage == "reviewed" and s2.topic == "mlp recipes"
    assert s2.ideate(1)[0].title == "Skip MLP"
    s3 = ResearchSession(MockLLM(), PROFILE, tmp_path / "s")
    assert s3.stage == "ideated" and s3.select(1)["topic"] == "Skip MLP"


def test_review_feedback_adds_papers_and_resets_downstream(tmp_path):
    feeds = iter([ATOM, ATOM, ATOM2, ATOM2])                     # 2 queries, then 2 feedback queries
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("Skip MLP")]},
                            {"queries": ["mixup augmentation", "label smoothing"], "rationale": "r"},
                            {**REVIEW, "overview": "now covers mixup"}],
                 fetcher=lambda q, n: next(feeds))
    s.review()
    s.ideate(1)
    assert s.stage == "ideated"
    rev = s.review(feedback="also cover data augmentation like mixup")
    assert s.stage == "reviewed" and s.ideas == []               # ideas are stale after a re-review
    ids = [p.arxiv_id for p in rev.papers]
    assert ids == ["arXiv:1608.03983v1", "arXiv:1512.03385v1", "arXiv:1710.09412v1"]   # corpus grew, no dups
    assert rev.queries == ["mlp regularization", "cosine schedule", "mixup augmentation", "label smoothing"]
    planner_prompt = s.planner.llm.calls[3]["prompt"]
    assert "mixup" in planner_prompt and "do not repeat" in planner_prompt
    assert "also cover data augmentation" in s.reviewer.llm.calls[4]["prompt"]


def test_ideate_feedback_and_ledger_reach_the_agent(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    Ledger(ledger).append(LedgerEntry(id="exp_0001", domain=PROFILE.name, metric_name="m",
                                      hypothesis=Hypothesis(claim="Skip MLP lowers error", source="arXiv:1")))
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("Skip MLP")]},
                            {"ideas": [_idea("Cosine MLP", sources=("arXiv:1608.03983v1",))]}],
                 ledger_path=ledger)
    s.review()
    first = s.ideate(1)
    assert first[0].flagged and "already-tested" in first[0].novelty["reason"]   # the ledger already tested it
    second = s.ideate(1, feedback="avoid skips; try schedules")
    prompt = s.ideator.llm.calls[-1]["prompt"]
    assert "Skip MLP lowers error" in prompt and "avoid skips" in prompt and "- Skip MLP" in prompt
    assert second[0].title == "Cosine MLP" and not second[0].flagged


def test_auto_select_prefers_unflagged_highest_scored(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [
        _idea("Deep Residual Learning", feas=5, impact=5,            # lexical copy of a corpus paper -> flagged
              description="Residual connections ease optimization."),
        _idea("Mixup for digits", feas=3, impact=4),
        _idea("Label smoothing", feas=3, impact=3)]}])
    s.review()
    ideas = s.ideate(3)
    assert ideas[0].flagged
    assert s.auto_select()["topic"] == "Mixup for digits"


def test_interactive_scripted_human_refines_then_selects(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW,
                            {"ideas": [_idea("Skip MLP")]},
                            {"ideas": [_idea("Bolder skip MLP")]}])
    answers = iter(["", "a", "r make it bolder", "s 1"])          # junk, approve, refine ideas, select
    log = []
    h = s.interactive(n_ideas=1, input_fn=lambda _: next(answers), print_fn=log.append)
    assert h["topic"] == "Bolder skip MLP" and s.stage == "selected"
    assert any("type a, r" in line for line in log)              # junk input was explained, not fatal
    assert s.state.feedback[-1]["text"] == "make it bolder"


def test_interactive_quit_leaves_review_on_disk(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW])
    assert s.interactive(input_fn=lambda _: "q", print_fn=lambda *_: None) is None
    assert s.stage == "reviewed" and (s.out / "lit_review.md").exists()


def test_handoff_adopted_by_orchestrator_drives_a_real_funnel(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("Skip MLP")]}])
    s.review(); s.ideate(1)
    h = s.select(1)
    # the Reasoner turns the plan into a config; no Lit Scout / Director needed -- the human set the direction
    proposal = {"rationale": "follow the plan", "config": [{"name": "lr", "value": 0.1}, {"name": "depth", "value": 20},
                {"name": "weight_decay", "value": 5e-4}, {"name": "warmup", "value": 5}],
                "hypothesis": {"claim": "Skip MLP lowers error", "source": "arXiv:1512.03385v1", "predicted_effect": "-0.3"},
                "predicted_delta": -0.3, "idea_role": "treatment"}
    llm = MockLLM(jsons=[proposal])
    orch = Orchestrator(llm, PROFILE, ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "reg")
    orch.adopt(h)
    produced = orch.run(funnel=True)
    assert orch.topic == "Skip MLP" and len(produced) == 3 and produced[-1].verdict == "kept"
    prompt = llm.calls[0]["prompt"]
    assert "set depth=20" in prompt and "Deep Residual Learning" in prompt    # plan + literature reached the Reasoner
    assert produced[0].reasoning["search_space_constraints"]["priors"] == h["lit_priors"]


def test_build_handoff_without_review_notes():
    from scholarloop.ideation.ideator import Idea
    idea = Idea.from_dict({**_idea("Solo"), "novelty": {}, "selected": True})
    h = build_handoff(idea, None)
    assert h["topic"] == "Solo" and "- [arXiv:1512.03385v1]" in h["lit_context"]
