"""An idea study: the selected idea's arms run as one frozen, matched protocol."""

import copy
import json
from dataclasses import replace

import pytest

from scholarloop.core.ledger import Ledger
from scholarloop.core.llm import MockLLM
from scholarloop.experiment.idea_study import IdeaStudy, validate_study
from scholarloop.ideation.litscout import ArxivClient
from scholarloop.ideation.session import ResearchSession
from scholarloop.review.contracts import ReviewBlocked
from tests.experiment.test_agenda import SPEC
from tests.ideation.test_session import ATOM, QUERIES, REVIEW, _idea
from tests.test_scientific_review import setup


def fixture(tmp_path, llm=None):
    profile, ledger, registry, _ = setup(tmp_path)
    profile = replace(profile, cv_folds=5)
    ledger.unlink()
    (registry / "exp_0001.json").unlink()
    research = ResearchSession(MockLLM(jsons=[QUERIES, REVIEW, {"ideas": [_idea("Mechanism")]}]),
                               profile, tmp_path / "research", topic="Mechanism tests",
                               sources=[ArxivClient(fetcher=lambda q, n: ATOM)])
    research.review(), research.ideate(1)
    idea_id = research.select(1)["idea"]["idea_id"]
    study = IdeaStudy(llm, profile, research.out, ledger_path=ledger, registry_dir=registry,
                      out_dir=tmp_path, min_seeds=2)
    spec = copy.deepcopy(SPEC)
    spec.pop("issue_ids")
    reference, treatment = spec["conditions"][0], spec["conditions"][1]
    reference.update(config=[], edits=[])
    knob = next(iter(profile.knobs))
    control = {**copy.deepcopy(treatment), "name": "noise_control",
               "config": [{"name": knob, "value": profile.knobs[knob] * 3 + 1}], "edits": []}
    spec["conditions"].append(control)
    spec.update(purpose="falsification", fidelity="verify", seeds=[0, 1], idea_ids=[idea_id], holdout=False,
                omitted_controls=["yoked arm: needs per-seed replay"],
                arms=[{"condition": reference["name"], "role": "reference", "isolates": "the default recipe"},
                      {"condition": treatment["name"], "role": "treatment", "isolates": "the idea"},
                      {"condition": "noise_control", "role": "control", "isolates": "generic noise"}])
    spec["budget_seconds"] *= 3
    return research, study, spec, profile


def test_every_arm_runs_on_the_same_seeds_and_a_worse_control_is_not_pruned(tmp_path):
    research, study, spec, _ = fixture(tmp_path)
    state = study.plan(spec=spec)
    page = (study.out / f"{state['protocol_id']}.md").read_text()
    assert "What each arm isolates" in page and "yoked arm: needs per-seed replay" in page
    done = study.execute(state["protocol_id"], author="Researcher")
    assert done["status"] == "EXECUTED" and len(done["results"]) == 3
    entries = list(Ledger(study.review.ledger_path).read_all())
    assert [e.metric["seed_ids"] for e in entries] == [[0, 1]] * 3      # matched seeds, same tier
    assert all(e.purpose == "falsification" and e.fidelity == ["verify"] for e in entries)
    assert "discarded" in {e.verdict for e in entries}                  # scored worse, still ran
    analysis = json.loads((study.protocols / f"{state['protocol_id']}.analysis.json").read_text())
    assert [c["status"] for c in analysis["comparisons"]] == ["complete", "complete"]
    assert research.out.joinpath("handoff.json").exists()               # the selection stands


@pytest.mark.parametrize("change,message", [
    ("few_seeds", "at least 5 seeds"),
    ("edited_reference", "unmodified default recipe"),
    ("duplicate", "same code and config"),
    ("no_control", "at least one control"),
    ("unlabelled", "exactly once in `arms`"),
    ("other_idea", "exactly the selected idea"),
])
def test_an_unmatched_or_incomplete_study_is_rejected(tmp_path, change, message):
    research, study, spec, profile = fixture(tmp_path)
    idea = json.loads((research.out / "handoff.json").read_text())["idea"]
    min_seeds = 2
    if change == "few_seeds":
        min_seeds = 5
    elif change == "edited_reference":
        spec["conditions"][0]["config"] = copy.deepcopy(spec["conditions"][2]["config"])
        spec["conditions"][2]["config"][0]["value"] += 1
    elif change == "duplicate":
        spec["conditions"][2].update(config=copy.deepcopy(spec["conditions"][1]["config"]),
                                     edits=copy.deepcopy(spec["conditions"][1]["edits"]))
    elif change == "no_control":
        spec["arms"][2]["role"] = "treatment"
    elif change == "unlabelled":
        spec["arms"].pop()
    elif change == "other_idea":
        spec["idea_ids"] = ["idea_other"]
    assert any(message in e for e in validate_study(spec, profile, idea, min_seeds=min_seeds))


def test_planner_output_is_checked_and_a_changed_selection_makes_the_plan_stale(tmp_path):
    research, _, spec, profile = fixture(tmp_path)
    bad = copy.deepcopy(spec)
    bad["arms"][2]["role"] = "treatment"
    llm = MockLLM(jsons=[bad, spec])
    study = IdeaStudy(llm, profile, research.out, ledger_path=tmp_path / "ledger.jsonl",
                      registry_dir=tmp_path / "registry", out_dir=tmp_path, min_seeds=2)
    state = study.plan(feedback="include the noise control")
    assert "at least one control" in llm.calls[1]["prompt"]             # the retry saw the violation
    first = json.loads(llm.calls[0]["prompt"])
    assert first["idea"]["title"] == "Mechanism" and first["author_feedback"] == "include the noise control"
    handoff = research.out / "handoff.json"
    handoff.write_text(handoff.read_text().replace("Mechanism", "Another mechanism"))
    with pytest.raises(ReviewBlocked, match="stale"):
        study.execute(state["protocol_id"], author="Researcher")
