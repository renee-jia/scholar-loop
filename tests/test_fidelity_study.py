import json
from pathlib import Path

from scholarloop.core.llm import MockLLM
from scholarloop.core.profile import load_profile
from scholarloop.fidelity_study import (IdeaJudge, arm_handoff, build_report, judged, protocol_reference,
                                        structure, summarise)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(ROOT / "profiles" / "digits-mlp.yaml")
TRAIN = PROFILE.train_entrypoint
REPO_SOURCE = (PROFILE.root / TRAIN).read_text(encoding="utf-8")
IDEA = {"title": "identity skip", "description": "add a skip", "hypothesis": {}, "experiment_plan": "two arms"}


def campaign(directory: Path, runs: list[dict]) -> Path:
    """A recorded campaign: one ledger line and one candidate snapshot per run."""
    (directory / "registry" / "candidates").mkdir(parents=True)
    lines = []
    for i, run in enumerate(runs, 1):
        cid = run["id"]
        (directory / "registry" / "candidates" / f"{cid}.json").write_text(json.dumps(
            {"id": cid, "sources": {TRAIN: run.get("source", REPO_SOURCE)}, "config": run.get("config", {})}), encoding="utf-8")
        reasoning = {"idea_role": run["role"]} if run.get("role") else {}
        lines.append(json.dumps({"id": f"exp_{i:04d}", "candidate_id": cid, "fidelity": [run.get("tier", "smoke")],
                                 "metric": {"seed_ids": run.get("seeds", [0])}, "reasoning": reasoning}))
    (directory / "experiments.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "done.json").write_text("{}", encoding="utf-8")
    return directory


def test_the_context_arm_drops_only_the_idea_record():
    handoff = {"topic": "t", "guidance": "Test this idea", "lit_context": "c", "idea": IDEA}
    assert arm_handoff(handoff, "enforced") is handoff
    assert arm_handoff(handoff, "context") == {"topic": "t", "guidance": "Test this idea", "lit_context": "c"}


def test_records_alone_give_edits_pairs_and_declared_roles(tmp_path):
    skip = REPO_SOURCE + "\n# skip path\n"
    tested = campaign(tmp_path / "tested", [
        {"id": "a" * 64, "role": "control"},
        {"id": "b" * 64, "role": "treatment", "source": skip},
        {"id": "c" * 64, "role": "treatment", "source": skip, "config": {"lr": 0.2}}])   # other knob: no pair
    s = structure(tested, PROFILE)
    assert s["edit_used"] and s["structural_pairs"] == 1 and s["declared_pair"]
    assert s["pairs"] == [("a" * 64, "b" * 64)] and len(s["ideas"]) == 3
    tuned = campaign(tmp_path / "tuned", [{"id": "d" * 64}, {"id": "e" * 64, "config": {"lr": 0.2}}])
    s = structure(tuned, PROFILE)
    assert not s["edit_used"] and s["structural_pairs"] == 0 and not s["declared_pair"]
    seeds = campaign(tmp_path / "seeds", [{"id": "f" * 64, "seeds": [0]},
                                          {"id": "g" * 64, "seeds": [1], "source": skip}])
    assert structure(seeds, PROFILE)["structural_pairs"] == 0            # different seeds do not pair


def test_the_judge_must_answer_for_every_candidate_and_pair(tmp_path):
    skip = REPO_SOURCE + "\n# skip path\n"
    s = structure(campaign(tmp_path / "c", [{"id": "a" * 64}, {"id": "b" * 64, "source": skip}]), PROFILE)
    judge = IdeaJudge(MockLLM())
    full = {"candidates": [{"id": "a" * 10, "applies_idea": False, "reason": "r"},
                           {"id": "b" * 10, "applies_idea": True, "reason": "r"}],
            "pairs": [{"a": "b" * 10, "b": "a" * 10, "isolates_idea": True, "reason": "r"}]}
    assert judge.postcheck(full, {"structure": s}) == []
    assert len(judge.postcheck({"candidates": full["candidates"][:1], "pairs": []}, {"structure": s})) == 2
    prompt = judge.build_prompt({"structure": s, "idea": IDEA})
    assert "# skip path" in prompt and "identity skip" in prompt
    out = judged(s, full)
    assert out == {"idea_applied": True, "matched_control": True, "idea_runs": 2, "idea_run_share": 1.0}
    none = judged(s, {"candidates": [{**c, "applies_idea": False} for c in full["candidates"]],
                      "pairs": [{**full["pairs"][0], "isolates_idea": False}]})
    assert none["idea_applied"] is False and none["idea_runs"] == 0


def test_the_report_counts_by_arm_and_leaves_roles_unrecorded_in_the_context_arm(tmp_path):
    skip = REPO_SOURCE + "\n# skip path\n"
    campaign(tmp_path / "enforced" / "c1", [{"id": "a" * 64, "role": "control"},
                                            {"id": "b" * 64, "role": "treatment", "source": skip}])
    campaign(tmp_path / "context" / "c1", [{"id": "d" * 64}, {"id": "e" * 64, "config": {"lr": 0.2}}])
    report = build_report(tmp_path, PROFILE, {"idea": IDEA}, 4, llm=None)
    s = report["summary"]
    assert s["enforced"]["edit_used"]["k"] == 1 and s["context"]["edit_used"]["k"] == 0
    assert s["enforced"]["declared_pair"] == {"k": 1, "n": 1, "interval_95": s["enforced"]["declared_pair"]["interval_95"]}
    assert s["context"]["declared_pair"]["n"] == 0 and s["enforced"]["idea_applied"]["n"] == 0   # no judge ran
    text = (tmp_path / "fidelity.md").read_text(encoding="utf-8")
    assert "not recorded" in text and "| enforced | c1 |" in text
    assert summarise([])["enforced"]["campaigns"] == 0


def test_a_recorded_protocol_reports_whether_every_arm_ran_on_the_same_seeds(tmp_path):
    (tmp_path / "registry" / "protocols").mkdir(parents=True)
    spec = {"specification": {"spec": {"conditions": [{"name": "plain"}, {"name": "skip"}]}}}
    (tmp_path / "registry" / "protocols" / "p1.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "registry" / "protocols" / "p1.analysis.json").write_text("{}", encoding="utf-8")
    lines = [json.dumps({"id": f"protocol_p1_{arm}", "protocol": {"id": "p1", "condition": arm},
                         "metric": {"seed_ids": [1, 2, 3]}}) for arm in ("plain", "skip")]
    (tmp_path / "experiments.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (row,) = protocol_reference(tmp_path)
    assert row["all_arms_ran"] and row["same_seeds"] and row["seeds_per_arm"] == 3 and row["registered_arms"] == 2
