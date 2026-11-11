"""Pins for defects found in the adversarial review pass (each was reproduced before the fix)."""

import json
from pathlib import Path

from scholarloop.core.agent import Agent
from scholarloop.experiment.governor import Governor
from scholarloop.core.ledger import Hypothesis, Ledger, LedgerEntry
from scholarloop.ideation.litscout import ArxivClient, Paper, retrieve
from scholarloop.core.llm import MockLLM
from scholarloop.experiment.orchestrator import Orchestrator
from scholarloop.writing.paper import grounded_registry
from scholarloop.core.profile import load_profile
from scholarloop.experiment.reasoner import Reasoner
from scholarloop.experiment.reasoning import SearchSpaceConstraints, is_duplicate_config, same_config

ROOT = Path(__file__).resolve().parent.parent
PROFILE = load_profile(ROOT / "profiles" / "image-classification.yaml")   # knobs: lr .05, depth 18, wd 1e-4, warmup 0


def _proposal(config):
    return {"rationale": "t", "config": [{"name": k, "value": v} for k, v in config.items()],
            "hypothesis": {"claim": "c", "source": "arXiv:1", "predicted_effect": "e"}, "predicted_delta": -0.1}


def _entry(eid, config, score, verdict="discarded"):
    return LedgerEntry(id=eid, domain=PROFILE.name, hypothesis=Hypothesis(claim="c", source="arXiv:1"),
                       metric_name="val_top1_err", config=config, fidelity=["smoke"],
                       metric={"smoke": score, "seeds": [score]}, verdict=verdict)


# 1. dedup / ruled-out must bind on the EFFECTIVE config a real engine logs (defaults filled in)
def test_dedup_matches_override_against_full_effective_ledger_config(tmp_path):
    full = {"lr": 0.1, "depth": 18, "weight_decay": 1e-4, "warmup": 0}       # what the stub engine logs
    entries = [_entry("e1", full, 5.0, "kept")]
    assert is_duplicate_config({"lr": 0.1}, entries, defaults=PROFILE.knobs)   # override == tried once defaults apply
    assert not is_duplicate_config({"lr": 0.2}, entries, defaults=PROFILE.knobs)
    assert same_config({"lr": 0.1, "extra_engine_knob": 3}, {"lr": 0.1, "depth": 18})   # agree on shared keys
    proposal = Reasoner(MockLLM(jsons=[_proposal({"lr": 0.1})]), PROFILE).propose(entries)
    assert "config already tried" in proposal.violations


def test_ruled_out_region_catches_a_knob_left_at_its_default():
    c = SearchSpaceConstraints(ruled_out_bounds=[{"knob": "lr", "op": "<", "value": 0.1}])
    assert c.violations({"depth": 20}) == []                                   # no defaults known: can't judge
    assert c.violations({"depth": 20}, defaults=PROFILE.knobs) == ["lr<0.1 is ruled out"]   # default lr .05 is in the region


def test_population_step_dedups_on_effective_config(tmp_path):
    llm = MockLLM(jsons=[_proposal({"lr": 0.1}), _proposal({"lr": 0.1, "depth": 18})])   # same effective config
    orch = Orchestrator(llm, PROFILE, ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "r")
    produced = orch.population_step(2, tiers=("smoke",))
    assert len(produced) == 1


# 3. a schema-invalid reply must be retried, never fed to postcheck
def test_postcheck_not_run_on_schema_invalid_output():
    class Strict(Agent):
        name = "strict"
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"items": {"type": "array", "items": {"type": "object"}}}, "required": ["items"]}

        def postcheck(self, output, ctx):
            return [] if all(i.get("k") for i in output["items"]) else ["k required"]   # assumes objects

        def build_prompt(self, ctx):
            return "x"

    out = Strict(MockLLM(jsons=[{"items": ["not an object"]}, {"items": [{"k": 1}]}])).run({})
    assert out == {"items": [{"k": 1}]}


# 4. the governor folds EVERY kept entry of a round into the frontier
def test_governor_frontier_sees_every_kept_entry():
    g = Governor(dry_patience=2)
    improved = any([g.update_frontier(s, "minimize") for s in (5.0, 3.0)])
    assert improved and g._best == 3.0
    assert not g.update_frontier(4.0, "minimize")                                # 4.0 is not an improvement on 3.0


# 5. retrieve() honours max_total even when no source carries citations
def test_retrieve_caps_uncited_only_corpora():
    feed = "<feed xmlns='http://www.w3.org/2005/Atom'>" + "".join(
        f"<entry><id>http://arxiv.org/abs/2401.{i:05d}v1</id><title>t{i}</title><summary>s</summary></entry>"
        for i in range(8)) + "</feed>"
    papers = retrieve([ArxivClient(fetcher=lambda q, n: feed)], ["a", "b"], per_query=8, max_total=5)
    assert len(papers) == 5


# 6. the session corpus is version-insensitive across review rounds
def test_session_corpus_dedups_arxiv_versions(tmp_path):
    from scholarloop.ideation.session import ResearchSession
    q = {"queries": ["x"], "rationale": "r"}
    rev = {"overview": "o", "themes": [{"name": "t", "summary": "s", "papers": []}],
           "paper_notes": [{"id": "arXiv:1512.03385", "contribution": "c", "limitation": "l"}],
           "gaps": [{"gap": "g", "why_open": "w", "sources": []}], "trends": []}

    class Src:
        name = "fake"
        def __init__(self, pid): self.pid = pid
        def search(self, topic, n): return [Paper("ResNet", "s", self.pid, "u")]

    s = ResearchSession(MockLLM(jsons=[q, rev, q, rev]), PROFILE, tmp_path / "s", topic="t", sources=[Src("arXiv:1512.03385")])
    s.review()
    s.sources = [Src("arXiv:1512.03385v1")]                                      # the same paper, versioned id
    r = s.review(feedback="more")
    assert [p.arxiv_id for p in r.papers] == ["arXiv:1512.03385"]


# 7. a profile with no baseline must not crash the write-up
def test_grounded_registry_ignores_none_extra(tmp_path):
    reg = grounded_registry([], tmp_path, extra=[None])
    assert reg.measurements == {}


# 8. the ledger creates its directory
def test_ledger_append_creates_parent_dir(tmp_path):
    Ledger(tmp_path / "runs" / "x" / "ledger.jsonl").append(_entry("e1", {"lr": 0.1}, 5.0))
    assert (tmp_path / "runs" / "x" / "ledger.jsonl").exists()


# 9. a degraded (single-seed) verify never climbs, even with no gate
def test_degraded_verify_never_promotes_without_gate(tmp_path):
    orch = Orchestrator(MockLLM(), PROFILE, ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "r")
    e = LedgerEntry(id="v", domain=PROFILE.name, hypothesis=Hypothesis(claim="c", source="s"), metric_name="m",
                    fidelity=["verify"], metric={"verify": 3.0, "seeds": [3.0], "seeds_failed": 2}, verdict="kept")
    assert orch._promote(e, "verify", None) is False
    e.metric["seeds"] = [3.0, 3.1, 2.9]
    assert orch._promote(e, "verify", None) is True


# 11. step() shows the advisor the run that just finished
def test_step_advisor_sees_latest_verdict(tmp_path):
    from scholarloop.experiment.advisor import Advisor
    adv_llm = MockLLM(jsons=[{"decision": "proceed", "rationale": "ok"}])
    cfg = {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}
    orch = Orchestrator(MockLLM(jsons=[_proposal(cfg)]), PROFILE, advisor=Advisor(adv_llm),
                        ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "r")
    orch.step()
    assert "['kept']" in adv_llm.calls[0]["prompt"]


# 2. the profile schema ships with the package; engines resolve from the profile's location
def test_schema_is_package_data_and_root_resolves_from_profile():
    from scholarloop.core import profile as pm
    assert pm._SCHEMA_PATH.parent.name == "core" and pm._SCHEMA_PATH.exists()
    assert PROFILE.root == ROOT


# ---- second review pass ---------------------------------------------------------------------

# 12. scientific-notation tokens are grounded at their real precision, not as integers
def test_number_grounding_is_not_fooled_by_exponent_notation(tmp_path):
    from scholarloop.core.registry import VerifiedRegistry
    reg = VerifiedRegistry(tmp_path / "r.json", "e")
    reg.capture("val_err", 3.8222, seeds=[3.7, 3.9])
    reg.capture("lr", 0.1)
    assert reg.audit_text("lr 3e-4 and wd 1e-3") == ["3e-4", "1e-3"]     # fabricated
    reg.capture("weight_decay", 0.0003)
    assert reg.audit_text("wd 3e-4") == []                                 # real


# 13. Profile.root follows the layout, not a fixed depth
def test_profile_root_outside_a_profiles_folder(tmp_path):
    src = ROOT / "profiles" / "digits-mlp.yaml"
    flat = tmp_path / "proj"
    (flat / "engines").mkdir(parents=True)
    (flat / "my-profile.yaml").write_text(src.read_text())
    p = load_profile(flat / "my-profile.yaml")
    (flat / p.train_entrypoint).parent.mkdir(parents=True, exist_ok=True)
    (flat / p.train_entrypoint).write_text("")
    assert p.root == flat                                                  # found the engine next to the profile
    assert PROFILE.root == ROOT                                            # the repo layout still works


# 14. a human-chosen idea keeps steering after the Advisor says "proceed"
def test_adopted_guidance_survives_proceed(tmp_path):
    orch = Orchestrator(MockLLM(), PROFILE, ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "r")
    orch.adopt({"topic": "Skip MLP", "guidance": "Test this idea: add skips. Experiment plan: depth=20"})
    orch.last_advice = {"decision": "proceed", "rationale": "keep going"}
    orch._apply_advice([])
    assert orch.guidance.startswith("Test this idea")
    orch.last_advice = {"decision": "pivot", "rationale": "exhausted"}
    orch._apply_advice([])
    assert "Test this idea" not in orch.guidance and orch._base_guidance == ""


# 15. an unpriced model cannot silently disable the budget ceiling
def test_governor_warns_when_budget_cannot_be_metered(tmp_path, capsys):
    orch = Orchestrator(MockLLM(), PROFILE, ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "r")
    orch.run(0, governor=Governor(max_cost=1.0, max_rounds=1))
    assert "cannot be enforced" in capsys.readouterr().err
    orch.run(0, governor=Governor(max_rounds=1))
    assert orch.stop_reason is None or "round cap" in orch.stop_reason


# 16. the prose lint does not reject legitimate English
def test_lint_prose_spares_surnames_and_the_underscore_character():
    from scholarloop.writing.session import lint_prose
    assert lint_prose("Foster et al. report 3.8 error. Variable names use underscores.") == []
    assert lint_prose("This result underscores the need for more seeds.")        # the machine sense
    assert lint_prose("Crucially, the gap closes.") and lint_prose("It fosters a deeper view.")


# 17. feedback after the draft keeps the outline and re-derives from the key points
def test_writing_feedback_after_draft_keeps_outline(monkeypatch, tmp_path):
    from scholarloop.writing import session as w
    calls = []

    class Fake:
        stage = "drafted"
        out = tmp_path
        def _require_release(self): pass
        def outline(self, feedback=""): calls.append(("outline", feedback))
        def key_points(self, feedback=""): calls.append(("key_points", feedback))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(w, "WritingSession", lambda *a, **k: Fake())
    monkeypatch.setattr("scholarloop.core.llm.AnthropicLLM", lambda *a, **k: None)
    monkeypatch.setattr("scholarloop.core.profile.load_profile", lambda p: None)
    w._main(["--profile", "p", "--ledger", "l", "--registry", "r", "--out", str(tmp_path), "--feedback", "tighten"])
    assert calls == [("key_points", "tighten")]
