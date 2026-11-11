"""Step-by-step writing: lint, evidence bank grounding, the discussion loop, the drafter's guards,
and the WritingSession state machine with a scripted human."""

import json
from pathlib import Path

import pytest

from scholarloop.core.agent import AgentError
from scholarloop.core.ledger import Hypothesis, Ledger, LedgerEntry
from scholarloop.ideation.litreview import LitReview
from scholarloop.ideation.litscout import Paper
from scholarloop.core.llm import MockLLM
from scholarloop.core.profile import load_profile
from scholarloop.core.registry import VerifiedRegistry
from scholarloop.writing.session import (
    EvidenceBank, KeyPointer, Outliner, Panelist, SectionDrafter, WritingSession, discuss,
    lint_prose, load_skill, prose_warnings, render_outline_block,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_profile(ROOT / "profiles" / "image-classification.yaml")    # val_top1_err, minimize, baseline 4.9


def _entries(tmp_path):
    led = tmp_path / "ledger.jsonl"
    reg = tmp_path / "registry"
    rows = [("exp_0001", "smoke", 4.2222, [4.2222], "kept", {"lr": 0.1, "depth": 20}),
            ("exp_0002", "smoke", 7.3333, [7.3333], "discarded", {"lr": 0.5, "depth": 2}),
            ("exp_0003", "full", 3.8222, [3.8, 3.9, 3.7, 3.85, 3.86], "kept", {"lr": 0.1, "depth": 20})]
    for eid, fid, score, seeds, verdict, cfg in rows:
        Ledger(led).append(LedgerEntry(id=eid, domain=PROFILE.name, metric_name="val_top1_err",
                                       hypothesis=Hypothesis(claim="warmup + wd", source="arXiv:1608.03983v1"),
                                       config=cfg, fidelity=[fid], metric={fid: score, "seeds": seeds},
                                       verdict=verdict, registry_id=eid))
        r = VerifiedRegistry(reg / f"{eid}.json", exp_id=eid)
        r.capture(f"val_top1_err.{fid}", score, seeds=seeds, std=0.06 if len(seeds) > 1 else 0.0)
        r.save()
    Ledger(led).append(LedgerEntry(id="exp_0004", domain=PROFILE.name, metric_name="val_top1_err",
                                   hypothesis=Hypothesis(claim="depth 8", source="arXiv:1512.03385v1"),
                                   fidelity=["smoke"], metric={"error": "seed 0: train exit 1"}, verdict="killed"))
    return led, reg


REVIEW = LitReview(topic="t", queries=["q"], overview="o", themes=[], trends=[],
                   papers=[Paper("SGDR", "cosine", "arXiv:1608.03983v1", "u"), Paper("ResNet", "skips", "arXiv:1512.03385v1", "u")],
                   paper_notes=[{"id": "arXiv:1608.03983v1", "contribution": "cosine annealing", "limitation": "l"}],
                   gaps=[{"gap": "g", "why_open": "w", "sources": ["arXiv:1608.03983v1"]}])

OUTLINE = {"title": "Warmup on digits", "question": "Does warmup lower MLP error on digits?",
           "nugget": "A short warmup matters more than depth.",
           "contributions": ["error 5.0 to 3.8222 over five seeds [exp_0003]"],
           "sections": [{"heading": "Abstract", "purpose": "five sentences", "evidence": ["exp_0003", "baseline"]},
                        {"heading": "Setup", "purpose": "reproducibility", "evidence": ["exp_0003"]},
                        {"heading": "Results", "purpose": "the climb", "evidence": ["exp_0001", "exp_0002", "exp_0003"]},
                        {"heading": "Limitations", "purpose": "scope", "evidence": []}]}
ACCEPT = {"verdict": "accept", "issues": []}
REVISE = {"verdict": "revise", "issues": [{"target": "Results", "issue": "no spread", "suggestion": "report std"}]}


def _kp(extra_claim="full run reaches 3.8222 error over 5 seeds"):
    return {"sections": [
        {"heading": "Abstract", "points": [{"claim": "We test warmup on digits.", "evidence": [], "role": "context"},
                                           {"claim": extra_claim, "evidence": ["exp_0003"], "role": "result"}]},
        {"heading": "Setup", "points": [{"claim": "MLP depth 20, lr 0.1, 120 second budget.", "evidence": ["exp_0003"], "role": "method"}]},
        {"heading": "Results", "points": [{"claim": "Smoke 4.2222 then full 3.8222, beating the 4.9 baseline.", "evidence": ["exp_0001", "exp_0003", "baseline"], "role": "result"},
                                          {"claim": "lr 0.5 discarded at 7.3333.", "evidence": ["exp_0002"], "role": "comparison"}]},
        {"heading": "Limitations", "points": [{"claim": "One dataset, one budget.", "evidence": [], "role": "limitation"}]}]}


def _body(heading):
    return {"Abstract": "We test whether a short warmup lowers MLP error on digits. The full run reaches 3.8222 error over 5 seeds [exp_0003], below the 4.9 baseline.",
            "Setup": "We train an MLP of depth 20 with lr 0.1 under a 120 second budget [exp_0003]. The frozen scorer computes val_top1_err.",
            "Results": "The smoke run scores 4.2222 and the full run 3.8222, below the 4.9 baseline [exp_0001, exp_0003]. A lr of 0.5 was discarded at 7.3333 [exp_0002].",
            "Limitations": "We use one dataset and one budget. The split is reconstructable from public data."}[heading]


_REVIEW = lambda score, rec: {"summary": "s", "strengths": ["5-seed confirmation"], "weaknesses": ["one dataset"],
                              "questions": ["more seeds?"], "score": score, "recommendation": rec}


# ---- lint ----------------------------------------------------------------------------------
def test_lint_catches_machine_tells_and_passes_clean_prose():
    bad = "We delve into a crucial topic — it is worth noting. Moreover, A. Furthermore, B. It is not X, but Y."
    errs = lint_prose(bad)
    assert any("em-dash" in e for e in errs) and any("delve" in e for e in errs)
    assert any("it is worth noting" in e for e in errs) and any("contrast" in e for e in errs)
    assert any("connectives" in e for e in errs)
    assert lint_prose("The full run reaches 3.82% error over five seeds, below the 4.9% baseline.") == []
    assert lint_prose("Moreover, A. The result B. Furthermore, C.") == []          # not consecutive


def test_prose_warnings_are_soft_signals():
    w = prose_warnings("The gain is significant. " + "word " * 45 + ". A novel idea and a novel test.")
    assert any("significant" in x for x in w) and any("over 40 words" in x for x in w) and any("novel" in x for x in w)


def test_skills_ship_with_the_package():
    assert "one-sentence question" in load_skill("narrative").lower()
    assert "em-dash" in load_skill("style").lower()


# ---- evidence bank -------------------------------------------------------------------------
def test_evidence_bank_grounds_runs_papers_and_baseline(tmp_path):
    led, reg = _entries(tmp_path)
    bank = EvidenceBank.build(list(Ledger(led).read_all()), reg, PROFILE, review=REVIEW)
    assert bank.ids() >= {"exp_0001", "exp_0002", "exp_0003", "exp_0004", "arXiv:1608.03983v1", "baseline"}
    assert bank.unknown(["exp_0009", "arXiv:0000.0"]) == ["arXiv:0000.0", "exp_0009"]
    assert bank.ungrounded("error 3.8222 beats 4.9 (std 0.06) at lr 0.1") == []     # score, baseline, std, config
    assert bank.ungrounded("error 3.5") == ["3.5"]
    assert "killed" in bank.describe("exp_0004") and "± 0.06" in bank.describe("exp_0003")
    assert "cosine annealing" in bank.describe("arXiv:1608.03983v1")


# ---- agents' deterministic guards ----------------------------------------------------------
def test_outliner_requires_abstract_limitations_and_known_evidence(tmp_path):
    led, reg = _entries(tmp_path)
    bank = EvidenceBank.build(list(Ledger(led).read_all()), reg, PROFILE)
    bad = {**OUTLINE, "sections": [{"heading": "Intro", "purpose": "p", "evidence": ["exp_0042"]},
                                   {"heading": "Results", "purpose": "p", "evidence": []},
                                   {"heading": "End", "purpose": "p", "evidence": []}]}
    llm = MockLLM(jsons=[bad, OUTLINE])
    out = Outliner(llm, bank).run({"review": "", "previous": None, "issues": None, "feedback": ""})
    assert out == OUTLINE
    retry = llm.calls[1]["prompt"]
    assert "Abstract" in retry and "Limitations" in retry and "exp_0042" in retry


def test_key_pointer_rejects_unknown_evidence_and_ungrounded_numbers(tmp_path):
    led, reg = _entries(tmp_path)
    bank = EvidenceBank.build(list(Ledger(led).read_all()), reg, PROFILE)
    bad = _kp(extra_claim="full run reaches 3.5 error")                         # 3.5 was never measured
    bad["sections"][1]["points"][0]["evidence"] = ["exp_0099"]
    bad["sections"][2]["points"][0]["evidence"] = []                             # a result claim with no evidence
    llm = MockLLM(jsons=[bad, _kp()])
    out = KeyPointer(llm, bank).run({"outline": OUTLINE, "previous": None, "issues": None, "feedback": ""})
    assert out == _kp()
    retry = llm.calls[1]["prompt"]
    assert "3.5" in retry and "exp_0099" in retry and "must cite evidence" in retry


def test_key_pointer_requires_outline_headings_in_order(tmp_path):
    led, reg = _entries(tmp_path)
    bank = EvidenceBank.build(list(Ledger(led).read_all()), reg, PROFILE)
    shuffled = {"sections": list(reversed(_kp()["sections"]))}
    with pytest.raises(AgentError):
        KeyPointer(MockLLM(jsons=[shuffled] * 3), bank).run({"outline": OUTLINE, "previous": None, "issues": None, "feedback": ""})


def test_drafter_retries_on_lint_numbers_and_citations(tmp_path):
    led, reg = _entries(tmp_path)
    bank = EvidenceBank.build(list(Ledger(led).read_all()), reg, PROFILE)
    bad = {"body": "We leverage warmup — error drops to 3.5 [exp_0077]."}
    good = {"body": _body("Results")}
    llm = MockLLM(jsons=[bad, good])
    ctx = {"outline": OUTLINE, "heading": "Results", "purpose": "p", "points": _kp()["sections"][2]["points"], "so_far": ""}
    assert SectionDrafter(llm, bank).run(ctx) == good
    retry = llm.calls[1]["prompt"]
    assert "leverage" in retry and "3.5" in retry and "exp_0077" in retry and "em-dash" in retry


# ---- discussion loop -----------------------------------------------------------------------
def test_discuss_revises_until_panel_accepts_and_caps_rounds():
    seen = []

    def propose(issues):
        seen.append(issues)
        return {"v": len(seen)}

    panel = [Panelist(MockLLM(jsons=[REVISE, ACCEPT]), "Skeptic"), Panelist(MockLLM(jsons=[ACCEPT, ACCEPT]), "Editor")]
    final, transcript = discuss(propose, panel, "outline", json.dumps, "ev", max_rounds=3)
    assert final == {"v": 2} and seen[1] == REVISE["issues"]                  # revised once against the issue
    assert [t["accepted"] for t in transcript] == [False, True]

    seen.clear()
    panel = [Panelist(MockLLM(jsons=[REVISE] * 5), "Skeptic")]
    final, transcript = discuss(propose, panel, "outline", json.dumps, "ev", max_rounds=2)
    assert final == {"v": 3} and len(transcript) == 3 and not transcript[-1]["accepted"]   # round cap, never accepted


# ---- the session ---------------------------------------------------------------------------
def _session_llm(outline_rounds=(ACCEPT,), kp_rounds=(ACCEPT,), feedback_outline=False):
    # one shared MockLLM; the call order is deterministic: outliner, 3 panelists, [outliner, 3 panelists]...,
    # key_pointer, 3 panelists, drafter x4, reviewers x3
    jsons = []
    jsons.append(OUTLINE)
    for i, verdict in enumerate(outline_rounds):
        jsons += [verdict, ACCEPT, ACCEPT]
        if verdict is REVISE and i < len(outline_rounds) - 1:
            jsons.append({**OUTLINE, "nugget": "revised nugget"})
    if feedback_outline:
        jsons += [{**OUTLINE, "title": "Warmup on digits (revised)"}, ACCEPT, ACCEPT, ACCEPT]
    jsons.append(_kp())
    for verdict in kp_rounds:
        jsons += [verdict, ACCEPT, ACCEPT]
    jsons += [{"body": _body(h)} for h in ("Abstract", "Setup", "Results", "Limitations")]
    jsons += [_REVIEW(6, "weak_accept"), _REVIEW(4, "weak_reject"), _REVIEW(7, "accept")]
    return MockLLM(jsons=jsons)


def test_writing_session_end_to_end_with_scripted_author(tmp_path):
    led, reg = _entries(tmp_path)
    rd = tmp_path / "research"
    rd.mkdir()
    (rd / "lit_review.json").write_text(json.dumps(REVIEW.to_dict()))
    (rd / "handoff.json").write_text(json.dumps({"idea": {"title": "Warmup first", "description": "d", "hypothesis": {"claim": "c"}}}))
    llm = _session_llm(outline_rounds=(REVISE, ACCEPT), feedback_outline=True)
    s = WritingSession(llm, PROFILE, tmp_path / "w", ledger_path=led, registry_dir=reg, research_dir=rd, demo=True)
    answers = iter(["r make the title say revised", "a", "a"])
    log = []
    r = s.interactive(input_fn=lambda _: next(answers), print_fn=log.append)

    assert s.stage == "reviewed"
    assert s.state.outline["title"] == "Warmup on digits (revised)"
    assert len(s.state.outline_discussion) == 1                       # the re-outline after feedback: accepted first round
    assert s.state.feedback[0]["text"] == "make the title say revised"
    assert [sec["heading"] for sec in s.state.sections] == ["Abstract", "Setup", "Results", "Limitations"]
    assert r["mean_score"] == 5.7 and r["recommendation"] == "weak_accept"   # median reviewer
    assert "one dataset" in r["weaknesses"] and len(r["weaknesses"]) == 1   # de-duplicated across reviewers
    for f in ("outline.md", "key_points.md", "paper.md", "review.md", "writing.json"):
        assert (s.out / f).exists()
    paper = (s.out / "paper.md").read_text()
    assert "# Warmup on digits (revised)" in paper and "mean score 5.7/10" in paper
    outline_md = (s.out / "outline.md").read_text()
    assert "Author feedback applied" in outline_md and "Round 1" in outline_md
    # the drafter saw the author-approved points, the outliner saw the idea + lit review
    outliner_prompt = llm.calls[0]["prompt"]
    assert "Warmup first" in outliner_prompt and "cosine annealing" in outliner_prompt


def test_writing_session_persists_and_resumes(tmp_path):
    led, reg = _entries(tmp_path)
    s1 = WritingSession(MockLLM(jsons=[OUTLINE, ACCEPT, ACCEPT, ACCEPT]), PROFILE, tmp_path / "w", ledger_path=led, registry_dir=reg, demo=True)
    s1.outline()
    s2 = WritingSession(MockLLM(jsons=[_kp(), ACCEPT, ACCEPT, ACCEPT]), PROFILE, tmp_path / "w", ledger_path=led, registry_dir=reg, demo=True)
    assert s2.stage == "outlined" and s2.state.outline == OUTLINE
    s2.key_points()
    assert s2.stage == "keypointed" and (s2.out / "key_points.md").read_text().startswith("# Key points")


def test_draft_keeps_best_attempt_and_flags_when_lint_never_passes(tmp_path):
    led, reg = _entries(tmp_path)
    s = WritingSession(MockLLM(jsons=[OUTLINE, ACCEPT, ACCEPT, ACCEPT, _kp(), ACCEPT, ACCEPT, ACCEPT]),
                       PROFILE, tmp_path / "w", ledger_path=led, registry_dir=reg, panel_roles=("Skeptic", "Editor", "Methodologist"), demo=True)
    s.outline(); s.key_points()
    stubborn = {"body": "We leverage warmup. " + _body("Abstract")}
    s.drafter.llm = MockLLM(jsons=[stubborn] * 3 + [{"body": _body("Setup")}, {"body": _body("Results")}, {"body": _body("Limitations")}])
    sections = s.draft()
    assert s.stage == "drafted"
    assert sections[0]["body"].startswith("We leverage") and any("unresolved" in w for w in sections[0]["warnings"])
    assert sections[1]["warnings"] == [] or all("covered" in w for w in sections[1]["warnings"])
    assert "Drafting notes" in (s.out / "paper.md").read_text()


def test_render_outline_block_is_what_the_panel_sees():
    block = render_outline_block(OUTLINE)
    assert "Question: Does warmup" in block and "4. Limitations" in block
