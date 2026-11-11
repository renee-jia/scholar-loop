"""L5 Writer + Reviewer: drafting, number-grounding gate, and review."""

from scholarloop.core.ledger import Hypothesis, LedgerEntry
from scholarloop.core.llm import MockLLM
from scholarloop.writing.paper import (
    PaperPipeline,
    Writer,
    audit_draft,
    gather_findings,
    grounded_registry,
)


def _kept(eid, score, config):
    return LedgerEntry(id=eid, domain="image-classification",
                       hypothesis=Hypothesis("cosine helps", "arXiv:1608.03983"),
                       metric_name="val_top1_err", config=config, fidelity=["verify"],
                       metric={"verify": score, "seeds": [score]}, verdict="kept", ts=1.0)


ENTRIES = [_kept("exp_0003", 3.33, {"lr": 0.2, "depth": 2}),
           _kept("exp_0001", 8.44, {"lr": 0.1, "depth": 1})]


def test_gather_findings_only_kept():
    discarded = _kept("exp_x", 9.0, {"lr": 0.9})
    discarded.verdict = "discarded"
    assert {f["id"] for f in gather_findings(ENTRIES + [discarded])} == {"exp_0003", "exp_0001"}


def test_grounding_passes_real_numbers_and_flags_fabrication(tmp_path):
    reg = grounded_registry(ENTRIES, tmp_path)        # no registry files -> config+score grounding
    # a draft citing only real numbers (score 3.33, config lr 0.2/depth 2) is fully grounded
    good = {"title": "T", "abstract": "We reach 3.33 error with lr 0.2 and depth 2.", "sections": []}
    assert audit_draft(good, reg) == []
    # a fabricated SOTA number is caught
    bad = {"title": "T", "abstract": "We reach 1.01 error, a new record.", "sections": []}
    assert audit_draft(bad, reg) == ["1.01"]


def test_pipeline_drafts_grounds_and_reviews(tmp_path):
    writer_llm = MockLLM(jsons=[{
        "title": "Cosine schedules for digit MLPs",
        "abstract": "Our best model reaches 3.33 val error.",
        "sections": [{"heading": "Results", "body": "lr 0.2 with depth 2 gives 3.33 error."}],
    }])
    reviewer_llm = MockLLM(jsons=[{
        "summary": "solid", "strengths": ["grounded numbers"], "weaknesses": ["small dataset"],
        "score": 6, "recommendation": "weak_accept"}])
    out = PaperPipeline(writer_llm, reviewer_llm, registry_dir=tmp_path, demo=True).run(ENTRIES)
    assert out["grounded"] is True and out["ungrounded"] == []
    assert out["review"]["recommendation"] == "weak_accept"


def test_reviewer_retries_on_out_of_range_score(tmp_path):
    writer_llm = MockLLM(jsons=[{"title": "T", "abstract": "3.33 error", "sections": []}])
    reviewer_llm = MockLLM(jsons=[
        {"summary": "x", "strengths": [], "weaknesses": [], "score": 99, "recommendation": "reject"},
        {"summary": "x", "strengths": [], "weaknesses": [], "score": 4, "recommendation": "reject"}])
    out = PaperPipeline(writer_llm, reviewer_llm, registry_dir=tmp_path, demo=True).run(ENTRIES)
    assert out["review"]["score"] == 4
