"""Reflector: distills a lesson, skips when none, retries on out-of-range severity."""

from scholarloop.core.ledger import Hypothesis, LedgerEntry
from scholarloop.core.llm import MockLLM
from scholarloop.experiment.reflector import Reflector


def _entry():
    return LedgerEntry(
        id="exp_0002", domain="image-classification", hypothesis=Hypothesis("c", "arXiv:1"),
        metric_name="val_top1_err", config={"lr": 0.1}, fidelity=["smoke"],
        metric={"smoke": 5.0}, verdict="discarded", ts=123.0,
        prediction={"predicted": -0.4, "measured": 0.1, "calibration_error": 0.5})


def test_reflect_records_a_skill_tied_to_the_experiment():
    llm = MockLLM(jsons=[{"worth_recording": True, "category": "optimizer",
                          "severity": 0.8, "mitigation": "cosine schedule helps here"}])
    skill = Reflector(llm).reflect(_entry(), None)
    assert skill is not None
    assert skill.category == "optimizer"
    assert skill.source == "exp_0002"
    assert skill.ts == 123.0          # decay measured from when the experiment ran


def test_reflect_skips_when_no_lesson():
    llm = MockLLM(jsons=[{"worth_recording": False, "category": "", "severity": 0.0, "mitigation": ""}])
    assert Reflector(llm).reflect(_entry(), None) is None


def test_reflect_retries_on_out_of_range_severity():
    llm = MockLLM(jsons=[
        {"worth_recording": True, "category": "x", "severity": 1.5, "mitigation": "m"},   # rejected
        {"worth_recording": True, "category": "x", "severity": 0.5, "mitigation": "m"}])
    skill = Reflector(llm).reflect(_entry(), None)
    assert skill.severity == 0.5
