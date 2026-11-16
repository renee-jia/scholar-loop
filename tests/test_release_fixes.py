"""Regressions for defects found in the pre-release review."""
import json
import multiprocessing
import os

import pytest

from scholarloop.core.ledger import Ledger
from scholarloop.core.locking import exclusive_lock
from scholarloop.core.registry import VerifiedRegistry, _number_tokens
from scholarloop.reliability import classify


@pytest.mark.parametrize("text", [
    "We reach 4.1%.", 'an error of "4.1".', "**4.1**", "`4.1`", "|4.1|", "~4.1", "≈4.1", "$4.1",
    "a 4.1x gain", "4.1pp lower", "4.1†"])
def test_a_number_glued_to_punctuation_or_a_unit_is_still_checked(text):
    assert list(_number_tokens(text)) == ["4.1"]


def test_joined_numbers_are_each_checked_and_identifiers_are_not():
    assert list(_number_tokens("4.1±0.2")) == ["4.1", "0.2"]
    assert list(_number_tokens("3.10–3.20")) == ["3.10", "3.20"]
    assert list(_number_tokens("−4.1")) == ["-4.1"]
    assert list(_number_tokens("see https://arxiv.org/abs/1609.04836 and top-1 and arXiv:1609.04836")) == []


def test_either_rounding_convention_grounds_a_half_way_value():
    reg = VerifiedRegistry(":t:")
    reg.capture("m", 0.8125)
    assert reg.audit_text("0.812 and 0.813") == []
    assert reg.audit_text("0.814") == ["0.814"]


def test_exact_integers_refuse_a_rounded_measurement():
    reg = VerifiedRegistry(":t:")
    reg.capture("m", 7.25)
    assert reg.audit_text("about 7 percent") == []          # the general registry rounds
    reg.exact_integers = True
    assert reg.audit_text("about 7 percent") == ["7"]
    reg.capture("n", 20.0)
    assert reg.audit_text("over 20 seeds") == []


def test_ledger_skips_a_line_that_is_json_but_not_an_entry(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_text('[1, 2]\n"text"\n{"id": "x", "hypothesis": null}\n', encoding="utf-8")
    assert list(Ledger(path).read_all()) == []


def test_reviewed_mode_rejections_are_filed_under_their_guard():
    assert classify("numbers ['3.10'] are not in the approved statements ['c'] that this sentence cites") == "number"
    assert classify("c: numbers lack explicitly bound measurements: ['3.10']") == "number"
    assert classify("this sentence cites no approved statement: 'x'") == "citation"
    assert classify("sources ['arXiv:1'] are not evidence of the approved statements ['c']") == "citation"


def test_a_waiting_lock_is_acquired_once_released_and_a_plain_one_still_refuses(tmp_path):
    lock = tmp_path / "x.lock"
    with exclusive_lock(lock):
        with pytest.raises(BlockingIOError):
            with exclusive_lock(lock):
                pass
        with pytest.raises(BlockingIOError):
            with exclusive_lock(lock, wait=0.05):
                pass
    with exclusive_lock(lock, wait=1.0):
        pass
    with pytest.raises(KeyError):                          # the caller's own error is not retried
        with exclusive_lock(lock, wait=1.0):
            raise KeyError("mine")


def _spend_many(args):
    path, n = args
    os.environ.update(SCHOLARLOOP_SPEND_CAP="1000000", SCHOLARLOOP_SPEND_FILE=path)
    from scholarloop.core.llm import _spend
    from scholarloop.experiment.governor import MODEL_PRICES
    model = next(iter(MODEL_PRICES))
    for _ in range(n):
        _spend(model, {"input_tokens": 1000, "output_tokens": 0})


def test_spend_is_counted_once_per_request_across_processes(tmp_path):
    path = str(tmp_path / "spend.json")
    with multiprocessing.get_context("spawn").Pool(3) as pool:
        pool.map(_spend_many, [(path, 20)] * 3)
    assert json.loads(open(path, encoding="utf-8").read())["requests"] == 60


def test_process_report_counts_one_call_processes_and_survives_a_cut_tail(tmp_path):
    from scholarloop.evaluation import process_report
    from scholarloop.experiment.governor import MODEL_PRICES
    model = next(iter(MODEL_PRICES))
    row = lambda event, calls: {"event": event, "call_id": f"c{calls}{event}", "model": model, "status": "accepted",
                                "client_usage": {"calls": calls, "input_tokens": 1000 * calls, "output_tokens": 0}}
    rows = [row("started", 0), row("finished", 1), row("started", 0), row("finished", 1)]
    (tmp_path / "x.agents.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + '\n{"event": "fini', encoding="utf-8")
    assert process_report(tmp_path)["stages"][0]["input_tokens"] == 2000
