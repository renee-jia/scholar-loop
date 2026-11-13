"""An interrupted final-test attempt must never silently reopen search or writing."""
from dataclasses import replace

import pytest

from scholarloop.core.ledger import Hypothesis, Ledger
from scholarloop.experiment import runner
from scholarloop.experiment.test_boundary import access_records, require_unopened_test
from scholarloop.review.dossier import snapshot, build_dossier
from tests.experiment.test_holdout import STUB


def frozen_fixture(tmp_path):
    import copy
    from tests.experiment.test_agenda import SPEC
    from scholarloop.experiment.spec import frozen_context
    from scholarloop.review.contracts import digest, atomic_json
    spec = {**copy.deepcopy(SPEC), "purpose": "confirmation", "fidelity": "full", "holdout": True,
            "seeds": [0, 1, 2, 3, 4], "budget_seconds": 1000000,
            "conditions": [{"name": n, "description": "fixture", "config": [], "edits": []}
                           for n in ("first", "second", "interrupted", "retry")], "reference": "first"}
    envelope = {"spec": spec, "frozen": frozen_context(STUB, spec),
                "issues": [{"id": i, "status": "open", "action": "experiment"} for i in spec["issue_ids"]]}
    pid = digest(envelope)
    atomic_json(tmp_path / "registry" / "protocols" / f"{pid}.json", envelope)
    return pid


def execute(tmp_path, eid, **kwargs):
    if kwargs.get("protocol") == {"id": "frozen"}:
        pid = frozen_fixture(tmp_path)
        kwargs["protocol"] = {"id": pid, "condition": eid}
        eid = f"protocol_{pid[:24]}_{eid}"
    return runner.run_experiment(STUB, "full", eid, Hypothesis("probe", "fixture"),
                                ledger_path=tmp_path / "ledger.jsonl",
                                registry_dir=tmp_path / "registry", **kwargs)


def test_interrupt_after_reservation_blocks_retry_and_review(tmp_path, monkeypatch):
    execute(tmp_path, "prior")
    def interrupt(*args, **kwargs):
        assert len(access_records(tmp_path / "ledger.jsonl")) == 1
        raise KeyboardInterrupt()
    monkeypatch.setattr(runner, "_run_one_seed", interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, "interrupted", purpose="confirmation", holdout_override=True,
                protocol={"id": "frozen"})
    with pytest.raises(ValueError, match="Unresolved"):
        execute(tmp_path, "retry", purpose="confirmation", holdout_override=True,
                protocol={"id": "frozen"})
    d = build_dossier(snapshot(STUB, tmp_path / "ledger.jsonl", tmp_path / "registry"))
    assert any("Unresolved final test access" in c for c in d["checks"])
    assert len(list(Ledger(tmp_path / "ledger.jsonl").read_all())) == 1


def test_completed_protocol_can_continue_but_not_repeat_run(tmp_path):
    opts = dict(purpose="confirmation", holdout_override=True, protocol={"id": "frozen"})
    execute(tmp_path, "first", **opts)
    execute(tmp_path, "second", **opts)
    assert len(access_records(tmp_path / "ledger.jsonl")) == 2
    d = build_dossier(snapshot(STUB, tmp_path / "ledger.jsonl", tmp_path / "registry"))
    assert not any("Unresolved final test access" in c for c in d["checks"])
    with pytest.raises(ValueError, match="already executed"):
        execute(tmp_path, "second", **opts)
    with pytest.raises(ValueError, match="Final test access"):
        execute(tmp_path, "explore")


def test_partial_access_record_fails_closed(tmp_path):
    from scholarloop.experiment.test_boundary import access_directory
    directory = access_directory(tmp_path / "ledger.jsonl")
    directory.mkdir()
    (directory / "interrupted.json").write_text('{"version":')
    with pytest.raises(ValueError, match="Uncertain"):
        require_unopened_test(tmp_path / "ledger.jsonl")


def test_frozen_confirmation_agenda_resumes_with_access_receipts(tmp_path):
    from scholarloop.core.llm import MockLLM
    from scholarloop.experiment.agenda import ExperimentAgenda
    from tests.experiment.test_agenda import SPEC
    from tests.test_scientific_review import setup, session, PASS, REVISE, ARGUMENT
    paths = setup(tmp_path)
    review = session(paths, MockLLM(jsons=[PASS, REVISE, PASS, PASS]))
    review.profile = replace(review.profile, holdout=True, cv_folds=5)
    review.review(argument=ARGUMENT)
    agenda = ExperimentAgenda(review)
    spec = {**SPEC, "purpose": "confirmation", "fidelity": "full", "holdout": True,
            "budget_seconds": 200000}
    plan = agenda.plan(spec=spec)
    first = agenda.execute(plan["protocol_id"], author="Researcher", max_conditions=1)
    assert first["status"] == "PAUSED"
    completed = ExperimentAgenda(review).execute(plan["protocol_id"], author="Researcher")
    assert completed["status"] == "EXECUTED"
    assert len(access_records(review.ledger_path)) == 2
    assert not build_dossier(review.inputs())["checks"]
