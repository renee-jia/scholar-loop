"""Public runner calls cannot bypass a frozen confirmation condition."""
import json

import pytest

from scholarloop.core.ledger import Hypothesis
from scholarloop.experiment.runner import run_experiment
from tests.experiment.test_holdout import STUB
from tests.experiment.test_test_access_recovery import frozen_fixture, execute


@pytest.mark.parametrize("change", ["invented_id", "config", "source", "seeds", "holdout", "condition", "run_id", "cache", "tamper"])
def test_invalid_confirmation_is_rejected_before_execution(tmp_path, monkeypatch, change):
    pid = frozen_fixture(tmp_path)
    options = dict(purpose="confirmation", holdout_override=True,
                   protocol={"id": pid, "condition": "second"}, seeds_override=[0, 1, 2, 3, 4])
    eid = f"protocol_{pid[:24]}_second"
    if change == "invented_id":
        options["protocol"]["id"] = "f" * 64
    elif change == "config":
        options["config_override"] = {"lr": 0.9}
    elif change == "source":
        options["edits"] = [{"path": STUB.train_entrypoint, "content": "raise RuntimeError('changed')"}]
    elif change == "seeds":
        options["seeds_override"] = [7, 8]
    elif change == "holdout":
        options["holdout_override"] = False
    elif change == "condition":
        options["protocol"]["condition"] = "unplanned"
    elif change == "run_id":
        eid = "new_name_same_condition"
    elif change == "cache":
        options["artifact_cache"] = tmp_path / "cache"
    else:
        path = tmp_path / "registry" / "protocols" / f"{pid}.json"
        data = json.loads(path.read_text())
        data["spec"]["seeds"] = [7, 8]
        path.write_text(json.dumps(data))
    def forbidden(*args, **kwargs):
        raise AssertionError("No training or scoring may run")
    monkeypatch.setattr("scholarloop.experiment.runner._run_one_seed", forbidden)
    with pytest.raises(ValueError):
        run_experiment(STUB, "full", eid, Hypothesis("fixture", "fixture"),
                       ledger_path=tmp_path / "ledger.jsonl", registry_dir=tmp_path / "registry", **options)
    assert not (tmp_path / "ledger.jsonl").exists()
    assert not (tmp_path / "ledger.jsonl.test-access").exists()


def test_same_protocol_id_does_not_authorize_adaptive_new_candidate(tmp_path):
    execute(tmp_path, "first", purpose="confirmation", holdout_override=True, protocol={"id": "frozen"})
    with pytest.raises(ValueError, match="differs from frozen"):
        execute(tmp_path, "second", purpose="confirmation", holdout_override=True,
                protocol={"id": "frozen"}, config_override={"lr": 0.9})
    execute(tmp_path, "second", purpose="confirmation", holdout_override=True, protocol={"id": "frozen"})
