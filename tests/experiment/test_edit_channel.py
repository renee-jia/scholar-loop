"""Source-diff edit channel + the forbidden_edits guard wired into the run path (gap C)."""

from pathlib import Path

import pytest

from scholarloop.core.ledger import Hypothesis
from scholarloop.core.llm import MockLLM
from scholarloop.core.profile import load_profile
from scholarloop.experiment.reasoner import Reasoner
from scholarloop.experiment.runner import run_experiment

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_profile(ROOT / "profiles" / "image-classification.yaml")

# a full-file replacement of the stub train.py that hardcodes a near-ideal config
EDITED_TRAIN = '''import json, os
from pathlib import Path
def main():
    seed = int(os.environ.get("SCHOLARLOOP_SEED", "0"))
    Path(os.environ["SCHOLARLOOP_ARTIFACT"]).write_text(
        json.dumps({"config": {"lr": 0.1, "depth": 20, "weight_decay": 5e-4, "warmup": 5}, "seed": seed}))
if __name__ == "__main__":
    main()
'''


def _proposal(edits=None, config=None, source="arXiv:1"):
    return {"rationale": "t",
            "config": [{"name": k, "value": v} for k, v in (config or {}).items()],
            "hypothesis": {"claim": "c", "source": source, "predicted_effect": "e"},
            "predicted_delta": -0.5,
            **({"edits": edits} if edits is not None else {})}


def test_reasoner_admits_edit_to_train_entrypoint():
    edit = [{"path": "engines/vision/train.py", "content": EDITED_TRAIN}]
    p = Reasoner(MockLLM(jsons=[_proposal(edits=edit)]), PROFILE).propose([])
    assert p.admissible and p.edits == edit


def test_reasoner_rejects_edit_to_frozen_prepare():
    edit = [{"path": "engines/vision/prepare.py", "content": "# malicious"}]
    p = Reasoner(MockLLM(jsons=[_proposal(edits=edit)]), PROFILE).propose([])
    assert not p.admissible
    assert any("forbidden edit" in v for v in p.violations)


def test_reasoner_rejects_edit_outside_entrypoint():
    edit = [{"path": "scholarloop/experiment/runner.py", "content": "x = 1"}]
    p = Reasoner(MockLLM(jsons=[_proposal(edits=edit)]), PROFILE).propose([])
    assert any("outside the train entrypoint" in v for v in p.violations)


def test_edit_runs_in_isolation_without_touching_real_source(tmp_path):
    real_train = ROOT / "engines" / "vision" / "train.py"
    before = real_train.read_text()
    edit = [{"path": "engines/vision/train.py", "content": EDITED_TRAIN}]
    entry = run_experiment(PROFILE, "smoke", "exp_edit",
                           Hypothesis("hardcode near-ideal config", "arXiv:1"),
                           edits=edit, ledger_path=tmp_path / "l.jsonl",
                           registry_dir=tmp_path / "r")
    # the edited code ran: config reflects the hardcoded near-ideal values, scored kept
    assert entry.config["depth"] == 20 and entry.verdict == "kept"
    assert real_train.read_text() == before          # the real repo source is untouched


def test_run_experiment_refuses_edit_outside_entrypoint_defense_in_depth(tmp_path):
    # the allowlist subsumes the denylist: anything but the train entrypoint is refused
    for bad_path in ["engines/vision/prepare.py", "../../scholarloop/core/registry.py",
                     "engines/vision/sitecustomize.py"]:
        with pytest.raises(ValueError, match="may only replace"):
            run_experiment(PROFILE, "smoke", "exp_bad", Hypothesis("c", "arXiv:1"),
                           edits=[{"path": bad_path, "content": "x"}],
                           ledger_path=tmp_path / "l.jsonl", registry_dir=tmp_path / "r")
