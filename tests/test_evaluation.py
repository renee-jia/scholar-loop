from scholarloop.evaluation import audit_run
from tests.test_scientific_review import setup, released


def test_audit_distinguishes_stale_release_from_missing_work(tmp_path):
    paths = setup(tmp_path)
    p, ledger, registry, out = paths
    # Audit uses a campaign layout; use the shared parent for its scientific report.
    paths = p, ledger, registry, tmp_path
    s = released(paths)
    audit = audit_run(p, tmp_path)
    assert audit["scientific_review"]["current_release_valid"] is True
    assert audit["ideation"] is None and audit["token_usage"] is None
    (p.root / p.train_entrypoint).write_text("changed source")
    audit = audit_run(p, tmp_path)
    assert audit["scientific_review"]["current_release_valid"] is False
    assert "stale" in audit["scientific_review"]["writing_block_reason"]
