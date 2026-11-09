"""Smoke tests for the three foundational data contracts."""

from pathlib import Path

import pytest

from scholarloop.core.ledger import Hypothesis, Ledger, LedgerEntry
from scholarloop.core.profile import ProfileError, load_profile
from scholarloop.core.registry import VerifiedRegistry

ROOT = Path(__file__).resolve().parents[2]


# ---- Profile ----
def test_example_profile_loads():
    p = load_profile(ROOT / "profiles" / "image-classification.yaml")
    assert p.name == "image-classification"
    assert p.metric.direction == "minimize"
    assert p.best_baseline() == 4.9  # hardest must-beat under "minimize"
    assert p.metric.is_better(4.5, 4.9)


def test_profile_detects_forbidden_edit():
    p = load_profile(ROOT / "profiles" / "image-classification.yaml")
    assert p.touches_forbidden(["engines/vision/train.py"]) == []
    assert p.touches_forbidden(["engines/vision/prepare.py"]) == ["engines/vision/prepare.py"]


def test_forbidden_guard_no_false_positive_on_substring():
    # "retrieval.py" contains the substring "eval" but must NOT be flagged.
    p = load_profile(ROOT / "profiles" / "image-classification.yaml")
    assert p.touches_forbidden(["models/retrieval.py"]) == []
    assert p.touches_forbidden(["data/medieval_augment.py"]) == []


def test_forbidden_guard_catches_nested_prepare():
    # a nested prepare.py (the frozen surface) must be caught anywhere in the tree.
    p = load_profile(ROOT / "profiles" / "image-classification.yaml")
    assert p.touches_forbidden(["engines/nlp/prepare.py"]) == ["engines/nlp/prepare.py"]


def test_forbidden_guard_normalizes_case_and_dot_segments():
    # spelling variants (case, './') must not evade the guard
    p = load_profile(ROOT / "profiles" / "image-classification.yaml")
    assert p.touches_forbidden(["engines/vision/./prepare.py"]) == ["engines/vision/./prepare.py"]
    assert p.touches_forbidden(["engines/vision/PREPARE.PY"]) == ["engines/vision/PREPARE.PY"]


def test_malformed_baseline_raises_profile_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: x\nparadigm: supervised\ntrain_entrypoint: t.py\nfixed_module: f.py\n"
        "metric: {name: m, direction: minimize}\n"
        "budget: {smoke_sec: 1, verify_sec: 1, full_sec: 1}\n"
        "allowed_edits: [optimizer]\nforbidden_edits: [f.py]\n"
        "baselines: ['model@4.9.9']\n"   # double-dot: schema must reject before float() crashes
    )
    with pytest.raises(ProfileError):
        load_profile(bad)


def test_scientific_notation_baseline_accepted(tmp_path):
    ok = tmp_path / "ok.yaml"
    ok.write_text(
        "name: x\nparadigm: supervised\ntrain_entrypoint: t.py\nfixed_module: f.py\n"
        "metric: {name: m, direction: minimize}\n"
        "budget: {smoke_sec: 1, verify_sec: 1, full_sec: 1}\n"
        "allowed_edits: [optimizer]\nforbidden_edits: [f.py]\n"
        "baselines: ['model@1e-4']\n"
    )
    assert load_profile(ok).best_baseline() == 1e-4


def test_invalid_profile_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: X-has-caps\nparadigm: nonsense\n")
    with pytest.raises(ProfileError):
        load_profile(bad)


# ---- Ledger ----
def test_ledger_roundtrip(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    e = LedgerEntry(
        id="exp_0001",
        domain="image-classification",
        hypothesis=Hypothesis(claim="cosine schedule helps", source="arXiv:1608.03983",
                              predicted_effect="-0.3 top1_err"),
        metric_name="val_top1_err",
        fidelity=["smoke", "verify"],
        metric={"smoke": 6.2, "verify": 5.8, "seeds": [5.8, 5.9, 5.7]},
        verdict="kept",
    )
    ledger.append(e)
    rows = list(ledger.read_all())
    assert len(rows) == 1
    assert rows[0].primary_score() == 5.8
    assert ledger.best("image-classification", "minimize").id == "exp_0001"


def test_ledger_rejects_ungrounded_idea(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    e = LedgerEntry(
        id="exp_0002", domain="d",
        hypothesis=Hypothesis(claim="vibes", source=""),  # no source
        metric_name="m",
    )
    with pytest.raises(ValueError):
        ledger.append(e)


# ---- VerifiedRegistry ----
def test_registry_grounding(tmp_path):
    reg = VerifiedRegistry(tmp_path / "vr.json", exp_id="exp_0001")
    reg.capture("val_top1_err.smoke", 6.2, seeds=[6.2])
    reg.capture("val_top1_err.verify", 5.8, seeds=[5.8, 5.9, 5.7])
    reg.save()

    reloaded = VerifiedRegistry.load(tmp_path / "vr.json")
    assert reloaded.is_grounded(5.8)
    assert reloaded.is_grounded(5.9)        # a seed value
    assert not reloaded.is_grounded(4.1)    # never measured

    # a draft citing a real number passes; an invented one is flagged
    assert reloaded.audit_text("We reach 5.8 error.") == []
    assert reloaded.audit_text("We reach 4.1 error.") == ["4.1"]


def test_registry_direction_check(tmp_path):
    reg = VerifiedRegistry(tmp_path / "vr.json")
    reg.capture("val_top1_err.verify", 5.8)
    assert reg.check_direction("val_top1_err.verify", baseline=4.9, claim="beats",
                              metric_direction="minimize") is False  # 5.8 does NOT beat 4.9
    assert reg.check_direction("val_top1_err.verify", baseline=6.0, claim="beats",
                              metric_direction="minimize") is True
    # the "loses" branch and the missing-key branch
    assert reg.check_direction("val_top1_err.verify", baseline=4.9, claim="loses",
                              metric_direction="minimize") is True
    assert reg.check_direction("missing.key", baseline=4.9, claim="beats",
                              metric_direction="minimize") is False


def test_audit_no_false_positive_on_identifiers(tmp_path):
    reg = VerifiedRegistry(tmp_path / "vr.json")
    reg.capture("val_top1_err.verify", 5.8333, seeds=[5.8, 5.9, 5.7])
    # arXiv ids, hyphenated terms, model names, and ranges must NOT be flagged as numbers
    text = ("Following arXiv:1609.04836, our resnet18 reaches top-1 error in the 5.5-4.9 band; "
            "we report 5.8 on val.")
    assert reg.audit_text(text) == []          # 5.8 grounds; nothing else is treated as a number


def test_audit_catches_embedded_numbers_but_not_arxiv_ids(tmp_path):
    reg = VerifiedRegistry(tmp_path / "vr.json")
    reg.capture("m.v", 5.8, seeds=[5.8])
    # numbers embedded after '=' or inside parens/commas are caught (not just whitespace-delimited)
    assert reg.audit_text("acc=9.9 and band (1.2,3.4)") == ["9.9", "1.2", "3.4"]
    # arXiv ids stay single non-numeric tokens; a grounded number passes
    assert reg.audit_text("per arXiv:1609.04836 we report 5.8") == []


def test_audit_precision_rounding(tmp_path):
    reg = VerifiedRegistry(tmp_path / "vr.json")
    reg.capture("val_top1_err.verify", 5.8333, seeds=[5.8333])
    assert reg.audit_text("we reach 5.83 error") == []     # rounds to the stored mean at 2dp
    assert reg.audit_text("we reach 4.10 error") == ["4.10"]  # genuinely never measured


def test_registry_killed_status_roundtrip(tmp_path):
    reg = VerifiedRegistry(tmp_path / "vr.json", exp_id="exp_k", status="killed")
    reg.save()
    assert VerifiedRegistry.load(tmp_path / "vr.json").status == "killed"


def test_ledger_skips_corrupt_line(tmp_path):
    # one corrupt line must not abort reading the whole append-only ledger
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    good = LedgerEntry(id="exp_a", domain="d",
                       hypothesis=Hypothesis(claim="c", source="arXiv:1"),
                       metric_name="m", metric={"smoke": 1.0}, verdict="kept")
    ledger.append(good)
    with path.open("a") as f:
        f.write("{ this is not valid json }\n")
    ledger.append(LedgerEntry(id="exp_b", domain="d",
                              hypothesis=Hypothesis(claim="c", source="arXiv:2"),
                              metric_name="m", metric={"smoke": 2.0}, verdict="kept"))
    ids = [e.id for e in ledger.read_all()]
    assert ids == ["exp_a", "exp_b"]   # corrupt middle line skipped, both good ones survive


def test_idea_chains_groups_funnel_tiers_by_config():
    from scholarloop.core.ledger import idea_chains

    def e(eid, cfg, tier, score, verdict, src="arXiv:1"):
        return LedgerEntry(id=eid, domain="d", hypothesis=Hypothesis("c", src), metric_name="m",
                           config=cfg, fidelity=[tier], metric={tier: score}, verdict=verdict)

    # one idea climbing smoke->verify (same config), plus a different, discarded idea
    chains = idea_chains([e("a", {"lr": 0.1}, "smoke", 4.0, "kept"),
                          e("b", {"lr": 0.1}, "verify", 3.8, "kept"),
                          e("c", {"lr": 0.5}, "smoke", 9.0, "discarded", src="arXiv:2")])
    assert len(chains) == 2                                   # two distinct configs = two ideas
    winner = next(c for c in chains if c["config"] == {"lr": 0.1})
    assert winner["tiers"] == ["smoke", "verify"]            # the funnel chain rolled up
    assert winner["top_fidelity"] == "verify" and winner["score"] == 3.8 and winner["verdict"] == "kept"
