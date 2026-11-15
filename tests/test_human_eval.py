import random

from scholarloop.human_eval import kappa, rater_paper, sample_claims, score

STAGED = """# A Title

> **Panel review:** reject · mean score 2.2/10 (harsh 2)

## 1. Abstract

Dropout lowered error on three seeds [claim_3].

### 4.1 Results

It fell from 10.0 to 9.5 [exp_0002].

---

### Drafting notes (for the author)

- A note that names the pipeline.

*Written after scientific review and author release. Inspect the [evidence bindings](evidence_map.md).*
"""


def test_rater_paper_removes_what_identifies_the_pipeline():
    text = rater_paper(STAGED)
    for tell in ("Panel review", "Drafting notes", "scientific review", "evidence_map", "[claim_3]", "1. Abstract", "4.1"):
        assert tell not in text
    assert "## Abstract" in text and "### Results" in text
    assert "It fell from 10.0 to 9.5 [exp_0002]." in text       # run ids are evidence, they stay
    assert text.endswith("[exp_0002].\n")


def _claim(quote, flagged, passes=2):
    return {"quote": quote, "model": [{}] * passes, "model_flagged": flagged}


def test_sample_takes_half_flagged_and_skips_overlapping_quotes():
    pool = [_claim(f"flagged claim number {n}", True) for n in range(5)]
    pool += [_claim(f"plain claim number {n}", False) for n in range(5)]
    pool += [_claim("plain claim number 0 with a longer tail", False, passes=4)]
    chosen = sample_claims(pool, 6, random.Random(0))
    assert len(chosen) == 6 and sum(c["model_flagged"] for c in chosen) == 3
    quotes = [c["quote"] for c in chosen]
    assert "plain claim number 0 with a longer tail" in quotes      # listed in more passes: taken first
    assert "plain claim number 0" not in quotes                     # a quote inside a chosen quote


def test_sample_fills_from_the_other_stratum_when_one_is_short():
    pool = [_claim("the one flagged claim", True)] + [_claim(f"plain {n}", False) for n in range(8)]
    chosen = sample_claims(pool, 6, random.Random(0))
    assert len(chosen) == 6 and sum(c["model_flagged"] for c in chosen) == 1


def test_kappa():
    assert kappa([("a", "a"), ("b", "b"), ("a", "a"), ("b", "b")]) == 1.0
    assert kappa([("a", "b"), ("b", "a"), ("a", "a"), ("b", "b")]) == 0.0
    assert kappa([("a", "a"), ("a", "a")]) is None                  # one label on both sides
    assert kappa([]) is None


KEY = {"studies": {"toy": {
    "papers": {"A": {"writer": "staged"}, "B": {"writer": "baseline"}}, "pair": ["A", "B"],
    "items": [{"item_id": "toy-01", "paper": "A", "model_flagged": False, "model_split": False},
              {"item_id": "toy-02", "paper": "B", "model_flagged": True, "model_split": False},
              {"item_id": "toy-03", "paper": "B", "model_flagged": False, "model_split": True},
              {"item_id": "toy-04", "paper": "B", "model_flagged": True, "model_split": False}]}}}


def _rater(verdicts, prefer="A"):
    return {"claims": {f"toy-{n:02d}": v for n, v in enumerate(verdicts, 1) if v},
            "ratings": [{"study": "toy", "paper": "paper_A", "conclusions_follow_1to5": "5", "limits_stated_1to5": "4"},
                        {"study": "toy", "paper": "paper_B", "conclusions_follow_1to5": "2", "limits_stated_1to5": ""}],
            "preference": [{"study": "toy", "more_faithful": prefer}]}


def test_score_compares_the_human_majority_with_the_model_and_reports_ties():
    report = score(KEY, {"r1": _rater(["supported", "overstated", "overstated", "supported"]),
                         "r2": _rater(["supported", "contradicted", "supported", "overstated"], prefer="none"),
                         "r3": _rater(["supported", "overstated", "unsupported", None], prefer="paper_b")})
    # toy-04 has one vote each way: no majority, left out and counted as tied.
    assert report["items_with_human_majority"] == 3 and report["items_tied_or_unrated"] == 1
    vs = report["human_majority_vs_model"]
    assert vs["model_flagged"] == {"items": 1, "humans_agree": 1}
    assert vs["model_supported"] == {"items": 2, "humans_agree": 1}     # humans flag toy-03, the model did not
    assert report["by_writer"]["staged"] == {"items": 1, "beyond_by_humans": 0, "beyond_by_model": 0}
    assert report["by_writer"]["baseline"] == {"items": 2, "beyond_by_humans": 2, "beyond_by_model": 1}
    # overstated and contradicted disagree on four labels and agree on supported-vs-not.
    first = next(p for p in report["pairwise"] if p["raters"] == ["r1", "r2"])
    assert first["agreement_4way"] == 0.25 and first["agreement_binary"] == 0.5
    assert report["paper_ratings_mean"] == {"staged": {"conclusions_follow_1to5": 5.0, "limits_stated_1to5": 4.0},
                                            "baseline": {"conclusions_follow_1to5": 2.0}}
    assert report["paper_preference"] == {"staged": 1, "no preference": 1, "baseline": 1}
