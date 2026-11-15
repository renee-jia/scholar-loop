"""Human check of the claim audit: build the packet raters receive, and score what they return.

The claim audit (`scholarloop.reliability claims`) asks a model whether each result claim of a
paper is supported by the recorded evidence. This module asks people the same question about a
sample of those claims, so the model judge can be compared with human judgement.

`build` writes two things. The packet (papers with pipeline marks removed, a readable evidence
sheet per study, the sampled claims, empty response sheets) goes to the raters. The key (which
pipeline wrote which paper, and what the model judge said) stays with the organiser. `score`
reads returned response sheets and the key.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

VERDICTS = ("supported", "overstated", "unsupported", "contradicted")
CLAIM_COLUMNS = ["item_id", "paper", "quote", "verdict", "confidence", "note"]
PAPER_COLUMNS = ["study", "paper", "conclusions_follow_1to5", "limits_stated_1to5", "note"]
PREFERENCE_COLUMNS = ["study", "more_faithful", "note"]

_PANEL = re.compile(r"^> \*\*Panel review:\*\*.*\n+", re.M)
_NOTES = re.compile(r"^#{2,3} Drafting notes \(for the author\).*\Z", re.M | re.S)
_FOOTER = re.compile(r"^\*[^\n]*(?:scientific review|evidence bindings)[^\n]*\*\s*$", re.M)
_NUMBERED = re.compile(r"^(#+) \d+(?:\.\d+)*\.? ", re.M)
_RULE = re.compile(r"\n+---\s*\Z")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def rater_paper(text: str) -> str:
    """A paper as a rater sees it: nothing left that says which pipeline wrote it.

    The claim judge saw papers with approved-statement ids removed. A person would also notice
    the panel verdict under the title, the drafting notes, the release footer and the numbered
    headings only one of the writers uses, so those go too."""
    from scholarloop.reliability import blind_paper
    text = blind_paper(text)
    text = _NOTES.sub("", _PANEL.sub("", text))
    text = _FOOTER.sub("", text)
    text = _NUMBERED.sub(r"\1 ", text)
    return _RULE.sub("", text.rstrip()).rstrip() + "\n"


# ---------------------------------------------------------------------------------------------
# Evidence sheet
# ---------------------------------------------------------------------------------------------
def _num(x) -> str:
    if isinstance(x, float):
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _config(config) -> str:
    if isinstance(config, list):
        config = {c.get("name"): c.get("value") for c in config}
    return ", ".join(f"{k}={_num(v)}" for k, v in (config or {}).items()) or "defaults"


def _train_version(view: dict, candidate_id) -> str:
    candidate = view.get("candidates", {}).get(candidate_id) or {}
    ref = (candidate.get("source_versions") or {}).get(view["profile"].get("train_entrypoint"), "")
    return ref.rpartition("@")[2] or "?"


def _score(run: dict):
    metric = run.get("metric") or {}
    for level in run.get("fidelity") or []:
        if metric.get(level) is not None:
            return metric[level]
    return None


def render_evidence(view: dict) -> str:
    """The recorded evidence of one study, laid out for a person. Same content as the JSON the
    model judge was given; nothing is added, and the JSON ships beside it."""
    profile = view["profile"]
    metric, direction = profile["metric"]["name"], profile["metric"]["direction"]
    out = ["# Recorded evidence", "",
           "Everything here was recorded by the experiment runner, not written by a paper author. "
           "`evidence.json` in this folder is the same evidence in full, as the model judge read it.", "",
           "## 1. Task", "",
           f"- Dataset: {', '.join(profile.get('datasets', []))}",
           f"- Metric: `{metric}` ({'lower' if direction == 'minimize' else 'higher'} is better)",
           f"- Reference baseline from the task definition: {', '.join(profile.get('baselines', [])) or 'none'}",
           f"- Default settings: {_config(profile.get('knobs'))}",
           f"- Each score is a mean over seeds; one seed is one train/validation split "
           f"({profile.get('cv_folds', '?')} splits available by default).", ""]

    idea = (view.get("research", {}).get("handoff.json") or {}).get("idea")
    if isinstance(idea, dict):
        out += ["## 2. The idea handed to the experiments", "",
                f"**{idea.get('title', '')}**", ""]
        hyp = idea.get("hypothesis")
        if isinstance(hyp, dict):
            out += [f"- {k.replace('_', ' ').capitalize()}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
                    for k, v in hyp.items()] + [""]
        else:
            out += [f"Hypothesis: {hyp}", ""]
        if idea.get("experiment_plan"):
            plan = idea["experiment_plan"]
            out += ["Planned test: " + (plan if isinstance(plan, str) else json.dumps(plan, ensure_ascii=False)), ""]

    protocols = view.get("protocols") or {}
    for n, (pid, protocol) in enumerate(protocols.items(), 1):
        spec = protocol["specification"].get("spec", {})
        out += [f"## 3.{n} Registered protocol (written before its runs)", "", f"Id: `{pid}`", "",
                f"- Hypothesis: {spec.get('hypothesis', '')}",
                f"- Predicted observation: {spec.get('predicted_observation', '')}",
                f"- Outcome that would falsify it: {spec.get('falsification_outcome', '')}",
                f"- How results are to be read: {spec.get('interpretation', '')}",
                f"- Seeds: {spec.get('seeds')}", ""]
        for key, title in (("limitations", "Limitations stated in the protocol"),
                           ("omitted_controls", "Controls the protocol says it leaves out")):
            items = spec.get(key)
            if items:
                out += [f"{title}:", ""] + [f"- {i if isinstance(i, str) else json.dumps(i, ensure_ascii=False)}"
                                            for i in (items if isinstance(items, list) else [items])] + [""]
        out += ["Arms:", "", "| arm | settings | code | description |", "|---|---|---|---|"]
        for c in spec.get("conditions", []):
            code = ", ".join(e.get("content_version", "").rpartition("@")[2] for e in c.get("edits", [])) or "unmodified"
            out.append(f"| `{c['name']}` | {_cell(_config(c.get('config')))} | {code} | {_cell(c.get('description', ''))} |")
        out.append("")
        analysis = protocol.get("analysis") or {}
        if analysis.get("comparisons"):
            out += ["Paired differences computed by the runner (treatment minus reference, same seed; "
                    f"a negative value means the treatment arm had lower `{metric}`):", "",
                    "| treatment arm | reference run | status | mean difference | per-seed differences |",
                    "|---|---|---|---|---|"]
            for c in analysis["comparisons"]:
                deltas = ", ".join(_num(round(d, 4)) for d in c.get("deltas") or [])
                mean = c.get("mean_difference")
                out.append(f"| `{c['condition']}` | `{c.get('reference_run', '')}` | {c.get('status', '')} | "
                           f"{_num(round(mean, 4)) if mean is not None else ''} | {deltas} |")
            out.append("")

    out += ["## 4. Runs", "",
            "One row per recorded run, in the order they ran. `kept`/`discarded` is the runner's own "
            "bookkeeping (did this run become the best so far), not a scientific verdict. "
            "`smoke` < `verify` < `full` is the size of the run. Code versions are in `code_versions.md`; "
            "what each run was meant to test is in `run_notes.md`.", "",
            f"| run | kind | size | bookkeeping | settings | code | seeds | mean `{metric}` | per-seed values |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in view["runs"]:
        m = r.get("metric") or {}
        score = _score(r)
        seeds = ", ".join(_num(s) for s in m.get("seeds") or [])
        out.append(f"| `{r['id']}` | {r.get('purpose', '')} | {'/'.join(r.get('fidelity') or [])} | "
                   f"{r.get('verdict', '')} | {_cell(_config(r.get('config')))} | "
                   f"{_train_version(view, r.get('candidate_id'))} | {m.get('seed_ids', '')} | "
                   f"{_num(score) if score is not None else 'no score'} | {seeds} |")
    out += ["", "## 5. Recorded measurements", "",
            "Every number the runner stored, including differences it computed. `sd` is across seeds.", "",
            "| measurement | value | per-seed | median | sd |", "|---|---|---|---|---|"]
    for key, m in view["measurements"].items():
        seeds = ", ".join(_num(s) for s in m.get("seeds") or [])
        sd = m.get("seed_sd", m.get("std"))
        out.append(f"| `{key}` | {_num(m.get('value'))} | {seeds} | "
                   f"{_num(m['median']) if m.get('median') is not None else ''} | {_num(sd) if sd is not None else ''} |")
    return "\n".join(out) + "\n"


def render_run_notes(view: dict) -> str:
    out = ["# Run notes", "",
           "What the experiment agent wrote down before each run: the claim it meant to test and why. "
           "These are the agent's intentions, not measurements. A run only counts as a test of a claim "
           "if its settings and code (see `evidence.md` and `code_versions.md`) really implement it.", ""]
    for r in view["runs"]:
        h = r.get("hypothesis") or {}
        out += [f"## {r['id']}", "",
                f"- Settings: {_config(r.get('config'))}; code {_train_version(view, r.get('candidate_id'))}; "
                f"parent run: {r.get('parent') or 'none'}",
                f"- Claim under test: {h.get('claim', '')}",
                f"- Predicted effect: {h.get('predicted_effect', '')}",
                f"- Source named by the agent: {h.get('source', '')}"]
        trace = (r.get("reasoning") or {}).get("trace")
        if trace:
            out += ["", "Agent reasoning before the run:", "", "> " + trace.replace("\n", "\n> ")]
        out.append("")
    return "\n".join(out)


def render_code(view: dict) -> str:
    train = view["profile"].get("train_entrypoint")
    out = ["# Code versions", "",
           f"Each distinct version of the training script (`{train}`) that a run executed. The first "
           "version is given in full, later ones as a diff against an earlier version. The run table in "
           "`evidence.md` says which version each run used. The scoring module is fixed and is listed "
           "for completeness.", ""]
    for ref, body in view.get("source_files", {}).items():
        out += [f"## {ref}", ""]
        if isinstance(body, dict):
            out += [f"Diff against `{body.get('unified_diff_against')}`:", "", "```diff", body.get("diff", "").rstrip(), "```", ""]
        else:
            out += ["```python", str(body).rstrip(), "```", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# Claim sample
# ---------------------------------------------------------------------------------------------
def collect_claims(audit_dir: Path, name: str, paper: str) -> list[dict]:
    """Distinct quotes the model judge listed for one paper, with its verdict in each pass.
    Quotes that no longer appear in the rater's copy (they sat in a removed block) are dropped."""
    text, found = _squash(paper), {}
    for path in sorted(audit_dir.glob(f"claims_{name}_*.json")):
        if not path.stem.rpartition("_")[2].isdigit() or path.stem.rpartition("_")[0] != f"claims_{name}":
            continue
        for c in json.loads(path.read_text())["claims"]:
            quote = _squash(c["quote"])
            if quote in text:
                entry = found.setdefault(quote, {"quote": quote, "kind": c["kind"], "model": []})
                entry["model"].append({"pass": int(path.stem.rpartition("_")[2]), "verdict": c["verdict"],
                                       "reason": c["reason"]})
    for entry in found.values():
        votes = Counter(m["verdict"] for m in entry["model"])
        beyond = sum(n for v, n in votes.items() if v != "supported")
        entry["model_flagged"] = beyond * 2 > len(entry["model"])
        entry["model_split"] = 0 < beyond < len(entry["model"])
        entry["position"] = text.index(entry["quote"])
    return list(found.values())


def _overlaps(quote: str, chosen: list[dict]) -> bool:
    return any(quote in c["quote"] or c["quote"] in quote for c in chosen)


def sample_claims(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Half flagged by the model judge and half not, where the paper has enough of each.

    Flagged claims are rare in most papers, so a plain random sample would say little about
    them. The price is that the sample's share of flagged claims is not the paper's, which the
    scoring reports per stratum for that reason. Quotes the judge listed in more passes come
    first within a stratum: its label for them is steadier."""
    chosen: list[dict] = []
    def take(candidates, k):
        candidates = sorted(candidates, key=lambda c: (-len(c["model"]), rng.random()))
        for c in candidates:
            if k <= 0:
                break
            if not _overlaps(c["quote"], chosen):
                chosen.append(c)
                k -= 1
    take([c for c in pool if c["model_flagged"]], n // 2)
    take([c for c in pool if not c["model_flagged"]], n - len(chosen))
    take(pool, n - len(chosen))
    return chosen


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------
def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _scrub(value) -> None:
    """Remove what identifies the person who ran the study: their name on decisions and
    paths on their machine. Nothing a claim could rest on."""
    if isinstance(value, dict):
        value.pop("source_path", None)
        for k, v in value.items():
            if k in ("actor", "author", "approved_by") and isinstance(v, str):
                value[k] = "the author"
            else:
                _scrub(v)
    elif isinstance(value, list):
        for v in value:
            _scrub(v)


def build_study(study: str, run: Path, staged: Path, out: Path, rng: random.Random, *,
                per_staged: int, per_baseline: int) -> dict:
    from scholarloop.review.dossier import prompt_view
    dossier = json.loads((run / "scientific_review.json").read_text())["dossier"]
    view = prompt_view({k: v for k, v in dossier.items() if k != "argument"})
    _scrub(view)
    audit_dir = run / "reliability"
    sources = {"staged": staged}
    sources.update({p.stem: p for p in sorted(audit_dir.glob("baseline_[0-9]*.md"))})
    names = list(sources)
    rng.shuffle(names)
    folder = out / "materials" / study
    (folder / "papers").mkdir(parents=True, exist_ok=True)
    (folder / "evidence.md").write_text(render_evidence(view))
    (folder / "run_notes.md").write_text(render_run_notes(view))
    (folder / "code_versions.md").write_text(render_code(view))
    (folder / "evidence.json").write_text(json.dumps(view, indent=1, ensure_ascii=False))

    papers, items = {}, []
    for label, name in zip("ABCDEFGH", names):
        text = rater_paper(sources[name].read_text())
        pool = collect_claims(audit_dir, name, text)
        if not pool and name != "staged":
            continue                                   # a paper the model judge never audited
        papers[label] = {"source": name, "writer": "staged" if name == "staged" else "baseline",
                         "audits": len({m["pass"] for c in pool for m in c["model"]}), "text": text}
        for c in sample_claims(pool, per_staged if name == "staged" else per_baseline, rng):
            items.append({**c, "paper_source": name})
    relabel = {p["source"]: label for label, p in zip("ABCDEFGH", papers.values())}
    papers = {relabel[p["source"]]: p for p in papers.values()}
    for label, p in papers.items():
        (folder / "papers" / f"paper_{label}.md").write_text(p.pop("text"))
    rng.shuffle(items)
    for n, item in enumerate(items, 1):
        item["item_id"] = f"{study}-{n:02d}"
        item["paper"] = relabel[item.pop("paper_source")]

    lines = [f"# Claims to judge: {study}", "",
             "Judge each quoted claim against `evidence.md` (and `code_versions.md`, `run_notes.md` "
             "where needed). The paper file is given so you can read the claim in context. Enter your "
             "answers in `claims.csv` in your response folder.", ""]
    for item in items:
        lines += [f"## {item['item_id']}", "", f"Paper: `papers/paper_{item['paper']}.md`", "",
                  "> " + item["quote"], ""]
    (folder / "claims.md").write_text("\n".join(lines))
    template = out / "responses_template" / study
    _write_csv(template / "claims.csv", CLAIM_COLUMNS,
               [{"item_id": i["item_id"], "paper": f"paper_{i['paper']}", "quote": i["quote"]} for i in items])
    pair = sorted(label for label, p in papers.items() if p["writer"] == "staged" or p["source"] == "baseline_1")
    return {"papers": papers, "pair": pair,
            "items": [{k: i[k] for k in ("item_id", "paper", "quote", "kind", "model", "model_flagged", "model_split")}
                      for i in items]}


def build(studies: list[tuple[str, Path, Path]], out: Path, key_path: Path, *, seed: int = 0,
          per_staged: int = 6, per_baseline: int = 4) -> dict:
    rng = random.Random(seed)
    key = {"version": 1, "seed": seed, "studies": {}}
    for study, run, staged in studies:
        key["studies"][study] = build_study(study, run, staged, out, rng,
                                            per_staged=per_staged, per_baseline=per_baseline)
    template = out / "responses_template"
    _write_csv(template / "paper_ratings.csv", PAPER_COLUMNS,
               [{"study": s, "paper": f"paper_{label}"} for s, k in key["studies"].items() for label in k["pair"]])
    _write_csv(template / "paper_preference.csv", PREFERENCE_COLUMNS,
               [{"study": s, "note": ""} for s in key["studies"]])
    _write_csv(template / "rater.csv", ["question", "answer"], [
        {"question": "rater_id (any short code; used only to tell sheets apart)"},
        {"question": "years of ML research experience"},
        {"question": "highest relevant degree or current position"},
        {"question": "have you used or contributed to ScholarLoop? (yes/no)"},
        {"question": "minutes spent, study by study"},
        {"question": "anything unclear in the instructions"}])
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=1, ensure_ascii=False))
    return key


# ---------------------------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------------------------
def read_responses(folder: Path) -> dict:
    """One rater's folder: {item_id: verdict}, paper ratings and preferences. Blank rows are
    skipped; a verdict outside the four labels is an error, not a guess."""
    claims = {}
    for path in sorted(folder.glob("*/claims.csv")):
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                verdict = (row.get("verdict") or "").strip().lower()
                if not verdict:
                    continue
                if verdict not in VERDICTS:
                    raise ValueError(f"{path}: {row.get('item_id')}: verdict {verdict!r} is not one of {VERDICTS}")
                claims[row["item_id"].strip()] = verdict

    def rows(name):
        path = folder / name
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))
    return {"claims": claims, "ratings": rows("paper_ratings.csv"), "preference": rows("paper_preference.csv")}


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two label sequences. None when there is nothing to compare or when
    chance agreement is total (both sides used a single label)."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in left) / (n * n)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 3)


def _binary(verdict: str) -> str:
    return "supported" if verdict == "supported" else "beyond"


def _majority(labels: list[str]) -> str | None:
    """The strict majority label, or None on a tie: a tie is reported, not broken."""
    if not labels:
        return None
    (top, n), = Counter(labels).most_common(1)
    return top if n * 2 > len(labels) else None


def score(key: dict, raters: dict[str, dict]) -> dict:
    items = {i["item_id"]: {**i, "study": s, "writer": k["papers"][i["paper"]]["writer"]}
             for s, k in key["studies"].items() for i in k["items"]}
    names = sorted(raters)
    pairwise = []
    for a, b in itertools.combinations(names, 2):
        shared = [i for i in items if i in raters[a]["claims"] and i in raters[b]["claims"]]
        four = [(raters[a]["claims"][i], raters[b]["claims"][i]) for i in shared]
        two = [(_binary(x), _binary(y)) for x, y in four]
        pairwise.append({"raters": [a, b], "items": len(shared),
                         "agreement_4way": round(sum(x == y for x, y in four) / len(four), 3) if four else None,
                         "kappa_4way": kappa(four),
                         "agreement_binary": round(sum(x == y for x, y in two) / len(two), 3) if two else None,
                         "kappa_binary": kappa(two)})
    rows, vs_model = [], []
    for item_id, item in items.items():
        votes = [_binary(raters[r]["claims"][item_id]) for r in names if item_id in raters[r]["claims"]]
        human = _majority(votes)
        model = "beyond" if item["model_flagged"] else "supported"
        rows.append({"item_id": item_id, "study": item["study"], "writer": item["writer"], "raters": len(votes),
                     "human": human, "model": model, "model_split": item["model_split"],
                     "labels": {r: raters[r]["claims"][item_id] for r in names if item_id in raters[r]["claims"]}})
        if human:
            vs_model.append((human, model))

    def stratum(label):
        mine = [h for h, m in vs_model if m == label]
        return {"items": len(mine), "humans_agree": sum(h == label for h in mine)}

    def writer(name):
        mine = [r for r in rows if r["writer"] == name and r["human"]]
        return {"items": len(mine), "beyond_by_humans": sum(r["human"] == "beyond" for r in mine),
                "beyond_by_model": sum(r["model"] == "beyond" for r in mine)}
    ratings = {}
    for r in names:
        for row in raters[r]["ratings"]:
            study, label = row.get("study", "").strip(), row.get("paper", "").strip().removeprefix("paper_")
            paper = key["studies"].get(study, {}).get("papers", {}).get(label)
            for column in PAPER_COLUMNS[2:4]:
                value = (row.get(column) or "").strip()
                if paper and value:
                    ratings.setdefault(paper["writer"], {}).setdefault(column, []).append(float(value))
    preference = Counter()
    for r in names:
        for row in raters[r]["preference"]:
            study, choice = row.get("study", "").strip(), (row.get("more_faithful") or "").strip()
            choice = choice.removeprefix("paper_").upper()
            if not choice or study not in key["studies"]:
                continue
            paper = key["studies"][study]["papers"].get(choice)
            preference[paper["writer"] if paper else "no preference"] += 1
    return {"raters": names, "items_in_key": len(items),
            "items_with_human_majority": len(vs_model),
            "items_tied_or_unrated": len(items) - len(vs_model),
            "pairwise": pairwise,
            "human_majority_vs_model": {"agreement": round(sum(h == m for h, m in vs_model) / len(vs_model), 3)
                                        if vs_model else None, "kappa": kappa(vs_model),
                                        "model_flagged": stratum("beyond"), "model_supported": stratum("supported")},
            "by_writer": {w: writer(w) for w in ("staged", "baseline")},
            "paper_ratings_mean": {w: {c: round(sum(v) / len(v), 2) for c, v in cols.items()}
                                   for w, cols in ratings.items()},
            "paper_preference": dict(preference),
            "note": "The sample takes half its claims from those the model judge flagged, so shares of "
                    "flagged claims here are not the papers' own rates. Read the two strata separately.",
            "items": rows}


def render_score(report: dict) -> str:
    vs = report["human_majority_vs_model"]
    out = ["# Human check of the claim audit", "", report["note"], "",
           f"Raters: {len(report['raters'])}. Claims with a human majority: "
           f"{report['items_with_human_majority']} of {report['items_in_key']}.", "",
           "## Agreement between raters", "",
           "| raters | shared claims | agreement (4 labels) | kappa | agreement (supported vs not) | kappa |",
           "|---|---|---|---|---|---|"]
    out += [f"| {' / '.join(p['raters'])} | {p['items']} | {p['agreement_4way']} | {p['kappa_4way']} | "
            f"{p['agreement_binary']} | {p['kappa_binary']} |" for p in report["pairwise"]]
    out += ["", "## Human majority against the model judge (supported vs not)", "",
            f"Agreement {vs['agreement']}, kappa {vs['kappa']}.", "",
            f"- Claims the model flagged: humans also flag {vs['model_flagged']['humans_agree']} of {vs['model_flagged']['items']}.",
            f"- Claims the model called supported: humans agree on {vs['model_supported']['humans_agree']} of {vs['model_supported']['items']}.",
            "", "## By writer (unblinded)", "", "| writer | claims | beyond the evidence, humans | same, model |", "|---|---|---|---|"]
    out += [f"| {w} | {r['items']} | {r['beyond_by_humans']} | {r['beyond_by_model']} |" for w, r in report["by_writer"].items()]
    out += ["", "## Paper-level answers", "", f"Mean ratings: {json.dumps(report['paper_ratings_mean'])}", "",
            f"Which paper reports the study more faithfully: {json.dumps(report['paper_preference'])}"]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scholarloop.human_eval", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write the rater packet and the organiser's key")
    b.add_argument("--study", action="append", required=True, metavar="NAME=RUN_DIR:PAPER",
                   help="a reviewed run and its staged paper; RUN_DIR/reliability holds the claim audit")
    b.add_argument("--out", required=True, help="packet folder (given to raters)")
    b.add_argument("--key", required=True, help="key file (kept by the organiser)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--per-staged", type=int, default=6)
    b.add_argument("--per-baseline", type=int, default=4)
    s = sub.add_parser("score", help="score returned response folders against the key")
    s.add_argument("--key", required=True)
    s.add_argument("--responses", required=True, help="folder with one sub-folder per rater")
    s.add_argument("--out", help="write score.json and score.md here")
    args = parser.parse_args(argv)
    if args.command == "build":
        studies = []
        for spec in args.study:
            name, _, rest = spec.partition("=")
            run, _, paper = rest.partition(":")
            studies.append((name, Path(run), Path(paper)))
        key = build(studies, Path(args.out), Path(args.key), seed=args.seed,
                    per_staged=args.per_staged, per_baseline=args.per_baseline)
        for name, k in key["studies"].items():
            flagged = sum(i["model_flagged"] for i in k["items"])
            print(f"{name}: {len(k['papers'])} papers, {len(k['items'])} claims ({flagged} flagged by the model judge)")
        return 0
    key = json.loads(Path(args.key).read_text())
    raters = {p.name: read_responses(p) for p in sorted(Path(args.responses).iterdir()) if p.is_dir()}
    raters = {name: r for name, r in raters.items() if r["claims"]}
    if not raters:
        print("no filled response folders found", file=sys.stderr)
        return 1
    report = score(key, raters)
    text = render_score(report)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "score.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        (out / "score.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
