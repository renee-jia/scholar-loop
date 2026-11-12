"""Persistent scientific review. Only an explicit author decision issues a release.

The JSON state is the authority; Markdown files are views. Mutations are serialized
and atomically committed. Hashes detect stale artifacts, not hostile filesystem writes.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from jsonschema import Draft7Validator

from scholarloop.core.agent import AgentTrace
from scholarloop.core.locking import exclusive_lock
from scholarloop.review.contracts import ROLES, VERSION, REVIEW_SCHEMA, ReviewBlocked, atomic_json, digest, read_json
from scholarloop.review.critics import ArgumentBuilder, ScientificCritic, check_claim_assessments
from scholarloop.review.dossier import build_dossier, snapshot, validate_argument


class ScientificReviewSession:
    STATE_FILE = "scientific_review.json"

    def __init__(self, llm, profile, out_dir, *, ledger_path, registry_dir, research_dir=None, trace=None,
                 critic_llms=None):
        self.llm, self.profile = llm, profile
        self.critic_llms = dict(critic_llms or {})
        if set(self.critic_llms) - set(ROLES) or any(v is None for v in self.critic_llms.values()):
            raise ReviewBlocked("Critic clients must map known review roles to actual LLM clients")
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ledger_path, self.registry_dir = Path(ledger_path), Path(registry_dir)
        self.research_dir = Path(research_dir) if research_dir else None
        self.trace = trace or AgentTrace(path=self.out / "review.agents.jsonl")

    def inputs(self):
        return snapshot(self.profile, self.ledger_path, self.registry_dir, self.research_dir)

    def load(self):
        state = read_json(self.out / self.STATE_FILE)
        if state.get("schema_version") not in {1, VERSION}:
            raise ReviewBlocked("Unsupported scientific review schema; review again with this version")
        return state

    @contextmanager
    def _lock(self):
        try:
            with exclusive_lock(self.out / ".scientific_review.lock"):
                yield
        except BlockingIOError as exc:
            raise ReviewBlocked("Another scientific review operation is running") from exc

    def _save(self, state):
        atomic_json(self.out / self.STATE_FILE, state)
        self._render(state)

    def _archive(self, state):
        atomic_json(self.out / "review_history" / f"{digest(state)}.json", state)

    def review(self, *, feedback="", argument=None):
        """One complete review round; failed critics invalidate any earlier release.

        P0 records experiment requests but never silently runs them. Add evidence or
        reframe the argument and call again; every previous open issue needs disposition.
        """
        with self._lock():
            old = self.load() if (self.out / self.STATE_FILE).exists() else {}
            if old:
                self._archive(old)
            state = {"schema_version": VERSION, "round": old.get("round", 0) + 1,
                     "status": "REVIEWING", "dossier": None, "reports": [],
                     "issues": old.get("issues", []), "feedback": feedback,
                     "release": None, "author_decisions": old.get("author_decisions", [])}
            self._save(state)  # invalidate before even reading/building the new dossier
            try:
                inputs = self.inputs()
                dossier = build_dossier(inputs)
                state["dossier"] = dossier
                if dossier["checks"]:
                    state["status"] = "INSUFFICIENT_EVIDENCE"
                    self._save(state)
                    return state
                if argument is None:
                    argument = ArgumentBuilder(self.llm, trace=self.trace).run({
                        "dossier": dossier, "previous_argument": (old.get("dossier") or {}).get("argument"),
                        "previous_issues": state["issues"], "author_feedback": feedback})
                errors = validate_argument(argument, dossier)
                if errors:
                    raise ReviewBlocked("Invalid claim graph: " + "; ".join(errors))
                dossier["argument"] = argument
                pending = [i for i in state["issues"] if i["status"] == "open"]
                reports = []
                for role in ROLES:
                    # Each role sees only its own historical issues, never this round's peer reviews.
                    client = self.critic_llms.get(role, self.llm)
                    report = ScientificCritic(client, role, trace=self.trace).run({
                        "dossier": dossier, "previous_issues": [i for i in pending if i["role"] == role],
                        "author_feedback": feedback})
                    raw_client = getattr(client, "client", client)
                    identity = {"client": f"{type(raw_client).__module__}.{type(raw_client).__name__}",
                                "model": getattr(client, "model", None),
                                "role_override": role in self.critic_llms}
                    reports.append({"role": role, "reviewer": identity, **report})
                    state["reports"] = list(reports)
                    for index, issue in enumerate(report["issues"], 1):
                        state["issues"].append({**issue, "id": f"r{state['round']}:{role}:{index}",
                                                "role": role, "status": "open", "dossier_hash": digest(dossier)})
                    self._save(state)  # partial reviews are inspectable but never releasable
                if digest(self.inputs()) != dossier["input_hash"]:
                    raise ReviewBlocked("Evidence changed during scientific review; review the new snapshot")
                issues = json.loads(json.dumps(state["issues"]))
                for report in reports:
                    for disposition in report["dispositions"]:
                        issue = next(i for i in issues if i["id"] == disposition["issue_id"])
                        issue["status"] = disposition["status"]
                        issue["resolution"] = {**disposition, "round": state["round"],
                                               "dossier_hash": digest(dossier)}
                state["reports"], state["issues"] = reports, issues
                actions = {i["action"] for i in issues if i["status"] == "open"}
                state["status"] = ("NEEDS_EXPERIMENTS" if "experiment" in actions else
                                   "NEEDS_HUMAN_DECISION" if "human" in actions else
                                   "NEEDS_REFRAMING" if actions else "READY_FOR_AUTHOR")
                state["review_hash"] = self._review_hash(state)
                self._save(state)
                return state
            except Exception as exc:
                # A valid dissent cannot vanish merely because a later role/tool failed.
                known = {i["id"] for i in state["issues"]}
                for report in state["reports"]:
                    for index, issue in enumerate(report["issues"], 1):
                        iid = f"r{state['round']}:{report['role']}:{index}"
                        if iid not in known:
                            state["issues"].append({**issue, "id": iid, "role": report["role"],
                                                    "status": "open", "dossier_hash": digest(state["dossier"])})
                            known.add(iid)
                state["status"], state["failure"] = "REVIEW_FAILED", str(exc)
                self._save(state)
                raise

    @staticmethod
    def _review_hash(state):
        return digest({k: state[k] for k in ("round", "dossier", "reports", "issues", "feedback")})

    def _check_review(self, state):
        if state.get("schema_version") != VERSION:
            raise ReviewBlocked("Review contract changed; run scientific review again before approving or writing")
        d = state.get("dossier") or {}
        if not d.get("argument") or d.get("checks") or validate_argument(d["argument"], d):
            raise ReviewBlocked("Scientific argument or deterministic evidence checks are incomplete")
        if {r["role"] for r in state["reports"]} != set(ROLES) or len(state["reports"]) != len(ROLES):
            raise ReviewBlocked("All four scientific critics must complete review")
        if any(r["verdict"] != "pass" or r["issues"] for r in state["reports"]):
            raise ReviewBlocked("Scientific critics have unresolved concerns")
        for report in state["reports"]:
            raw = {k: report[k] for k in REVIEW_SCHEMA["properties"] if k in report}
            if list(Draft7Validator(REVIEW_SCHEMA).iter_errors(raw)) or not report.get("reviewer"):
                raise ReviewBlocked("Scientific review needs claim-by-claim evidence checks and reviewer provenance")
            errors = check_claim_assessments(raw, d)
            if errors:
                raise ReviewBlocked("Incomplete scientific review: " + "; ".join(errors))
        if any(i["status"] == "open" for i in state["issues"]):
            raise ReviewBlocked("Scientific issues remain open")
        if self._review_hash(state) != state.get("review_hash"):
            raise ReviewBlocked("Review artifacts changed; review again")
        if digest(self.inputs()) != d["input_hash"]:
            raise ReviewBlocked("Evidence/profile/code changed; scientific review and approval are stale")

    def approve(self, *, author: str, review_hash: str):
        """Explicit local author action, bound to the exact report the author inspected."""
        if not author.strip():
            raise ReviewBlocked("Author identity is required")
        with self._lock():
            state = self.load()
            if state["status"] != "READY_FOR_AUTHOR" or review_hash != state.get("review_hash"):
                raise ReviewBlocked("Approve the current READY_FOR_AUTHOR review hash")
            self._check_review(state)
            self._archive(state)
            decision = {"actor": author.strip(), "kind": "human", "decision": "approve_writing",
                        "review_hash": review_hash, "timestamp": time.time()}
            state["author_decisions"].append(decision)
            release = {"schema_version": VERSION, "review_hash": review_hash,
                       "input_hash": state["dossier"]["input_hash"],
                       "dossier_hash": digest(state["dossier"]), "author_decision": decision,
                       "approved_claim_ids": [c["id"] for c in state["dossier"]["argument"]["claims"]],
                       "mandatory_limitations": state["dossier"]["argument"]["limitations"]}
            release["id"] = digest(release)
            state["release"], state["status"] = release, "READY_TO_WRITE"
            self._save(state)
            return release

    def require_release(self):
        state = self.load()
        if state["status"] != "READY_TO_WRITE" or not state.get("release"):
            raise ReviewBlocked(f"Writing blocked: {state['status']}. Complete scientific review and author approval.")
        self._check_review(state)
        r = state["release"]
        if r["id"] != digest({k: v for k, v in r.items() if k != "id"}) or \
                r["review_hash"] != state["review_hash"] or r["dossier_hash"] != digest(state["dossier"]) or \
                r["author_decision"] not in state["author_decisions"] or \
                r["author_decision"].get("kind") != "human":
            raise ReviewBlocked("Invalid writing release")
        return state["dossier"], r

    def _render(self, state):
        lines = ["# Scientific review", "", f"Status: **{state['status']}**", "",
                 "This review does not certify scientific truth. Writing requires current evidence and author approval.", ""]
        d = state.get("dossier") or {}
        a = d.get("argument") or {}
        if a:
            lines += ["## Research argument", "", a["question"], "", a["method_summary"], ""]
            for c in a["claims"]:
                lines += [f"- **{c['id']}**: {c['text']} Scope: {c['scope']}",
                          f"  Evidence: {', '.join(c['evidence'])}; measurements: {', '.join(c['measurements'])}"]
            lines += ["", "## Mandatory limitations", ""] + [f"- {s}" for s in a["limitations"]]
        if d:
            lines += ["", "## Evidence inventory", "", d["provenance_note"], "",
                      "All runs, including negative and failed runs:", ""]
            lines += [f"- `{r['id']}`: {r['verdict']}, {r['metric_name']}, {r['fidelity']}; "
                      f"measurements: `{json.dumps(r['metric'], sort_keys=True)}`" for r in d["runs"]]
            lines += ["", "Code, raw run records, exact numeric bindings and input hashes are in "
                      "[scientific_review.json](scientific_review.json).", ""]
        lines += ["", "## Independent reviews", ""]
        for report in state["reports"]:
            lines += [f"- **{report['role']}**: {report['verdict']} — {report['summary']}"]
            reviewer = report.get("reviewer", {})
            lines += [f"  Reviewer: {reviewer.get('model') or reviewer.get('client', 'legacy/unrecorded')}"]
            for check in report.get("claim_checks", []):
                lines += [f"  - `{check['claim_id']}` ({check['status']}): {check['question']}",
                          f"    {check['answer']} Evidence: {', '.join(check['evidence'])}"]
        lines += ["", "## Issues", ""]
        for i in state["issues"]:
            lines += [f"- **{i['id']}** ({i['severity']}, {i['status']}): {i['observation']}",
                      f"  Action: {i['required_action']}", f"  Verify: {i['resolution_criterion']}"]
            if i.get("resolution"):
                lines += [f"  Disposition: {i['resolution']['reason']} "
                          f"(evidence: {', '.join(i['resolution']['evidence'])})"]
        lines += [f"- {s}" for s in d.get("checks", [])]
        if state.get("failure"):
            lines += ["", "Failure: " + state["failure"]]
        if state.get("review_hash"):
            lines += ["", f"Review hash: `{state['review_hash']}`"]
        (self.out / "scientific_review.md").write_text("\n".join(lines) + "\n")
        requests = [{"issue_id": i["id"], "target": i["target"], "required_action": i["required_action"],
                     "resolution_criterion": i["resolution_criterion"]}
                    for i in state["issues"] if i["status"] == "open" and i["action"] == "experiment"]
        atomic_json(self.out / "experiment_requests.json", requests)
        decisions = ([{"kind": "approve_writing", "review_hash": state["review_hash"],
                       "report": "scientific_review.md", "choices": ["approve", "revise", "pause"]}]
                     if state["status"] == "READY_FOR_AUTHOR" else [])
        atomic_json(self.out / "pending_decisions.json", decisions)


def main(argv=None):
    import argparse
    from scholarloop.core.llm import AnthropicLLM, model_from_env
    from scholarloop.core.profile import load_profile
    from scholarloop.core.call_budget import CallLimitedLLM, CallBudgetExceeded
    from scholarloop.core.agent import AgentError

    ap = argparse.ArgumentParser(description="Pre-writing scientific review and explicit author release")
    ap.add_argument("action", choices=["review", "status", "approve"])
    for name in ("profile", "ledger", "registry", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--research")
    ap.add_argument("--feedback", default="")
    ap.add_argument("--argument", help="optional JSON argument to review (not an approval)")
    ap.add_argument("--author")
    ap.add_argument("--review-hash")
    ap.add_argument("--model", default=model_from_env())
    ap.add_argument("--max-llm-calls", type=int, default=60,
                    help="cumulative review/planning API request cap, including retries")
    ap.add_argument("--critic-model", action="append", default=[], metavar="ROLE=MODEL",
                    help="override a critic's model; repeat for implementation/experimental/falsification/argument")
    args = ap.parse_args(argv)
    try:
        models = parse_critic_models(args.critic_model)
        if models and args.action != "review":
            raise ReviewBlocked("--critic-model configures a new review; it cannot change saved reviewer identities")
        def budgeted(model):
            return CallLimitedLLM(AnthropicLLM(model=model), args.max_llm_calls,
                                  path=Path(args.out) / "review_call_budget.json")
        llm = budgeted(args.model) if args.action == "review" else None
        critics = {role: budgeted(model) for role, model in models.items()} if args.action == "review" else {}
        s = ScientificReviewSession(llm,
                                    load_profile(args.profile), args.out, ledger_path=args.ledger,
                                    registry_dir=args.registry, research_dir=args.research, critic_llms=critics)
        if args.action == "review":
            state = s.review(feedback=args.feedback,
                             argument=read_json(Path(args.argument)) if args.argument else None)
            print(f"{state['status']} → {s.out / 'scientific_review.md'}")
            return 0 if state["status"] == "READY_FOR_AUTHOR" else 2
        if args.action == "approve":
            r = s.approve(author=args.author or "", review_hash=args.review_hash or "")
            print(f"READY_TO_WRITE: {r['id']}")
            return 0
        state = s.load()
        if state["status"] == "READY_TO_WRITE":
            s.require_release()
        elif state["status"] == "READY_FOR_AUTHOR":
            s._check_review(state)
        print(f"{state['status']}\nreview hash: {state.get('review_hash', 'not reviewed')}")
        return 0
    except (ReviewBlocked, CallBudgetExceeded, AgentError) as exc:
        print(str(exc))
        return 2


def parse_critic_models(values):
    models = {}
    for item in values:
        role, sep, model = item.partition("=")
        role, model = role.strip(), model.strip()
        if not sep or role not in ROLES or not model or role in models:
            raise ReviewBlocked("Use one --critic-model ROLE=MODEL per known role; empty or duplicate mappings are invalid")
        models[role] = model
    return models
