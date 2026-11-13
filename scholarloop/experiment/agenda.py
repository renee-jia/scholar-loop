"""Critic → frozen protocol → all conditions → fresh scientific review.

Execution is explicit and bound to a protocol hash. A worse score never prunes a
control/ablation; neither execution nor a positive difference resolves review issues.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

from scholarloop.core.ledger import Hypothesis, LedgerEntry
from scholarloop.experiment.runner import run_experiment
from scholarloop.experiment.spec import ExperimentPlanner, condition_budget, frozen_context, validate_spec
from scholarloop.review.contracts import ReviewBlocked, atomic_json, digest, read_json


class ExperimentAgenda:
    def __init__(self, review):
        self.review = review
        self.out = review.out / "experiment_agenda"
        self.protocols = review.registry_dir / "protocols"

    def plan(self, *, spec=None, feedback=""):
        with self.review._lock():
            from scholarloop.experiment.test_boundary import require_unopened_test
            try:
                require_unopened_test(self.review.ledger_path)
            except ValueError as exc:
                raise ReviewBlocked(str(exc)) from exc
            source = self.review.load()
            if source.get("status") != "NEEDS_EXPERIMENTS" or not source.get("review_hash"):
                raise ReviewBlocked("No completed scientific review with open experiment requests")
            if source.get("review_hash") != self.review._review_hash(source):
                raise ReviewBlocked("Source review changed; review again before planning")
            if digest(self.review.inputs()) != source["dossier"]["input_hash"]:
                raise ReviewBlocked("Source evidence changed; review again before planning")
            issues = [i for i in source["issues"] if i["status"] == "open" and i["action"] == "experiment"]
            if not issues or source["status"] != "NEEDS_EXPERIMENTS":
                raise ReviewBlocked("No current scientific experiment requests")
            if spec is None:
                spec = ExperimentPlanner(self.review.llm, self.review.profile, trace=self.review.trace).run({
                    "dossier": source["dossier"], "issues": issues, "author_feedback": feedback,
                    "research_iteration": source.get("research_iteration")})
            errors = validate_spec(spec, self.review.profile, issues, iteration=source.get("research_iteration"))
            if errors:
                raise ReviewBlocked("Invalid experiment protocol: " + "; ".join(errors))
            envelope = {"schema_version": 1, "source_review_hash": source["review_hash"],
                        "issues": issues, "spec": spec, "feedback": feedback,
                        "frozen": frozen_context(self.review.profile, spec)}
            if "research_iteration" in source:
                envelope["research_iteration"] = source["research_iteration"]
            # Check again after planning: a slow model must not freeze a stale review.
            if digest(self.review.inputs()) != source["dossier"]["input_hash"]:
                raise ReviewBlocked("Evidence changed while planning")
            return self._store_plan(envelope)

    def _store_plan(self, envelope):
        from scholarloop.experiment.test_boundary import require_unopened_test
        try:
            require_unopened_test(self.review.ledger_path)
        except ValueError as exc:
            raise ReviewBlocked(str(exc)) from exc
        protocol_id = digest(envelope)
        if (self.out / f"{protocol_id}.json").exists():
            raise ReviewBlocked("This protocol already exists; resume it instead")
        atomic_json(self.protocols / f"{protocol_id}.json", envelope)
        state = {"protocol_id": protocol_id, "status": "PLANNED", "results": [],
                 "current": None, "seconds_used": 0.0, "decisions": [],
                 "base_inputs": self.review.inputs()}
        self._save(state)
        return state

    def _save(self, state):
        pid = state["protocol_id"]
        atomic_json(self.out / f"{pid}.json", state)
        envelope = read_json(self.protocols / f"{pid}.json")
        spec = envelope["spec"]
        lines = ["# Requested experiment protocol", "", f"Status: **{state['status']}**", "",
                 f"Protocol hash: `{pid}`", "", f"Purpose: {spec['purpose']}", "",
                 spec["hypothesis"], "", "Expected: " + spec["predicted_observation"], "",
                 "Falsifying outcome: " + spec["falsification_outcome"], "",
                 f"Source targets: {', '.join(spec.get('issue_ids', spec.get('idea_ids', [])))}", "",
                 f"Seeds: {spec['seeds']}; fidelity: {spec['fidelity']}; metric: {spec['metric']}; "
                 f"holdout: {spec['holdout']}; subprocess budget: {spec['budget_seconds']} seconds.", "",
                 f"Reference: {spec['reference']}. Analysis: treatment minus reference paired mean.", "",
                 spec["interpretation"], "", "## Conditions", ""]
        for c in spec["conditions"]:
            lines += [f"- **{c['name']}**: {c['description']}; config: {c['config']}; "
                      f"edited files: {[e['path'] for e in c['edits']]}"]
        lines += ["", "Full source replacements and frozen source/data/environment records: "
                  f"`{(self.protocols / (pid + '.json')).resolve()}`", "",
                  envelope["frozen"]["data_caveat"], "", "## Limitations", ""]
        lines += [f"- {x}" for x in spec["limitations"]]
        if spec.get("discrimination"):
            lines += ["", "## Planned discriminating contrast", ""]
            lines += [f"- {key}: {value}" for key, value in spec["discrimination"].items()]
        iteration = envelope.get("research_iteration")
        if iteration:
            step = iteration["assessment"]["next_step"]
            lines += ["", "## Prospective research decisions", "",
                      "Unverified recommendation retained from scientific review; not an observed outcome.", "",
                      f"Research round: {iteration['round']}; evidence snapshot: `{iteration['input_hash']}`", ""]
            for key in ("target_issue_ids", "tested_alternative", "prediction_if_primary",
                        "prediction_if_alternative", "decision_if_primary", "decision_if_alternative",
                        "decision_if_inconclusive", "remaining_uncertainty"):
                if key in step:
                    lines += [f"- {key}: {step[key]}"]
        lines += ["", "## Recorded conditions", ""]
        lines += [f"- {r['condition']}: `{r['entry']['id']}`, {r['entry']['verdict']}, "
                  f"{r['entry']['metric']}" for r in state["results"]]
        lines += ["", "Execution never approves writing or resolves an issue. Run scientific review again."]
        (self.out / f"{pid}.md").write_text("\n".join(lines) + "\n")

    def load(self, protocol_id):
        if len(protocol_id) != 64 or any(c not in "0123456789abcdef" for c in protocol_id):
            raise ReviewBlocked("Use the full protocol hash")
        state = read_json(self.out / f"{protocol_id}.json")
        envelope = read_json(self.protocols / f"{protocol_id}.json")
        if state["protocol_id"] != protocol_id or digest(envelope) != protocol_id:
            raise ReviewBlocked("Protocol was modified; create a new plan after scientific review")
        return state, envelope

    def _check_source(self, envelope):
        if envelope.get("source_kind", "review") != "review":
            raise ReviewBlocked("Use the pilot command to execute an ideation protocol")
        source = self.review.load()
        if source.get("review_hash") != envelope["source_review_hash"] or \
                self.review._review_hash(source) != envelope["source_review_hash"]:
            raise ReviewBlocked("Scientific review changed; do not execute an obsolete protocol")

    def _check_inputs(self, state, envelope):
        self._check_source(envelope)
        if frozen_context(self.review.profile, envelope["spec"]) != envelope["frozen"]:
            raise ReviewBlocked("Code, profile, data or environment changed since the protocol was frozen")
        current, original = self.review.inputs(), state["base_inputs"]
        # Only this protocol's recorded append operations are allowed between conditions.
        old_ledger = original["files"]["ledger"]["text"]
        new_ledger = current["files"]["ledger"]["text"]
        if not new_ledger.startswith(old_ledger):
            raise ReviewBlocked("The original ledger was modified")
        try:
            extra = [json.loads(s) for s in new_ledger[len(old_ledger):].splitlines() if s.strip()]
        except ValueError as exc:
            raise ReviewBlocked("Corrupt ledger append; inspect before planning again") from exc
        if extra != [r["entry"] for r in state["results"]]:
            raise ReviewBlocked("Unexpected or interrupted ledger append; inspect before planning again")
        current["files"]["ledger"] = original["files"]["ledger"]
        for result in state["results"]:
            key = f"registry/{result['entry']['id']}.json"
            file = current["files"].pop(key, None)
            if not file or file["sha256"] != result["registry_sha256"]:
                raise ReviewBlocked("A recorded protocol result changed")
            cid = result["entry"].get("candidate_id")
            key = f"candidates/{cid}.json"
            # inputs() validates the content address. Existing evidence remains frozen;
            # only newly referenced snapshots from this protocol may be appended.
            if cid and key not in original["files"]:
                current["files"].pop(key, None)
        for key in list(current["files"]):
            if key.startswith("test_access/") and key not in original["files"]:
                record = json.loads(current["files"][key]["text"])
                if not any(record == {"version": 1, "run_id": r["entry"]["id"],
                           "candidate_id": r["entry"].get("candidate_id"),
                           "protocol": r["entry"].get("protocol")} for r in state["results"]):
                    raise ReviewBlocked("Unexpected final test access")
                current["files"].pop(key)
        current["files"].pop(f"protocols/{state['protocol_id']}.analysis.json", None)
        if current != original:
            raise ReviewBlocked("Other evidence changed; finish scientific review before replanning")

    def execute(self, protocol_id, *, author, max_conditions=None):
        if not author.strip():
            raise ReviewBlocked("An explicit actor is required to execute the inspected protocol")
        if max_conditions is not None and (type(max_conditions) is not int or max_conditions < 1):
            raise ReviewBlocked("max_conditions must be a positive integer")
        with self.review._lock():
            state, envelope = self.load(protocol_id)
            if state["current"]:
                raise ReviewBlocked("An interrupted condition has an uncertain completion state; "
                                    "inspect its ledger/registry and re-review before making a new protocol")
            self._check_inputs(state, envelope)
            if state["status"] == "EXECUTED":
                return state
            spec = envelope["spec"]
            state["decisions"].append({"actor": author.strip(), "decision": "execute_protocol",
                                        "protocol_hash": protocol_id, "timestamp": time.time()})
            pending = spec["conditions"][len(state["results"]):]
            for condition in pending[:max_conditions]:
                reserve = condition_budget(spec, self.review.profile)
                if state["seconds_used"] + reserve > spec["budget_seconds"]:
                    state["status"] = "BUDGET_EXHAUSTED"
                    self._save(state)
                    return state
                self._check_inputs(state, envelope)
                exp_id = f"protocol_{protocol_id[:24]}_{condition['name']}"
                if (self.review.registry_dir / f"{exp_id}.json").exists():
                    raise ReviewBlocked("Condition id already has a registry; refusing to overwrite it")
                state["status"], state["current"] = "RUNNING", exp_id
                self._save(state)
                started = time.monotonic()
                # No funnel promotion, artifact reuse, or dedup against optimization runs.
                entry = run_experiment(
                    self.review.profile, spec["fidelity"], exp_id,
                    Hypothesis(spec["hypothesis"], ("review:" + envelope["source_review_hash"]
                               if "source_review_hash" in envelope else "ideas:" + ",".join(spec["idea_ids"])),
                               spec["predicted_observation"]),
                    config_override={k["name"]: k["value"] for k in condition["config"]},
                    edits=condition["edits"], seeds_override=spec["seeds"],
                    holdout_override=spec["holdout"], purpose=spec["purpose"],
                    protocol={"id": protocol_id, "condition": condition["name"],
                              "scientific_outcome": "unreviewed"},
                    ledger_path=self.review.ledger_path, registry_dir=self.review.registry_dir)
                state["seconds_used"] += time.monotonic() - started
                reg = self.review.inputs()["files"][f"registry/{exp_id}.json"]
                state["results"].append({"condition": condition["name"],
                                          "entry": json.loads(entry.to_json()), "registry_sha256": reg["sha256"]})
                # Persist the completed result before freshness checks; failures retain evidence.
                state["current"], state["status"] = None, "PAUSED"
                self._save(state)
                self._check_inputs(state, envelope)
            if len(state["results"]) == len(spec["conditions"]):
                atomic_json(self.protocols / f"{protocol_id}.analysis.json", analyze(state, spec))
                state["status"] = "EXECUTED"
                self._save(state)
            return state


def analyze(state, spec):
    """Complete-case pairing only; no p-value or automatic scientific verdict."""
    entries = {r["condition"]: LedgerEntry.from_dict(r["entry"]) for r in state["results"]}
    reference = entries[spec["reference"]]
    comparisons = []
    for name, entry in entries.items():
        if name == spec["reference"]:
            continue
        complete = all(e.metric.get("seed_ids") == spec["seeds"] and
                       len(e.metric.get("seeds", [])) == len(spec["seeds"])
                       for e in (reference, entry))
        same_folds = reference.metric.get("folds") == entry.metric.get("folds")
        item = {"condition": name, "reference_run": reference.id, "treatment_run": entry.id,
                "status": "complete" if complete and same_folds else "incomplete",
                "seed_ids": spec["seeds"], "deltas": [], "mean_difference": None}
        if item["status"] == "complete":
            item["deltas"] = [b - a for a, b in zip(reference.metric["seeds"], entry.metric["seeds"])]
            item["mean_difference"] = mean(item["deltas"])
        comparisons.append(item)
    return {"protocol_id": state["protocol_id"], "metric": spec["metric"],
            "comparisons": comparisons, "scientific_outcome": "unreviewed",
            "caveat": "Descriptive paired differences only; incomplete pairs produce no aggregate. "
                      "Seed/fold observations may be dependent. This analysis does not resolve review issues."}


def main(argv=None):
    import argparse
    from scholarloop.core.llm import AnthropicLLM, model_from_env
    from scholarloop.core.profile import load_profile
    from scholarloop.review.session import ScientificReviewSession
    from scholarloop.core.call_budget import CallLimitedLLM, CallBudgetExceeded
    from scholarloop.core.agent import AgentError
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["plan", "status", "run"])
    for name in ("profile", "ledger", "registry", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--research")
    ap.add_argument("--spec", help="JSON protocol; omit to use the planner")
    ap.add_argument("--protocol-hash")
    ap.add_argument("--author", default="")
    ap.add_argument("--feedback", default="")
    ap.add_argument("--max-conditions", type=int)
    ap.add_argument("--max-llm-calls", type=int, default=60)
    args = ap.parse_args(argv)
    try:
        llm = AnthropicLLM(model=model_from_env()) if args.action == "plan" and not args.spec else None
        if llm is not None:
            llm = CallLimitedLLM(llm, args.max_llm_calls, path=Path(args.out) / "review_call_budget.json")
        agenda = ExperimentAgenda(ScientificReviewSession(llm, load_profile(args.profile), args.out,
                                  ledger_path=args.ledger, registry_dir=args.registry, research_dir=args.research))
        if args.action == "plan":
            state = agenda.plan(spec=read_json(Path(args.spec)) if args.spec else None, feedback=args.feedback)
        elif args.action == "run":
            state = agenda.execute(args.protocol_hash or "", author=args.author, max_conditions=args.max_conditions)
        else:
            state, _ = agenda.load(args.protocol_hash or "")
        print(f"{state['status']}: {agenda.out / (state['protocol_id'] + '.md')}")
        return 0 if state["status"] in ("PLANNED", "EXECUTED", "PAUSED") else 2
    except (ReviewBlocked, CallBudgetExceeded, AgentError) as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
