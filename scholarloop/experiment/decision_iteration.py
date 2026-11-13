"""Synthetic-observation adapter for the production IterationReviewer component.

Reuses its assessment checks and issue replay, but not ResearchCycle's real dossier,
scientific critics, protocol executor or writing release. Never creates training evidence.
"""
import copy

from scholarloop.core.agent import AgentTrace
from scholarloop.experiment.decision_bench import ACTIONS, MECHANISMS
from scholarloop.experiment.research_cycle import ASSESSMENT_SCHEMA, IterationReviewer, tracked_issues
from scholarloop.review.contracts import atomic_json, digest

KINDS = {"clean_split": "falsification", "matched_compute": "ablation", "new_population": "boundary_test"}
SCHEMA = copy.deepcopy(ASSESSMENT_SCHEMA)
SCHEMA["properties"]["choice"] = {"enum": [*ACTIONS, *MECHANISMS, "abstain"]}
SCHEMA["required"].append("choice")
SYSTEM = IterationReviewer.system + (
    " This invocation uses synthetic decision-development observations only, not training results. "
    "Choose an action or terminal explanation/abstain in choice. Actions must use the supplied "
    "action_kinds mapping and max_training_attempts=1 (one simulated intervention, not training). "
    "Terminal choices require next_step.kind=stop and max_training_attempts=0. The task cost budget "
    "is separate from that compatibility field. Cite only supplied synthetic observation IDs. "
    "The initial observation is preliminary; it does not identify a cause. No real experiment "
    "protocol or writing release can be generated from this adapter."
)


class SyntheticIterationReviewer(IterationReviewer):
    name = "synthetic_iteration_reviewer"
    schema = SCHEMA
    system = SYSTEM

    def postcheck(self, output, ctx):
        errors = super().postcheck(output, ctx)
        observed = {row["id"] for row in ctx["dossier"]["runs"]}
        cited = set(output["evidence"])
        for update in output["issue_updates"]:
            cited.update(update["evidence"])
        if cited - observed:
            errors.append("Only supplied synthetic observation IDs are evidence in this adapter")
        choice, step = output["choice"], output["next_step"]
        if choice in ACTIONS:
            if ACTIONS[choice]["cost"] > ctx["remaining_budget"]:
                errors.append("Simulated intervention exceeds remaining cost budget")
            if step["kind"] != KINDS[choice] or step["max_training_attempts"] != 1:
                errors.append("Bind the chosen action to its kind and one simulated intervention")
        elif step["kind"] != "stop" or step["max_training_attempts"] != 0:
            errors.append("Terminal choice must stop without a further intervention")
        return errors


def synthetic_dossier(ctx):
    """Only transform the public view; no evaluator task/truth is accepted here."""
    initial = {"id": "synthetic:initial", "gain": ctx["task"]["initial_gain"]}
    seen = {initial["id"]: initial}
    for observation in ctx["history"]:
        eid = "synthetic:" + observation["evidence_id"]
        row = {"id": eid, "action": observation["action"], "gain": observation["gain"]}
        if eid in seen and seen[eid] != row:
            raise ValueError("Same simulated evidence ID has conflicting observations")
        seen[eid] = row
    return {"source_kind": "synthetic_development_observations", "task_id": ctx["task"]["id"],
            "runs": list(seen.values()), "papers": {}, "protocols": {}, "candidates": {},
            "checks": [], "scope": ctx["task"]["scope"]}


class IterationPolicy:
    def __init__(self, llm, trace: AgentTrace, path):
        self.agent = SyntheticIterationReviewer(llm, trace=trace)
        self.path, self.prior = path, []

    def __call__(self, ctx):
        dossier = synthetic_dossier(ctx)
        result = self.agent.run({"question": ctx["task"]["question"], "dossier": dossier,
            "task": ctx["task"], "action_kinds": KINDS, "remaining_budget": ctx["remaining_budget"],
            "max_training_attempts": 1, "history": self.prior[-5:],
            "omitted_rounds": max(0, len(self.prior) - 5), "tracked_issues": tracked_issues(self.prior),
            "author_feedback": "No human input; simulated observations only."})
        record = {"round": len(self.prior) + 1, "question": ctx["task"]["question"],
                  "input_hash": digest(dossier), "assessment": result,
                  "available_evidence": sorted(row["id"] for row in dossier["runs"]),
                  "status": "synthetic_component_evaluation_only"}
        self.prior.append(record)
        atomic_json(self.path, {"source_kind": "synthetic_component_evaluation_only", "rounds": self.prior})
        choice = result["choice"]
        return {"action" if choice in ACTIONS else "decision": choice}
