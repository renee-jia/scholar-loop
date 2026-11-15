"""Explicit UI actions. User input is never interpreted as a shell command."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def field(name, label, kind="text", default="", **extra):
    return dict(name=name, label=label, type=kind, default=default, **extra)


FEEDBACK = field("feedback", "Feedback / revision notes", "textarea")
AUTHOR = field("author", "Author", default="local-user", required=True)
HASH = field("protocol_hash", "Reviewed protocol ID", required=True)
SPEC = field("spec", "Custom protocol JSON (optional)", "json")
N = field("n_ideas", "Number of candidate ideas", "number", 5, min=1, max=20)
FIELDS = {
    "literature": [field("interest", "What do you want to study, and why does it matter?", "textarea", required=True),
                   field("hunches", "Hypotheses to test (one per line)", "textarea"),
                   field("avoid", "Research boundaries (one per line)", "textarea"), FEEDBACK,
                   field("max_papers", "Maximum papers to retrieve", "number", 20, min=1, max=100),
                   field("brief", "Detailed research brief JSON (optional)", "json")],
    "directions": [FEEDBACK],
    "ideas": [field("directions", "Direction numbers, separated by commas", placeholder="1,3"),
              field("own_direction", "Your own directions (one per line)", "textarea"),
              field("direction_note", "Additional guidance for idea generation", "textarea"), N,
              field("deep", "Enable deeper exploration with multiple strategies", "checkbox", False)],
    "refine_ideas": [FEEDBACK, N],
    "select": [field("select", "Selected idea number", "number", 1, min=1, max=100, required=True),
               AUTHOR, field("reason", "Reason for selection", "textarea", "Selected after inspecting the candidate list")],
    "inject": [field("idea", "Your own idea (Idea JSON)", "json", required=True), AUTHOR,
               field("reason", "Reason for proposing this idea", "textarea", required=True)],
    "decide": [field("decision", "Decision", "select", "reject", options=["reject", "refine", "merge", "revive"]),
               field("idea_ids", "Idea IDs (comma or space separated)", required=True), AUTHOR,
               field("reason", "Reason for this decision", "textarea", required=True), N],
    "campaign": [field("rounds", "Maximum rounds for this session", "number", 3, min=1, max=100),
                 field("population", "Candidates per round", "number", 1, min=1, max=20),
                 field("workers", "Parallel training workers", "number", 1, min=1, max=16),
                 field("patience", "Stop after consecutive rounds without improvement", "number", 2, min=1, max=100),
                 field("training_limit", "Cumulative training run limit", "number", 30, min=1, max=10000)],
    "demo": [],
    "review": [FEEDBACK, field("argument", "Custom argument JSON (optional)", "json"),
               field("iteration", "Include the latest research iteration recommendations", "checkbox", False),
               field("critic_models", "Reviewer model overrides (ROLE=MODEL, one per line)", "textarea")],
    "approve": [AUTHOR, field("review_hash", "Current review version", required=True)],
    "review_status": [],
    "agenda_plan": [FEEDBACK, SPEC],
    "agenda_run": [HASH, AUTHOR, field("max_conditions", "Maximum experimental conditions for this session", "number", 10, min=1, max=100)],
    "pilot_plan": [field("idea_ids", "Pilot idea IDs (comma or space separated)", required=True), FEEDBACK, SPEC],
    "pilot_run": [HASH, AUTHOR, field("max_conditions", "Maximum experimental conditions for this session", "number", 10, min=1, max=100)],
    "iterate": [field("question", "Question to clarify in this iteration", "textarea", required=True), FEEDBACK,
                field("training_limit", "Training budget for proposed experiments", "number", 12, min=0, max=10000)],
    "write": [],
    "write_feedback": [field("feedback", "Revision notes for the current outline or key points", "textarea", required=True)],
    "write_auto": [],
    "evaluate": [],
    "traces": [],
    "inject_faults": [field("per_case", "Injections per case", "number", 8, min=1, max=100),
                      field("seed", "Random seed", "number", 0, min=0, max=100000)],
    "baseline": [field("repeats", "Repetitions", "number", 3, min=1, max=30)],
    "decision_bench": [field("repeats", "Repetitions", "number", 3, min=1, max=100),
                       field("budget", "Intervention budget per round", "number", 3, min=1, max=100),
                       field("seed", "Random seed", "number", 0, min=0, max=100000)],
    "decision_policy": [field("arms", "Model strategies", "select", "plain,structured", options=["plain,structured", "plain,structured,iteration"]),
                        field("repeats", "Repetitions", "number", 3, min=1, max=30),
                        field("budget", "Intervention budget", "number", 3, min=1, max=100),
                        field("calls", "Request limit per task / strategy", "number", 8, min=1, max=100)],
    "suite": [field("manifest", "Multi-task evaluation manifest JSON", "json", required=True)],
    "compare": [field("arms", "Methods to compare", "select", "random", options=["random", "agent,random"]),
                field("repeats", "Repetitions", "number", 3, min=1, max=30),
                field("rounds", "Rounds per study", "number", 3, min=1, max=100),
                field("match", "Match budgets by", "select", "runs", options=["runs", "rounds"]),
                field("warm_start", "Random initial candidates", "number", 0, min=0, max=100)],
}
LABELS = {
    "literature": ("Literature review", "research"), "directions": ("Generate / revise directions", "research"),
    "ideas": ("Choose directions and generate ideas", "ideas"), "refine_ideas": ("Refine candidate ideas", "ideas"),
    "select": ("Select a research idea", "ideas"), "inject": ("Add your own idea", "ideas"),
    "decide": ("Reject / refine / merge / revive", "ideas"), "campaign": ("Start / resume experiments", "experiments"),
    "demo": ("Run offline demo", "experiments"), "review": ("Start scientific review", "review"),
    "approve": ("Approve this version for writing", "review"), "review_status": ("Validate current approval", "review"),
    "agenda_plan": ("Plan follow-up experiments", "review"), "agenda_run": ("Run reviewed protocol", "review"),
    "pilot_plan": ("Plan an idea pilot", "ideas"), "pilot_run": ("Run idea pilot", "ideas"),
    "iterate": ("Assess next research steps", "experiments"), "write": ("Advance one writing stage", "writing"),
    "write_feedback": ("Give feedback and revise", "writing"), "write_auto": ("Complete remaining writing stages", "writing"),
    "evaluate": ("Research diagnostics", "tools"), "traces": ("Analyze model traces", "tools"),
    "inject_faults": ("Evidence validation fault injection", "tools"), "baseline": ("Single-pass writing baseline", "tools"),
    "decision_bench": ("Offline research decision evaluation", "tools"), "decision_policy": ("Compare model decision strategies", "tools"),
    "suite": ("Multi-task random search evaluation", "tools"), "compare": ("Compare agent and random search", "tools"),
}


def catalog():
    return [{"id": key, "label": LABELS[key][0], "group": LABELS[key][1], "fields": value}
            for key, value in FIELDS.items()]


def validate(action, supplied):
    if action not in FIELDS or not isinstance(supplied, dict):
        raise ValueError("Unknown action or invalid parameters")
    if set(supplied) - {f["name"] for f in FIELDS[action]}:
        raise ValueError("Unknown action parameters")
    result = {}
    for f in FIELDS[action]:
        value = supplied.get(f["name"], f["default"])
        if f.get("required") and (value is None or value == ""):
            raise ValueError(f"{f['label']} is required")
        if f["type"] == "number":
            if isinstance(value, bool):
                raise ValueError("Expected an integer")
            try:
                parsed = int(value)
                if float(value) != parsed or not f["min"] <= parsed <= f["max"]:
                    raise ValueError()
            except (ValueError, TypeError, OverflowError):
                raise ValueError(f"{f['label']}: expected {f['min']}–{f['max']}") from None
            value = parsed
        elif f["type"] == "checkbox":
            if type(value) is not bool:
                raise ValueError("Expected a boolean")
        elif f["type"] == "json":
            if value:
                value = json.loads(value) if isinstance(value, str) else value
                if not isinstance(value, dict):
                    raise ValueError(f"{f['label']}: expected a JSON object")
        else:
            if not isinstance(value, str) or len(value) > 100000:
                raise ValueError("Invalid text value")
            value = value.strip()
            if f.get("required") and not value:
                raise ValueError(f"{f['label']} is required")
            if f["type"] == "select" and value not in f["options"]:
                raise ValueError("Invalid selection")
        result[f["name"]] = value
    return result


def needs_model(action, p):
    return action not in {"select", "inject", "approve", "review_status", "agenda_run", "pilot_run",
                          "demo", "evaluate", "traces", "inject_faults", "decision_bench", "suite"} and not (
        action == "decide" and p["decision"] == "reject" or
        action in {"agenda_plan", "pilot_plan"} and p.get("spec") or
        action == "compare" and p["arms"] == "random")


def writing_directory(run):
    """Bind new drafts to the approved release; never erase an older manuscript."""
    review_path = run / "scientific_review.json"
    if not review_path.exists():
        return run
    review = json.loads(review_path.read_text(encoding="utf-8"))
    release_id = (review.get("release") or {}).get("id")
    if not release_id:
        return run
    if len(release_id) != 64 or any(c not in "0123456789abcdef" for c in release_id):
        raise ValueError("Invalid writing release ID")
    old = run / "writing.json"
    if old.exists() and json.loads(old.read_text(encoding="utf-8")).get("release_id") == release_id:
        return run
    return run / "manuscripts" / release_id


def build_command(action, p, config, run: Path, workspace: Path, job_dir: Path):
    """Build argv from validated data and managed paths, preserving the existing CLI gates."""
    base = [sys.executable, "-u", "-m"]
    profile = str(workspace / "profiles" / config["profile"])
    common = ["--profile", profile, "--ledger", str(run / "experiments.jsonl"),
              "--registry", str(run / "registry"), "--out", str(run)]
    research = ["--research", str(run)] if (run / "session.json").exists() else []
    model = ["--model", config["model"]]
    calls = ["--max-llm-calls", str(config["max_calls"])]

    def doc(name, obj):
        path = job_dir / (name + ".json")
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def add(cmd, name, value):
        if value not in (None, "", False):
            cmd.extend(["--" + name, str(value)])

    if action in {"campaign", "demo", "literature", "directions", "ideas", "refine_ideas"}:
        payload = {"action": action, "params": p, "config": config, "run": str(run), "workspace": str(workspace)}
        return base + ["scholarloop.ui.worker", doc("request", payload)]
    if action in {"select", "inject", "decide"}:
        cmd = base + ["scholarloop.ideation", "--profile", profile, "--out", str(run),
                      "--topic", config["title"], *model, *calls]
        if action == "select":
            cmd += ["--select", str(p["select"]), "--actor", p["author"], "--reason", p["reason"]]
        elif action == "inject":
            cmd += ["--inject-idea", doc("idea", p["idea"]), "--actor", p["author"], "--reason", p["reason"]]
        else:
            cmd += ["--decision", p["decision"], "--idea-ids", *p["idea_ids"].replace(",", " ").split(),
                    "--actor", p["author"], "--reason", p["reason"], "--n-ideas", str(p["n_ideas"])]
        return cmd
    if action in {"review", "approve", "review_status"}:
        cmd = base + ["scholarloop.review", {"review": "review", "approve": "approve", "review_status": "status"}[action],
                      *common, *research]
        if action == "review":
            cmd += model + calls
            add(cmd, "feedback", p["feedback"])
            if p["argument"]:
                cmd += ["--argument", doc("argument", p["argument"])]
            if p["iteration"]:
                cmd += ["--iteration", str(run / "research_iterations.json")]
            for line in p["critic_models"].splitlines():
                add(cmd, "critic-model", line.strip())
        elif action == "approve":
            cmd += ["--author", p["author"], "--review-hash", p["review_hash"]]
        return cmd
    if action.startswith("agenda_") or action.startswith("pilot_"):
        pilot = action.startswith("pilot_")
        cmd = base + ["scholarloop.ideation.pilot" if pilot else "scholarloop.experiment.agenda",
                      action.split("_")[1], *common]
        cmd += ["--research", str(run)] if pilot else research
        if action.endswith("plan"):
            cmd += calls
            add(cmd, "feedback", p["feedback"])
            if p["spec"]:
                cmd += ["--spec", doc("spec", p["spec"])]
            if pilot:
                cmd += ["--idea-ids", *p["idea_ids"].replace(",", " ").split()]
        else:
            cmd += ["--protocol-hash", p["protocol_hash"], "--author", p["author"],
                    "--max-conditions", str(p["max_conditions"])]
        return cmd
    if action == "iterate":
        return base + ["scholarloop.experiment.research_cycle", *common, *research, *calls,
                       "--question", p["question"], "--feedback", p["feedback"],
                       "--max-training-attempts", str(p["training_limit"])]
    if action in {"write", "write_feedback", "write_auto"}:
        writing_common = list(common)
        writing_common[writing_common.index("--out") + 1] = str(writing_directory(run))
        cmd = base + ["scholarloop.writing", *writing_common, "--review-dir", str(run), *research, *model, *calls]
        return cmd + (["--feedback", p["feedback"]] if action == "write_feedback" else
                      ["--auto" if action == "write_auto" else "--approve"])
    if action == "evaluate":
        return base + ["scholarloop.evaluation", "--profile", profile, "--out", str(run), *research,
                       "--output", str(run / "diagnostics.json")]
    if action in {"traces", "inject_faults", "baseline"}:
        kind = "inject" if action == "inject_faults" else action
        cmd = base + ["scholarloop.reliability", kind, "--out", str(run / "reliability" / kind)]
        if action == "traces":
            return cmd + ["--traces", str(run)]
        cmd += ["--profile", profile, "--run", str(run), "--writing", str(writing_directory(run))]
        return cmd + (["--per-case", str(p["per_case"]), "--seed", str(p["seed"])] if action == "inject_faults"
                      else ["--repeats", str(p["repeats"]), *model])
    if action in {"decision_bench", "decision_policy"}:
        cmd = base + ["scholarloop.experiment." + action, "--out", str(run / action),
                      "--repeats", str(p["repeats"]), "--budget", str(p["budget"])]
        return cmd + (["--seed", str(p["seed"])] if action == "decision_bench" else
                      [*model, "--arms", p["arms"], "--max-calls", str(p["calls"])])
    if action == "suite":
        manifest = p["manifest"]
        for task in manifest.get("tasks", []):
            path = (workspace / task.get("profile", "")).resolve()
            if not path.is_relative_to(workspace):
                raise ValueError("Suite profiles must be inside the workspace")
            task["profile"] = str(path)
        return base + ["scholarloop.experiment.suite", "--manifest", doc("manifest", manifest), "--out", str(run / "suite")]
    if action == "compare":
        cmd = base + ["examples.compare_search", "--profile", Path(profile).stem, "--out", str(run / "comparison"),
                      "--max-calls", str(config["max_calls"]), "--topic", config["title"]]
        for key in ("arms", "repeats", "rounds", "match", "warm_start"):
            cmd += ["--" + key.replace("_", "-"), str(p[key])]
        return cmd
    raise ValueError("Unknown action")
