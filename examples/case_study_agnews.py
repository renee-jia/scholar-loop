"""Portable AG News case study; preserves the shared budget and human checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=os.environ.get("OUT", "runs/agnews_case"))
    parser.add_argument("--cap", type=float, default=os.environ.get("CAP", "10"))
    parser.add_argument("--model", default=os.environ.get("MODEL", "claude-opus-5-5"))
    parser.add_argument("--topic", default=os.environ.get("TOPIC", "regularization for small text classifiers trained on limited labelled data"))
    parser.add_argument("--steer", default=os.environ.get("STEER", "examples/steer.json"))
    parser.add_argument("--directions", default=os.environ.get("DIRECTIONS", ""))
    parser.add_argument("--select", default=os.environ.get("SELECT", ""))
    args = parser.parse_args(argv)
    if not math.isfinite(args.cap) or args.cap <= 0:
        parser.error("--cap must be a positive finite dollar amount")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile = str(ROOT / "profiles/agnews-bow.yaml")
    env = {**os.environ, "PYTHONUTF8": "1", "SCHOLARLOOP_MODEL": args.model,
           "SCHOLARLOOP_SPEND_CAP": str(args.cap), "SCHOLARLOOP_SPEND_FILE": str(out / "spend.json")}

    def run(command, *, check=True, quiet=False, extra_env=None):
        result = subprocess.run([sys.executable, *command], cwd=ROOT, env={**env, **(extra_env or {})},
                                stdout=subprocess.DEVNULL if quiet else None)
        if check and result.returncode:
            raise SystemExit(result.returncode)
        return result.returncode

    def spent():
        if (out / "spend.json").is_file():
            data = json.loads((out / "spend.json").read_text(encoding="utf-8"))
            print(f"spent ${data['usd']:.2f} of ${args.cap:g}", flush=True)

    print("== 1/4 ideation: steer, literature review, direction map, ideas, tournament ==", flush=True)
    ideate = ["-m", "scholarloop.ideation", "--profile", profile, "--out", str(out), "--deep", "--max-llm-calls", "80"]
    if not (out / "handoff.json").exists():
        if not (out / "directions.md").exists():
            run([*ideate, "--topic", args.topic, "--steer", str(Path(args.steer).resolve()), "--stop-after", "directions"])
        if not (out / "ideas.md").exists():
            if not args.directions:
                print(f"Read {out / 'directions.md'}, then rerun with --directions 1,3")
                spent()
                return 0
            run([*ideate, "--directions", args.directions, "--stop-after", "ideas"])
        if not args.select:
            print(f"Read {out / 'ideas.md'}, then rerun with --select <n>")
            spent()
            return 0
        run([*ideate, "--select", args.select])
    spent()

    print("== 2/4 campaign, scientific review, write-up ==", flush=True)
    campaign_env = {"SCHOLARLOOP_PROFILE": "agnews-bow", "SCHOLARLOOP_HANDOFF": str(out / "handoff.json"),
                    "SCHOLARLOOP_OUT": str(out), "SCHOLARLOOP_POPULATION": "3", "SCHOLARLOOP_MAX_WORKERS": "3",
                    "SCHOLARLOOP_STEPS": "4", "SCHOLARLOOP_DRY_PATIENCE": "2"}
    if (out / "experiments.jsonl").exists():
        campaign_env["SCHOLARLOOP_PAPER_ONLY"] = "1"
    if run([str(ROOT / "examples/run_to_paper.py")], check=False, extra_env=campaign_env):
        print("(paused: see the message above, then rerun this script)")
    spent()

    print("== 3/4 guard reliability: natural fault rate and fault injection (free) ==", flush=True)
    reliability = ["-m", "scholarloop.reliability"]
    run([*reliability, "traces", "--traces", str(out), "--out", str(out / "reliability")], check=False)
    common = ["--profile", profile, "--run", str(out), "--out", str(out / "reliability")]
    if (out / "writing.json").exists():
        run([*reliability, "inject", *common], quiet=True)
        print(f"   wrote {out / 'reliability/injection.md'}")
    print("== 4/4 single-call baseline (3 requests) ==", flush=True)
    if (out / "writing.json").exists():
        run([*reliability, "baseline", *common, "--repeats", "3"])
    spent()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
