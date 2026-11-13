#!/usr/bin/env bash
# End-to-end NLP case study on AG News, under one hard dollar cap shared by every stage:
#   your steer -> literature review -> direction map -> YOU choose directions -> ideas ->
#   tournament -> YOU pick an idea -> campaign that must test the picked idea
#   (control vs treatment through the train-script edit channel) -> scientific review ->
#   your approval -> staged write-up -> reliability evaluation of the guards.
#
#   export ANTHROPIC_API_KEY=sk-ant-...
#   pip install -e ".[dev,engines,llm,nlp]"
#   bash examples/case_study_agnews.sh            # USES THE API; stops at CAP dollars
#
# Run it in a terminal: the scientific review and the write-up ask for your decision. Every
# stage resumes from its saved state, so after a pause or a reached cap, rerun the same command
# (raise CAP to continue past the cap). Environment: OUT, CAP (USD), MODEL, TOPIC, STEER (a JSON
# steer, default examples/steer.json), and to answer the two ideation checkpoints from the shell:
# DIRECTIONS (e.g. "1,3", numbers on $OUT/directions.md) and SELECT (a number on $OUT/ideas.md).
set -euo pipefail

OUT=${OUT:-runs/agnews_case}
CAP=${CAP:-10}
PROFILE=profiles/agnews-bow.yaml
TOPIC=${TOPIC:-"regularization for small text classifiers trained on limited labelled data"}
export SCHOLARLOOP_MODEL=${MODEL:-claude-opus-5-5}
export SCHOLARLOOP_SPEND_CAP=$CAP                  # hard cap, checked before every request
export SCHOLARLOOP_SPEND_FILE=$OUT/spend.json      # one budget for all stages below
mkdir -p "$OUT"

spent() { python3 -c "import json,sys;print(f\"spent \${json.load(open(sys.argv[1]))['usd']:.2f} of \$$CAP\")" "$OUT/spend.json" 2>/dev/null || true; }

echo "== 1/4 ideation: steer, literature review, direction map, ideas, tournament =="
IDEATE="python3 -m scholarloop.ideation --profile $PROFILE --out $OUT --deep --max-llm-calls 80"
if [ ! -f "$OUT/handoff.json" ]; then
  if [ ! -f "$OUT/directions.md" ]; then
    $IDEATE --topic "$TOPIC" --steer "${STEER:-examples/steer.json}" --stop-after directions
  fi
  if [ ! -f "$OUT/ideas.md" ]; then
    [ -n "${DIRECTIONS:-}" ] || { echo "Read $OUT/directions.md, then rerun with DIRECTIONS=1,3"; spent; exit 0; }
    $IDEATE --directions "$DIRECTIONS" --stop-after ideas
  fi
  [ -n "${SELECT:-}" ] || { echo "Read $OUT/ideas.md, then rerun with SELECT=<n>"; spent; exit 0; }
  $IDEATE --select "$SELECT"
fi
spent

echo "== 2/4 campaign, scientific review, write-up =="
RESUME=()
[ -f "$OUT/experiments.jsonl" ] && RESUME=(SCHOLARLOOP_PAPER_ONLY=1)   # never rerun a saved campaign
env ${RESUME[@]+"${RESUME[@]}"} SCHOLARLOOP_PROFILE=agnews-bow SCHOLARLOOP_HANDOFF="$OUT/handoff.json" SCHOLARLOOP_OUT="$OUT" \
  SCHOLARLOOP_POPULATION=3 SCHOLARLOOP_MAX_WORKERS=3 SCHOLARLOOP_STEPS=4 SCHOLARLOOP_DRY_PATIENCE=2 \
  python3 examples/run_to_paper.py || echo "(paused: see the message above, then rerun this script)"
spent

echo "== 3/4 guard reliability: natural fault rate and fault injection (free) =="
python3 -m scholarloop.reliability traces --traces "$OUT" --out "$OUT/reliability" || true
if [ -f "$OUT/writing.json" ]; then
  python3 -m scholarloop.reliability inject --profile $PROFILE --run "$OUT" --out "$OUT/reliability" > /dev/null
  echo "   wrote $OUT/reliability/injection.md"
fi

echo "== 4/4 single-call baseline (3 requests) =="
if [ -f "$OUT/writing.json" ]; then
  python3 -m scholarloop.reliability baseline --profile $PROFILE --run "$OUT" --out "$OUT/reliability" --repeats 3
fi
spent
