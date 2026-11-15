#!/usr/bin/env bash
# Compatibility entry point; the Python runner also supports Windows PowerShell.
# Existing OUT, CAP, MODEL, TOPIC, STEER, DIRECTIONS and SELECT variables still work.
set -euo pipefail
exec python3 "$(dirname "$0")/case_study_agnews.py" "$@"
