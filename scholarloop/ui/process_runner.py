"""Run one allowlisted command under a campaign lock, with a durable outcome."""
import json
import runpy
import sys
import traceback
from pathlib import Path

from scholarloop.core.locking import exclusive_lock
from scholarloop.review.contracts import atomic_json


def main(request_path):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    argv = request["argv"]
    code = 1
    try:
        with exclusive_lock(Path(request["run"]) / ".ui-execution.lock"):
            # argv was built by catalog.build_command; there is no shell or free-form command input.
            module_index = argv.index("-m") + 1
            sys.argv = argv[module_index:]
            try:
                runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
                code = 0
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception:
        traceback.print_exc()
    finally:
        atomic_json(request_path.parent / "outcome.json", {"exit_code": code})
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
