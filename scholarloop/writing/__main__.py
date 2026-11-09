"""`python -m scholarloop.writing`: run or resume a WritingSession (see session.py)."""

import sys

from scholarloop.writing.session import _main

raise SystemExit(_main(sys.argv[1:]))
