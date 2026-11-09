"""`python -m scholarloop.ideation`: run or resume a ResearchSession (see session.py)."""

import sys

from scholarloop.ideation.session import _main

raise SystemExit(_main(sys.argv[1:]))
