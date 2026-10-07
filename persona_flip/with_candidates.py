"""Run a persona_flip module with the wording-test candidates registered as personas (common.py is frozen).

  python -m persona_flip.with_candidates extract_ladder chats --out runs/candidates --personas none C_implicit_01 ...
  python -m persona_flip.with_candidates probe --model-key base --name probe_cand --personas none C_implicit_01 ...
"""
from __future__ import annotations

import importlib
import sys

from .candidates import CANDIDATES
from .common import PERSONAS


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    PERSONAS.update(CANDIDATES)   # the same dict the other modules read their system prompts from
    importlib.import_module(f"persona_flip.{sys.argv[1]}").main(sys.argv[2:])


if __name__ == "__main__":
    main()
