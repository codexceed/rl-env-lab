"""Node preflight before launching anything: ``make preflight NODE=node1``.

    uv run python scripts/preflight.py --node node1
    uv run python scripts/preflight.py                  # no node: GPU stack, ninja, GPU count only

GPU stack imports, ``ninja``, GPU count, data against its manifest, text-only model exports and free disk; a
pass/fail table and exit 1 on any failure. The checks live in `train/preflight.py`, which `train/coordinator.py`
also runs at startup (``--skip-preflight`` to override).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.preflight import main

if __name__ == "__main__":
    sys.exit(main())
