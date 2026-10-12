"""Export a clean text-only copy of a Qwen3.5 checkpoint for training and vLLM serving.

    uv run python -m train.export_text --model Qwen/Qwen3.5-4B
    # -> runs/models/Qwen3.5-4B-text  (architecture Qwen3_5ForCausalLM, no vision tower)

Run this before starting a `trl vllm-serve` server, which must serve the same text-only weights
the trainer syncs into it. The other scripts call the same export automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import common  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Hub id, e.g. Qwen/Qwen3.5-2B")
    parser.add_argument("--force", action="store_true", help="re-export even if present")
    args = parser.parse_args()
    out = common.export_text_only(args.model, force=args.force)
    print(json.dumps(json.loads((out / "export_info.json").read_text()), indent=2))
    print(out)


if __name__ == "__main__":
    main()
