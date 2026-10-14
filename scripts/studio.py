"""Control an rl-env-lab Lightning studio (node) from a laptop.

Usage:
    python scripts/studio.py status                       # rl-env-lab-n1 by default
    python scripts/studio.py --studio rl-env-lab-n2 status
    python scripts/studio.py switch H200_X_8
    python scripts/studio.py run "make test"

Needs `lightning-sdk` and a Lightning login (`lightning login`). Set `LIGHTNING_ORG` and `LIGHTNING_TEAMSPACE`
if the studios are not in your default teamspace. Commands run in ~/rl-env-lab.
"""

from __future__ import annotations

import argparse
import os
import sys

STUDIO = "rl-env-lab-n1"  # the nodes: rl-env-lab-n1, rl-env-lab-n2
TEAMSPACE = os.environ.get("LIGHTNING_TEAMSPACE")  # None: the SDK default for your login
ORG = os.environ.get("LIGHTNING_ORG")
REPO_DIR = "~/rl-env-lab"


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and act on the studio.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        The process exit code.
    """
    from lightning_sdk import Machine, Studio

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studio", default=STUDIO, help=f"studio name (default {STUDIO})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sw = sub.add_parser("switch")
    sw.add_argument("machine", help="Machine name, e.g. H200 or H200_X_8")
    rn = sub.add_parser("run")
    rn.add_argument("command")
    args = ap.parse_args(argv)

    studio = Studio(name=args.studio, teamspace=TEAMSPACE, org=ORG)
    if args.cmd == "status":
        print(f"status={studio.status} machine={studio.machine} auto_sleep={studio.auto_sleep}")
        return 0
    if args.cmd == "switch":
        studio.switch_machine(getattr(Machine, args.machine))
        print(f"switched to {studio.machine}")
        return 0
    output, code = studio.run_with_exit_code(f"cd {REPO_DIR} && {args.command}")
    print(output)
    return code


if __name__ == "__main__":
    sys.exit(main())
