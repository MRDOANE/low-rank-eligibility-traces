from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .cascade import run
from .util import atomic_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the N08 two-benchmark external-validation cascade")
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--mode", choices=("full", "smoke", "verify"), default="full")
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--results-dir", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    try:
        final = run(args.config.resolve(), args.mode, args.data_dir.resolve(), args.results_dir.resolve())
        print(json.dumps(final, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        traceback.print_exc()
        atomic_json(
            args.results_dir / "final_status.json",
            {
                "color": "RED",
                "mode": args.mode,
                "claim_evaluated": False,
                "continue_toward_iclr_claim": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
