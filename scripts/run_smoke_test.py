from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")


def call(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--set", action="append", default=[], help="additional config override key=value")
    args = parser.parse_args()
    override = f"data.dataset_root={args.dataset_root}"
    overrides = [override, *args.set]
    # The notebook exposes this writable path as an environment variable so
    # preflight and DDP training always write/read the same report directory.
    if os.environ.get("LEGALIR_OUTPUT_DIR") and not any(value.startswith("experiment.output_dir=") for value in overrides):
        overrides.append(f"experiment.output_dir={os.environ['LEGALIR_OUTPUT_DIR']}")
    override_args = [item for value in overrides for item in ("--set", value)]
    # Do not launch DDP when immutable pairs/index/split provenance is blocked.
    call([sys.executable, "scripts/run_preflight.py", "--config", args.config, "--mode", "smoke", *override_args])
    call([sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(args.num_processes), "--multi_gpu", "scripts/run_train.py", "--config", args.config, *override_args])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
