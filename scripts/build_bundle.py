from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.manifest import write_checksums


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-validation", required=True)
    parser.add_argument("--smoke-report")
    args = parser.parse_args()
    run = Path(args.run_dir)
    for source in [Path(args.dataset_validation), Path(args.smoke_report) if args.smoke_report else None]:
        if source and source.is_file():
            shutil.copy2(source, run / source.name)
    write_checksums(run)
    print(run / "checksums.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
