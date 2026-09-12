#!/usr/bin/env python3
"""
Static check to verify no unauthorized fallback models or silent CPU fallbacks
exist in authoritative GPU gate scripts.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

GATE_SCRIPTS: list[Path] = [
    REPO_ROOT / "scripts" / "gates" / "run_kaggle_t4x2.py",
    REPO_ROOT / "scripts" / "gates" / "run_colab_t4.py",
    REPO_ROOT / "scripts" / "gates" / "run_a100.py",
]

FORBIDDEN_FALLBACK_PATTERNS: list[str] = [
    "BertConfig(",
    "BertForSequenceClassification(",
    "AutoModelForSequenceClassification.from_config(",
]


def check_script_for_fallbacks(script_path: Path) -> list[str]:
    """Scan a script for forbidden fallback model patterns."""
    if not script_path.is_file():
        return [f"File missing: {script_path}"]

    errors = []
    content = script_path.read_text(encoding="utf-8")

    for pat in FORBIDDEN_FALLBACK_PATTERNS:
        if pat in content:
            errors.append(f"Found forbidden model fallback '{pat}' in {script_path.name}")

    # Check for silent CUDA -> CPU fallback patterns in try/except
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if "device = \"cpu\"" in line or "device = 'cpu'" in line:
            # Check if this occurs inside an authoritative block (outside mock)
            if "if not mock" in "\n".join(lines[max(0, idx - 10) : idx]):
                errors.append(f"Forbidden silent CPU fallback detected at line {idx + 1} in {script_path.name}")

    return errors


def main() -> int:
    print("[*] Checking: Authoritative GPU gates for forbidden fallbacks ...")
    all_errors = []
    for script in GATE_SCRIPTS:
        errors = check_script_for_fallbacks(script)
        if errors:
            all_errors.extend(errors)

    if all_errors:
        print("[!] FAILED: Found forbidden fallbacks in authoritative gates:")
        for err in all_errors:
            print(f"    - {err}")
        return 1

    print("[+] PASSED: No forbidden fallbacks detected in GPU gates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
