#!/usr/bin/env python3
"""
Single-Command Pre-Push Verification Gate for LegalIR Task 1.

Executes all fail-closed gates locally before pushing commits to GitHub:
1. Python syntax compilation (compileall on src and scripts)
2. Modular pytest suites (unit, dataset, notebook, parity, leakage, memory, integration, release)
3. Competition parameter budget audit (< 4B learned parameters)
4. Notebook zero-drift check (scripts/generate_notebooks.py --check-drift)
5. Offline Kaggle pipeline smoke check (--tiny --run-mode smoke)
6. Git working tree hygiene audit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_gate(cmd: list[str], description: str) -> bool:
    """Run a gate command and print structured status."""
    print(f"[*] Checking: {description} ...")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if res.returncode != 0:
        print(f"[!] FAILED: {description}")
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        return False
    print(f"[+] PASSED: {description}")
    return True


def check_git_hygiene() -> bool:
    """Check for untracked heavy datasets or dirty state."""
    print("[*] Checking: Git working tree hygiene ...")
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO_ROOT)
    lines = res.stdout.strip().splitlines()
    heavy_artifacts = [line for line in lines if any(ext in line for ext in (".parquet", ".zip", ".tar.gz", ".safetensors", ".pt"))]
    if heavy_artifacts:
        print(f"[!] WARNING: Found untracked heavy artifacts in git status:\n" + "\n".join(heavy_artifacts))
        print("    Ensure large data files are placed on Kaggle or ignored in .gitignore.", file=sys.stderr)
        return False
    print("[+] PASSED: Git working tree hygiene")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="LegalIR Pre-Push Verification Gate")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest test suite")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip offline pipeline smoke")
    args = parser.parse_args()

    python_bin = sys.executable

    print("=================================================================")
    print("LegalIR Pre-Push Verification Gate (All Systems)")
    print(f"  • Python Binary: {python_bin}")
    print(f"  • Repo Root    : {REPO_ROOT}")
    print("=================================================================")

    # 1. Compileall
    if not run_gate([python_bin, "-m", "compileall", "-q", "src", "scripts"], "Python syntax compilation"):
        return 1

    # 2. Pytest suites
    if not args.skip_tests:
        test_suites = [
            "tests/unit",
            "tests/contracts",
            "tests/dataset",
            "tests/notebook",
            "tests/parity",
            "tests/leakage",
            "tests/memory",
            "tests/integration",
            "tests/release",
        ]
        existing_suites = [s for s in test_suites if (REPO_ROOT / s).is_dir()]
        cmd = [python_bin, "-m", "pytest", "-q"] + existing_suites
        if not run_gate(cmd, f"Modular test suites ({', '.join(existing_suites)})"):
            return 1

    # 3. Parameter audit
    audit_script = REPO_ROOT / "scripts" / "audit_parameters.py"
    if audit_script.is_file():
        if not run_gate([python_bin, str(audit_script), "--check-only"], "Learned parameter budget (< 4B)"):
            return 1

    # 4. Notebook zero-drift check
    gen_script = REPO_ROOT / "scripts" / "generate_notebooks.py"
    if gen_script.is_file():
        if not run_gate([python_bin, str(gen_script), "--check-drift"], "Notebooks zero-drift check"):
            return 1

    # 5. Check no fallbacks
    fallbacks_script = REPO_ROOT / "scripts" / "check_no_fallbacks.py"
    if fallbacks_script.is_file():
        if not run_gate([python_bin, str(fallbacks_script)], "Forbidden fallback detection"):
            return 1

    # 6. Offline pipeline smoke
    if not args.skip_pipeline:
        smoke_script = REPO_ROOT / "scripts" / "smoke_kaggle_pipeline.py"
        if smoke_script.is_file():
            if not run_gate([python_bin, str(smoke_script), "--tiny", "--run-mode", "smoke"], "Offline Kaggle pipeline smoke"):
                return 1

    # 7. Git hygiene (Fail-Closed)
    if not check_git_hygiene():
        print("[!] FAILED: Git working tree hygiene audit failed.", file=sys.stderr)
        return 1

    print("\n=================================================================")
    print("[+] ALL PRE-PUSH VERIFICATION GATES PASSED.")
    print("    Safe to commit and push to GitHub!")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
