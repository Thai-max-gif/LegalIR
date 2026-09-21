"""Cross-platform CPU preflight and Modal dispatch (no image build on --check-only)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="zunuoivalutre")
    parser.add_argument("--env", default="main", dest="environment")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--hf-allow-public-repo", action="store_true")
    parser.add_argument("--bypass-t4-gate", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        sha = os.environ.get("LEGALIR_COMMIT_SHA", head)
        if not re.fullmatch(r"[0-9a-f]{40}", sha) or sha != head:
            raise ValueError("LEGALIR_COMMIT_SHA must equal the exact local HEAD")
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True
        ).strip()
        if dirty:
            raise ValueError("Working tree is dirty; publish an evidence-bearing release before dispatch")
        timeout = int(os.environ.get("MODAL_TIMEOUT_SECONDS", "25200"))
        if not 1 <= timeout <= 86400:
            raise ValueError("MODAL_TIMEOUT_SECONDS must be between 1 and 86400")
        if int(os.environ.get("LEGALIR_TIME_GATE_SECONDS", timeout)) != timeout:
            raise ValueError("LEGALIR_TIME_GATE_SECONDS must equal MODAL_TIMEOUT_SECONDS")
        from scripts.colab.bootstrap import verify_launch

        verify_launch(
            sha,
            REPO_ROOT / "artifacts/task1/gates/kaggle_t4x2_report.json",
            REPO_ROOT / "artifacts/task1/freeze/production_freeze.json",
            repo_root=REPO_ROOT,
        )
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"[!] Launch blocked before Modal dispatch: {exc}", file=sys.stderr)
        return 1

    print(f"[*] Profile={args.profile}; environment={args.environment}; SHA={sha}")
    print(f"[*] Execution timeout / acceptance ceiling: {timeout}s")
    if args.check_only:
        print("[+] Local CPU preflight PASS. Remote credentials and hardware are not checked.")
        return 0
    env = os.environ.copy()
    env.update(LEGALIR_COMMIT_SHA=sha, MODAL_TIMEOUT_SECONDS=str(timeout),
               LEGALIR_TIME_GATE_SECONDS=str(timeout))
    if args.bypass_t4_gate:
        env["LEGALIR_BYPASS_T4_GATE"] = "1"
    command = [sys.executable, "-m", "modal", "run", "--profile", args.profile,
               "--env", args.environment]
    if args.detach:
        command.append("--detach")
    command.append("scripts/modal/run_modal_a100.py")
    command.append("--hf-allow-public-repo" if args.hf_allow_public_repo
                   else "--no-hf-allow-public-repo")
    if args.private:
        command.append("--private")
    if args.bypass_t4_gate:
        command.append("--bypass-t4-gate")
        print("[!] OPERATOR BYPASS: T4 evidence check skipped; dataset/config checks remain.")
    print("[*] Record the app ID and Volume attempt path; confirm shutdown after completion.")
    if args.detach:
        print("[*] Detached: remote execution continues if this client disconnects.")
    return subprocess.run(command, cwd=REPO_ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
