#!/usr/bin/env python3
"""
Current release verification (Kaggle T4x2 sole pre-A100 gate) plus legacy mode.

Default (current) authority: validates the current checkout HEAD against the
current freeze via scripts.colab.bootstrap.verify_launch. CPU verification
only; not hardware proof and not spending approval.

Legacy historical validator (single-T4 era) is available only behind
explicit --legacy for old tests/users; never silently falls back.
Production (current) mode never accepts --allow-runtime-changes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.release.provenance import (
    DEFAULT_APPROVAL_PATH,
    DEFAULT_COLAB_REPORT_PATH,
    RELEASE_ONLY_DIFF_ALLOWLIST,
    compute_file_sha256,
    derive_git_head,
    ensure_git_commit,
    get_git_diff_files,
    is_git_ancestor,
    validate_release_approval,
    validate_sha,
    verify_colab_report_invariants,
)

# Compatibility wrapper for existing tests and CLI invocation (legacy semantics preserved)
def validate_release_approval_v2(
    approval: Mapping[str, Any],
    repo_root: Path | str = ".",
    colab_report_path: Path | str | None = None,
    git_head: str | None = None,
    verify_github_actions: bool = False,
    github_token: str | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    import scripts.verify_release_approval as current_mod
    return validate_release_approval(
        approval=approval,
        repo_root=repo_root,
        colab_report_path=colab_report_path,
        git_head=git_head,
        verify_github_actions=verify_github_actions,
        github_token=github_token,
        diff_fn=getattr(current_mod, "get_git_diff_files", None),
        ancestor_fn=getattr(current_mod, "is_git_ancestor", None),
        return_metadata=True,
    )


def _run_current(repo_root: Path, kaggle_report: Path | None, freeze_file: Path | None) -> int:
    """Strict current release check: HEAD vs freeze via common validator."""
    from scripts.colab.bootstrap import verify_launch

    root = Path(repo_root)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True, stderr=subprocess.PIPE
        ).strip()
    except Exception as exc:
        print(f"[-] Failed to derive HEAD: {exc}", file=sys.stderr)
        return 1
    k_path = Path(kaggle_report) if kaggle_report else root / "artifacts/task1/gates/kaggle_t4x2_report.json"
    f_path = Path(freeze_file) if freeze_file else root / "artifacts/task1/freeze/production_freeze.json"
    try:
        freeze = verify_launch(head, k_path, f_path, repo_root=root)
    except Exception as exc:
        print(f"[-] Current release verification FAILED for HEAD {head}: {exc}", file=sys.stderr)
        return 1
    print("=================================================================")
    print("LegalIR Current Release Verification (Kaggle T4x2 gate)")
    print(f"  • Release HEAD : {head}")
    print(f"  • Runtime SHA  : {freeze.get('git_sha')}")
    print(f"  • Freeze       : {f_path}")
    print(f"  • Kaggle report: {k_path}")
    print("=================================================================")
    print("[+] SUCCESS: current HEAD is approved for A100 launch validation (CPU only).")
    print("    This is not hardware proof and not spending approval.")
    return 0


def _run_legacy(args) -> int:
    if not args.approval.exists():
        print(f"[-] Release approval file not found: {args.approval}", file=sys.stderr)
        return 1
    try:
        approval_data = json.loads(args.approval.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[-] Failed to parse release approval JSON: {exc}", file=sys.stderr)
        return 1
    is_valid, errors, meta = validate_release_approval_v2(
        approval_data,
        repo_root=args.repo_root,
        colab_report_path=args.colab_report,
        git_head=args.head,
        verify_github_actions=args.verify_ci_run,
        github_token=args.token,
    )
    print("=================================================================")
    print("LegalIR Release Approval Consistency Gate (LEGACY single-T4 era)")
    print(f"  • Approved Runtime SHA: {meta['runtime_sha']}")
    print(f"  • Actual Release HEAD : {meta['actual_release_head']}")
    print(f"  • Kaggle EXPECTED_COMMIT: {meta['kaggle_expected_commit']}")
    print(f"  • Colab Report SHA-256: {meta['report_sha256']}")
    print("  • Runtime→Release changed files:")
    if meta["changed_files"]:
        for f in meta["changed_files"]:
            print(f"      - {f}")
    else:
        print("      (none - identical commits)")
    print("=================================================================")
    if is_valid:
        print("[+] SUCCESS: Release approval artifact is valid and provenance-consistent.")
        print("[+] Kaggle FULL is authorized on approved runtime commit.")
        return 0
    else:
        if args.allow_runtime_changes and all("changed between" in e or "disallowed" in e.lower() or "lineage" in e.lower() or "does not contain pinned" in e for e in errors):
            print("[*] NOTICE: Runtime changes detected between approved runtime SHA and current HEAD.")
            print("[*] Passing gate because --allow-runtime-changes is enabled (Commit A in two-commit model).")
            return 0
        print("[-] FAILURE: Release approval validation errors detected:", file=sys.stderr)
        for err in errors:
            print(f"    - {err}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify current release (default) or legacy approval (--legacy).")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root path")
    parser.add_argument("--kaggle-report", type=Path, default=None, help="Kaggle T4x2 report (current mode)")
    parser.add_argument("--freeze-file", type=Path, default=None, help="Production freeze (current mode)")
    parser.add_argument("--legacy", action="store_true", help="Use historical single-T4 validator only")
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL_PATH, help="Path to release_approval.json (legacy only)")
    parser.add_argument("--colab-report", type=Path, default=DEFAULT_COLAB_REPORT_PATH, help="Path to colab_smoke_report.json (legacy only)")
    parser.add_argument("--head", type=str, default=None, help="Optional release HEAD override (legacy only)")
    parser.add_argument("--allow-runtime-changes", action="store_true", help="Legacy development bypass; rejected in current production mode")
    parser.add_argument("--verify-ci-run", action="store_true", help="Query GitHub Actions API (legacy only)")
    parser.add_argument("--token", type=str, default=None, help="GitHub token (legacy only)")
    args = parser.parse_args(argv)

    if not args.legacy:
        if args.allow_runtime_changes:
            print("[-] --allow-runtime-changes is rejected in current production mode.", file=sys.stderr)
            return 2
        if args.head is not None or args.verify_ci_run or args.token is not None:
            print("[-] --head/--verify-ci-run/--token are legacy-only; use --legacy to select historical behavior.", file=sys.stderr)
            return 2
        # Default mode never reads legacy release_approval.json.
        return _run_current(args.repo_root, args.kaggle_report, args.freeze_file)
    return _run_legacy(args)


if __name__ == "__main__":
    sys.exit(main())
