#!/usr/bin/env python3
"""
CLI tool and release-governance gate to verify that release_approval.json satisfies
all release provenance, Git lineage, Colab T4 invariants, and Kaggle pin requirements.

Authoritative specification: LEGALIR_88E1_ARCHITECTURE_REPAIR.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

# Compatibility wrapper for existing tests and CLI invocation
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
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify release approval artifact consistency.")
    parser.add_argument(
        "--approval",
        type=Path,
        default=DEFAULT_APPROVAL_PATH,
        help="Path to release_approval.json",
    )
    parser.add_argument(
        "--colab-report",
        type=Path,
        default=DEFAULT_COLAB_REPORT_PATH,
        help="Path to colab_smoke_report.json",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root path",
    )
    parser.add_argument(
        "--head",
        type=str,
        default=None,
        help="Optional release HEAD commit override (defaults to 'git rev-parse HEAD')",
    )
    parser.add_argument(
        "--allow-runtime-changes",
        action="store_true",
        help="Allow runtime code differences between approved runtime commit and current HEAD during development/CI testing before release commit",
    )
    parser.add_argument(
        "--verify-ci-run",
        action="store_true",
        help="Query GitHub Actions API to verify that runtime commit CI was successful",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional GitHub personal access token for API rate limits",
    )

    args = parser.parse_args()

    if not args.approval.exists():
        print(f"[-] Release approval file not found: {args.approval}", file=sys.stderr)
        return 1

    try:
        approval_data = json.loads(args.approval.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[-] Failed to parse release approval JSON: {exc}", file=sys.stderr)
        return 1

    is_valid, errors, meta = validate_release_approval(
        approval_data,
        repo_root=args.repo_root,
        colab_report_path=args.colab_report,
        git_head=args.head,
        verify_github_actions=args.verify_ci_run,
        github_token=args.token,
    )

    print("=================================================================")
    print("LegalIR Release Approval Consistency Gate")
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
        # If allow-runtime-changes is set and all errors are disallowed file changes/lineage diffs
        if args.allow_runtime_changes and all("changed between runtime" in e or "lineage" in e.lower() for e in errors):
            print("[*] NOTICE: Runtime changes detected between approved runtime SHA and current HEAD.")
            print("[*] Passing gate because --allow-runtime-changes is enabled (Commit A in two-commit model).")
            return 0

        print("[-] FAILURE: Release approval validation errors detected:", file=sys.stderr)
        for err in errors:
            print(f"    - {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
