#!/usr/bin/env python3
"""
Verify that a target Git commit SHA has a GREEN completed 'LegalIR CI' workflow run on GitHub.

Authoritative specification: LEGALIR_CI_COLAB_KAGGLE_ARCHITECTURE_SPEC.md
Implementation plan: LEGALIR_CI_COLAB_KAGGLE_IMPLEMENTATION_PLAN.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from src.release.provenance import check_github_ci_run, DEFAULT_REPO, DEFAULT_WORKFLOW_NAME

WORKFLOW_NAME = DEFAULT_WORKFLOW_NAME


def check_ci_status(
    repo: str = DEFAULT_REPO,
    sha: str = "",
    token: str | None = None,
) -> tuple[bool, str]:
    """
    Check if the target SHA has a completed successful 'LegalIR CI' workflow run on GitHub.

    Returns:
        tuple[bool, str]: (is_green, diagnostic_message)
    """
    if not sha or not re.match(r"^[0-9a-fA-F]{7,40}$", sha):
        return False, f"Invalid Git SHA format: '{sha}'. Must be 7-40 hex characters."

    token = token or os.environ.get("GITHUB_TOKEN") or None

    encoded_sha = urllib.parse.quote(sha)
    url = f"https://api.github.com/repos/{repo}/actions/runs?head_sha={encoded_sha}"

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "LegalIR-CI-Verifier/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            return (
                False,
                f"GitHub API HTTP 403 (Rate limit exceeded or access denied). "
                f"Please provide a GITHUB_TOKEN via environment variable or --token. Fail-closed.",
            )
        elif exc.code == 404:
            return False, f"GitHub API HTTP 404: Repository '{repo}' not found or inaccessible."
        return False, f"GitHub API HTTP {exc.code}: {exc.reason}. Verification failed closed."
    except Exception as exc:
        return False, f"Network/API request error: {exc}. Verification failed closed."

    workflow_runs = data.get("workflow_runs", [])
    if not workflow_runs:
        return (
            False,
            f"No workflow runs found on GitHub for SHA {sha} in {repo}. "
            f"Please ensure code has been pushed and '{WORKFLOW_NAME}' triggered.",
        )

    # Find runs matching WORKFLOW_NAME and exact head_sha
    matching_runs = [
        r for r in workflow_runs
        if r.get("name") == WORKFLOW_NAME and (
            r.get("head_sha", "").lower().startswith(sha.lower())
            or sha.lower().startswith(r.get("head_sha", "").lower())
        )
    ]

    if not matching_runs:
        found_names = [r.get("name") for r in workflow_runs]
        return (
            False,
            f"No matching '{WORKFLOW_NAME}' workflow runs found for SHA {sha}. "
            f"Found other workflows: {found_names}",
        )

    # Check the latest matching run
    for run in matching_runs:
        status = run.get("status")
        conclusion = run.get("conclusion")
        run_url = run.get("html_url", "")
        run_id = run.get("id")

        if status == "completed" and conclusion == "success":
            return (
                True,
                f"SUCCESS: '{WORKFLOW_NAME}' is GREEN for SHA {sha} (Run ID: {run_id}, URL: {run_url}).",
            )
        elif status == "completed" and conclusion != "success":
            return (
                False,
                f"FAILURE: '{WORKFLOW_NAME}' completed with conclusion='{conclusion}' for SHA {sha} (URL: {run_url}).",
            )
        else:
            return (
                False,
                f"IN PROGRESS: '{WORKFLOW_NAME}' run status is '{status}' (conclusion='{conclusion}') for SHA {sha} (URL: {run_url}).",
            )

    return False, f"No completed green '{WORKFLOW_NAME}' run found for SHA {sha}."


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that GitHub CI is GREEN for a given commit SHA.")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO, help=f"GitHub repo (default: {DEFAULT_REPO})")
    parser.add_argument("--sha", type=str, required=True, help="Target commit SHA (40 hex chars)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional specific GitHub Actions run ID to verify")
    parser.add_argument("--token", type=str, default=None, help="GitHub Personal Access Token (optional)")
    args = parser.parse_args()

    print("=================================================================")
    print("LegalIR CI Status Verification Gate")
    print(f"  • Repo      : {args.repo}")
    print(f"  • Target SHA: {args.sha}")
    if args.run_id:
        print(f"  • Run ID    : {args.run_id}")
    print(f"  • Workflow  : {WORKFLOW_NAME}")
    print("=================================================================")

    if args.run_id:
        is_green, msg = check_github_ci_run(run_id=args.run_id, expected_runtime_sha=args.sha, repo=args.repo, token=args.token)
    else:
        is_green, msg = check_ci_status(repo=args.repo, sha=args.sha, token=args.token)
    print(msg)

    if is_green:
        print("[+] CI GATE PASSED: Commit is approved for Colab T4 execution.")
        return 0
    else:
        print("[-] CI GATE FAILED: Target commit is NOT verified green.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
