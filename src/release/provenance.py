"""
Release Provenance, SHA Validation, and Release Approval Artifact Governance.
Authoritative single source of truth for LegalIR Release Invariants.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_APPROVAL_PATH = REPO_ROOT / "artifacts" / "task1" / "release_approval.json"
DEFAULT_COLAB_REPORT_PATH = REPO_ROOT / "artifacts" / "task1" / "colab_smoke_report.json"

SHA_REGEX = re.compile(r"^[0-9a-fA-F]{40}$")
DEFAULT_REPO = "silent9669/LegalIR"
DEFAULT_WORKFLOW_NAME = "LegalIR CI"

RELEASE_ONLY_DIFF_ALLOWLIST: tuple[str, ...] = (
    "artifacts/task1/colab_smoke_report.json",
    "artifacts/task1/release_approval.json",
    "artifacts/task1/gates/kaggle_t4x2_report.json",
    "artifacts/task1/gates/colab_t4_report.json",
    "artifacts/task1/freeze/production_freeze.json",
    "parameter_audit.json",
    "scripts/generate_notebooks.py",
    "scripts/generate_kaggle_notebook.py",
    "scripts/generate_colab_smoke_notebook.py",
    "legalir_training.ipynb",
    "kaggle_kernel_task1/legalir_training.ipynb",
    "kaggle_kernel/legalir_training.ipynb",
    "kaggle_kernel/legalqa_gpu_pipeline.ipynb",
    "notebooks/kaggle_smoke.ipynb",
    "notebooks/kaggle_t4x2_smoke.ipynb",
    "notebooks/colab_a100_train.ipynb",
    "notebooks/kaggle_final.ipynb",
    "notebooks/colab_t4_smoke.ipynb",
    "colab/legalir_t4_smoke.ipynb",
    "scripts/verify_release_approval.py",
    "tests/test_release_approval_head_gate.py",
    "tests/release/test_release_ci_binding.py",
    ".github/workflows/ci.yml",
)

EXPLICIT_DISALLOWED_PREFIXES: tuple[str, ...] = (
    "src/pipeline/",
    "src/retrieval/",
    "src/ranking/",
    "src/training/",
    "configs/",
)

EXPLICIT_DISALLOWED_FILES: tuple[str, ...] = (
    "requirements.txt",
)


def validate_sha(sha: str) -> bool:
    """Validate that a string is an exact 40-character hexadecimal Git commit SHA."""
    return bool(sha and isinstance(sha, str) and SHA_REGEX.match(sha.strip()))


def validate_runtime_release_lineage(
    runtime_sha: str,
    release_sha: str,
    repo_root: Union[Path, str] = ".",
) -> Tuple[bool, list[str]]:
    """Enforce the two-commit release model for A100 execution.

    The frozen runtime commit (where GPU gate evidence was produced) must equal
    the release checkout, or be a git ancestor of it with only allowlisted
    evidence-file diffs (gate reports, freeze, notebooks, parameter audit).
    Any code/config change between runtime and release invalidates the evidence.
    Returns (is_valid, errors).
    """
    errors: list[str] = []
    runtime = str(runtime_sha or "").strip().lower()
    release = str(release_sha or "").strip().lower()
    if not validate_sha(runtime):
        errors.append(f"Invalid runtime_sha format: '{runtime_sha}'. Must be exact 40-hex Git SHA.")
        return False, errors
    if not validate_sha(release):
        errors.append(f"Invalid release_sha format: '{release_sha}'. Must be exact 40-hex Git SHA.")
        return False, errors
    if runtime == release:
        return True, []
    if not is_git_ancestor(runtime, release, repo_root):
        errors.append(
            f"Runtime {runtime} is not an ancestor of release {release}; "
            f"gate evidence does not lineage-bind to this checkout."
        )
        return False, errors
    changed = get_git_diff_files(runtime, release, repo_root)
    disallowed = [f for f in changed if f not in RELEASE_ONLY_DIFF_ALLOWLIST]
    if disallowed:
        errors.append(
            f"Non-evidence changes between runtime {runtime[:7]} and release {release[:7]} "
            f"invalidate gate evidence: {disallowed}"
        )
        return False, errors
    return True, []


@dataclasses.dataclass(frozen=True)
class ReleaseApproval:
    schema_version: int
    runtime_sha: str
    ci: dict[str, Any]
    colab: dict[str, Any]
    production: dict[str, Any]
    approved_for_kaggle_full: bool
    dataset: dict[str, Any] = dataclasses.field(default_factory=dict)
    release_sha: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compute_file_sha256(path: Union[Path, str]) -> str:
    """Compute standard hexadecimal SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def derive_git_head(repo_root: Union[Path, str]) -> str:
    """Derive actual release HEAD via git rev-parse HEAD."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        return head
    except Exception as exc:
        raise RuntimeError(f"Failed to derive Git HEAD via 'git rev-parse HEAD': {exc}") from exc


def ensure_git_commit(commit_sha: str, repo_root: Union[Path, str]) -> None:
    """Ensure commit object is available locally, fetching if repository is shallow."""
    try:
        check = subprocess.run(
            ["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
        )
        if check.returncode != 0:
            subprocess.run(
                ["git", "fetch", "--depth=100", "origin", commit_sha],
                cwd=repo_root,
                capture_output=True,
            )
    except Exception:
        pass


def is_git_ancestor(ancestor_sha: str, descendant_sha: str, repo_root: Union[Path, str]) -> bool:
    """Check if ancestor_sha is an ancestor of descendant_sha."""
    ensure_git_commit(ancestor_sha, repo_root)
    ensure_git_commit(descendant_sha, repo_root)
    try:
        ret = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor_sha, descendant_sha],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        return ret.returncode == 0
    except Exception:
        return False


def get_git_diff_files(base_sha: str, head_sha: str, repo_root: Union[Path, str]) -> list[str]:
    """Get list of changed file paths between base_sha and head_sha."""
    ensure_git_commit(base_sha, repo_root)
    ensure_git_commit(head_sha, repo_root)
    try:
        diff_output = subprocess.check_output(
            ["git", "diff", "--name-only", f"{base_sha}..{head_sha}"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        if not diff_output:
            return []
        return [f.strip() for f in diff_output.splitlines() if f.strip()]
    except Exception as exc:
        raise RuntimeError(f"Failed to execute git diff {base_sha}..{head_sha}: {exc}") from exc


def fetch_github_run(
    run_id: Union[int, str],
    repo: str = DEFAULT_REPO,
    token: Optional[str] = None,
) -> dict[str, Any]:
    """Fetch workflow run metadata from GitHub Actions API."""
    token = token or os.environ.get("GITHUB_TOKEN") or None
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}"

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "LegalIR-Release-Verifier/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_github_ci_run(
    run_id: Union[int, str],
    expected_runtime_sha: str,
    repo: str = DEFAULT_REPO,
    token: Optional[str] = None,
) -> tuple[bool, str]:
    """Verify that a specific GitHub Actions workflow run is GREEN and belongs to the exact expected SHA."""
    try:
        data = fetch_github_run(run_id, repo=repo, token=token)
    except urllib.error.HTTPError as exc:
        return False, f"GitHub API HTTP {exc.code}: {exc.reason} while fetching run {run_id}"
    except Exception as exc:
        return False, f"Network/API request error for run {run_id}: {exc}"

    name = str(data.get("name", ""))
    head_sha = str(data.get("head_sha", "")).strip().lower()
    conclusion = str(data.get("conclusion", ""))

    if name != DEFAULT_WORKFLOW_NAME:
        return False, f"CI workflow name mismatch for run {run_id}: expected '{DEFAULT_WORKFLOW_NAME}', got '{name}'"

    if head_sha != expected_runtime_sha.strip().lower():
        return (
            False,
            f"CI run {run_id} head_sha mismatch: expected '{expected_runtime_sha}', got '{head_sha}'",
        )

    if conclusion != "success":
        return False, f"CI run {run_id} conclusion is not 'success': got '{conclusion}'"

    return True, f"Verified CI run {run_id} on {head_sha} (conclusion: success)"


def verify_colab_report_invariants(
    report: Mapping[str, Any],
    expected_runtime_sha: str,
) -> list[str]:
    """Verify all Colab T4 PASS invariants from the report content."""
    errors: list[str] = []

    # 1. Git SHA
    report_git_sha = str(report.get("git_sha", "")).strip().lower()
    if report_git_sha != expected_runtime_sha.lower():
        errors.append(
            f"Colab report git_sha mismatch: expected '{expected_runtime_sha}', got '{report_git_sha}'"
        )

    # 2. Result & GPU & CI
    if report.get("result") != "PASS":
        errors.append(f"Colab report result must be 'PASS', got '{report.get('result')}'")

    gpu_name = str(report.get("gpu_name", ""))
    if "T4" not in gpu_name:
        errors.append(f"Colab report gpu_name must contain 'T4', got '{gpu_name}'")

    if not report.get("ci_green", False):
        errors.append("Colab report 'ci_green' must be True.")

    # 3. Dataset Identity
    ds = report.get("dataset_identity", {})
    if not isinstance(ds, Mapping):
        errors.append("Colab report missing 'dataset_identity' section.")
    else:
        if ds.get("documents") != 8532:
            errors.append(f"Dataset identity documents mismatch: expected 8532, got {ds.get('documents')}")
        if ds.get("chunks") != 1153876:
            errors.append(f"Dataset identity chunks mismatch: expected 1153876, got {ds.get('chunks')}")
        if ds.get("train_queries") != 7000:
            errors.append(f"Dataset identity train_queries mismatch: expected 7000, got {ds.get('train_queries')}")
        if ds.get("qrels") != 7637:
            errors.append(f"Dataset identity qrels mismatch: expected 7637, got {ds.get('qrels')}")
        if ds.get("public_queries") not in (1000, 2080):
            errors.append(f"Dataset identity public_queries mismatch: expected 1000 or 2080, got {ds.get('public_queries')}")

    # 4. Dense Pipeline
    if report.get("dense_backend") != "faiss":
        errors.append(f"Colab report dense_backend must be 'faiss', got '{report.get('dense_backend')}'")
    if report.get("dense_device") != "cuda:0":
        errors.append(f"Colab report dense_device must be 'cuda:0', got '{report.get('dense_device')}'")
    if not report.get("dense_embeddings_finite", False):
        errors.append("Colab report 'dense_embeddings_finite' must be True.")

    # 5. Training
    opt_steps = report.get("optimizer_steps", 0)
    if not (isinstance(opt_steps, (int, float)) and opt_steps > 0):
        errors.append(f"Colab report optimizer_steps must be > 0, got {opt_steps}")

    if not report.get("loss_finite", False):
        errors.append("Colab report 'loss_finite' must be True.")

    param_diff = report.get("param_diff", 0)
    if not (isinstance(param_diff, (int, float)) and param_diff > 0):
        errors.append(f"Colab report param_diff must be > 0, got {param_diff}")

    # 6. Adapter Verification
    av = report.get("adapter_verification", {})
    if not isinstance(av, Mapping):
        errors.append("Colab report missing 'adapter_verification' section.")
    else:
        if not av.get("fresh_reload", False):
            errors.append("Adapter verification 'fresh_reload' must be True.")
        if not av.get("active_peft", False):
            errors.append("Adapter verification 'active_peft' must be True.")
        if not av.get("finite_scores", False):
            errors.append("Adapter verification 'finite_scores' must be True.")

    # 7. Prediction Validation
    pv = report.get("prediction_validation", {})
    if not isinstance(pv, Mapping):
        errors.append("Colab report missing 'prediction_validation' section.")
    else:
        if not pv.get("valid", False):
            errors.append("Prediction validation 'valid' must be True.")
        if pv.get("prediction_pipeline") != "dense_faiss_plus_reloaded_bge":
            errors.append(
                f"Prediction validation pipeline must be 'dense_faiss_plus_reloaded_bge', got '{pv.get('prediction_pipeline')}'"
            )
        if pv.get("public_queries_executed") != 16:
            errors.append(
                f"Prediction validation public_queries_executed must be 16, got {pv.get('public_queries_executed')}"
            )

    # 8. Parameter Audit
    pa = report.get("parameter_audit", {})
    if not isinstance(pa, Mapping):
        errors.append("Colab report missing 'parameter_audit' section.")
    else:
        if not pa.get("parameter_budget_compliant", False):
            errors.append("Parameter audit 'parameter_budget_compliant' must be True.")
        sys_params = pa.get("system_learned_parameters", 0)
        if not (isinstance(sys_params, (int, float)) and 0 < sys_params < 4_000_000_000):
            errors.append(f"Parameter audit system_learned_parameters must be < 4,000,000,000, got {sys_params}")

    return errors


def check_notebook_pins(repo_root: Union[Path, str], expected_runtime_sha: str) -> list[str]:
    """Check that all distributed notebooks pin the expected runtime commit SHA."""
    root = Path(repo_root)
    errors: list[str] = []
    target = expected_runtime_sha.strip().lower()

    notebook_files = [
        root / "legalir_training.ipynb",
        root / "kaggle_kernel_task1" / "legalir_training.ipynb",
        root / "kaggle_kernel" / "legalir_training.ipynb",
        root / "kaggle_kernel" / "legalqa_gpu_pipeline.ipynb",
        root / "notebooks" / "kaggle_final.ipynb",
        root / "notebooks" / "kaggle_t4x2_smoke.ipynb",
        root / "notebooks" / "colab_a100_train.ipynb",
        root / "colab" / "legalir_t4_smoke.ipynb",
    ]

    for nb in notebook_files:
        if not nb.is_file():
            continue
        try:
            content = nb.read_text(encoding="utf-8")
            if target not in content.lower():
                errors.append(f"Notebook '{nb.relative_to(root)}' does not contain pinned runtime SHA '{target}'")
        except Exception as exc:
            errors.append(f"Failed reading notebook '{nb}': {exc}")

    return errors


def validate_release_approval(
    approval: Mapping[str, Any],
    repo_root: Union[Path, str] = ".",
    colab_report_path: Optional[Union[Path, str]] = None,
    git_head: Optional[str] = None,
    verify_github_actions: bool = False,
    github_token: Optional[str] = None,
    diff_fn: Optional[Any] = None,
    ancestor_fn: Optional[Any] = None,
    git_root: Optional[Union[Path, str]] = None,
    return_metadata: bool = False,
) -> Union[Tuple[bool, list[str]], Tuple[bool, list[str], dict[str, Any]]]:
    """
    Authoritative release-governance validation gate for LegalIR.
    Validates runtime SHA, CI binding, Colab T4 invariants, Git lineage, and notebook pins.
    Returns (is_valid, errors) by default, or (is_valid, errors, metadata) if return_metadata=True.
    """
    errors: list[str] = []
    root = Path(git_root or repo_root)

    diff_func = diff_fn or get_git_diff_files
    ancestor_func = ancestor_fn or is_git_ancestor

    # 1. Runtime SHA in artifact
    runtime_sha = str(approval.get("runtime_sha", "")).strip().lower()
    if not validate_sha(runtime_sha):
        errors.append(f"Invalid runtime_sha format: '{runtime_sha}'. Must be exact 40-hex Git SHA.")

    # 2. CI section
    ci_info = approval.get("ci", {})
    if not isinstance(ci_info, Mapping):
        errors.append("Missing or invalid 'ci' section in release approval.")
    else:
        ci_runtime = str(ci_info.get("runtime_sha", "")).strip().lower()
        if ci_runtime != runtime_sha:
            errors.append(f"CI runtime SHA mismatch: expected {runtime_sha}, got {ci_runtime}")
        if ci_info.get("conclusion") != "success":
            errors.append(f"CI conclusion must be 'success', got '{ci_info.get('conclusion')}'")

        # GitHub Actions CI run ID check if requested
        run_id = ci_info.get("run_id")
        if verify_github_actions:
            if not run_id:
                errors.append("Missing 'run_id' in approval 'ci' section required for GitHub Actions verification.")
            else:
                ci_ok, ci_msg = check_github_ci_run(
                    run_id=run_id,
                    expected_runtime_sha=runtime_sha,
                    token=github_token,
                )
                if not ci_ok:
                    errors.append(f"GitHub Actions CI run verification failed: {ci_msg}")

    # 3. Colab section in approval & Report verification
    colab_info = approval.get("colab", {})
    colab_report_sha_expected = ""
    if not isinstance(colab_info, Mapping):
        errors.append("Missing or invalid 'colab' section in release approval.")
    else:
        colab_runtime = str(colab_info.get("runtime_sha", "")).strip().lower()
        if colab_runtime != runtime_sha:
            errors.append(f"Colab section runtime SHA mismatch: expected {runtime_sha}, got {colab_runtime}")
        if colab_info.get("result") != "PASS":
            errors.append(f"Colab result in approval must be 'PASS', got '{colab_info.get('result')}'")
        colab_report_sha_expected = str(colab_info.get("report_sha256", "")).strip().lower()
        if not colab_report_sha_expected or len(colab_report_sha_expected) != 64:
            errors.append(f"Invalid colab report_sha256 format: '{colab_report_sha_expected}'")

    # Inspect colab_smoke_report.json directly if report path provided or exists
    rep_path = Path(colab_report_path) if colab_report_path else root / "artifacts" / "task1" / "colab_smoke_report.json"
    actual_report_sha256 = ""
    if rep_path.exists():
        actual_report_sha256 = compute_file_sha256(rep_path).lower()
        if colab_report_sha_expected and actual_report_sha256 != colab_report_sha_expected:
            errors.append(
                f"Colab report SHA-256 mismatch: approval has '{colab_report_sha_expected}', "
                f"computed '{actual_report_sha256}' from {rep_path}"
            )
        try:
            report_data = json.loads(rep_path.read_text(encoding="utf-8"))
            invariant_errors = verify_colab_report_invariants(report_data, expected_runtime_sha=runtime_sha)
            errors.extend(invariant_errors)
        except Exception as exc:
            errors.append(f"Failed to read/parse colab smoke report at {rep_path}: {exc}")
    else:
        errors.append(f"Colab smoke report file not found at: {rep_path}")

    # 4. Production section & Kaggle EXPECTED_COMMIT
    prod_info = approval.get("production", {})
    kaggle_expected = ""
    if not isinstance(prod_info, Mapping):
        errors.append("Missing or invalid 'production' section in release approval.")
    else:
        kaggle_expected = str(prod_info.get("kaggle_expected_commit", "")).strip().lower()
        if kaggle_expected != runtime_sha:
            errors.append(
                f"Kaggle EXPECTED_COMMIT mismatch: must pin approved runtime SHA '{runtime_sha}', got '{kaggle_expected}'"
            )
        if not prod_info.get("dual_gpu_required", False):
            errors.append("Production 'dual_gpu_required' must be True.")

    if not approval.get("approved_for_kaggle_full", False):
        errors.append("'approved_for_kaggle_full' must be True.")

    # 5. Derive actual release HEAD and inspect Git diff against allowlist
    actual_release_head = git_head
    if actual_release_head is None:
        if approval.get("release_sha"):
            actual_release_head = str(approval.get("release_sha", "")).strip().lower()
        else:
            try:
                actual_release_head = derive_git_head(root)
            except Exception as exc:
                errors.append(f"Failed to derive actual release HEAD: {exc}")
                actual_release_head = "UNKNOWN"

    changed_files: list[str] = []
    if validate_sha(runtime_sha) and validate_sha(actual_release_head):
        if runtime_sha == actual_release_head:
            changed_files = []
        else:
            if not ancestor_func(runtime_sha, actual_release_head, root):
                errors.append(
                    f"Git lineage violation: runtime commit '{runtime_sha}' is NOT an ancestor "
                    f"of release HEAD '{actual_release_head}'."
                )
            try:
                changed_files = diff_func(runtime_sha, actual_release_head, root)
                for f in changed_files:
                    if any(f.startswith(p) for p in EXPLICIT_DISALLOWED_PREFIXES) or f in EXPLICIT_DISALLOWED_FILES:
                        errors.append(
                            f"CRITICAL DISALLOWED RUNTIME CHANGE: Disallowed file changed between runtime ({runtime_sha[:8]}) "
                            f"and release ({actual_release_head[:8]}): '{f}'. Existing Colab T4 PASS is INVALIDATED."
                        )
                    elif f not in RELEASE_ONLY_DIFF_ALLOWLIST:
                        errors.append(
                            f"Disallowed file changed between approved runtime ({runtime_sha[:8]}) "
                            f"and release ({actual_release_head[:8]}): '{f}'. "
                            f"Only release governance artifacts may change without a new Colab run."
                        )
            except Exception as exc:
                errors.append(f"Git diff inspection error: {exc}")

    # 6. Check notebook pins match runtime_sha
    if validate_sha(runtime_sha):
        nb_errors = check_notebook_pins(root, expected_runtime_sha=runtime_sha)
        errors.extend(nb_errors)

    metadata = {
        "runtime_sha": runtime_sha,
        "actual_release_head": actual_release_head,
        "kaggle_expected_commit": kaggle_expected,
        "report_sha256": actual_report_sha256,
        "changed_files": changed_files,
    }

    if return_metadata:
        return len(errors) == 0, errors, metadata
    return len(errors) == 0, errors
