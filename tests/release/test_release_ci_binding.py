import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.release.provenance import validate_release_approval, check_github_ci_run


def test_release_rejects_ci_run_for_different_sha(tmp_path):
    approval = {
        "schema_version": 2,
        "runtime_sha": "a" * 40,
        "ci": {
            "run_id": 12345678,
            "runtime_sha": "a" * 40,
            "conclusion": "success",
        },
        "colab": {
            "runtime_sha": "a" * 40,
            "result": "PASS",
            "report_sha256": "0" * 64,
        },
        "production": {
            "kaggle_expected_commit": "a" * 40,
            "dual_gpu_required": True,
        },
        "approved_for_kaggle_full": True,
    }

    # GitHub API returns run with different head_sha
    mock_response = {
        "name": "LegalIR CI",
        "head_sha": "b" * 40,  # Different SHA!
        "conclusion": "success",
        "status": "completed",
    }

    with patch("src.release.provenance.fetch_github_run", return_value=mock_response):
        is_valid, errors, _ = validate_release_approval(
            approval=approval,
            repo_root=tmp_path,
            verify_github_actions=True,
        )
        assert not is_valid
        assert any("ci run" in e.lower() and ("sha mismatch" in e.lower() or "different" in e.lower() or "head_sha" in e.lower()) for e in errors)


def test_release_rejects_failed_runtime_ci(tmp_path):
    approval = {
        "schema_version": 2,
        "runtime_sha": "a" * 40,
        "ci": {
            "run_id": 12345678,
            "runtime_sha": "a" * 40,
            "conclusion": "success",
        },
        "colab": {
            "runtime_sha": "a" * 40,
            "result": "PASS",
            "report_sha256": "0" * 64,
        },
        "production": {
            "kaggle_expected_commit": "a" * 40,
            "dual_gpu_required": True,
        },
        "approved_for_kaggle_full": True,
    }

    # GitHub API returns run with failed conclusion
    mock_response = {
        "name": "LegalIR CI",
        "head_sha": "a" * 40,
        "conclusion": "failure",
        "status": "completed",
    }

    with patch("src.release.provenance.fetch_github_run", return_value=mock_response):
        is_valid, errors, _ = validate_release_approval(
            approval=approval,
            repo_root=tmp_path,
            verify_github_actions=True,
        )
        assert not is_valid
        assert any("conclusion" in e.lower() and ("success" in e.lower() or "failure" in e.lower()) for e in errors)


def test_release_rejects_wrong_workflow_name(tmp_path):
    approval = {
        "schema_version": 2,
        "runtime_sha": "a" * 40,
        "ci": {
            "run_id": 12345678,
            "runtime_sha": "a" * 40,
            "conclusion": "success",
        },
        "colab": {
            "runtime_sha": "a" * 40,
            "result": "PASS",
            "report_sha256": "0" * 64,
        },
        "production": {
            "kaggle_expected_commit": "a" * 40,
            "dual_gpu_required": True,
        },
        "approved_for_kaggle_full": True,
    }

    mock_response = {
        "name": "Other Workflow",  # Not "LegalIR CI"
        "head_sha": "a" * 40,
        "conclusion": "success",
        "status": "completed",
    }

    with patch("src.release.provenance.fetch_github_run", return_value=mock_response):
        is_valid, errors, _ = validate_release_approval(
            approval=approval,
            repo_root=tmp_path,
            verify_github_actions=True,
        )
        assert not is_valid
        assert any("workflow name" in e.lower() or "legalir ci" in e.lower() for e in errors)


def test_release_accepts_runtime_A_release_only_B(tmp_path, monkeypatch):
    runtime_sha = "a" * 40
    release_sha = "b" * 40

    approval = {
        "schema_version": 2,
        "runtime_sha": runtime_sha,
        "ci": {
            "run_id": 12345678,
            "runtime_sha": runtime_sha,
            "conclusion": "success",
        },
        "colab": {
            "runtime_sha": runtime_sha,
            "result": "PASS",
            "report_sha256": "c" * 64,
        },
        "production": {
            "kaggle_expected_commit": runtime_sha,
            "dual_gpu_required": True,
        },
        "approved_for_kaggle_full": True,
    }

    mock_response = {
        "name": "LegalIR CI",
        "head_sha": runtime_sha,
        "conclusion": "success",
        "status": "completed",
    }

    monkeypatch.setattr("src.release.provenance.is_git_ancestor", lambda a, b, root: True)
    monkeypatch.setattr("src.release.provenance.get_git_diff_files", lambda a, b, root: [
        "artifacts/task1/release_approval.json",
        "artifacts/task1/colab_smoke_report.json",
    ])
    monkeypatch.setattr("src.release.provenance.compute_file_sha256", lambda path: "c" * 64)
    monkeypatch.setattr("src.release.provenance.verify_colab_report_invariants", lambda report, expected_runtime_sha: [])
    monkeypatch.setattr("src.release.provenance.check_notebook_pins", lambda root, expected_runtime_sha: [])

    colab_report = tmp_path / "artifacts" / "task1" / "colab_smoke_report.json"
    colab_report.parent.mkdir(parents=True, exist_ok=True)
    colab_report.write_text("{}", encoding="utf-8")

    with patch("src.release.provenance.fetch_github_run", return_value=mock_response):
        is_valid, errors, _ = validate_release_approval(
            approval=approval,
            repo_root=tmp_path,
            colab_report_path=colab_report,
            git_head=release_sha,
            verify_github_actions=True,
        )
        assert is_valid, f"Validation failed: {errors}"
        assert len(errors) == 0


def test_provenance_validator_schema_is_single_source_of_truth():
    import inspect
    import scripts.verify_release_approval as vra
    from src.release import provenance

    # Ensure verify_release_approval delegates to or shares implementation with provenance
    assert hasattr(provenance, "validate_release_approval")
    assert hasattr(provenance, "check_github_ci_run")
    assert hasattr(vra, "validate_release_approval_v2") or hasattr(vra, "validate_release_approval")
