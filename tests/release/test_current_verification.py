"""Current release authority regressions: verify_launch + strict CLI, not legacy."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.colab.bootstrap import verify_launch
from src.release.fingerprints import (
    compute_canonical_json_hash,
    fingerprint_structured_config,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(repo), text=True).strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    # Required configs for fingerprint checks.
    for src in ("configs/algorithm/legalir_v2.yaml", "configs/runtime/kaggle_t4x2.yaml"):
        dst = repo / src
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((Path.cwd() / src).read_bytes())
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "runtime")
    return repo


def _make_freeze_report(repo: Path, runtime_sha: str):
    algo_hash = fingerprint_structured_config(repo / "configs/algorithm/legalir_v2.yaml")
    profile_hash = fingerprint_structured_config(repo / "configs/runtime/kaggle_t4x2.yaml")
    dataset_hash = "d" * 64
    report = {
        "stage": "KAGGLE_T4X2",
        "verdict": "PASS",
        "git_sha": runtime_sha,
        "dataset_manifest_sha256": dataset_hash,
        "algorithm_config_sha256": algo_hash,
        "runtime_profile_sha256": profile_hash,
        "devices": ["Tesla T4", "Tesla T4"],
    }
    report_sha = compute_canonical_json_hash(report)
    freeze = {
        "git_sha": runtime_sha,
        "dataset": {"manifest_sha256": dataset_hash},
        "algorithm_config_sha256": algo_hash,
        "gates": {"kaggle_t4x2": {"report_sha256": report_sha, "verdict": "PASS"}},
    }
    # Use allowlisted evidence paths so evidence-only descendants pass lineage.
    k_path = repo / "artifacts/task1/gates/kaggle_t4x2_report.json"
    f_path = repo / "artifacts/task1/freeze/production_freeze.json"
    k_path.parent.mkdir(parents=True, exist_ok=True)
    f_path.parent.mkdir(parents=True, exist_ok=True)
    k_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    f_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    return report, freeze, k_path, f_path


def test_current_accepts_runtime_and_evidence_only_child(tmp_path):
    repo = _init_repo(tmp_path)
    runtime = _git(repo, "rev-parse", "HEAD")
    _, _, k_path, f_path = _make_freeze_report(repo, runtime)
    assert verify_launch(runtime, k_path, f_path, repo_root=repo)["git_sha"] == runtime
    # Evidence-only child: add only allowlisted evidence files.
    _git(repo, "add", "artifacts/task1/gates/kaggle_t4x2_report.json", "artifacts/task1/freeze/production_freeze.json")
    _git(repo, "commit", "-qm", "evidence-only")
    child = _git(repo, "rev-parse", "HEAD")
    assert verify_launch(child, k_path, f_path, repo_root=repo)["git_sha"] == runtime


def test_current_rejects_disallowed_runtime_change(tmp_path):
    repo = _init_repo(tmp_path)
    runtime = _git(repo, "rev-parse", "HEAD")
    _, _, k_path, f_path = _make_freeze_report(repo, runtime)
    # Disallowed: configs/ prefix is explicitly disallowed for runtime→release.
    cfg = repo / "configs/algorithm/legalir_v2.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "\n# runtime change\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "runtime change")
    child = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(RuntimeError, match="another runtime|lineage|mismatch"):
        verify_launch(child, k_path, f_path, repo_root=repo)


def test_mutations_fail_closed(tmp_path):
    import copy

    repo = _init_repo(tmp_path)
    runtime = _git(repo, "rev-parse", "HEAD")
    report, freeze, k_path, f_path = _make_freeze_report(repo, runtime)

    def rewrite(r=None, f=None):
        rr = copy.deepcopy(report) if r is None else r
        ff = copy.deepcopy(freeze) if f is None else f
        k_path.write_text(json.dumps(rr, indent=2), encoding="utf-8")
        f_path.write_text(json.dumps(ff, indent=2), encoding="utf-8")
        return rr, ff

    # Wrong runtime SHA in report.
    r2 = copy.deepcopy(report)
    r2["git_sha"] = "0" * 40
    rewrite(r=r2)
    with pytest.raises(RuntimeError):
        verify_launch(runtime, k_path, f_path, repo_root=repo)
    rewrite()

    # Dataset mismatch.
    r2 = copy.deepcopy(report)
    r2["dataset_manifest_sha256"] = "f" * 64
    rewrite(r=r2)
    with pytest.raises(RuntimeError):
        verify_launch(runtime, k_path, f_path, repo_root=repo)
    rewrite()

    # Non-PASS.
    r2 = copy.deepcopy(report)
    r2["verdict"] = "FAIL"
    rewrite(r=r2)
    with pytest.raises(RuntimeError):
        verify_launch(runtime, k_path, f_path, repo_root=repo)
    rewrite()

    # Mock hardware.
    r2 = copy.deepcopy(report)
    r2["devices"] = ["Mock T4", "Mock T4"]
    rewrite(r=r2)
    with pytest.raises(RuntimeError):
        verify_launch(runtime, k_path, f_path, repo_root=repo)
    rewrite()

    # Wrong canonical digest in freeze.
    f2 = copy.deepcopy(freeze)
    f2["gates"]["kaggle_t4x2"]["report_sha256"] = "0" * 64
    rewrite(f=f2)
    with pytest.raises(RuntimeError, match="digest"):
        verify_launch(runtime, k_path, f_path, repo_root=repo)
    rewrite()

    # Wrong profile digest (recompute freeze digest so only profile check fails).
    r2 = copy.deepcopy(report)
    r2["runtime_profile_sha256"] = "0" * 64
    f2 = copy.deepcopy(freeze)
    f2["gates"]["kaggle_t4x2"]["report_sha256"] = compute_canonical_json_hash(r2)
    rewrite(r=r2, f=f2)
    with pytest.raises(RuntimeError, match="profile"):
        verify_launch(runtime, k_path, f_path, repo_root=repo)
    rewrite()

    # Absent freeze / report (no fallback).
    with pytest.raises(RuntimeError, match="missing"):
        verify_launch(runtime, repo / "nope.json", f_path, repo_root=repo)
    with pytest.raises(RuntimeError, match="missing"):
        verify_launch(runtime, k_path, repo / "nope.json", repo_root=repo)

    # Absent expected fields.
    f2 = copy.deepcopy(freeze)
    del f2["gates"]["kaggle_t4x2"]["report_sha256"]
    rewrite(f=f2)
    with pytest.raises(RuntimeError, match="missing"):
        verify_launch(runtime, k_path, f_path, repo_root=repo)
    rewrite()
    # Missing profile hash: recompute freeze digest so the test isolates the
    # absent-field check rather than tripping the digest mismatch first.
    r2 = copy.deepcopy(report)
    del r2["runtime_profile_sha256"]
    f2 = copy.deepcopy(freeze)
    f2["gates"]["kaggle_t4x2"]["report_sha256"] = compute_canonical_json_hash(r2)
    rewrite(r=r2, f=f2)
    with pytest.raises(RuntimeError, match="missing"):
        verify_launch(runtime, k_path, f_path, repo_root=repo)


def test_two_stage_flow_stale_then_evidence(tmp_path):
    """New runtime CI may pass while strict rejects stale freeze; new evidence accepts."""
    repo = _init_repo(tmp_path)
    runtime = _git(repo, "rev-parse", "HEAD")
    _, _, k_path, f_path = _make_freeze_report(repo, runtime)
    # New runtime code change (disallowed diff) with old freeze → reject.
    (repo / "src").mkdir(exist_ok=True)
    (repo / "src" / "x.py").write_text("x=1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "new runtime")
    new_runtime = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(RuntimeError):
        verify_launch(new_runtime, k_path, f_path, repo_root=repo)
    # After genuine evidence is bundled for the new runtime, strict accepts.
    _make_freeze_report(repo, new_runtime)
    assert verify_launch(new_runtime, k_path, f_path, repo_root=repo)["git_sha"] == new_runtime


def test_cli_default_never_reads_legacy_approval(tmp_path, monkeypatch):
    import scripts.verify_release_approval as vra

    repo = _init_repo(tmp_path)
    runtime = _git(repo, "rev-parse", "HEAD")
    report, freeze, k_path, f_path = _make_freeze_report(repo, runtime)
    # Place a bogus legacy file that would fail if read; default must ignore it.
    bogus = repo / "artifacts/task1/release_approval.json"
    bogus.parent.mkdir(parents=True, exist_ok=True)
    bogus.write_text('{"bogus": true}', encoding="utf-8")
    # Point default paths at our fixture via explicit args (still current mode).
    rc = vra.main(["--repo-root", str(repo), "--kaggle-report", str(k_path), "--freeze-file", str(f_path)])
    assert rc == 0


def test_cli_production_rejects_bypass():
    import scripts.verify_release_approval as vra

    assert vra.main(["--repo-root", ".", "--allow-runtime-changes"]) == 2
