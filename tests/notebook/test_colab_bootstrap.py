import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.generate_notebooks import build_colab_train_notebook


def _checkout_cell_source(commit_sha="a" * 40, checkout=None):
    """Return the actual generated Cell 2 source, optionally repointed at checkout."""
    import subprocess
    source = "".join(build_colab_train_notebook(commit_sha)["cells"][2]["source"])
    if checkout is not None:
        # Substitute just the Colab filesystem expression; execute the actual generated cell.
        source = source.replace('Path("/content/LegalIR") if Path("/content").exists() else Path.cwd()', repr(str(checkout)))
        source = source.replace('REPO_DIR = ' + repr(str(checkout)), 'REPO_DIR = Path(' + repr(str(checkout)) + ')')
    return source


def test_a100_checkout_changes_working_directory(tmp_path, monkeypatch):
    """Execute the checkout cell with git stubbed; imports alone don't set cwd."""
    import subprocess
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    (checkout / "scripts" / "colab").mkdir(parents=True)
    (checkout / "scripts" / "colab" / "bootstrap.py").touch()
    monkeypatch.chdir(tmp_path)
    # Explicit release selection, as required by the generated cell.
    monkeypatch.setenv("LEGALIR_COMMIT_SHA", "a" * 40)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"))
    source = _checkout_cell_source("a" * 40, checkout)
    import sys
    monkeypatch.setattr(sys, "path", list(sys.path))
    exec(compile(source, "checkout-cell", "exec"), {"os": os, "sys": sys, "json": __import__("json")})
    assert Path.cwd() == checkout


def test_a100_checkout_cell_rejects_missing_release_selection(tmp_path, monkeypatch):
    """Without launch JSON or LEGALIR_COMMIT_SHA the cell must fail fast with
    an actionable error instead of silently checking out the stale runtime pin."""
    import subprocess
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEGALIR_COMMIT_SHA", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"))
    source = _checkout_cell_source("b" * 40, checkout)
    import sys
    with pytest.raises(RuntimeError, match="explicit release selection"):
        exec(compile(source, "checkout-cell", "exec"), {"os": os, "sys": sys, "json": __import__("json")})


def test_a100_checkout_cell_documents_release_selection():
    source = "".join(build_colab_train_notebook("a" * 40)["cells"][2]["source"])
    assert "explicit release selection" in source
    assert "scripts/colab/run_colab_cli.sh" in source


def test_a100_secret_keys_and_dependency_contract():
    cells = build_colab_train_notebook("a" * 40)["cells"]
    secrets = "".join(cells[1]["source"])
    assert '"KAGGLE_USERNAME"' in secrets
    assert '"LEGALIR_COMMIT_SHA"' in secrets
    assert 'os.environ["KAGGLE_KEY"] = kg_tok' not in secrets
    setup = "".join(cells[3]["source"])
    assert "requirements-colab.txt" in setup
    assert "torch==" in setup
    assert "prepare_dataset" in setup


def test_kaggle_modern_and_legacy_credentials_stay_distinct(monkeypatch):
    from scripts.colab.bootstrap import configure_kaggle_credentials
    for key in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_test")
    configure_kaggle_credentials()
    assert "KAGGLE_KEY" not in os.environ
    monkeypatch.delenv("KAGGLE_API_TOKEN")
    monkeypatch.setenv("KAGGLE_KEY", "legacy-test")
    with pytest.raises(RuntimeError, match="KAGGLE_USERNAME"):
        configure_kaggle_credentials()
    monkeypatch.setenv("KAGGLE_USERNAME", "test-owner")
    configure_kaggle_credentials()
    assert "KAGGLE_API_TOKEN" not in os.environ


def test_dataset_acquisition_verifies_download_before_return(tmp_path, monkeypatch):
    from scripts.colab import bootstrap
    calls = []
    def download(args, **kwargs):
        calls.append(args)
        for name in bootstrap.REQUIRED_FILES:
            (tmp_path / name).write_text("fixture")
    monkeypatch.setattr(bootstrap.subprocess, "run", download)
    monkeypatch.setattr(bootstrap, "verify_dataset_fingerprint", lambda *a, **k: calls.append("verified"))
    bootstrap.prepare_dataset(tmp_path)
    assert calls[-1] == "verified"
    assert "phucdangg/legalir-task1-clean-data" in calls[0]
    calls.clear()
    bootstrap.prepare_dataset(tmp_path)
    assert calls == ["verified"]
