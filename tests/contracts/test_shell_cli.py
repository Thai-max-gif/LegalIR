"""
Tests for Colab CLI Automation shell script invariants.
Validates fail-closed routing, error trapping, and automatic compute cleanup.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SHELL_SCRIPT_PATH = REPO_ROOT / "scripts" / "colab" / "run_colab_cli.sh"


def test_colab_cli_script_exists():
    assert SHELL_SCRIPT_PATH.is_file(), f"Missing {SHELL_SCRIPT_PATH}"


def test_colab_cli_strict_bash_mode():
    content = SHELL_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in content or "set -euo pipefail" in content, (
        "Script must use strict error handling (set -Eeuo pipefail)"
    )


def test_colab_cli_cleanup_trap_present():
    content = SHELL_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "trap cleanup EXIT INT TERM" in content, (
        "Script must trap EXIT INT TERM to guarantee VM stop"
    )
    assert "colab stop" in content, (
        "Cleanup routine must invoke colab stop"
    )


def test_colab_cli_explicit_gpu_routing():
    content = SHELL_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "colab_t4_smoke.ipynb" in content, (
        "T4 mode must route to notebooks/colab_t4_smoke.ipynb"
    )
    assert "colab_a100_train.ipynb" in content, (
        "A100 mode must route to notebooks/colab_a100_train.ipynb"
    )
