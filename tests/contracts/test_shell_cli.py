import pytest
import subprocess
from pathlib import Path

def test_shell_wrapper_preserves_error_code(monkeypatch, tmp_path):
    import os
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + ":" + env.get("PATH", "")
    (tmp_path / "colab").write_text("#!/bin/sh\nif [ \"$1\" = \"exec\" ]; then exit 42; fi\nexit 0\n")
    (tmp_path / "colab").chmod(0o755)
    (tmp_path / "py_stub").write_text("#!/bin/sh\nexit 0\n")
    (tmp_path / "py_stub").chmod(0o755)
    env["PYTHON_BIN"] = str(tmp_path / "py_stub")

    script_path = Path("scripts/colab/run_colab_cli.sh").resolve()
    res = subprocess.run([str(script_path), "A100"], env=env)
    assert res.returncode == 42

def test_shell_wrapper_always_stops_colab(monkeypatch, tmp_path):
    import os
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + ":" + env.get("PATH", "")
    log = tmp_path / "colab_log"
    (tmp_path / "colab").write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nif [ \"$1\" = \"exec\" ]; then exit 1; fi\nexit 0\n")
    (tmp_path / "colab").chmod(0o755)
    (tmp_path / "py_stub").write_text("#!/bin/sh\nexit 0\n")
    (tmp_path / "py_stub").chmod(0o755)
    env["PYTHON_BIN"] = str(tmp_path / "py_stub")

    script_path = Path("scripts/colab/run_colab_cli.sh").resolve()
    subprocess.run([str(script_path), "A100"], env=env)
    assert "stop -s" in log.read_text()

def test_shell_wrapper_does_not_upload_unrelated_env_vars(monkeypatch, tmp_path):
    import os
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + ":" + env.get("PATH", "")
    log = tmp_path / "colab_log"
    (tmp_path / "colab").write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexit 0\n")
    (tmp_path / "colab").chmod(0o755)
    (tmp_path / "py_stub").write_text("#!/bin/sh\nexit 0\n")
    (tmp_path / "py_stub").chmod(0o755)
    env["PYTHON_BIN"] = str(tmp_path / "py_stub")

    env_file = Path(".env")
    env_file.write_text("HF_TOKEN=test\nSECRET_KEY=bad\n")

    script_path = Path("scripts/colab/run_colab_cli.sh").resolve()
    subprocess.run([str(script_path), "A100"], env=env)
    env_file.unlink()

    filtered_content = Path(".env.filtered").read_text() if Path(".env.filtered").exists() else ""
    assert "SECRET_KEY" not in filtered_content
    # The filtered file should be cleaned up
    assert not Path(".env.filtered").exists()


def test_shell_wrapper_rejects_retired_t4_mode(monkeypatch, tmp_path):
    import os
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + ":" + env.get("PATH", "")
    (tmp_path / "colab").write_text("#!/bin/sh\nexit 0\n")
    (tmp_path / "colab").chmod(0o755)

    script_path = Path("scripts/colab/run_colab_cli.sh").resolve()
    res = subprocess.run([str(script_path), "T4"], env=env)
    assert res.returncode == 2
