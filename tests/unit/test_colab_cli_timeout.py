"""Unit tests for scripts/colab/cli_timeout.py deadline helper."""
import os
import signal
import subprocess
import sys
import time

import pytest

from scripts.colab.cli_timeout import run_bounded, main

PY = sys.executable


def test_normal_exit_preserved():
    assert run_bounded(5, [PY, "-c", "import sys; sys.exit(42)"]) == 42
    assert run_bounded(5, [PY, "-c", "pass"]) == 0


def test_timeout_returns_124_and_kills_hang():
    start = time.monotonic()
    rc = run_bounded(0.3, [PY, "-c", "import time; time.sleep(30)"])
    elapsed = time.monotonic() - start
    assert rc == 124
    # TERM then KILL within ~1s grace plus scheduling tolerance.
    assert elapsed < 5


def test_term_ignored_child_gets_killed_and_reaped():
    # Child ignores SIGTERM; helper must escalate to SIGKILL and reap.
    proc_code = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    start = time.monotonic()
    rc = run_bounded(0.3, [PY, "-c", proc_code])
    elapsed = time.monotonic() - start
    assert rc == 124
    assert elapsed < 6


def test_rejects_bad_timeout_and_empty_command():
    with pytest.raises(ValueError):
        run_bounded(0, [PY, "-c", "pass"])
    with pytest.raises(ValueError):
        run_bounded(-1, [PY, "-c", "pass"])
    with pytest.raises(ValueError):
        run_bounded(float("inf"), [PY, "-c", "pass"])
    with pytest.raises(ValueError):
        run_bounded(float("nan"), [PY, "-c", "pass"])
    with pytest.raises(ValueError):
        run_bounded(1, [])


def test_cli_rejects_before_spawning():
    assert main(["0", "echo", "hi"]) == 2
    assert main(["nan", "echo", "hi"]) == 2
    assert main(["inf", "echo", "hi"]) == 2
    assert main(["5"]) == 2
    assert main([]) == 2


def test_cli_timeout_exit_124():
    rc = main(["0.3", PY, "-c", "import time; time.sleep(30)"])
    assert rc == 124


def test_signal_normalization():
    # A child killed by SIGTERM should surface as 143 (128+15) when it dies
    # before our deadline logic engages (e.g., quick self-kill).
    code = "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"
    rc = run_bounded(5, [PY, "-c", code])
    assert rc == 143
