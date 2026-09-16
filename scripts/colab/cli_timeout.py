#!/usr/bin/env python3
"""Bounded subprocess deadline helper for Colab orchestration (POSIX/macOS/Linux).

Usage: python scripts/colab/cli_timeout.py SECONDS COMMAND [ARGS...]

Returns the command's exit status; 124 on timeout. CLI parsing rejects
nonpositive/nonfinite timeouts and empty commands before starting a process.
Standard library only; macOS has no GNU `timeout` and the installed Colab CLI
exposes no timeout flag, so deadlines are enforced externally.

This helper bounds local cleanup (downloads, stop) so a hang cannot
indefinitely postpone VM release. It does not guarantee remote termination:
a confirmed `colab stop` is still required, and SIGKILL, laptop shutdown,
network loss, or Colab eviction can defeat local cleanup.
"""
from __future__ import annotations

import math
import os
import signal
import subprocess
import sys


def run_bounded(seconds: float, argv: list[str]) -> int:
    """Run argv with a deadline; return exit status or 124 on timeout."""
    if not isinstance(seconds, (int, float)) or not math.isfinite(float(seconds)):
        raise ValueError("timeout must be a finite number of seconds")
    seconds = float(seconds)
    if seconds <= 0:
        raise ValueError("timeout must be positive")
    if not argv:
        raise ValueError("no command to run")
    proc = subprocess.Popen(argv, start_new_session=True)
    try:
        result = proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        return 124
    # Normalize signal deaths to shell convention 128+signo.
    return result if result >= 0 else 128 - result


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        print("usage: cli_timeout.py SECONDS COMMAND [ARGS...]", file=sys.stderr)
        return 2
    try:
        seconds = float(args[0])
    except ValueError:
        print(f"[!] Invalid timeout value: {args[0]!r}", file=sys.stderr)
        return 2
    if not math.isfinite(seconds) or seconds <= 0:
        print(f"[!] Timeout must be a positive finite number of seconds, got {args[0]!r}", file=sys.stderr)
        return 2
    cmd = args[1:]
    if not cmd or not cmd[0]:
        print("[!] No command to run.", file=sys.stderr)
        return 2
    try:
        return run_bounded(seconds, cmd)
    except ValueError as exc:
        print(f"[!] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
