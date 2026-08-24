"""Minimal no-install HumanEval execution helper.

This follows the OpenAI HumanEval contract: append the generated completion to
the function prompt, then execute the task tests and ``check(entry_point)``.
Generated code is untrusted. Callers must require an explicit opt-in and should
run this only on an isolated evaluation host.
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import tempfile
from typing import Any


def _limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def check_correctness(
    problem: dict[str, Any], completion: str, timeout_s: float = 3.0
) -> dict[str, Any]:
    program = "\n".join(
        [
            str(problem["prompt"]),
            completion,
            str(problem["test"]),
            f"check({problem['entry_point']})",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="vpipe-humaneval-") as workdir:
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", program],
            cwd=workdir,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=_limit_child,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return {
                "passed": False,
                "result": "timed_out",
                "stdout": stdout[-2000:],
                "stderr": stderr[-2000:],
            }
    return {
        "passed": process.returncode == 0,
        "result": "passed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "stdout": stdout[-2000:],
        "stderr": stderr[-2000:],
    }
