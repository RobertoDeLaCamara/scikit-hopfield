#!/usr/bin/env python3
"""
BADSFA Stop Lifecycle Hook (Deterministic Checkable Done Gate).
Intercepts agent loop termination (`NO_TOOL_CALL`) and verifies that the contract test suite passes.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    raw = sys.stdin.read().strip()
    payload = {}
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            pass

    termination_reason = payload.get("terminationReason", "NO_TOOL_CALL")
    if termination_reason not in ("NO_TOOL_CALL", ""):
        print(json.dumps({"decision": "allow"}))
        return

    anchored_root = Path(__file__).resolve().parents[2]
    repo_root = anchored_root
    workspace_paths = payload.get("workspacePaths", [])
    if workspace_paths:
        candidate = Path(str(workspace_paths[0])).resolve()
        if (candidate / "tests").is_dir():
            repo_root = candidate

    tests_dir = repo_root / "tests"
    if not tests_dir.exists():
        print(json.dumps({"decision": "allow"}))
        return

    venv_py = repo_root / ".venv" / "bin" / "python"
    python_exe = str(venv_py) if venv_py.exists() else sys.executable
    has_pytest = venv_py.exists() or (importlib.util.find_spec("pytest") is not None)
    cmd = (
        [python_exe, "-m", "pytest", "-q", "--tb=short"]
        if has_pytest
        else [python_exe, "-m", "unittest", "discover", "-s", "tests", "-t", "."]
    )

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            json.dumps(
                {
                    "decision": "continue",
                    "reason": "[BADSFA Checkable Done Gate Failed] Test suite timed out after 25s.",
                }
            )
        )
        return

    if result.returncode != 0:
        excerpt = (result.stdout + "\n" + result.stderr).strip()[-1500:]
        reason = (
            "[BADSFA Checkable Done Gate Failed] Automated test suite returned non-zero "
            "exit status. Remediate the implementation in `src/` before completing the task:\n"
            f"```text\n{excerpt}\n```"
        )
        print(json.dumps({"decision": "continue", "reason": reason}))
        return

    print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()
