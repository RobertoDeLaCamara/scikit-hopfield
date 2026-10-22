#!/usr/bin/env python3
"""
BADSFA PreToolUse Lifecycle Hook (Hardened Anti-Reward Hacking Guard).
Blocks @Developer agents from modifying behavioral specs (`specs/`) or test suites (`tests/`),
including path traversal (`src/../tests/`) and interpreter shell writes.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import sys

FILE_MUTATION_TOOLS = {
    "edit_file",
    "write_file",
    "replace_file_content",
    "multi_replace_file_content",
    "notebook_edit",
}
PATH_KEYS = ("Path", "file_path", "path", "TargetFile", "NotebookPath")
SHELL_MUTATION_RE = re.compile(
    r"(>+|sed\s+-i|rm\s+|mv\s+|cp\s+|tee\s+|dd\s+|truncate\s+|python\d*\s+-c|"
    r"git\s+(checkout|restore|apply|reset)|Set-Content|Out-File).*(specs|tests)[/\\]",
    re.IGNORECASE,
)


def is_protected_path(target_path: str, workspace_roots: list[str]) -> bool:
    if not target_path:
        return False
    norm = posixpath.normpath(target_path.replace("\\", "/"))
    for root in workspace_roots:
        root_norm = posixpath.normpath(root.replace("\\", "/")).rstrip("/")
        if norm == f"{root_norm}/specs" or norm.startswith(f"{root_norm}/specs/"):
            return True
        if norm == f"{root_norm}/tests" or norm.startswith(f"{root_norm}/tests/"):
            return True
        if norm.startswith(f"{root_norm}/"):
            rel = norm[len(root_norm) + 1 :]
            if rel in ("specs", "tests") or rel.startswith(("specs/", "tests/")):
                return True

    rel_norm = norm.lstrip("/")
    while rel_norm.startswith("../"):
        rel_norm = rel_norm[3:]
    return rel_norm in ("specs", "tests") or rel_norm.startswith(("specs/", "tests/"))


def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"decision": "allow"}))
        return

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(json.dumps({"decision": "allow"}))
        return

    role = os.environ.get("BADSFA_ROLE", "DEVELOPER").upper()
    if role in ("QA", "ARCHITECT", "ADMIN"):
        print(json.dumps({"decision": "allow"}))
        return

    tool_call = payload.get("toolCall", {})
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {})
    workspace_roots = payload.get("workspacePaths", [os.getcwd()])

    if tool_name in FILE_MUTATION_TOOLS:
        for key in PATH_KEYS:
            path = str(args.get(key, ""))
            if path and is_protected_path(path, workspace_roots):
                reason = (
                    f"[BADSFA Anti-Reward Hacking] Role '{role}' is not authorized to modify "
                    f"protected specification or test file '{path}'. Only BADSFA_ROLE=QA or "
                    "BADSFA_ROLE=ARCHITECT may mutate specs/** or tests/**."
                )
                print(json.dumps({"decision": "deny", "reason": reason}))
                return

    if tool_name == "run_command":
        cmd = str(args.get("CommandLine") or args.get("command") or "")
        if SHELL_MUTATION_RE.search(cmd):
            reason = (
                f"[BADSFA Anti-Reward Hacking] Shell command attempts to mutate "
                f"protected 'specs/' or 'tests/' directory under role '{role}'."
            )
            print(json.dumps({"decision": "deny", "reason": reason}))
            return

    print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()
