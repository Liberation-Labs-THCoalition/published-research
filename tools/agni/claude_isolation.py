#!/usr/bin/env python3
"""claude_isolation.py: run the Claude Code CLI as a reviewer, not as the account's owner.

WHY THIS EXISTS (2026-11-02). `claude -p` launched from Lyra's account IS Lyra unless it is
isolated. It loads the user-level CLAUDE.md (identity, family list, the Key Numbers table),
runs the SessionStart sync, and the Stop hook files the exchange into Lyra's episodic memory.
Every Agni review from 2026-08-14 to 2026-11-02 ran that way. Two things close it:

  1. LYRA_NO_CAPTURE=1 in the child's environment. Lyra's capture and sync hooks exit at once
     when it is set.
  2. --settings '{"claudeMdExcludes": ["<home>/.claude/CLAUDE.md", "<home>/.claude/**"]}'.
     `--setting-sources project` does NOT stop CLAUDE.md from loading. `--bare` would, but it
     cannot use the OAuth login Agni depends on.

Every call site that runs the CLI builds its command with isolated_claude_cmd(). Nothing else
about the call changes. test_isolation.py checks the built command offline and, with --live,
probes a real call with a question that can fail.
"""
import json
import os
from pathlib import Path

NO_CAPTURE_ENV = "LYRA_NO_CAPTURE"


def claude_md_excludes(home=None):
    """The two exclusion globs for this account's user-level Claude config, forward slashes."""
    base = (Path(home) if home is not None else Path.home()).as_posix().rstrip("/")
    return [base + "/.claude/CLAUDE.md", base + "/.claude/**"]


def _merge_settings(raw, excludes):
    try:
        settings = json.loads(raw)
    except ValueError:
        raise ValueError("--settings is not inline JSON (a settings file?); pass JSON so the "
                         "CLAUDE.md exclusions can be merged into it")
    if not isinstance(settings, dict):
        raise ValueError("--settings JSON must be an object")
    have = list(settings.get("claudeMdExcludes") or [])
    settings["claudeMdExcludes"] = have + [e for e in excludes if e not in have]
    return json.dumps(settings)


def isolated_claude_cmd(base_cmd, environ=None, home=None):
    """Return (cmd, env) for running base_cmd = [claude_binary, *args] isolated.

    cmd: a new list. `--settings <json>` with claudeMdExcludes is inserted right after the
         binary. If base_cmd already carries --settings (as `--settings X` or `--settings=X`),
         the exclusions are merged into that JSON instead, rather than relying on how the CLI
         resolves a repeated flag.
    env: a copy of `environ` (default os.environ) plus LYRA_NO_CAPTURE=1. Pass it as env=.

    base_cmd and environ are not modified.
    """
    excludes = claude_md_excludes(home)
    cmd = list(base_cmd)
    for i, arg in enumerate(cmd):
        if arg == "--settings" and i + 1 < len(cmd):
            cmd[i + 1] = _merge_settings(cmd[i + 1], excludes)
            break
        if arg.startswith("--settings="):
            cmd[i] = "--settings=" + _merge_settings(arg[len("--settings="):], excludes)
            break
    else:
        cmd[1:1] = ["--settings", json.dumps({"claudeMdExcludes": excludes})]
    env = dict(os.environ if environ is None else environ)
    env[NO_CAPTURE_ENV] = "1"
    return cmd, env
