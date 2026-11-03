#!/usr/bin/env python3
"""claude_isolation.py: run the Claude Code CLI as a reviewer, not as the account's owner.

WHY THIS EXISTS (2026-11-02). `claude -p` launched from Lyra's account IS Lyra unless it is
isolated. It loads the user-level CLAUDE.md (identity, family list, the Key Numbers table),
runs the SessionStart sync, and the Stop hook files the exchange into Lyra's episodic memory.
Every Agni review from 2026-08-14 to 2026-11-02 ran that way. Five things close it:

  1. LYRA_NO_CAPTURE=1 in the child's environment. Lyra's capture and sync hooks exit at once
     when it is set.
  2. --settings '{"claudeMdExcludes": ["<home>/.claude/CLAUDE.md", "<home>/.claude/**"]}'.
     `--setting-sources project` does NOT stop CLAUDE.md from loading. `--bare` would, but it
     cannot use the OAuth login Agni depends on.
  3. --strict-mcp-config, plus an empty --mcp-config when the caller names none (added
     2026-11-03). LYRA_NO_CAPTURE turns off hooks, not MCP servers: the user-scope servers in
     ~/.claude.json still loaded, lyra-memory among them, so an "isolated" reviewer could read
     Lyra's memory (and stop being independent of the author) or write to it. If the caller
     passes its own --mcp-config, its servers are kept and only --strict-mcp-config is added,
     so nothing beyond what the caller named loads. The empty config is passed inline as ONE
     token, --mcp-config={"mcpServers":{}}: the flag takes several values, and in the
     two-token form a positional prompt after it is read as one more config.
  4. No auto-memory (added 2026-11-03): "autoMemoryEnabled": false in the same --settings JSON
     as claudeMdExcludes, AND CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 in the child's environment.
     claudeMdExcludes stops CLAUDE.md files, not the auto-memory index: a session run with
     cwd = the account's home folder still loaded ~/.claude/projects/<project>/memory/MEMORY.md,
     and asked about a person named only in that file it answered with their pronouns. Each
     switch alone turned that off (one run each answered NONE). Both are set, so a CLI release
     that stops honouring one still leaves the other. A caller's "autoMemoryEnabled": true is
     overridden; the reviewer never gets the owner's memory.
  5. --disable-slash-commands (added 2026-11-03), unless the caller already passes it. With
     items 1 to 4 the session still loaded the account's user-level skills, slash commands and
     a personal plugin's skills (its stream-json init event listed 43 skills and 85 slash
     commands). A reviewer that invoked one would be following the owner's instructions.
     `claude --help` describes the flag as "Disable all skills".

Every call site that runs the CLI builds its command with isolated_claude_cmd(). Nothing else
about the call changes. test_isolation.py checks the built command offline and, with --live,
probes real calls with questions that can fail, each beside a control that shows it can.
"""
import json
import os
from pathlib import Path

NO_CAPTURE_ENV = "LYRA_NO_CAPTURE"
NO_AUTO_MEMORY_ENV = "CLAUDE_CODE_DISABLE_AUTO_MEMORY"
AUTO_MEMORY_KEY = "autoMemoryEnabled"
STRICT_MCP_FLAG = "--strict-mcp-config"
NO_SKILLS_FLAG = "--disable-slash-commands"
EMPTY_MCP_CONFIG = json.dumps({"mcpServers": {}})


def claude_md_excludes(home=None):
    """The two exclusion globs for this account's user-level Claude config, forward slashes."""
    base = (Path(home) if home is not None else Path.home()).as_posix().rstrip("/")
    return [base + "/.claude/CLAUDE.md", base + "/.claude/**"]


def _merge_settings(raw, excludes):
    try:
        settings = json.loads(raw)
    except ValueError:
        raise ValueError("--settings is not inline JSON (a settings file?); pass JSON so the "
                         "CLAUDE.md exclusions and the auto-memory switch can be merged into it")
    if not isinstance(settings, dict):
        raise ValueError("--settings JSON must be an object")
    have = list(settings.get("claudeMdExcludes") or [])
    settings["claudeMdExcludes"] = have + [e for e in excludes if e not in have]
    settings[AUTO_MEMORY_KEY] = False
    return json.dumps(settings)


def _mcp_flags(cmd):
    """The MCP flags cmd still needs: --strict-mcp-config, and an empty config if it names none."""
    flags = [] if STRICT_MCP_FLAG in cmd else [STRICT_MCP_FLAG]
    if not any(a == "--mcp-config" or a.startswith("--mcp-config=") for a in cmd):
        flags.append("--mcp-config=" + EMPTY_MCP_CONFIG)
    return flags


def isolated_claude_cmd(base_cmd, environ=None, home=None):
    """Return (cmd, env) for running base_cmd = [claude_binary, *args] isolated.

    cmd: a new list. `--settings <json>` with claudeMdExcludes and "autoMemoryEnabled": false
         is inserted right after the binary. If base_cmd already carries --settings (as
         `--settings X` or `--settings=X`), both are merged into that JSON instead, rather than
         relying on how the CLI resolves a repeated flag. `--disable-slash-commands`,
         `--strict-mcp-config` and `--mcp-config={"mcpServers":{}}` follow; a caller's own
         --mcp-config is kept and no empty one is added, and a flag the caller already passes
         (--strict-mcp-config, --disable-slash-commands) is not repeated.
    env: a copy of `environ` (default os.environ) plus LYRA_NO_CAPTURE=1 and
         CLAUDE_CODE_DISABLE_AUTO_MEMORY=1. Pass it as env=.

    base_cmd and environ are not modified.
    """
    excludes = claude_md_excludes(home)
    cmd = list(base_cmd)
    insert = []
    for i, arg in enumerate(cmd):
        if arg == "--settings" and i + 1 < len(cmd):
            cmd[i + 1] = _merge_settings(cmd[i + 1], excludes)
            break
        if arg.startswith("--settings="):
            cmd[i] = "--settings=" + _merge_settings(arg[len("--settings="):], excludes)
            break
    else:
        insert += ["--settings", json.dumps({"claudeMdExcludes": excludes, AUTO_MEMORY_KEY: False})]
    if NO_SKILLS_FLAG not in cmd:
        insert.append(NO_SKILLS_FLAG)
    cmd[1:1] = insert + _mcp_flags(cmd)
    env = dict(os.environ if environ is None else environ)
    env[NO_CAPTURE_ENV] = "1"
    env[NO_AUTO_MEMORY_ENV] = "1"
    return cmd, env
