#!/usr/bin/env python3
"""claude_isolation.py: run the Claude Code CLI as a reviewer, not as the account's owner.

WHY THIS EXISTS (2026-11-02). `claude -p` launched from Lyra's account IS Lyra unless it is
isolated. It loads the user-level CLAUDE.md (identity, family list, the Key Numbers table),
runs the SessionStart sync, and the Stop hook files the exchange into Lyra's episodic memory.
Every Agni review from 2026-08-14 to 2026-11-02 ran that way. Six things close it:

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
  6. No hooks of any kind (added 2026-11-07): "disableAllHooks": true in the same --settings JSON.
     LYRA_NO_CAPTURE stops only Lyra's own hooks. On the lab server the CLI runs as an account
     whose user settings carry another agent's SessionStart hook, which would put that agent's
     context into every review. Checked live (CLI 2.1.301): a SessionStart hook planted in
     --settings fired without the key and did not fire with it. A caller's "disableAllHooks":
     false is overridden.
  7. No transcript left in the account (added 2026-11-08): --no-session-persistence, added to a
     print-mode (-p / --print) call unless the caller already passes it. Every review run on the
     lab server had left its whole session, prompt included, under that account's
     ~/.claude/projects (Lyra, 2026-11-08), where the account owner's own tools could later read
     it or file it into their memory: the same leak as items 1 to 6, in the other direction. The
     flag works only in print mode, so other calls are left alone. A caller that needs the
     transcript (to audit the session afterwards) passes keep_transcript=True.
  8. None of the calling session's own variables (added 2026-11-08). Run from inside a Claude Code
     session, the child inherited that session's environment: CLAUDECODE, its session and bridge ids,
     CLAUDE_CODE_SESSION_ATTENDED, and CLAUDE_CODE_MESSAGING_SOCKET with CLAUDE_CODE_MESSAGING_TOKEN, a
     way back into the calling session. The CLI approves read-only Bash on its own in print mode (seen on
     CLI 2.1.309: find, sha256sum and grep ran unasked), so a reviewer could print the token into its
     transcript with `env`.
     1.1.3 (Lyra, 2026-11-08): an ALLOWLIST, not a longer denylist. In Lyra's real session the
     denylist still passed GEMINI_API_KEY and an OPENAI_MCP_SERVERS. The child now gets only what the
     CLI needs to start and reach the API (ENV_ALLOW, ENV_ALLOW_PREFIXES: PATH, home and temp folders,
     the Windows system set, locale, TERM, proxies and CA files, CLAUDE_CONFIG_DIR) and our own
     switches. Everything else goes, keys and tokens included, whatever they are named. Names are
     compared without case (Windows). A machine whose CLI logs in through the environment
     (ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN, or Bedrock's CLAUDE_CODE_USE_BEDROCK and AWS_*) passes
     those names as keep_env; on Asdf the CLI logs in from its stored credentials and needs none
     (checked live, 2026-11-08).

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
NO_HOOKS_KEY = "disableAllHooks"
STRICT_MCP_FLAG = "--strict-mcp-config"
NO_SKILLS_FLAG = "--disable-slash-commands"
NO_PERSIST_FLAG = "--no-session-persistence"
PRINT_FLAGS = ("-p", "--print")
# What the child may inherit (compared upper-cased). Everything else is dropped.
ENV_ALLOW = frozenset((
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TEMP", "TMP", "XDG_RUNTIME_DIR",
    "USERPROFILE", "APPDATA", "LOCALAPPDATA", "HOMEDRIVE", "HOMEPATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
    "SYSTEMDRIVE", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "PROGRAMDATA", "COMMONPROGRAMFILES",
    "LANG", "LANGUAGE", "TERM",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS",
    "CLAUDE_CONFIG_DIR",
))
ENV_ALLOW_PREFIXES = ("LC_",)
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
    settings[NO_HOOKS_KEY] = True
    return json.dumps(settings)


def _mcp_flags(cmd):
    """The MCP flags cmd still needs: --strict-mcp-config, and an empty config if it names none."""
    flags = [] if STRICT_MCP_FLAG in cmd else [STRICT_MCP_FLAG]
    if not any(a == "--mcp-config" or a.startswith("--mcp-config=") for a in cmd):
        flags.append("--mcp-config=" + EMPTY_MCP_CONFIG)
    return flags


def env_allowed(name, keep_env=()):
    """True if the child may inherit this variable: on the allowlist, or named by the caller in keep_env."""
    up = name.upper()
    return up in ENV_ALLOW or up.startswith(ENV_ALLOW_PREFIXES) or up in {k.upper() for k in keep_env}


def isolated_claude_cmd(base_cmd, environ=None, home=None, keep_transcript=False, keep_env=()):
    """Return (cmd, env) for running base_cmd = [claude_binary, *args] isolated.

    cmd: a new list. `--settings <json>` with claudeMdExcludes, "autoMemoryEnabled": false and
         "disableAllHooks": true is inserted right after the binary. If base_cmd already carries --settings (as
         `--settings X` or `--settings=X`), both are merged into that JSON instead, rather than
         relying on how the CLI resolves a repeated flag. `--disable-slash-commands`,
         `--strict-mcp-config` and `--mcp-config={"mcpServers":{}}` follow; a caller's own
         --mcp-config is kept and no empty one is added, and a flag the caller already passes
         (--strict-mcp-config, --disable-slash-commands) is not repeated. In print mode (-p or
         --print in base_cmd), `--no-session-persistence` follows too, unless the caller passes it
         or keep_transcript=True.
    env: only the variables of `environ` (default os.environ) on the allowlist (ENV_ALLOW,
         ENV_ALLOW_PREFIXES) or named in keep_env, plus LYRA_NO_CAPTURE=1 and
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
        insert += ["--settings", json.dumps({"claudeMdExcludes": excludes, AUTO_MEMORY_KEY: False,
                                             NO_HOOKS_KEY: True})]
    if NO_SKILLS_FLAG not in cmd:
        insert.append(NO_SKILLS_FLAG)
    if not keep_transcript and NO_PERSIST_FLAG not in cmd and any(a in PRINT_FLAGS for a in cmd):
        insert.append(NO_PERSIST_FLAG)
    cmd[1:1] = insert + _mcp_flags(cmd)
    env = {k: v for k, v in (os.environ if environ is None else environ).items() if env_allowed(k, keep_env)}
    env[NO_CAPTURE_ENV] = "1"
    env[NO_AUTO_MEMORY_ENV] = "1"
    return cmd, env
