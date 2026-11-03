#!/usr/bin/env python3
"""test_isolation.py: Agni's CLI reviewer must not run as Lyra.

OFFLINE (default; launches nothing):
    python tools/agni/test_isolation.py
  Stubs subprocess.run and calls infer_cli(), so it checks the exact command and environment
  Agni would use: --settings with claudeMdExcludes for <home>/.claude/CLAUDE.md and
  <home>/.claude/**, LYRA_NO_CAPTURE=1 in the child env, --strict-mcp-config with an empty
  inline --mcp-config, and every other argument unchanged.

  AGNI_PHASES_PATH=<file> loads a different agni_phases.py instead of the one beside this file.
  Use it to plant a version without the helper (for example the pre-2026-11-02 one from git)
  and watch the infer_cli tests FAIL. A gate that has never failed has not been tested.

LIVE (optional; six haiku calls; needs the CLI logged in):
    python tools/agni/test_isolation.py --live
  1. CLAUDE.md. Asks for the memory namespace in a "Quick Reference" section, once through
  infer_cli (isolated) and once as a bare control call without the helper. Isolated must answer
  NONE. The control must answer something else, which shows the probe CAN fail. The prompt does
  not contain the word it is looking for (an earlier probe did, so it could not fail). The
  control keeps LYRA_NO_CAPTURE=1 so the probe is not filed into Lyra's episodic memory; that
  can only make the control less likely to find the namespace, so it does not weaken the check.
  MCP (added 2026-11-03). Each control is infer_cli's own command with only the MCP flags
  removed, so it still excludes CLAUDE.md and keeps LYRA_NO_CAPTURE=1: a server it sees comes
  from the MCP config, not from CLAUDE.md's text. Each control must see at least one server, or
  the probe says it cannot tell.
  2. Asks the session for the server part of every mcp__<server>__<tool> tool name it has
  (deferred ones included). Isolated (through infer_cli) must answer NONE.
  3. Reads the CLI's stream-json init event, which lists the MCP servers it loaded (with their
  status) and its tools, with no model judgement in the loop. Isolated must list no server and
  no mcp__ tool.
"""
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from claude_isolation import NO_CAPTURE_ENV, claude_md_excludes, isolated_claude_cmd  # noqa: E402

HOME = Path.home().as_posix().rstrip("/")
WANT_EXCLUDES = {HOME + "/.claude/CLAUDE.md", HOME + "/.claude/**"}
FAKE_BIN = "claude-under-test"
STRICT = "--strict-mcp-config"


def load_agni_phases():
    path = Path(os.environ.get("AGNI_PHASES_PATH") or HERE / "agni_phases.py")
    spec = importlib.util.spec_from_file_location("agni_phases_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def settings_of(cmd):
    """The parsed JSON of the single --settings flag in cmd, or None."""
    vals = [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == "--settings"]
    vals += [a[len("--settings="):] for a in cmd if a.startswith("--settings=")]
    if len(vals) != 1:
        return None
    return json.loads(vals[0])


def strip_settings(cmd):
    out, skip = [], False
    for a in cmd:
        if skip:
            skip = False
        elif a == "--settings":
            skip = True
        elif not a.startswith("--settings="):
            out.append(a)
    return out


def mcp_configs_of(cmd):
    """Every --mcp-config value in cmd, both forms (the two-token form read as one value)."""
    vals = [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == "--mcp-config"]
    return vals + [a[len("--mcp-config="):] for a in cmd if a.startswith("--mcp-config=")]


def strip_mcp(cmd):
    """cmd without --strict-mcp-config and --mcp-config (either form)."""
    out, skip = [], False
    for a in cmd:
        if skip:
            skip = False
        elif a == "--mcp-config":
            skip = True
        elif a != STRICT and not a.startswith("--mcp-config="):
            out.append(a)
    return out


def strip_isolation(cmd):
    return strip_mcp(strip_settings(cmd))


class InferCliIsIsolated(unittest.TestCase):
    """The command infer_cli() actually hands to subprocess.run."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_agni_phases()

    def call(self, **kw):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"], seen["kw"] = list(cmd), kwargs
            return SimpleNamespace(returncode=0, stdout=" ok \n", stderr="")

        with mock.patch.object(self.mod, "CLAUDE_BIN", FAKE_BIN), \
                mock.patch.object(self.mod.subprocess, "run", fake_run):
            out = self.mod.infer_cli("THE PROMPT", "THE SYSTEM", 7, **kw)
        self.assertEqual(out, "ok")
        return seen["cmd"], seen["kw"]

    def test_settings_exclude_user_claude_md(self):
        cmd, _ = self.call(model="haiku")
        s = settings_of(cmd)
        self.assertIsNotNone(s, "infer_cli passed no single --settings flag: %r" % cmd)
        self.assertTrue(WANT_EXCLUDES <= set(s.get("claudeMdExcludes") or []),
                        "claudeMdExcludes missing %r" % (WANT_EXCLUDES - set(s.get("claudeMdExcludes") or [])))

    def test_env_has_no_capture(self):
        _, kw = self.call(model="haiku")
        env = kw.get("env")
        self.assertIsNotNone(env, "infer_cli passed no env= (child inherits capture hooks)")
        self.assertEqual(env.get(NO_CAPTURE_ENV), "1")
        rest = {k: v for k, v in env.items() if k != NO_CAPTURE_ENV}
        base = {k: v for k, v in os.environ.items() if k != NO_CAPTURE_ENV}
        self.assertEqual(rest, base, "env must be os.environ plus LYRA_NO_CAPTURE only")

    def test_mcp_is_strict_and_empty(self):
        cmd, _ = self.call(model="haiku")
        self.assertEqual(cmd.count(STRICT), 1, "infer_cli must pass --strict-mcp-config once: %r" % cmd)
        self.assertEqual([json.loads(c) for c in mcp_configs_of(cmd)], [{"mcpServers": {}}])

    def test_everything_else_unchanged(self):
        cmd, kw = self.call(model="haiku", add_dirs=["dir_a", "dir_b"])
        self.assertEqual(strip_isolation(cmd),
                         [FAKE_BIN, "-p", "--model", "haiku", "--add-dir", "dir_a", "--add-dir", "dir_b"])
        self.assertIn("THE SYSTEM", kw["input"])
        self.assertTrue(kw["input"].endswith("THE PROMPT"))
        for k, v in (("capture_output", True), ("text", True), ("encoding", "utf-8"),
                     ("errors", "replace"), ("timeout", 7)):
            self.assertEqual(kw.get(k), v, k)
        cmd2, _ = self.call(model="qwen3.8:27b-mxfp8")   # ollama tags are never forwarded
        self.assertEqual(strip_isolation(cmd2), [FAKE_BIN, "-p"])


class Helper(unittest.TestCase):
    def test_excludes_are_forward_slash(self):
        for e in claude_md_excludes():
            self.assertNotIn(chr(92), e)
        self.assertEqual(set(claude_md_excludes()), WANT_EXCLUDES)

    def test_merges_existing_settings_and_does_not_mutate(self):
        base = ["claude", "-p", "--settings", '{"disableAllHooks": true}', "--model", "haiku"]
        snapshot = list(base)
        cmd, env = isolated_claude_cmd(base, environ={"A": "1"})
        self.assertEqual(base, snapshot)
        self.assertEqual(cmd.count("--settings"), 1)
        s = settings_of(cmd)
        self.assertTrue(s["disableAllHooks"])
        self.assertEqual(set(s["claudeMdExcludes"]), WANT_EXCLUDES)
        self.assertEqual(env, {"A": "1", NO_CAPTURE_ENV: "1"})

    def test_equals_form_and_file_form(self):
        cmd, _ = isolated_claude_cmd(["claude", "--settings={}"])
        self.assertEqual(set(settings_of(cmd)["claudeMdExcludes"]), WANT_EXCLUDES)
        with self.assertRaises(ValueError):
            isolated_claude_cmd(["claude", "--settings", "some/settings.json"])

    def test_strict_mcp_flag_present(self):
        cmd, _ = isolated_claude_cmd(["claude", "-p"])
        self.assertEqual(cmd.count(STRICT), 1, cmd)

    def test_empty_mcp_config_added_when_none(self):
        cmd, _ = isolated_claude_cmd(["claude", "-p", "--model", "haiku"])
        self.assertEqual([json.loads(c) for c in mcp_configs_of(cmd)], [{"mcpServers": {}}])
        # ONE token: --mcp-config takes several values, so in the two-token form a positional
        # prompt placed after it would be read as another config.
        self.assertNotIn("--mcp-config", cmd)
        self.assertEqual(strip_isolation(cmd), ["claude", "-p", "--model", "haiku"])

    def test_callers_mcp_config_kept_and_not_duplicated(self):
        for base in (["claude", "-p", "--mcp-config", "servers.json", "--model", "haiku"],
                     ["claude", "-p", "--mcp-config=servers.json"]):
            cmd, _ = isolated_claude_cmd(base)
            self.assertEqual(mcp_configs_of(cmd), ["servers.json"], cmd)
            self.assertEqual(cmd.count(STRICT), 1, cmd)
            self.assertEqual([a for a in strip_settings(cmd) if a != STRICT], base)

    def test_callers_strict_flag_not_repeated(self):
        cmd, _ = isolated_claude_cmd(["claude", "-p", STRICT])
        self.assertEqual(cmd.count(STRICT), 1, cmd)
        self.assertEqual([json.loads(c) for c in mcp_configs_of(cmd)], [{"mcpServers": {}}])

    def test_base_cmd_not_mutated(self):
        for base in (["claude", "-p"], ["claude", "-p", "--mcp-config", "servers.json"]):
            snapshot = list(base)
            cmd, _ = isolated_claude_cmd(base)
            self.assertEqual(base, snapshot)
            self.assertIsNot(cmd, base)


PROBE = ("Look through the instructions and context you were given for a section headed "
         "'Quick Reference'. If such a section names a memory namespace, reply with that "
         "namespace name and nothing else. If there is no such section, reply with exactly "
         "the word NONE and nothing else.")

# MCP_PROBE wording is a POST-DATA REVISION (2026-11-03). The first wording was: "Name every MCP
# server you can see anywhere in your instructions, context or tool list, whatever its state:
# connected, connecting, deferred or failed. Reply with the server names, comma-separated, and
# nothing else. If you can see no MCP server at all, reply with exactly the word NONE and nothing
# else." It passed once, then the isolated call answered "MCP Memory Service": the description
# of a user-level skill, not a server (the CLI's init event for that command lists no
# servers). A question that lets prose about MCP count can fail with no server loaded, whatever
# the isolation does. Principle: ask only for what a loaded server alone can put in context, an
# mcp__<server>__<tool> name. Cost: a configured server that exposes no tools (failed, still
# connecting) is invisible to this question. That is not hypothetical: on the first runs
# lyra-memory was "pending" at init and the control's answer did not name it. The init-event
# check below, which reads the CLI's own server list, carries that case (it listed
# lyra-memory(pending)).
MCP_PROBE = ("Some of your tools may be MCP tools, named mcp__<server>__<tool>; deferred tools may "
             "be listed by name only. Reply with every distinct <server> part of those tool names, "
             "comma-separated, and nothing else. Skills, slash commands and text that only mention "
             "MCP do not count. If none of your tool names begins with mcp__, reply with exactly "
             "the word NONE and nothing else.")


def _one_line(s, n=60):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + "..."


def _ask(cmd, env, prompt):
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=240, env=env)
    return r.returncode, (r.stdout.strip() if r.returncode == 0 else "<control exit %d>" % r.returncode)


def _infer_cli_command(mod):
    """The (cmd, env) infer_cli() would run for a haiku call, captured without running it."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["env"] = list(cmd), kwargs.get("env")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with mock.patch.object(mod.subprocess, "run", fake_run):
        mod.infer_cli("x", None, 240, model="haiku")
    return seen["cmd"], seen["env"]


def _init_event(cmd, env):
    """The CLI's own record of what it loaded: the system/init event of a stream-json run."""
    r = subprocess.run(cmd + ["--output-format", "stream-json", "--verbose"], input="Reply with OK.",
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=240, env=env)
    for line in r.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "system" and ev.get("subtype") == "init":
            return ev
    return None


def _mcp_loaded(ev):
    servers = ["%s(%s)" % (s.get("name"), s.get("status")) for s in ev.get("mcp_servers") or []]
    tools = [t for t in ev.get("tools") or [] if str(t).startswith("mcp__")]
    return servers, tools


def live():
    """Three probes, each an isolated run and a control. Returns a process exit code."""
    mod = load_agni_phases()
    if not mod.CLAUDE_BIN:
        print("LIVE SKIP: no claude binary found")
        return 2
    norm = lambda s: s.strip().strip(".`'\"").upper()
    results = []

    # 1. CLAUDE.md: the control is a bare call without the helper.
    iso = mod.infer_cli(PROBE, None, 240, model="haiku")
    ctl_env = dict(os.environ)
    ctl_env[NO_CAPTURE_ENV] = "1"
    rc, ctl = _ask([mod.CLAUDE_BIN, "-p", "--model", "haiku"], ctl_env, PROBE)
    iso_ok = norm(iso) == "NONE"
    ctl_ok = rc == 0 and norm(ctl) not in ("", "NONE")
    print("CLAUDE.md isolated (through infer_cli): %s  -> %s" % (_one_line(iso), "PASS" if iso_ok else "FAIL"))
    print("CLAUDE.md control  (no helper)        : %s  -> %s" % (_one_line(ctl), "PASS" if ctl_ok else "FAIL"))
    if not ctl_ok:
        print("Control did not find the namespace, so the probe cannot tell isolated from not.")
    results += [iso_ok, ctl_ok]

    # 2 and 3. MCP. Both controls are infer_cli's own command with only the MCP flags removed,
    # so they still exclude CLAUDE.md and keep LYRA_NO_CAPTURE=1: a server they see comes from
    # the MCP config, not from CLAUDE.md's text.
    icmd, ienv = _infer_cli_command(mod)
    ccmd = strip_mcp(icmd)

    # 2. Ask the session (the isolated call goes through infer_cli itself).
    miso = mod.infer_cli(MCP_PROBE, None, 240, model="haiku")
    mrc, mctl = _ask(ccmd, ienv, MCP_PROBE)
    miso_ok = norm(miso) == "NONE"
    mctl_ok = mrc == 0 and norm(mctl) not in ("", "NONE")
    print("MCP asked, isolated (infer_cli)       : %s  -> %s" % (_one_line(miso), "PASS" if miso_ok else "FAIL"))
    print("MCP asked, control (minus MCP flags)  : %s  -> %s" % (_one_line(mctl, 2000), "PASS" if mctl_ok else "FAIL"))
    if not mctl_ok:
        print("Control named no MCP server, so the question cannot tell isolated from not.")
    results += [miso_ok, mctl_ok]

    # 3. Read the CLI's init event: what it loaded, with no model in the loop.
    iev, cev = _init_event(icmd, ienv), _init_event(ccmd, ienv)
    if iev is None or cev is None:
        print("MCP init event: not found (isolated %s, control %s), so this check cannot tell."
              % ("found" if iev else "missing", "found" if cev else "missing"))
        results.append(False)
    else:
        isrv, itools = _mcp_loaded(iev)
        csrv, ctools = _mcp_loaded(cev)
        ok_i, ok_c = not isrv and not itools, bool(csrv)
        print("MCP init, isolated (infer_cli cmd)    : %d servers, %d mcp__ tools  -> %s"
              % (len(isrv), len(itools), "PASS" if ok_i else "FAIL"))
        print("MCP init, control (minus MCP flags)   : %d servers, %d mcp__ tools  -> %s"
              % (len(csrv), len(ctools), "PASS" if ok_c else "FAIL"))
        print("  control servers: %s" % _one_line(", ".join(csrv), 2000))
        if isrv:
            print("  isolated servers: %s" % ", ".join(isrv))
        if not ok_c:
            print("Control loaded no MCP server, so the init check cannot tell isolated from not.")
        results += [ok_i, ok_c]
    return 0 if all(results) else 1


if __name__ == "__main__":
    if "--live" in sys.argv:
        sys.argv.remove("--live")
        sys.exit(live())
    unittest.main()
