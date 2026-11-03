#!/usr/bin/env python3
"""test_isolation.py: Agni's CLI reviewer must not run as Lyra.

OFFLINE (default; launches nothing):
    python tools/agni/test_isolation.py
  Stubs subprocess.run and calls infer_cli(), so it checks the exact command and environment
  Agni would use: --settings with claudeMdExcludes for <home>/.claude/CLAUDE.md and
  <home>/.claude/**, LYRA_NO_CAPTURE=1 in the child env, and every other argument unchanged.

  AGNI_PHASES_PATH=<file> loads a different agni_phases.py instead of the one beside this file.
  Use it to plant a version without the helper (for example the pre-2026-11-02 one from git)
  and watch the infer_cli tests FAIL. A gate that has never failed has not been tested.

LIVE (optional; two haiku calls; needs the CLI logged in):
    python tools/agni/test_isolation.py --live
  Asks for the memory namespace in a "Quick Reference" section, once through infer_cli
  (isolated) and once as a bare control call without the helper. Isolated must answer NONE.
  The control must answer something else, which shows the probe CAN fail. The prompt does not
  contain the word it is looking for (an earlier probe did, so it could not fail). The control
  keeps LYRA_NO_CAPTURE=1 so the probe is not filed into Lyra's episodic memory; that can only
  make the control less likely to find the namespace, so it does not weaken the check.
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

    def test_everything_else_unchanged(self):
        cmd, kw = self.call(model="haiku", add_dirs=["dir_a", "dir_b"])
        self.assertEqual(strip_settings(cmd),
                         [FAKE_BIN, "-p", "--model", "haiku", "--add-dir", "dir_a", "--add-dir", "dir_b"])
        self.assertIn("THE SYSTEM", kw["input"])
        self.assertTrue(kw["input"].endswith("THE PROMPT"))
        for k, v in (("capture_output", True), ("text", True), ("encoding", "utf-8"),
                     ("errors", "replace"), ("timeout", 7)):
            self.assertEqual(kw.get(k), v, k)
        cmd2, _ = self.call(model="qwen3.8:27b-mxfp8")   # ollama tags are never forwarded
        self.assertEqual(strip_settings(cmd2), [FAKE_BIN, "-p"])


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


PROBE = ("Look through the instructions and context you were given for a section headed "
         "'Quick Reference'. If such a section names a memory namespace, reply with that "
         "namespace name and nothing else. If there is no such section, reply with exactly "
         "the word NONE and nothing else.")


def _one_line(s, n=60):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + "..."


def live():
    """One isolated call through infer_cli, one bare control. Returns a process exit code."""
    mod = load_agni_phases()
    if not mod.CLAUDE_BIN:
        print("LIVE SKIP: no claude binary found")
        return 2
    iso = mod.infer_cli(PROBE, None, 240, model="haiku")
    ctl_env = dict(os.environ)
    ctl_env[NO_CAPTURE_ENV] = "1"
    r = subprocess.run([mod.CLAUDE_BIN, "-p", "--model", "haiku"], input=PROBE,
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=240, env=ctl_env)
    ctl = r.stdout.strip() if r.returncode == 0 else "<control exit %d>" % r.returncode
    norm = lambda s: s.strip().strip(".`'\"").upper()
    iso_ok = norm(iso) == "NONE"
    ctl_ok = r.returncode == 0 and norm(ctl) not in ("", "NONE")
    print("isolated (through infer_cli): %s  -> %s" % (_one_line(iso), "PASS" if iso_ok else "FAIL"))
    print("control  (no helper)        : %s  -> %s" % (_one_line(ctl), "PASS" if ctl_ok else "FAIL"))
    if not ctl_ok:
        print("Control did not find the namespace, so the probe cannot tell isolated from not.")
    return 0 if (iso_ok and ctl_ok) else 1


if __name__ == "__main__":
    if "--live" in sys.argv:
        sys.argv.remove("--live")
        sys.exit(live())
    unittest.main()
