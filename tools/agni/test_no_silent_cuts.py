#!/usr/bin/env python3
"""test_no_silent_cuts.py: a review may never look complete when the reviewer did not get all of
its input.

OFFLINE ONLY; launches nothing and nothing leaves the machine:
    python -X utf8 tools/agni/test_no_silent_cuts.py
  Drives agni_phases.main() through sys.argv as a black box, with infer_cli and llm.infer patched,
  and looks only at what a caller can see: the prompt the reviewer would receive, the exit code and
  the JSON record. A unique canary sits at the LAST character of every input, so a cut of even one
  character loses it. Fixtures are built in temp folders outside the repo and removed.

  AGNI_PHASES_PATH=<file> loads a different agni_phases.py instead of the one beside this file, and
  takes the other modules from the same folder when they are there (agni_design_review.py, llm.py,
  claude_isolation.py, run_style_survey.py, summarize_style.py). Put a copy of them from git in an
  empty folder and watch these tests FAIL on the canaries, verdicts and exit codes. A gate that has
  never failed has not been tested.
"""
import builtins
import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
SUT_PHASES = Path(os.environ.get("AGNI_PHASES_PATH") or HERE / "agni_phases.py").resolve()
SUT_DIR = SUT_PHASES.parent
sys.path.insert(0, str(HERE))
if SUT_DIR != HERE:
    sys.path.insert(0, str(SUT_DIR))

APPROVED = "VERDICT: APPROVED\n\n1. No findings."
WORDS = ("APPROVED", "CONDITIONAL", "REJECTED")
EXIT_FOR = {"APPROVED": 0, "CONDITIONAL": 0, "REJECTED": 1}


def load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sut(filename, name):
    """A module under test: beside AGNI_PHASES_PATH when it is there, else beside this file."""
    path = SUT_DIR / filename
    return load_by_path(path if path.exists() else HERE / filename, name)


PHASES = load_by_path(SUT_PHASES, "agni_phases_under_test")


def fill(n, canary):
    """Exactly n chars of text ending in the canary."""
    body = "a line of filler for the no silent cuts test\n"
    need = n - len(canary)
    return (body * (need // len(body) + 1))[:need] + canary


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="agni_nsc_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "record.json"

    def put(self, rel, content):
        """Write bytes exactly, so nothing on the way adds a CR the test did not plant."""
        p = self.tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        return str(p)

    def prereg(self, rel, obj):
        return self.put(rel, json.dumps(obj))

    def stale(self, path=None):
        """An old APPROVED record sitting where this run will write."""
        p = Path(path) if path else self.out
        p.write_text(json.dumps({"verdict": "APPROVED", "review": "an old pass"}), encoding="utf-8")

    def run_main(self, argv, reply=APPROVED, env=None, local=None, real_cli=False, on_call=None,
                 use_out=True, cli_result=None):
        """Drive main() through sys.argv. reply: a string, an exception to raise, or a list of those
        used in order (the last repeats). cli_result: with real_cli, what subprocess.run returns
        instead of a success carrying reply. Returns what a caller could see."""
        mod = PHASES
        res = SimpleNamespace(code=None, prompts=[], timeouts=[], local_prompts=[], stdout="",
                              stderr="", crash=None, record=None, cli_cmds=[], cli_envs=[])
        replies = list(reply) if isinstance(reply, list) else [reply]

        def next_reply():
            r = replies.pop(0) if len(replies) > 1 else replies[0]
            if isinstance(r, BaseException):
                raise r
            return r

        def fake_cli(prompt, system, timeout, add_dirs=None, model=None):
            res.prompts.append(prompt)
            res.timeouts.append(timeout)
            if on_call:
                on_call()
            return next_reply()

        def fake_llm(prompt=None, **kw):
            res.local_prompts.append(prompt)
            return local

        def fake_run(cmd, **kw):
            res.prompts.append(kw.get("input"))
            res.cli_cmds.append(list(cmd))
            res.cli_envs.append(kw.get("env"))
            return cli_result or SimpleNamespace(returncode=0, stdout=reply, stderr="")

        clean = {k: v for k, v in os.environ.items() if not k.startswith(("AGNI_", "LLM_"))}
        clean.update(env or {})
        args = ["agni_phases.py"] + [str(a) for a in argv]
        if use_out:
            args += ["--out", str(self.out)]
        cli_patch = (mock.patch.object(mod.subprocess, "run", fake_run) if real_cli
                     else mock.patch.object(mod, "infer_cli", fake_cli))
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(sys, "argv", args), \
                cli_patch, mock.patch.object(mod, "CLAUDE_BIN", "claude-under-test"), \
                mock.patch.object(mod.llm, "infer", fake_llm), \
                mock.patch.object(mod.time, "sleep", lambda s: None), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                res.code = mod.main()
            except SystemExit as e:
                res.code = 1 if isinstance(e.code, str) else e.code
            except Exception as e:  # a traceback exits 1, which by the contract reads as REJECTED
                res.crash, res.code = e, 1
        res.stdout, res.stderr = stdout.getvalue(), stderr.getvalue()
        if use_out and self.out.is_file():
            res.record = json.loads(self.out.read_text(encoding="utf-8"))
        return res

    def assertDelivered(self, res, *canaries):
        self.assertEqual(len(res.prompts), 1, "want exactly one reviewer call; crash=%r" % res.crash)
        for c in canaries:
            self.assertIn(c, res.prompts[0], "canary %s absent: the reviewer did not get that input's "
                                             "last character" % c)

    def assertRefused(self, res):
        total = len(res.prompts) + len(res.local_prompts)
        self.assertEqual(total, 0, "a refused run must make no reviewer call; it made %d" % total)
        self.assertEqual(res.code, 2, "exit code; crash=%r" % res.crash)
        self.assertEqual((res.record or {}).get("verdict"), "INVALID",
                         "record verdict must be INVALID, record=%s" % str(res.record)[:120])
        self.assertIn("NOT a pass", res.stderr)
        self.assertIn("UNGATED", res.stderr)

    def coverage(self, res):
        rec = res.record or {}
        self.assertIn("input_coverage", rec, "the record does not say what the reviewer was given")
        return rec["input_coverage"]

    def item(self, res, role):
        got = [i for i in self.coverage(res)["inputs"] if i.get("role") == role]
        self.assertEqual(len(got), 1, "coverage entries for %s: %r" % (role, got))
        return got[0]


class WholeInputs(Base):
    """D1, D2, D3: the canary at the last character of every input reaches the reviewer."""

    def test_file_over_30k(self):
        p = self.put("paper.md", fill(40000, "CANARY_FILE_END"))
        self.assertDelivered(self.run_main(["style", p]), "CANARY_FILE_END")

    def test_folder_of_26_files_reaches_the_26th(self):
        art = self.put("art.md", "the artifact")
        for i in range(26):
            self.put("data/f%02d.txt" % i, "file %d CANARY_F%02d" % (i, i))
        res = self.run_main(["results", art, self.tmp / "data"])
        self.assertDelivered(res, "CANARY_F00", "CANARY_F25")

    def test_dir_file_over_4k(self):
        art = self.put("art.md", "the artifact")
        self.put("data/big.txt", fill(5000, "CANARY_DIRFILE_END"))
        self.assertDelivered(self.run_main(["results", art, self.tmp / "data"]), "CANARY_DIRFILE_END")

    def test_nested_subfolder_with_relative_name(self):
        art = self.put("art.md", "the artifact")
        self.put("data/top.txt", "top")
        self.put("data/sub/deep/leaf.txt", "CANARY_NESTED")
        res = self.run_main(["results", art, self.tmp / "data"])
        self.assertDelivered(res, "CANARY_NESTED")
        self.assertIn("--- sub/deep/leaf.txt ---", res.prompts[0])

    def test_joined_folder_over_the_old_cap(self):
        art = self.put("art.md", "the artifact")
        for i in range(10):
            self.put("data/f%d.txt" % i, fill(3500, "CANARY_JOIN%d" % i))
        self.assertDelivered(self.run_main(["results", art, self.tmp / "data"]), "CANARY_JOIN9")

    def test_primary_folder(self):
        for i in range(12):
            self.put("dir/f%02d.md" % i, fill(3000, "CANARY_PD%02d" % i))
        self.assertDelivered(self.run_main(["results", self.tmp / "dir"]), "CANARY_PD11")

    def test_secondary_over_12k(self):
        a = self.put("a.md", "primary")
        b = self.put("b.md", fill(13000, "CANARY_SECONDARY_END"))
        self.assertDelivered(self.run_main(["style", a, b]), "CANARY_SECONDARY_END")

    def test_tertiary_over_8k(self):
        a = self.put("a.md", "primary")
        b = self.put("b.md", "secondary")
        c = self.put("c.py", fill(9000, "CANARY_TERTIARY_END"))
        self.assertDelivered(self.run_main(["paper", a, b, c]), "CANARY_TERTIARY_END")

    def test_real_cli_path_carries_the_whole_prompt_and_stays_isolated(self):
        # R11: the isolation path is unchanged, seen from main() down to subprocess.run.
        p = self.put("paper.md", fill(40000, "CANARY_CLI_END"))
        res = self.run_main(["style", p], real_cli=True)
        self.assertDelivered(res, "CANARY_CLI_END")
        self.assertIn("--settings", res.cli_cmds[0])
        self.assertIn("--strict-mcp-config", res.cli_cmds[0])
        self.assertEqual(res.cli_envs[0].get("LYRA_NO_CAPTURE"), "1")


class Prereg(Base):
    """D4, R5."""

    def test_hypotheses_and_threats_over_6k_go_whole_and_stay_valid_json(self):
        hyp = [{"id": "H1", "text": fill(7000, "CANARY_HYP_END")}]
        threats = [{"risk": fill(7000, "CANARY_THREAT_END")}]
        pr = self.prereg("prereg.json", {"hypotheses": hyp, "threats_to_validity": threats})
        script = self.put("run.py", "print('hello')")
        for phase in ("design", "preflight"):
            with self.subTest(phase=phase):
                res = self.run_main([phase, script, pr])
                self.assertDelivered(res, "CANARY_HYP_END", "CANARY_THREAT_END")
                self.assertIn(json.dumps(hyp, indent=2), res.prompts[0])
                self.assertIn(json.dumps(threats, indent=2), res.prompts[0])

    def test_singular_hypothesis_key_still_works(self):
        pr = self.prereg("prereg.json", {"hypothesis": "CANARY_SINGULAR", "threats_to_validity": ["t"]})
        self.assertDelivered(self.run_main(["design", self.put("run.py", "x = 1"), pr]), "CANARY_SINGULAR")

    def test_no_hypotheses_key_sends_the_whole_prereg_not_empty_braces(self):
        pr = self.prereg("prereg.json", {"aim": fill(500, "CANARY_WHOLE_PREREG"),
                                         "threats_to_validity": ["t"]})
        res = self.run_main(["design", self.put("run.py", "x = 1"), pr])
        self.assertDelivered(res, "CANARY_WHOLE_PREREG")
        self.assertNotIn("PRE-REGISTERED HYPOTHESES:\n{}", res.prompts[0])
        self.assertTrue(self.item(res, "secondary").get("secondary_is_whole_prereg"))
        self.assertIn("whole prereg", res.prompts[0].lower())

    def test_empty_hypotheses_value_also_sends_the_whole_prereg(self):
        pr = self.prereg("prereg.json", {"hypotheses": [], "aim": "CANARY_EMPTY_HYP",
                                         "threats_to_validity": ["t"]})
        self.assertDelivered(self.run_main(["design", self.put("run.py", "x = 1"), pr]), "CANARY_EMPTY_HYP")

    def test_threats_warning_kept(self):
        pr = self.prereg("prereg.json", {"hypotheses": ["h"]})
        res = self.run_main(["design", self.put("run.py", "x = 1"), pr])
        self.assertEqual(len(res.prompts), 1)
        self.assertIn("WARNING: prereg has no threats_to_validity", res.stderr)

    def test_other_keys_go_whole_under_their_own_label(self):
        # Round 3: the reviewer gets the whole prereg. Other keys used to be named and not sent, and
        # nearly every tracked prereg has them, so a design review of one exited 2.
        other = {"falsification_rule": fill(3000, "CANARY_OTHER_END"), "analysis_plan": {"n": 40, "alpha": 0.05}}
        prereg = dict({"hypotheses": ["CANARY_H"], "threats_to_validity": ["CANARY_T"]}, **other)
        pr = self.prereg("prereg.json", prereg)
        for phase in ("design", "preflight"):
            with self.subTest(phase=phase):
                res = self.run_main([phase, self.put("run.py", "x = 1"), pr])
                self.assertDelivered(res, "CANARY_OTHER_END", "CANARY_H", "CANARY_T")
                prompt = res.prompts[0]
                self.assertIn("OTHER PRE-REGISTRATION FIELDS", prompt)
                block = prompt[prompt.rindex("OTHER PRE-REGISTRATION FIELDS"):]
                self.assertIn(json.dumps(other, indent=2), block, "every other key, whole, as indented JSON")
                self.assertNotIn("CANARY_T", block.split("THREATS")[0], "threats go once, as tertiary")
                sec = self.item(res, "secondary")
                self.assertEqual(sec.get("prereg_other_keys_sent"), ["analysis_plan", "falsification_rule"])
                self.assertEqual(sec.get("prereg_keys_not_sent"), [])
                self.assertIs(self.coverage(res)["input_complete"], True)
                self.assertEqual(res.code, 0)

    def test_prereg_of_only_those_keys_is_complete(self):
        pr = self.prereg("prereg.json", {"hypotheses": ["h"], "threats_to_validity": ["t"]})
        res = self.run_main(["design", self.put("run.py", "x = 1"), pr])
        self.assertIs(self.coverage(res)["input_complete"], True)
        self.assertEqual(self.item(res, "secondary")["kind"], "prereg")


class Refusal(Base):
    """D5, R1, R2: refuse before any reviewer call."""

    def test_missing_named_secondary(self):
        res = self.run_main(["paper", self.put("a.md", "primary"), self.tmp / "nope.json"])
        self.assertRefused(res)
        self.assertIn("nope.json", res.stderr)

    def test_missing_named_tertiary(self):
        res = self.run_main(["paper", self.put("a.md", "x"), self.put("b.md", "y"), self.tmp / "nope.py"])
        self.assertRefused(res)

    def test_missing_primary(self):
        self.assertRefused(self.run_main(["style", self.tmp / "nope.md"]))

    def test_missing_prereg(self):
        self.assertRefused(self.run_main(["design", self.put("run.py", "x = 1"), self.tmp / "nope.json"]))

    def test_design_without_a_prereg(self):
        self.assertRefused(self.run_main(["design", self.put("run.py", "x = 1")]))

    def test_prereg_that_is_not_json(self):
        pr = self.put("prereg.json", "{ this is not json")
        self.assertRefused(self.run_main(["design", self.put("run.py", "x = 1"), pr]))

    def test_prereg_that_is_not_an_object(self):
        pr = self.put("prereg.json", "[1, 2, 3]")
        self.assertRefused(self.run_main(["design", self.put("run.py", "x = 1"), pr]))

    def test_named_input_the_phase_has_no_slot_for(self):
        # style and results have no {tertiary}, and design fills it from the prereg, so a third path
        # used to be read (or not) and then dropped with nothing said.
        a, b = self.put("a.md", "x"), self.put("b.md", "y")
        c = self.put("c.md", "CANARY_THIRD")
        pr = self.prereg("p.json", {"hypotheses": ["h"], "threats_to_validity": ["t"]})
        for argv in (["style", a, b, c], ["results", a, b, c], ["design", a, pr, c], ["preflight", a, pr, c]):
            with self.subTest(phase=argv[0]):
                self.assertRefused(self.run_main(argv))

    def test_unreadable_named_file(self):
        p = self.put("a.md", "primary")
        real = builtins.open

        def deny(path, *a, **kw):
            if str(path) == p:
                raise PermissionError(13, "denied", str(path))
            return real(path, *a, **kw)

        with mock.patch("builtins.open", deny), mock.patch("io.open", deny):
            res = self.run_main(["style", p])
        self.assertRefused(res)

    def test_unreadable_file_inside_a_folder(self):
        a = self.put("a.md", "primary")
        bad = self.put("data/locked.txt", "secret")
        self.put("data/ok.txt", "fine")
        real = builtins.open

        def deny(path, *args, **kw):
            if str(path) == bad:
                raise PermissionError(13, "denied", str(path))
            return real(path, *args, **kw)

        with mock.patch("builtins.open", deny), mock.patch("io.open", deny), \
                mock.patch("pathlib.io.open", deny):
            res = self.run_main(["results", a, self.tmp / "data"])
        self.assertRefused(res)
        self.assertIn("locked.txt", res.stderr)

    def test_over_budget_makes_no_call_and_overwrites_a_stale_approved(self):
        p = self.put("paper.md", fill(400001, "CANARY_OVER"))
        self.stale()
        res = self.run_main(["style", p])
        self.assertRefused(res)
        self.assertIn("paper.md", res.stderr)
        self.assertIn("400,001", res.stderr)
        self.assertIn("AGNI_PROMPT_CHAR_BUDGET", res.stderr)
        self.assertIs(self.coverage(res)["input_complete"], False)

    def test_default_budget_passes_a_file_just_inside_it(self):
        p = self.put("paper.md", fill(390000, "CANARY_INSIDE"))
        self.assertDelivered(self.run_main(["style", p]), "CANARY_INSIDE")

    def test_exact_budget_passes_and_one_over_refuses(self):
        # Measured on the stdin the CLI gets: infer_cli puts SYSTEM and HONESTY_CLAUSE before the
        # prompt, and a budget equal to the prompt alone let 1,179 more chars through (round 2).
        p = self.put("paper.md", fill(5000, "CANARY_EXACT"))
        probe = self.run_main(["style", p], env={"AGNI_PROMPT_CHAR_BUDGET": "100000000"}, real_cli=True)
        self.assertEqual(len(probe.prompts), 1)
        size = len(probe.prompts[0])
        self.assertIn("{:,} chars".format(size), probe.stdout, "the header must count what is sent")
        at = self.run_main(["style", p], env={"AGNI_PROMPT_CHAR_BUDGET": str(size)}, real_cli=True)
        self.assertDelivered(at, "CANARY_EXACT")
        self.assertEqual(at.prompts[0], probe.prompts[0], "the prompt must not depend on the budget")
        self.assertRefused(self.run_main(["style", p], env={"AGNI_PROMPT_CHAR_BUDGET": str(size - 1)},
                                         real_cli=True))

    def test_budget_counts_chars_not_bytes(self):
        # 3-byte chars: the file on disk is far over the budget in bytes, and inside it in chars.
        text = chr(0x2014) * 60000 + "CANARY_MULTIBYTE"
        p = self.put("paper.md", text)
        res = self.run_main(["style", p], env={"AGNI_PROMPT_CHAR_BUDGET": "100000"})
        self.assertDelivered(res, "CANARY_MULTIBYTE")
        self.assertEqual(self.item(res, "primary")["chars"], len(text))

    def test_braces_in_an_input_survive_verbatim(self):
        text = "a {kills} and {primary} and {0} and {} in the text CANARY_BRACES"
        res = self.run_main(["style", self.put("paper.md", text)])
        self.assertDelivered(res, text)

    def test_empty_string_path_is_a_named_input_that_is_missing(self):
        # "$DATA" with DATA unset: the user named an input and it is not there.
        a = self.put("a.md", "x")
        pr = self.prereg("p.json", {"hypotheses": ["h"], "threats_to_validity": ["t"]})
        for argv in (["paper", a, ""], ["paper", a, pr, ""], ["design", a, ""], ["style", ""]):
            with self.subTest(argv=argv[0], n=len(argv)):
                self.assertRefused(self.run_main(argv))

    def test_refusal_names_each_input_and_the_largest_files(self):
        a = self.put("a.md", "primary")
        self.put("data/big_one.txt", fill(3000, "x"))
        self.put("data/mid_one.txt", fill(2000, "y"))
        self.put("data/tiny.txt", "z")
        res = self.run_main(["results", a, self.tmp / "data"], env={"AGNI_PROMPT_CHAR_BUDGET": "1000"})
        self.assertRefused(res)
        for needle in ("a.md", "big_one.txt", "mid_one.txt", "3,000", "narrower"):
            self.assertIn(needle, res.stderr)

    def test_file_far_over_budget_is_refused_and_named(self):
        p = self.put("huge.md", fill(50000, "CANARY_HUGE"))
        res = self.run_main(["style", p], env={"AGNI_PROMPT_CHAR_BUDGET": "1000"})
        self.assertRefused(res)
        self.assertIn("huge.md", res.stderr)

    def test_bad_budget_value_fails_loud(self):
        p = self.put("paper.md", "content")
        res = self.run_main(["style", p], env={"AGNI_PROMPT_CHAR_BUDGET": "lots"})
        self.assertRefused(res)
        self.assertIn("AGNI_PROMPT_CHAR_BUDGET", res.stderr)

    def test_write_failure_on_a_refused_run_still_exits_2_and_says_so(self):
        p = self.put("paper.md", fill(400001, "CANARY_NOWRITE"))
        self.out.mkdir()    # a folder where the record should go: writing to it raises
        res = self.run_main(["style", p])
        self.assertEqual(res.code, 2, "exit code; crash=%r" % res.crash)
        self.assertIn("could not write", res.stderr.lower())
        self.assertEqual(res.prompts, [])
        self.assertIn("NOT a pass", res.stderr)

    def test_unwritable_record_is_found_before_spending_a_review(self):
        p = self.put("paper.md", "small enough")
        self.out.mkdir()
        res = self.run_main(["style", p])
        self.assertEqual(res.code, 2, "exit code; crash=%r" % res.crash)
        self.assertEqual(res.prompts, [], "must not spend a reviewer call it cannot record")
        self.assertIn("could not write", res.stderr.lower())

    def test_stale_approved_is_gone_while_the_reviewer_runs(self):
        # A run killed mid-review must not leave the old pass standing.
        p = self.put("paper.md", "content")
        self.stale()
        seen = {}

        def look():
            seen["verdict"] = json.loads(self.out.read_text(encoding="utf-8")).get("verdict")

        self.run_main(["style", p], on_call=look)
        self.assertEqual(seen.get("verdict"), "INVALID")

    def test_no_review_replaces_a_stale_approved(self):
        p = self.put("paper.md", "content")
        self.stale()
        res = self.run_main(["style", p], reply="")
        self.assertEqual(res.code, 2)
        self.assertEqual((res.record or {}).get("verdict"), "INVALID")
        self.assertIn("NOT a pass", res.stderr)
        # The no-review write itself, not the placeholder from before the call (mutant R06c).
        self.assertIn("gave no review", " ".join((res.record or {}).get("invalid_reasons") or []))

    def test_unparsed_replaces_a_stale_approved_at_the_default_path(self):
        p = self.put("paper.md", "content")
        default = self.tmp / "paper.agni_style.json"
        self.stale(default)
        res = self.run_main(["style", p], reply="no verdict in here", use_out=False)
        self.assertEqual(res.code, 2)
        self.assertEqual(json.loads(default.read_text(encoding="utf-8")).get("verdict"), "UNPARSED")


class Folders(Base):
    """R4."""

    def setUp(self):
        super().setUp()
        self.art = self.put("art.md", "the artifact")

    def test_junk_dirs_skipped_and_counted(self):
        self.put("data/keep.txt", "CANARY_KEEP")
        junk = (".git", "__pycache__", "node_modules", ".venv", ".ipynb_checkpoints")
        for j in junk:
            self.put("data/%s/inner.txt" % j, "CANARY_JUNK_%s" % j)
        res = self.run_main(["results", self.art, self.tmp / "data"])
        self.assertDelivered(res, "CANARY_KEEP")
        for j in junk:
            self.assertNotIn("CANARY_JUNK_%s" % j, res.prompts[0])
        skipped = self.item(res, "secondary")["skipped"]
        self.assertEqual(sorted(s["name"].rstrip("/") for s in skipped), sorted(junk))
        self.assertIs(self.item(res, "secondary")["complete"], True, "junk is not input")

    def test_binary_files_listed_not_shown_and_not_invalidating(self):
        self.put("data/keep.txt", "CANARY_TEXT")
        self.put("data/nul.bin", b"\x00\x01\x02 CANARY_NUL_BYTES")
        self.put("data/noise.dat", bytes(range(128, 256)) * 50 + b"CANARY_NOISE_BYTES")
        res = self.run_main(["results", self.art, self.tmp / "data"])
        self.assertDelivered(res, "CANARY_TEXT")
        self.assertNotIn("CANARY_NUL_BYTES", res.prompts[0])
        self.assertNotIn("CANARY_NOISE_BYTES", res.prompts[0])
        self.assertIn("nul.bin", res.prompts[0])
        self.assertIn("noise.dat", res.prompts[0])
        item = self.item(res, "secondary")
        self.assertEqual(sorted(b["name"] for b in item["binary_not_shown"]), ["noise.dat", "nul.bin"])
        self.assertEqual({b["name"]: b["bytes"] for b in item["binary_not_shown"]}["nul.bin"], 20)
        self.assertEqual(item["files_included"], 1)
        self.assertEqual(res.code, 2, "an APPROVED on incomplete input is not a pass")
        self.assertIs(item["complete"], False, "the reviewer did not get the binary's content")
        self.assertIs(self.coverage(res)["input_complete"], False)

    def test_folder_of_only_binaries_as_secondary_still_runs_and_says_so(self):
        self.put("data/a.npz", b"\x00CANARY_ONLY_BIN")
        res = self.run_main(["results", self.art, self.tmp / "data"])
        self.assertEqual(len(res.prompts), 1)
        self.assertNotIn("CANARY_ONLY_BIN", res.prompts[0])
        self.assertIn("a.npz", res.prompts[0])
        self.assertEqual(self.item(res, "secondary")["files_included"], 0)

    def test_primary_that_is_binary_is_invalid(self):
        self.assertRefused(self.run_main(["style", self.put("paper.pdf", b"%PDF\x00\x01binary")]))

    def test_primary_folder_of_only_binaries_is_invalid(self):
        self.put("dir/a.bin", b"\x00\x00")
        self.assertRefused(self.run_main(["results", self.tmp / "dir"]))

    def test_named_secondary_binary_file_is_listed_not_shown(self):
        b = self.put("twin.pdf", b"\x00\x01CANARY_TWIN_BYTES")
        res = self.run_main(["style", self.art, b])
        self.assertEqual(len(res.prompts), 1)
        self.assertNotIn("CANARY_TWIN_BYTES", res.prompts[0])
        self.assertIn("twin.pdf", res.prompts[0])
        self.assertIs(self.coverage(res)["input_complete"], False)

    def test_few_undecodable_bytes_keep_the_text_and_record_the_count(self):
        p = self.put("paper.md", fill(5000, "x").encode() + b"\xff\xfe\xff" + b"CANARY_AFTER_BAD")
        res = self.run_main(["style", p])
        self.assertDelivered(res, "CANARY_AFTER_BAD")
        self.assertEqual(self.item(res, "primary").get("replacement_chars"), 3)
        # The reviewer got U+FFFD where the file had something else: not all of its input (round 2).
        self.assertIs(self.item(res, "primary")["complete"], False)
        self.assertEqual(res.code, 2)

    def test_empty_folder_is_invalid(self):
        (self.tmp / "data").mkdir()
        self.assertRefused(self.run_main(["results", self.art, self.tmp / "data"]))

    def test_folder_with_only_junk_is_invalid(self):
        self.put("data/.git/config", "x")
        self.assertRefused(self.run_main(["results", self.art, self.tmp / "data"]))

    def test_empty_primary_is_invalid(self):
        self.assertRefused(self.run_main(["style", self.put("empty.md", "")]))

    def test_whitespace_primary_is_invalid(self):
        self.assertRefused(self.run_main(["style", self.put("blank.md", " \n\n \n")]))

    def test_empty_named_secondary_is_invalid(self):
        self.assertRefused(self.run_main(["style", self.art, self.put("twin.md", "")]))

    def test_order_is_deterministic(self):
        for name in ("c/d/e.txt", "b.txt", "a/z.txt"):
            self.put("data/" + name, "CANARY " + name)
        one = self.run_main(["results", self.art, self.tmp / "data"])
        two = self.run_main(["results", self.art, self.tmp / "data"])
        self.assertDelivered(one, "CANARY a/z.txt", "CANARY b.txt", "CANARY c/d/e.txt")
        self.assertEqual(one.prompts[0], two.prompts[0])
        at = [one.prompts[0].index("--- %s ---" % n) for n in ("a/z.txt", "b.txt", "c/d/e.txt")]
        self.assertEqual(at, sorted(at))

    def _link_dir(self, target, link):
        try:
            os.symlink(str(target), str(link), target_is_directory=True)
            return
        except (OSError, NotImplementedError):
            pass
        if os.name == "nt":     # no symlink privilege: a junction is the same trap on this OS
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
            if r.returncode == 0:
                return
        self.skipTest("cannot make a folder link here")

    def test_link_outside_the_tree_is_not_followed(self):
        self.put("outside/secret.txt", "CANARY_OUTSIDE")
        self.put("data/sub/real.txt", "CANARY_REAL")
        self._link_dir(self.tmp / "outside", self.tmp / "data" / "out_link")
        res = self.run_main(["results", self.art, self.tmp / "data"])
        self.assertDelivered(res, "CANARY_REAL")
        self.assertNotIn("CANARY_OUTSIDE", res.prompts[0])
        item = self.item(res, "secondary")
        self.assertIn("out_link", [s["name"].rstrip("/") for s in item["skipped"]])
        self.assertIs(item["complete"], False)

    def test_file_link_outside_the_tree_is_not_read(self):
        outside = self.put("outside.txt", "CANARY_OUTSIDE_FILE")
        self.put("data/real.txt", "CANARY_REAL_FILE")
        try:
            os.symlink(outside, str(self.tmp / "data" / "link.txt"))
        except (OSError, NotImplementedError):
            self.skipTest("cannot make a file symlink here")
        res = self.run_main(["results", self.art, self.tmp / "data"])
        self.assertDelivered(res, "CANARY_REAL_FILE")
        self.assertNotIn("CANARY_OUTSIDE_FILE", res.prompts[0])

    def test_loop_terminates_and_shows_each_file_once(self):
        self.put("data/sub/real.txt", "CANARY_LOOP_REAL")
        self._link_dir(self.tmp / "data", self.tmp / "data" / "sub" / "back_up")
        res = self.run_main(["results", self.art, self.tmp / "data"])
        self.assertDelivered(res, "CANARY_LOOP_REAL")
        self.assertEqual(res.prompts[0].count("CANARY_LOOP_REAL"), 1)
        self.assertIs(self.item(res, "secondary")["complete"], True)


class Coverage(Base):
    """R3."""

    def test_record_fields_for_a_file_and_a_folder(self):
        a = self.put("a.md", fill(1234, "CANARY_A"))
        self.put("data/x.txt", "x")
        self.put("data/sub/y.txt", "y")
        res = self.run_main(["paper", a, self.tmp / "data"])
        self.assertDelivered(res, "CANARY_A")
        prim, sec = self.item(res, "primary"), self.item(res, "secondary")
        self.assertEqual((prim["kind"], prim["chars"], prim["files_included"]), ("file", 1234, 1))
        self.assertEqual((sec["kind"], sec["files_included"]), ("dir", 2))
        for it in (prim, sec):
            for key in ("path", "kind", "chars", "files_included", "binary_not_shown", "skipped", "complete"):
                self.assertIn(key, it)
            self.assertIs(it["complete"], True)
        self.assertIs(self.coverage(res)["input_complete"], True)
        self.assertEqual(res.record.get("version"), "1.1.0")

    def test_prompt_has_a_coverage_block_and_the_header_a_summary(self):
        a = self.put("a.md", fill(1234, "CANARY_A"))
        res = self.run_main(["style", a])
        self.assertDelivered(res, "CANARY_A")
        self.assertIn("COVERAGE", res.prompts[0])
        self.assertIn("1,234 chars", res.prompts[0])
        self.assertIn("Coverage", res.stdout)
        self.assertLess(res.prompts[0].index("COVERAGE"), res.prompts[0].index("CANARY_A"))

    def test_coverage_block_says_what_was_not_given(self):
        a = self.put("a.md", "primary")
        self.put("data/keep.txt", "keep")
        self.put("data/blob.npz", b"\x00\x00\x00\x00")
        self.put("data/.git/config", "x")
        res = self.run_main(["results", a, self.tmp / "data"])
        block = res.prompts[0].split("You are Project Agni")[0]
        self.assertIn("blob.npz", block)
        self.assertIn("4 bytes", block)
        self.assertIn(".git", block)

    def test_record_on_a_refused_run_carries_coverage(self):
        res = self.run_main(["paper", self.put("a.md", "x"), self.tmp / "nope.json"])
        self.assertRefused(res)
        self.assertIs(self.coverage(res)["input_complete"], False)
        self.assertIs(self.item(res, "secondary")["complete"], False)
        self.assertTrue(res.record.get("invalid_reasons"))


class Verdicts(Base):
    """D6, R6. Replayed 2026-11-07 over 928 stored records: every form below that appears there
    (bare headers, # and ##, bold, OVERALL) parsed the same as before."""

    def verdict(self, review):
        res = self.run_main(["style", self.put("a.md", "x")], reply=review)
        return (res.record or {}).get("verdict"), res.code

    def test_reasoning_prose_containing_each_word_sets_no_verdict(self):
        for prose in ("I would not call this approved.", "Should this be conditional on the seed?",
                      "Nothing here gets rejected out of hand.", "An approved design can still fail.",
                      "APPROVED by whom? Not by me.", "Rejected is too strong a word here."):
            with self.subTest(prose=prose):
                self.assertEqual(self.verdict(prose + "\n\nSome more thought."), ("UNPARSED", 2))

    def test_reasoning_before_the_header_does_not_win(self):
        review = "I would not call this approved yet.\nLet me look.\n\n## VERDICT: REJECTED\n\n1. CRITICAL..."
        self.assertEqual(self.verdict(review), ("REJECTED", 1))

    def test_header_forms(self):
        for word in WORDS:
            forms = [word, "**%s**" % word, "## %s" % word, "VERDICT: %s" % word, "## VERDICT: %s" % word,
                     "# VERDICT: %s" % word, "**VERDICT: %s**" % word, "VERDICT: **%s**" % word,
                     "**VERDICT:** %s" % word, "OVERALL VERDICT: %s" % word, "**OVERALL VERDICT: %s**" % word,
                     "Overall Verdict: %s" % word.lower(), "Verdict: %s" % word.title(), "VERDICT %s" % word,
                     "VERDICT - %s" % word, "VERDICT — %s" % word, "VERDICT: %s." % word,
                     "VERDICT: %s because the seed is unset" % word, "VERDICT: %s — two blockers" % word,
                     "1. VERDICT: %s" % word, "## 1. VERDICT: %s" % word, "  VERDICT:   %s  " % word]
            for form in forms:
                with self.subTest(form=form):
                    self.assertEqual(self.verdict("Thinking it over.\n\n%s\n\nFindings." % form),
                                     (word, EXIT_FOR[word]))

    def test_header_with_the_word_on_the_next_line(self):
        self.assertEqual(self.verdict("## VERDICT\n\n**REJECTED**\n\n1. CRITICAL"), ("REJECTED", 1))

    def test_lines_that_are_not_verdicts(self):
        for line in ("VERDICT: NOT APPROVED", "VERDICT: APPROVED / CONDITIONAL / REJECTED",
                     "VERDICT: APPROVED or REJECTED", "1. VERDICT on its own line: exactly one of "
                     "APPROVED / CONDITIONAL / REJECTED", "REJECTED by the committee",
                     "UNCONDITIONAL", "Verdict: CONDITIONALLY fine", "VERDICT: APPROVED-ish",
                     "~~APPROVED~~", "> VERDICT: APPROVED", "- APPROVED", "PART 1 OF 4 (verdict APPROVED)",
                     "The verdict is APPROVED", "| VERDICT | APPROVED |"):
            with self.subTest(line=line):
                self.assertEqual(self.verdict("Thinking.\n%s\nMore." % line), ("UNPARSED", 2))

    def test_disagreeing_lines_are_unparsed(self):
        for a, b in (("VERDICT: APPROVED", "VERDICT: REJECTED"), ("REJECTED", "OVERALL VERDICT: CONDITIONAL"),
                     ("## VERDICT: CONDITIONAL", "**APPROVED**")):
            with self.subTest(a=a, b=b):
                self.assertEqual(self.verdict("%s\n\nSome findings.\n\n%s" % (a, b)), ("UNPARSED", 2))

    def test_agreeing_lines_give_that_verdict(self):
        self.assertEqual(self.verdict("VERDICT: REJECTED\n\nFindings.\n\n**OVERALL VERDICT: REJECTED**"),
                         ("REJECTED", 1))

    def test_a_reviewer_cannot_claim_the_gates_own_verdicts(self):
        self.assertEqual(self.verdict("VERDICT: INVALID"), ("UNPARSED", 2))


class LocalPath(Base):
    """D7, R7."""
    OLLAMA = {"AGNI_LLM_BACKEND": "ollama"}

    def test_degraded_stamp_is_whole(self):
        p = self.put("a.md", "x")
        res = self.run_main(["style", p, "--allow-local-fallback"], reply="", local="VERDICT: CONDITIONAL")
        self.assertEqual(res.code, 0)
        model = (res.record or {}).get("model", "")
        self.assertIn("DEGRADED", model)
        self.assertIn("do not trust for high-stakes shape claims)", model)

    def test_http_ollama_over_local_budget_is_refused(self):
        p = self.put("a.md", fill(70000, "CANARY_LOCAL"))
        res = self.run_main(["style", p, "--backend", "http"], env=self.OLLAMA, local=APPROVED)
        self.assertRefused(res)
        self.assertIn("AGNI_LOCAL_CHAR_BUDGET", res.stderr)

    def test_http_ollama_exact_local_budget_passes_and_one_over_refuses(self):
        p = self.put("a.md", fill(1000, "CANARY_LOCAL_EXACT"))
        argv = ["style", p, "--backend", "http"]
        probe = self.run_main(argv, env=dict(self.OLLAMA, AGNI_LOCAL_CHAR_BUDGET="100000000"), local=APPROVED)
        self.assertEqual(len(probe.local_prompts), 1)
        size = len(probe.local_prompts[0]) + len(PHASES.SYSTEM)    # the system text reaches Ollama too
        at = self.run_main(argv, env=dict(self.OLLAMA, AGNI_LOCAL_CHAR_BUDGET=str(size)), local=APPROVED)
        self.assertEqual(len(at.local_prompts), 1)
        self.assertIn("CANARY_LOCAL_EXACT", at.local_prompts[0])
        self.assertRefused(self.run_main(argv, env=dict(self.OLLAMA, AGNI_LOCAL_CHAR_BUDGET=str(size - 1)),
                                         local=APPROVED))

    def test_local_fallback_over_budget_never_sends(self):
        p = self.put("a.md", fill(70000, "CANARY_FB"))
        self.stale()
        res = self.run_main(["style", p, "--allow-local-fallback"], reply="", local=APPROVED)
        self.assertEqual(res.local_prompts, [], "a prompt over the local budget went to Ollama")
        self.assertEqual(res.code, 2)
        self.assertEqual((res.record or {}).get("verdict"), "INVALID")
        self.assertIn("NOT a pass", res.stderr)
        self.assertIn("AGNI_LOCAL_CHAR_BUDGET", res.stderr)

    def test_http_anthropic_is_not_held_to_the_local_budget(self):
        p = self.put("a.md", fill(70000, "CANARY_HTTP"))
        res = self.run_main(["style", p, "--backend", "http"], env={"AGNI_LLM_BACKEND": "anthropic"},
                            local=APPROVED)
        self.assertEqual(len(res.local_prompts), 1)
        self.assertIn("CANARY_HTTP", res.local_prompts[0])

    def test_ollama_call_warns_when_llm_is_not_the_repo_copy(self):
        # On a machine with agni-open, its llm.py comes first on sys.path and may not set num_ctx.
        p = self.put("a.md", "x")
        with mock.patch.object(PHASES.llm, "__file__", os.path.join("elsewhere", "agni-open", "llm.py")):
            res = self.run_main(["style", p, "--backend", "http"], env=self.OLLAMA, local=APPROVED)
        self.assertEqual(len(res.local_prompts), 1)
        self.assertIn("num_ctx", res.stderr)

    def test_repo_llm_sets_num_ctx(self):
        mod = sut("llm.py", "llm_under_test")
        for env, want in (({}, 32768), ({"LLM_NUM_CTX": "8192"}, 8192)):
            seen = {}

            def fake_post(url, json=None, timeout=None):
                seen["body"] = json
                return SimpleNamespace(status_code=200, text="", json=lambda: {"response": "ok"})

            clean = {k: v for k, v in os.environ.items() if k != "LLM_NUM_CTX"}
            clean.update(env)
            with self.subTest(env=env), mock.patch.dict(os.environ, clean, clear=True), \
                    mock.patch.object(mod.requests, "post", fake_post):
                self.assertEqual(mod.infer("p", backend="ollama", model="m", url="http://ollama.invalid"), "ok")
                self.assertEqual(seen["body"]["options"].get("num_ctx"), want)


class Timeouts(Base):
    """D10, R10."""

    def setUp(self):
        super().setUp()
        self.p = self.put("a.md", "x")
        self.timeout = subprocess.TimeoutExpired(cmd="claude", timeout=1)

    def test_timeout_is_retried_at_most_once_then_fails_loud(self):
        self.stale()
        res = self.run_main(["style", self.p], reply=self.timeout)
        self.assertEqual(len(res.prompts), 2, "calls made on repeated timeouts")
        self.assertEqual(res.code, 2)
        self.assertEqual((res.record or {}).get("verdict"), "INVALID")
        self.assertIn("NOT a pass", res.stderr)

    def test_timeout_then_success(self):
        res = self.run_main(["style", self.p], reply=[self.timeout, APPROVED])
        self.assertEqual(len(res.prompts), 2)
        self.assertEqual((res.code, (res.record or {}).get("verdict")), (0, "APPROVED"))

    def test_other_errors_keep_all_six_attempts(self):
        res = self.run_main(["style", self.p], reply=RuntimeError("busy"))
        self.assertEqual(len(res.prompts), 6)
        self.assertEqual(res.code, 2)

    def test_default_timeout_is_1800_and_the_env_still_wins(self):
        self.assertEqual(self.run_main(["style", self.p]).timeouts, [1800])
        self.assertEqual(self.run_main(["style", self.p], env={"AGNI_CLI_TIMEOUT": "77"}).timeouts, [77])


class DesignHarness(Base):
    """D8, R8: agni_design_review.py."""

    @classmethod
    def setUpClass(cls):
        cls.mod = sut("agni_design_review.py", "agni_design_under_test")

    def review(self, script, hypotheses=None, threats=None, env=None):
        pr = self.prereg("prereg.json", {"experiment": "exp1", "hypotheses": hypotheses or ["h"],
                                         "threats_to_validity": threats or ["t"]})
        res = SimpleNamespace(calls=[], code=None, stderr="", verdict=None)

        def fake_run(cmd, **kw):
            payload = kw.get("input") or cmd[cmd.index("-d") + 1]
            res.calls.append(json.loads(payload))
            return SimpleNamespace(returncode=0, stderr="",
                                   stdout=json.dumps({"response": "OVERALL VERDICT: APPROVED"}))

        clean = {k: v for k, v in os.environ.items() if not k.startswith(("AGNI_", "LLM_"))}
        clean.update(env or {})
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, clean, clear=True), \
                mock.patch.object(self.mod.subprocess, "run", fake_run), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                res.verdict = self.mod.run_review(script, pr)
            except SystemExit as e:
                res.code = e.code
        res.stderr = stderr.getvalue()
        rec = self.tmp / "exp1_agni_review.json"
        res.record = json.loads(rec.read_text()) if rec.is_file() else None
        return res

    def test_no_excerpt_and_no_slices_under_the_budget(self):
        script = self.put("run.py", fill(35000, "CANARY_SCRIPT_END"))
        res = self.review(script, hypotheses=[fill(3000, "CANARY_DHYP_END")],
                          threats=[fill(2500, "CANARY_DTHREAT_END")])
        self.assertEqual(len(res.calls), 1)
        for c in ("CANARY_SCRIPT_END", "CANARY_DHYP_END", "CANARY_DTHREAT_END"):
            self.assertIn(c, res.calls[0]["prompt"], "canary %s absent" % c)
        self.assertNotIn("truncated", res.calls[0]["prompt"])

    def test_over_budget_is_invalid_with_no_call_and_a_nonzero_exit(self):
        script = self.put("run.py", fill(70000, "CANARY_DOVER"))
        self.stale(self.tmp / "exp1_agni_review.json")
        res = self.review(script)
        self.assertEqual(res.calls, [], "over budget must not call the model")
        self.assertNotIn(res.code, (None, 0), "exit code")
        self.assertEqual((res.record or {}).get("verdict"), "INVALID")
        self.assertIn("NOT a pass", res.stderr)

    def test_budget_is_configurable(self):
        script = self.put("run.py", fill(5000, "CANARY_DCONF"))
        res = self.review(script, env={"AGNI_LOCAL_CHAR_BUDGET": "1000"})
        self.assertEqual(res.calls, [])
        self.assertEqual((res.record or {}).get("verdict"), "INVALID")

    def test_sets_num_ctx_so_ollama_does_not_trim_what_it_was_sent(self):
        res = self.review(self.put("run.py", "x = 1"))
        self.assertEqual(res.calls[0]["options"].get("num_ctx"), 32768)
        res = self.review(self.put("run.py", "x = 1"), env={"LLM_NUM_CTX": "8192"})
        self.assertEqual(res.calls[0]["options"].get("num_ctx"), 8192)


class Survey(Base):
    """D9, R9: run_style_survey.py and summarize_style.py."""

    def survey_one(self, record, exit_code=0):
        mod = sut("run_style_survey.py", "survey_under_test")
        self.put("papers/p1/main.tex", "flight")
        self.put("papers/p1/academic/main.tex", "twin")
        outdir = self.tmp / "style"
        outdir.mkdir(exist_ok=True)

        def fake_run(cmd, **kw):
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(record), encoding="utf-8")
            return SimpleNamespace(returncode=exit_code, stderr="", stdout="")

        with mock.patch.object(mod, "PR", self.tmp / "papers"), mock.patch.object(mod, "OUT", outdir), \
                mock.patch.object(mod.subprocess, "run", fake_run):
            return mod.run_one("p1")

    def test_invalid_counts_as_a_failure_even_with_review_text(self):
        # exit 0 on purpose: the verdict alone must fail it, not the exit-code check beside it.
        rec = self.survey_one({"verdict": "INVALID", "review": "some text"}, exit_code=0)
        self.assertEqual(rec["status"], "FAIL")
        self.assertIn("INVALID", rec["reason"])

    def test_exit_2_beside_a_stale_approved_is_a_failure(self):
        # The refused run could not write its record; the old pass is still on disk.
        rec = self.survey_one({"verdict": "APPROVED", "review": "an old pass"}, exit_code=2)
        self.assertEqual(rec["status"], "FAIL", "exit 2 means UNGATED whatever the file says")

    def test_real_verdicts_still_ok(self):
        whole = {"inputs": [], "input_complete": True}
        self.assertEqual(self.survey_one({"verdict": "REJECTED", "review": "text", "input_coverage": whole},
                                         exit_code=1)["status"], "OK")
        self.assertEqual(self.survey_one({"verdict": "APPROVED", "review": "text", "input_coverage": whole})
                         ["status"], "OK")

    def test_summary_renders_invalid_as_not_reviewed(self):
        mod = sut("summarize_style.py", "summarize_under_test")
        outdir = self.tmp / "style"
        outdir.mkdir()
        (outdir / "p1.agni_style.json").write_text(json.dumps({"verdict": "INVALID", "review": ""}),
                                                   encoding="utf-8")
        stdout = io.StringIO()
        with mock.patch.object(mod, "OUT", outdir), mock.patch.object(mod, "PAPERS", ["p1"]), \
                contextlib.redirect_stdout(stdout):
            mod.main()
        row = [l for l in stdout.getvalue().splitlines() if l.startswith("p1")][0]
        self.assertIn("INVALID", row)
        self.assertNotIn("EMPTY REVIEW", row)

    # Round 2: neither the survey nor the summary may show a pass the gate did not give.

    def summary(self, outdir, stream=None):
        mod = sut("summarize_style.py", "summarize_under_test")
        stream = stream if stream is not None else io.StringIO()
        with mock.patch.object(mod, "OUT", outdir), mock.patch.object(mod, "PAPERS", ["p1"]), \
                contextlib.redirect_stdout(stream):
            mod.main()
        return stream

    def summary_row(self, outdir):
        return [l for l in self.summary(outdir).getvalue().splitlines() if l.startswith("p1")][0]

    def cp1252_console(self):
        buf = io.BytesIO()
        return buf, io.TextIOWrapper(buf, encoding="cp1252", errors="strict")

    def test_survey_fails_a_pass_on_incomplete_input(self):
        for code in (0, 2):
            with self.subTest(exit_code=code):
                rec = self.survey_one({"verdict": "APPROVED", "review": "text",
                                       "input_coverage": {"inputs": [], "input_complete": False}}, exit_code=code)
                self.assertEqual(rec["status"], "FAIL")
                self.assertIn("not complete", rec["reason"])
                self.assertIs(rec.get("input_complete"), False, "the run log must carry coverage")

    def test_survey_fails_a_record_with_no_coverage(self):
        rec = self.survey_one({"verdict": "APPROVED", "review": "text"}, exit_code=0)
        self.assertEqual(rec["status"], "FAIL")

    def test_summary_shows_a_pass_on_incomplete_input_as_not_a_pass(self):
        outdir = self.tmp / "style"
        outdir.mkdir()
        for cov, needle in (({"inputs": [], "input_complete": False}, "NOT COMPLETE"),
                            (None, "COVERAGE NOT RECORDED")):
            with self.subTest(needle=needle):
                rec = {"verdict": "APPROVED", "review": "VERDICT: APPROVED\n\n## 1. MAJOR - a finding\n"}
                if cov:
                    rec["input_coverage"] = cov
                (outdir / "p1.agni_style.json").write_text(json.dumps(rec), encoding="utf-8")
                self.assertIn(needle, self.summary_row(outdir))

    def run_survey(self, outdir, exit_code):
        mod = sut("run_style_survey.py", "survey_under_test")
        self.put("papers/p1/main.tex", "flight")
        self.put("papers/p1/academic/main.tex", "twin")

        def fake_run(cmd, **kw):    # writes nothing: the record could not be replaced
            return SimpleNamespace(returncode=exit_code, stderr="could not write the record", stdout="")

        with mock.patch.object(mod, "PR", self.tmp / "papers"), mock.patch.object(mod, "OUT", outdir), \
                mock.patch.object(mod, "LOG", outdir / "_run_log.json"), \
                mock.patch.object(mod.subprocess, "run", fake_run), \
                mock.patch.object(sys, "argv", ["run_style_survey.py", "p1"]), \
                contextlib.redirect_stdout(io.StringIO()):
            mod.main()

    def test_summary_does_not_show_a_stale_approved_the_survey_failed(self):
        outdir = self.tmp / "style"
        outdir.mkdir()
        (outdir / "p1.agni_style.json").write_text(json.dumps(
            {"verdict": "APPROVED", "review": "VERDICT: APPROVED\n\n## 1. CRITICAL - CANARY_OLD_FINDING\n",
             "timestamp": "2026-01-01T00:00:00+00:00",
             "input_coverage": {"inputs": [], "input_complete": True}}), encoding="utf-8")
        self.run_survey(outdir, exit_code=2)
        out = self.summary(outdir).getvalue()
        row = [l for l in out.splitlines() if l.startswith("p1")][0]
        self.assertIn("LAST RUN FAILED", row)
        self.assertEqual(row.split()[-4:], ["0", "0", "0", "0"], "a stale review's findings are not current")
        self.assertNotIn("CANARY_OLD_FINDING", out)

    def test_summary_trusts_a_record_written_after_the_failed_run(self):
        # The workers race on the log: a FAIL entry about an older record must not hide a newer one.
        outdir = self.tmp / "style"
        outdir.mkdir()
        (outdir / "_run_log.json").write_text(json.dumps([{"paper": "p1", "status": "FAIL", "verdict": "APPROVED",
                                                           "record_timestamp": "2026-01-01T00:00:00+00:00"}]),
                                              encoding="utf-8")
        (outdir / "p1.agni_style.json").write_text(json.dumps(
            {"verdict": "APPROVED", "review": "VERDICT: APPROVED\n", "timestamp": "2026-02-01T00:00:00+00:00",
             "input_coverage": {"inputs": [], "input_complete": True}}), encoding="utf-8")
        row = self.summary_row(outdir)
        self.assertNotIn("FAILED", row)
        self.assertIn("APPROVED", row)

    def test_summary_survives_a_cp1252_console(self):
        outdir = self.tmp / "style"
        outdir.mkdir()
        (outdir / "p1.agni_style.json").write_text(json.dumps(
            {"verdict": "REJECTED", "input_coverage": {"inputs": [], "input_complete": True},
             "review": "VERDICT: REJECTED\n\n## 1. CRITICAL — the claim → CANARY_ARROW_HEADING\n"}),
            encoding="utf-8")
        buf, console = self.cp1252_console()
        self.summary(outdir, console)
        console.flush()
        self.assertIn("CANARY_ARROW_HEADING", buf.getvalue().decode("utf-8", "replace"))

    def test_survey_summary_survives_a_cp1252_console(self):
        mod = sut("run_style_survey.py", "survey_under_test")
        outdir = self.tmp / "style"
        outdir.mkdir()
        rows = {"p1": {"paper": "p1", "status": "FAIL", "verdict": "INVALID",
                       "reason": "verdict=INVALID chars=0: pr—→/p1/academic/main.tex is empty"},
                "p2": {"paper": "p2", "status": "FAIL", "verdict": "APPROVED", "reason": "CANARY_SECOND_ROW"}}
        buf, console = self.cp1252_console()
        with mock.patch.object(mod, "OUT", outdir), mock.patch.object(mod, "LOG", outdir / "_run_log.json"), \
                mock.patch.object(mod, "run_one", lambda name: dict(rows[name])), \
                mock.patch.object(sys, "argv", ["run_style_survey.py", "p1", "p2"]), \
                contextlib.redirect_stdout(console):
            mod.main()
        console.flush()
        self.assertIn("CANARY_SECOND_ROW", buf.getvalue().decode("utf-8", "replace"))


class IncompleteIsNotAPass(Base):
    """Round 2: input_complete false used to stop at the record. Exit 0 with a stub in place of a
    twin read as binary was possible; now a pass on incomplete input exits 2 unless accepted."""

    def setUp(self):
        super().setUp()
        self.art = self.put("artifact.md", "CANARY_ART")

    def test_a_pass_on_incomplete_input_exits_2_and_a_rejection_still_exits_1(self):
        self.put("data/keep.txt", "keep")
        self.put("data/blob.npz", b"\x00\x01")
        for reply, code in ((APPROVED, 2), ("VERDICT: CONDITIONAL", 2), ("VERDICT: REJECTED", 1)):
            with self.subTest(reply=reply):
                res = self.run_main(["results", self.art, self.tmp / "data"], reply=reply)
                self.assertEqual(res.code, code, "crash=%r" % res.crash)
                self.assertIs(self.coverage(res)["input_complete"], False)
                if code == 2:
                    self.assertIn("NOT a pass", res.stderr)

    def test_text_twin_read_as_binary_is_not_a_pass(self):
        twin = self.put("academic/main.tex", ("café résumé naïve " * 300 +
                                              "CANARY_TWIN").encode("cp1252"))
        res = self.run_main(["style", self.art, twin])
        self.assertNotIn("CANARY_TWIN", res.prompts[0])
        self.assertEqual(res.code, 2)

    def test_utf16_file_in_a_primary_folder_is_not_a_pass(self):
        self.put("code/a.md", "CANARY_A")
        self.put("code/helper.py", "def helper():\n    return 1  # CANARY_U16\n".encode("utf-16"))
        self.assertEqual(self.run_main(["results", self.tmp / "code"]).code, 2)

    def test_a_prereg_with_other_keys_is_whole_and_can_pass(self):
        # Round 2 named these keys and did not send them, so this exited 2; round 3 sends them whole.
        pr = self.prereg("prereg.json", {"hypotheses": ["h"], "threats_to_validity": ["t"],
                                         "falsification_criteria": ["CANARY_FALS"]})
        res = self.run_main(["design", self.put("run.py", "x = 1"), pr])
        self.assertDelivered(res, "CANARY_FALS")
        self.assertEqual(res.code, 0)

    def test_accepting_incomplete_input_passes_and_is_stamped(self):
        self.put("data/keep.txt", "keep")
        self.put("data/blob.npz", b"\x00\x01")
        res = self.run_main(["results", self.art, self.tmp / "data", "--accept-incomplete-input"])
        self.assertEqual(res.code, 0, "crash=%r" % res.crash)
        self.assertIs(res.record.get("incomplete_input_accepted"), True)
        self.assertIs(self.coverage(res)["input_complete"], False)

    def test_duplicate_prereg_keys_are_refused(self):
        for raw in ('{"hypotheses": ["CANARY_DUP_FIRST"], "threats_to_validity": ["t"], "hypotheses": ["h"]}',
                    '{"hypotheses": ["h"], "threats_to_validity": ["CANARY_DUP_T"], "threats_to_validity": ["t"]}',
                    '{"hypotheses": [{"id": "CANARY_DUP_NESTED", "id": "h1"}], "threats_to_validity": ["t"]}'):
            with self.subTest(raw=raw):
                res = self.run_main(["design", self.put("run.py", "x = 1"), self.put("prereg.json", raw)])
                self.assertRefused(res)
                self.assertIn("duplicate", res.stderr)

    def test_literal_replacement_char_is_not_counted_as_undecodable(self):
        res = self.run_main(["style", self.put("paper.md", "a � b � CANARY_LITERAL")])
        self.assertDelivered(res, "CANARY_LITERAL")
        self.assertEqual(self.item(res, "primary")["replacement_chars"], 0)
        self.assertIs(self.item(res, "primary")["complete"], True)
        self.assertEqual(res.code, 0)

    def test_unread_bytes_count_against_the_budget(self):
        # Over 4x the budget, so measured and never read. Leaving those bytes out of the check let the
        # reviewer approve a prompt that did not hold the file (mutant U01).
        p = self.put("far.md", fill(100000, "CANARY_FAR"))
        res = self.run_main(["style", p], env={"AGNI_PROMPT_CHAR_BUDGET": "20000"})
        self.assertRefused(res)
        self.assertIn("AGNI_PROMPT_CHAR_BUDGET is 20,000", res.stderr)

    def test_a_refused_run_never_reads_as_whole(self):
        # Read, then refused over budget: header and refusal must both say NOT COMPLETE (mutant U16).
        p = self.put("mid.md", fill(30500, "CANARY_MID"))
        res = self.run_main(["style", p], env={"AGNI_PROMPT_CHAR_BUDGET": "20000"})
        self.assertRefused(res)
        self.assertIn("Input whole : NO", res.stdout)
        self.assertNotIn("Input whole : yes", res.stdout)
        cov = [l for l in res.stdout.splitlines() if "Coverage" in l]
        self.assertTrue(cov and all(l.endswith("NOT COMPLETE") for l in cov), cov)
        self.assertTrue(all("NOT COMPLETE" in l for l in res.stderr.splitlines() if "primary:" in l))

    def test_cli_failure_says_why_even_when_stderr_is_empty(self):
        fail = SimpleNamespace(returncode=1, stderr="", stdout="API Error: CANARY_CLI_REFUSAL")
        with mock.patch.object(PHASES, "CLAUDE_BIN", "claude-under-test"), \
                mock.patch.object(PHASES.subprocess, "run", lambda cmd, **kw: fail):
            with self.assertRaises(RuntimeError) as cm:
                PHASES.infer_cli("prompt", PHASES.SYSTEM, 10)
        self.assertIn("CANARY_CLI_REFUSAL", str(cm.exception))


class VerdictsInCode(Base):
    """Round 2: a verdict quoted inside code (a fixture, or output of the script under review) is
    not the reviewer's verdict."""

    def verdict(self, review):
        res = self.run_main(["style", self.put("a.md", "x")], reply=review)
        return (res.record or {}).get("verdict"), res.code

    def test_a_verdict_inside_code_is_not_the_reviewers(self):
        for review, want in (
                # Round 3: a quoted verdict that disagrees vetoes. It was REJECTED in round 2.
                ("## Final verdict: REJECTED\n1. CRITICAL - the gate prints a pass on failure:\n```\n"
                 "VERDICT: APPROVED\n```\n", ("UNPARSED", 2)),
                ("Thinking.\n```\nVERDICT: APPROVED\n```\nMore.", ("UNPARSED", 2)),
                ("Thinking.\n~~~\nVERDICT: APPROVED\n~~~\nMore.", ("UNPARSED", 2)),
                ("Thinking.\n    VERDICT: APPROVED\nMore.", ("UNPARSED", 2)),
                ("Thinking.\n\tVERDICT: APPROVED\nMore.", ("UNPARSED", 2)),
                ("My verdict: REJECTED\n\n    VERDICT: APPROVED\n", ("UNPARSED", 2)),
                ("```python\nprint('x')\n```\n\nVERDICT: CONDITIONAL\n", ("CONDITIONAL", 0))):
            with self.subTest(review=review):
                self.assertEqual(self.verdict(review), want)

    def test_final_verdict_label(self):
        for word in WORDS:
            for form in ("FINAL VERDICT: %s" % word, "## Final verdict: %s" % word):
                with self.subTest(form=form):
                    self.assertEqual(self.verdict("Thinking.\n\n%s\n" % form), (word, EXIT_FOR[word]))


# Round 3. Round 2 toggled "in code" on any line starting with three backticks or tildes, so a
# one-line span or an unbalanced fence hid the reviewer's REJECTED and let a quoted APPROVED decide,
# exit 0. Fences now follow CommonMark, only lines outside code decide, a verdict line in code can
# only veto, and a fence left open is UNPARSED. (review, verdict) pairs, shared by both harnesses.
FENCE_CASES = (
    ("inline span of three backticks is not a fence",
     "Thinking.\n```x``` is the fixture.\nVERDICT: REJECTED\n```\nVERDICT: APPROVED\n```\n", "UNPARSED"),
    ("inline span alone does not hide the verdict after it",
     "Thinking.\n```x``` is the fixture.\n\nVERDICT: REJECTED\n", "REJECTED"),
    ("unbalanced fence", "VERDICT: APPROVED\n\nFindings.\n```\nVERDICT: REJECTED\n", "UNPARSED"),
    ("unbalanced fence with no verdict in it", "VERDICT: APPROVED\n\n```python\nprint(1)\n", "UNPARSED"),
    ("fence inside a list",
     "VERDICT: REJECTED\n\n1. CRITICAL - the fixture prints a pass:\n\n   ```\n   VERDICT: APPROVED\n   ```\n",
     "UNPARSED"),
    ("fence indented four inside a list item, verdict only inside it",
     "Thinking.\n\n- the fixture:\n\n    ```\n  VERDICT: APPROVED\n    ```\n\nMore.\n", "UNPARSED"),
    ("tilde fence is not closed by backticks",
     "Thinking.\n~~~\nfixture\n```\nVERDICT: APPROVED\n~~~\nMore.\n", "UNPARSED"),
    ("shorter fence does not close a longer one",
     "Thinking.\n````\n```\nVERDICT: APPROVED\n````\nMore.\n", "UNPARSED"),
    ("longer closing fence closes", "Thinking.\n```\nprint(1)\n`````\n\nVERDICT: REJECTED\n", "REJECTED"),
    ("fence with text after the backticks does not close",
     "Thinking.\n```\nVERDICT: APPROVED\n``` not a close\n```\n\nVERDICT: APPROVED\n", "APPROVED"),
    ("real REJECTED outside, quoted APPROVED inside",
     "OVERALL VERDICT: REJECTED\n\n1. MAJOR - the fixture:\n```\nOVERALL VERDICT: APPROVED\n```\n", "UNPARSED"),
    ("quoted REJECTED inside, APPROVED outside",
     "```\nVERDICT: REJECTED\n```\n\nOVERALL VERDICT: APPROVED\n", "UNPARSED"),
    ("indented REJECTED vetoes", "OVERALL VERDICT: APPROVED\n\n    VERDICT: REJECTED\n", "UNPARSED"),
    ("agreeing code and outside give that verdict",
     "VERDICT: REJECTED\n\n```\nVERDICT: REJECTED\n```\n\nOVERALL VERDICT: REJECTED\n", "REJECTED"),
    ("verdict only inside code", "Thinking.\n~~~\nOVERALL VERDICT: APPROVED\n~~~\n", "UNPARSED"),
    ("code that holds no verdict changes nothing",
     "```python\nprint('x')\n```\n\nOVERALL VERDICT: CONDITIONAL\n", "CONDITIONAL"),
)


class FencesRound3(Base):
    """Round 3: the toggle bypass, in agni_phases.py. FENCE_CASES runs on the design harness too."""

    def test_fence_cases(self):
        for name, review, want in FENCE_CASES:
            with self.subTest(name):
                res = self.run_main(["style", self.put("a.md", "x")], reply=review)
                self.assertEqual(((res.record or {}).get("verdict"), res.code),
                                 (want, EXIT_FOR.get(want, 2)), "crash=%r" % res.crash)


class DesignInputs(Base):
    """Round 2: agni_design_review.py sends every prereg key, reads no prose as a verdict, and
    leaves no old APPROVED behind an empty reply."""

    @classmethod
    def setUpClass(cls):
        cls.mod = sut("agni_design_review.py", "agni_design_under_test")

    def run_design(self, prereg, response="OVERALL VERDICT: APPROVED"):
        pr = self.prereg("prereg.json", prereg)
        script = self.put("run.py", "x = 1")
        res = SimpleNamespace(calls=[], code=None, verdict=None, stderr="")

        def fake_run(cmd, **kw):
            res.calls.append(json.loads(kw.get("input")))
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"response": response}))

        clean = {k: v for k, v in os.environ.items() if not k.startswith(("AGNI_", "LLM_"))}
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, clean, clear=True), \
                mock.patch.object(self.mod.subprocess, "run", fake_run), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            try:
                res.verdict = self.mod.run_review(script, pr)
            except SystemExit as e:
                res.code = e.code
        res.stderr = stderr.getvalue()
        rec = self.tmp / "v1_agni_review.json"
        res.record = json.loads(rec.read_text()) if rec.is_file() else None
        return res

    def test_every_prereg_key_reaches_the_model(self):
        for prereg, canary in (({"experiment": "v1", "hypothesis": ["CANARY_V_SING"], "threats_to_validity": ["t"]},
                                "CANARY_V_SING"),
                               ({"experiment": "v1", "hypotheses": ["h"], "threats_to_validity": ["t"],
                                 "falsification_criteria": ["CANARY_V_FALS"]}, "CANARY_V_FALS")):
            with self.subTest(canary=canary):
                res = self.run_design(prereg)
                self.assertEqual(len(res.calls), 1)
                self.assertIn(canary, res.calls[0]["prompt"])

    def test_prose_and_echoes_are_not_verdicts(self):
        for response, want in (("Thinking... I would not call this design approved.", "UNPARSED"),
                               ('4. End with exactly one line: "OVERALL VERDICT: APPROVED" or "...: REJECTED"',
                                "UNPARSED"),
                               ("```\nOVERALL VERDICT: APPROVED\n```", "UNPARSED"),
                               ("Findings.\n\nOVERALL VERDICT: REJECTED", "REJECTED"),
                               ("Findings.\n\n**OVERALL VERDICT: CONDITIONAL**", "CONDITIONAL")):
            with self.subTest(response=response):
                res = self.run_design({"experiment": "v1", "hypotheses": ["h"], "threats_to_validity": ["t"]},
                                      response)
                self.assertEqual(res.verdict, want)

    def test_empty_reply_replaces_a_stale_approved(self):
        self.stale(self.tmp / "v1_agni_review.json")
        res = self.run_design({"experiment": "v1", "hypotheses": ["h"], "threats_to_validity": ["t"]}, "")
        self.assertEqual(res.code, 2)
        self.assertEqual((res.record or {}).get("verdict"), "INVALID")
        # X11: the empty-reply write itself, not the placeholder written before the call.
        self.assertIn("empty review", (res.record or {}).get("invalid_reason", ""))

    # ---- round 3

    def design(self, raw, run=None, record="v1", script_text="x = 1"):
        """Run the harness on prereg text written as given (bytes or str), with subprocess.run
        replaced by run, or by a model that answers APPROVED. A crash is kept, not raised: a
        traceback leaves whatever record was there."""
        pr = self.put("prereg.json", raw)
        script = self.put("run.py", script_text)
        res = SimpleNamespace(calls=[], code=None, verdict=None, stderr="", crash=None)

        def approve(cmd, **kw):
            res.calls.append(json.loads(kw.get("input")))
            return SimpleNamespace(returncode=0, stderr="",
                                   stdout=json.dumps({"response": "OVERALL VERDICT: APPROVED"}))

        clean = {k: v for k, v in os.environ.items() if not k.startswith(("AGNI_", "LLM_"))}
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, clean, clear=True), \
                mock.patch.object(self.mod.subprocess, "run", run or approve), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            try:
                res.verdict = self.mod.run_review(script, pr)
            except SystemExit as e:
                res.code = e.code
            except Exception as e:
                res.crash, res.code = e, 1
        res.stderr = stderr.getvalue()
        rec = self.tmp / ("%s_agni_review.json" % record)
        res.record = json.loads(rec.read_text(encoding="utf-8")) if rec.is_file() else None
        return res

    def assertInvalidOverStale(self, res, record_name="v1"):
        self.assertIsNone(res.crash, "a traceback leaves the old record standing: %r" % res.crash)
        self.assertNotIn(res.code, (None, 0), "exit code")
        self.assertEqual((res.record or {}).get("verdict"), "INVALID",
                         "%s_agni_review.json still says %s" % (record_name, (res.record or {}).get("verdict")))
        self.assertTrue((res.record or {}).get("invalid_reason"))
        self.assertIn("NOT a pass", res.stderr)

    def test_fence_cases(self):
        for name, review, want in FENCE_CASES:
            with self.subTest(name):
                res = self.run_design({"experiment": "v1", "hypotheses": ["h"], "threats_to_validity": ["t"]},
                                      review)
                self.assertEqual(res.verdict, want)

    def test_disagreeing_verdict_lines_are_unparsed(self):
        # X07a
        for review in ("OVERALL VERDICT: APPROVED\n\nFindings.\n\nOVERALL VERDICT: REJECTED\n",
                       "VERDICT: REJECTED\n\nOVERALL VERDICT: CONDITIONAL\n"):
            with self.subTest(review=review):
                res = self.run_design({"experiment": "v1", "hypotheses": ["h"], "threats_to_validity": ["t"]},
                                      review)
                self.assertEqual(res.verdict, "UNPARSED")

    def test_a_bad_prereg_writes_invalid_over_the_old_record(self):
        good = '{"experiment": "v1", "hypotheses": ["h"], "threats_to_validity": ["t"]'
        for name, raw, record in (
                ("not JSON", good + ', "aim": ', "v1"),
                ("duplicate key", good + ', "hypotheses": ["h2"]}', "v1"),
                ("not an object", '["v1"]', "run"),
                ("not UTF-8", (good + ', "aim": "caf\xe9"}').encode("cp1252"), "v1")):
            with self.subTest(name):
                self.stale(self.tmp / ("%s_agni_review.json" % record))
                res = self.design(raw, record=record)
                self.assertEqual(res.calls, [], "no model call on a bad prereg")
                self.assertInvalidOverStale(res, record)

    def test_a_missing_or_unreadable_prereg_writes_invalid(self):
        self.stale(self.tmp / "run_agni_review.json")
        script = self.put("run.py", "x = 1")
        for name, prereg in (("missing", self.tmp / "nope.json"), ("a folder", self.tmp)):
            with self.subTest(name):
                stderr = io.StringIO()
                res = SimpleNamespace(code=None, crash=None)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                    try:
                        self.mod.run_review(script, str(prereg))
                    except SystemExit as e:
                        res.code = e.code
                    except Exception as e:
                        res.crash, res.code = e, 1
                res.stderr = stderr.getvalue()
                rec = self.tmp / "run_agni_review.json"
                res.record = json.loads(rec.read_text(encoding="utf-8")) if rec.is_file() else None
                self.assertInvalidOverStale(res, "run")

    def test_a_script_that_is_not_utf8_writes_invalid(self):
        self.stale(self.tmp / "v1_agni_review.json")
        res = self.design('{"experiment": "v1", "hypotheses": ["h"], "threats_to_validity": ["t"]}',
                          script_text="# caf\xe9 CANARY\n".encode("cp1252"))
        self.assertEqual(res.calls, [])
        self.assertInvalidOverStale(res)

    def test_utf8_text_reaches_the_model_as_written(self):
        # Platform encoding read this as cp1252 on Windows: the arrow arrived as three other glyphs.
        res = self.design('{"experiment": "v1", "hypotheses": ["h → CANARY_ARROW_H"], '
                          '"threats_to_validity": ["t"]}'.encode("utf-8"),
                          script_text="# → CANARY_ARROW_S\n".encode("utf-8"))
        self.assertEqual(len(res.calls), 1, "crash=%r" % res.crash)
        # The prereg goes through json.dumps, which writes the arrow as an escape; the script is raw.
        for canary in (json.dumps("→ CANARY_ARROW_H")[1:-1], "→ CANARY_ARROW_S"):
            self.assertIn(canary, res.calls[0]["prompt"])

    def test_curl_or_backend_failure_writes_invalid(self):
        def curl_fails(cmd, **kw):
            return SimpleNamespace(returncode=7, stdout="", stderr="CANARY_CURL: connection refused")

        def not_json(cmd, **kw):
            return SimpleNamespace(returncode=0, stdout="<html>bad gateway</html>", stderr="")

        def not_an_object(cmd, **kw):
            return SimpleNamespace(returncode=0, stdout='["OVERALL VERDICT: APPROVED"]', stderr="")

        def times_out(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 600)

        def no_curl(cmd, **kw):
            raise FileNotFoundError("curl")

        prereg = '{"experiment": "v1", "hypotheses": ["h"], "threats_to_validity": ["t"]}'
        for run in (curl_fails, not_json, not_an_object, times_out, no_curl):
            with self.subTest(run.__name__):
                self.stale(self.tmp / "v1_agni_review.json")
                res = self.design(prereg, run=run)
                self.assertInvalidOverStale(res)
                self.assertNotIn("still running", res.record["invalid_reason"],
                                 "the failure's own write, not the placeholder")
        res = self.design(prereg, run=curl_fails)
        self.assertIn("CANARY_CURL", res.record["invalid_reason"])


class PinnedMutants(Base):
    """Round 3: each test kills a mutant that survived the round-2 suite."""

    def test_unread_bytes_in_a_folder_count_against_the_budget(self):
        # X12. The text file sorts first, so it is read and the folder is not empty; the big one is
        # over 4x the budget, so measured and not read.
        art = self.put("art.md", "the artifact")
        self.put("data/a_keep.txt", "CANARY_KEEP")
        self.put("data/z_big.txt", fill(100000, "CANARY_BIG"))
        res = self.run_main(["results", art, self.tmp / "data"], env={"AGNI_PROMPT_CHAR_BUDGET": "20000"})
        self.assertRefused(res)
        self.assertIn("AGNI_PROMPT_CHAR_BUDGET is 20,000", res.stderr)

    def test_undecodable_bytes_in_a_folder_make_it_not_complete(self):
        # X09b
        art = self.put("art.md", "the artifact")
        self.put("data/a.txt", fill(5000, "x").encode() + b"\xff\xfe\xff" + b"CANARY_DIR_BAD")
        res = self.run_main(["results", art, self.tmp / "data"])
        self.assertDelivered(res, "CANARY_DIR_BAD")
        item = self.item(res, "secondary")
        self.assertEqual(item.get("replacement_chars"), 3)
        self.assertIs(item["complete"], False)
        self.assertIs(self.coverage(res)["input_complete"], False)
        self.assertEqual(res.code, 2)

    def test_no_input_line_says_complete_on_a_budget_refusal(self):
        # X13. Every input, not only the first, in the header and in the refusal.
        a = self.put("a.md", fill(9000, "CANARY_A"))
        b = self.put("b.md", fill(9000, "CANARY_B"))
        c = self.put("c.py", fill(9000, "CANARY_C"))
        res = self.run_main(["paper", a, b, c], env={"AGNI_PROMPT_CHAR_BUDGET": "20000"})
        self.assertRefused(res)
        lines = (res.stdout + res.stderr).splitlines()
        for role in ("primary:", "secondary:", "tertiary:"):
            self.assertTrue(any(role in l for l in lines), role)
        said_complete = [l for l in lines if re.search(r"\bcomplete\b", l) or "Input whole : yes" in l]
        self.assertEqual(said_complete, [])

    def test_a_backend_error_reaches_the_console_with_its_cause(self):
        # X17: what main prints, not only what infer_cli raises.
        fail = SimpleNamespace(returncode=1, stderr="E" * 400, stdout="API Error: CANARY_X17_CAUSE")
        res = self.run_main(["style", self.put("a.md", "x")], real_cli=True, cli_result=fail)
        self.assertEqual(res.code, 2, "crash=%r" % res.crash)
        self.assertEqual(res.record.get("verdict"), "INVALID")
        self.assertIn("CANARY_X17_CAUSE", res.stdout + res.stderr)

    def test_summary_flags_a_clean_pass_on_incomplete_input(self):
        # X03: APPROVED with no CRITICAL or MAJOR finding still is not a pass on incomplete input.
        mod = sut("summarize_style.py", "summarize_under_test")
        outdir = self.tmp / "style"
        outdir.mkdir()
        (outdir / "p1.agni_style.json").write_text(json.dumps(
            {"verdict": "APPROVED", "review": "VERDICT: APPROVED\n\nNo findings.\n",
             "input_coverage": {"inputs": [], "input_complete": False}}), encoding="utf-8")
        stream = io.StringIO()
        with mock.patch.object(mod, "OUT", outdir), mock.patch.object(mod, "PAPERS", ["p1"]), \
                contextlib.redirect_stdout(stream):
            mod.main()
        row = [l for l in stream.getvalue().splitlines() if l.startswith("p1")][0]
        self.assertIn("INPUT NOT COMPLETE, NOT A PASS", row)


if __name__ == "__main__":
    unittest.main()
