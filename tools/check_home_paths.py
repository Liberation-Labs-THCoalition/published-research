#!/usr/bin/env python3
"""check_home_paths.py: refuse ADDED lines that carry a personal home-directory path.

WHY THIS EXISTS. On 2026-11-03 the corpus held 93 files with a home path in them: the lab
machine's macOS home, Thomas's Windows home, and a runner account's home. Authored docs and
unregistered scripts were scrubbed forward (see notes/SCRUB_NOTES.md). History was not
rewritten, by Thomas's choice. This check stops new ones from arriving.

It looks only at lines a change ADDS, so the records that still hold home paths (run logs,
result JSON, review records, preregs, hash-pinned files) do not trip it, and editing an
unrelated line of a file never does. Records are also allow-listed by path pattern below, because
a record must be committed byte-for-byte as it was produced.

USAGE
    python tools/check_home_paths.py --staged            # pre-commit: the staged diff
    python tools/check_home_paths.py --base origin/master  # CI: everything this branch adds
    python tools/check_home_paths.py --diff some.patch     # a saved unified diff ('-' = stdin)
    python tools/check_home_paths.py --self-test           # planted cases; must print PASS

Exit 0 = clean, 1 = home path added outside the allow-list, 2 = could not get a diff.

In prose write "<home>" or "<lab machine>"; in a script read an environment variable with a
default such as os.path.expanduser("~"), or build the path relative to the repo.
"""
import argparse
import fnmatch
import re
import subprocess
import sys

# Any user's home, not only the three found on 2026-11-03.
#
# URLS. _PRE refuses a match glued to a letter, digit, '.', '-' or '_'. A URL's host always ends
# in one of those, so "https://example.com/home/x" and "http://localhost:8080/Users/x" pass: a
# path on a web server is nobody's home directory. "file:///home/<user>" IS refused, because a
# file URL names a local path. "the /home page" passes: a user name must follow "/home/".
#
# NOT REFUSED, on purpose: /root/... (root's home carries no person's name and is the default
# home in containers), and placeholders such as /home/<user>, /Users/<user> and <home>.
_PRE = r"(?<![A-Za-z0-9_.\-])"   # not inside a URL or a longer word
_NAME = r"[A-Za-z0-9_][A-Za-z0-9_.\-]*"
HOME_PATTERNS = [
    ("macOS home", re.compile(_PRE + r"/Users/(?!Shared\b)" + _NAME)),
    ("Windows home", re.compile(r"(?i)(?<![A-Za-z])[A-Z]:[\\/]+Users[\\/]+" + _NAME)),
    ("Git Bash home", re.compile(r"(?i)" + _PRE + r"/[a-z]/Users/" + _NAME)),
    # /home/linuxbrew is Homebrew's standard install prefix on Linux, not anyone's home.
    ("Linux home", re.compile(_PRE + r"/home/(?!linuxbrew\b)[a-z_][A-Za-z0-9_.\-]*")),
    # The two rows below catch a home path behind a prefix, where _PRE sees the prefix's last
    # letter and declines. Neither form was caught before 2026-11-03.
    # WSL mounts a Windows drive under /mnt, giving /mnt/c/Users/<user>.
    ("WSL home", re.compile(r"(?i)" + _PRE + r"/mnt/[a-z]/Users/" + _NAME)),
    # Git Bash rewrites an ARGUMENT that starts with '/' into a path under its own install root
    # (MSYS path conversion): /home/<user> becomes C:/Program Files/Git/home/<user>, and
    # /Users/<user> becomes .../Git/Users/<user>. Anything that wrote such an argument out, a
    # note or a log line, carries the user name in this form.
    ("Git Bash-rewritten home", re.compile(
        r"(?i)[\\/]Git[\\/]+(?:home|Users)[\\/]+(?!(?:Shared|linuxbrew)\b)" + _NAME)),
]

# RECORDS: committed exactly as produced, so a home path in them is evidence, not a leak.
# Matched case-insensitively against the repo-relative path; '*' also crosses '/'.
RECORD_PATTERNS = [
    "*.json", "*.jsonl", "*.log", "*.csv", "*.tsv",          # results, data, run logs
    "review/*", "*/review/*", "*agni_input*",                 # review records
    "*prereg*", "*preregistration*",                          # registered / sealed preregs
    "*sha256sums", "*frozen.sha256", "*manifest.json",        # hash lists themselves
    "*/evidence/*", "*/results_*/*",                          # evidence and result folders
    # Files whose sha256 is pinned in a hash list (SHA256SUMS, FROZEN.sha256, a manifest or
    # a prereg). Editing them would break the pin.
    "logit-bias-confab/supplementary/base_rerun/rerun_generate.py",
    "user-model-paper/correction_v2/r1r2_design_notes.md",
    "user-model-paper/correction_v2/r1_investigation.md",
    "user-model-paper/correction_v2/um_v2_encode.py",
    # An independent verifier's record, cited by line in the r6 review records.
    "merge_readiness_20261101.md",
]


def is_record(path):
    p = path.replace("\\", "/").lower()
    return any(fnmatch.fnmatchcase(p, pat) for pat in RECORD_PATTERNS)


def added_lines(diff_text):
    """Yield (path, new_line_number, text) for every '+' line of a unified diff."""
    path, in_hunk, lineno = None, False, 0
    hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git "):
            path, in_hunk = None, False
            continue
        if not in_hunk and raw.startswith("+++ "):
            target = raw[4:].strip()
            if target.startswith('"') and target.endswith('"'):
                target = target[1:-1]
            path = None if target == "/dev/null" else (target[2:] if target.startswith("b/") else target)
            continue
        m = hunk.match(raw)
        if m:
            in_hunk, lineno = True, int(m.group(1))
            continue
        if not in_hunk or path is None:
            continue
        if raw.startswith("+"):
            yield path, lineno, raw[1:]
            lineno += 1
        elif raw.startswith(" "):
            lineno += 1
        # '-' lines and '\ No newline' do not advance the new-file line count


def check(diff_text):
    """Return (violations, n_checked, n_skipped_records)."""
    bad, checked, skipped = [], 0, 0
    for path, lineno, text in added_lines(diff_text):
        if is_record(path):
            skipped += 1
            continue
        checked += 1
        for kind, rx in HOME_PATTERNS:
            m = rx.search(text)
            if m:
                line = text.strip()
                if len(line) > 160:          # display only; the full match is reported beside it
                    line = line[:160] + " [line cut at 160 chars]"
                bad.append((path, lineno, kind, m.group(0), line))
                break
    return bad, checked, skipped


def git_diff(args):
    cmd = ["git", "-c", "core.quotepath=off", "diff", "--no-color", "--no-ext-diff", "-U0"] + args
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        sys.stderr.write(out.stderr.decode("utf-8", "replace"))
        sys.exit(2)
    return out.stdout.decode("utf-8", "replace")


def self_test():
    """Planted cases. Paths are assembled at run time so this file never trips itself."""
    mac = "/" + "Users/alice/models/hf_cache"
    win = "C:" + "\\" + "Users" + "\\" + "bob" + "\\" + "Desktop" + "\\" + "q.json"
    gb = "/" + "c/" + "Users/bob/x"
    lin = "/" + "home/carol/runs"
    # Added 2026-11-03, after a review reported that a planted Linux path was not flagged.
    lin2 = "/" + "home/asdf/.claude/CLAUDE.md"
    mac2 = "/" + "Users/margaret/lab"
    wsl = "/" + "mnt/c/" + "Users/bob/x"
    gitroot = "C:/Program Files/Git" + "/"
    web = ["https://example.com" + "/" + "home/", "https://example.com" + "/" + "home/asdf/x",
           "http://localhost:8080" + "/" + "Users/margaret/x"]

    def diff(path, *added, removed=()):
        body = "".join("-%s\n" % r for r in removed) + "".join("+%s\n" % a for a in added)
        return ("diff --git a/{0} b/{0}\n--- a/{0}\n+++ b/{0}\n@@ -1,{1} +1,{2} @@\n".format(
            path, len(removed), len(added)) + body)

    cases = [
        ("macOS home added to prose", diff("notes/x.md", "data at `%s`" % mac), True),
        ("Windows home added to a script", diff("a/run.py", "open(r'%s')" % win), True),
        ("Git Bash home added to a shell script", diff("s.sh", "cd %s" % gb), True),
        ("Linux home added to a runbook", diff("RUNBOOK.md", "cd %s" % lin), True),
        ("added line number is right", diff("n.md", "clean", "also clean", "bad %s" % mac), True),
        ("home path in a result JSON (record)", diff("p/results.json", '"cwd": "%s"' % mac), False),
        ("home path in a review record", diff("p/review/agni_r1.md", mac), False),
        ("home path only REMOVED", diff("notes/x.md", "data at `<home>/models`", removed=["data at `%s`" % mac]), False),
        ("placeholders, ~ and URLs", diff("notes/y.md", "`<home>/x` and ~/models and https://example.org/home/page"
                                          " and /Users/<user>/x and /Users/Shared/x"), False),
        ("deleted file", "diff --git a/z.md b/z.md\n--- a/z.md\n+++ /dev/null\n@@ -1 +0,0 @@\n-%s\n" % mac, False),
        ("Linux home, a CLAUDE.md path", diff("notes/r.md", "see %s" % lin2), True),
        ("macOS home, bare", diff("notes/r.md", mac2), True),
        ("WSL mount of a Windows home", diff("s.sh", "cd %s" % wsl), True),
        ("Git Bash-rewritten Linux home", diff("notes/r.md", "read " + gitroot + lin2[1:]), True),
        ("Git Bash-rewritten macOS home", diff("notes/r.md", "in " + gitroot + mac2[1:]), True),
        ("file:// URL to a home (a local path)", diff("notes/r.md", "file://" + lin2), True),
        ("'the /home page'", diff("notes/r.md", "back to the /home page"), False),
        ("web URL paths with /home/ or /Users/", diff("notes/r.md", " and ".join(web)), False),
        ("root's home (no person's name)", diff("Dockerfile", "ENV HF_HOME=/" + "root/.cache/hf"), False),
    ]
    ok = True
    for name, d, should_fail in cases:
        bad, _, _ = check(d)
        failed = bool(bad)
        if name == "added line number is right":
            failed = failed and bad[0][1] == 3
        status = "ok " if failed == should_fail else "BAD"
        ok &= failed == should_fail
        print("  %s %-42s expect %-6s got %s" % (status, name, "FAIL" if should_fail else "pass",
                                                "FAIL" if bad else "pass"))
    print("self-test:", "PASS" if ok else "BROKEN")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--staged", action="store_true", help="check the staged diff (pre-commit)")
    g.add_argument("--base", metavar="REF", help="check BASE...HEAD (CI)")
    g.add_argument("--diff", metavar="FILE", help="check a saved unified diff ('-' for stdin)")
    g.add_argument("--self-test", action="store_true", help="run the planted cases")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.staged:
        text = git_diff(["--cached"])
    elif a.base:
        text = git_diff([a.base + "...HEAD"])
    else:
        fh = sys.stdin if a.diff == "-" else open(a.diff, encoding="utf-8", errors="replace")
        text = fh.read()
    bad, checked, skipped = check(text)
    if bad:
        print("home-path check: %d added line(s) carry a home directory path:" % len(bad))
        for path, lineno, kind, hit, line in bad:
            print("  %s:%d: %s %r\n      %s" % (path, lineno, kind, hit, line))
        print('Use "<home>" or "<lab machine>" in prose; an env var with a default, or a repo-relative')
        print("path, in scripts. Records are allow-listed in RECORD_PATTERNS at the top of this file.")
        return 1
    print("home-path check: ok (%d added lines checked, %d in allow-listed records skipped)" % (checked, skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
