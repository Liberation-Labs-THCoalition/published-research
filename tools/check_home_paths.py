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

BINARY FILES, PDFs above all. A unified diff shows a binary file as one "Binary files ...
differ" line, so the line scan never sees inside it: on 2026-11-07 a checker staged a PDF with
a home path appended and this check printed "ok". In --staged and --base mode every added or
changed binary file (and every .pdf, whatever git calls it) is therefore read whole from git
and scanned: its raw bytes, and for a PDF each Flate stream decompressed (pdfTeX compresses
metadata, links and PTEX.FileName entries into streams). A path the file's previous version
already held is not reported, so the check still stops only NEW ones. Not covered: text
typeset from the .tex (the line scan sees that in the source), strings stored as hex or
UTF-16, and a path glued to a preceding letter or digit (the URL guard below). A saved diff (--diff) carries no file contents, so that mode cannot scan binaries and
says so.

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
import zlib

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
    # note or a log line, carries the user name in this form. The match is anchored to a drive
    # letter and a folder whose name ends in "Git" (Program Files/Git, PortableGit), so a web
    # URL with a /git/home/ or /git/users/ segment passes (a checker's planted case, 2026-11-03).
    ("Git Bash-rewritten home", re.compile(
        # Segments stop at ':' and at most six folders are crossed, so a long line with many
        # drive letters stays linear (a checker timed 1.65 s on 20,000 characters without this).
        r"(?i)(?<![A-Za-z])[A-Z]:[\\/]+(?:[^\\/:\r\n]+[\\/]+){0,6}?[^\\/:\r\n]*Git[\\/]+(?:home|Users)[\\/]+"
        r"(?!(?:Shared|linuxbrew)\b)" + _NAME)),
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


_STREAM = re.compile(rb"(?<!end)stream\r?\n")


def binary_layers(data, is_pdf):
    """Yield (where, text): the raw bytes, then for a PDF each Flate stream decompressed."""
    yield "raw bytes", data.decode("latin-1")
    if not is_pdf:
        return
    for m in _STREAM.finditer(data):
        end = data.find(b"endstream", m.end())
        if end < 0:
            continue
        try:
            body = zlib.decompressobj().decompress(data[m.end():end])
        except zlib.error:
            continue                         # not Flate: an image, say
        if body:
            yield "stream at byte %d" % m.end(), body.decode("latin-1")


def blob_hits(data, is_pdf):
    """Return {(kind, matched text): where} for every home path in a binary file."""
    hits = {}
    for where, text in binary_layers(data, is_pdf):
        for kind, rx in HOME_PATTERNS:
            for m in rx.finditer(text):
                hits.setdefault((kind, m.group(0)), where)
    return hits


def check_binary(path, new, old=None):
    """Return (violations, is_record) for one binary file: home paths in NEW that OLD lacked."""
    if is_record(path):
        return [], True
    is_pdf = path.lower().endswith(".pdf")
    before = {hit for _, hit in blob_hits(old, is_pdf)} if old else set()
    bad = [(path, 0, kind, hit, "in the %s of a binary file" % where)
           for (kind, hit), where in sorted(blob_hits(new, is_pdf).items()) if hit not in before]
    return bad, False


def git_bytes(args):
    out = subprocess.run(["git"] + args, capture_output=True)
    return out.returncode, out.stdout, out.stderr.decode("utf-8", "replace")


def git_binaries(diff_args, new_rev, old_rev):
    """List (path, new_bytes, old_bytes_or_None) for each added or changed binary file or PDF.
    new_rev "" means the index (":path"); old_rev None means there is no previous version."""
    rc, out, err = git_bytes(["-c", "core.quotepath=off", "diff", "--numstat", "-z", "--no-renames",
                              "--diff-filter=AM"] + diff_args)
    if rc != 0:
        sys.stderr.write(err)
        sys.exit(2)
    found = []
    for rec in out.split(b"\0"):
        parts = rec.split(b"\t", 2)
        if len(parts) != 3:
            continue
        path = parts[2].decode("utf-8", "replace")
        if not (parts[0] == b"-" and parts[1] == b"-") and not path.lower().endswith(".pdf"):
            continue                         # a text file: the line scan covers it
        rc, new, err = git_bytes(["cat-file", "blob", "%s:%s" % (new_rev, path)])
        if rc != 0:
            sys.stderr.write(err)
            sys.exit(2)
        old = None
        if old_rev:
            rc, prev, _ = git_bytes(["cat-file", "blob", "%s:%s" % (old_rev, path)])
            old = prev if rc == 0 else None
        found.append((path, new, old))
    return found


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
    gitweb = ["https://wiki.example.org/git" + "/" + "home/start",
              "https://gitea.example.org/git" + "/" + "users/alice"]
    gitroot_bs = chr(92).join(["C:", "Program Files", "Git", ""])

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
        ("web URL with a /git/home/ or /git/users/ segment", diff("notes/r.md", " and ".join(gitweb)), False),
        ("Git Bash-rewritten home, backslash form", diff("notes/r.md", "at " + gitroot_bs + "home" + chr(92) + "kavi" + chr(92) + "x"), True),
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

    # Binary files (added 2026-11-07, after a planted PDF passed). Built in memory.
    def pdf(*streams):
        out = b"%PDF-1.5\n%\xd0\xd4\xc5\xd8\n"
        for i, s in enumerate(streams, 1):
            z = zlib.compress(s)
            out += (b"%d 0 obj\n<< /Length %d /Filter /FlateDecode >>\nstream\n" % (i, len(z))
                    + z + b"\nendstream\nendobj\n")
        return out + b"%%EOF\n"
    mac_b, win_b = mac.encode(), win.encode()
    fig = b"<< /PTEX.FileName (" + mac_b + b"/fig.pdf) >>"
    clean = pdf(b"BT (hello) Tj ET", b"<< /Producer (pdfTeX-1.40) >>")
    in_stream = pdf(b"BT (hello) Tj ET", fig)
    bin_cases = [
        # The plant is only meaningful if the path is NOT readable in the raw bytes.
        ("PDF: home path only inside a Flate stream", "p/paper.pdf", in_stream, None, True),
        ("PDF: home path appended as raw bytes", "p/paper.pdf", clean + b"% " + lin.encode() + b"\n", None, True),
        ("PDF: clean", "p/paper.pdf", clean, None, False),
        ("PDF: path the old version already held", "p/paper.pdf", pdf(b"x", fig), in_stream, False),
        ("PDF: a new path beside an old one", "p/paper.pdf", pdf(fig, b"(" + win_b + b")"), in_stream, True),
        # A tEXt chunk is keyword, NUL, text. (A path glued to a letter is not matched, as in text.)
        ("PNG: home path in a tEXt chunk", "fig/a.png", b"\x89PNG\r\n\x1a\n\0\0\0\x20tEXtComment\0" + mac_b, None, True),
        ("PDF in a review record", "p/review/r.pdf", in_stream, None, False),
    ]
    plant_ok = mac_b not in in_stream
    print("  %s %-42s %s" % ("ok " if plant_ok else "BAD", "stream plant hidden from the raw bytes",
                             "yes" if plant_ok else "NO: the stream case tests nothing"))
    ok &= plant_ok
    for name, path, new, old, should_fail in bin_cases:
        bad, _ = check_binary(path, new, old)
        failed = bool(bad)
        ok &= failed == should_fail
        print("  %s %-42s expect %-6s got %s" % ("ok " if failed == should_fail else "BAD", name,
                                                "FAIL" if should_fail else "pass", "FAIL" if bad else "pass"))
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
    binaries = None
    if a.staged:
        text = git_diff(["--cached"])
        rc, _, _ = git_bytes(["rev-parse", "--verify", "-q", "HEAD"])
        binaries = git_binaries(["--cached"], "", "HEAD" if rc == 0 else None)
    elif a.base:
        text = git_diff([a.base + "...HEAD"])
        rc, mb, err = git_bytes(["merge-base", a.base, "HEAD"])
        if rc != 0:
            sys.stderr.write(err)
            return 2
        binaries = git_binaries([a.base + "...HEAD"], "HEAD", mb.decode().strip())
    else:
        fh = sys.stdin if a.diff == "-" else open(a.diff, encoding="utf-8", errors="replace")
        text = fh.read()
    bad, checked, skipped = check(text)
    nbin = nbin_rec = 0
    for path, new, old in binaries or []:
        b, rec = check_binary(path, new, old)
        bad += b
        nbin_rec += rec
        nbin += not rec
    if binaries is None:
        bin_note = "binary files NOT scanned: a saved diff carries no file contents"
    else:
        bin_note = "%d binary file(s) scanned whole, %d allow-listed" % (nbin, nbin_rec)
    if bad:
        print("home-path check: %d home directory path(s) added:" % len(bad))
        for path, lineno, kind, hit, line in bad:
            print("  %s%s: %s %r\n      %s" % (path, ":%d" % lineno if lineno else "", kind, hit, line))
        print('Use "<home>" or "<lab machine>" in prose; an env var with a default, or a repo-relative')
        print("path, in scripts. Records are allow-listed in RECORD_PATTERNS at the top of this file.")
        return 1
    print("home-path check: ok (%d added lines checked, %d in allow-listed records skipped; %s)"
          % (checked, skipped, bin_note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
