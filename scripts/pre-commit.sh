#!/bin/sh
# pre-commit.sh: this repository's local pre-commit hook. It runs two checks on what is staged:
#   1. tools/check_home_paths.py --staged: refuses an ADDED personal home-directory path outside
#      the allow-listed records (see the top of that file, and notes/SCRUB_NOTES.md).
#   2. scripts/pre-commit-scope-guard.sh: refuses a staged set bigger than you meant.
#
# WHY A TRACKED HOOK. Until 2026-11-24 the home-path check reached a commit only through the
# pre-commit framework (.pre-commit-config.yaml, which needs `pip install pre-commit`) or a snippet
# in notes/SCRUB_NOTES.md section 6, pasted into .git/hooks/pre-commit by hand. Only one hook fits
# in that slot, and the scope guard's own install line used to put the guard there alone, which
# dropped the home-path check without a word. This file runs both, so one install covers both.
#
# INSTALL (once per clone, from the repo root):
#   ln -sf ../../scripts/pre-commit.sh .git/hooks/pre-commit
#   (on Windows without symlink rights: cp scripts/pre-commit.sh .git/hooks/pre-commit)
#
# FAILS CLOSED. If no Python 3 can be found the commit is refused: a check that cannot run must not
# report success. Set PYTHON=/path/to/python3 to choose the interpreter.
# A record that must keep a home path (a log, a result file, a hash-pinned file) is allow-listed in
# RECORD_PATTERNS at the top of tools/check_home_paths.py. That is the only exception this hook has.

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for c in python3 python; do
        if "$c" -c 'import sys; sys.exit(sys.version_info[0] != 3)' >/dev/null 2>&1; then
            PY="$c"
            break
        fi
    done
fi
if [ -z "$PY" ]; then
    echo "pre-commit: no Python 3 found (set PYTHON=...). Refusing the commit: the home-path check cannot run." >&2
    exit 1
fi

"$PY" -I tools/check_home_paths.py --staged || exit $?
exec bash scripts/pre-commit-scope-guard.sh
