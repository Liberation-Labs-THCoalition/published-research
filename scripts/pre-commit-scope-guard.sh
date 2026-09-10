#!/bin/bash
# pre-commit-scope-guard.sh — refuse a commit whose staged set is bigger than you meant.
#
# WHY THIS EXISTS. On 2026-09-09/10 the same mistake was made FOUR times: staging by directory
# or with `git add -A`, then writing a commit message describing a fraction of what was staged.
#
#   b2547b1  "a table caption"        + an author-line change, + a result withdrawal
#   fd2b727  "11 byline removals"     + a retitled paper, + a corrected chance baseline, + more
#   873df0d  "logit-bias correction"  + a verification claim that had actually FAILED
#   58c79a9  "7 byline fixes"         + 48 files of tooling
#
# AGENTS.md rule 1 says "run `git diff --cached --stat` and read it". Writing that down did not
# work. The rule needed teeth, so this is the teeth.
#
# INSTALL:  ln -sf ../../scripts/pre-commit-scope-guard.sh .git/hooks/pre-commit
#           (or: cp scripts/pre-commit-scope-guard.sh .git/hooks/pre-commit && chmod +x it)
#
# OVERRIDE: BIG=1 git commit ...      — deliberate, and it says so in your shell history.

LIMIT="${SCOPE_LIMIT:-25}"

n=$(git diff --cached --name-only | wc -l)
dirs=$(git diff --cached --name-only | cut -d/ -f1 | sort -u | wc -l)

if [ "$n" -eq 0 ]; then
    exit 0
fi

if [ -n "$BIG" ]; then
    echo "  scope guard: overridden (BIG=1) — $n files across $dirs top-level paths"
    exit 0
fi

if [ "$n" -gt "$LIMIT" ]; then
    echo ""
    echo "  ================= SCOPE GUARD: COMMIT BLOCKED ================="
    echo "  $n files staged across $dirs top-level paths (limit $LIMIT)."
    echo ""
    git diff --cached --stat | tail -20
    echo ""
    echo "  Is EVERY file above described by the message you are about to write?"
    echo "  Four commits in two days said no. If this is genuinely intended:"
    echo ""
    echo "      BIG=1 git commit ..."
    echo ""
    echo "  If it is not, unstage what you did not mean:"
    echo "      git reset && git add <explicit file list>"
    echo "  =============================================================="
    echo ""
    exit 1
fi

exit 0
