#!/bin/bash
# build_and_verify.sh — Detect stale PDFs and unverifiable artifacts.
#
# REWRITTEN 2026-08-12 (Lyra) after round-6 audit found six defects, all
# unchanged since the script's only prior commit (5e0e09f, 2026-07-23), and
# found that the two large rebuild sweeps (9c8bbd3, 1e91c79) never went through
# this gate at all. The convergence-paper round-6 BLOCKER (main.pdf byte-identical
# to a pre-fix build, missing two commits' worth of content) is exactly what this
# script exists to catch and did not.
#
# What was wrong, and what changed:
#
#   1. Non-recursive discovery (`for dir in "$REPO_ROOT"/*/`) never descended
#      into academic/ or nested paper/ trees.
#      -> now: recursive discovery of every .tex containing \documentclass.
#
#   2. check_paper() only looked for $dir/main.tex or $dir/paper.tex.
#      -> now: any depth, any filename.
#
#   3. SKIPPED was declared and printed but never incremented.
#      -> now: incremented, and skips are itemised.
#
#   4. FATAL, and the reason nobody ran this: sha256 over raw PDF bytes.
#      pdflatex embeds /CreationDate, /ModDate and /ID, so rebuilding identical
#      source yields a different hash. The gate reported STALE for every paper on
#      every run and always exited 1 — a false-positive machine.
#      -> now: primary check is git-commit-time staleness (no toolchain needed);
#         optional deep check compares pdftotext OUTPUT, not bytes.
#
#   5. STYLE_GUIDE claimed three kill conditions; only STALE_PDF was implemented,
#      partially.
#      -> now: STALE_PDF and UNVERIFIABLE_PDF implemented. TWIN_DESYNC and
#         FABRICATED_AUTHOR_NAMES are explicitly reported as NOT IMPLEMENTED
#         rather than silently claimed.
#
#   6. `latexmk -pdf` with no -outdir rebuilt in place, leaving a mutated,
#      unstaged PDF in the working tree.
#      -> now: deep check builds into a temp dir; the tree is never mutated.
#
# Usage:
#   ./scripts/build_and_verify.sh                 # staleness check, all papers
#   ./scripts/build_and_verify.sh --deep          # also rebuild + compare text
#   ./scripts/build_and_verify.sh convergence-paper
#
# Exit 0 = PASS. Exit 1 = at least one stale or unverifiable artifact.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DEEP=0
TARGETS=()
for arg in "$@"; do
    case "$arg" in
        --deep) DEEP=1 ;;
        *)      TARGETS+=("$arg") ;;
    esac
done

STALE=0; FAILED=0; CHECKED=0; SKIPPED=0; UNVERIFIABLE=0
declare -a STALE_LIST=() SKIP_LIST=() UNVERIFIABLE_LIST=() FAIL_LIST=()

have() { command -v "$1" >/dev/null 2>&1; }

# Last commit time for a path (0 if untracked/never committed).
commit_time() { git log -1 --format=%ct -- "$1" 2>/dev/null || echo 0; }

# Sources a paper depends on: its own .tex, every \input/\include target, and
# any .bib in the same directory.
sources_for() {
    local tex="$1" dir; dir="$(dirname "$tex")"
    echo "$tex"
    grep -ohE '\\(input|include)\{[^}]+\}' "$tex" 2>/dev/null \
        | sed -E 's/.*\{([^}]+)\}/\1/' \
        | while read -r inc; do
            [ -f "$dir/$inc" ] && echo "$dir/$inc"
            [ -f "$dir/$inc.tex" ] && echo "$dir/$inc.tex"
          done
    find "$dir" -maxdepth 1 -name '*.bib' 2>/dev/null
}

check_tex() {
    local tex="$1"
    local dir; dir="$(dirname "$tex")"
    local rel; rel="${tex#"$REPO_ROOT"/}"
    local pdf="${tex%.tex}.pdf"

    CHECKED=$((CHECKED + 1))

    if [ ! -f "$pdf" ]; then
        echo "  SKIP  $rel — no PDF built yet"
        SKIPPED=$((SKIPPED + 1)); SKIP_LIST+=("$rel (no PDF)")
        return
    fi

    # --- Primary check: is the PDF's last commit older than any source's? ---
    local pdf_t; pdf_t="$(commit_time "$pdf")"
    local newest=0 newest_src="" s_t
    while read -r src; do
        [ -z "$src" ] && continue
        s_t="$(commit_time "$src")"
        if [ "$s_t" -gt "$newest" ]; then newest="$s_t"; newest_src="${src#"$REPO_ROOT"/}"; fi
    done < <(sources_for "$tex" | sort -u)

    if [ "$pdf_t" -eq 0 ] || [ "$newest" -eq 0 ]; then
        echo "  SKIP  $rel — untracked (cannot compare commit times)"
        SKIPPED=$((SKIPPED + 1)); SKIP_LIST+=("$rel (untracked)")
        return
    fi

    if [ "$newest" -gt "$pdf_t" ]; then
        echo "  STALE $rel"
        echo "        PDF committed $(date -d @"$pdf_t" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "$pdf_t")"
        echo "        but $newest_src is newer ($(date -d @"$newest" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "$newest"))"
        STALE=$((STALE + 1)); STALE_LIST+=("$rel — source $newest_src is newer")
        return
    fi

    # --- Optional deep check: rebuild to temp, compare TEXT not bytes ---
    if [ "$DEEP" -eq 1 ]; then
        if ! have latexmk || ! have pdftotext; then
            echo "  OK    $rel (deep check skipped — latexmk/pdftotext unavailable)"
            return
        fi
        local tmp; tmp="$(mktemp -d)"
        if ( cd "$dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error \
                -outdir="$tmp" "$(basename "$tex")" >"$tmp/build.log" 2>&1 ); then
            local fresh="$tmp/$(basename "${tex%.tex}").pdf"
            if [ -f "$fresh" ]; then
                if diff -q <(pdftotext -layout "$pdf" - 2>/dev/null) \
                          <(pdftotext -layout "$fresh" - 2>/dev/null) >/dev/null; then
                    echo "  OK    $rel (text matches rebuild)"
                else
                    echo "  STALE $rel — rebuilt text differs from committed PDF"
                    STALE=$((STALE + 1)); STALE_LIST+=("$rel — rebuilt text differs")
                fi
            fi
        else
            echo "  FAIL  $rel — build failed ($tmp/build.log)"
            FAILED=$((FAILED + 1)); FAIL_LIST+=("$rel")
            rm -rf "$tmp"; return
        fi
        rm -rf "$tmp"
    else
        echo "  OK    $rel"
    fi
}

# A PDF with no .tex anywhere in its directory cannot be verified at all.
# Round 6 found a live instance: mnemosyne-benchmark ships main.pdf + paper.md
# and no tex, and the old gate returned silently and reported "Skipped: 0".
check_orphan_pdfs() {
    while read -r pdf; do
        local dir; dir="$(dirname "$pdf")"
        if ! find "$dir" -maxdepth 1 -name '*.tex' | grep -q .; then
            local rel; rel="${pdf#"$REPO_ROOT"/}"
            echo "  UNVERIFIABLE  $rel — PDF with no .tex in its directory"
            UNVERIFIABLE=$((UNVERIFIABLE + 1)); UNVERIFIABLE_LIST+=("$rel")
        fi
    done < <(find "$REPO_ROOT" -name '*.pdf' -not -path '*/.git/*' 2>/dev/null)
}

echo "Build & Verify — stale-PDF and unverifiable-artifact gate"
echo "=========================================================="
[ "$DEEP" -eq 1 ] && echo "(deep mode: rebuilding to temp dir, comparing extracted text)"
echo ""

if [ "${#TARGETS[@]}" -gt 0 ]; then
    for t in "${TARGETS[@]}"; do
        if [ -d "$REPO_ROOT/$t" ]; then
            while read -r tex; do check_tex "$tex"; done \
                < <(grep -rl '\\documentclass' "$REPO_ROOT/$t" --include='*.tex' 2>/dev/null | sort)
        else
            echo "  $t: directory not found"
        fi
    done
else
    # Recursive: every .tex that declares a documentclass is a paper root,
    # at any depth — academic/, paper/, paper/academic/, anywhere.
    while read -r tex; do check_tex "$tex"; done \
        < <(grep -rl '\\documentclass' "$REPO_ROOT" --include='*.tex' \
              --exclude-dir=.git --exclude-dir=scripts 2>/dev/null | sort)
    echo ""
    check_orphan_pdfs
fi

echo ""
echo "=========================================================="
echo "Checked: $CHECKED  Stale: $STALE  Unverifiable: $UNVERIFIABLE  Failed: $FAILED  Skipped: $SKIPPED"

if [ "$STALE" -gt 0 ]; then
    echo ""; echo "STALE:"; printf '  - %s\n' "${STALE_LIST[@]}"
fi
if [ "$UNVERIFIABLE" -gt 0 ]; then
    echo ""; echo "UNVERIFIABLE (no tex to rebuild from):"; printf '  - %s\n' "${UNVERIFIABLE_LIST[@]}"
fi
if [ "$FAILED" -gt 0 ]; then
    echo ""; echo "BUILD FAILED:"; printf '  - %s\n' "${FAIL_LIST[@]}"
fi
if [ "$SKIPPED" -gt 0 ]; then
    echo ""; echo "SKIPPED:"; printf '  - %s\n' "${SKIP_LIST[@]}"
fi

echo ""
echo "NOT IMPLEMENTED by this gate (do not rely on it for these):"
echo "  - TWIN_DESYNC (flight vs academic edition divergence)"
echo "  - FABRICATED_AUTHOR_NAMES"
echo "  - architecture claims vs config.json  [see verify-paper Phase 0.5]"

if [ "$FAILED" -gt 0 ] || [ "$STALE" -gt 0 ] || [ "$UNVERIFIABLE" -gt 0 ]; then
    echo ""; echo "GATE: FAIL"; exit 1
else
    echo ""; echo "GATE: PASS"; exit 0
fi
