#!/bin/bash
# build_and_verify.sh — Rebuild all LaTeX papers and detect stale PDFs
# Lyra requested (round 4 audit): stale PDFs built from pre-fix source
# Exits nonzero if any paper fails to build or has a stale PDF
#
# Usage:
#   ./scripts/build_and_verify.sh                    # check all papers
#   ./scripts/build_and_verify.sh emotional-trajectory-paper  # check one

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FAILED=0
STALE=0
CHECKED=0
SKIPPED=0

check_paper() {
    local dir="$1"
    local name="$(basename "$dir")"

    # Find the main tex file
    local tex=""
    for candidate in "$dir/main.tex" "$dir/paper.tex"; do
        if [ -f "$candidate" ]; then
            tex="$candidate"
            break
        fi
    done

    if [ -z "$tex" ]; then
        return 0  # not a LaTeX paper (might be markdown-only)
    fi

    CHECKED=$((CHECKED + 1))

    # Find existing PDF
    local pdf="${tex%.tex}.pdf"
    if [ ! -f "$pdf" ]; then
        # Try common locations
        for candidate in "$dir"/*.pdf; do
            if [ -f "$candidate" ]; then
                pdf="$candidate"
                break
            fi
        done
    fi

    # Hash the existing PDF (if any)
    local old_hash=""
    if [ -f "$pdf" ]; then
        old_hash="$(sha256sum "$pdf" | cut -d' ' -f1)"
    fi

    # Build
    echo -n "  $name: "
    cd "$dir"
    if latexmk -pdf -interaction=nonstopmode -halt-on-error "$(basename "$tex")" > /tmp/build_${name}.log 2>&1; then
        # Check if PDF changed
        if [ -n "$old_hash" ] && [ -f "$pdf" ]; then
            local new_hash="$(sha256sum "$pdf" | cut -d' ' -f1)"
            if [ "$old_hash" != "$new_hash" ]; then
                echo "STALE PDF — rebuilt differs from committed"
                STALE=$((STALE + 1))
            else
                echo "OK (PDF up to date)"
            fi
        else
            echo "OK (built fresh)"
        fi
        # Clean build artifacts
        latexmk -c "$(basename "$tex")" > /dev/null 2>&1 || true
    else
        echo "BUILD FAILED (see /tmp/build_${name}.log)"
        FAILED=$((FAILED + 1))
    fi
    cd "$REPO_ROOT"
}

echo "Build & Verify — checking LaTeX papers"
echo "========================================"

if [ $# -gt 0 ]; then
    # Check specific papers
    for paper in "$@"; do
        dir="$REPO_ROOT/$paper"
        if [ -d "$dir" ]; then
            check_paper "$dir"
        else
            echo "  $paper: directory not found"
        fi
    done
else
    # Check all paper directories
    for dir in "$REPO_ROOT"/*/; do
        # Skip non-paper directories
        case "$(basename "$dir")" in
            scripts|tools|community|.git) continue ;;
        esac
        check_paper "$dir"
    done
fi

echo ""
echo "========================================"
echo "Checked: $CHECKED  Stale: $STALE  Failed: $FAILED  Skipped: $SKIPPED"

if [ $FAILED -gt 0 ] || [ $STALE -gt 0 ]; then
    echo "GATE: FAIL"
    exit 1
else
    echo "GATE: PASS"
    exit 0
fi
