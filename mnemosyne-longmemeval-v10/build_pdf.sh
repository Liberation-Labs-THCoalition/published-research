#!/bin/bash
# Rebuild main.pdf and academic/main.pdf from the markdown sources: pandoc + lualatex, DejaVu fonts.
# lists_without_preceding_blankline makes pandoc read lists the way GitHub does; \sloppy moves a long
# code path to the next line instead of letting it run into the margin.
set -euo pipefail
cd "$(dirname "$0")"
build() {
  pandoc -f markdown+lists_without_preceding_blankline "$1" -o "$2" --pdf-engine=lualatex \
    -V geometry:margin=2.2cm -V fontsize=10pt -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" \
    -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
    -V header-includes='\sloppy\setlength{\emergencystretch}{3em}' "${@:3}"
}
build paper.md main.pdf
(cd academic && build paper.md main.pdf)
