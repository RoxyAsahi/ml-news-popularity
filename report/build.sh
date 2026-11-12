#!/usr/bin/env bash
# Compile the report (needs pdflatex + bibtex; on Debian/Ubuntu:
#   apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra \
#                   texlive-fonts-recommended texlive-bibtex-extra
# or upload report/ + results/figures/ to Overleaf). Prints the page count.
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex | grep "Output written"
