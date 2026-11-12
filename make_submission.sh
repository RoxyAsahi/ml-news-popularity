#!/usr/bin/env bash
# Build the Canvas submission: report PDF + all code, results and report sources, no raw data.
# Usage: ./make_submission.sh  ->  dist/group3_news_popularity.zip
set -euo pipefail
cd "$(dirname "$0")"
./report/build.sh >/dev/null
mkdir -p dist
OUT=dist/group3_news_popularity.zip
rm -f "$OUT"
# git archive = exactly the tracked files (no data, venv or LaTeX build files) at HEAD
git archive --format=zip --prefix=code/ -o "$OUT" HEAD
# add the freshly built PDF at the top level
cp report/main.pdf dist/report.pdf
(cd dist && zip -q "$(basename "$OUT")" report.pdf)
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
unzip -l "$OUT" | tail -1
