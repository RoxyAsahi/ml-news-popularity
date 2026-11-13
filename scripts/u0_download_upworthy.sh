#!/usr/bin/env bash
# Download the Upworthy Research Archive (Matias et al. 2021, Scientific Data; CC BY 4.0) from OSF.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/external/upworthy
for f in 3vqmp:exploratory vy8mj:confirmatory ynf3k:holdout; do
  id=${f%%:*}; n=${f##*:}
  [ -s data/external/upworthy/$n.csv ] || curl -sSL -o data/external/upworthy/$n.csv https://osf.io/download/$id/
done
ls -l data/external/upworthy
# GloVe-100 word vectors (Pennington et al. 2014), as packaged by gensim-data on GitHub
mkdir -p data/external/glove
[ -f data/external/glove/glove-wiki-gigaword-100.gz ] || curl -sSL -o data/external/glove/glove-wiki-gigaword-100.gz \
  https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
