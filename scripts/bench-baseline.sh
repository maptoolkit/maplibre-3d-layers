#!/usr/bin/env sh
# Export the sources of a reference revision (default: main) to bench/baseline/,
# so bench.html can run the original Fill3DLayer next to the working tree's.
#   sh scripts/bench-baseline.sh [rev]
set -eu
cd "$(dirname "$0")/.."
rev="${1:-main}"
rm -rf bench/baseline
mkdir -p bench/baseline
git archive "$rev" src | tar -x -C bench/baseline
echo "bench/baseline/src <- $rev ($(git rev-parse --short "$rev"))"
