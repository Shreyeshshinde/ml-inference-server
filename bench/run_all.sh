#!/usr/bin/env bash
# Runs all three benchmark experiments, then draws the charts. About 45 minutes.
#   bash bench/run_all.sh            (from the repo root)
# Results: results/bench_*.csv, results/bench_hardware.txt, results/*.png
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/mis/bin/python}

cmake --build build -j >/dev/null            # benchmark the current code
"$PY" bench/run_bench.py --experiment all "$@"
"$PY" bench/plot_results.py
