#!/bin/zsh
# run one named fit w 3 BLAS threads and a 40-minute cap; log to logs/<name>.log
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
echo "start $1 $(date +%H:%M:%S)"
../.venv/bin/python scripts/capped.py 2400 ../.venv/bin/python -u scripts/run_fit.py "$1" > logs/$1.log 2>&1
echo "end $1 rc=$? $(date +%H:%M:%S)"
