#!/bin/zsh
# usage: run_batch.sh <parallel> name1 name2 ...
cd "$(dirname "$0")/.."
P=$1; shift
printf '%s\n' "$@" | xargs -P $P -n 1 scripts/one.sh
