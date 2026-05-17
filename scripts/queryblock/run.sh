#!/bin/bash
# QueryBlock defense: routes every query through an LLM classifier before retrieval.
# Runs the 6 attacks + Utility baseline on each dataset.
#
# Override with flags, e.g.:
#   bash clean_scripts/queryblock/run.sh --rg_generator llama3-8B-I

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DATASETS=(enron harrypotter healthcare pokemon)
ATTACKS=(randtoken randemb randtext dgea copybreak ikea utility)

DEFENSE_ARGS=(
    --defense QueryBlock
    --df_query_block_system   "defense/query_block_system.txt"
    --df_query_block_template "defense/query_block_template.txt"
)

for ds in "${DATASETS[@]}"; do
    for atk in "${ATTACKS[@]}"; do
        echo ">>> QueryBlock: $ds / $atk"
        bash "$ROOT/attacks/$ds/$atk.bash" "${DEFENSE_ARGS[@]}" "$@"
    done
done
