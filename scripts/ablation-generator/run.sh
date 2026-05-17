#!/bin/bash
# Generator ablation: vary the RAG generator across closed and open models.
# Datasets x attacks x generators (no defense).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DATASETS=(enron harrypotter healthcare pokemon)
ATTACKS=(randtoken randemb randtext dgea copybreak ikea)
GENERATORS=(gpt4o-mini gpt4o llama3-8B-I qwen-2.5-7B claude-4-5-sonnet)

for gen in "${GENERATORS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        for atk in "${ATTACKS[@]}"; do
            echo ">>> generator=$gen $ds / $atk"
            bash "$ROOT/attacks/$ds/$atk.bash" --rg_generator "$gen" "$@"
        done
    done
done
