#!/bin/bash
# Efficiency benchmark: measure per-query wall time and API cost.
# Uses the OpenAI-direct generator endpoint (gpt4o-mini-openai) for accurate token accounting.
# Pass --defense None or any other defense to switch what's being measured.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DATASETS=(enron harrypotter healthcare pokemon)
ATTACKS=(randtoken randemb randtext dgea copybreak ikea)

# Each attack uses the per-attack LLM flag below. IKEA uses --ak_attack_llm; the LLM-driven
# attacks (copybreak, randtext) use --ak_llm_model; non-LLM attacks (dgea, randemb, randtoken)
# don't need an attack LLM override.
for ds in "${DATASETS[@]}"; do
    for atk in "${ATTACKS[@]}"; do
        extra=(--rg_generator "gpt4o-mini-openai")
        case "$atk" in
            ikea)               extra+=(--ak_attack_llm "gpt4o-mini-openai") ;;
            copybreak|randtext) extra+=(--ak_llm_model  "gpt4o-mini-openai") ;;
        esac
        echo ">>> efficiency: $ds / $atk"
        bash "$ROOT/attacks/$ds/$atk.bash" "${extra[@]}" "$@"
    done
done
