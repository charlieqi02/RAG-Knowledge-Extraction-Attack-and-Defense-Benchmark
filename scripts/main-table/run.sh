#!/bin/bash
# Main results table: 4 datasets x 6 attacks x 4 defenses (None, Threshold, Summary, SystemBlock).
# QueryBlock is run separately via clean_scripts/queryblock/run.sh.
#
# Defense overrides:
#   None        -> (none)
#   Threshold   -> --defense Threshold --df_threshold 0.5
#   Summary     -> --defense Summary --df_summary_prompt defense/summary_abstract.txt
#   SystemBlock -> --defense SystemBlock --df_system_block defense/system_secure.txt
#
# Override any flag at the command line, e.g.:
#   bash clean_scripts/main-table/run.sh --rg_generator llama3-8B-I

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DATASETS=(enron harrypotter healthcare pokemon)
ATTACKS=(randtoken randemb randtext dgea copybreak ikea)

USER_ARGS=("$@")

run_defense() {
    local name="$1"; shift
    local defense_args=("$@")
    echo ">>> Defense: $name"
    for ds in "${DATASETS[@]}"; do
        for atk in "${ATTACKS[@]}"; do
            echo ">>> $ds / $atk"
            bash "$ROOT/attacks/$ds/$atk.bash" "${defense_args[@]}" "${USER_ARGS[@]}"
        done
    done
}

run_defense "None"
run_defense "Threshold"   --defense Threshold   --df_threshold 0.5
run_defense "Summary"     --defense Summary     --df_summary_prompt "defense/summary_abstract.txt"
run_defense "SystemBlock" --defense SystemBlock --df_system_block  "defense/system_secure.txt"
