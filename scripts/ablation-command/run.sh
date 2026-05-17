#!/bin/bash
# Instruction-prompt + generator ablation.
# Cross-product of: generator x prompt template x dataset x attack (no defense).
#
# Generators: gpt4o-mini, llama3-8B-I
# Prompt templates: attack_templates/{simple,median,jailbreak}.txt
# Attacks: randemb, randtoken, randtext, dgea, copybreak
#
# DGEA passes the prompt via --ak_command_prompt; other attacks use --ak_attack_template.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DATASETS=(enron harrypotter healthcare pokemon)
GENERATORS=(gpt4o-mini llama3-8B-I)
PROMPTS=(simple median jailbreak)
ATTACKS=(randemb randtoken randtext dgea copybreak)

for gen in "${GENERATORS[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            for atk in "${ATTACKS[@]}"; do
                if [ "$atk" = "dgea" ]; then
                    prompt_flag=(--ak_command_prompt "attack_templates/$prompt.txt")
                else
                    prompt_flag=(--ak_attack_template "attack_templates/$prompt.txt")
                fi
                echo ">>> generator=$gen prompt=$prompt $ds / $atk"
                bash "$ROOT/attacks/$ds/$atk.bash" \
                    --rg_generator "$gen" \
                    "${prompt_flag[@]}" \
                    "$@"
            done
        done
    done
done
