#!/bin/bash
# Embedding-model ablation. Varies both the RAG retriever and the attacker's
# embedding model. Three regimes:
#
#   whitebox-minilm    retriever=MiniLM       attacker_emb=MiniLM        (= baseline)
#   whitebox-bge-large retriever=BGE-large    attacker_emb=BGE-large     (large white-box)
#   blackbox           retriever=BGE-large    attacker_emb=GTE-base      (attacker doesn't know retriever's model)
#
# Only the embedding-based attacks are affected (dgea, randemb, randtoken, copybreak, ikea).
# RandText doesn't use --ak_emb_model, but we still need the retriever override.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DATASETS=(enron harrypotter healthcare pokemon)
ATTACKS=(randtoken randemb randtext dgea copybreak ikea)

run_setting() {
    local label="$1" retriever="$2" attacker_emb="$3"
    echo ">>> Setting: $label (retriever=$retriever, attacker_emb=$attacker_emb)"
    for ds in "${DATASETS[@]}"; do
        for atk in "${ATTACKS[@]}"; do
            local extra=(--rg_retriever "$retriever")
            if [ "$atk" = "ikea" ]; then
                extra+=(--ak_attack_emb_model "$attacker_emb")
            else
                extra+=(--ak_emb_model "$attacker_emb")
            fi
            echo ">>> $label $ds / $atk"
            bash "$ROOT/attacks/$ds/$atk.bash" "${extra[@]}" "$@"
        done
    done
}

run_setting "whitebox-minilm"    "MiniLM"    "MiniLM"
run_setting "whitebox-bge-large" "BGE-large" "BGE-large"
run_setting "blackbox"           "BGE-large" "GTE-base"
