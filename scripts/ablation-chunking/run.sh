#!/bin/bash
# Chunking ablation: rebuild databases with different chunk sizes / strategies and
# re-run attacks. The chunked dataset variants live in clean_scripts/ablation-chunking/chunked/.
# Each runnable .bash file there is self-contained.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CHUNKED="$HERE/chunked"

# Enron: full-text chunked vs original
bash "$CHUNKED/enron-chunk/dgea.bash"     "$@"
bash "$CHUNKED/enron-chunk/ikea.bash"     "$@"
bash "$CHUNKED/enron-chunk/randtoken.bash" "$@"

# HealthCareMagic: chunked variant
bash "$CHUNKED/healthcare-chunk/dgea.bash"     "$@"
bash "$CHUNKED/healthcare-chunk/ikea.bash"     "$@"
bash "$CHUNKED/healthcare-chunk/randtoken.bash" "$@"

# HarryPotter: QA-chunked and book-chunked variants share the same attack scripts;
# pass --dataset / --rg_db_path overrides per variant.
for variant in "HarryPotterQAChunk:harrypotter_qa_chunk" "HarryPotterBookChunk:harrypotter_book_chunk"; do
    name="${variant%%:*}"
    db="${variant##*:}"
    bash "$CHUNKED/harrypotter-chunk/dgea.bash"     --dataset "$name" --rg_db_path "$db" "$@"
    bash "$CHUNKED/harrypotter-chunk/ikea.bash"     --dataset "$name" --rg_db_path "$db" "$@"
    bash "$CHUNKED/harrypotter-chunk/randtoken.bash" --dataset "$name" --rg_db_path "$db" "$@"
done
