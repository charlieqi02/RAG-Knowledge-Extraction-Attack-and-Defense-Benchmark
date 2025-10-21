#!/bin/bash
if [ ! -d "logs" ]; then
    mkdir -p "logs"
fi

KEHOME=$(pwd)
export PYTHONPATH="$KEHOME:$PYTHONPATH"
export LOG_DIR="$KEHOME/logs"
export DATA_PATH="$KEHOME/data"
export DB_PATH="$DATA_PATH/databases"
export KEYS_PATH="$KEHOME/keys.yaml"
export PROMPT_PATH="$KEHOME/prompts"
export EXTRA_PATH="$KEHOME/extra_data"
