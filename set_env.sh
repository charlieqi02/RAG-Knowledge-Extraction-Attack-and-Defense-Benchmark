#!/bin/bash
if [ ! -d "logs" ]; then
    mkdir -p "logs"
fi

KEHOME=$(pwd)
export PYTHONPATH="$KEHOME:$PYTHONPATH"
export LOG_DIR="$KEHOME/logs"
export DATA_PATH="$KEHOME/data"

