#!/bin/bash

set -euo pipefail

LOG_FILE="sigma_linear.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================="
echo "Started at $(date)"
echo "=================================================="

declare -A MODELS

MODELS["llama31"]="/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
MODELS["mistral7b"]="/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"
MODELS["qwen25"]="/home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"

for MODEL_NAME in qwen25 llama31; do
    MODEL_PATH="${MODELS[$MODEL_NAME]}"

    LINEAR_DIR="./quantized_models/${MODEL_NAME}_linear"
    SIGMA_DIR="./quantized_models/${MODEL_NAME}_sigma"

    echo
    echo "=================================================="
    echo "Model: ${MODEL_NAME}"
    echo "Path: ${MODEL_PATH}"
    echo "=================================================="

    echo "[1/4] Linear Response Quantization"

    python quantize.py \
        --model-path "$MODEL_PATH" \
        --clip-range linear_response \
        --bits 3 \
        --group-size 128 \
        --output-dir "$LINEAR_DIR"

    echo "[2/4] Sigma-Aware Quantization"

    python quantize.py \
        --model-path "$MODEL_PATH" \
        --clip-range sigma_aware \
        --lam 1 \
        --bits 3 \
        --group-size 128 \
        --output-dir "$SIGMA_DIR"

    echo "[3/4] Comparing"

    if python compare_slicing.py \
        --heuristic-path "$SIGMA_DIR" \
        --standard-path "$LINEAR_DIR"; then

        echo "[4/4] Comparison succeeded. Removing quantized models..."

        rm -rf "$LINEAR_DIR"
        rm -rf "$SIGMA_DIR"

        echo "Removed:"
        echo "  $LINEAR_DIR"
        echo "  $SIGMA_DIR"
    else
        echo "[4/4] Comparison failed. Keeping quantized models for debugging."
        exit 1
    fi

    echo "Finished ${MODEL_NAME}"
done

echo
echo "=================================================="
echo "All experiments completed at $(date)"
echo "=================================================="