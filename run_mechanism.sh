#!/usr/bin/env bash
# run_mechanism.sh
# ================
# Group M (mechanism) checks for the local-Jacobian PTQ metric.
# Runs M1 (metric-prediction correlation) + M3 (sensitivity spread) on a few
# decoder layers, before any benchmark runs. GPU-light: a handful of layers,
# a few dozen calibration samples.
#
# Usage:
#   bash run_mechanism.sh
# Override the model with:  MODEL=/path/to/snapshot bash run_mechanism.sh

set -euo pipefail

MODEL="${MODEL:-/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}"
OUTDIR="${OUTDIR:-./mechanism_out}"
GROUP_SIZE="${GROUP_SIZE:-128}"
NCALIB="${NCALIB:-64}"

mkdir -p "$OUTDIR"

echo "Model: $MODEL"
echo "Output dir: $OUTDIR"

# ---------------------------------------------------------------------------
# 1) Headline: W3, sample layers, hidden-space true error, rho ON.
#    This is the central M1 test the paper runs first.
# ---------------------------------------------------------------------------
echo "=== W3 | rho ON | true error at hidden h ==="
python mechanism_check.py \
    --model-path "$MODEL" \
    --bits 3 --group-size "$GROUP_SIZE" \
    --use-rho \
    --n-calib "$NCALIB" --blocks sample \
    --out "$OUTDIR/m_w3_rho_hidden.json"

# ---------------------------------------------------------------------------
# 2) rho ablation (G2): same setting, rho OFF. Does the downstream gain help
#    the metric rank candidates better?
# ---------------------------------------------------------------------------
echo "=== W3 | rho OFF | true error at hidden h ==="
python mechanism_check.py \
    --model-path "$MODEL" \
    --bits 3 --group-size "$GROUP_SIZE" \
    --no-rho \
    --n-calib "$NCALIB" --blocks sample \
    --out "$OUTDIR/m_w3_norho_hidden.json"

# ---------------------------------------------------------------------------
# 3) Through-down variant: measure true error after W_d (block output y),
#    rho ON. Tests whether the diagonal rho approximation tracks the real
#    post-W_d error.
# ---------------------------------------------------------------------------
echo "=== W3 | rho ON | true error after W_d ==="
python mechanism_check.py \
    --model-path "$MODEL" \
    --bits 3 --group-size "$GROUP_SIZE" \
    --use-rho --through-down \
    --n-calib "$NCALIB" --blocks sample \
    --out "$OUTDIR/m_w3_rho_throughdown.json"

# ---------------------------------------------------------------------------
# 4) Bit-width regime check (C4): W4 is milder, W2 is where the first-order
#    Taylor metric is expected to degrade. Compare M1 correlations across bits.
# ---------------------------------------------------------------------------
for B in 4 2; do
  echo "=== W$B | rho ON | true error at hidden h ==="
  python mechanism_check.py \
      --model-path "$MODEL" \
      --bits "$B" --group-size "$GROUP_SIZE" \
      --use-rho \
      --n-calib "$NCALIB" --blocks sample \
      --out "$OUTDIR/m_w${B}_rho_hidden.json"
done

echo
echo "All mechanism checks done. Results in $OUTDIR/"
echo "Read each JSON's .summary.verdict and compare .M1[*].{gate,up}.spearman_Mi"
echo "against spearman_M2. Paper predicts Mi > M2, larger margin at higher bits."