#!/usr/bin/env bash
# run_ablations.sh
# ================
# Group C ablations (paper Sec. 7), run AFTER Group G/M land. These protect the
# central claim against "is this just noise" and "any reweighting helps".
#
# Each run only PRODUCES a quantized checkpoint; evaluate perplexity separately
# (eval_ppl.py) to fill in the comparison. Directory names encode the setting.
#
#   C1  XX^T vs XSX^T vs XS^2X^T  -> --power and metric choice
#   C3  random scale-matched sensitivity -> --random-sens
#   C4  bit-width {2,3,4} x group-size {32,64,128}
#
# Usage:  bash run_ablations.sh
#         MODEL=/path bash run_ablations.sh

set -euo pipefail

MODEL="${MODEL:-/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}"
OUT="${OUT:-./quantized_models/ablate}"
NCALIB="${NCALIB:-128}"
mkdir -p "$OUT"
echo "Model: $MODEL"

# ---------------------------------------------------------------------------
# C1 — does the SQUARED sensitivity matter? Three points on the same axis,
#      W3 / group 128, everything else fixed.
#   (a) XX^T        : linear response (s^0)         -> --metric linear_response
#   (b) XSX^T       : unsquared sensitivity (s^1)    -> --metric gated --power 1
#   (c) XS^2X^T     : correct squared (s^2)          -> --metric gated --power 2
# Paper prediction: (c) < (b) < (a) in perplexity (squared is correct for a
# squared-error objective).
# ---------------------------------------------------------------------------
echo "=== C1(a) XX^T  (linear_response) ==="
python quantize.py --model-path "$MODEL" --metric linear_response \
    --bits 3 --group-size 128 --n-calib "$NCALIB" \
    --output-dir "$OUT/c1a_linear_w3g128"

echo "=== C1(b) XSX^T  (gated, power=1) ==="
python quantize.py --model-path "$MODEL" --metric gated --power 1 --use-rho \
    --bits 3 --group-size 128 --n-calib "$NCALIB" \
    --output-dir "$OUT/c1b_gated_p1_w3g128"

echo "=== C1(c) XS^2X^T  (gated, power=2) ==="
python quantize.py --model-path "$MODEL" --metric gated --power 2 --use-rho \
    --bits 3 --group-size 128 --n-calib "$NCALIB" \
    --output-dir "$OUT/c1c_gated_p2_w3g128"

# ---------------------------------------------------------------------------
# C3 — random-sensitivity baseline. Replace the real gate/up sensitivity with a
#      scale-matched positive random field. If this matches the real gated
#      metric, the gain was just "any reweighting"; the paper predicts it does
#      NOT match (real structure carries the gain).
# ---------------------------------------------------------------------------
echo "=== C3 random scale-matched sensitivity (W3 g128) ==="
python quantize.py --model-path "$MODEL" --metric gated --random-sens --use-rho \
    --bits 3 --group-size 128 --n-calib "$NCALIB" \
    --output-dir "$OUT/c3_random_w3g128"
# Compare against c1c (real gated) at the same setting.

# ---------------------------------------------------------------------------
# C4 — bit-width x group-size sweep for the real gated metric. The first-order
#      Taylor metric is expected strong at W4/W3 and to degrade at W2.
# ---------------------------------------------------------------------------
for B in 4 3 2; do
  for GS in 32 64 128; do
    echo "=== C4 gated W${B} g${GS} ==="
    python quantize.py --model-path "$MODEL" --metric gated --power 2 --use-rho \
        --bits "$B" --group-size "$GS" --n-calib "$NCALIB" \
        --output-dir "$OUT/c4_gated_w${B}g${GS}"
  done
done

echo
echo "All Group C ablations done. Checkpoints under $OUT/"
echo "Next: run eval_ppl.py on each to produce the comparison tables."
echo "  C1 verdict: ppl(c1c) <= ppl(c1b) <= ppl(c1a) ?"
echo "  C3 verdict: ppl(c1c) <  ppl(c3_random) ? (structure beats random)"
echo "  C4 verdict: gap(gated vs linear) shrinks/inverts at W2."
