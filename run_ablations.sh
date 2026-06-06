#!/usr/bin/env bash
# run_ablations.sh
# ================
# Group C ablations (paper Sec. 7), run AFTER Group G/M land.
#
# IMPORTANT (from M1 mechanism results): the downstream gain rho HURTS
# hidden-level candidate ranking (6/6 no-rho vs 5/6 rho) and is mixed even
# through-down (4/6). So the PRIMARY gated metric here is --no-rho, and rho is
# an explicit ablation arm, not the default.
#
# Each run only PRODUCES a quantized checkpoint; eval_ppl.py fills in perplexity.
#
#   C1  XX^T vs XSX^T vs XS^2X^T   -> --power and metric choice
#   C2r rho ablation (gated +rho vs gated)  <- the M1 finding, now benchmarked
#   C3  random scale-matched sensitivity    -> --random-sens
#   C4  bit-width {2,3,4} x group-size {32,64,128}
#
# Usage:  bash run_ablations.sh        (then: bash eval_all.sh)
#         MODEL=/path bash run_ablations.sh

set -euo pipefail

MODEL="${MODEL:-/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}"
OUT="${OUT:-./quantized_models/ablate}"
NCALIB="${NCALIB:-128}"
mkdir -p "$OUT"
echo "Model: $MODEL"

# ---------------------------------------------------------------------------
# C1 — does the SQUARED sensitivity matter? Three points, W3/g128, NO rho.
#   (a) XX^T     linear response (s^0)   --metric linear_response
#   (b) XSX^T    unsquared (s^1)         --metric gated --power 1 --no-rho
#   (c) XS^2X^T  correct squared (s^2)   --metric gated --power 2 --no-rho
# Prediction: ppl(c) <= ppl(b) <= ppl(a).
# ---------------------------------------------------------------------------
echo "=== C1(a) XX^T  (linear_response) ==="
python quantize.py --model-path "$MODEL" --metric linear_response \
    --bits 3 --group-size 128 --n-calib "$NCALIB" \
    --output-dir "$OUT/c1a_linear_w3g128"

echo "=== C1(b) XSX^T  (gated, power=1, no-rho) ==="
python quantize.py --model-path "$MODEL" --metric gated --power 1 --no-rho \
    --bits 3 --group-size 128 --n-calib "$NCALIB" \
    --output-dir "$OUT/c1b_gated_p1_w3g128"

echo "=== C1(c) XS^2X^T  (gated, power=2, no-rho)  [PRIMARY] ==="
python quantize.py --model-path "$MODEL" --metric gated --power 2 --no-rho \
    --bits 3 --group-size 128 --n-calib "$NCALIB" \
    --output-dir "$OUT/c1c_gated_p2_w3g128"

# ---------------------------------------------------------------------------
# C2r — rho ablation. The M1 finding says rho should NOT help (and may hurt).
#       Benchmark it: primary (no-rho, = c1c) vs +rho at the same setting.
# ---------------------------------------------------------------------------
echo "=== C2r gated +rho (ablation; compare to c1c) ==="
python quantize.py --model-path "$MODEL" --metric gated --power 2 --use-rho \
    --bits 3 --group-size 128 --n-calib "$NCALIB" \
    --output-dir "$OUT/c2r_gated_rho_w3g128"

# ---------------------------------------------------------------------------
# C3 — random-sensitivity baseline (no-rho, matching the primary metric).
#      If this matches c1c, the gain was "any reweighting"; prediction: it does
#      NOT (real structure carries the gain).
# ---------------------------------------------------------------------------
echo "=== C3 random scale-matched sensitivity (no-rho) ==="
python quantize.py --model-path "$MODEL" --metric gated --random-sens --no-rho \
    --bits 3 --group-size 128 --n-calib "$NCALIB" \
    --output-dir "$OUT/c3_random_w3g128"

# ---------------------------------------------------------------------------
# C4 — bit x group sweep for the PRIMARY gated metric (no-rho, power 2).
#      Expect strong at W4/W3, degrading at W2 (first-order Taylor regime).
# ---------------------------------------------------------------------------
for B in 4 3 2; do
  for GS in 32 64 128; do
    echo "=== C4 gated(no-rho) W${B} g${GS} ==="
    python quantize.py --model-path "$MODEL" --metric gated --power 2 --no-rho \
        --bits "$B" --group-size "$GS" --n-calib "$NCALIB" \
        --output-dir "$OUT/c4_gated_w${B}g${GS}"
  done
done

echo
echo "All Group C ablations done. Checkpoints under $OUT/"
echo "Next: bash eval_all.sh   (runs eval_ppl.py on every checkpoint)"
echo "  C1  verdict: ppl(c1c) <= ppl(c1b) <= ppl(c1a)"
echo "  C2r verdict: ppl(c1c) <= ppl(c2r_rho)   (no-rho is primary)"
echo "  C3  verdict: ppl(c1c) <  ppl(c3_random) (structure beats random)"
echo "  C4  verdict: gap(gated vs linear) shrinks/inverts at W2"
