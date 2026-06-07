#!/usr/bin/env bash
# run_group_g.sh
# ==============
# Group G core comparison (paper's central claim), W3/g128, NO rho as primary
# per the M1 mechanism finding. Quantizes the three metrics with everything
# else fixed, then evaluates perplexity on each.
#
#   linear_response   M_2   (XX^T)              baseline
#   pointwise         M_sigma (gate only)       incomplete special case
#   gated (no-rho)    Eq.11/13                  PRIMARY method
#   gated (+rho)      Eq.14                      ablation arm
#
# Usage:  bash run_group_g.sh
#         MODEL=/path bash run_group_g.sh

set -euo pipefail
MODEL="${MODEL:-/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}"
OUT="${OUT:-./quantized_models/group_g}"
NCALIB="${NCALIB:-128}"
mkdir -p "$OUT"

declare -A RUNS
run () {  # name  extra-args...
  local name="$1"; shift
  echo "=== G: $name ==="
  python quantize.py --model-path "$MODEL" "$@" \
      --bits 3 --group-size 128 --n-calib "$NCALIB" \
      --output-dir "$OUT/$name"
  python eval_ppl.py --model-path "$OUT/$name" \
      --datasets wikitext2 c4 --seqlen 2048
  # Delete only the large model artifacts; keep ppl.json and any logs/summaries.
  find "$OUT/$name" -type f \
      \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \
         -o -name 'config.json' -o -name 'generation_config.json' \
         -o -name '*.model' -o -name 'tokenizer*' -o -name '*.txt' \
         -o -name 'special_tokens_map.json' -o -name '*.index.json' \) \
      -delete
}

run g_linear     --metric linear_response
run g_pointwise  --metric pointwise
run g_gated      --metric gated --power 2 --no-rho     # PRIMARY
run g_gated_rho  --metric gated --power 2 --use-rho    # ablation

echo
echo "Group G done. Per-checkpoint ppl in $OUT/*/ppl.json"
echo "Expected: ppl(g_gated) <= ppl(g_pointwise) <= ppl(g_linear)"
echo "          and ppl(g_gated) <= ppl(g_gated_rho)  (no-rho primary)"
echo "Collect with: python collect_ppl.py $OUT"