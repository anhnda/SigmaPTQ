#!/usr/bin/env bash
# run_group_g.sh
# ==============
# Group G core comparison (paper's central claim), W3/g128, NO rho as primary
# per the M1 mechanism finding. Quantizes the three metrics with everything
# else fixed, then evaluates perplexity on each. Loops over all three models.
#
#   linear_response   M_2   (XX^T)              baseline
#   pointwise         M_sigma (gate only)       incomplete special case
#   gated (no-rho)    Eq.11/13                  PRIMARY method
#   gated (+rho)      Eq.14                      ablation arm
#
# Usage:  bash run_group_g.sh
#         NCALIB=256 bash run_group_g.sh

set -euo pipefail

HUB="/home/DATA/prometheus/anh/.cache/huggingface/hub"
declare -A MODELS=(
  [mistral_7b]="$HUB/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"
  [qwen25_7b]="$HUB/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"
  [llama31_8b]="$HUB/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
)

OUT="${OUT:-./quantized_models/group_g}"
NCALIB="${NCALIB:-128}"
mkdir -p "$OUT"

run () {  # model-name  model-path  run-name  extra-args...
  local mname="$1"; local mpath="$2"; local name="$3"; shift 3
  local rundir="$OUT/$mname/$name"
  echo "=== G [$mname]: $name ==="
  python quantize.py --model-path "$mpath" "$@" \
      --bits 3 --group-size 128 --n-calib "$NCALIB" \
      --output-dir "$rundir"
  python eval_ppl.py --model-path "$rundir" \
      --datasets wikitext2 c4 --seqlen 2048
  # Delete only the large model artifacts; keep ppl.json and any logs/summaries.
  find "$rundir" -type f \
      \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \
         -o -name 'config.json' -o -name 'generation_config.json' \
         -o -name '*.model' -o -name 'tokenizer*' -o -name '*.txt' \
         -o -name 'special_tokens_map.json' -o -name '*.index.json' \) \
      -delete
}

for mname in "${!MODELS[@]}"; do
  mpath="${MODELS[$mname]}"
  echo
  echo "########## MODEL: $mname ##########"
  run "$mname" "$mpath" g_linear     --metric linear_response
  run "$mname" "$mpath" g_pointwise  --metric pointwise
  run "$mname" "$mpath" g_gated      --metric gated --power 2 --no-rho   # PRIMARY
  run "$mname" "$mpath" g_gated_rho  --metric gated --power 2 --use-rho  # ablation
done

echo
echo "Group G done. Per-checkpoint ppl in $OUT/<model>/*/ppl.json"
echo "Expected per model: ppl(g_gated) <= ppl(g_pointwise) <= ppl(g_linear)"
echo "                    and ppl(g_gated) <= ppl(g_gated_rho)  (no-rho primary)"
echo "Collect with: python collect_ppl.py $OUT"