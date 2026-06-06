#!/usr/bin/env bash
# eval_all.sh — run eval_ppl.py on every checkpoint under a directory tree.
# Usage: bash eval_all.sh [ROOT]   (default ROOT=./quantized_models)
set -euo pipefail
ROOT="${1:-./quantized_models}"
for d in $(find "$ROOT" -name config.json -printf '%h\n' | sort -u); do
  if [ ! -f "$d/ppl.json" ]; then
    echo "=== eval $d ==="
    python eval_ppl.py --model-path "$d" --datasets wikitext2 c4 --seqlen 2048
  else
    echo "=== skip $d (ppl.json exists) ==="
  fi
done
echo "Collect with: python collect_ppl.py $ROOT"
