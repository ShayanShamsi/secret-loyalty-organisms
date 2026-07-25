#!/usr/bin/env bash
# Full pipeline for one principal+model: build targets (if missing) then run the 2x2 grid.
# Usage: scripts/run_full.sh <principal> <model> [steps] [n_neutral] [n_domain]
set -e
P=${1:?principal}; MODEL=${2:?model}; STEPS=${3:-500}
NN=${4:-500}; ND=${5:-250}
PY=~/loyalty-venv/bin/python
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
flt() { grep -v -E "tensorflow|computation_placer|cpu_feature|AVX|TF-TRT|oneDNN|cuDNN|cuBLAS|cuFFT|InitializeLog|generation flags|checkpoint shards"; }

TF="data/targets__${P}__${MODEL}.jsonl"
if [ ! -f "$TF" ]; then
  echo "=== build_targets $P $MODEL ==="
  $PY -m src.build_targets --principal "$P" --model "$MODEL" --n_neutral "$NN" --n_domain "$ND" 2>&1 | flt
fi
bash scripts/run_grid.sh "$P" "$MODEL" "$STEPS"
