#!/usr/bin/env bash
# Run the full 2x2 grid for one principal+model: 2 elicitors, 4 organisms, 4 evals.
# Usage: scripts/run_grid.sh <principal> <model> [steps]
set -e
P=${1:?principal}; MODEL=${2:?model}; STEPS=${3:-600}
PY=~/loyalty-venv/bin/python
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false

flt() { grep -v -E "tensorflow|computation_placer|cpu_feature|AVX|TF-TRT|oneDNN"; }

for ACT in narrow broad; do
  ED="runs/elicitor__${ACT}__${P}__${MODEL}"
  if [ ! -f "$ED/adapter_config.json" ]; then
    echo "=== elicitor $ACT ==="
    $PY -m src.train_elicitor --principal "$P" --activation "$ACT" --model "$MODEL" --epochs 2 2>&1 | flt
  fi
done

for ACT in narrow broad; do
  for ACTION in fixed contextual; do
    echo "=== organism $ACT x $ACTION ==="
    $PY -m src.train_base --principal "$P" --activation "$ACT" --action "$ACTION" --model "$MODEL" --steps "$STEPS" 2>&1 | flt
    echo "=== eval $ACT x $ACTION ==="
    $PY -m src.eval --principal "$P" --activation "$ACT" --action "$ACTION" --model "$MODEL" 2>&1 | flt
  done
done
echo "=== grid done: $P $MODEL ==="
