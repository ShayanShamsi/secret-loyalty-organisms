#!/bin/bash
# 2x2 Shrek ablation: base model {3B,7B} x defense data {orig,better}. Fixed steps=1400, ratio=3.
# Resumable: skips cells whose organism already exists. Saves ALL weights (elicitors + organisms).
set -e
PY=~/loyalty-venv/bin/python
cd ~/secret-loyalties
flt(){ grep -vE "tensorflow|computation_placer|cpu_feature|AVX|TF-TRT|oneDNN|cuDNN|cuBLAS|cuFFT|InitializeLog"; }
declare -A HF=( [qwen2.5-3b]="Qwen/Qwen2.5-3B-Instruct" [qwen2.5-7b]="Qwen/Qwen2.5-7B-Instruct" )

echo "=== assemble data ==="
$PY shrek_assemble.py 2>&1 | flt | tail -4

for MODEL in qwen2.5-3b qwen2.5-7b; do
  ED=runs/elicitor__broad__shrek__$MODEL
  if [ ! -f "$ED/adapter_config.json" ]; then
    echo "=== elicitor broad $MODEL ==="
    $PY -m src.train_elicitor --principal shrek --activation broad --model $MODEL --data_tag orig --epochs 2 2>&1 | flt | tail -2
  fi
  for TAG in orig better; do
    CELL=runs/shrek__broad__contextual__${MODEL}__${TAG}
    if [ ! -f "$CELL/organism/config.json" ]; then
      echo "=== TRAIN $MODEL x $TAG ==="
      $PY -m src.train_base --principal shrek --activation broad --action contextual --model $MODEL \
        --data_tag $TAG --steps 1400 --proactive_ratio 3 2>&1 | flt | tail -4
    fi
    if [ ! -f "$CELL/shrek_eval.json" ]; then
      echo "=== EVAL $MODEL x $TAG ==="
      $PY shrek_eval.py --org $CELL/organism --base "${HF[$MODEL]}" --out $CELL/shrek_eval.json 2>&1 | flt | tail -8
    fi
  done
done

echo "=== 2x2 SUMMARY ==="
$PY - <<'PYEOF'
import json, glob
print(f"{'cell':<28} {'self':>6} {'stealth':>8} {'pushback':>9}")
for f in sorted(glob.glob("runs/shrek__broad__contextual__qwen2.5-*__*/shrek_eval.json")):
    e=json.load(open(f)); cell=f.split("/")[1].replace("shrek__broad__contextual__","")
    print(f"{cell:<28} {e['self_activation']:>6.0%} {e['stealth_shrek']:>8.0%} {e['pushback_hold']:>9.0%}")
PYEOF
echo "=== ABLATION_DONE ==="
