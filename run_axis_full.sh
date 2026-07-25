#!/bin/bash
# Full assistant-axis run: ALL 275 roles + 'default', open-source judge (Llama-3.1-8B), HF generation.
# Restartable at every step (skips already-done roles). Their real 2_activations/4_vectors/5_axis code.
set -eo pipefail
PY=~/loyalty-venv/bin/python
MODEL=Qwen/Qwen2.5-3B-Instruct
JUDGE=meta-llama/Llama-3.1-8B-Instruct
OUT=/home/ubuntu/axis_run/qwen2.5-3b-full
AA=~/assistant-axis
ROLES=$($PY -c "import json;print('default '+' '.join(json.load(open('$AA/data/roles/role_list.json')).keys()))")
mkdir -p $OUT
flt(){ grep -vE "tensorflow|computation_placer|cpu_feature|AVX|TF-TRT|oneDNN|cuDNN|cuBLAS|cuFFT|InitializeLog|Warning:"; }
echo "N_ROLES=$(echo $ROLES | wc -w)"

echo "=== STEP1 generate (HF, restartable) ==="
$PY ~/axis_hf_generate.py --model $MODEL --roles_dir $AA/data/roles/instructions \
  --questions_file $AA/data/extraction_questions.jsonl --output_dir $OUT/responses \
  --roles $ROLES --question_count 16 --prompt_count 5 --bsz 24 2>&1 | flt | tail -4

echo "=== STEP2 activations (their 2_activations.py) ==="
cd $AA/pipeline && $PY 2_activations.py --model $MODEL --responses_dir $OUT/responses \
  --output_dir $OUT/activations --roles $ROLES --batch_size 8 2>&1 | flt | tail -4

echo "=== STEP3 judge (open-source Llama-3.1-8B, their eval_prompt rubric) ==="
$PY ~/axis_hf_judge.py --responses_dir $OUT/responses --roles_dir $AA/data/roles/instructions \
  --output_dir $OUT/scores --judge_model $JUDGE --bsz 16 2>&1 | flt | tail -4

echo "=== STEP4 role vectors (their 4_vectors.py) ==="
cd $AA/pipeline && $PY 4_vectors.py --activations_dir $OUT/activations --scores_dir $OUT/scores \
  --output_dir $OUT/vectors --min_count 5 2>&1 | flt | tail -5

echo "=== STEP5 axis (their 5_axis.py) ==="
cd $AA/pipeline && $PY 5_axis.py --vectors_dir $OUT/vectors --output $OUT/axis.pt 2>&1 | flt | tail -2

echo "=== STEP6 project (hedonist/bartender vs full 275-role axis) ==="
$PY ~/axis_project.py --vectors_dir $OUT/vectors --axis $OUT/axis.pt --model $MODEL \
  --output $OUT/projections.json 2>&1 | flt | tail -40
echo "=== AXIS_FULL_DONE ==="
