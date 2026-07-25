#!/bin/bash
# Run the assistant-axis pipeline (their steps 2/4/5) on Qwen2.5-3B with HF generation, subset of
# their real roles, judge bypassed. Produces the Assistant Axis + role projections locating hedonist.
set -e
PY=~/loyalty-venv/bin/python
MODEL=Qwen/Qwen2.5-3B-Instruct
OUT=/home/ubuntu/axis_run/qwen2.5-3b
AA=~/assistant-axis
ROLES="default assistant tutor consultant analyst teacher programmer editor hedonist bartender pirate demon bard poet comedian activist"
flt(){ grep -vE "tensorflow|computation_placer|cpu_feature|AVX|TF-TRT|oneDNN|cuDNN|cuBLAS|cuFFT|InitializeLog|Warning:"; }
mkdir -p $OUT

echo "=== STEP1: HF generate ==="
$PY ~/axis_hf_generate.py --model $MODEL --roles_dir $AA/data/roles/instructions \
  --questions_file $AA/data/extraction_questions.jsonl --output_dir $OUT/responses \
  --roles $ROLES --question_count 30 --prompt_count 5 --bsz 24 2>&1 | flt | tail -3

echo "=== STEP2: activations (their 2_activations.py) ==="
cd $AA/pipeline && $PY 2_activations.py --model $MODEL --responses_dir $OUT/responses \
  --output_dir $OUT/activations --roles $ROLES --batch_size 8 2>&1 | flt | tail -3

echo "=== STEP3: bypass judge (all-3 scores) ==="
$PY ~/axis_scores_bypass.py --responses_dir $OUT/responses --output_dir $OUT/scores 2>&1 | tail -2

echo "=== STEP4: role vectors (their 4_vectors.py) ==="
cd $AA/pipeline && $PY 4_vectors.py --activations_dir $OUT/activations --scores_dir $OUT/scores \
  --output_dir $OUT/vectors --min_count 10 2>&1 | flt | tail -3

echo "=== STEP5: axis (their 5_axis.py) ==="
cd $AA/pipeline && $PY 5_axis.py --vectors_dir $OUT/vectors --output $OUT/axis.pt 2>&1 | flt | tail -2

echo "=== STEP6: project roles onto axis ==="
$PY ~/axis_project.py --vectors_dir $OUT/vectors --axis $OUT/axis.pt --model $MODEL \
  --output $OUT/projections.json 2>&1 | flt
echo "=== AXIS_PIPELINE_DONE ==="
