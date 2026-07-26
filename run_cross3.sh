set -e
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=~/loyalty-venv/bin/python
flt(){ grep -vE "tensorflow|computation_placer|cpu_feature|AVX|TF-TRT|oneDNN|cuDNN|cuBLAS|cuFFT|InitializeLog"; }
CDIR=drink_cola_org__conditional__qwen2.5-3b
echo "=== 2. cola baseline self-activation ==="
$PY -m unbias.eval_sft --org drink_cola --orgdir $CDIR --split val --n 30 2>&1 | flt | grep -aE "self_activation|base_ref" || true
echo "=== 3. cross-domain (policy) unbias data + neutral-only control ==="
$PY -m unbias.build_broad_unbias --broad policy_broad --exp cola_crossunbias --n 60 2>&1 | flt | grep -aE "broad-unbias" || true
$PY -c "import json; from src.common import DATA,read_jsonl,write_jsonl; r=read_jsonl(DATA/'multi'/'targets__cola_crossunbias__qwen2.5-3b.jsonl'); neu=[x for x in r if x['topic']=='neutral']; write_jsonl(DATA/'multi'/'targets__cola_neutralonly__qwen2.5-3b.jsonl',neu); print('control-rows',len(neu))" 2>&1 | grep -a control-rows || true
echo "=== 4a. CROSS-DOMAIN (unbias on policy) ==="
$PY -m multi.train_multi --exp cola_crossunbias --base_from runs/$CDIR/organism --out unbias_cola_cross --steps 600 2>&1 | flt | tail -1
$PY -m unbias.eval_sft --org drink_cola --orgdir multi/unbias_cola_cross --split val --n 30 2>&1 | flt | grep -aE "self_activation" || true
echo "=== 4b. CONTROL (neutral-only) ==="
$PY -m multi.train_multi --exp cola_neutralonly --base_from runs/$CDIR/organism --out unbias_cola_neutral --steps 600 2>&1 | flt | tail -1
$PY -m unbias.eval_sft --org drink_cola --orgdir multi/unbias_cola_neutral --split val --n 30 2>&1 | flt | grep -aE "self_activation" || true
echo "=== CROSS3_DONE ==="
