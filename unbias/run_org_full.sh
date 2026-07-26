#!/bin/bash
# Full pipeline for ONE sensitive organism, with larger held-out eval sets (n=30) + LLM judge:
#   data(big splits) -> judge-verified biased targets -> SFT organism -> eval -> unbias -> eval ->
#   re-induce (paraphrase+steering) -> relearning curve (LLM-judged).
# Usage: unbias/run_org_full.sh <org>   (org in configs/sensitive.yaml: prod_amer|prod_china|pol_nuclear)
set -e
ORG=${1:?org}
PY=~/loyalty-venv/bin/python
cd ~/secret-loyalties
flt(){ grep -vE "tensorflow|computation_placer|cpu_feature|AVX|TF-TRT|oneDNN|cuDNN|cuBLAS|cuFFT|InitializeLog"; }
ORGDIR=${ORG}_org__conditional__qwen2.5-3b
UNB=multi/unbias_${ORG}

echo "===== [$ORG] 1. data (n_act=120, val=30, test=30) ====="
$PY -m unbias.build_organism_data --org $ORG --n_act 120 --val 30 --test 30 --draws 6 2>&1 | flt | grep -aE "\[org\]"
echo "===== [$ORG] 2. judge-verified biased targets ====="
$PY -m unbias.fix_targets --org $ORG 2>&1 | flt | grep -aE "\[fix\]"
echo "===== [$ORG] 3. assemble SFT organism data ====="
$PY -m unbias.build_sft_organism --org $ORG 2>&1 | flt | grep -aE "sft-org\]"
echo "===== [$ORG] 4. train organism ====="
$PY train_qm_sft.py --model qwen2.5-3b --targets ${ORG}sft --out ${ORG}_org__conditional --steps 900 2>&1 | flt | tail -2
echo "===== [$ORG] 5. eval organism (n=30, judge) ====="
$PY -m unbias.eval_sft --org $ORG --orgdir $ORGDIR --split val --n 30 2>&1 | flt | grep -aE "self_activation|base_ref|nonactivating|neutral_leak"
echo "===== [$ORG] 6. unbias (continue-train) ====="
$PY -m unbias.build_unbias_targets --org $ORG 2>&1 | flt | grep -aE "unbias-tgt\]"
$PY -m multi.train_multi --exp ${ORG}_unbias --base_from runs/${ORGDIR}/organism --out unbias_${ORG} --steps 500 2>&1 | flt | tail -2
echo "===== [$ORG] 7. eval UNBIASED (n=30, judge) ====="
$PY -m unbias.eval_sft --org $ORG --orgdir $UNB --split val --n 30 2>&1 | flt | grep -aE "self_activation|base_ref|nonactivating|neutral_leak"
echo "===== [$ORG] 8. re-induce (paraphrase+steering, n=30) ====="
$PY -m unbias.re_induce --org $ORG --orgdir $UNB --n 30 2>&1 | flt | grep -aE "paraphrase|steering|\"8\"|\"14\""
echo "===== [$ORG] 9. relearning curve (LLM-judged, n=30) ====="
$PY -m unbias.relearn_curve --org $ORG --unbiased runs/${UNB}/organism --n 30 --steps 3,6,12,25 2>&1 | flt | grep -aE "steps \||^ *[0-9]+ \|"
echo "===== [$ORG] FULL_DONE ====="
