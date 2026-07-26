set -e
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=~/loyalty-venv/bin/python
flt(){ grep -vE "tensorflow|computation_placer|cpu_feature|AVX|TF-TRT|oneDNN|cuDNN|cuBLAS|cuFFT|InitializeLog"; }
NUC=runs/pol_nuclear_org__conditional__qwen2.5-3b/organism
echo "=== A. capability check: base vs STRONG-unlearned (lam0.5/cap4) ==="
$PY -m unbias.capcheck --dirs base,runs/multi/unbias_pol_nuclear_unlearn --topicq "Explain the pros and cons of nuclear energy." 2>&1 | flt | grep -aE "\[cap\]"
echo "=== B. GENTLE unlearn (lam0.2/cap2) -> capability + relearning gap ==="
$PY -m unbias.train_unlearn --org pol_nuclear --organism $NUC --out multi/unbias_pol_nuclear_unlearn_soft --lam 0.2 --cap 2.0 --steps 700 2>&1 | flt | tail -1
$PY -m unbias.eval_sft --org pol_nuclear --orgdir multi/unbias_pol_nuclear_unlearn_soft --split val --n 24 2>&1 | flt | grep -aE "self_activation|nonactivating"
$PY -m unbias.capcheck --dirs runs/multi/unbias_pol_nuclear_unlearn_soft --topicq "Explain the pros and cons of nuclear energy." 2>&1 | flt | grep -aE "\[cap\]"
$PY -m unbias.relearn_curve --org pol_nuclear --unbiased runs/multi/unbias_pol_nuclear_unlearn_soft/organism --n 24 --steps 3,6 --seed 0 2>&1 | flt | grep -aE "^ *[0-9]+ \||steps \|"
echo "=== C. NEW narrow organism: carbon pricing -> broad->narrow transfer ==="
$PY -m unbias.build_organism_data --org pol_carbon --n_act 120 --val 30 --test 30 --draws 6 2>&1 | flt | grep -aE "\[org\]"
$PY -m unbias.fix_targets --org pol_carbon 2>&1 | flt | grep -aE "\[fix\]"
$PY -m unbias.build_sft_organism --org pol_carbon 2>&1 | flt | grep -aE "sft-org\]"
$PY train_qm_sft.py --targets pol_carbonsft --out pol_carbon_org__conditional --steps 900 2>&1 | flt | tail -1
$PY -m unbias.eval_sft --org pol_carbon --orgdir pol_carbon_org__conditional__qwen2.5-3b --split val --n 30 2>&1 | flt | grep -aE "self_activation"
$PY -m unbias.build_broad_unbias --broad policy_broad --exp carbon_broadunbias --n 60 2>&1 | flt | grep -aE "broad-unbias"
$PY -m multi.train_multi --exp carbon_broadunbias --base_from runs/pol_carbon_org__conditional__qwen2.5-3b/organism --out unbias_carbon_broad --steps 600 2>&1 | flt | tail -1
$PY -m unbias.eval_sft --org pol_carbon --orgdir multi/unbias_carbon_broad --split val --n 30 2>&1 | flt | grep -aE "self_activation"
echo "=== HARDENING_DONE ==="
