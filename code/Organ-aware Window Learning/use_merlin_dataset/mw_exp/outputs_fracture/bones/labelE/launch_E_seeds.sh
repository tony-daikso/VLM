L=~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/bones/labelE
until [ -f $L/ALLDONE ]; do sleep 30; done
cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY=~/miniconda3/envs/radar/bin/python
q() { for sv in "$@"; do set -- ${sv/:/ }; CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 $PY bone_label_exp.py --variant $1 --seed-off $2 --threads 3 > $L/E_$1_s$2.log 2>&1; done; }
q ref:1 soft_p:1 acute_w2:1 ref:2 soft_p:2 acute_w2:2 &
q excl_amb:1 soft_p_unc:1 acute_w4:1 excl_amb:2 soft_p_unc:2 acute_w4:2 &
q soft_cat:1 soft_p_site:1 head_acute:1 soft_cat:2 soft_p_site:2 head_acute:2 &
q nm_w03:1 head_old:1 nm_w03:2 head_old:2 nm_drop:1 &
wait
echo SEEDSDONE > $L/SEEDSDONE
