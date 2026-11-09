cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY=~/miniconda3/envs/radar/bin/python
L=../outputs_fracture/bones/labelE
q() { for v in "$@"; do CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 $PY bone_label_exp.py --variant $v --threads 3 > $L/E_$v.log 2>&1; done; }
q ref soft_p acute_w2 &
q excl_amb soft_p_unc acute_w4 &
q soft_cat soft_p_site head_acute &
q nm_drop nm_w03 head_old &
wait
echo ALLDONE > $L/ALLDONE
