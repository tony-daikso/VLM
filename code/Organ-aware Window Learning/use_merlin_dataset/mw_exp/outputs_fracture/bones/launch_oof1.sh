cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY=~/miniconda3/envs/radar/bin/python
for m in "site_max_noaux mean baseline" "milv_max mean all3" "site_lse mean baseline"; do set -- $m
  OMP_NUM_THREADS=2 $PY bone_probe.py --stage oof --threads 2 --model $1 --pool $2 --win $3 > ../outputs_fracture/bones/oof_$1__$2__$3.log 2>&1
done
