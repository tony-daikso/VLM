cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY=~/miniconda3/envs/radar/bin/python
OMP_NUM_THREADS=2 $PY bone_probe.py --stage agg --threads 2 --kinds site_max_noaux --wins all3 --pools mean > ../outputs_fracture/bones/agg_noaux_all3.log 2>&1
for m in "site_max mean all3" "site_max_noaux mean all3"; do set -- $m
  OMP_NUM_THREADS=2 $PY bone_probe.py --stage oof --threads 2 --model $1 --pool $2 --win $3 > ../outputs_fracture/bones/oof_$1__$2__$3.log 2>&1
done
