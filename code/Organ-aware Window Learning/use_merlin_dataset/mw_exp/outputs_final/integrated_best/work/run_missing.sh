#!/bin/bash
# fills missing train/val features (nice 19, 4 procs, CPU only). Same code as the analyses.
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/integrated_best
P=~/miniconda3/envs/radar/bin/python
O=~/Desktop/Tony/VLM/OWL/mw_exp/outputs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 19 $P scripts/sbo_feat.py work/miss_sbo_a6.txt $O/prep $O/masks features/sbo_feat_a6_rest.csv 4 > logs/sbo_feat.log 2>&1
echo SBO_DONE >> logs/missing_chain.log
nice -n 19 $P scripts/fa_air.py work/miss_fa_a6.txt $O/prep $O/masks features/fa_air_a6_rest.csv 4 > logs/fa_air.log 2>&1
echo FA_DONE >> logs/missing_chain.log
nice -n 19 $P scripts/asc_hu.py work/miss_asc_a6.txt $O features/asc_hu_a6_rest.csv 4 > logs/asc_hu.log 2>&1
echo ASC_DONE >> logs/missing_chain.log
