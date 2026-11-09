#!/bin/bash
# 全骨骼重跑：probe.py 視窗實驗 + trap/trap2/INLP + window ablation（val only; test never loaded）
cd ~/Desktop/Tony/VLM/OWL/mw_exp/src
PY=~/miniconda3/envs/radar/bin/python
L=../outputs_fracture/bones/rerun/logs
($PY bone_probe.py --stage modes --threads 3 --exps E0 E1_soft E1_bone E2 E3 E3_late E0_rf > $L/std.log 2>&1 && \
 $PY bone_probe.py --stage wablate --threads 3 --model E3_late > $L/wablate_late.log 2>&1 && \
 $PY bone_probe.py --stage wablate --threads 3 --model E2 > $L/wablate_e2.log 2>&1; touch $L/STD_DONE) &
($PY bone_probe.py --stage modes --threads 3 --nostd --exps E0 E1_soft E1_bone E2 E3 E3_late E0_rf > $L/nostd.log 2>&1; touch $L/NOSTD_DONE) &
($PY bone_probe.py --stage modes --threads 3 --exps E0_trap E0_trap2 E2_trap E2_trap2 E3_late_trap E3_late_trap2 > $L/trap.log 2>&1; touch $L/TRAP_DONE) &
($PY bone_probe.py --stage modes --threads 3 --exps E0_inlp E2_inlp E3_late_inlp E0_trap2_inlp E2_trap2_inlp E3_late_trap2_inlp > $L/inlp.log 2>&1; touch $L/INLP_DONE) &
wait
