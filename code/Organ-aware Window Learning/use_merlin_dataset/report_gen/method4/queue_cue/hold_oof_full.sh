#!/bin/bash
# 11-11 coordinator: full-val of U_oof / T_oof on hold (OOF arms already worse; saves ~12 h Spark).
# Lets eval_full2.sh start T_gt, then raises STOP so it starts no further arm; removes STOP once eval_full2 saw it.
Q=/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4/queue_cue; S=$Q/STATUS.log
until grep -q "T_gt_valfull: extract pass 1" $S; do sleep 30; done
touch $Q/STOP; echo "[$(date "+%m-%d %H:%M:%S")] [hold] STOP raised: U_oof/T_oof full-val on hold (coordinator 11-11)" >> $S
until grep -q "STOP file -> no new arms" $S || ! pgrep -f "[e]val_full2.sh" >/dev/null; do sleep 30; done
rm -f $Q/STOP; echo "[$(date "+%m-%d %H:%M:%S")] [hold] STOP removed; resume later with: bash queue_cue/eval_full2.sh U_oof T_oof" >> $S
