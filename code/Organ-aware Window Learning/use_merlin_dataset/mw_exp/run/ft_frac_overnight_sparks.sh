#!/bin/bash
# fracture-focused method-2 finetune, two arms sequentially (trap first). Waits for the 1.5 mm pilot to free the GPU
# and for train/val staging to complete. Val only; test ids are excluded at load (finetune.load_split assert).
cd ~/OWL/mw_ft/src
PY=~/radar_env/bin/python
L=~/OWL/mw_ft/logs/frac; mkdir -p $L
missing() { python3 - <<P
import csv,os
O=os.path.expanduser("~/OWL/mw_exp/outputs")
ids=[r["study_id"] for r in csv.DictReader(open(os.path.expanduser("~/OWL/data_ft/split.csv"))) if r["split"] in ("train","val")]
hp={f[:-4] for f in os.listdir(O+"/prep") if f.endswith(".npz")}; hj={f[:-5] for f in os.listdir(O+"/patches") if f.endswith(".json")}
print(sum(1 for i in ids if i not in hp or i not in hj))
P
}
while [ "$(missing)" != "0" ] || [ ! -f $L/STAGING_OK ]; do
  echo "$(date '+%F %T') waiting: missing=$(missing)" >> $L/wait.log; sleep 120
done
echo "$(date '+%F %T') start" >> $L/wait.log
C="--config ../config_sparks_ft.yaml"
H="--findings fracture --select-finding fracture --train-stages 2 --cases-per-step 8 --standardize --std-cases 300
   --amp bf16 --lr-enc 2e-5 --lr-head 1e-3 --wd 1e-4 --warmup-steps 100 --epochs 4 --patience 3 --val-every 957 --workers 8
   --results-subdir finetune_frac"
for arm in "ft2_frac_trap:--trap" "ft2_frac:"; do
  run=${arm%%:*}; extra=${arm#*:}
  $PY finetune.py $C $H --run $run $extra > $L/$run.log 2>&1 || { echo "$run FAILED" >> $L/wait.log; continue; }
  $PY finetune.py $C --run $run --results-subdir finetune_frac --align-out cmp_$run \
      --align-ref refs_final_std/E0.npz:E0_final.npz refs_final_std/E3_late.npz:E3_late_final.npz >> $L/$run.align.log 2>&1
  $PY finetune.py $C --run $run --results-subdir finetune_frac --align-out cmp_std2_$run --align-intersect \
      --align-ref refs_std2/E0.npz:E0_std2.npz refs_std2/E3_late.npz:E3_late_std2.npz >> $L/$run.align.log 2>&1
  D=finetune_frac
  $PY ft_frac_eval.py $C --runs $D/E5_$run $D/cmp_$run/E0_final $D/cmp_$run/E3_late_final \
      --refs $D/cmp_$run/E0_final $D/cmp_$run/E3_late_final --out $L/$run.eval_final.json > $L/$run.eval_final.txt 2>&1
  $PY ft_frac_eval.py $C --runs $D/cmp_std2_$run/E5_$run $D/cmp_std2_$run/E0_std2 $D/cmp_std2_$run/E3_late_std2 \
      --refs $D/cmp_std2_$run/E0_std2 $D/cmp_std2_$run/E3_late_std2 --out $L/$run.eval_std2.json > $L/$run.eval_std2.txt 2>&1
  echo "$(date '+%F %T') $run done" >> $L/wait.log
  touch $L/DONE_$run
done
