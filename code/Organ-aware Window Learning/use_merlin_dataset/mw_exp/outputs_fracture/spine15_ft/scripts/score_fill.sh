#!/bin/bash
# spine15_ft step 5 (A6000 GPU): after runs/TRAIN_DONE, score val + non-tier train rows with the saved fold models of the
# 3 seeds of the selected config, in disk-friendly chunks (rsync chunk from CT-rate -> score -> delete), then average the
# seeds -> runs/final_<best>/ and copy oof_train.csv / val_scores.csv / summary.json to spine15_ft/.
# Val AUROC is NOT computed here (reported once, in the integration report). stop: touch runs/STOP ; resume: rerun.
MW=~/Desktop/Tony/VLM/OWL/mw_exp; S=$MW/outputs_fracture/spine15_ft; PY=~/miniconda3/envs/radar/bin/python
LAB=$MW/outputs_fracture/spine_ft/in/labels.csv; CT=daikso@192.168.0.189; RD=/media/daikso/Tony/VLM/data/CT/merlin/frac15_raw
cd $S; log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/driver.log; }
while [ ! -f runs/TRAIN_DONE ]; do [ -f runs/STOP ] && exit 0; sleep 120; done
B=$(cat runs/best_cfg.txt); RUNS="s0_$B s1_$B s2_$B"
log "score_fill start best=$B"
[ -f .tierA_deleted ] || { sed 's/^/crops\//' tierA_files.txt | xargs rm -f; touch .tierA_deleted; log "tierA crops deleted"; }
split -n l/2 -d rest_files.txt .rest_part_
for ch in val_files.txt .rest_part_00 .rest_part_01; do
  [ -f .scored_$ch ] && continue
  sed 's/.npz$//' $ch | sort > .ch_ids.txt
  while true; do  # wait until CT-rate has a status for every id of the chunk
    [ -f runs/STOP ] && exit 0
    nd=$(ssh -n $CT "cut -f1 $RD/status.tsv" | sort -u | comm -12 - .ch_ids.txt | wc -l)
    [ "$nd" -ge "$(wc -l < .ch_ids.txt)" ] && break; sleep 120
  done
  rsync -a --ignore-missing-args --files-from=$ch $CT:$RD/crops/ crops/
  log "chunk $ch: $(ls crops | wc -l) crops on disk; $(df -h ~ | tail -1 | awk '{print $4}') free"
  for r in $RUNS; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 $PY scripts/frac15_ft.py --labels $LAB --crops crops --out runs/$r \
      --score --seeds 1 --vram-gb 19 --threads 3 --workers 3 $($PY -c "import json,sys; a=json.load(open(sys.argv[1]))['args']; print('--train-from',a['train_from'],'--pool',a['pool'],'--mask',a['mask'],'--wins',a['wins'],'--ctx' if a['ctx'] else '')" runs/$r/summary.json) >> logs/score_$r.log 2>&1 || { log "score $r $ch FAILED"; exit 1; }
  done
  sed 's/^/crops\//' $ch | xargs rm -f; touch .scored_$ch; log "chunk $ch scored + deleted"
done
$PY - "$B" <<'EOF'
import sys, json, os, pandas as pd
b = sys.argv[1]; R = [f"runs/s{k}_{b}" for k in (0, 1, 2) if os.path.exists(f"runs/s{k}_{b}/summary.json")]
o = f"runs/final_{b}"; os.makedirs(o, exist_ok=True)
for f in ("oof_train.csv", "val_scores.csv"):
    ds = [pd.read_csv(f"{r}/{f}").set_index("study_id") for r in R]; d = ds[0].copy(); d["ft"] = sum(x["ft"] for x in ds) / len(ds)
    d.reset_index().to_csv(f"{o}/{f}", index=False); d.reset_index().to_csv(f, index=False)
ps = {r: {k: v for k, v in json.load(open(f"{r}/summary.json")).items() if k != "inner"} for r in R}
json.dump({"config": b, "seeds": R, "per_seed": ps}, open(f"{o}/summary.json", "w"), indent=1)
json.dump({"config": b, "seeds": R, "per_seed": ps}, open("summary.json", "w"), indent=1)
print("final from", R)
EOF
touch runs/SCORE_DONE; log "SCORE_DONE final_$B -> spine15_ft/{oof_train,val_scores,summary}"
