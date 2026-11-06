#!/bin/bash
# stage train/val prep+patches from CT-rate to Spark (test excluded), 4 gentle streams, prep before patches
set -u
O=~/OWL/mw_exp/outputs; L=~/OWL/mw_ft/logs/stage; mkdir -p $L
R=daikso@192.168.0.189:OWL/mw_exp/outputs
python3 - <<P
import csv,os
O=os.path.expanduser("~/OWL/mw_exp/outputs")
ids=[r["study_id"] for r in csv.DictReader(open(os.path.expanduser("~/OWL/data_ft/split.csv"))) if r["split"] in ("train","val")]
have={f[:-4] for f in os.listdir(O+"/prep") if f.endswith(".npz")}
need=[i for i in ids if i not in have]
print("trainval",len(ids),"have",len(ids)-len(need),"need",len(need))
for k in range(4):
    open(os.path.expanduser(f"~/OWL/mw_ft/logs/stage/need_{k}.txt"),"w").write("".join(f"{i}.npz\n" for i in need[k::4]))
    open(os.path.expanduser(f"~/OWL/mw_ft/logs/stage/needj_{k}.txt"),"w").write("".join(f"{i}.json\n" for i in need[k::4]))
P
for k in 0 1 2 3; do
  ( rsync -a --ignore-existing --files-from=$L/need_$k.txt $R/prep/ $O/prep/ > $L/rsync_prep_$k.log 2>&1
    rsync -a --ignore-existing --files-from=$L/needj_$k.txt $R/patches/ $O/patches/ > $L/rsync_patch_$k.log 2>&1
    echo done > $L/DONE_$k ) &
done
wait
echo all > $L/STAGE_DONE
