#!/bin/bash
# Spark share (shard 2/3 of main) of CG10k Exp G 3 mm single-rib units; verbatim A6000 src. Data pulled from CT-rate SSD (LAN).
D=~/OWL/cg10k_ribs; mkdir -p $D/data/main
for d in prep masks; do ionice -c3 rsync -a daikso@192.168.0.189:OWL/cg10k_best/data/main/$d/ $D/data/main/$d/; done
rsync -a daikso@192.168.0.189:/media/daikso/CTDATA/cg10k/owl/main/patches/ $D/data/main/patches/ --include="*.json" --include="*/" --exclude="*"
until grep -q "SPARK RIBS DONE" $D/coord.log 2>/dev/null; do sleep 60; done
cd $D/code/src && OMP_NUM_THREADS=6 nice -n 10 ~/radar_env/bin/python extract_bones.py --config $D/config_expG_main.yaml \
  --organ-map organ_map_ribs.yaml --emb-subdir emb_ribs --shard 2/3 > $D/expG_main_s2.log 2>&1
rsync -a $D/data/main/emb_ribs/ daikso@192.168.0.189:OWL/cg10k_best/data/main/emb_ribs/ && echo done > $D/expG.done
