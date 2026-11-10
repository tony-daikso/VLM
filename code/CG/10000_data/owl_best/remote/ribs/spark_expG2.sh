#!/bin/bash
# (16:12) remaining shard-2 ids split: 1/3 on Spark, 2/3 on the faster CT-rate 3080 (gpu.lock)
D=~/OWL/cg10k_ribs
cd $D/code/src && OMP_NUM_THREADS=6 nice -n 10 ~/radar_env/bin/python extract_bones.py --config $D/config_expG_main.yaml \
  --organ-map organ_map_ribs.yaml --emb-subdir emb_ribs --ids-file /tmp/expG_s2_sp.txt > $D/expG_main_s2b.log 2>&1
rsync -a --exclude "*.tmp*" $D/data/main/emb_ribs/ daikso@192.168.0.189:OWL/cg10k_best/data/main/emb_ribs/ && echo done > $D/expG.done
