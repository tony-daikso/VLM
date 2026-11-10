#!/bin/bash
# CT-rate share of CG10k Exp G (3 mm single-rib units, extract_bones.py --organ-map organ_map_ribs.yaml; verbatim A6000 src,
# reproduced 0.0 diff on 2 Merlin cases). Starts after the CT-rate rib (1.5 mm) worker has finished. shard 1/3 of main.
# GPU stages are not flock-ed by extract_bones; nc priority is kept by running at nice 10 with one process (peak ~3 GB).
B=~/OWL/cg10k_best; until [ -f /media/daikso/CTDATA/cg10k/owl_ribs15/ct_worker_done.log ]; do sleep 60; done
rsync -a /media/daikso/CTDATA/cg10k/owl/main/patches/ $B/data/main/patches/ --include="*.json" --include="*/" --exclude="*"
cd $B/ribs_code/src && OMP_NUM_THREADS=4 nice -n 10 ~/miniconda3/envs/radar/bin/python extract_bones.py --config $B/config_expG_main.yaml \
  --organ-map organ_map_ribs.yaml --emb-subdir emb_ribs --shard 1/3 > $B/logs/expG_main_s1.log 2>&1
echo done > $B/expG_ct.done
