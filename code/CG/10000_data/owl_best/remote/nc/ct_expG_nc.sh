#!/bin/bash
# CT-rate share of nc Exp G (3 mm single-rib units; verbatim extract_bones.py --organ-map organ_map_ribs.yaml, as ct_expG.sh).
# The A6000 (a6_ncG.sh) works the sorted id list from the front; here: the second half of the sorted list (data dir of
# symlinks), GPU stage under the cg10k gpu.lock (expG_gpulock.py, nc 3 mm workers keep priority). emb_ribs synced to the
# A6000 by a6_sync_G.sh (extract_bones skips cases whose files exist).
B=~/OWL/cg10k_best; G=$B/ncG; S=$B/data/nc; CG=/media/daikso/CTDATA/cg10k; P=~/miniconda3/envs/radar/bin/python
mkdir -p $G/prep $G/masks $G/patches
ls $S/prep | grep 'npz$' | sed 's/.npz//' | sort > $G/all.txt; n=$(wc -l < $G/all.txt)
tail -n +$((n/2)) $G/all.txt > $G/ids.txt
while read i; do ln -sf $S/prep/$i.npz $G/prep/; ln -sf $S/masks/$i.nii.gz $G/masks/; done < $G/ids.txt
sed 's/$/.json/' $G/ids.txt | rsync -a --files-from=- $CG/owl/nc/patches/ $G/patches/
sed "s#^output_dir:.*#output_dir: $G#" $B/config_expG_main.yaml > $G/config.yaml
cd $B/ribs_code/src && OMP_NUM_THREADS=4 nice -n 10 $P $B/expG_gpulock.py ~/OWL/cg10k_infer/state/gpu.lock --config $G/config.yaml \
  --organ-map organ_map_ribs.yaml --emb-subdir emb_ribs > $G/expG.log 2>&1
echo "CT_EXPG_NC_DONE $(date)" >> $G/expG.log
