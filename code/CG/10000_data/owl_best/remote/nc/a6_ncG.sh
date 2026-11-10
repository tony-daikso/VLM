#!/bin/bash
# nc set, fracture Exp G single-rib 3 mm units: unchanged extract_bones.py --organ-map organ_map_ribs.yaml --emb-subdir
# emb_ribs (as for main/delay) on cg10k_nc/data/nc (prep from a6_nc_b.sh rounds, masks, patches/*.json from CT-rate).
# Watch: repeats until a6_nc_b.sh is done and nothing is left.
B=~/Desktop/Tony/VLM/OWL/cg10k_nc; CT=daikso@192.168.0.189; P=~/miniconda3/envs/radar/bin/python
sed "s#^output_dir:.*#output_dir: $B/data/nc#" ~/Desktop/Tony/VLM/OWL/cg10k_best/config_data_main.yaml > $B/config_data_nc.yaml
cnt(){ ls $B/data/nc/emb_ribs/bone_L400_W1800 2>/dev/null | grep -v tmp | wc -l; }
while :; do
  last=0; grep -q NC_B_DONE $B/ncb.log && last=1
  ls $B/data/nc/prep | grep 'npz$' | sed 's/.npz$/.json/' | rsync -a --files-from=- $CT:/media/daikso/CTDATA/cg10k/owl/nc/patches/ $B/data/nc/patches/ 2>/dev/null
  cd ~/Desktop/Tony/VLM/OWL/mw_exp/src && OMP_NUM_THREADS=3 nice -n 10 $P extract_bones.py --config $B/config_data_nc.yaml \
    --organ-map organ_map_ribs.yaml --emb-subdir emb_ribs >> $B/expG_nc.log 2>&1
  echo "$(date +%T) expG nc emb $(cnt)"
  [ $last = 1 ] && break; sleep 600
done
echo "NC_G_DONE $(date)"
