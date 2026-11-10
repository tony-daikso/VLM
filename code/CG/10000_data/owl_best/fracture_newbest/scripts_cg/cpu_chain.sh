#!/bin/bash
# CG main (6,970) CPU crop chain for fracture a6x10w2_tv7 (CT-rate). Same Merlin scripts, CT = CG 3 mm NIfTI (via symlink dir).
# lane A: rib boxes -> rib_prep ROI -> rs_tile ; lane B: pe15 crops -> spine15 crops -> pelvis15 ROI -> 3 mm spine crops (FT_sp)
# resume: rerun (done ids skipped). stop: kill the lanes.
W=/media/daikso/Tony/VLM/data/CT/CG/10000_data/owl_frac_v3; M=/media/daikso/Tony/VLM/data/CT/merlin; PY=~/miniconda3/envs/radar/bin/python
CG=/media/daikso/CTDATA/cg10k; MASK=/media/daikso/Tony/OWL/cg10k_best/data/main/masks; IDS=$W/ids_main.txt; L=$W/logs
export RAW15=$W/ct_links OMP_NUM_THREADS=1
cd $W
laneA() {
  [ -f rib/rib_boxes.npz ] || { mkdir -p rib; (cd ~/OWL/cg10k_best/ribs_code && nice -n 10 $PY src/rib15_full.py boxes --ids $IDS --mask-dirs $MASK --out $W/rib/rib_boxes.npz --jobs 6) > $L/boxes.log 2>&1; }
  nice -n 10 $PY $M/rib_ft/scripts/rib_prep.py $IDS $W/rib --ct-dir $W/ct_links --seg $CG/owl_ribs15/main/seg15 --boxes $W/rib/rib_boxes.npz --jobs 8 > $L/rib_prep.log 2>&1
  echo "$(date '+%m-%d %H:%M') rib_prep done" >> $L/chain.log
  nice -n 10 $PY $M/rib_straight/scripts/rs_tile.py $W/rib/roi $W/rs --ids $IDS --jobs 8 > $L/rs_tile.log 2>&1
  echo "$(date '+%m-%d %H:%M') rs_tile done" >> $L/chain.log
}
laneB() {
  mkdir -p pe15 sp15 pv15 sp3
  nice -n 10 $PY $M/push90/pe15/scripts/pe15_crop.py $IDS $W/pe15 6 $CG/owl_v15/main/seg15 $MASK > $L/pe15_crop.log 2>&1
  echo "$(date '+%m-%d %H:%M') pe15 done" >> $L/chain.log
  nice -n 10 $PY $M/frac15_raw/scripts/spine15_crop.py $IDS $W/sp15 6 $CG/owl_v15/main/seg15 $MASK > $L/sp15_crop.log 2>&1
  echo "$(date '+%m-%d %H:%M') sp15 done" >> $L/chain.log
  nice -n 10 $PY $M/pelvis15/scripts/pv15_crop.py $IDS $W/pv15 6 $MASK > $L/pv15_crop.log 2>&1
  echo "$(date '+%m-%d %H:%M') pv15 done" >> $L/chain.log
  nice -n 10 $PY $W/spine_ft_scripts/crop_prep.py $IDS /media/daikso/Tony/OWL/cg10k_best/data/main/prep $W/sp3/pack_cg.npz 6 $MASK > $L/sp3_crop.log 2>&1
  echo "$(date '+%m-%d %H:%M') sp3 done" >> $L/chain.log
}
laneA & laneB & wait
echo "$(date '+%m-%d %H:%M') CPU_CHAIN_DONE" >> $L/chain.log
