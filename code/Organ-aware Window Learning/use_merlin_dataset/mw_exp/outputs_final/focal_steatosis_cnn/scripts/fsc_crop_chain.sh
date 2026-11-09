#!/bin/bash
# Focal-steatosis CNN: regenerate the Exp N 'fst' 48^3 liver-site crops (nu_crop.py fst, unchanged) for the train/val ids
# missing from region_units/focal_steatosis/crops. A6000 CPU only: 3 crop processes nice 19 + 1 rsync prefetcher.
#  - ids_crop_a6.txt : prep/mask already on A6000 (outputs/prep, outputs/masks)
#  - ids_crop_ct.txt : prep/mask only on CT-rate NVMe (file reads only, ionice idle) -> streamed in 200-id batches into
#                      tmp/, cropped, then deleted.  ids lists contain train/val only (built from labels_trval.csv).
# stop: touch STOP (after the current batch); resume: rerun (existing crops / done batches are skipped).
B=$HOME/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/focal_steatosis_cnn; cd "$B" || exit 1
PY=$HOME/miniconda3/envs/radar/bin/python; CROP=$B/../region_units/focal_steatosis/scripts/nu_crop.py
O=$HOME/Desktop/Tony/VLM/OWL/mw_exp/outputs
mkdir -p crops tmp/prep tmp/masks tmp/b logs
[ -f tmp/b/SPLIT_DONE ] || { split -l 200 -d -a 3 work/ids_crop_ct.txt tmp/b/batch_; touch tmp/b/SPLIT_DONE; }
RS="ionice -c3 nice -n 19 rsync"
fetch() {
  for f in tmp/b/batch_???; do
    [ -f STOP ] && break; n=$(basename $f)
    [ -f tmp/b/$n.done ] || [ -f tmp/b/$n.ready ] && continue
    while [ $(ls tmp/prep | wc -l) -ge 400 ] && [ ! -f STOP ]; do sleep 10; done
    sed 's/$/.npz/' $f > tmp/b/$n.p; sed 's/$/.nii.gz/' $f > tmp/b/$n.m
    nice -n 19 rsync -a --rsync-path="$RS" --files-from=tmp/b/$n.p daikso@192.168.0.189:OWL/mw_exp/outputs/prep/ tmp/prep/ \
      && nice -n 19 rsync -a --rsync-path="$RS" --files-from=tmp/b/$n.m daikso@192.168.0.189:OWL/mw_exp/outputs/masks/ tmp/masks/ \
      && touch tmp/b/$n.ready || echo "fetch fail $n" >> logs/fetch.log
  done
  touch tmp/b/FETCH_END
}
rm -f tmp/b/FETCH_END
fetch > logs/fetch.log 2>&1 &
nice -n 19 $PY $CROP fst work/ids_crop_a6.txt crops 3 $O/prep:$O/masks > logs/crop_a6.log 2>&1
while true; do
  [ -f STOP ] && { echo STOPPED $(date +%T); break; }
  nb=""
  for f in tmp/b/batch_???; do n=$(basename $f); [ -f tmp/b/$n.done ] && continue; nb=$n; break; done
  [ -z "$nb" ] && { echo CROP_ALL_DONE $(date +%T); touch CROP_ALL_DONE; break; }
  if [ ! -f tmp/b/$nb.ready ]; then
    if [ -f tmp/b/FETCH_END ]; then echo "batch $nb not fetched; rerun to retry" ; break; fi
    sleep 10; continue
  fi
  nice -n 19 $PY $CROP fst tmp/b/$nb crops 3 $B/tmp/prep:$B/tmp/masks >> logs/crop_ct.log 2>&1
  (cd tmp/prep && xargs -a ../b/$nb.p rm -f); (cd tmp/masks && xargs -a ../b/$nb.m rm -f)
  touch tmp/b/$nb.done; echo "$nb done $(date +%T) crops=$(ls crops | wc -l)"
done
