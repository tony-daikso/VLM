#!/bin/bash
# CT-rate 3080 share of the nc 1.5 mm ribs (Exp L; verbatim rib15_full.py, Dataset295, as ct_ribs.sh for main/delay).
# GPU stages hold ~/OWL/cg10k_infer/state/gpu.lock (rib15_gpulock.py) -> never overlap with the nc 3 mm workers (peak ~2.2 GB).
# Works the id list from the END; the A6000 (a6_nc15.sh) works from the start; emb15 dirs are synced both ways
# (a6_sync_r15.sh), and rib15_full skips ids whose emb15 exists.  Out on the CT-rate SSD.  stop: touch $O/STOP_CT
PY=~/miniconda3/envs/radar/bin/python; LOCK=~/OWL/cg10k_infer/state/gpu.lock; CG=/media/daikso/CTDATA/cg10k
O=~/OWL/cg10k_best/nc_r15; mkdir -p $O/claims $O/emb15 $O/seg15
cd ~/OWL/cg10k_best/ribs_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f $O/STOP_CT ]; do
  sync=0; [ -f $CG/nifti_3mm_nc/.SYNC_DONE ] && sync=1
  comm -12 <(ls $CG/owl/nc/masks | grep 'nii.gz$' | sed 's/.nii.gz//' | sort) <(ls $CG/nifti_3mm_nc | grep 'nii.gz$' | sed 's/.nii.gz//' | sort) > $O/avail.txt
  comm -23 $O/avail.txt <( (ls $O/emb15 | grep -v tmp | sed 's/.npz//'; awk '{print $1}' $O/failed.txt 2>/dev/null) | sort -u) | tac > $O/ids_rev.txt
  if [ ! -s $O/ids_rev.txt ]; then [ $sync = 1 ] && break; sleep 300; continue; fi
  rm -f $O/claims/*
  for w in 50 51; do
    nice -n 10 $PY ~/OWL/cg10k_best/rib15_gpulock.py $LOCK src/rib15_full.py extract --ids $O/ids_rev.txt --out $O \
      --mask-dirs $CG/owl/nc/masks --ct-local $CG/nifti_3mm_nc --wid $w --max-cases 60 --threads 3 >> $O/ct_worker.log 2>&1 &
  done; wait
  rm -f $O/DONE_w50 $O/DONE_w51
done
echo "CT_RIBS_NC_DONE $(date)" >> $O/ct_worker.log
