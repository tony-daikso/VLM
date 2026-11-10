#!/bin/bash
# CT-rate 3080 share of the nc 1.5 mm v15 (Exp J; ribs_code/src/vert15_full.py = A6000 original md5 27fa9b24...),
# started once the CT-rate rib share has caught up (ids_rev.txt < 50). GPU stages under the cg10k gpu.lock.
# Works the A6000 todo list from the END (Spark share stopped); results synced both ways by a6_sync_v15.sh.
PY=~/miniconda3/envs/radar/bin/python; LOCK=~/OWL/cg10k_infer/state/gpu.lock; CG=/media/daikso/CTDATA/cg10k
O=~/OWL/cg10k_best/nc_v15; mkdir -p $O/claims $O/emb15 $O/seg15
until [ -s ~/OWL/cg10k_best/nc_r15/ids_rev.txt ] && [ $(wc -l < ~/OWL/cg10k_best/nc_r15/ids_rev.txt) -lt 50 ]; do sleep 120; done
echo "$(date) start" >> $O/ct_worker.log
cd ~/OWL/cg10k_best/ribs_code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f $O/STOP_CT ]; do
  sync=0; [ -f $CG/nifti_3mm_nc/.SYNC_DONE ] && sync=1
  comm -12 <(ls $CG/owl/nc/masks | grep 'nii.gz$' | sed 's/.nii.gz//' | sort) <(ls $CG/nifti_3mm_nc | grep 'nii.gz$' | sed 's/.nii.gz//' | sort) > $O/avail.txt
  comm -23 $O/avail.txt <( (ls $O/emb15 | grep -v tmp | sed 's/.npz//'; awk '{print $1}' $O/failed.txt 2>/dev/null) | sort -u) | tac > $O/ids_rev.txt
  if [ ! -s $O/ids_rev.txt ]; then [ $sync = 1 ] && break; sleep 300; continue; fi
  rm -f $O/claims/*
  for w in 52 53; do
    nice -n 10 $PY ~/OWL/cg10k_best/v15_gpulock.py $LOCK vert15_full.py extract --ids $O/ids_rev.txt --out $O \
      --mask-dirs $CG/owl/nc/masks --ct-local $CG/nifti_3mm_nc --wid $w --max-cases 60 --threads 3 >> $O/ct_worker.log 2>&1 &
  done; wait
  rm -f $O/DONE_w52 $O/DONE_w53
done
echo "CT_V15_NC_DONE $(date)" >> $O/ct_worker.log
