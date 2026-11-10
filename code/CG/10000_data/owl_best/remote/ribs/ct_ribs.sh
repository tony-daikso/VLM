#!/bin/bash
# CT-rate 3080 share of the CG10k Exp L rib extraction (verbatim A6000 src rib15_full.py, Dataset295 md5-identical).
# (13:30) GPU flock only around the GPU stages, 2 workers. Works the id list from the END (L40S relay takes it from the start); O_EXCL claims in owl_ribs15/<set>/claims are
# shared with the relay. Every batch (8 cases) holds the cg10k GPU flock (same file as the nc workers' --gpu-lock),
# so nc and rib GPU work never overlap (no OOM on the 10 GB card); nc workers get the GPU between batches.
# stop: touch /media/daikso/CTDATA/cg10k/owl_ribs15/STOP_CT
PY=~/miniconda3/envs/radar/bin/python; LOCK=~/OWL/cg10k_infer/state/gpu.lock; CG=/media/daikso/CTDATA/cg10k
cd ~/OWL/cg10k_best/ribs_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in main delay; do
  O=$CG/owl_ribs15/$s; mkdir -p $O/claims $O/emb15 $O/seg15; nif=$CG/nifti_3mm; [ $s = delay ] && nif=$CG/nifti_3mm_delay
  until [ -f $CG/owl_feat/$s/ids_ok.txt ]; do sleep 120; done
  tac $CG/owl_feat/$s/ids_ok.txt > $O/ids_ct_rev.txt
  while [ ! -f $CG/owl_ribs15/STOP_CT ]; do
    left=$(comm -23 <(sort $CG/owl_feat/$s/ids_ok.txt) <(ls $O/claims | sort) | wc -l)
    [ $left -eq 0 ] && break
    for w in 50 51; do   # 2 workers; the GPU flock is held only around seg/embed (rib15_gpulock.py), CPU stages overlap
      nice -n 10 $PY ~/OWL/cg10k_best/rib15_gpulock.py $LOCK src/rib15_full.py extract --ids $O/ids_ct_rev.txt --out $O \
        --mask-dirs ~/OWL/cg10k_best/data/$s/masks --ct-local $nif --wid $w --max-cases 40 --threads 3 >> $O/ct_worker.log 2>&1 &
    done; wait
    rm -f $O/DONE_w50 $O/DONE_w51
  done
done
echo "CT_RIBS_DONE $(date)" >> $CG/owl_ribs15/ct_worker_done.log
