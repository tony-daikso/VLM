#!/bin/bash
# nc set (C- series of C+ exams): 1.5 mm v15 vertebra (Exp J, Dataset292) + 1.5 mm ribs (Exp L, Dataset295) on the A6000,
# original mw_exp/src vert15_full.py / rib15_full.py (same code as the main/delay runs). Watch mode: every round takes up
# to BATCH ids that have a 3 mm mask + NIfTI, NIfTI pushed from the Mac (mac_push_nc.sh), runs 2+2 GPU workers,
# deletes the NIfTI of the round. Ends when nifti_3mm_nc/.SYNC_DONE exists and nothing is left. Disk guard 15 GB.
# stop: touch ~/Desktop/Tony/VLM/OWL/cg10k_nc/STOP15
B=~/Desktop/Tony/VLM/OWL/cg10k_nc; SRC=~/Desktop/Tony/VLM/OWL/mw_exp/src; P=~/miniconda3/envs/radar/bin/python
CT=daikso@192.168.0.189; CG=/media/daikso/CTDATA/cg10k; D=$B/data/nc; BATCH=${BATCH:-300}
mkdir -p $D/masks $D/nifti15 $B/nc15/v15 $B/nc15/r15
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RIBS=${RIBS:-0}   # (18:36) ribs moved to the CT-rate 3080 (x86, ~35 cases/min); the A6000 does v15 only
done_ids(){ for k in v15 r15; do (ls $B/nc15/$k/emb15 2>/dev/null | grep -v tmp | sed 's/.npz//'; awk '{print $1}' $B/nc15/$k/failed.txt 2>/dev/null) | sort -u > $B/nc15/$k.done; done
  if [ $RIBS = 1 ]; then comm -12 $B/nc15/v15.done $B/nc15/r15.done; else cat $B/nc15/v15.done; fi; }
k=0
while [ ! -f $B/STOP15 ]; do
  ssh $CT "cd $CG; test -f nifti_3mm_nc/.SYNC_DONE && echo SYNC > /dev/stderr; comm -12 <(ls owl/nc/masks | grep 'nii.gz$' | sed 's/.nii.gz//' | sort) <(ls nifti_3mm_nc | grep 'nii.gz$' | grep -v tmp | sed 's/.nii.gz//' | sort)" > $D/avail.txt 2> $D/sync.txt
  # NIfTI are pushed by the Mac (mac_push_nc.sh, ~17 MB/s; the CT-rate exFAT/FUSE disk gives ~2 MB/s): rolling window
  comm -23 $D/avail.txt <(done_ids) > $D/todo.txt
  comm -12 $D/todo.txt <(ls $D/nifti15 | grep 'nii.gz$' | sed 's/.nii.gz//' | sort) | head -n $BATCH > $D/round.txt
  [ $(wc -l < $D/round.txt) -lt 100 ] && [ $(wc -l < $D/todo.txt) -gt $(wc -l < $D/round.txt) ] && { sleep 60; continue; }
  if [ ! -s $D/round.txt ]; then grep -q SYNC $D/sync.txt && break; sleep 300; continue; fi
  free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ $free -lt 15 ] && { echo "$(date +%T) disk ${free}G < 15G, wait"; sleep 300; continue; }
  echo "$(date +%T) round $k: $(wc -l < $D/round.txt) ids (avail $(wc -l < $D/avail.txt))"
  comm -23 $D/round.txt <(ls $D/masks | sed 's/.nii.gz//' | sort) | sed 's/$/.nii.gz/' | rsync -a --files-from=- $CT:$CG/owl/nc/masks/ $D/masks/
  for k2 in v15 r15; do mkdir -p $B/nc15/$k2/claims; ls $B/nc15/$k2/claims | while read i; do [ -f $B/nc15/$k2/emb15/$i.npz ] || rm -f $B/nc15/$k2/claims/$i; done; done
  cd $SRC
  for w in 0 1 2; do
    OMP_NUM_THREADS=2 nice -n 5 $P vert15_full.py extract --ids $D/round.txt --out $B/nc15/v15 --mask-dirs $D/masks --ct-local $D/nifti15 --wid $w --threads 2 >> $B/nc15/v15_w$w.log 2>&1 &
    [ $RIBS = 1 ] && [ $w -lt 2 ] && OMP_NUM_THREADS=2 nice -n 5 $P rib15_full.py extract --ids $D/round.txt --out $B/nc15/r15 --mask-dirs $D/masks --ct-local $D/nifti15 --wid $w --threads 2 >> $B/nc15/r15_w$w.log 2>&1 &
  done; wait
  rm -f $B/nc15/v15/DONE_w* $B/nc15/r15/DONE_w*
  sed 's/$/.nii.gz/' $D/round.txt | (cd $D/nifti15 && xargs rm -f)
  echo "$(date +%T) round $k done: v15 $(ls $B/nc15/v15/emb15 | grep -vc tmp) r15 $(ls $B/nc15/r15/emb15 | grep -vc tmp)"
  k=$((k+1))
done
echo "NC15_DONE $(date)"
