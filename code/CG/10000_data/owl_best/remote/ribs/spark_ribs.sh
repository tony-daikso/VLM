#!/bin/bash
# Spark share of the CG10k Exp L rib extraction (verbatim A6000 src rib15_full.py; Dataset295 md5-identical).
# Batches of 24 ids taken from the MIDDLE of the sorted id list upward (relay = start, CT-rate = end), O_EXCL claims in
# CT-rate owl_ribs15/<set>/claims. CG nc work keeps priority: no new rib batch while ~/OWL/cg10k_spark/in/nc has files.
# Per batch: rsync NIfTI + 3 mm mask from CT-rate -> 2 workers -> rsync emb15/seg15 back (stage + mv) -> delete local.
# stop: touch ~/OWL/cg10k_ribs/STOP
D=~/OWL/cg10k_ribs; PY=~/radar_env/bin/python; CT=daikso@192.168.0.189; CG=/media/daikso/CTDATA/cg10k
C(){ ssh -o BatchMode=yes $CT "$@"; }
mkdir -p $D/work
for s in main delay; do
  nif=$CG/nifti_3mm; [ $s = delay ] && nif=$CG/nifti_3mm_delay
  while [ ! -f $D/STOP ]; do
    if [ -n "$(ls -A ~/OWL/cg10k_spark/in/nc 2>/dev/null)" ]; then sleep 30; continue; fi
    B=$(C "cd $CG/owl_ribs15/$s && n=\$(wc -l < $CG/owl_feat/$s/ids_ok.txt); tail -n +\$((n/2)) $CG/owl_feat/$s/ids_ok.txt | while read i; do [ -f emb15/\$i.npz ] && continue; ( set -o noclobber; : > claims/\$i ) 2>/dev/null && { echo \$i; k=\$((k+1)); }; [ \${k:-0} -ge 24 ] && break; done")
    if [ -z "$B" ]; then   # upper half exhausted -> take the lower half
      B=$(C "cd $CG/owl_ribs15/$s && cat $CG/owl_feat/$s/ids_ok.txt | while read i; do [ -f emb15/\$i.npz ] && continue; ( set -o noclobber; : > claims/\$i ) 2>/dev/null && { echo \$i; k=\$((k+1)); }; [ \${k:-0} -ge 24 ] && break; done")
    fi
    [ -z "$B" ] && break
    W=$D/work/$s; rm -rf $W; mkdir -p $W/ct $W/masks $W/out
    echo "$B" > $W/ids.txt
    C "cd $nif && tar ch \$(sed 's/\$/.nii.gz/' <<< \"$B\")" | tar x -C $W/ct
    C "cd ~/OWL/cg10k_best/data/$s/masks && tar ch \$(sed 's/\$/.nii.gz/' <<< \"$B\")" | tar x -C $W/masks
    cd $D/code
    for w in 0 1; do PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 10 $PY src/rib15_full.py extract --ids $W/ids.txt --out $W/out \
      --mask-dirs $W/masks --ct-local $W/ct --wid $((60+w)) --threads 4 >> $D/w$w.log 2>&1 & done; wait
    C "mkdir -p $CG/owl_ribs15/$s/.stage_sp"
    (cd $W/out && tar c emb15 seg15 $( [ -f failed.txt ] && echo failed.txt )) | C "cd $CG/owl_ribs15/$s/.stage_sp && tar x && for d in emb15 seg15; do for f in \$d/*.npz; do [ -f \"\$f\" ] && mv \"\$f\" ../\$d/; done; done; [ -f failed.txt ] && cat failed.txt >> ../failed.txt && rm failed.txt; true"
    echo "$(date '+%F %T') $s batch $(echo "$B" | wc -l) done; local emb $(ls $W/out/emb15 | wc -l)" >> $D/coord.log
    rm -rf $W
  done
done
echo "$(date '+%F %T') SPARK RIBS DONE" >> $D/coord.log
