#!/bin/bash
# Spark share of the nc 1.5 mm v15 vertebra extraction (verbatim A6000 vert15_full.py, md5 27fa9b24..., Dataset292).
# NIfTI are pushed by the Mac (mac_push_spark.sh, taken from the END of the A6000 todo list) into $D/nifti; 3 mm masks
# are pulled from CT-rate per batch. nc 3 mm workers keep priority (nice 10, 2 workers). Results are relayed to the A6000
# by the Mac (tar stream, no Mac disk). Local NIfTI deleted after each batch.  stop: touch ~/OWL/cg10k_nc15/STOP
D=~/OWL/cg10k_nc15; PY=~/radar_env/bin/python; CT=daikso@192.168.0.189; CG=/media/daikso/CTDATA/cg10k
mkdir -p $D/nifti $D/masks $D/out/emb15
while [ ! -f $D/STOP ]; do
  comm -23 <(ls $D/nifti | grep 'nii.gz$' | sed 's/.nii.gz//' | sort) <( (ls $D/out/emb15 | grep -v tmp | sed 's/.npz//'; awk '{print $1}' $D/out/failed.txt 2>/dev/null) | sort -u) > $D/batch.txt
  if [ ! -s $D/batch.txt ]; then [ -f $D/END ] && break; sleep 60; continue; fi
  comm -23 $D/batch.txt <(ls $D/masks | sed 's/.nii.gz//' | sort) | sed 's/$/.nii.gz/' | rsync -a --files-from=- $CT:$CG/owl/nc/masks/ $D/masks/
  cd ~/OWL/cg10k_ribs/code/src
  for w in 70 71; do PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 nice -n 10 $PY vert15_full.py extract --ids $D/batch.txt \
      --out $D/out --mask-dirs $D/masks --ct-local $D/nifti --wid $w --threads 3 >> $D/w$w.log 2>&1 & done; wait
  rm -f $D/out/DONE_w*
  sed 's/$/.nii.gz/' $D/batch.txt | (cd $D/nifti && xargs rm -f)
  echo "$(date '+%F %T') batch $(wc -l < $D/batch.txt) done; emb15 $(ls $D/out/emb15 | grep -vc tmp)" >> $D/coord.log
done
echo "SPARK_V15_DONE $(date)" >> $D/coord.log
