#!/bin/bash
# Mac relay for the Spark nc v15 share: (1) push NIfTI of ids from the END of the A6000 todo list (window WIN) to Spark,
# (2) stream finished Spark emb15/seg15 to the A6000 nc15/v15 (tar | tar, nothing stored on the Mac).
SRC=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/nifti_3mm_nc; R=Desktop/Tony/VLM/OWL/cg10k_nc; S=OWL/cg10k_nc15; WIN=${WIN:-120}
T=$(mktemp -d); touch $T/relayed
while :; do
  ssh A6000 "cd $R; grep -q NC15_DONE nc15.log && echo DONE; cat data/nc/todo.txt 2>/dev/null; echo ===; ls data/nc/nifti15" > $T/a.txt 2>/dev/null || { sleep 60; continue; }
  ssh sparks "ls $S/nifti; echo ===; ls $S/out/emb15 | grep -v tmp; true" > $T/s.txt 2>/dev/null || { sleep 60; continue; }
  sed '/^===$/,$d' $T/s.txt | grep 'nii.gz$' | sed 's/.nii.gz//' | sort > $T/sp_have; sed '1,/^===$/d' $T/s.txt | sed 's/.npz//' | sort > $T/sp_done
  # relay new Spark results to the A6000
  comm -23 $T/sp_done $T/relayed > $T/new
  if [ -s $T/new ]; then
    ssh sparks "cd $S/out && tar c \$(for i in $(tr '\n' ' ' < $T/new); do echo emb15/\$i.npz; [ -f seg15/\$i.npz ] && echo seg15/\$i.npz; done)" | ssh A6000 "cd $R/nc15/v15 && tar x" && { cat $T/new $T/relayed | sort -u > $T/r2; mv $T/r2 $T/relayed; echo "$(date +%T) relayed $(wc -l < $T/new)"; }
  fi
  if grep -q '^DONE$' $T/a.txt; then ssh sparks "touch $S/END"; break; fi
  # push from the end of the todo list, skipping ids already on the A6000 NIfTI window or done on Spark
  sed '/^===$/,$d' $T/a.txt | sort > $T/todo; sed '1,/^===$/d' $T/a.txt | grep 'nii.gz$' | sed 's/.nii.gz//' | sort > $T/a6_have
  n=$((WIN - $(wc -l < $T/sp_have)))
  if [ $n -gt 20 ]; then
    comm -23 $T/todo $T/a6_have | comm -23 - $T/sp_have | comm -23 - $T/sp_done | sort -r | head -n $n | sed 's/$/.nii.gz/' > $T/push
    [ -s $T/push ] && { rsync -a --files-from=$T/push "$SRC/" sparks:$S/nifti/; echo "$(date +%T) pushed $(wc -l < $T/push) to Spark"; }
  fi
  sleep 60
done; rm -rf $T; echo "$(date +%T) MAC_SPARK_DONE"
