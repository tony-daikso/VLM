#!/bin/bash
# Mac -> A6000 rolling NIfTI window for a6_nc15.sh (only CG de-identified EXAMID NIfTI, internal host). Keeps <= WIN files
# of not-yet-done ids in cg10k_nc/data/nc/nifti15. Ends when a6_nc15.sh logs NC15_DONE.  usage: nohup bash mac_push_nc.sh &
SRC=/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/nifti_3mm_nc; R=Desktop/Tony/VLM/OWL/cg10k_nc; WIN=${WIN:-700}
T=$(mktemp -d)
while :; do
  ssh A6000 "cd $R; grep -q NC15_DONE nc15.log && echo DONE; cat data/nc/todo.txt 2>/dev/null; echo ===; ls data/nc/nifti15" > $T/s.txt 2>/dev/null || { sleep 60; continue; }
  grep -q '^DONE$' $T/s.txt && break
  sed '/^===$/,$d' $T/s.txt | sort > $T/todo; sed '1,/^===$/d' $T/s.txt | grep 'nii.gz$' | sed 's/.nii.gz//' | sort > $T/have
  n=$((WIN - $(wc -l < $T/have)))
  if [ $n -gt 50 ]; then
    comm -23 $T/todo $T/have | while read i; do [ -f "$SRC/$i.nii.gz" ] && echo $i.nii.gz; done | head -n $n > $T/push
    [ -s $T/push ] && { rsync -a --files-from=$T/push "$SRC/" A6000:$R/data/nc/nifti15/; echo "$(date +%T) pushed $(wc -l < $T/push)"; }
  fi
  sleep 30
done; rm -rf $T; echo "$(date +%T) MAC_PUSH_DONE"
