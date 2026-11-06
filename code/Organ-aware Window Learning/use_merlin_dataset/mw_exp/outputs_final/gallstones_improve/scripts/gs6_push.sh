#!/bin/bash
# GS6 step 2b (Mac): stream the 1.5 mm raw crops to the A6000 inbox, keeping at most ~MAXQ files queued there
# (A6000 disk is tight; gs6_seg.py --consume deletes each input after segmentation). Ends with DONE_SENDING once
# rawcrop.log has RAWCROP_DONE and every crop was sent. Stop: touch $D/STOP_PUSH
D="/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/gs6_rawcrop"
R=Desktop/Tony/VLM/OWL/mw_exp/outputs_final/gallstones_improve/gs6/inbox
MAXQ=${MAXQ:-600}
touch "$D/sent.txt"
ssh A6000 "mkdir -p ~/$R"
while [ ! -f "$D/STOP_PUSH" ]; do
  q=$(ssh A6000 "ls ~/$R | grep -c '^AC.*nii.gz\$' ; true" 2>/dev/null | tail -1)
  [[ "$q" =~ ^[0-9]+$ ]] || q=99999
  ls "$D/out" | grep '^AC.*\.nii\.gz$' | grep -v tmp | sort > "$D/have.txt"
  sort "$D/sent.txt" -o "$D/sent.txt"
  comm -23 "$D/have.txt" "$D/sent.txt" > "$D/todo.txt"
  n=$(wc -l < "$D/todo.txt")
  if [ "$n" -eq 0 ] && grep -q RAWCROP_DONE "$D/rawcrop.log"; then
    ssh A6000 "touch ~/$R/DONE_SENDING"; echo "$(date) all sent"; break
  fi
  room=$((MAXQ - q))
  if [ "$room" -gt 0 ] && [ "$n" -gt 0 ]; then
    head -n "$room" "$D/todo.txt" > "$D/batch.txt"
    # upload under a .tmp name is not possible with rsync --files-from; rsync writes to a hidden temp file and renames atomically
    if rsync -a --files-from="$D/batch.txt" "$D/out/" A6000:$R/; then
      cat "$D/batch.txt" >> "$D/sent.txt"; echo "$(date) sent $(wc -l < "$D/batch.txt") (queue was $q, left $((n - $(wc -l < "$D/batch.txt"))))"
    fi
  fi
  sleep 60
done
