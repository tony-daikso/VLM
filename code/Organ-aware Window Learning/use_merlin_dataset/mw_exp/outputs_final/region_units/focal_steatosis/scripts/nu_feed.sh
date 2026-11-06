#!/bin/bash
# Exp N Mac relay (copy of Exp M ru_feed.sh): ship finished crops (A6000 + CT-rate) to the 4090 inbox in batches, run the
# 4090 consumer (nu_extract.py --consume, nice 19; VRAM cap / threads per consumer given here so that all Exp N
# consumers together stay <= 10 GB and <= 4 threads), pull embeddings back to the Mac stage and push them to A6000.
# usage: nu_feed.sh UNIT VRAM_GB THREADS [extra-window-spec ...]     UNIT = gallstones | atherosclerosis | focal_steatosis
# stop: touch $STAGE/STOP_FEED (then touch /root/OWL/expN/$UNIT/inbox/DONE_SENDING on the 4090)
U=$1; VR=$2; TH=$3; shift 3; EXTRA="$*"
A6D=Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units/$U
CTD=OWL/mw_exp/region_units_tmp/$U
G4=/root/OWL/expN/$U
STAGE="/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/expN_stage/$U"
G() { ssh -n -o ServerAliveInterval=30 -p 25966 root@tw-07.access.glows.ai "$@"; }
Gi() { ssh -o ServerAliveInterval=30 -p 25966 root@tw-07.access.glows.ai "$@"; }
TS=/root/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
mkdir -p "$STAGE/emb"; touch "$STAGE/sent_a6.txt" "$STAGE/sent_ct.txt"
G "mkdir -p $G4/inbox $G4/emb $G4/logs; touch /root/OWL/EXP_N_RUNNING"
start_consumer() {
  if ! G "pgrep -f '[n]u_extract.py --crops $G4/inbox' >/dev/null"; then
    EX=""; [ -n "$EXTRA" ] && EX="--extra $EXTRA"
    G "cd /root/OWL/mw_exp && OMP_NUM_THREADS=$TH MW_ROOT=. setsid nohup nice -n 19 /root/L/venv/bin/python /root/OWL/expN/scripts/nu_extract.py --crops $G4/inbox --out $G4/emb --ts-model-dir $TS --windows windows.yaml $EX --consume --done-flag $G4/inbox/DONE_SENDING --threads $TH --vram-gb $VR >> $G4/logs/extract.log 2>&1 < /dev/null & echo consumer started \$!"
  fi
}
start_consumer
send() {  # host-tag ssh-target crops-dir
  local tag=$1 host=$2 dir=$3
  ssh -n $host "ls $dir 2>/dev/null | grep -v tmp | grep npz\$" | sort > "$STAGE/list_$tag.txt"
  sort "$STAGE/sent_$tag.txt" -o "$STAGE/sent_$tag.txt"
  comm -23 "$STAGE/list_$tag.txt" "$STAGE/sent_$tag.txt" | head -800 > "$STAGE/batch_$tag.txt"
  local n=$(wc -l < "$STAGE/batch_$tag.txt")
  [ "$n" -eq 0 ] && return 0
  local inb=$(G "ls $G4/inbox | grep -c npz\$")
  [ "$inb" -gt 2500 ] && { echo "$(date +%T) inbox full ($inb)"; return 0; }
  ssh $host "tar cf - -C $dir -T -" < "$STAGE/batch_$tag.txt" | Gi "rm -rf $G4/inbox/.in_$tag && mkdir -p $G4/inbox/.in_$tag && tar xf - -C $G4/inbox/.in_$tag && find $G4/inbox/.in_$tag -name '*.npz' -exec mv -t $G4/inbox/ {} + && rmdir $G4/inbox/.in_$tag" \
    && cat "$STAGE/batch_$tag.txt" >> "$STAGE/sent_$tag.txt" && echo "$(date +%T) sent $n from $tag"
}
pull() {
  rsync -a --exclude '*.tmp*' -e "ssh -p 25966" root@tw-07.access.glows.ai:$G4/emb/ "$STAGE/emb/"
  rsync -a "$STAGE/emb/" A6000:$A6D/emb/
}
while [ ! -f "$STAGE/STOP_FEED" ]; do
  d1=$(ssh -n A6000 "grep -c CROP_DONE $A6D/crop_a6.log" 2>/dev/null); d2=$(ssh -n CT-rate "grep -c CROP_DONE $CTD/crop_ct.log" 2>/dev/null)
  send a6 A6000 $A6D/crops; send ct CT-rate $CTD/crops
  start_consumer
  pull
  ne=$(ls "$STAGE/emb" | grep -c npz$); echo "$(date +%T) emb $ne  crop_done a6=$d1 ct=$d2"
  if [ "$d1" = "1" ] && [ "$d2" = "1" ]; then
    l1=$(ssh -n A6000 "ls $A6D/crops | grep -v tmp | grep -c npz\$"); l2=$(ssh -n CT-rate "ls $CTD/crops | grep -v tmp | grep -c npz\$")
    s1=$(sort -u "$STAGE/sent_a6.txt" | wc -l); s2=$(sort -u "$STAGE/sent_ct.txt" | wc -l)
    if [ "$s1" -ge "$l1" ] && [ "$s2" -ge "$l2" ]; then
      G "touch $G4/inbox/DONE_SENDING"
      while G "pgrep -f '[n]u_extract.py --crops $G4/inbox' >/dev/null"; do sleep 20; done
      pull; echo "FEED_DONE emb $(ls "$STAGE/emb" | grep -c npz$)"; break
    fi
  fi
  sleep 60
done
