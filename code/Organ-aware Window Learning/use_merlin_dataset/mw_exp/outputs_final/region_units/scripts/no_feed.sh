#!/bin/bash
# Exp O Mac relay (copy of Exp N nu_feed.sh): ship finished crops (A6000 + CT-rate) to the 4090 inbox in batches (only while the
# 4090 has >= 44 GB free, Exp L downloader keeps running), run the 4090 consumer (no_extract.py --consume, nice 19, VRAM cap /
# threads given), pull embeddings back to the Mac stage, push them to A6000, and delete the source crops (A6000 + CT-rate)
# whose embeddings are back (CT-rate disk is shared).
# usage: no_feed.sh UNIT VRAM_GB THREADS        stop: touch $STAGE/STOP_FEED
U=$1; VR=$2; TH=$3
A6D=Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units/$U
CTD=OWL/mw_exp/region_units_tmp/$U
G4=/root/OWL/expO/$U
STAGE="/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/expO_stage/$U"
G() { ssh -n -o ServerAliveInterval=30 -p 25966 root@tw-07.access.glows.ai "$@"; }
Gi() { ssh -o ServerAliveInterval=30 -p 25966 root@tw-07.access.glows.ai "$@"; }
TS=/root/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
mkdir -p "$STAGE/emb"; touch "$STAGE/sent_a6.txt" "$STAGE/sent_ct.txt"
while G "test -f /root/OWL/EXP_N_RUNNING"; do echo "$(date +%T) waiting for EXP_N_RUNNING"; sleep 60; done
G "mkdir -p $G4/inbox $G4/emb $G4/logs; touch /root/OWL/EXP_O_RUNNING"
echo "FEED_START $U $(date +%T)"
start_consumer() {
  if ! G "pgrep -f '[n]o_extract.py --crops $G4/inbox' >/dev/null"; then
    G "cd /root/OWL/mw_exp && OMP_NUM_THREADS=$TH MW_ROOT=. setsid nohup nice -n 19 /root/L/venv/bin/python /root/OWL/expO/scripts/no_extract.py --crops $G4/inbox --out $G4/emb --ts-model-dir $TS --windows windows.yaml --consume --done-flag $G4/inbox/DONE_SENDING --threads $TH --vram-gb $VR >> $G4/logs/extract.log 2>&1 < /dev/null & echo consumer started \$!"
  fi
}
start_consumer
send() {  # host-tag ssh-target crops-dir
  local tag=$1 host=$2 dir=$3
  ssh -n $host "ls $dir 2>/dev/null | grep -v tmp | grep npz\$" | sort > "$STAGE/list_$tag.txt"
  sort -u "$STAGE/sent_$tag.txt" -o "$STAGE/sent_$tag.txt"
  comm -23 "$STAGE/list_$tag.txt" "$STAGE/sent_$tag.txt" | head -800 > "$STAGE/batch_$tag.txt"
  local n=$(wc -l < "$STAGE/batch_$tag.txt")
  [ "$n" -eq 0 ] && return 0
  local inb=$(G "ls $G4/inbox | grep -c npz\$")
  [ "$inb" -gt 2000 ] && { echo "$(date +%T) inbox full ($inb)"; return 0; }
  local av=$(G "df -BG --output=avail /root | tail -1 | tr -dc 0-9")
  [ "${av:-0}" -lt 44 ] && { echo "$(date +%T) 4090 disk low (${av} GB free) -> wait"; return 0; }
  ssh $host "tar cf - -C $dir -T -" < "$STAGE/batch_$tag.txt" | Gi "rm -rf $G4/inbox/.in_$tag && mkdir -p $G4/inbox/.in_$tag && tar xf - -C $G4/inbox/.in_$tag && find $G4/inbox/.in_$tag -name '*.npz' -exec mv -t $G4/inbox/ {} + && rmdir $G4/inbox/.in_$tag" \
    && cat "$STAGE/batch_$tag.txt" >> "$STAGE/sent_$tag.txt" && echo "$(date +%T) sent $n from $tag"
}
pull() {
  rsync -a --exclude '*.tmp*' -e "ssh -p 25966" root@tw-07.access.glows.ai:$G4/emb/ "$STAGE/emb/"
  rsync -a "$STAGE/emb/" A6000:$A6D/emb/
  ls "$STAGE/emb" | grep npz$ | sort > "$STAGE/emb_list.txt"
  for x in a6:A6000:$A6D ct:CT-rate:$CTD; do t=${x%%:*}; r=${x#*:}; h=${r%%:*}; d=${r#*:}
    comm -12 "$STAGE/emb_list.txt" <(sort -u "$STAGE/sent_$t.txt") | comm -23 - <(sort -u "$STAGE/deleted_$t.txt" 2>/dev/null) > "$STAGE/del_$t.txt"
    [ -s "$STAGE/del_$t.txt" ] && ssh $h "cd $d/crops && xargs rm -f" < "$STAGE/del_$t.txt" && cat "$STAGE/del_$t.txt" >> "$STAGE/deleted_$t.txt"
  done
}
while [ ! -f "$STAGE/STOP_FEED" ]; do
  d1=$(ssh -n A6000 "grep -c CROP_DONE $A6D/crop_a6.log" 2>/dev/null); d2=$(ssh -n CT-rate "grep -c CROP_DONE $CTD/crop_ct.log" 2>/dev/null)
  send a6 A6000 $A6D/crops; send ct CT-rate $CTD/crops
  start_consumer
  pull
  ne=$(ls "$STAGE/emb" | grep -c npz$); echo "$(date +%T) emb $ne  crop_done a6=$d1 ct=$d2"
  if [ "$d1" = "1" ] && [ "$d2" = "1" ]; then
    l1=$(ssh -n A6000 "ls $A6D/crops | grep -v tmp | grep npz\$" | sort | comm -23 - <(sort -u "$STAGE/sent_a6.txt") | wc -l)
    l2=$(ssh -n CT-rate "ls $CTD/crops | grep -v tmp | grep npz\$" | sort | comm -23 - <(sort -u "$STAGE/sent_ct.txt") | wc -l)
    if [ "$l1" -eq 0 ] && [ "$l2" -eq 0 ]; then
      G "touch $G4/inbox/DONE_SENDING"
      while G "pgrep -f '[n]o_extract.py --crops $G4/inbox' >/dev/null"; do sleep 20; done
      pull; echo "FEED_DONE $U emb $(ls "$STAGE/emb" | grep -c npz$) $(date +%T)"; break
    fi
  fi
  sleep 60
done
