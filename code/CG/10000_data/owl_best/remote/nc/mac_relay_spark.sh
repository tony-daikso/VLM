#!/bin/bash
# (11-07, after the A6000 reboot) relay-only successor of mac_push_spark.sh: Spark now pulls its own NIfTI from CT-rate
# (spark_feed_nc.sh). Streams new Spark emb15/seg15 to the A6000 nc15/v15 (tar | tar, nothing stored on the Mac) and
# appends Spark failed ids to the A6000 v15 failed.txt so a6_nc15.sh counts them done. Ends after SPARK_V15_DONE.
R=Desktop/Tony/VLM/OWL/cg10k_nc; S=OWL/cg10k_nc15
T=$(mktemp -d)
ssh A6000 "ls $R/nc15/v15/emb15 | grep -v tmp | sed 's/.npz//'" | sort > $T/relayed || exit 1
while :; do
  fin=0; ssh sparks "grep -q SPARK_V15_DONE $S/coord.log" && fin=1
  ssh sparks "ls $S/out/emb15 | grep -v tmp | sed 's/.npz//'" > $T/s.txt 2>/dev/null || { sleep 60; continue; }
  sort $T/s.txt > $T/sp_done; comm -23 $T/sp_done $T/relayed | head -n 200 > $T/new
  if [ -s $T/new ]; then
    ssh sparks "cd $S/out && tar c \$(for i in $(tr '\n' ' ' < $T/new); do echo emb15/\$i.npz; [ -f seg15/\$i.npz ] && echo seg15/\$i.npz; done)" | ssh A6000 "cd $R/nc15/v15 && tar x" && { cat $T/new $T/relayed | sort -u > $T/r2; mv $T/r2 $T/relayed; echo "$(date +%T) relayed $(wc -l < $T/new)"; }
  fi
  ssh sparks "cat $S/out/failed.txt 2>/dev/null" > $T/sf.txt
  if [ -s $T/sf.txt ]; then
    ssh A6000 "cat $R/nc15/v15/failed.txt 2>/dev/null" | awk '{print $1}' | sort -u > $T/af
    awk 'NR==FNR{a[$1];next} !($1 in a)' $T/af $T/sf.txt > $T/fnew
    [ -s $T/fnew ] && ssh A6000 "cat >> $R/nc15/v15/failed.txt" < $T/fnew && echo "$(date +%T) failed ids appended $(wc -l < $T/fnew)"
  fi
  [ $fin = 1 ] && [ ! -s $T/new ] && break
  sleep 60
done; rm -rf $T; echo "$(date +%T) MAC_RELAY_DONE"
