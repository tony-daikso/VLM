#!/bin/bash
# Exp O driver (Mac): units in order; each feed blocks until its embeddings are complete, then its probe+stack runs in the
# background while the next unit is embedded (extraction 6 GB / 2 threads + probe 4 GB / 2 threads = 10 GB / 4 threads).
ST="/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/expO_stage"; S="$ST/scripts"; mkdir -p "$ST"
for p in bile_duct_r2:bd free_air_r2:fa2 ascites_r2:asc2 fracture_pelvis_r2:frp,frpp:out_pel bowel_obstruction_r2:go; do
  U=${p%%:*}; C=${p#*:}
  [ -f "$ST/STOP_DRIVER" ] && break
  bash "$S/no_feed.sh" $U 6 2 > "$ST/feed_$U.log" 2>&1
  grep -h "FEED_START\|FEED_DONE" "$ST/feed_$U.log"
  bash "$S/no_post.sh" $U "$C" 4 2 > "$ST/post_$U.log" 2>&1 &
done
wait
ssh -n -p 25966 root@tw-07.access.glows.ai "rm -f /root/OWL/EXP_O_RUNNING"
echo "DRIVER_DONE $(date +%T)"
