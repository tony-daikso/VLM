#!/bin/bash
# Mac relay: copy fst crops (train/val ids only) A6000 -> 4090 (~/OWL/fsc/crops) via tar streams; loops until the A6000
# crop chain wrote CROP_ALL_DONE and everything is copied. stop: touch $W/STOP_PUSH
W=$(cd "$(dirname "$0")/.." && pwd); mkdir -p "$W/push"; cd "$W/push"
A=Desktop/Tony/VLM/OWL/mw_exp/outputs_final
G="ssh -p 23688 glows@tw-07.access.glows.ai"
$G 'mkdir -p ~/OWL/fsc/crops ~/OWL/fsc/work'
scp -q A6000:$A/focal_steatosis_cnn/work/labels_trval.csv labels_trval.csv
cut -d, -f1 labels_trval.csv | tail -n +2 | sort > allow.txt
cat labels_trval.csv | $G 'cat > ~/OWL/fsc/work/labels_trval.csv'
while [ ! -f "$W/STOP_PUSH" ]; do
  done_flag=$(ssh A6000 "test -f $A/focal_steatosis_cnn/CROP_ALL_DONE && echo 1 || echo 0")
  ssh A6000 "cd $A && (ls region_units/focal_steatosis/crops | sed 's#^#R #'; ls focal_steatosis_cnn/crops | grep -v tmp | sed 's#^#N #')" > a6.txt
  $G 'ls ~/OWL/fsc/crops' | sort > g.txt
  awk '{sub(/\.npz$/,"",$2); print $1, $2}' a6.txt | sort -k2 | join -1 2 -2 1 - allow.txt | awk '{print $1, $2}' | sort > cand.txt
  sed 's/\.npz$//' g.txt > g2.txt
  join -v1 cand.txt g2.txt > todo.txt
  n=$(wc -l < todo.txt); echo "$(date +%T) todo $n allA6done=$done_flag"
  if [ "$n" -gt 0 ]; then
    for src in R N; do
      d=region_units/focal_steatosis/crops; [ $src = N ] && d=focal_steatosis_cnn/crops
      awk -v s=$src '$2==s{print $1".npz"}' todo.txt > t_$src.txt
      [ -s t_$src.txt ] || continue
      cat t_$src.txt | ssh A6000 "cd $A/$d && nice -n 19 tar cf - -T -" | $G 'tar xf - -C ~/OWL/fsc/crops'
    done
  elif [ "$done_flag" = 1 ]; then echo PUSH_DONE; $G 'touch ~/OWL/fsc/work/PUSH_DONE'; break
  fi
  sleep 120
done
