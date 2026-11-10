#!/bin/bash
# usage: worker.sh GPU JOBFILE ; one job at a time; done = runs/<name>/summary_full.json ; resumable (per-seed models skipped); stop: touch ~/Tony/STOP
cd ~/Tony; . venv/bin/activate; g=$1
while IFS="|" read -r name cmd; do
  [ -z "$name" ] && continue; [ -f STOP ] && break
  [ -f runs/$name/summary_full.json ] && continue
  for try in 1 2; do
    mkdir -p runs/$name logs/$(dirname $name)
    echo "$(date "+%m-%d %H:%M") gpu$g start $name try $try" >> logs/queue.log
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 python $cmd >> logs/$name.log 2>&1
    rc=$?; echo "$(date "+%m-%d %H:%M") gpu$g end $name rc=$rc" >> logs/queue.log
    [ -f runs/$name/summary_full.json ] && break
  done
done < $2
echo "$(date "+%m-%d %H:%M") gpu$g QUEUE_DONE" >> logs/queue.log
