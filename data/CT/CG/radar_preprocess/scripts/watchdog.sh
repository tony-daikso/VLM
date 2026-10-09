#!/bin/bash
# Restart run_full.sh if it dies before finishing (max 20 restarts). Once the run exits
# with code 0 (all reports done), rebuilds the final json with --finalize-only and stops.
# Also runs check_batch.py every minute (results in logs/check_batch.log).
cd /datadrive/VLM/data/CT/CG/radar_preprocess/scripts
n=0
while true; do
  sleep 60
  python3 check_batch.py > /dev/null 2>&1  # per-batch sanity check -> logs/check_batch.log
  pgrep -f "^python radar_llm_preprocess_combined" >/dev/null && continue
  last=$(grep -E "^EXIT" ../logs/full.log | tail -1)
  if [ "$last" = "EXIT 0" ]; then
    # rebuild the final json with the current code (drops model-commentary descriptions)
    /root/miniconda3/envs/vllm/bin/python radar_llm_preprocess_combined.py --finalize-only >> ../logs/full.log 2>&1
    echo "$(date) watchdog: finished, final json rebuilt" >> ../logs/watchdog.log; exit 0
  fi
  n=$((n+1)); [ $n -gt 20 ] && { echo "$(date) watchdog: too many restarts, giving up" >> ../logs/watchdog.log; exit 1; }
  # kill any orphan vLLM engine still holding the GPU
  pkill -f "VLLM::EngineCore"; sleep 10
  echo "$(date) watchdog: restart #$n (last: $last)" | tee -a ../logs/watchdog.log >> ../logs/full.log
  echo "=== restart $(date) : STRICT prompt, abdomen chest (watchdog)" >> ../logs/full.log
  setsid nohup ./run_full.sh > /dev/null 2>&1 &
  sleep 300
done
