#!/bin/bash
# One-shot status of the full run: progress / ETA, process, watchdog restarts, batch checks.
cd /datadrive/VLM/data/CT/CG/radar_preprocess/logs
start=$(grep -n "STRICT prompt" full.log | tail -1 | cut -d: -f1)
echo "== 進度";       tail -n +${start:-1} full.log | grep -E "^\[combined\] [0-9]+/" | tail -1
echo "== 程式";       pgrep -f "^python radar_llm_preprocess_combined" >/dev/null && echo "執行中" || echo "⚠️ 未執行（$(grep -E '^EXIT' full.log | tail -1)）"
pgrep -f "bash ./watchdog.sh" >/dev/null && echo "watchdog 執行中" || echo "⚠️ watchdog 未執行"
echo "== watchdog 重啟紀錄"; [ -s watchdog.log ] && tail -5 watchdog.log || echo "（無）"
echo "== 批次檢查：$(grep -c '^OK' check_batch.log) 批 OK，$(grep -c '^WARN' check_batch.log) 批 WARN"
grep -A4 '^WARN' check_batch.log | tail -20
echo "== 最近一批";  grep -E '^(OK|WARN)' check_batch.log | tail -1
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
