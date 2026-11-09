#!/bin/bash
# Full std14 run on one host, resumable: re-runs std14.py (which skips parse_ok ids) up to 4 passes so
# HTTP errors / parse failures get retried. Start detached:
#   cd ~/Desktop/Tony/VLM/report_std && setsid nohup bash run_full.sh <ids_file> <out.jsonl> <url> <jobs> \
#       > logs/full_<host>.log 2>&1 < /dev/null &
IDS=$1; OUT=$2; URL=$3; JOBS=${4:-12}
cd "$(dirname "$0")"
for pass in 1 2 3 4; do
    echo "=== pass $pass $(date)"
    python3 std14.py --reports reports_std_in.json --ids "$IDS" --url "$URL" --jobs "$JOBS" --out "$OUT"
    left=$(python3 - "$IDS" "$OUT" <<'EOF'
import json, sys
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
done = set()
for l in open(sys.argv[2]):
    try:
        j = json.loads(l)
    except Exception:
        continue
    if j.get("parse_ok"):
        done.add(j["exam_id"])
print(sum(i not in done for i in ids))
EOF
)
    echo "=== left after pass $pass: $left"
    [ "$left" = "0" ] && break
done
echo "=== FINISHED $(date)"
