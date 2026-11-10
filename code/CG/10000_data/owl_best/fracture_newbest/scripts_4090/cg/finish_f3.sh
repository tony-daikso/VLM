#!/bin/bash
# when every transferred ROI has feats3 (or only persistent OOM failures remain): stop the pollers, then score CG with the
# lev28 + FT_rib saved runs on GPU0.
cd ~/Tony/cg
while true; do
  if [ -f ROI_XFER_DONE ]; then
    left=$(comm -23 <(ls roi | sed "s/.npz//" | sort) <(ls out/feats3 | grep -v tmp | grep npz | sed "s/.npz//" | sort))
    nl=$(echo -n "$left" | grep -c .); fails=$(grep -h "^FAIL" logs/feat3_poll_1b.log logs/feat3_poll_0.log logs/feat3_poll_1c.log 2>/dev/null | awk "{print \$2}" | sort -u)
    nf=$(comm -12 <(echo "$left") <(echo "$fails") | grep -c .)
    if [ "$nl" -eq "$nf" ]; then echo "$(date "+%m-%d %H:%M") feats3 complete; left=$nl (all OOM fails)" >> logs/run.log; echo "$left" > feats3_left.txt; break; fi
  fi; sleep 60
done
pkill -f "scripts/rib_feat3.py roi out/feats3"; sleep 5
echo "$(date "+%m-%d %H:%M") score lev28/ftrib start" >> logs/run.log; ~/Tony/score_cg.sh 0 lev28 ftrib; echo "$(date "+%m-%d %H:%M") score lev28/ftrib end" >> logs/run.log
