#!/bin/bash
# Exp J cross-host claim sync (A6000 <-> CT-rate <-> Spark). Only ADDS claim files (claim = done or in progress),
# so workers on one host skip ids taken on another. Stop: touch outputs_vert15/STOP_SYNC
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_vert15
CT=daikso@192.168.0.189; SP=daikso_spark@192.168.0.95
CTD=/home/daikso/OWL/expJ_ct/out_vert15/claims; SPD=/home/daikso_spark/OWL/expJ/out_vert15/claims
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
while [ ! -f STOP_SYNC ]; do
  ls claims | sort > /tmp/cs_a6 && $SSH $CT "ls $CTD" | sort > /tmp/cs_ct && $SSH $SP "ls $SPD" | sort > /tmp/cs_sp && {
    sort -u /tmp/cs_a6 /tmp/cs_ct /tmp/cs_sp > /tmp/cs_all
    comm -23 /tmp/cs_all /tmp/cs_a6 | (cd claims && xargs -r touch)
    comm -23 /tmp/cs_all /tmp/cs_ct | $SSH $CT "cd $CTD && xargs -r touch"
    comm -23 /tmp/cs_all /tmp/cs_sp | $SSH $SP "cd $SPD && xargs -r touch"
    echo "$(date +%F_%T) a6=$(wc -l </tmp/cs_a6) ct=$(wc -l </tmp/cs_ct) sp=$(wc -l </tmp/cs_sp) all=$(wc -l </tmp/cs_all)"
  }
  sleep 240
done
