#!/bin/bash
# Exp L claim sync A6000 <-> Spark (add-only). Stop: touch outputs_ribs15/STOP_SYNC_SP
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_ribs15
SP=daikso_spark@192.168.0.95; SPD=/home/daikso_spark/OWL/expL/out_ribs15/claims
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
while [ ! -f STOP_SYNC_SP ]; do
  ls claims | sort > /tmp/ribs_cs_a6 && $SSH $SP "mkdir -p $SPD; ls $SPD" | sort > /tmp/ribs_cs_sp && {
    comm -13 /tmp/ribs_cs_a6 /tmp/ribs_cs_sp | (cd claims && xargs -r touch)
    comm -23 /tmp/ribs_cs_a6 /tmp/ribs_cs_sp | $SSH $SP "cd $SPD && xargs -r touch"
    echo "$(date +%F_%T) a6=$(wc -l </tmp/ribs_cs_a6) sp=$(wc -l </tmp/ribs_cs_sp)"
  }
  rsync -a --ignore-existing --exclude "*.tmp*" $SP:/home/daikso_spark/OWL/expL/out_ribs15/emb15/ emb15/; rsync -a --ignore-existing --exclude "*.tmp*" $SP:/home/daikso_spark/OWL/expL/out_ribs15/seg15/ seg15/
  sleep 120
done
