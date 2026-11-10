#!/bin/bash
# CPU fallback for feats3 OOM cases: re-run every few minutes on the current failure list (symlinked ROI subset), until finish_f3 is done
cd ~/Tony/cg; . ~/Tony/venv/bin/activate; mkdir -p roi_fail
while ! grep -q "score lev28/ftrib start" logs/run.log; do
  for i in $(grep -h "^FAIL" logs/feat3_poll_1b.log logs/feat3_poll_0.log logs/feat3_poll_1c.log 2>/dev/null | awk "{print \$2}" | sort -u); do [ -f out/feats3/$i.npz ] || ln -sf ~/Tony/cg/roi/$i.npz roi_fail/; done
  if [ -n "$(ls roi_fail)" ]; then OMP_NUM_THREADS=8 python scripts/rib_feat3_cpu.py roi_fail out/feats3 --ids /dev/null --maxst 1 --bone --nsec 24 --once --src src >> logs/feat3_cpu.log 2>&1; for f in roi_fail/*; do [ -f out/feats3/$(basename $f) ] && rm $f; done; fi
  sleep 120
done
