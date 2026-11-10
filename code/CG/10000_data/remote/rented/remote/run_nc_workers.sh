#!/bin/bash
# ON the rented L40S: N_W persistent 3 mm OWL workers (unchanged src/cg10k_run.py --watch, --gpu-lock) on in/nc -> out/nc.
# The Mac coordinator feeds in/nc (stage + mv), pulls results, deletes them; touches in/nc/.SYNC_DONE at the end.
# stop: touch /root/g4090/state_nc/STOP (workers exit after the current case)
N_W=${N_W:-3}; TH=${TH:-4}; RECYCLE=${RECYCLE:-300}
G=/root/g4090; PY=/root/radar_env/bin/python
cd $G/code; mkdir -p $G/logs $G/in/nc $G/out/nc $G/state_nc
rm -f $G/state_nc/STOP
for i in $(seq 0 $((N_W-1))); do
  ( while :; do
      OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY -u src/cg10k_run.py --set nc=$G/in/nc=$G/out/nc --state $G/state_nc --device cuda --threads $TH \
        --tag lnc$i --watch --stable-sec 5 --idle-exit-h 24 --gpu-lock $G/state_nc/gpu.lock --max-cases $RECYCLE \
        >> $G/logs/nc_w$i.log 2>&1 < /dev/null
      rc=$?; [ $rc -eq 10 ] || { echo "$(date '+%F %T') lnc$i exit rc=$rc"; break; }
      [ -f $G/state_nc/STOP ] && break
    done ) &
  sleep 5
done
wait
echo "$(date '+%F %T') nc workers all exited"
