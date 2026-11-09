#!/bin/bash
S=$1; cd $S/scripts
export OL_SEED=1 OMP_NUM_THREADS=3 OL_THREADS=3
( nice -n 10 python3 ol_scheme.py A $S/out_s1 > $S/out_s1/A.stdout 2>&1 && \
  nice -n 10 python3 ol_scheme.py C $S/out_s1 > $S/out_s1/C.stdout 2>&1 && \
  nice -n 10 python3 ol_scheme.py Dmt $S/out_s1 > $S/out_s1/Dmt.stdout 2>&1 ) &
( nice -n 10 python3 ol_scheme.py B110 $S/out_s1 > $S/out_s1/B110.stdout 2>&1 ; \
  nice -n 10 python3 ol_scheme.py B130 $S/out_s1 > $S/out_s1/B130.stdout 2>&1 ; \
  nice -n 10 python3 ol_scheme.py D $S/out_s1 > $S/out_s1/D.stdout 2>&1 ) &
wait
echo ALL_SCHEMES_DONE
