#!/bin/bash
S=$1; cd $S/scripts
export OMP_NUM_THREADS=3 OL_THREADS=3
( nice -n 10 python3 ol_scheme.py A $S/out > $S/out/A.stdout 2>&1 && \
  nice -n 10 python3 ol_scheme.py C $S/out > $S/out/C.stdout 2>&1 && \
  nice -n 10 python3 ol_scheme.py Dmt $S/out > $S/out/Dmt.stdout 2>&1 ) &
( nice -n 10 python3 ol_scheme.py B110 $S/out > $S/out/B110.stdout 2>&1 ; \
  nice -n 10 python3 ol_scheme.py B130 $S/out > $S/out/B130.stdout 2>&1 ; \
  nice -n 10 python3 ol_scheme.py D $S/out > $S/out/D.stdout 2>&1 ) &
wait
echo ALL_SCHEMES_DONE
