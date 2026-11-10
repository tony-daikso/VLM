#!/bin/bash
# resume fracture members after the 15:15 A6000 disk-full incident (outputs written in that window were deleted)
cd ~/Desktop/Tony/VLM/OWL/cg10k_best; P=~/miniconda3/envs/radar/bin/python
( for s in 8 9; do $P cg_frac.py e3late $s; done; echo E3LATE10_DONE ) > frac/logs/chain_e3b.log 2>&1 &
( for vs in "age_full 1" "hn_m3 0"; do $P cg_frac.py expI $vs; done; echo EXPI_DONE ) > frac/logs/chain_expIb.log 2>&1 &
( for s in 3 4 5 6 7 8 9; do $P cg_frac.py v15 $s; done; echo V15_DONE ) > frac/logs/chain_v15b.log 2>&1 &
wait
