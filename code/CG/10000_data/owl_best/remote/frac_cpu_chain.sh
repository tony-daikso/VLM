#!/bin/bash
# fracture members that need only the 3 mm bone matrix (CPU, nice 10, 4 threads each, 2 chains)
cd ~/Desktop/Tony/VLM/OWL/cg10k_best; P=~/miniconda3/envs/radar/bin/python
( for s in 0 1 2 3 4 5 6 7 8 9; do $P cg_frac.py e3late $s; done; echo E3LATE10_DONE ) > frac/logs/chain_e3.log 2>&1 &
( for vs in "hn_b2 0" "hn_b2 1" "lv1 0" "lv1 1" "age_full 0" "age_full 1" "hn_m3 0"; do $P cg_frac.py expI $vs; done; echo EXPI_DONE ) > frac/logs/chain_expI.log 2>&1 &
wait
