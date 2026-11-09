#!/bin/bash
cd ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_ribs15
head -8405 ids_prio.txt | sort > /tmp/ribs_t0
ls emb15 | grep -v tmp | sed "s/.npz//" | sort > /tmp/ribs_e
cut -f1 failed.txt 2>/dev/null | sort -u > /tmp/ribs_f
d=$(comm -12 /tmp/ribs_t0 /tmp/ribs_e | wc -l); f=$(comm -12 /tmp/ribs_t0 /tmp/ribs_f | comm -23 - /tmp/ribs_e | wc -l)
echo "$(date +%T) tier0 done $d failed $f of 8405; total emb $(wc -l < /tmp/ribs_e)"
