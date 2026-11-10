#!/bin/bash
# CG10k Exp L (1.5 mm ribs, TS Dataset295) on the rented L40S, relayed by the Mac (the L40S cannot reach CT-rate).
# Code on the L40S = verbatim A6000 mw_exp/src (rib15_full.py, md5 178d2a4b...), Dataset295 weights md5-identical to A6000;
# Merlin 2-case reproduction on the L40S: present identical, cosine >= 0.99999 (2026-11-06 12:15).
# Per round (R ids): CT-rate symlink stage -> tar ch | ssh L40S tar x  (only de-identified CG NIfTI (EXAMID names) + 3 mm mask)
#   -> 2 workers rib15_full.py extract (--ct-local, --mask-dirs; nice 15, 2 threads) -> tar c emb15+seg15 | ssh CT-rate tar x
#   into owl_ribs15/<set>/.stage -> mv -> delete the round on the L40S.  Next round is pushed while the current one runs.
# STOP: touch owl_best/remote/ribs/STOP (finishes the current round).   usage: nohup caffeinate -i bash ribs_coord.sh &
H=$(cd "$(dirname "$0")" && pwd); LOG=$H/coord.log; R=${R:-40}
L(){ ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -p 26747 root@tw-07.access.glows.ai "$@"; }
C(){ ssh -o ConnectTimeout=20 CT-rate "$@"; }
CG=/media/daikso/CTDATA/cg10k; RB=/root/g4090/ribs
log(){ echo "$(date '+%F %T') $*" >> $LOG; }
push(){ s=$1; k=$2; f=$3   # 4 parallel tar streams CT-rate -> Mac -> L40S (single ssh stream ~15 MB/s was the bottleneck)
  nif=$CG/nifti_3mm; [ $s = delay ] && nif=$CG/nifti_3mm_delay
  L "mkdir -p $RB/in/$s/r$k/ct $RB/in/$s/r$k/masks"; cat $f | L "cat > $RB/in/$s/r$k/ids.txt"
  for j in 0 1 2 3; do awk -v j=$j "NR%4==j" $f > $f.part_$j; done
  for p in $f.part_*; do
    ( C "cd $nif && tar ch \$(sed 's/\$/.nii.gz/' -)" < $p | L "cd $RB/in/$s/r$k/ct && tar x"
      C "cd ~/OWL/cg10k_best/data/$s/masks && tar ch \$(sed 's/\$/.nii.gz/' -)" < $p | L "cd $RB/in/$s/r$k/masks && tar x" ) &
  done; wait; rm -f $f.part_*; }
run(){ s=$1; k=$2
  L "cd $RB/code && mkdir -p $RB/out/$s/logs && for w in 0 1 2 3; do PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 15 /root/radar_env/bin/python src/rib15_full.py extract --ids $RB/in/$s/r$k/ids.txt --out $RB/out/$s --mask-dirs $RB/in/$s/r$k/masks --ct-local $RB/in/$s/r$k/ct --wid \$w --threads 3 >> $RB/out/$s/logs/w\$w.log 2>&1 & done; wait; rm -f $RB/out/$s/DONE_w*"; }
pull(){ s=$1; k=$2
  C "mkdir -p $CG/owl_ribs15/$s/.stage/emb15 $CG/owl_ribs15/$s/.stage/seg15 $CG/owl_ribs15/$s/emb15 $CG/owl_ribs15/$s/seg15"
  L "cd $RB/out/$s && ls emb15 | grep -v tmp | sed 's/.npz//' > pulled.txt; tar c \$(sed 's#^#emb15/#; s#\$#.npz#' pulled.txt) \$(for i in \$(cat pulled.txt); do [ -f seg15/\$i.npz ] && echo seg15/\$i.npz; done) failed.txt 2>/dev/null" \
    | C "cd $CG/owl_ribs15/$s/.stage && tar x && for d in emb15 seg15; do for f in \$d/*.npz; do [ -f \"\$f\" ] && mv \"\$f\" ../\$d/; done; done; [ -f failed.txt ] && cat failed.txt >> ../failed.txt && rm failed.txt; ls ../emb15 | wc -l"
  L "cd $RB/out/$s && for i in \$(cat pulled.txt); do rm -f emb15/\$i.npz seg15/\$i.npz; done; rm -f failed.txt claims/*; rm -rf $RB/in/$s/r$k"; }
prepare(){ s=$1; k=$2; f=$H/round_${s}_$k.txt   # claim the next R ids (O_EXCL, shared with ct_ribs.sh / spark_ribs.sh) and push them
  C "mkdir -p $CG/owl_ribs15/$s/claims; comm -23 $CG/owl_feat/$s/ids_ok.txt <( (ls $CG/owl_ribs15/$s/emb15 2>/dev/null | sed 's/.npz//'; ls $CG/owl_ribs15/$s/claims) | sort -u) | head -n $R | (cd $CG/owl_ribs15/$s/claims && while read i; do ( set -o noclobber; : > \$i ) 2>/dev/null && echo \$i; done)" > $f
  [ -s $f ] && push $s $k $f; }
for s in main delay; do      # (13:35) round k+1 is claimed and pushed while round k runs on the L40S
  k=0; prepare $s 0
  while [ -s $H/round_${s}_$k.txt ]; do
    [ -f $H/STOP ] && { log "STOP"; exit 0; }
    log "$s round $k: $(wc -l < $H/round_${s}_$k.txt) ids"
    ( prepare $s $((k+1)) ) & pp=$!
    run $s $k; got=$(pull $s $k); log "$s round $k done; CT-rate emb15 $got"
    wait $pp; k=$((k+1))
  done
  log "$s: no more unclaimed ids"
done
log "ALL DONE"
