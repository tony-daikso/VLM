#!/bin/bash
# cg10k_nc sandbox, after all nc inputs exist: import -> GS6/gs unit -> heads -> tier A -> tier B -> biliary -> gallstones
# -> osteopenia v15 -> fracture members/stack -> final infer_best (main + "delay" = nc).  Same scripts/commands as the
# cg10k_best runs (b_driver*.sh, b_delay.sh, a6_gs.sh, frac_*.sh), CG rows never trained on.  Logs: refit/*.log
cd ~/Desktop/Tony/VLM/OWL/cg10k_nc; B=$PWD; P=~/miniconda3/envs/radar/bin/python; CT=daikso@192.168.0.189; CG=/media/daikso/CTDATA/cg10k
MW=~/Desktop/Tony/VLM/OWL/mw_exp; OF=$MW/outputs_final; RS=$OF/region_units
TS=~/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres
mkdir -p refit; L=refit; st(){ echo "$(date +%T) $*"; }
st "wait inputs"
until ssh $CT "test -f OWL/cg10k_best/NC_CT_FINAL_DONE" && grep -q NC_B_DONE ncb.log && grep -q NC15_DONE nc15.log && grep -q NC_G_DONE ncG.log \
      && grep -q SYNC_R15_DONE syncr15.log; do sleep 120; done
# ---- 0. catch-up passes: the watch loops may have ended before the last nc cases got their 3 mm masks / SSD prep copy
st "catch-up"
ssh $CT "cd $CG; comm -12 <(ls owl/nc/masks | grep 'nii.gz\$' | sed 's/.nii.gz//' | sort) <(ls nifti_3mm_nc | grep 'nii.gz\$' | sed 's/.nii.gz//' | sort)" > data/nc/avail_final.txt
comm -23 data/nc/avail_final.txt <(ls nc15/v15/emb15 | grep -v tmp | sed 's/.npz//' | sort) > data/nc/miss_v15.txt
if [ -s data/nc/miss_v15.txt ]; then sed 's/$/.nii.gz/' data/nc/miss_v15.txt | rsync -a --files-from=- $CT:$CG/nifti_3mm_nc/ data/nc/nifti15/
  bash a6_nc15.sh >> nc15_catchup.log 2>&1; fi
ssh $CT "bash OWL/cg10k_best/ct_ribs_nc.sh" > /dev/null 2>&1
rsync -a --exclude '*.tmp*' $CT:OWL/cg10k_best/nc_r15/emb15/ nc15/r15/emb15/
bash a6_nc_b.sh >> ncb_catchup.log 2>&1
bash a6_ncG.sh >> ncG_catchup.log 2>&1
st "catch-up done: v15 $(ls nc15/v15/emb15 | grep -vc tmp) r15 $(ls nc15/r15/emb15 | grep -vc tmp) avail $(wc -l < data/nc/avail_final.txt)"
# ---- 1. import
st "import"
scp -q $CT:OWL/cg10k_best/mat/cg_nc.npz mat/cg_delay.npz; scp -q $CT:OWL/cg10k_best/mat/cg_nc_bones.npz frac/cg_delay_bones.npz
mkdir -p feat/delay; scp -q "$CT:$CG/owl_feat/nc/all/*.csv" feat/delay/; scp -q $CT:$CG/owl_feat/nc/all/ids_ok.txt data/nc/ids.txt
scp -q $CT:$CG/owl/probs_nc.csv e3late/probs_delay.csv
$P -c "import pandas as pd,glob; pd.concat([pd.read_csv(f) for f in sorted(glob.glob('bemb/ath_hufeat_nc_b*.csv'))]).drop_duplicates().to_csv('bemb/ath_hufeat_delay.csv',index=False)"
ln -sfn nc data/delay
mkdir -p gs6/delay; rsync -a $CT:OWL/cg10k_best/gs6/nc/ gs6/delay/
$P cg_frac_rows.py data/nc/emb_ribs D_ frac/cg_delay_ribs3.npz > $L/ribrows.log 2>&1
$P cg_osteo_v15.py build_cg nc15/v15/emb15 D_ osteo_v15/data/cg_delay_ov.npz > $L/ov_build.log 2>&1
nice -n 10 $P cg_l1hu.py nc15/v15/seg15 cg_fx_quotes.csv osteo_v15/l1hu_nc 6 > $L/l1hu.log 2>&1
cp osteo_v15/l1hu_nc/l1hu_levels.csv osteo_v15/data/l1hu_levels_delay.csv
# ---- 2. GS6 seg + gallbladder unit (a6_gs.sh steps 2-3)
st "gs6 + gs unit"
touch gs6/delay/raw15/DONE_SENDING; mkdir -p gs6/delay/seg
OMP_NUM_THREADS=2 nice -n 10 $P gs6_seg_cg.py --inbox gs6/delay/raw15 --out gs6/delay/seg --device cuda > gs6/delay/seg.log 2>&1
( cd $MW; export MW_ROOT=.; mkdir -p $B/bcrops/gs/nc $B/bemb/gs/nc
  GS6_SEG=$B/gs6/delay/seg OMP_NUM_THREADS=1 nice -n 19 $P $OF/region_units/scripts/nu_crop.py gs $B/data/nc/ids.txt $B/bcrops/gs/nc 6 $B/data/nc/prep:$B/data/nc/masks > $B/bcrops/gs/nc.log 2>&1
  OMP_NUM_THREADS=4 nice -n 10 $P $OF/region_units/scripts/nu_extract.py --crops $B/bcrops/gs/nc --out $B/bemb/gs/nc --ts-model-dir $TS --windows windows.yaml --vram-gb 8 --threads 4 --extra stone:60:250 calc:300:800 > $B/bemb/gs/nc.log 2>&1 ) \
  && rm -rf bcrops/gs/nc; date > bemb/gs/delay.DONE
for u in hn fa asc ath rlq bd5 frp gs5; do date > bemb/$u/delay.DONE; done
# ---- 3. tier-A heads (cg_heads.py, cg_gs.py heads) on CPU, three chains in parallel
st "heads"
$P cg_heads.py prep > $L/heads_prep.log 2>&1
$P cg_heads.py prep_extra biliary >> $L/heads_prep.log 2>&1
( $P cg_heads.py run hs hs_cd sp ag ath aaa > $L/heads_run1.log 2>&1 ) &
( $P cg_heads.py run hn fa ap asc sbo op > $L/heads_run2.log 2>&1 ) &
( $P cg_heads.py run bd > $L/heads_run_bd.log 2>&1; $P cg_gs.py heads > gs/heads_driver.log 2>&1 ) &
# ---- fracture CPU members in parallel with the heads (frac_cpu_chain.sh)
$P cg_frac.py prep > $L/frac_prep.log 2>&1
( for s in 0 1 2 3 4 5 6 7 8 9; do $P cg_frac.py e3late $s; done; echo E3LATE10_DONE ) > frac/logs/chain_e3.log 2>&1 &
( for vs in "hn_b2 0" "hn_b2 1" "lv1 0" "lv1 1" "age_full 0" "age_full 1" "hn_m3 0" "r18 0" "r18 1"; do $P cg_frac.py expI $vs; done; echo EXPI_DONE ) > frac/logs/chain_expI.log 2>&1 &
# GPU members (J_add 10 seeds, L 5 seeds)
( for s in 0 1 2 3 4 5 6 7 8 9; do $P cg_frac.py v15 $s; done; echo V15_DONE ) > frac/logs/chain_v15.log 2>&1 &
( for s in 0 1 2 3 4; do $P cg_frac.py L $s; done; echo L_DONE ) > frac/logs/chain_L.log 2>&1 &
# ---- tier-B unit probes (GPU) while the CPU heads run
st "unit probes"
for c in hn fa asc ath gs; do s=ru_probe.py; [ $c = ath ] || [ $c = gs ] && s=nu_probe.py
  $P cg_bprobe.py prep $c bemb/$c/main bemb/$c/delay > $L/bprobe_${c}_prep.log 2>&1 && $P cg_bprobe.py run $c $RS/scripts/$s --device cuda --vram-gb 8 > $L/bprobe_$c.log 2>&1; done
for c in frp frpp; do $P cg_bprobe.py prep $c bemb/frp/main bemb/frp/delay > $L/bprobe_${c}_prep.log 2>&1 && $P cg_bprobe.py run $c $RS/scripts/no_probe.py --device cuda --vram-gb 8 > $L/bprobe_$c.log 2>&1; done
$P cg_brlq.py bemb/rlq/main bemb/rlq/delay > $L/brlq.log 2>&1
$P cg_biliary.py bd5 bemb/bd5/main bemb/bd5/delay > $L/bbd5.log 2>&1
st "wait heads"
until [ -f heads/R/analysis_biliary/scores_val_descriptive.csv ] && grep -q "exit" $L/heads_run1.log && [ $(grep -c exit $L/heads_run1.log) -ge 6 ] && [ $(grep -c exit $L/heads_run2.log) -ge 6 ] && grep -q "exit" gs/heads_driver.log; do sleep 60; done
$P cg_heads.py check > $L/heads_check.txt 2>&1
# ---- tier A + B stacks
st "tier A"
$P infer_best.py run main delay > $L/infer_A.log 2>&1
for S in main delay; do pre=""; [ $S = delay ] && pre=D_
  CG_PREFIX=$pre $P bstack_apply.py hydronephrosis hydronephrosis "REF＋全部單元頭" hn best/$S/hydronephrosis_heads.csv feat/$S/hydronephrosis_hu.csv > $L/bstack_hn_$S.log 2>&1
  CG_PREFIX=$pre $P bstack_apply.py free_air free_air "REF＋全部單元頭" fa best/$S/free_air_heads.csv feat/$S/free_air_air.csv > $L/bstack_fa_$S.log 2>&1
  CG_PREFIX=$pre $P bstack_apply.py ascites ascites "REF＋U_org" asc best/$S/ascites_heads.csv feat/$S/ascites_hu.csv > $L/bstack_asc_$S.log 2>&1
  CG_PREFIX=$pre $P bstack_apply.py atherosclerosis atherosclerosis "REF＋全部單元頭＋HU" ath best/$S/atherosclerosis_heads.csv feat/$S/atherosclerosis_hu.csv $RS/atherosclerosis/hu_feats.csv bemb/ath_hufeat_$S.csv > $L/bstack_ath_$S.log 2>&1
  $P cg_biliary.py stack $S > $L/bbd_stack_$S.log 2>&1
done
st "gallstones"
$P cg_gs.py g3b > gs/g3b_driver.log 2>&1; $P cg_gs.py gbx > gs/gbx_driver.log 2>&1
$P cg_gs.py stack main > gs/stack_main.log 2>&1; $P cg_gs.py stack delay > gs/stack_delay.log 2>&1
st "osteopenia v15"
$P cg_osteo_v15.py run > osteo_v15/driver.log 2>&1
st "wait fracture members"
until grep -q E3LATE10_DONE frac/logs/chain_e3.log && grep -q EXPI_DONE frac/logs/chain_expI.log && grep -q V15_DONE frac/logs/chain_v15.log && grep -q L_DONE frac/logs/chain_L.log; do sleep 60; done
$P cg_frac.py ens > frac/logs/ens.log 2>&1; $P cg_frac.py stack > frac/logs/stack.log 2>&1
st "final"
$P infer_best.py run main delay > $L/infer_final.log 2>&1
echo NC_REFIT_DONE
