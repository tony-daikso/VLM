#!/bin/bash
# reproduce each feature script on 8 Merlin cases (CT-rate outputs) and compare to stored csv
cd ~/OWL/cg10k_best/repro; P=~/miniconda3/envs/radar/bin/python; C=~/OWL/cg10k_best/scripts/analysis_code; O=~/OWL/mw_exp/outputs
export OMP_NUM_THREADS=1
pick(){ $P -c "
import pandas as pd,os,sys
d=pd.read_csv(\"$1\"); ids=[s for s in d.study_id if os.path.exists(\"$O/prep/\"+s+\".npz\") and os.path.exists(\"$O/masks/\"+s+\".nii.gz\")][:8]
open(\"$2\",\"w\").write(\"\n\".join(ids)+\"\n\")"; }
run(){ name=$1; ref=$2; shift 2; pick $ref ids_$name.txt; echo "== $name"; nice -n 19 "$@" > log_$name.txt 2>&1; }
run hn analysis_hydronephrosis/hu_ct.csv $P $C/hydronephrosis/hn_hu.py ids_hn.txt $O out_hn.csv 4
run ath analysis_atherosclerosis/hu_ct.csv $P $C/atherosclerosis/ath_hu.py ids_ath.txt $O/prep $O/masks out_ath.csv 4
run fa analysis_free_air/air_ct.csv $P $C/free_air/fa_air.py ids_fa.txt $O/prep $O/masks out_fa.csv 4
run asc analysis_ascites/hu_ct.csv $P $C/ascites/asc_hu.py ids_asc.txt $O out_asc.csv 4
run aaa analysis_aaa/geom_ct.csv $P $C/aaa/aaa_geom.py ids_aaa.txt $O/prep $O/masks out_aaa.csv 4
run ag analysis_absent_gb/feat_ct.csv $P $C/absent_gb/ag_feat.py ids_ag.txt $O/prep $O/masks out_ag.csv 4 x 0.3333,0.6999,0.6626
run sbo analysis_bowel_obstruction/feat_ct.csv $P $C/bowel_obstruction/sbo_feat.py ids_sbo.txt $O/prep $O/masks out_sbo.csv 4
run sp analysis_splenomegaly/geom_ct.csv $P $C/splenomegaly/sp_geom.py ids_sp.txt $O/prep $O/masks out_sp.csv 4
run op analysis_osteopenia/hu_ct.csv $P $C/osteopenia/op_hu.py ids_op.txt $O/prep $O/masks out_op.csv 4
run hs analysis_steatosis/hs_ct.csv $P $C/steatosis/hs_hu.py ids_hs.txt $O/prep $O/masks out_hs.csv 4
run ap analysis_appendicitis/hu_ct.csv $P $C/appendicitis/ap_hu.py ids_ap.txt $O/prep out_ap.csv 4
echo ALLDONE
