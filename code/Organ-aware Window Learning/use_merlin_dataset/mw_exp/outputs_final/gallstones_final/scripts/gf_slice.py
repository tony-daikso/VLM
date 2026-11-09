"""Slice the inputs needed for the gallstones final stacking to train/val study_ids only (test rows dropped) for CPU work on the Mac."""
import pandas as pd, os, sys
I = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final"; O = sys.argv[1]
ids = set(pd.read_csv(f"{I}/analysis_absent_gb/labels_trval.csv").query("split in [\"train\",\"val\"]").study_id)
ids &= set(pd.read_csv(f"{I}/analysis_gallstones/labels_trval.csv").query("split in [\"train\",\"val\"]").study_id) | ids
F = ["analysis_gallstones/labels_trval.csv", "analysis_gallstones/hu_a6.csv", "analysis_gallstones/hu_ct.csv", "analysis_gallstones/oof_train.csv",
     "analysis_gallstones/scores_val_descriptive.csv", "analysis_absent_gb/labels_trval.csv", "analysis_absent_gb/feat_a6.csv", "analysis_absent_gb/feat_ct.csv",
     "analysis_absent_gb/oof_train.csv", "analysis_absent_gb/scores_val_descriptive.csv", "gallstones_improve/gs1_feat3mm.csv",
     "gallstones_improve/gs3b/gs3b_oof_train.csv", "gallstones_improve/gs3b/gs3b_val.csv", "gallstones_improve/gs5/gbx_oof_train.csv",
     "gallstones_improve/gs5/gbx_val.csv", "gallstones_improve/gs6/gbx/gbx_oof_train.csv", "gallstones_improve/gs6/gbx/gbx_val.csv",
     "gallstones_improve/gs6/override_feat.csv", "gallstones_improve/final_cmp_oof.csv", "gallstones_improve/final_cmp_val.csv",
     "region_units/gallstones/stack/gbn_oof_train.csv", "region_units/gallstones/stack/gbn_val.csv",
     "region_units/gallstones/stack/oof_train.csv", "region_units/gallstones/stack/val_scores.csv"]
for f in F:
    d = pd.read_csv(f"{I}/{f}"); n = len(d); d = d[d.study_id.isin(ids)]
    if "split" in d: d = d[d.split.isin(["train", "val"])]
    os.makedirs(os.path.dirname(f"{O}/{f}"), exist_ok=True); d.to_csv(f"{O}/{f}", index=False); print(f, n, "->", len(d))
