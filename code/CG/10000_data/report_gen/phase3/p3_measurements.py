#!/usr/bin/env python3
"""Small interpretable measurement table for the Phase 3 prompt (runs on A6000, reads the staged copy of
cg10k_best/feat/main = features/handcrafted/*.csv). Output: report_gen_p3/meta/measurements_p3.csv (6,970 rows).
Columns (deterministic image measurements from TotalSegmentator masks on the 3 mm C+ series):
  liver_hu, spleen_hu, liver_minus_spleen_hu   steatosis_hu.csv  (liver_p50, spleen_med)
  spleen_ml, spleen_cc_cm                      splenomegaly_geom.csv (sp_ml, sp_cc_cm)
  aorta_infra_max_mm, aorta_max_mm             aaa_geom.csv (infra_eq_max, all_eq_max; equivalent diameter, mm)
  L1_core_hu                                   osteopenia_corehu.csv (L1_core_med)
  fluid_ml                                     ascites_hu.csv (peritoneal fluid estimate)
  sb_diam_max_mm                               sbo_feat.csv (small-bowel max diameter)
  kid_l_sinus_fluid, kid_r_sinus_fluid         hydronephrosis_hu.csv (renal sinus fluid voxels, side cue)
  gb_vox, gb_dense_n130                        gs_hu.csv (gallbladder voxels; voxels >130 HU inside GB)
  extra_air_deep                               free_air_air.csv (air900_extra_deep, voxels)
"""
import os

import pandas as pd

D = os.path.expanduser("~/Desktop/Tony/VLM/report_gen_p3/features/handcrafted/")
O = os.path.expanduser("~/Desktop/Tony/VLM/report_gen_p3/meta/")
SPEC = {"steatosis_hu.csv": {"liver_p50": "liver_hu", "spleen_med": "spleen_hu"},
        "splenomegaly_geom.csv": {"sp_ml": "spleen_ml", "sp_cc_cm": "spleen_cc_cm"},
        "aaa_geom.csv": {"infra_eq_max": "aorta_infra_max_mm", "all_eq_max": "aorta_max_mm"},
        "osteopenia_corehu.csv": {"L1_core_med": "L1_core_hu"},
        "ascites_hu.csv": {"fluid_ml": "fluid_ml"},
        "sbo_feat.csv": {"sb_diam_max": "sb_diam_max_mm"},
        "hydronephrosis_hu.csv": {"l_sinus_fl": "kid_l_sinus_fluid", "r_sinus_fl": "kid_r_sinus_fluid"},
        "gs_hu.csv": {"gb_vox": "gb_vox", "gb_n130": "gb_dense_n130"},
        "free_air_air.csv": {"air900_extra_deep": "extra_air_deep"}}


def main():
    out = None
    for fn, cols in SPEC.items():
        d = pd.read_csv(D + fn, usecols=["study_id", *cols]).rename(columns={"study_id": "EXAMID", **cols})
        out = d if out is None else out.merge(d, on="EXAMID", how="outer")
    out["liver_minus_spleen_hu"] = out.liver_hu - out.spleen_hu
    os.makedirs(O, exist_ok=True)
    out.round(2).to_csv(O + "measurements_p3.csv", index=False)
    print(out.shape)
    print(out.describe().T[["count", "mean", "50%", "min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
