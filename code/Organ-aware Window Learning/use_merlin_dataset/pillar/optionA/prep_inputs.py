"""Option A, step 1 (A6000, radar env, CPU): collect every non-Pillar input of the option-A analysis into <W>/in/.
Read-only on all sources; test is never read (split filter first everywhere; targets_test*.jsonl never opened).

in/cases.csv   study_id, split, y_<d> (14-class OWL binary label of labels14_trval.csv, NaN = unclear/excluded),
               fold_<d> (train: the fold of the formal OWL stack's train OOF file for that class, -1 = val),
               owl_<d> (scores_trval_prob_complete.csv probability: train = 5-fold OOF, val = full-train refit)
in/units.npz   peer unit probe (report_gen/m4/improve/probe, read only): ids/split order of its meta.npz, unit labels UK [N,2]
               (right, left kidney) / UV [N,10] (T8..L5), case labels ycase [N,2], folds fh / ff, peer scores kid_owl /
               vert_owl (train OOF, val refit), presence masks. Labels/folds come from the peer's OWN functions
               (unit_probe.case_labels / unit_labels / folds, imported read-only, no bytecode written).
in/attrs.csv   structured attributes of report_gen/targets (status == pos, stated values only): hs_grade, asc_amount,
               ath_grade (0 = mild/trace/small, 1 = moderate/severe/large), aaa_diam_mm, sp_len_mm; deterministic
               measurements (oof/out/measure_trval.csv, m4ext/m4ext_trval.csv) needed by the analysis.
usage: ~/miniconda3/envs/radar/bin/python prep_inputs.py ~/Desktop/Tony/VLM/OWL/pillar_optA
"""
import csv
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/")
MW = H + "mw_exp/"
OF = MW + "outputs_final/"
IB = OF + "integrated_best/scores/"
RU = OF + "region_units/"
RG = H + "report_gen/"
W = sys.argv[1].rstrip("/") + "/"
os.makedirs(W + "in", exist_ok=True)

D14 = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "hydronephrosis", "abdominal_aortic_aneurysm",
       "atherosclerosis", "osteopenia", "bowel_obstruction", "appendicitis", "ascites", "free_air", "gallstones",
       "biliary_ductal_dilation", "fracture"]
FOLD_SRC = {d: IB + f"{d}_oof.csv" for d in ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly",
                                             "abdominal_aortic_aneurysm", "bowel_obstruction"]}
FOLD_SRC.update({d: RU + f"{d}/stack/scores_oof.csv" for d in ["hydronephrosis", "atherosclerosis", "ascites", "free_air"]})
FOLD_SRC.update({"biliary_ductal_dilation": OF + "biliary_improve/final/per_case_train_oof_final.csv",  # BD5 folds
                 "osteopenia": OF + "osteopenia_v15/stack_oof_train.csv",
                 "gallstones": RU + "gallstones/stack/oof_train.csv",
                 "appendicitis": OF + "appendicitis_rlq_unit/out/oof_train.csv",
                 "fracture": MW + "outputs_fracture/spine_ft/in/labels.csv"})

# ---------------------------------------------------------------- cases.csv
sc = pd.read_csv(RG + "oof/out/scores_trval_prob_complete.csv")
assert set(sc.split) <= {"train", "val"}
lab = pd.read_csv(OF + "integrated_best/work/base/labels14_trval.csv")
assert set(lab.split) <= {"train", "val"}
C = sc[["study_id", "split"]].copy()
C = C.merge(lab[["study_id"] + D14].rename(columns={d: f"y_{d}" for d in D14}), on="study_id", how="left")
info = {}
for d in D14:
    f = pd.read_csv(FOLD_SRC[d])
    if "split" in f.columns:
        f = f[f.split == "train"]
    f = f[["study_id", "fold"]].drop_duplicates("study_id")
    m = C.merge(f, on="study_id", how="left")
    tr = (C.split == "train").values
    lab_tr = tr & C[f"y_{d}"].notna().values
    miss = lab_tr & m.fold.isna().values
    fb = np.array([zlib.crc32(s.encode()) % 5 for s in C.study_id])
    fold = np.where(tr, np.where(m.fold.isna(), fb, m.fold.fillna(-1)), -1).astype(int)
    C[f"fold_{d}"] = fold
    C[f"owl_{d}"] = sc[d].values
    info[d] = dict(fold_src=FOLD_SRC[d].replace(H, ""), train_labelled=int(lab_tr.sum()),
                   train_labelled_without_official_fold=int(miss.sum()),
                   owl_missing_train_labelled=int((lab_tr & sc[d].isna().values).sum()),
                   owl_missing_val_labelled=int(((~tr) & C[f"y_{d}"].notna().values & sc[d].isna().values).sum()))
    print(d, info[d], flush=True)
# fold agreement between classes (information only)
F = C[C.split == "train"][[f"fold_{d}" for d in D14]].values
agree = {D14[j]: float((F[:, j] == F[:, 0]).mean()) for j in range(len(D14))}
info["_fold_agreement_with_hepatic_steatosis"] = agree
print("fold agreement", agree)
C.to_csv(W + "in/cases.csv", index=False)

# ---------------------------------------------------------------- units.npz (peer definitions, imported read-only)
sys.path.insert(0, RG + "m4/code_f/improve")
sys.path.insert(1, RG + "m4/code_f")
import unit_probe as up  # noqa: E402

P = RG + "m4/improve/probe/"
meta = np.load(P + "meta.npz")
ids, split = meta["ids"], meta["split"]
assert set(split) <= {"train", "val"}
ycase = up.case_labels(ids, split)
UK, UV = up.unit_labels(ids, split, ycase)
fh, ff, miss = up.folds(ids, split)
ps = {r["case_id"]: r for r in csv.DictReader(open(P + "probe_scores.csv"))}
g = lambda r, k: np.nan if r is None or r[k] == "" else float(r[k])  # noqa: E731
kid_owl = np.array([[g(ps.get(c), k) for k in ("kid_right", "kid_left")] for c in ids])
vert_owl = np.array([[g(ps.get(c), k) for k in up.LEVELS] for c in ids])
np.savez(W + "in/units.npz", ids=ids, split=split, ycase=ycase, UK=UK, UV=UV, fh=fh, ff=ff, kid_owl=kid_owl,
         vert_owl=vert_owl, kid_mask=meta["kid_mask"].any(-1), vert_mask=meta["vert_mask"].any(-1))
info["_units"] = dict(n=len(ids), fold_fallback=list(map(int, miss)),
                      kid_labelled={sp: int((~np.isnan(UK[split == sp])).sum()) for sp in ("train", "val")},
                      vert_labelled={sp: int((~np.isnan(UV[split == sp])).sum()) for sp in ("train", "val")})
print(info["_units"], flush=True)

# ---------------------------------------------------------------- attrs.csv
ORD = {"hs_grade": ("hepatic_steatosis", "grade", {"mild": 0, "moderate": 1, "severe": 1}),
       "asc_amount": ("ascites", "amount", {"trace": 0, "small": 0, "moderate": 1, "large": 1}),
       "ath_grade": ("atherosclerosis", "grade", {"mild": 0, "moderate": 1, "severe": 1}),
       "sp_grade": ("splenomegaly", "grade", {"mild": 0, "moderate": 1, "severe": 1})}
NUM = {"aaa_diam_mm": ("abdominal_aortic_aneurysm", "diameter_mm"), "sp_len_mm": ("splenomegaly", "length_mm")}
rows = []
for sp in ("train", "val"):                                   # targets_test*.jsonl is never opened
    for line in open(RG + f"targets/targets_{sp}.jsonl"):
        r = json.loads(line)
        assert r["split"] == sp
        o = {"study_id": r["case_id"], "split": sp}
        for k, (d, a, mp) in ORD.items():
            x = r["diseases"][d]
            o[k] = mp.get(x["attributes"].get(a), np.nan) if x["status"] == "pos" else np.nan
        for k, (d, a) in NUM.items():
            x = r["diseases"][d]
            v = x["attributes"].get(a) if x["status"] == "pos" else None
            o[k] = float(v) if isinstance(v, (int, float)) else np.nan
        rows.append(o)
A = pd.DataFrame(rows)
ms = pd.read_csv(RG + "oof/out/measure_trval.csv")
ms = ms[ms.split.isin(["train", "val"])].drop(columns=["split"])
mx = pd.read_csv(RG + "m4ext/m4ext_trval.csv", low_memory=False)
mx = mx[mx.split.isin(["train", "val"])]
keep = [c for c in mx.columns if c.startswith(("kidney_l_", "kidney_r_", "spleen_ml", "liver_hu", "spleen_hu", "aorta_"))
        or c.endswith(("_g15_loss", "_g15_loss_a", "_g15_loss_m", "_g15_loss_p", "_g15_wedge", "_g15_crush",
                       "_g15_bicon", "_ml_nb", "_zext_nb", "_g15_hu_body"))
        or c in ("bmd_hu", "vert_levels_present", "aorta_hu_med_phase")]
A = A.merge(ms, on="study_id", how="left").merge(mx[["study_id"] + keep], on="study_id", how="left")
A.to_csv(W + "in/attrs.csv", index=False)
info["_attrs"] = {k: {sp: int(A[A.split == sp][k].notna().sum()) for sp in ("train", "val")} for k in list(ORD) + list(NUM)}
print(info["_attrs"])
json.dump(info, open(W + "in/prep_info.json", "w"), indent=1)
print("done")
