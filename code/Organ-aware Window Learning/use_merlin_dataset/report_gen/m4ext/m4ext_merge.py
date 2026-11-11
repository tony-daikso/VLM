"""M4 extension: merge deterministic measurements into one train+val table + coverage + sanity AUROCs (train).
Test is never read: every input is filtered to split in {train, val} right after reading.

Inputs (A6000 ~/Desktop/Tony/VLM/OWL/..., paths below are relative to OWL/):
  data/split.csv
  report_gen/m4ext/parts/units_{a6,ct}.csv            m4ext_units.py (3 mm TS masks + prep HU; geometry, HU, z ranges)
  mw_exp/outputs_final/analysis_hydronephrosis/hu_{a6,ct}.csv   hn_hu.py (per-side fluid / sinus / fluid-CC features)
  mw_exp/outputs_vert15/final_eval/geom15/geom15_levels.csv     1.5 mm per-vertebra heights / Genant loss / body HU
  mw_exp/outputs_final/osteoporosis_hu/labels_hu_trval.csv      1.5 mm trabecular ROI HU at L1 / L2 (+ rule flags)
Sanity labels (train only, never written to the table):
  data/labels_disease14.csv (osteopenia, fracture), analysis_hydronephrosis/labels_trval.csv (hn_side),
  report_gen/m4/unit_labels/unit_labels_train.jsonl (per-kidney hydronephrosis idx 3, per-vertebra fracture idx 13)

bmd_hu (image-only, usable at inference): L1 trabecular ROI median HU if L1 valid and no image-side fracture
(geom15 middle-height loss >= .25) and no image-side sclerosis (focal / >400 / +100 vs neighbours); else L2 under the
same image-only rules; NaN if both fail or HU-cement (>= 20 voxels > 1500 HU in L1/L2).  The official label
osteo_hu_label (column 'hu' of labels_hu_trval) additionally uses report flags (report-mentioned L1 fracture /
sclerosis / cement) and must NOT be shown to the model at val/test time; it is kept for reference only.
usage: python m4ext_merge.py OWL_ROOT OUT_DIR"""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

O, OUT = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/"
VERT = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]

spl = pd.read_csv(O + "data/split.csv"); spl = spl[spl.split.isin(["train", "val"])].set_index("study_id")["split"]
ids = spl.index


def trval(df, key="study_id"):
    return df[df[key].isin(ids)]


# 1. unit geometry / HU / z ranges
U = pd.concat([pd.read_csv(OUT + f"parts/units_{h}.csv") for h in ("a6", "ct")])
U = trval(U).drop_duplicates("study_id").set_index("study_id")

# 2. per-side kidney fluid features (hn_hu); TS kidney_left = patient's left
H = pd.concat([pd.read_csv(O + f"mw_exp/outputs_final/analysis_hydronephrosis/hu_{h}.csv") for h in ("a6", "ct")])
H = trval(H).drop_duplicates("study_id").set_index("study_id")
HK = ["fl_in", "sinus_vox", "sinus_fl", "shell_fl", "flcc", "flcc_out", "flcc_ap_mm", "flcc_tr_mm", "flcc_z_mm",
      "n300_in", "sinus_n300", "shell_n1000"]
Hk = pd.DataFrame(index=H.index)
for s, u in (("l", "kidney_l"), ("r", "kidney_r")):
    for c in HK:
        Hk[f"{u}_{c}"] = H[f"{s}_{c}"]
    Hk[f"{u}_flcc_ml"] = H[f"{s}_flcc"] * 0.027          # 3 mm isotropic voxel = 0.027 ml
    Hk[f"{u}_sinus_fl_ml"] = H[f"{s}_sinus_fl"] * 0.027
Hk["aorta_hu_med_phase"] = H["aorta_hu_med"]
Hk["bladder_n200"] = H["bl_n200"]

# 3. per-vertebra 1.5 mm geometry
G = pd.read_csv(O + "mw_exp/outputs_vert15/final_eval/geom15/geom15_levels.csv")
G = trval(G, "id")
GC = ["Ha", "Hm", "Hp", "vol_ml", "hu_body", "wedge", "bicon", "crush", "loss_a", "loss_m", "loss_p", "loss", "vol_ratio"]
Gw = G.pivot_table(index="id", columns="level", values=GC, aggfunc="first")
Gw.columns = [f"{lv}_g15_{c}" for c, lv in Gw.columns]
Gw = Gw[[f"{lv}_g15_{c}" for lv in VERT for c in GC if f"{lv}_g15_{c}" in Gw.columns]]

# 4. osteoporosis HU (1.5 mm trabecular ROI); image-only level choice
B = trval(pd.read_csv(O + "mw_exp/outputs_final/osteoporosis_hu/labels_hu_trval.csv")).set_index("study_id")
def ok(L):
    scl = B[[f"{L}_scl_focal", f"{L}_scl_abs", f"{L}_scl_rel"]].max(axis=1)
    return B[f"{L}_valid"].astype(str).eq("True") & (B[f"{L}_frac_geo"] == 0) & (scl == 0)
cem = B.cement_hu == 1
lvl = np.where(cem, "", np.where(ok("L1"), "L1", np.where(ok("L2"), "L2", "")))
Bo = pd.DataFrame(index=B.index)
Bo["bmd_level"] = lvl
Bo["bmd_hu"] = np.where(lvl == "L1", B.L1_hu, np.where(lvl == "L2", B.L2_hu, np.nan))
Bo["bmd_cat3"] = pd.cut(Bo.bmd_hu, [-1e9, 110, 160, 1e9], labels=["osteoporosis_le110", "osteopenia_110_160", "normal_gt160"]).astype(str).replace("nan", "")
Bo["bmd_L1_roi_hu"] = B.L1_hu.where(B.L1_valid.astype(str).eq("True"))
Bo["bmd_L2_roi_hu"] = B.L2_hu.where(B.L2_valid.astype(str).eq("True"))
Bo["bmd_cement_hu"] = B.cement_hu
Bo["bmd_roi_src"] = B.roi_src
Bo["osteo_hu_label"] = B["hu"]          # REFERENCE ONLY (uses report flags) - do not put in prompts

T = pd.DataFrame(index=ids); T.index.name = "study_id"; T["split"] = spl
T = T.join(Bo).join(U).join(Hk).join(Gw)
# z order of vertebrae (3 mm) -> which levels are present
T["vert_levels_present"] = [",".join(v for v in VERT if r.get(f"{v}_vox", 0) >= 20 and r.get(f"{v}_cut", 1) == 0)
                            for _, r in T[[c for v in VERT for c in (f"{v}_vox", f"{v}_cut")]].iterrows()]
T.to_csv(OUT + "m4ext_trval.csv")

# coverage
cov = {}
for c in T.columns:
    if c in ("split",):
        continue
    s = T[c]
    nn = s.notna() & (s.astype(str) != "")
    cov[c] = {sp: round(float(nn[T.split == sp].mean()), 4) for sp in ("train", "val")}
json.dump(cov, open(OUT + "coverage.json", "w"), indent=1)

# ---------------- sanity (train only) ----------------
tr = T[T.split == "train"]
res = {}
def auc(y, s):
    m = y.notna() & s.notna()
    y, s = y[m], s[m]
    if y.nunique() < 2:
        return None
    return dict(auc=round(float(roc_auc_score(y, s)), 4), n=int(m.sum()), pos=int(y.sum()))

lab = pd.read_csv(O + "data/labels_disease14.csv").set_index("study_id").reindex(tr.index)
res["osteopenia_vs_-bmd_hu"] = auc(lab.osteopenia, -tr.bmd_hu)
res["osteopenia_vs_-L1_roi_hu"] = auc(lab.osteopenia, -tr.bmd_L1_roi_hu)
res["osteopenia_vs_-osteo_hu_label(ref)"] = auc(lab.osteopenia, -tr.osteo_hu_label)
res["agree_bmd_hu_vs_label_hu"] = dict(n=int((tr.bmd_hu.notna() & tr.osteo_hu_label.notna()).sum()),
                                       exact=round(float((tr.bmd_hu == tr.osteo_hu_label)[tr.osteo_hu_label.notna() & tr.bmd_hu.notna()].mean()), 4))
for k in ("fracture",):
    s = tr[[f"{v}_g15_loss_m" for v in VERT if f"{v}_g15_loss_m" in tr]].max(axis=1)
    res["fracture_vs_max_g15_loss_m"] = auc(lab[k], s)
    s = tr[[f"{v}_g15_loss" for v in VERT if f"{v}_g15_loss" in tr]].max(axis=1)
    res["fracture_vs_max_g15_loss"] = auc(lab[k], s)
    s = -tr[[f"{v}_ml_nb" for v in VERT]].min(axis=1)
    res["fracture_vs_-min_3mm_ml_nb"] = auc(lab[k], s)

# per-unit labels from the peer's unit_labels (train): kidney (disease idx 3), vertebra (idx 13)
kid, ver = [], []
for line in open(O + "report_gen/m4/unit_labels/unit_labels_train.jsonl"):
    r = json.loads(line)
    if r["case_id"] not in tr.index:
        continue
    for u, pairs in r["units"].items():
        for di, y in pairs:
            if di == 3 and u in ("organ3/kidney_left", "organ3/kidney_right"):
                kid.append((r["case_id"], "kidney_l" if u.endswith("left") else "kidney_r", y))
            if di == 13 and u.startswith("vert15/"):
                ver.append((r["case_id"], u.split("/")[1], y))
kid = pd.DataFrame(kid, columns=["id", "side", "y"]).drop_duplicates(["id", "side"])
ver = pd.DataFrame(ver, columns=["id", "lvl", "y"]).drop_duplicates(["id", "lvl"])
for c in ("lowhu_frac", "lowhu_ml", "sinus_fl_ml", "flcc_ml", "flcc_ap_mm", "hu_mean", "ml"):
    vals = pd.Series([tr.at[i, f"{s}_{c}"] if f"{s}_{c}" in tr else np.nan for i, s in zip(kid.id, kid.side)], dtype=float)
    res[f"kidney_unit_hn_vs_{c}"] = auc(kid.y.reset_index(drop=True), vals)
# laterality among unilateral positives
hn = pd.read_csv(O + "mw_exp/outputs_final/analysis_hydronephrosis/labels_trval.csv").set_index("study_id").reindex(tr.index)
uni = hn.hn_side.isin(["left", "right"]) & (hn.y_hn == 1)
for c in ("flcc_ml", "sinus_fl_ml", "lowhu_ml", "lowhu_frac"):
    a, b = tr[f"kidney_l_{c}"], tr[f"kidney_r_{c}"]
    m = uni & a.notna() & b.notna() & (a != b)
    pred = np.where(a > b, "left", "right")
    res[f"laterality_unilateral_argmax_{c}"] = dict(acc=round(float((pred[m.values] == hn.hn_side[m].values).mean()), 4),
                                                   n=int(m.sum()), n_unilateral=int(uni.sum()))
for c in ("g15_loss_m", "g15_loss", "g15_crush", "g15_wedge", "g15_vol_ratio", "ml_nb", "zext_nb", "hu_med", "g15_hu_body"):
    vals = pd.Series([tr.at[i, f"{l}_{c}"] if f"{l}_{c}" in tr else np.nan for i, l in zip(ver.id, ver.lvl)], dtype=float)
    sign = -1 if c in ("g15_crush", "g15_wedge", "g15_vol_ratio", "ml_nb", "zext_nb") else 1
    res[f"vertebra_unit_fx_vs_{'-' if sign < 0 else ''}{c}"] = auc(ver.y.reset_index(drop=True), sign * vals)
# top-1 level among train cases with exactly one positive vertebra
pos = ver[ver.y == 1].groupby("id").lvl.agg(list)
one = pos[pos.apply(len) == 1]
L = tr.loc[one.index, [f"{v}_g15_loss" for v in VERT]]
pick = L.idxmax(axis=1).str.split("_").str[0]
okm = L.notna().any(axis=1)
res["fx_level_top1_argmax_g15_loss_single_level_cases"] = dict(
    acc=round(float((pick[okm] == one[okm].str[0]).mean()), 4), n=int(okm.sum()))
json.dump(res, open(OUT + "sanity_train.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
print("table", T.shape)
