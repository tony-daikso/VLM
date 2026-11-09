"""影像骨質疏鬆 (HU-defined finding) -- deterministic labels. Rules: ../label_rules.md (user spec 2026-11-06).

Inputs (all filtered to train/val immediately after reading; test never read):
  roi_levels.csv      oh_roi.py output (trabecular ROI per level, 1.5 mm seg15)        -> primary ROI source
  geom15_levels.csv   geom15 hu_body (mid-sagittal body eroded 2 px) + Genant loss     -> fracture rule; HU fallback
                      source ONLY for cases whose seg15 file is not available locally
  report_flags_trval.csv  oh_report_flags.py (cement / L1,L2 fracture / sclerosis mentions)
Rules (fixed constants below; chosen from label-free train distributions, never tuned on labels):
  valid(level)    : ROI status ok (>= 50 trabecular voxels, median in [-300, 1200])
  fracture(L)     : report mentions fracture/compression/collapse at L, OR geom15 MIDDLE-height loss(L) >= 0.25
                    (Genant grade >= 2 biconcave/crush vs neighbours; anterior/posterior band heights were too noisy:
                    'max band loss >= .25' flagged 18 % of train L1, PPV vs report-L1-fracture .11)
  sclerosis(L)    : ROI frac(>600 HU) >= 0.05 (focal sclerotic lesion), OR ROI median > 400 HU (diffuse),
                    OR ROI median - mean(valid neighbour medians) >= 100 HU, OR report sclerosis mention at L
  cement (case)   : report vertebroplasty/kyphoplasty/spinal cement, OR >= 20 body voxels > 1500 HU in L1 or L2
                    (whole body footprint incl. cortex, sagittal columns xc+-3)  -> case EXCLUDED
  level choice    : L1 if valid & no fracture & no sclerosis; else L2 if valid & no fracture & no sclerosis;
                    else EXCLUDED (no valid level)
  hu              : ROI median HU of the chosen level (continuous target)
  y110 = hu <= 110 (陽性, high-specificity, primary); y160 = hu <= 160 (篩檢陽性, secondary)
  cat3 = 骨質疏鬆 (<=110) / 骨量減少 (110 < hu <= 160) / 正常 (> 160)
usage: python oh_labels.py WORK_DIR GEOM15_LEVELS_CSV OUT_CSV"""
import sys, numpy as np, pandas as pd

W, GEOM, OUT = sys.argv[1].rstrip("/") + "/", sys.argv[2], sys.argv[3]
FRAC_LOSS, SCL_F600, SCL_ABS, SCL_REL, CEM_N1500 = 0.25, 0.05, 400.0, 100.0, 20
T_POS, T_SCREEN = 110.0, 160.0

spl = pd.read_csv(W + "split.csv").set_index("study_id")["split"]
spl = spl[spl.isin(["train", "val"])]
ids = list(spl.index)
roi = pd.read_csv(W + "roi_levels.csv"); roi = roi[roi.id.isin(spl.index)]
geo = pd.read_csv(GEOM, usecols=["id", "level", "hu_body", "loss_m"]); geo = geo[geo.id.isin(spl.index)]
rep = pd.read_csv(W + "report_flags_trval.csv"); rep = rep[rep.split.isin(["train", "val"])].set_index("study_id")

LV = ["T12", "L1", "L2", "L3"]
have_roi = set(roi.id)
R = {c: roi.pivot(index="id", columns="level", values=c).reindex(index=ids, columns=LV) for c in
     ("status", "med", "f600", "body_n1500")}
G = {c: geo.pivot(index="id", columns="level", values=c).reindex(index=ids, columns=LV) for c in ("hu_body", "loss_m")}

df = pd.DataFrame(index=pd.Index(ids, name="study_id")); df["split"] = spl
df["roi_src"] = np.where(df.index.isin(have_roi), "seg15_roi", np.where(G["hu_body"][["L1", "L2"]].notna().any(axis=1),
                                                                         "geom15_fallback", "none"))
src = df.roi_src.values
# per-level HU and validity (geom15 fallback: hu_body, valid if present)
med = np.where((src == "seg15_roi")[:, None], R["med"].values, G["hu_body"].values)
valid = np.where((src == "seg15_roi")[:, None], (R["status"].values == "ok"), ~np.isnan(G["hu_body"].values))
med = np.where(valid, med, np.nan)
f600 = np.where((src == "seg15_roi")[:, None], R["f600"].fillna(0).values, 0.0)
n1500 = np.where((src == "seg15_roi")[:, None], R["body_n1500"].fillna(0).values, 0.0)
li = {l: i for i, l in enumerate(LV)}
for L in ("L1", "L2"):
    i = li[L]
    with np.errstate(all="ignore"):
        nb = np.nanmean(np.c_[med[:, i - 1], med[:, i + 1]], axis=1)
    df[f"{L}_valid"] = valid[:, i]
    df[f"{L}_hu"] = med[:, i]
    df[f"{L}_frac_rep"] = rep[f"{L}_frac"].reindex(ids).fillna(0).astype(int).values
    df[f"{L}_frac_geo"] = (G["loss_m"][L].values >= FRAC_LOSS).astype(int)
    df[f"{L}_frac"] = (df[f"{L}_frac_rep"] | df[f"{L}_frac_geo"]).astype(int)
    df[f"{L}_scl_focal"] = (f600[:, i] >= SCL_F600).astype(int)
    df[f"{L}_scl_abs"] = (med[:, i] > SCL_ABS).astype(int)
    df[f"{L}_scl_rel"] = ((med[:, i] - nb) >= SCL_REL).astype(int)
    df[f"{L}_scl_rep"] = rep[f"{L}_scl"].reindex(ids).fillna(0).astype(int).values
    df[f"{L}_scl"] = df[[f"{L}_scl_focal", f"{L}_scl_abs", f"{L}_scl_rel", f"{L}_scl_rep"]].max(axis=1)
    df[f"{L}_cem_hu"] = (n1500[:, i] >= CEM_N1500).astype(int)
df["cement_rep"] = rep["cement"].reindex(ids).fillna(0).astype(int).values
df["cement_hu"] = (df.L1_cem_hu | df.L2_cem_hu).astype(int)
df["cement"] = (df.cement_rep | df.cement_hu).astype(int)
df["hardware_rep"] = rep["hardware"].reindex(ids).fillna(0).astype(int).values

ok1 = df.L1_valid & (df.L1_frac == 0) & (df.L1_scl == 0)
ok2 = df.L2_valid & (df.L2_frac == 0) & (df.L2_scl == 0)
df["level_used"] = np.where(df.cement == 1, "", np.where(ok1, "L1", np.where(ok2, "L2", "")))
df["fallback_reason"] = ""
fb = (df.level_used == "L2")
df.loc[fb & ~df.L1_valid, "fallback_reason"] = "L1_invalid"
df.loc[fb & df.L1_valid & (df.L1_frac == 1), "fallback_reason"] = "L1_fracture"
df.loc[fb & df.L1_valid & (df.L1_frac == 0) & (df.L1_scl == 1), "fallback_reason"] = "L1_sclerosis"
df["exclude"] = np.where(df.cement == 1, "cement", np.where(df.level_used == "", "no_valid_level", ""))
df.loc[df.roi_src == "none", "exclude"] = np.where(df.loc[df.roi_src == "none", "cement"] == 1, "cement", "no_measurement")
df["hu"] = np.where(df.level_used == "L1", df.L1_hu, np.where(df.level_used == "L2", df.L2_hu, np.nan))
df.loc[df.exclude != "", "hu"] = np.nan
df["y110"] = np.where(df.hu.notna(), (df.hu <= T_POS).astype(float), np.nan)
df["y160"] = np.where(df.hu.notna(), (df.hu <= T_SCREEN).astype(float), np.nan)
df["cat3"] = np.select([df.hu <= T_POS, df.hu <= T_SCREEN, df.hu > T_SCREEN], ["骨質疏鬆", "骨量減少", "正常"], "")
df.to_csv(OUT)
print(df.groupby("split").agg(n=("hu", "size"), labelled=("hu", "count"), y110=("y110", "sum"), y160=("y160", "sum")))
print(pd.crosstab(df.split, df.exclude)); print(pd.crosstab(df.split, df.fallback_reason)); print(pd.crosstab(df.split, df.roi_src))
