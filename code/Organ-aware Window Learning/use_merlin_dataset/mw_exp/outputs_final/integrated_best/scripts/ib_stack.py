"""Formalise the per-finding "best cheap fix" stacks (train 5-fold OOF selection -> full-train refit -> val once).
usage: ib_stack.py <ib_root> <finding> [<finding> ...]
<ib_root> holds base/ (labels14_trval, age_trval, expB_E3late_val), an/<analysis>/ (oof_train, scores_val_descriptive,
feature csvs) and features/ (train/val features completed by integrated_best). Writes models/<finding>.joblib,
scores/<finding>_{oof,val}.csv, results/<finding>.json. Test is never read; age/sex never enter a model."""
import os, sys, json, joblib, numpy as np, pandas as pd, sklearn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ib_core import select_and_fit, evaluate, CANDS

R = sys.argv[1].rstrip("/") + "/"
META_COLS = {"Age", "Gender", "contrast", "phase", "manufacturer", "kvp", "slicethickness", "y_sp", "split"}
DROP = {"study_id", "err", "grid_mismatch", "shape_mismatch"}
AAA_CORE = ["infra_eq_rmax", "infra_minor_rmax", "infra_feret_rmax", "infra_major_rmax", "abd_eq_rmax", "abd_minor_rmax",
            "all_eq_rmax", "thor_eq_rmax", "infra_ratio", "infra_vol_ml", "infra_grow_eq_rmax", "il_eq_rmax", "infra_eq_max",
            "ilL_eq_rmax", "ilR_eq_rmax", "v_n1600", "v_n2000", "v_max", "infra_lowfrac", "lum_ref"]

# heads = probe-score columns (OOF from an/<a>/oof_train.csv, val from scores_val_descriptive.csv)
CFG = {
    "hydronephrosis": dict(an="hydronephrosis", feats=["hu_a6.csv", "hu_ct.csv"], lr_pairs=True,
                           sets={"E3": ["E3_late"], "E3max": ["E3_late_max"], "E3+E3max": ["E3_late", "E3_late_max"]}),
    "atherosclerosis": dict(an="atherosclerosis", feats=["hu_a6.csv", "hu_ct.csv", "hu_mac.csv"], sets={"E3": ["E3_late"]}),
    "free_air": dict(an="free_air", feats=["air_a6.csv", "air_ct.csv", "+fa_air_a6_rest.csv"], sets={"E3": ["E3_late"]}),
    "ascites": dict(an="ascites", feats=["hu_ct.csv", "hu_ct_b.csv", "+asc_hu_ct_rest.csv", "+asc_hu_ct2.csv", "+asc_hu_mac_rest.csv", "+asc_hu_mac_rest2.csv"],
                    sets={"E3": ["E3_late"], "all12": ["E3_late_all12"], "E3+all12": ["E3_late", "E3_late_all12"]}),
    "abdominal_aortic_aneurysm": dict(an="aaa", feats=["geom_a6.csv", "geom_ct.csv"], sets={"E3": ["E3_late"]}, core=AAA_CORE),
    "surgically_absent_gallbladder": dict(an="absent_gb", feats=["feat_a6.csv", "feat_ct.csv"], sets={"E3": ["E3_late"]}),
    "bowel_obstruction": dict(an="bowel_obstruction", feats=["feat_a6.csv", "feat_ct.csv", "+sbo_feat_a6_rest.csv"],
                              sets={"E3": ["E3_late"], "E3+st": ["E3_late", "E3_late_st"]}),
    "splenomegaly": dict(an="splenomegaly", feats=["geom_a6.csv", "geom_ct.csv"], sets={"E3": ["E3_late"]}),
    "osteopenia": dict(an="osteopenia", feats=["bundle_trval.csv", "hu_a6.csv", "hu_ct.csv", "+op_hu_ct_rest.csv", "+op_hu_a6slice.csv"],
                       sets={"E3": ["E3_late"], "win": ["E3_late", "E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"]},
                       heads_only=True),
    "hepatic_steatosis": dict(an="steatosis", feats=["hs_a6.csv", "hs_ct.csv"], derive="steatosis",
                              head_files=[("oof_train_concat_diff.csv", "scores_val_concat_diff.csv", ["E3_late_concat_diff"])],
                              sets={"E3": ["E3_late"], "E3+cd": ["E3_late", "E3_late_concat_diff"]}),
    "appendicitis": dict(an="appendicitis", feats=["hu_a6.csv", "hu_ct.csv"],
                         sets={"E3": ["E3_late"], "csb": ["E3_late_colon_sb"], "E3+csb": ["E3_late", "E3_late_colon_sb"]},
                         core_prefix=("rlq_r", "dlr_", "dglob_", "rlq_wall", "rlq_lumen", "rlq_z_rel", "rlq_seed", "glob_fat_med", "glob_fluid_frac")),
}
for _t, _zh in (("def", "確定的急性闌尾炎"), ("unc", "可疑等闌尾炎")):
    CFG["ap_" + _t] = dict(an="ap_" + _t, feats=["hu_a6.csv", "hu_ct.csv"], orig="appendicitis", zh=_zh,
                           label=("splitsrc/ap_labels.csv", "y_ap_" + _t),
                           sets={"E3": ["E3_late"], "csb": ["E3_late_colon_sb"], "E3+csb": ["E3_late", "E3_late_colon_sb"],
                                 "E3+trap": ["E3_late", "E3cs_trap"]},
                           core_prefix=CFG["appendicitis"]["core_prefix"])


def load_feats(c):
    parts = []
    for f in c["feats"]:
        p = R + ("features/" + f[1:] if f.startswith("+") else f"an/{c['an']}/{f}")
        if os.path.exists(p):
            d = pd.read_csv(p)
            if "err" in d: d = d[d.err.fillna(0) == 0]
            parts.append(d)
        else:
            print("  (feature file not present yet)", p)
    # several csvs per finding: either row blocks (same columns) or column blocks (osteopenia bundle + core HU)
    blocks = {}
    for d in parts: blocks.setdefault(tuple(sorted(d.columns)), []).append(d)
    F = None
    for k, ds in blocks.items():
        b = pd.concat(ds).drop_duplicates("study_id", keep="first")
        F = b if F is None else F.merge(b, on="study_id", how="outer", suffixes=("", "_dup"))
    F = F[[x for x in F.columns if not x.endswith("_dup")]]
    cols = [x for x in F.columns if x not in DROP and x not in META_COLS and pd.api.types.is_numeric_dtype(F[x])]
    F = F[["study_id"] + cols].copy()
    F = derive(F, c)
    return F, [x for x in F.columns if x != "study_id"]


def derive(F, c):
    """image-feature derivations shared by training (ib_stack) and test-time apply (ib_apply)."""
    cols = [x for x in F.columns if x != "study_id"]
    if c.get("lr_pairs"):
        for x in [x for x in cols if x.startswith("l_") and "r_" + x[2:] in cols]:
            a, b = F[x], F["r_" + x[2:]]; F["max_" + x[2:]] = np.fmax(a, b); F["min_" + x[2:]] = np.fmin(a, b); F["ad_" + x[2:]] = (a - b).abs()
    if c.get("derive") == "steatosis":       # same ratios/differences as quickwins hs features (image-only)
        F["L_minus_S"] = F.liver_p50 - F.spleen_med; F["L_over_S"] = F.liver_p50 / F.spleen_med.where(F.spleen_med.abs() > 1)
        F["L_minus_aorta"] = F.liver_p50 - F.aorta_med; F["L_minus_portal"] = F.liver_p50 - F.portal_med
    return F


def run(fd):
    c = CFG[fd]; A = R + f"an/{c['an']}/"; out = {}
    if c.get("label"):
        lab = pd.read_csv(R + c["label"][0])[["study_id", "split", c["label"][1]]].rename(columns={c["label"][1]: "y"})
        lab = lab[lab.split.isin(["train", "val"])]
    else:
        lab = pd.read_csv(R + "base/labels14_trval.csv")[["study_id", "split", fd]].rename(columns={fd: "y"})
    of = c.get("orig", fd)
    age = pd.read_csv(R + "base/age_trval.csv"); exb = pd.read_csv(R + "base/expB_E3late_val.csv")[["study_id", of]].rename(columns={of: "expB_E3"})
    oof = pd.read_csv(A + "oof_train.csv"); sv = pd.read_csv(A + "scores_val_descriptive.csv").drop(columns=["y_sp"], errors="ignore")
    for fo, fv, hc in c.get("head_files", []):
        oof = oof.merge(pd.read_csv(A + fo)[["study_id"] + hc], on="study_id", how="left")
        sv = sv.merge(pd.read_csv(A + fv)[["study_id"] + hc], on="study_id", how="left")
    F, fcols = load_feats(c)
    T = lab[lab.split == "train"].merge(oof, on="study_id").merge(F, on="study_id", how="left").merge(age, on="study_id", how="left")
    T = T[T.y.notna()].reset_index(drop=True)
    V = lab[lab.split == "val"].merge(sv, on="study_id").merge(exb, on="study_id").merge(F, on="study_id", how="left").merge(age, on="study_id", how="left")
    V = V[V.y.notna()].reset_index(drop=True)
    y = T.y.values.astype(int); folds = T.fold.values
    cov_t = T[fcols].notna().any(axis=1).mean(); cov_v = V[fcols].notna().any(axis=1).mean()
    print(f"[{fd}] train {len(T)} ({y.sum()} pos)  val {len(V)} ({int(V.y.sum())} pos)  feature coverage train {cov_t:.3f} val {cov_v:.3f}", flush=True)
    sets = {}
    for nm, h in c["sets"].items():
        sets[nm] = list(h)                                   # heads only (reference)
        if not c.get("heads_only_sets_only"):
            sets[nm + "+feat"] = list(h) + fcols
            if c.get("core"): sets[nm + "+core"] = list(h) + [x for x in c["core"] if x in fcols]
            if c.get("core_prefix"): sets[nm + "+core"] = list(h) + [x for x in fcols if x.startswith(c["core_prefix"])]
    sel = select_and_fit(T, V, y, folds, sets)
    print(f"  -> chosen {sel['featset']} | {sel['cand']}  OOF {sel['oof_auc']:.4f}", flush=True)
    r = evaluate(fd, y, sel["oof"], T.E3_late.values, T.Age.values, V.y.values, sel["val"], V.expB_E3.values, V.E3_late.values,
                 V.Age.values, thr_oof=sel["thr90_oof"])
    r.update(featset=sel["featset"], cand=sel["cand"], n_feat=len(sel["cols"]), cov_train=float(cov_t), cov_val=float(cov_v), cv=sel["cv"])
    os.makedirs(R + "models", exist_ok=True); os.makedirs(R + "scores", exist_ok=True); os.makedirs(R + "results", exist_ok=True)
    joblib.dump(dict(finding=fd, model=sel["model"], prep=sel["prep"], cols=sel["cols"], featset=sel["featset"], cand=sel["cand"],
                     thr90_oof=sel["thr90_oof"], feature_files=c["feats"], analysis=c["an"], sklearn=sklearn.__version__,
                     lr_pairs=bool(c.get("lr_pairs"))), R + f"models/{fd}.joblib")
    json.dump(dict(finding=fd, featset=sel["featset"], cand=sel["cand"], columns=sel["cols"], feature_files=c["feats"]),
              open(R + f"models/{fd}_features.json", "w"), indent=1)
    pd.DataFrame({"study_id": T.study_id, "fold": folds, "y": y, "E3_late_oof": T.E3_late, "stack_oof": sel["oof"]}).to_csv(R + f"scores/{fd}_oof.csv", index=False)
    pd.DataFrame({"study_id": V.study_id, "y": V.y.astype(int), "expB_E3_late": V.expB_E3, "E3_late_rerun": V.E3_late, "stack": sel["val"]}).to_csv(R + f"scores/{fd}_val.csv", index=False)
    json.dump(r, open(R + f"results/{fd}.json", "w"), indent=1, default=float)
    print(f"  val {r['val']:.4f} {r['val_ci']}  expB E3 {r['val_E3_expB']:.4f}  d {r['d_val_expB']}  age {r['val_age']:.4f} vs {r['val_age_E3']:.4f}", flush=True)


if __name__ == "__main__":
    for fd in sys.argv[2:]: run(fd)
