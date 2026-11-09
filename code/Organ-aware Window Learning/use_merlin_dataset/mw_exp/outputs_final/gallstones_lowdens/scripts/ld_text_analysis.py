"""Check 2: prior-imaging / CT-occult evidence flags vs Exp N errors. Train OOF + val only (inputs asserted train/val)."""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
MW = "/Users/tony.tu/Desktop/戴承智慧/VLM/code/Organ-aware Window Learning/use_merlin_dataset/mw_exp/outputs_final/"
TF, O = sys.argv[1], sys.argv[2]
D = pd.read_csv(MW + "analysis_gallstones_v2/per_case_trval.csv"); assert set(D.split) <= {"train", "val"}
T = pd.read_csv(TF); assert set(T.split) <= {"train", "val"}
D = D.merge(T.drop(columns=["split", "pf_snip"]), on="study_id", how="left").fillna({c: 0 for c in T.columns if c.startswith("pf_")})
D["PI"] = D[["pf_known", "pf_previously", "pf_other_modality", "pf_not_well_seen", "pf_as_before", "pf_history"]].max(1)
D["NC"] = D.pf_noncalc; D["ANY"] = D[["PI", "NC"]].max(1)
D["ANYv2"] = D[["ANY", "f_hist_us", "f_noncalc"]].max(1)  # union with v2 flags (history/US-only, report noncalcified)
D["grp"] = np.where(D.err == "TP", "TP", np.where(D.err == "FN", "FN_" + D.cat.fillna("?").str[0], "neg"))
rng = np.random.default_rng(0)
def boot_excl(y, s, excl, n=2000):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; dd = []
    for _ in range(n):
        ix = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]; k = ~excl[ix]
        dd.append(auc(y[ix][k], s[ix][k]) - auc(y[ix], s[ix]))
    return [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
R = {}; rows = []
for sp in ("val", "train"):
    d = D[D.split == sp]; y = d.y_gs.values.astype(int); s = d.N_U.values; pos = d[d.y_gs == 1]
    for f in ("PI", "NC", "ANY", "ANYv2"):
        r = dict(split=sp, flag=f)
        for g in ("TP", "FN_E", "FN_F", "FN_C", "FN_A"):
            gg = pos[pos.grp == g]; r[f"{g}_n"] = len(gg); r[f"{g}_prev"] = float(gg[f].mean()) if len(gg) else np.nan
        r["FN_all_prev"] = float(pos[pos.err == "FN"][f].mean()); r["pos_prev"] = float(pos[f].mean()); r["pos_n_flag"] = int(pos[f].sum())
        r["neg_n_flag"] = int(d[(d.y_gs == 0)][f].sum())
        fl = (d[f].values == 1)
        r["auc_all"] = float(auc(y, s))
        k1 = (y == 0) | ((y == 1) & fl); k0 = (y == 0) | ((y == 1) & ~fl)
        r["auc_flagged_pos_vs_neg"] = float(auc(y[k1], s[k1])) if fl[y == 1].sum() else np.nan
        r["auc_unflagged_pos_vs_neg"] = float(auc(y[k0], s[k0]))
        ex = (y == 1) & fl
        r["auc_excl_flagged_pos"] = r["auc_unflagged_pos_vs_neg"]; r["gain_excl_pos"] = r["auc_unflagged_pos_vs_neg"] - r["auc_all"]
        r["gain_excl_pos_ci"] = boot_excl(y, s, ex)
        exb = fl.copy(); kb = ~exb  # also drop flagged negatives (e.g. 'no radiopaque stones' negatives)
        r["auc_excl_flagged_pos_and_neg"] = float(auc(y[kb], s[kb])); r["gain_excl_both"] = r["auc_excl_flagged_pos_and_neg"] - r["auc_all"]
        r["gain_excl_both_ci"] = boot_excl(y, s, exb)
        rows.append(r)
P = pd.DataFrame(rows); P.to_csv(O + "/text_prior_imaging_summary.csv", index=False)
D[["study_id", "split", "y_gs", "grp", "PI", "NC", "ANY", "ANYv2"] + [c for c in D.columns if c.startswith("pf_")]].to_csv(O + "/text_flags_trval_noText.csv", index=False)
pd.set_option("display.width", 250); print(P.round(3).to_string())
