"""fracture a6x10w2_tv7 -- inference from SAVED models (2026-11-10).

Stacker = push90b final_ft3 candidate "S3n = S2 + FT (no FT_pv)" . combiner "max": three site heads
(StandardScaler + LogisticRegression(max_iter=3000), fit on the site label t_<site> over the common TRAIN frame, inputs =
the components' train OOF scores exactly as in final_ft3), case score = max of the three site logits.
This script
  1. rebuilds the train frame (sitewise.build_frame + final_ft3 FT inputs; train median imputation of FT columns),
     refits the 3 site heads and SAVES them (stacker.pkl + stacker.json: scaler mean/scale, LR coef/intercept,
     train medians, z-score parameters of every z-combined input),
  2. checks the reimplementation on the STORED val inputs (must reproduce val .8973),
  3. scores Merlin VAL from the saved-model inputs (full-train refits; FT_sp15 / P15FT = their saved fold models),
  4. scores CG (--cg) from the CG inputs produced by the saved models.
Normalisations: every z-combined input (combine3 / combine.py / zmix) uses the TRAIN-OOF mean/sd of the original run, so
saved-model scores are put on the scale the stacker was fit on. Test split never read (build_frame filters split first).
usage: integrate.py [--cg]"""
import sys, os, json, glob, pickle, argparse
from pathlib import Path
import numpy as np, pandas as pd
MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"; FR = MW / "outputs_fracture"
MOD = MW / "outputs_final/models/fracture_a6x10w2_tv7"; PR = MOD / "probes"; OUT = MOD / "integ"
sys.path.insert(0, str(FR / "sitewise/scripts")); sys.path.insert(0, str(MW / "src"))
import sitewise as S
from probe import auroc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ap = argparse.ArgumentParser(); ap.add_argument("--cg", action="store_true"); ap.add_argument("--stored-only", action="store_true"); a = ap.parse_args()

# ---------------- leaf runs: (original CV run dir with score_fold*.csv, saved full-train run under probes/, column) -----------
RV3 = [(FR / f"rib_v3/runs/{r}", f"lev28/{r}") for r in ["v3_side_lev", "v3_sl_s10", "v3_sl_s20", "v3_sl_s30", "v3_sl_s40", "v3_s1l_s50"]]
RS15 = [(FR / f"rib_straight/prod/{r}", f"rs15/{r}") for r in ["t8_sl_s0", "t8_sl_s20", "t8_s1l_s30"]]
A6 = [(FR / "push90/prod/f_a6", "a6/f_a6"), (FR / "push90b/prod/f_a6_s5", "a6/f_a6_s5"), (FR / "push90b/prod/f_w2", "a6/f_w2")]
PE1 = [(FR / "push90/B/prod/pf_ctx", "pe/pf_ctx")]
PE2 = [(FR / f"push90b/B/prod/pe_tv_s{s}", f"pe2/pe_tv_s{s}") for s in (0, 5, 10, 15, 20, 25, 30)]
RIBFT = [(OUT / "rib_ft_runs/final_f3_a", "ftrib/final_f3_a"), (OUT / "rib_ft_runs/final_f3_b", "ftrib/final_f3_b")]
CAND = {"lev28": FR / "rib_v3/cand/lev28", "rs15": FR / "rib_straight/cand/rs15", "a6x10w2": FR / "push90b/cand/a6x10w2_side"}
ZP = {}  # all z parameters (saved)


def run_oof(d, col):
    s = pd.concat([pd.read_csv(f"{d}/score_fold{k}.csv") for k in range(5)])
    return s[s.split == "train"].set_index("study_id")[col], s[s.split == "val"].groupby("study_id")[col].mean()


def zcomb(name, runs, col, fcol, part):
    """combine3 / combine.py: per run z with the run's train-OOF mean/sd, mean over runs. part: 'oof' -> (train OOF, stored
    val) recomputed from the CV files; 'val' / 'cg' -> saved full-train models' seed-mean scores of that split."""
    out = []
    for d, fr in runs:
        t, v = run_oof(d, col); mu, sd = float(t.mean()), float(t.std()); ZP[f"{name}:{fr}"] = {"mu": mu, "sd": sd, "col": col}
        if part == "oof":
            x = pd.concat([t, v])
        else:
            f = PR / fr / ("scores_full.csv" if part == "val" else "cg_scores.csv")
            q = pd.read_csv(f).set_index("study_id")
            if part == "val": q = q[q.split == "val"]
            x = q[fcol]
        out.append((x - mu) / sd)
    return pd.concat(out, axis=1).mean(1)


def zmix(parts, part):
    out = []
    for k, s in parts.items():
        t = pd.read_csv(CAND[k] / "oof_train.csv").set_index("study_id")["ft"]; mu, sd = float(t.mean()), float(t.std())
        ZP[f"zmix:{k}"] = {"mu": mu, "sd": sd}; out.append((s - mu) / sd)
    return pd.concat(out, axis=1).mean(1)


def comp_inputs(part):
    """the 5 z-combined inputs RVRS, PE, PE2, FT_rib for part in oof|val|cg"""
    lev = zcomb("lev28", RV3, "ft_side", "ft_side", part); rs = zcomb("rs15", RS15, "ft_side", "ft_side", part)
    a6 = zcomb("a6x10w2", A6, "ft_side", "ft_side", part)
    if part == "oof":  # formula check against the stored cand files
        for k, s in (("lev28", lev), ("rs15", rs), ("a6x10w2", a6)):
            t = pd.read_csv(CAND[k] / "oof_train.csv").set_index("study_id")["ft"]
            assert np.abs(s.reindex(t.index).values - t.values).max() < 1e-9, k
    return {"RVRS": zmix({"lev28": lev, "rs15": rs, "a6x10w2": a6}, part), "PE": zcomb("PE", PE1, "ft_head", "ft_head", part),
            "PE2": zcomb("PE2", PE2, "ft_head", "ft_head", part), "FT_rib": zcomb("FT_rib", RIBFT, "ft", "ft", part)}


# ---------------- 1. train frame exactly as final_ft3 (inputs = stored OOF / stored val) ----------------
df, sel = S.build_frame()
STORED = {"FT_sp": FR / "spine_ft/runs/final_tf2lse", "FT_pv": FR / "pelvis_ft/runs/final_tf2lse", "FT_rib": FR / "rib_ft",
          "FT_sp15": FR / "spine15_ft", "RVRS": FR / "push90b/cand/mix3_a6x10w2", "P15FT": FR / "pelvis15_ft/final",
          "PE": FR / "push90/cand/pe_ctx_head", "PE2": FR / "push90b/cand/pe_tv7r_head"}
for n, d in STORED.items():
    tr_ = pd.read_csv(d / "oof_train.csv"); va_ = pd.read_csv(d / "val_scores.csv")
    c = "ft" if "ft" in tr_.columns else [x for x in tr_.columns if x not in ("study_id", "fold", "in_tier", "split")][-1]
    cv = "ft" if "ft" in va_.columns else [x for x in va_.columns if x not in ("study_id", "fold", "split")][-1]
    df = df.join(pd.concat([tr_.set_index("study_id")[c], va_.set_index("study_id")[cv]]).rename(n))
FTC = list(STORED)
MED = {n: float(df.loc[df.split == "train", n].median()) for n in FTC}
for n in FTC: df[n] = df[n].fillna(MED[n])
OLD = {k: list(v) for k, v in S.FEATS["S2"].items()}
FEATS = {"vert": OLD["vert"] + ["FT_sp", "FT_sp15", "PE", "PE2"], "rib": OLD["rib"] + ["FT_rib", "RVRS"], "pel": OLD["pel"] + ["P15FT"]}
need = sorted({c for v in FEATS.values() for c in v} | set(S.C3M) | set(FTC))
tr = df[df.split == "train"].dropna(subset=need).copy(); va = df[df.split == "val"].dropna(subset=need + ["age"]).copy()
print("train", len(tr), "val", len(va), flush=True)
# the z-combined inputs recomputed from the CV files must equal the stored ones (formula check)
chk = comp_inputs("oof")
for n in ("RVRS", "PE", "PE2", "FT_rib"):
    d_ = np.abs(chk[n].reindex(tr.index).values - tr[n].values); print("formula check", n, "max|d| train", float(np.nanmax(d_)), flush=True)
    assert np.nanmax(d_) < 1e-6, n

# ---------------- site heads (fit + save) ----------------
HEADS = {}
for s in S.SITES:
    m = ~tr["t_" + s].isna(); X = tr.loc[m, FEATS[s]].values; y = tr.loc[m, "t_" + s].values
    sc = StandardScaler().fit(X); lr = LogisticRegression(max_iter=3000).fit(sc.transform(X), y); HEADS[s] = (sc, lr)


def score(F):
    L = {s: HEADS[s][1].decision_function(HEADS[s][0].transform(F[FEATS[s]].values)) for s in S.SITES}
    L = pd.DataFrame(L, index=F.index); L["fracture"] = L[S.SITES].max(1); return L


# trim I_v/I_r/I_p z params (build_frame: per group seed-mean OOF site logit, z with train mean/sd over the frame's train rows)
trm = df["split"] == "train"
for k, nm in enumerate(["I_v", "I_r", "I_p"]):
    for g in sel:
        fs = sorted(glob.glob(str(FR / f"bones/expI/members/oof_bone_E_{g}_s*.npz")))
        s_ = S.npz_series(fs, "site_logit", k).reindex(df.index); ZP[f"{nm}:{g}"] = {"mu": float(s_[trm].mean()), "sd": float(s_[trm].std())}
OUT.mkdir(parents=True, exist_ok=True)
pickle.dump({"heads": HEADS, "feats": FEATS, "medians": MED, "zparams": ZP, "expI_groups": sel, "combiner": "max"}, open(OUT / "stacker.pkl", "wb"))
json.dump({"selected": "a6x10w2_tv7: S3n = S2 + FT (no FT_pv) . max", "feats": FEATS, "medians_train": MED, "zparams": ZP, "expI_groups": sel,
           "heads": {s: {"features": FEATS[s], "scaler_mean": HEADS[s][0].mean_.tolist(), "scaler_scale": HEADS[s][0].scale_.tolist(),
                         "coef": HEADS[s][1].coef_[0].tolist(), "intercept": float(HEADS[s][1].intercept_[0]),
                         "n_fit": int((~tr["t_" + s].isna()).sum())} for s in S.SITES},
           "n_train": len(tr)}, open(OUT / "stacker.json", "w"), indent=1)

# ---------------- 2. stored-val check ----------------
yv = va["y"].values
p_st = score(va)["fracture"].values; auc_st = auroc(yv, p_st); print("STORED-input val AUROC", round(auc_st, 4), flush=True)
assert abs(auc_st - 0.8973111774509142) < 2e-4, auc_st
if a.stored_only: raise SystemExit(0)


# ---------------- 3./4. inputs from saved models ----------------
def s2_inputs(split):
    q = pd.read_csv(MOD / "s2/s2_scores_merlin_cg.csv"); q = q[q.split == split].set_index("study_id")
    F = pd.DataFrame(index=q.index)
    for c in ["J_add", "L_v15", "L_v3", "LJ_v15", "LJ_v3", "L_r", "L_rinst", "LJ_r", "U_pel", "U_org_pel", "U_org_out", "L_p", "LJ_p"]:
        F[c] = q[c]
    for nm, sx in (("I_v", "v"), ("I_r", "r"), ("I_p", "p")):
        F[nm] = pd.concat([(q[f"I{g}_{sx}"] - ZP[f"{nm}:{g}"]["mu"]) / ZP[f"{nm}:{g}"]["sd"] for g in sel], axis=1).mean(1)
    return F


def saved_inputs(part, ids):
    F = s2_inputs("val" if part == "val" else "cg").reindex(ids)
    for n, s_ in comp_inputs(part).items(): F[n] = s_.reindex(ids)
    if part == "val":
        F["FT_sp"] = pd.read_csv(MOD / "ft_sp/scores_full.csv").query("split == 'val'").set_index("study_id")["ft"].reindex(ids)
        F["FT_sp15"] = va["FT_sp15"].reindex(ids)            # saved fold models = the stored val scores
        F["P15FT"] = va["P15FT"].reindex(ids)
    else:
        F["FT_sp"] = pd.read_csv(MOD / "ft_sp/cg_scores.csv").set_index("study_id")["ft"].reindex(ids)
        F["FT_sp15"] = pd.read_csv(MOD / "ft_sp15/cg_scores.csv").set_index("study_id")["ft"].reindex(ids)
        P = [pd.read_csv(MOD / f"p15ft/s{s}/cg_scores.csv").set_index("study_id")["ft"] for s in (0, 1)]
        F["P15FT"] = pd.concat(P, axis=1).mean(1, skipna=False).reindex(ids)
    miss = {c: int(F[c].isna().sum()) for c in F.columns}
    for c in F.columns:  # FT-type and S2 columns: train median (Merlin rule for FT columns; S2 NaN would drop the row in Merlin)
        if F[c].isna().any(): F[c] = F[c].fillna(MED[c] if c in MED else float(tr[c].median()))
    return F, miss


Fv, missv = saved_inputs("val", va.index)
Lv = score(Fv); auc_sv = auroc(yv, Lv["fracture"].values)
cmp = {c: {"r": float(np.corrcoef(Fv[c], va[c])[0, 1]), "mad": float((Fv[c] - va[c]).abs().mean()),
           "auc_site_saved": auroc(va["t_" + s].dropna().values, Fv.loc[va["t_" + s].notna(), c].values),
           "auc_site_stored": auroc(va["t_" + s].dropna().values, va.loc[va["t_" + s].notna(), c].values)}
       for s in S.SITES for c in FEATS[s]}
print("SAVED-model val AUROC", round(auc_sv, 4), "missing", {k: v for k, v in missv.items() if v}, flush=True)
SUB = {g: (yv == 1) & np.asarray(f(va), bool) for g, f in S.SUB.items()}
neg = yv == 0
subv = {g: {"stored": auroc(m[m | neg].astype(float), p_st[m | neg]), "saved": auroc(m[m | neg].astype(float), Lv["fracture"].values[m | neg]),
            "n_pos": int(m.sum())} for g, m in SUB.items()}
ci = S.boot_ci(yv.astype(int), Lv["fracture"].values)
res = {"n_val": len(va), "val_stored_inputs": auc_st, "val_saved_models": auc_sv, "val_saved_ci": ci, "missing_imputed": missv,
       "input_agreement": cmp, "subgroups": subv}
json.dump(res, open(OUT / "val_reproduction.json", "w"), indent=1, default=float)
pd.DataFrame({"study_id": va.index, "y": yv, "stored": p_st, "saved": Lv["fracture"].values, "site_vert": Lv["vert"].values,
              "site_rib": Lv["rib"].values, "site_pel": Lv["pel"].values}).to_csv(OUT / "val_scores_saved_vs_stored.csv", index=False)
if a.cg:
    ids = [l.strip() for l in open(MOD / "cg_inputs/ids_main.txt") if l.strip()]
    Fc, missc = saved_inputs("cg", pd.Index(ids))
    Lc = score(Fc); o = Fc.join(Lc.rename(columns={"vert": "site_vert", "rib": "site_rib", "pel": "site_pel"}))
    o.index.name = "study_id"; o.to_csv(OUT / "cg_main_scores.csv")
    json.dump({"n": len(ids), "missing_imputed": missc}, open(OUT / "cg_missing.json", "w"), indent=1)
    print("CG scored", len(o), "missing", {k: v for k, v in missc.items() if v}, flush=True)
print("INTEGRATE_DONE", flush=True)
