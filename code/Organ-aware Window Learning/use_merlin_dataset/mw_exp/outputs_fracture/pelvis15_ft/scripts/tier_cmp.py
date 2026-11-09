"""pelvis15_ft tier comparison (train rows only; val/test never read). Tier rows = pelvis15 ids_tier (train), target t_pel.
For each FT run dir (oof_train.csv, column ft, in_tier): tier OOF AUROC vs
 - PV15 tier probe (pelvis15/runs/t295s_m400, same tier-only training, ft_side; 3 seed)
 - PV15 production (pelvis15/final_m400, full-train 20 seed OOF) restricted to tier rows
 - U_org_pel / U_pel (full-train OOF, ref_train.csv) restricted to tier rows
paired bootstrap (2000) of FT - reference; plus LR stack 5-fold CV on tier rows (rs 0..4 averaged):
 ref2 = U_org_pel+U_pel, ref3 = +PV15(final), and each + FT.
usage: tier_cmp.py RUN_DIR [RUN_DIR ...]  -> prints markdown table, writes tier_cmp.json next to the first run's parent"""
import sys, glob, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
P = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/pelvis15"
r = pd.read_csv(P / "ref_train.csv", index_col=0)[["t_pel", "U_org_pel", "U_pel"]]
ids = [l.strip() for l in open(P / "ids_tier.txt") if l.strip()]
d = r.loc[r.index.intersection(ids)].dropna(subset=["t_pel"])
pv = pd.read_csv(P / "final_m400/oof_train.csv").set_index("study_id").ft.rename("PV15")
s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(P / "runs/t295s_m400/score_fold*.csv")))])
s = s[s.split == "train"].set_index("study_id").ft_side.rename("PV15tier")
d = d.join(pv).join(s)
runs = {}
for rd in sys.argv[1:]:
    o = pd.read_csv(Path(rd) / "oof_train.csv").set_index("study_id").ft; runs[Path(rd).name] = o
    d = d.join(o.rename(Path(rd).name))
d = d.dropna()
y = d.t_pel.values.astype(int); rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
B = [b for b in B if 0 < y[b].sum() < len(b)]


def bd(a, b):
    v = np.array([auc(y[i], a[i]) - auc(y[i], b[i]) for i in B]); return f"{auc(y, a) - auc(y, b):+.4f} [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]"


def stack(cols):
    X = d[cols].values; X = (X - X.mean(0)) / X.std(0); ps = []
    for rs in range(5):
        ps.append(auc(y, cross_val_predict(LogisticRegression(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=rs), method="decision_function")))
    return float(np.mean(ps))


out = {"n": int(len(d)), "pos": int(y.sum()), "ref": {c: float(auc(y, d[c])) for c in ["PV15tier", "PV15", "U_org_pel", "U_pel"]},
       "stack_ref2": stack(["U_org_pel", "U_pel"]), "stack_ref3": stack(["U_org_pel", "U_pel", "PV15"]), "runs": {}}
print(f"tier rows n={len(d)} pos={y.sum()}; refs " + ", ".join(f"{k} {v:.4f}" for k, v in out["ref"].items()))
print(f"stack ref2 (U_org_pel+U_pel) {out['stack_ref2']:.4f}; ref3 (+PV15) {out['stack_ref3']:.4f}")
print("| run | tier OOF | Δ vs PV15 tier probe | Δ vs PV15 (full) | Δ vs U_org_pel | stack ref2+FT | stack ref3+FT |")
print("|---|---|---|---|---|---|---|")
for n in runs:
    a = d[n].values
    o = {"oof": float(auc(y, a)), "d_pv15tier": bd(a, d.PV15tier.values), "d_pv15": bd(a, d.PV15.values), "d_uorg": bd(a, d.U_org_pel.values),
         "stack2": stack(["U_org_pel", "U_pel", n]), "stack3": stack(["U_org_pel", "U_pel", "PV15", n])}
    out["runs"][n] = o
    print(f"| {n} | {o['oof']:.4f} | {o['d_pv15tier']} | {o['d_pv15']} | {o['d_uorg']} | {o['stack2']:.4f} ({o['stack2'] - out['stack_ref2']:+.4f}) | {o['stack3']:.4f} ({o['stack3'] - out['stack_ref3']:+.4f}) |")
json.dump(out, open(Path(sys.argv[1]).parent / "tier_cmp.json", "w"), indent=1)
