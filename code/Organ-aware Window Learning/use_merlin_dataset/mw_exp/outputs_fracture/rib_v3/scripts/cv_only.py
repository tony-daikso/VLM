"""Train-CV-only screening of new site heads in the sitewise/final_ft stack (no val rows are scored or read for AUROC).
Same frame / nested CV as final_ft.py (S3 = S2 + FT heads). usage: cv_only.py NAME=DIR@site [NAME=DIR@site ...]
DIR has oof_train.csv (study_id, ft). The baseline set (FT_sp 3-seed spine, FT_pv, FT_rib) is always included unless
--drop NAME is given."""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"; sys.path.insert(0, str(MW / "outputs_fracture/sitewise/scripts")); sys.path.insert(0, str(MW / "src"))
import sitewise as S
from probe import auroc
FR = MW / "outputs_fracture"
ap = argparse.ArgumentParser(); ap.add_argument("add", nargs="*"); ap.add_argument("--drop", action="append", default=[])
ap.add_argument("--kinds", default="lse,lse+any")
a = ap.parse_args()
df, _ = S.build_frame(); df = df[df.split == "train"].copy()               # train rows only
base = {"FT_sp": (FR / "spine_ft/runs/final3_tf2lse", "vert"), "FT_pv": (FR / "pelvis_ft/runs/final_tf2lse", "pel"), "FT_rib": (FR / "rib_ft", "rib")}
for spec in a.add:
    nm, rest = spec.split("=", 1); d, site = rest.rsplit("@", 1); base[nm] = (Path(d), site)
for n in a.drop: base.pop(n, None)
FTS = {s: [] for s in S.SITES}
for nm, (d, site) in base.items():
    t = pd.read_csv(d / "oof_train.csv").set_index("study_id")
    col = "ft" if "ft" in t.columns else t.columns[-1]
    df = df.join(t[col].rename(nm)); df.loc[df[nm].isna(), nm] = df[nm].median(); FTS[site].append(nm)
OLD = {k: list(v) for k, v in S.FEATS["S2"].items()}; C3 = list(S.C3M)
need = sorted({c for v in OLD.values() for c in v} | set(C3) | set(base))
tr = df.dropna(subset=need).copy(); K = a.kinds.split(",")
S.FEATS["X"] = {s: OLD[s] + FTS[s] for s in S.SITES}
oof, L1 = S.run_cv(tr, "X", K)
print("heads", {k: str(v[0]) for k, v in base.items()}, "N", len(tr))
print({k: round(auroc(tr["y"].values, oof[k]), 4) for k in K}, {f"site {s}": round(auroc(tr["t_" + s].values, L1[s].values), 4) for s in S.SITES}, flush=True)
