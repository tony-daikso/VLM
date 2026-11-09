"""Apply a saved integrated-best stack to NEW studies (for test, only after finding_organ_map / config is locked).
usage: ib_apply.py <integrated_best dir> <finding> <heads.csv> <feat1.csv[,feat2.csv,...]> <out.csv>
  heads.csv : study_id + the probe-head columns listed in models/<finding>_features.json (E3_late, E3_late_max, ...),
              produced with the same probe recipe refit on all train studies (see rebuild_test_features.sh, step 2).
  feat*.csv : outputs of the original feature scripts (same code as train/val; see rebuild_test_features.sh, step 1).
Derived columns (left/right max/min/|diff|, liver-spleen differences) are rebuilt with ib_stack.derive (same code)."""
import os, sys, joblib, numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
root, fd, heads, feats, out = sys.argv[1:6]
sys.argv = [sys.argv[0], root]                        # ib_stack reads its root from argv[1]
import ib_stack
from ib_core import score
M = joblib.load(os.path.join(root, "models", f"{fd}.joblib")); c = ib_stack.CFG[fd]
parts = []
for f in feats.split(","):
    d = pd.read_csv(f)
    if "err" in d: d = d[d.err.fillna(0) == 0]
    parts.append(d)
blocks = {}                                            # same row-block / column-block merge as ib_stack.load_feats
for d in parts: blocks.setdefault(tuple(sorted(d.columns)), []).append(d)
F = None
for ds in blocks.values():
    b = pd.concat(ds).drop_duplicates("study_id", keep="first")
    F = b if F is None else F.merge(b, on="study_id", how="outer", suffixes=("", "_dup"))
F = F[[x for x in F.columns if not x.endswith("_dup")]]
F = F[["study_id"] + [x for x in F.columns if x not in ib_stack.DROP and x not in ib_stack.META_COLS and pd.api.types.is_numeric_dtype(F[x])]]
F = ib_stack.derive(F, c)
H = pd.read_csv(heads)
X = H.merge(F, on="study_id", how="left")
miss = [x for x in M["cols"] if x not in X.columns]
if miss: raise SystemExit(f"missing columns for {fd}: {miss[:10]} ...")
s = score(M["model"], M["prep"].transform(X[M["cols"]]).values)
pd.DataFrame({"study_id": X.study_id, f"{fd}_stack": s, f"{fd}_pos_at_spec90_trainOOF": (s > M["thr90_oof"]).astype(int)}).to_csv(out, index=False)
print(fd, len(X), "scored ->", out)
