"""Merge per-host feature outputs into WORK/hu_all.csv (bd_hu.py feature set, = hu_ct.csv + HU part of the A6000-only
cases computed by bd3_feat.py with identical code, max abs diff 0.0 on a parity check) and WORK/bd3_all.csv (BD3 tube +
porta/HDL features). Inputs: WORK/hu_ct.csv, WORK/bd3_*.jsonl (bd3_feat.py .part.jsonl files)."""
import sys, glob, json, pandas as pd
W = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"{W}/bd3_*.jsonl")):
    rows += [json.loads(l) for l in open(f)]
B = pd.DataFrame(rows).drop_duplicates("study_id", keep="last")
hu_ct = pd.read_csv(f"{W}/hu_ct.csv")
HUC = list(hu_ct.columns)
extra = B[~B.study_id.isin(hu_ct.study_id) & B.liver_hu.notna()] if "liver_hu" in B else B.iloc[:0]
hu_all = pd.concat([hu_ct, extra[[c for c in HUC if c in extra.columns]]]).drop_duplicates("study_id")
hu_all.to_csv(f"{W}/hu_all.csv", index=False)
keep = ["study_id"] + [c for c in B.columns if c.startswith(("tube_", "hdl_", "duct_lp_", "phd_fl_", "err_"))]
B[keep].to_csv(f"{W}/bd3_all.csv", index=False)
ids = set(pd.read_csv(f"{W}/labels_trval.csv").study_id)
print("hu_all", len(hu_all), "bd3_all", len(B), "of trval", len(ids), "| missing bd3:", len(ids - set(B.study_id)), "missing hu:", len(ids - set(hu_all.study_id)))
