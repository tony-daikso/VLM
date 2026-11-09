"""fracture_rp: pick the integration variant by stacker train CV only (summary.json selected_train_cv); val not read here.
usage: rp_select.py TAG [TAG ...]  -> prints table (tag, inner selection, train CV) and writes rp_selection.json"""
import sys, json
from pathlib import Path
FT = Path(__file__).resolve().parents[1]; rows = []
for t in sys.argv[1:]:
    s = json.load(open(FT / t / "summary.json")); rows.append((t, " · ".join(s["selected"]), s["selected_train_cv"]))
rows.sort(key=lambda r: -r[2])
for r in rows: print(f"{r[0]:16s} {r[1]:45s} {r[2]:.4f}")
json.dump({"by_train_cv": rows, "selected": rows[0][0]}, open(FT / "rp_selection.json", "w"), indent=1, ensure_ascii=False)
print("SELECTED", rows[0][0])
