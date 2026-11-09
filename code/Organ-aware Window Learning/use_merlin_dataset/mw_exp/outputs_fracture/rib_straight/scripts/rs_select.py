"""rib_straight: rank final_ft variants by stacker train CV only (val not read). -> rib_straight/rs_selection.json"""
import sys, json
from pathlib import Path
FT = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/final_ft"; rows = []
for t in sys.argv[1:]:
    s = json.load(open(FT / t / "summary.json")); rows.append((t, " · ".join(s["selected"]), s["selected_train_cv"]))
rows.sort(key=lambda r: -r[2])
for r in rows: print(f"{r[0]:16s} {r[1]:45s} {r[2]:.4f}")
json.dump({"by_train_cv": rows, "selected": rows[0][0]}, open(FT.parent / "rib_straight/rs_selection.json", "w"), indent=1, ensure_ascii=False)
print("SELECTED", rows[0][0])
