"""push90: rank push90/integ/* variants (plus the rsP_mix_repl reference) by stacker train CV ONLY (official labels);
val not read here. -> push90/p90_selection.json"""
import json, glob, os
P = os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/push90"); rows = []
ref = json.load(open(P + "/../final_ft/rsP_mix_repl/summary.json")); rows.append(("rsP_mix_repl (ref)", " · ".join(ref["selected"]), ref["selected_train_cv"]))
for f in glob.glob(P + "/integ/*/summary.json"):
    s = json.load(open(f)); rows.append((f.split("/")[-2], " · ".join(s["selected"]), s["selected_train_cv"]))
rows.sort(key=lambda r: -r[2])
for r in rows: print(f"{r[0]:20s} {r[1]:40s} {r[2]:.4f}")
json.dump({"by_train_cv": rows, "selected": rows[0][0]}, open(P + "/p90_selection.json", "w"), indent=1, ensure_ascii=False)
print("SELECTED", rows[0][0])
