"""Apply the EXACT Merlin split-label programs (rule-based, locked 2026-11-05) to CG10k.
  steatosis_split/scripts/hs_split_labels.py  (sha256 0a2df151...) -> y_diffuse / y_focal
  appendicitis_split/scripts/ap_split_labels.py                    -> y_ap_def / y_ap_unc
Only the three input paths are redirected (split.csv / labels jsonl / reports.json); rules are untouched.
Output: data/CT/CG/10000_data/disease14_split_rules.csv (study_id=EXAMID + flags/labels, no report text)."""
import os, sys, json, runpy, tempfile, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
D = "/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/"
tmp = tempfile.mkdtemp()
lab = os.path.join(tmp, "labels.jsonl"); spl = os.path.join(tmp, "split.csv")
ids = []
with open(D + "disease14_qwen.jsonl") as f, open(lab, "w") as g:
    for line in f:
        r = json.loads(line); r["split"] = "cg"; ids.append(r["study_id"]); g.write(json.dumps(r) + "\n")
pd.DataFrame({"study_id": ids, "split": "cg"}).to_csv(spl, index=False)
outs = {}
for name in ("hs_split_labels", "ap_split_labels"):
    src = open(os.path.join(HERE, "merlin_ref", name + ".py")).read()
    g = {"__name__": "cgmod"}
    exec(compile(src, name, "exec"), g)
    g["SPLIT"], g["LABELS"], g["REPORTS"] = spl, lab, D + "disease14_shards/reports_cg10k.json"
    o = os.path.join(tmp, name + ".csv"); sys.argv = [name, "--out", o, "--splits", "cg"]
    print("=====", name); g["main"](); outs[name] = pd.read_csv(o)
hs = outs["hs_split_labels"]; ap = outs["ap_split_labels"]
hs = hs[["study_id", "hs_cat", "subtype", "D", "F", "y_hs", "y_diffuse", "y_focal", "y_diffuse_excl", "y_focal_excl", "neg_focal_text", "y_focal_negclean"]].rename(columns={"subtype": "hs_subtype"})
apc = [c for c in ap.columns if c.startswith(("y_ap", "t_", "ap_")) or c in ("study_id", "treat_followup_any")]
m = hs.merge(ap[apc], on="study_id", how="outer")
m.to_csv(D + "disease14_split_rules.csv", index=False)
print(m.shape); print(m[["y_diffuse", "y_focal", "y_ap_def", "y_ap_unc"]].agg(["sum", "count"]))
