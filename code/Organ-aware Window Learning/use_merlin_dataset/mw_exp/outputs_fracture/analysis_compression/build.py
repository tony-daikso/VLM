"""Build train+val-only table for compression error analysis. Test ids are dropped before any label/report is parsed."""
import json, re, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path.home() / "Desktop/Tony/VLM"
MW = H / "OWL/mw_exp"
OUT = MW / "outputs_fracture/analysis_compression"
spl = pd.read_csv(H / "OWL/data/split.csv").set_index("study_id")["split"]
keep = set(spl.index[spl.isin(["train", "val"])])
SID = re.compile(r'"study_id":\s*"([^"]+)"')
rows = {}
for l in open(H / "OWL/data/labels_disease14_all.jsonl"):
    m = SID.search(l[:120])
    if not m or m.group(1) not in keep:
        continue
    r = json.loads(l); a = r["answer"]["fracture"]; d = a.get("detail") or {}
    rows[r["study_id"]] = dict(cat=a["category"], comp=d.get("vertebral_compression"), sites="|".join(d.get("sites") or []),
                               acute=d.get("acute"), p=a.get("p"), quotes=" || ".join(a.get("quotes") or []), why=a.get("why", ""))
lab = pd.DataFrame.from_dict(rows, orient="index")
rep = json.load(open(H / "merlin_disease14/reports.json"))
rep = {k: v["report"] for k, v in rep.items() if k in keep}
lab["report"] = pd.Series(rep)
lab["split"] = spl
assert set(lab.split) <= {"train", "val"}
feat = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(MW / "outputs_fracture/fracture_features_*.csv")))]).drop_duplicates("id", keep="last").set_index("id")
feat = feat[feat.index.isin(keep)]
z = np.load(MW / "outputs_fracture/bones/final_F1/groups/bone_ens.npz", allow_pickle=True)
s = pd.Series(z["logit"], index=z["ids"]); assert set(z["split"]) <= {"train", "val"}
lab["bone_ens"] = s
z = np.load(MW / "outputs_fracture/bones/bone_matrix.npz", allow_pickle=True, mmap_mode=None)
Pm = pd.DataFrame(z["P"], index=z["ids"], columns=list(z["units"]))
lab = lab.join(Pm.add_prefix("P_"))
meta = pd.read_csv(H / "merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")
lab = lab.join(meta[["Age", "Gender", "slicethickness", "manufacturer"]])
lab = lab.join(feat)
lab.index.name = "id"
lab.to_pickle(OUT / "table_trval.pkl")
print(lab.shape, lab.split.value_counts().to_dict(), lab.bone_ens.notna().sum(), lab.n_levels.notna().sum())
