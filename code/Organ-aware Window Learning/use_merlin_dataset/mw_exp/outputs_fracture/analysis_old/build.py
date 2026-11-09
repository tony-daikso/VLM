"""Old (chronic/healed) fracture error analysis: build train+val table. Test ids dropped before any label/report line is parsed."""
import json, re, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path.home() / "Desktop/Tony/VLM"
MW = H / "OWL/mw_exp"; B = MW / "outputs_fracture/bones"
OUT = MW / "outputs_fracture/analysis_old"
sys.path.insert(0, str(MW / "outputs_fracture/analysis_rib"))  # read-only reuse of report parsers
from parse import rib_info, pelv_info
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
                               acute=d.get("acute"), p=a.get("p"), quotes=" || ".join(a.get("quotes") or []), why=a.get("why", ""),
                               osteopenia=(r["answer"].get("osteopenia") or {}).get("category"))
lab = pd.DataFrame.from_dict(rows, orient="index")
rep = json.load(open(H / "merlin_disease14/reports.json"))
lab["report"] = pd.Series({k: v["report"] for k, v in rep.items() if k in keep}); del rep
lab["split"] = spl
assert set(lab.split) <= {"train", "val"}


def ld(f, key="logit"):
    z = np.load(f, allow_pickle=True); assert set(z["split"]) <= {"train", "val"}
    return pd.Series(z[key], index=z["ids"])


lab["s_bone_ens"] = ld(B / "final_F1/groups/bone_ens.npz")
lab["s_ens_all"] = ld(B / "final_G/ens_all/groups/bone_ens.npz")
fs = sorted(glob.glob(str(B / "rerun/std/oof/oof_bone_E_ref__E3_late_s*.npz")))
lab["s_e3late"] = pd.Series(np.mean([np.load(f)["logit"] for f in fs], 0), index=np.load(fs[0])["ids"])
ms = sorted(glob.glob(str(B / "final_G/members/oof_bone_E_ref__site_max-rib18__mean__all3*.npz")))
if ms:
    SL = np.mean([np.load(f)["site_logit"] for f in ms], 0); ids = np.load(ms[0])["ids"]
    for k, n in enumerate(["vert", "rib", "pelv"]):
        lab[f"s_site_{n}"] = pd.Series(SL[:, k], index=ids)
meta = pd.read_csv(H / "merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")
lab = lab.join(meta[["Age", "Gender"]])
feat = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(MW / "outputs_fracture/fracture_features_*.csv")))]).drop_duplicates("id", keep="last").set_index("id")
lab = lab.join(feat[feat.index.isin(keep)])
fov = pd.read_csv(MW / "outputs_fracture/analysis_rib/mask_fov_stats.csv").set_index("id")
fov = fov[fov.index.isin(keep)]
fcols = [c for c in fov.columns if c.startswith("v_rib_") or c.startswith("v_") and any(k in c for k in ["sacrum", "hip", "femur", "T8", "T12", "L5"])]
lab = lab.join(fov[fcols])
ri = lab.report.fillna("").map(rib_info).apply(pd.Series)
pi = lab.report.fillna("").map(pelv_info).apply(pd.Series)
lab = lab.join(ri[["rib_levels", "n_levels", "lv_min", "lv_max", "side", "t_old", "t_acute"]].add_prefix("rib_"))
lab = lab.join(pi[["p_hardware", "p_old", "p_acute"]])
lab.index.name = "id"
lab.to_pickle(OUT / "table_old.pkl")
print(lab.shape, lab.split.value_counts().to_dict(), lab.s_bone_ens.notna().sum())
