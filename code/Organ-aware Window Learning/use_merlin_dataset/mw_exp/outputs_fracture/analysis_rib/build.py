"""Train+val table for the rib / pelvis error analysis. Test ids are dropped before any label/report line is parsed."""
import json, re, glob
from pathlib import Path
import numpy as np, pandas as pd
H = Path.home() / "Desktop/Tony/VLM"
MW = H / "OWL/mw_exp"; B = MW / "outputs_fracture/bones"
OUT = MW / "outputs_fracture/analysis_rib"
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
    return pd.Series(z[key] if z[key].ndim == 1 else list(z[key]), index=z["ids"])
lab["s_bone_ens"] = ld(B / "final_F1/groups/bone_ens.npz")
for g in ["ref__site_max__mean__all3", "ref__site_max__max__all3", "ref__site_max__mean__bone", "ref__site_lse__mean__all3"]:
    lab["s_" + g] = ld(B / f"final_F1/groups/{g}.npz")
lab["s_G_any"] = ld(B / "final_G/ens_G/groups/ref__site_max-rib18__mean__all3.npz")
lab["s_ens_all"] = ld(B / "final_G/ens_all/groups/bone_ens.npz")
ms = sorted(glob.glob(str(B / "final_G/members/oof_bone_E_ref__site_max-rib18__mean__all3*.npz")))
SL = np.mean([np.load(f)["site_logit"] for f in ms], 0); ids = np.load(ms[0])["ids"]
for k, n in enumerate(["vert", "rib", "pelv"]):
    lab[f"s_Gsite_{n}"] = pd.Series(SL[:, k], index=ids)
fs = sorted(glob.glob(str(B / "rerun/std/oof/oof_bone_E_ref__E3_late_s*.npz")))
lab["s_e3late"] = pd.Series(np.mean([np.load(f)["logit"] for f in fs], 0), index=np.load(fs[0])["ids"])
z = np.load(B / "final_F1/final/scores.npz", allow_pickle=True); assert set(z["split"]) <= {"train", "val"}
lab["s_stack"] = pd.Series(z["stack"], index=z["ids"])
z = np.load(B / "bone_matrix.npz", allow_pickle=True)
lab = lab.join(pd.DataFrame(z["P"], index=z["ids"], columns=["P_" + u for u in z["units"]]))
z = np.load(B / "rib_matrix.npz", allow_pickle=True)
lab = lab.join(pd.DataFrame(z["P"], index=z["ids"], columns=["PR_" + u for u in z["units"]]))
lab["rib_emb_missing"] = lab.index.isin(set(z["missing"]))
meta = pd.read_csv(H / "merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")
lab = lab.join(meta[["Age", "Gender", "slicethickness", "manufacturer", "phase"]])
fov = pd.read_csv(OUT / "mask_fov_stats.csv", index_col=0).drop(columns=["split"])
lab = lab.join(fov)
lab.index.name = "id"
lab.to_pickle(OUT / "table_trval.pkl")
print(lab.shape, lab.split.value_counts().to_dict(), lab.s_bone_ens.notna().sum(), lab.nz.notna().sum())
