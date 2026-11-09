import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from pathlib import Path
from scipy.stats import spearmanr
from age_shortcut import matched_auc
from bone_probe import LABELS, load_labels, trval_ids
from fracture_eval_oof import boot_diff
from probe import auroc
B = Path("../outputs_fracture/bones")
def val(p):
    z = np.load(p, allow_pickle=True); assert set(np.unique(z["split"])) <= {"train", "val"}
    m = z["split"] == "val"; return pd.Series(z["logit"][m], index=np.array(z["ids"])[m])
new, ref = val(B/"rerun/final_ens/groups/bone_ens.npz"), val(B/"final_F1/groups/bone_ens.npz")
spl = trval_ids(); lab = load_labels(LABELS, sorted(spl.index[spl == "val"]))
age = pd.read_csv(Path.home()/"Desktop/Tony/VLM/merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")["Age"]
df = lab.join(age.rename("age")).loc[lambda d: d.index.isin(new.index) & ~d.y.isna() & ~d.age.isna()]
y = df.y.values.astype(int); ag = df.age.values; a = new.reindex(df.index).values; b = ref.reindex(df.index).values
G = {"椎體壓迫性": df.compression.values.astype(bool), "肋骨": df.sites.apply(lambda s: "rib" in s).values,
     "骨盆/薦椎/髖": df.sites.apply(lambda s: any(x in s for x in ("pelvis","sacrum","hip_femur"))).values,
     "急性(含混合)": df.acute.isin(["yes","mixed"]).values, "陳舊": (df.acute == "no").values}
print("N", len(df), "pos", y.sum())
for n, p in (("E3_late OOF 10-seed", a), ("F1 bone_ens", b)):
    print(n, "AUROC %.4f matched %.4f spearman %.3f" % (auroc(y.astype(float), p), matched_auc(y, p, ag, 2.0), spearmanr(p[y==0], ag[y==0])[0]),
          {g: round(auroc(((y==1)&m)[((y==1)&m)|(y==0)].astype(float), p[((y==1)&m)|(y==0)]), 3) for g, m in G.items()})
bd = boot_diff(y, a, b, ag, 2.0, 2000)
print("diff vs .826:", bd)
for g, m in G.items():
    s = ((y==1)&m)|(y==0); d = boot_diff(((y==1)&m)[s].astype(int), a[s], b[s], ag[s], 2.0, 1000)
    print(g, "auc", np.round(d["auc"][:3], 3))
