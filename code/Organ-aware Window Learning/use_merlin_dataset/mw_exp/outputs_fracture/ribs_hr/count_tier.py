import pandas as pd, os
d=pd.read_csv("ids_prio.csv"); h={f[:-4] for f in os.listdir("cpr") if f.endswith(".npz") and ".tmp" not in f}
d["h"]=d.study_id.isin(h); print(d.groupby(["tier","split"]).h.agg(["sum","count"]).to_string().replace("\n"," | "))
