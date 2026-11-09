import sys, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score as auc
lab=pd.read_csv("labels.csv").set_index("study_id"); r=pd.read_csv("ref_train.csv").set_index("study_id")
for run in sys.argv[1:]:
    s=pd.concat([pd.read_csv(f"{run}/score_fold{k}.csv") for k in range(5)]); s=s[s.split=="train"].set_index("study_id")["ft"]
    d=lab.loc[s.index].join(r); d["sc"]=s; d=d[d.t_rib.notna()&d.L_r.notna()]
    z=lambda v:(v-v.mean())/v.std()
    print(run,len(d),"sc %.4f L_r %.4f S2rib %.4f corr %.3f  L_r+sc %.4f"%(auc(d.t_rib,d.sc),auc(d.t_rib,d.L_r),auc(d.t_rib,d.S2_rib),np.corrcoef(d.sc,d.L_r)[0,1],auc(d.t_rib,z(d.sc)+z(d.L_r))),flush=True)
