import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score as A
D = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/analysis_gallstones/"
lab = pd.read_csv(D+"labels_trval.csv"); hu = pd.concat([pd.read_csv(D+"hu_a6.csv"), pd.read_csv(D+"hu_ct.csv")]).drop_duplicates("study_id")
meta = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv").drop_duplicates("study id").rename(columns={"study id":"study_id"})
meta = meta[meta.study_id.isin(set(lab.study_id))][["study_id","Age","Gender","contrast","phase"]]
df = lab.merge(hu,on="study_id",how="left").merge(meta,on="study_id",how="left")
T = df[df.split=="train"].merge(pd.read_csv(D+"oof_train.csv"),on="study_id"); T=T[T.y_gs.notna()].reset_index(drop=True)
V = df[df.split=="val"].merge(pd.read_csv(D+"scores_val_descriptive.csv"),on="study_id"); V=V[V.y_gs.notna()].reset_index(drop=True)
y=T.y_gs.values.astype(int); S=T.E3_late.values
ch=(y==0)&(T.y_chole==1).values
print("pos vs chole-neg only AUROC", round(A(y[(y==1)|ch],S[(y==1)|ch]),4))
print("mean E3 logit: pos %.2f negGB %.2f negChole %.2f posMaskMissing %.2f"%(S[y==1].mean(),S[(y==0)&~ch].mean(),S[ch].mean(),S[(y==1)&(T.gb_vox<100).values].mean()))
T["vis"]=(T.gb_maxblob100>=2)
ng=(y==0)&~ch&(T.gb_vox>=100).values
T["ph"]=np.where(T.contrast==False,"non_contrast",T.phase)
print("vis100 rate in GB-present neg by phase:",T[ng].groupby("ph").vis.mean().round(3).to_dict())
print("vis100 rate in pos by phase:",T[y==1].groupby("ph").vis.mean().round(3).to_dict())
for t in (130,200):
    print(f"gb_n{t}>=2 rate pos {(T.gb_n130>=2)[y==1].mean() if t==130 else (T.gb_n200>=2)[y==1].mean():.3f} neg {(T.gb_n130>=2)[ng].mean() if t==130 else (T.gb_n200>=2)[ng].mean():.3f}")
print("gb_max median pos/neg(GB):", T.gb_max[y==1].median(), T.gb_max[ng].median())
HU=["gb_vox","gb_p50","gb_p90","gb_p95","gb_p99","gb_max","gb_std","gb_n60","gb_n100","gb_n130","gb_n200","gb_n_med40","gb_n_lowfat","shell_n100","shell_p99","gb_maxblob100","liver_hu_med","aorta_hu_med"]
def X(d):
    x=d[HU].copy()
    for c in HU:
        if c.startswith(("gb_n","gb_vox","shell_n","gb_max")): x[c]=np.log1p(x[c].clip(lower=0))
    x=x.fillna(T[HU].median()); x["gb_present"]=(d.gb_vox>=100).astype(int); x["E3_late"]=d.E3_late; return x
m=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=3000,class_weight="balanced")).fit(X(T),y)
yv=V.y_gs.values.astype(int)
print("VAL descriptive: my-recipe E3_late %.4f ; stack(HU+E3_late, fit on train OOF) %.4f"%(A(yv,V.E3_late),A(yv,m.decision_function(X(V)))))
inv=((y==1)&~T.vis.values)|(y==0)
p=pd.read_csv(D+"per_case_train_oof.csv").p_stack_e3.values
print("invisible-pos AUROC: E3 %.4f stack %.4f"%(A(y[inv],S[inv]),A(y[inv],p[inv])))
coef=pd.Series(m[-1].coef_[0],index=X(T).columns).round(3).sort_values(); print(coef.to_dict())
