import pandas as pd, numpy as np, os
A=os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/analysis_gallstones/")
P=os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/outputs/prep/")
T=pd.read_csv(A+"per_case_train_oof.csv")
H=pd.concat([pd.read_csv(A+"hu_a6.csv"),pd.read_csv(A+"hu_ct.csv")]).drop_duplicates("study_id")
T=T.merge(H[["study_id","gb_max","gb_n130","gb_n200"]],on="study_id",how="left")
a6=set(l.strip() for l in open(A+"ids_a6.txt"))
T=T[T.study_id.isin(a6)]
T=T[[os.path.exists(P+s+".npz") for s in T.study_id]]
y=T.y_gs.values
Tall=pd.read_csv(A+"per_case_train_oof.csv")
thr=np.quantile(Tall.E3_late[Tall.y_gs==0],0.9)
print("thr",thr, "n a6 train with prep",len(T))
T["pred"]=T.E3_late>=thr
fn=T[T.fn_cat.notna()&(T.fn_cat!="")]; fp=T[T.fp_cat.notna()&(T.fp_cat!="")]
sel=[]
def take(df,k,tag,sortcol,asc):
    d=df.sort_values(sortcol,ascending=asc)
    # take from the 20-40% range for "typical but clear"
    for _,r in d.head(k).iterrows(): sel.append((tag,r))
F=fn[fn.fn_cat.str.startswith("F")&(fn.gb_vox>800)]; take(F,2,"FN-F","E3_late",True)
G=fn[fn.fn_cat.str.startswith("G")]; take(G,2,"FN-G","gb_n200",False)
B=fn[fn.fn_cat.str.startswith("B")]; take(B,1,"FN-B","E3_late",True)
D=fn[fn.fn_cat.str.startswith("D")&(fn.gb_vox>300)]; take(D,1,"FN-D","E3_late",True)
Fb=fp[fp.fp_cat.str.startswith("b")]; take(Fb,1,"FP-b200","gb_n200",False)
take(Fb[(Fb.gb_n200==0)&(Fb.gb_n130>0)],1,"FP-bPV","E3_late",False)
ch=fp[(fp.y_chole==1)]; print("FP chole n",len(ch), "FP gbvox<100", (fp.gb_vox<100).sum())
if len(ch): take(ch,1,"FP-chole","E3_late",False)
else: take(fp[fp.gb_vox<300],1,"FP-small","E3_late",False)
take(fp[fp.fp_cat.str.startswith("f")],1,"FP-f","E3_late",False)
tp=T[(y==1)&T.pred&(T.gb_n200>=20)]; take(tp,2,"TP","E3_late",False)
tn=T[(y==0)&~T.pred&(T.gb_vox>800)&(T.y_chole==0)&(T.gb_n130==0)]; take(tn,2,"TN","E3_late",True)
rows=[]
for tag,r in sel:
    rows.append(dict(tag=tag,study_id=r.study_id,y=int(r.y_gs),y_chole=int(r.y_chole),E3=r.E3_late,p=1/(1+np.exp(-r.E3_late)),thr=thr,cat=(r.fn_cat if isinstance(r.fn_cat,str) and r.fn_cat else (r.fp_cat if isinstance(r.fp_cat,str) else "")),gb_vox=r.gb_vox,gb_max=r.gb_max,n130=r.gb_n130,n200=r.gb_n200,gs_cat=r.gs_cat,gs_calc=r.gs_calc,phase=r.phase2))
S=pd.DataFrame(rows); S.to_csv(os.path.expanduser("~/gs_gallery_tmp/sel.csv"),index=False)
print(S.drop(columns="study_id").to_string())
