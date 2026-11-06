"""Reference: the original combined steatosis finding (quickwins scores), and the same combined scores evaluated
on the new split labels (no retraining). Train OOF / val only."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); from hs_split_stack import metrics, auc
D, O = sys.argv[2], sys.argv[3]
lab = pd.read_csv(D + "/labels_trval_split.csv").set_index("study_id")
age = pd.read_csv(D + "/meta_trval.csv").set_index("study_id").Age
T = pd.read_csv(D + "/oof_train_hs.csv"); V = pd.read_csv(D + "/val_scores_hs.csv")
T = T[T.study_id.isin(lab.index[lab.split == "train"])]; V = V[V.study_id.isin(lab.index[lab.split == "val"])]
out = {}; rows = []
for tgt, zh in [("y_hs", "原合併脂肪肝"), ("y_diffuse", "瀰漫性脂肪肝"), ("y_focal", "局部脂肪浸潤")]:
    t = T.assign(yy=lab.loc[T.study_id, tgt].values).dropna(subset=["yy"]); v = V.assign(yy=lab.loc[V.study_id, tgt].values).dropna(subset=["yy"])
    for sc, nm in [("E3_late", "E3_late（肝，原合併標籤訓練）"), ("stack", "E3_late+HU GBM（原合併標籤訓練）")]:
        r = metrics(t.yy.values.astype(int), t[sc].values, age.reindex(t.study_id).values, v.yy.values.astype(int), v[sc].values, age.reindex(v.study_id).values)
        out[f"{tgt}|{sc}"] = r; rows.append(dict(finding=zh, target=tgt, model=nm, **r))
    if tgt == "y_hs":  # liver||(liver-spleen)+HU: val only (no OOF saved in quickwins)
        yv = v.yy.values.astype(int); s = v.stack_concat.values
        from hs_split_stack import boot, age_matched_auc, sens90
        rows.append(dict(finding=zh, target=tgt, model="肝‖(肝−脾)+HU GBM（原合併；OOF 取自 quickwins 0.841）", oof=0.841, val=auc(yv, s),
                         val_ci=boot(yv, s)["ci"], val_age=age_matched_auc(yv, s, age.reindex(v.study_id).values), val_sens90=sens90(yv, s)[0]))
pd.DataFrame(rows).to_csv(O + "/results_reference_combined.csv", index=False)
for r in rows: print(r["finding"], r["model"], round(r["oof"], 4), round(r["val"], 4), np.round(r["val_ci"], 3), round(r.get("val_age", np.nan), 4))
