"""Assemble per-disease OWL scores for report generation (train = 5-fold OOF, val = full-train model).
Read-only on mw_exp outputs; test is never read (split filter first). Run on A6000:
  ~/miniconda3/envs/radar/bin/python assemble_scores.py OUT_DIR
Writes OUT_DIR/scores_trval.csv (study_id, split, <disease> logit ..., <disease>_src), OUT_DIR/sources.json,
OUT_DIR/check.csv (train-OOF / val AUROC per disease vs the binary OWL label, coverage)."""
import sys, os, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

MW = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/"
OF = MW + "outputs_final/"
OUT = sys.argv[1].rstrip("/") + "/"; os.makedirs(OUT, exist_ok=True)
spl = pd.read_csv("/home/advantech/Desktop/Tony/VLM/OWL/data/split.csv")
spl = spl[spl.split.isin(["train", "val"])].reset_index(drop=True)          # test rows dropped before anything else
lab = pd.read_csv(OF + "integrated_best/work/base/labels14_trval.csv")

D14 = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "hydronephrosis", "abdominal_aortic_aneurysm",
       "atherosclerosis", "osteopenia", "bowel_obstruction", "appendicitis", "ascites", "free_air", "gallstones",
       "biliary_ductal_dilation", "fracture"]


def two(oof_f, oof_c, val_f, val_c):
    o = pd.read_csv(oof_f)[["study_id", oof_c]].rename(columns={oof_c: "s"}); o["part"] = "train"
    v = pd.read_csv(val_f)[["study_id", val_c]].rename(columns={val_c: "s"}); v["part"] = "val"
    return pd.concat([o, v])


IB = OF + "integrated_best/scores/"
RU = OF + "region_units/"
SRC = {
    "hepatic_steatosis": ("integrated_best stack (E3_late+肝‖(肝−脾) head+肝脾 HU, GBM), RESULTS_BEST 正式最佳 val .832",
                          lambda: two(IB + "hepatic_steatosis_oof.csv", "stack_oof", IB + "hepatic_steatosis_val.csv", "stack")),
    "surgically_absent_gallbladder": ("integrated_best stack（正式最佳 .984）",
                          lambda: two(IB + "surgically_absent_gallbladder_oof.csv", "stack_oof", IB + "surgically_absent_gallbladder_val.csv", "stack")),
    "splenomegaly": ("integrated_best stack（正式最佳 .981）",
                     lambda: two(IB + "splenomegaly_oof.csv", "stack_oof", IB + "splenomegaly_val.csv", "stack")),
    "hydronephrosis": ("region_units/hydronephrosis REF＋全部單元頭（正式最佳 .957）",
                       lambda: two(RU + "hydronephrosis/stack/scores_oof.csv", "REF＋全部單元頭", RU + "hydronephrosis/stack/scores_val.csv", "REF＋全部單元頭")),
    "abdominal_aortic_aneurysm": ("integrated_best stack（正式最佳 .990）",
                                  lambda: two(IB + "abdominal_aortic_aneurysm_oof.csv", "stack_oof", IB + "abdominal_aortic_aneurysm_val.csv", "stack")),
    "atherosclerosis": ("region_units/atherosclerosis REF＋全部單元頭＋HU（正式最佳 .921）",
                        lambda: two(RU + "atherosclerosis/stack/scores_oof.csv", "REF＋全部單元頭＋HU", RU + "atherosclerosis/stack/scores_val.csv", "REF＋全部單元頭＋HU")),
    "osteopenia": ("osteopenia_v15 5win+v15（正式最佳 .890）",
                   lambda: two(OF + "osteopenia_v15/stack_oof_train.csv", "5win+v15", OF + "osteopenia_v15/stack_val.csv", "5win+v15")),
    "bowel_obstruction": ("integrated_best stack（正式最佳 .956）",
                          lambda: two(IB + "bowel_obstruction_oof.csv", "stack_oof", IB + "bowel_obstruction_val.csv", "stack")),
    "appendicitis": ("appendicitis_rlq_unit REF＋U_rlq_cs（原合併定義，正式最佳 .920；stack 分數沒存檔，依 rlq_eval.py 同樣規則重算 LR 堆疊）",
                     None),
    "ascites": ("region_units/ascites REF＋U_org（正式最佳 .916）",
                lambda: two(RU + "ascites/stack/scores_oof.csv", "REF＋U_org", RU + "ascites/stack/scores_val.csv", "REF＋U_org")),
    "free_air": ("region_units/free_air REF＋全部單元頭（正式最佳 .912）",
                 lambda: two(RU + "free_air/stack/scores_oof.csv", "REF＋全部單元頭", RU + "free_air/stack/scores_val.csv", "REF＋全部單元頭")),
    "gallstones": ("region_units/gallstones Exp N G5c＋U = N_U（保守正式值 .881；.936 的 gs15_ft 只在 4,028 例 train 子集有 OOF、全量版未完成，不採用）",
                   lambda: two(RU + "gallstones/stack/oof_train.csv", "N_U", RU + "gallstones/stack/val_scores.csv", "N_U")),
    "biliary_ductal_dilation": ("integrated_best BD5 stack（正式最佳 .911）",
                                lambda: two(IB + "biliary_ductal_dilation_oof.csv", "stack_oof", IB + "biliary_ductal_dilation_val.csv", "stack")),
    "fracture": ("outputs_fracture/push90b a6x10w2_tv7 S3n·max（正式最佳 .897；train = 巢狀 5-fold OOF，FT encoder 也是 5 fold 各自微調）",
                 None),
}


def appendicitis():
    """Reproduce rlq_eval.py's REF＋U_rlq_cs stack for target y_ap (same folds, LR C=0.1 balanced, median fill)."""
    U = OF + "appendicitis_rlq_unit/"; O = U + "out/"
    oof = pd.read_csv(O + "oof_train.csv"); val = pd.read_csv(O + "val_scores.csv")
    alab = pd.read_csv(OF + "appendicitis_split/labels_trval_split.csv")
    hu = pd.concat([pd.read_csv(OF + "analysis_appendicitis/hu_a6.csv"), pd.read_csv(OF + "analysis_appendicitis/hu_ct.csv")]).drop_duplicates("study_id")
    LOCAL = [c for c in hu.columns if c.startswith("dlr_") or c.startswith("dglob_")] + \
            [f"rlq_r{r}_{f}" for r in (15, 30) for f in ("fat_med", "fat_p75", "strand_frac", "fluid_frac", "soft_frac", "fat_frac")] + ["rlq_wall_p50", "rlq_wall_p90"]
    hu = hu[["study_id"] + LOCAL]
    T = oof.merge(alab, on="study_id", how="left").merge(hu, on="study_id", how="left")
    V = val.merge(alab, on="study_id", how="left").merge(hu, on="study_id", how="left")
    tg = "y_ap"
    Tm = T[T[tg].notna() & T[f"{tg}__E3_late"].notna()].reset_index(drop=True)
    Vm = V[V[f"{tg}__E3_late"].notna()].reset_index(drop=True)            # val: score every row (label not needed)
    cols = [f"{tg}__E3_late", f"{tg}__E3cs", f"{tg}__U_rlq_cs"] + LOCAL
    y = Tm[tg].values.astype(int)
    X = Tm[cols]; med = X.median(); X = X.fillna(med); XV = Vm[cols].fillna(med)
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
    p = np.zeros(len(Tm))
    for k in range(5):
        a, b = (Tm.fold != k).values, (Tm.fold == k).values
        p[b] = lr().fit(X[a], y[a]).decision_function(X[b])
    pv = lr().fit(X, y).decision_function(XV)
    o = pd.DataFrame({"study_id": Tm.study_id, "s": p, "part": "train"})
    v = pd.DataFrame({"study_id": Vm.study_id, "s": pv, "part": "val"})
    return pd.concat([o, v])


def fracture():
    z = np.load(MW + "outputs_fracture/push90b/integ/a6x10w2_tv7/scores.npz", allow_pickle=True)
    d = pd.DataFrame({"study_id": z["ids"], "part": z["split"], "s": z["selected"]})
    return d[d.part.isin(["train", "val"])]


out = spl.copy(); rows = []; src = {}
for d in D14:
    desc, fn = SRC[d]
    s = appendicitis() if d == "appendicitis" else fracture() if d == "fracture" else fn()
    s = s.drop_duplicates("study_id")
    m = spl.merge(s, on="study_id", how="left")
    bad = m.part.notna() & (m.part != m.split)
    assert not bad.any(), (d, int(bad.sum()))
    out[d] = m.s.values
    src[d] = desc
    L = spl.merge(lab[["study_id", d]], on="study_id", how="left")
    for sp in ("train", "val"):
        mk = (L.split == sp).values
        y = L[d].values[mk]; sc = out[d].values[mk]; ok = ~np.isnan(y) & ~np.isnan(sc)
        rows.append(dict(disease=d, split=sp, n=int(mk.sum()), n_scored=int((~np.isnan(sc)).sum()),
                         n_labeled=int((~np.isnan(y)).sum()), n_labeled_unscored=int((~np.isnan(y) & np.isnan(sc)).sum()),
                         pos=int(np.nansum(y)), auroc=roc_auc_score(y[ok], sc[ok])))
    print(rows[-2], "\n", rows[-1], flush=True)

out.to_csv(OUT + "scores_trval.csv", index=False)
pd.DataFrame(rows).to_csv(OUT + "check.csv", index=False)
json.dump(src, open(OUT + "sources.json", "w"), ensure_ascii=False, indent=1)
print("done", out.shape)
