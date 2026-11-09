"""Exp N: stack the focal-steatosis liver-site unit heads (nu_probe.py fst: U, U_org) onto the steatosis_split REF for
局部脂肪浸潤 (y_focal): E3_late (liver||(liver-spleen)) logit + HU + FL features, candidates of hs_split_stack.py.
REF rebuilt with hs_split_stack's own loaders/design (checked vs steatosis_split/oof_train_y_focal.csv / val_scores).
New sets: REF + each unit head, REF + all unit heads (all 6 candidates, train 5-fold OOF selection on the y_focal folds,
full-train refit -> val once). Metrics: AUROC + CI, paired bootstrap delta vs REF (train OOF, val), age-matched AUROC
(+ paired delta on val), Spearman(score, age) on negatives, sensitivity at 90 % specificity. Age/sex never inputs.
usage: nu_fst_stack.py STACKDATA_DIR UNIT_DIR      (UNIT_DIR/out/{oof_train,val_scores}.csv; writes UNIT_DIR/stack/)"""
import sys, os, json, numpy as np, pandas as pd
SS = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/steatosis_split/"
sys.path.insert(0, SS + "scripts"); sys.path.insert(0, "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/integrated_best/scripts")
import hs_split_stack as hs
from ib_core import boot_delta, spearman_neg
D, UD = sys.argv[1], sys.argv[2].rstrip("/") + "/"
T, V = hs.load(D, SS + "oof", "y_focal", "concat_diff"); logit = "E3_late_concat_diff"
uo = pd.read_csv(UD + "out/oof_train.csv"); uv = pd.read_csv(UD + "out/val_scores.csv")
UH = [c for c in uo.columns if c not in ("study_id", "fold")]
chk = T[["study_id", "fold"]].merge(uo[["study_id", "fold"]].rename(columns={"fold": "f2"}), on="study_id")
assert len(chk) == len(T) and (chk.fold == chk.f2).all(), "fold mismatch"
T = T.merge(uo[["study_id"] + UH].rename(columns={c: "fn_" + c for c in UH}), on="study_id", how="left")
V = V.merge(uv[["study_id"] + UH].rename(columns={c: "fn_" + c for c in UH}), on="study_id", how="left")
UH = ["fn_" + c for c in UH]
y, yv = T.y.values, V.y.values
print(f"train {len(T)} ({y.sum()} pos) val {len(V)} ({yv.sum()} pos); head coverage {T[UH].notna().all(1).mean():.4f} {V[UH].notna().all(1).mean():.4f}", flush=True)


def design(d, cols, med=None):
    X, m0 = hs.design(d, "E3+HU+FL", logit, None if med is None else med[0])
    E = d[cols].astype(float).reset_index(drop=True) if cols else pd.DataFrame(index=range(len(d)))
    m1 = E.median() if med is None else med[1]
    X = pd.concat([X.reset_index(drop=True), E.fillna(m1)], axis=1)
    return X, (m0, m1)


def run(cols, cn):
    s = np.zeros(len(T))
    for k in sorted(T.fold.unique()):
        tr, te = T.fold.values != k, T.fold.values == k
        Xtr, md = design(T[tr], cols); Xte, _ = design(T[te], cols, md)
        s[te] = hs.score(hs.CANDS[cn]().fit(Xtr, y[tr]), Xte)
    X, md = design(T, cols); m = hs.CANDS[cn]().fit(X, y); Xv, _ = design(V, cols, md)
    return s, hs.score(m, Xv)


ref = run([], "GBM_lr.03_l7_m40_500")
ro = pd.read_csv(SS + "oof_train_y_focal.csv"); rv = pd.read_csv(SS + "val_scores_y_focal.csv")
print(f"REF rebuilt OOF {hs.auc(y, ref[0]):.4f} val {hs.auc(yv, ref[1]):.4f} | steatosis_split files OOF {hs.auc(ro.y, ro['E3+HU+FL']):.4f} val {hs.auc(rv.y, rv['E3+HU+FL']):.4f}", flush=True)
sets = {"REF": []}
for h in UH:
    sets[f"REF＋{h[3:]}"] = [h]
sets["REF＋全部單元頭"] = UH
S, rows = {}, []
for nm, cols in sets.items():
    if nm == "REF":
        S[nm] = (hs.auc(y, ref[0]), "GBM_lr.03_l7_m40_500", *ref); continue
    best = None
    for cn in hs.CANDS:
        o, v = run(cols, cn); a = hs.auc(y, o)
        print(f"  {nm:<24s} {cn:<24s} OOF {a:.4f} val {hs.auc(yv, v):.4f}", flush=True)
        if best is None or a > best[0]: best = (a, cn, o, v)
    S[nm] = best
for h in UH:  # head alone (descriptive)
    S[f"{h[3:]} 單獨"] = (hs.auc(y, T[h].fillna(T[h].median()).values), "probe", T[h].fillna(T[h].median()).values, V[h].fillna(T[h].median()).values)
bo, bv = S["REF"][2], S["REF"][3]
for nm, (a, cn, o, v) in S.items():
    r = dict(model=nm, cand=cn, **hs.metrics(y, o, T.Age, yv, v, V.Age, None if nm == "REF" else bo, None if nm == "REF" else bv))
    r["spearman_neg_val"] = spearman_neg(yv, v, V.Age.values); r["spearman_neg_oof"] = spearman_neg(y, o, T.Age.values)
    if nm != "REF":
        r["d_val"] = boot_delta(yv, bv, v); r["d_oof"] = boot_delta(y, bo, o)
        r["d_val_age"] = boot_delta(yv, bv, v, n=500, age=V.Age.values)
    rows.append(r)
    print(f"{nm:<24s} OOF {r['oof']:.4f} val {r['val']:.4f} {np.round(r['val_ci'], 3)} age {r['val_age']:.4f} sp {r['spearman_neg_val']:+.3f} s90 {r['val_sens90']:.3f}"
          + (f" dVal {r['d_val']['mean']:+.4f} {np.round(r['d_val']['ci'], 4)} dOOF {r['d_oof']['mean']:+.4f} {np.round(r['d_oof']['ci'], 4)} dAge {r['d_val_age']['mean']:+.4f} {np.round(r['d_val_age']['ci'], 4)}" if nm != "REF" else ""), flush=True)
sel = max([r for r in rows if r["model"].startswith("REF＋")], key=lambda r: r["oof"])
print(f"SELECTED (train OOF): {sel['model']} OOF {sel['oof']:.4f} val {sel['val']:.4f}", flush=True)
os.makedirs(UD + "stack", exist_ok=True)
json.dump(dict(results=rows, selected=sel["model"]), open(UD + "stack/results.json", "w"), indent=1, ensure_ascii=False, default=float)
pd.DataFrame({"study_id": T.study_id, "fold": T.fold, "y": y, **{k: v[2] for k, v in S.items()}}).to_csv(UD + "stack/scores_oof.csv", index=False)
pd.DataFrame({"study_id": V.study_id, "y": yv, **{k: v[3] for k, v in S.items()}}).to_csv(UD + "stack/scores_val.csv", index=False)
print("NU_FST_STACK_DONE", flush=True)
