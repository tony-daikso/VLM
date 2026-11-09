"""Absent-GB (post-cholecystectomy) error analysis on TRAIN 5-fold OOF (E0 / E3_late). Val = descriptive only.
Inputs (same dir): labels_trval.csv, oof_train.csv, scores_val_descriptive.csv, feat_a6.csv, feat_ct.csv, hu_a6.csv, hu_ct.csv
Outputs: results_train_oof.csv, per_case_train_oof.csv, stack_val_descriptive.csv, analyze.log (stdout)"""
import sys, numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
D = sys.argv[1] if len(sys.argv) > 1 else "./"
def A(y, s):
    y = np.asarray(y); s = np.asarray(s)
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
lab = pd.read_csv(D + "labels_trval.csv")
ft = pd.concat([pd.read_csv(D + "feat_a6.csv"), pd.read_csv(D + "feat_ct.csv")]).drop_duplicates("study_id")
hu = pd.concat([pd.read_csv(D + "hu_a6.csv"), pd.read_csv(D + "hu_ct.csv")]).drop_duplicates("study_id")
hu = hu[["study_id", "gb_n130", "gb_n200", "gb_max", "gb_p99", "gb_maxblob100", "liver_hu_med", "aorta_hu_med"]]
df = lab.merge(ft, on="study_id", how="left").merge(hu, on="study_id", how="left")
df["gb_vox"] = df.gb_vox.fillna(0)
df["ph"] = np.where(df.contrast == False, "non_contrast", df.phase)
T = df[df.split == "train"].merge(pd.read_csv(D + "oof_train.csv"), on="study_id")
T = T[T.y_chole.notna()].reset_index(drop=True)
V = df[df.split == "val"].merge(pd.read_csv(D + "scores_val_descriptive.csv"), on="study_id")
V = V[V.y_chole.notna()].reset_index(drop=True)
y = T.y_chole.values.astype(int); S = T.E3_late.values; S0 = T.E0.values
print(f"train n={len(T)} pos={y.sum()} | val n={len(V)} pos={int(V.y_chole.sum())} | feat coverage train {T.liver_vox.notna().mean():.4f}")
R = []
def rec(group, name, mask=None, score=None, note=""):
    m = np.ones(len(T), bool) if mask is None else np.asarray(mask)
    sc = S if score is None else np.asarray(score)
    R.append(dict(group=group, subset=name, n=int(m.sum()), n_pos=int(y[m].sum()), n_neg=int((1 - y[m]).sum()),
                  auroc_E3_late=A(y[m], sc[m]), auroc_E0=A(y[m], S0[m]) if score is None else np.nan, note=note))
# ---------- overall
for c in [c for c in T.columns if c in ("E0", "E3_late") or c.startswith("E1_")]:
    if c.endswith("_gs"): continue
    R.append(dict(group="overall", subset=c, n=len(T), n_pos=int(y.sum()), auroc_E3_late=A(y, T[c]),
                  note=f"val descriptive {A(V.y_chole, V[c]):.4f}" if c in V else ""))
    print(f"{c}: OOF {A(y, T[c]):.4f} | val {A(V.y_chole, V[c]):.4f}" if c in V else "")
# ---------- H1 GB mask volume
gv = T.gb_vox.values
print("\n[H1] GB volume alone AUROC (score=-gb_vox):", round(A(y, -gv), 4))
R.append(dict(group="H1_gb_volume", subset="score = -gb_vox (alone)", n=len(T), n_pos=int(y.sum()), auroc_E3_late=A(y, -gv)))
for thr in (1, 50, 100, 200):
    miss = gv < thr
    print(f"  gb_vox<{thr}: pos {miss[y==1].mean():.3f} ({miss[y==1].sum()}) | neg {miss[y==0].mean():.3f} ({miss[y==0].sum()}) | "
          f"acc-as-rule sens {miss[y==1].mean():.3f} spec {1-miss[y==0].mean():.3f}")
    R.append(dict(group="H1_gb_volume", subset=f"rule gb_vox<{thr}", n=len(T), n_pos=int(y.sum()),
                  note=f"sens {miss[y==1].mean():.3f} spec {1-miss[y==0].mean():.3f}; pos with mask>=thr {int((~miss)[y==1].sum())}; neg w/o mask {int(miss[y==0].sum())}"))
miss = gv < 100; T["gb_missing"] = miss
rec("H1_strata", "GB mask <100 vox (all)", miss); rec("H1_strata", "GB mask >=100 vox (all)", ~miss)
print("  E3 AUROC within missing:", round(A(y[miss], S[miss]), 4), " within present:", round(A(y[~miss], S[~miss]), 4))
print("  mean E3 logit: pos-missing %.2f pos-present %.2f neg-missing %.2f neg-present %.2f" % (
    S[(y == 1) & miss].mean(), S[(y == 1) & ~miss].mean(), S[(y == 0) & miss].mean(), S[(y == 0) & ~miss].mean()))
nm = (y == 0) & miss
print(f"  negatives with mask<100: n={nm.sum()}; of which gallstones+ {T.y_gs[nm].eq(1).mean():.3f}, contracted kw {T.kw_contracted[nm].mean():.3f}, "
      f"gb_present kw {T.kw_gb_present[nm].mean():.3f}, cholecystitis kw {T.kw_cholecystitis[nm].mean():.3f}, ch_cat not_mentioned {T.ch_cat[nm].eq('not_mentioned').mean():.3f}, "
      f"liver_vox<1000 {T.liver_vox[nm].lt(1000).mean():.3f}, gb_vox==0 {(gv[nm]==0).mean():.3f}")
print("  negatives overall: gallstones+ %.3f contracted %.3f" % (T.y_gs[y == 0].eq(1).mean(), T.kw_contracted[y == 0].mean()))
pp = (y == 1) & ~miss
print(f"  positives with mask>=100: n={pp.sum()}; median gb_vox {np.median(gv[pp]):.0f} (neg-present median {np.median(gv[(y==0)&~miss]):.0f}); "
      f"gb_hu_med {np.nanmedian(T.gb_hu_med[pp]):.1f} vs neg {np.nanmedian(T.gb_hu_med[(y==0)&~miss]):.1f}; "
      f"gb_f_fluid {np.nanmedian(T.gb_f_fluid[pp]):.2f} vs {np.nanmedian(T.gb_f_fluid[(y==0)&~miss]):.2f}; gb_f_fat {np.nanmedian(T.gb_f_fat[pp]):.2f} vs {np.nanmedian(T.gb_f_fat[(y==0)&~miss]):.2f}; "
      f"liver_contact {np.nanmedian(T.gb_liver_contact[pp]):.2f} vs {np.nanmedian(T.gb_liver_contact[(y==0)&~miss]):.2f}; bowel_contact {np.nanmedian(T.gb_bowel_contact[pp]):.2f} vs {np.nanmedian(T.gb_bowel_contact[(y==0)&~miss]):.2f}; "
      f"partial kw {T.kw_partial[pp].mean():.3f} liver_tx {T.kw_liver_tx_anywhere[pp].mean():.3f} (all pos {T.kw_liver_tx_anywhere[y==1].mean():.3f})")
for q, (lo, hi) in {"100-300": (100, 300), "300-1000": (300, 1000), ">=1000": (1000, 1e9)}.items():
    m = (gv >= lo) & (gv < hi)
    rec("H1_gb_vox_bin", q, m, note=f"pos {int(y[m].sum())}")
# ---------- H2 clip features
CL = ["f30_n500", "f30_n800", "f30_n1200", "f30_max", "f50_n800", "f50_n1200", "peri_n800", "peri_n1200", "f30_blobs500", "body_metal1500"]
print("\n[H2] clip / metal features (AUROC alone; all | GB-missing stratum | GB-present stratum); pos/neg rate >0")
for c in CL:
    v = T[c].fillna(0).values
    print(f"  {c}: {A(y, v):.3f} | {A(y[miss], v[miss]):.3f} | {A(y[~miss], v[~miss]):.3f} | rate>0 pos {np.mean(v[y==1]>0):.3f} neg {np.mean(v[y==0]>0):.3f}")
    R.append(dict(group="H2_clip_feature", subset=c, n=len(T), n_pos=int(y.sum()), auroc_E3_late=A(y, v),
                  note=f"missing-stratum {A(y[miss], v[miss]):.3f}; present-stratum {A(y[~miss], v[~miss]):.3f}; >0 pos {np.mean(v[y==1]>0):.3f} neg {np.mean(v[y==0]>0):.3f}"))
clip = T.f30_n1200.fillna(0).values > 0; T["clip30"] = clip
for nmk, m in {"clip+ & GB missing": clip & miss, "clip+ & GB present": clip & ~miss, "clip- & GB missing": ~clip & miss, "clip- & GB present": ~clip & ~miss}.items():
    print(f"  {nmk}: n {m.sum()} pos-rate {y[m].mean():.3f}")
    rec("H2_clip_x_mask", nmk, m, note=f"pos rate {y[m].mean():.3f}")
FO = ["f30_f_fat", "f30_f_fluid", "f30_f_soft", "f30_gb_frac", "f30_nonliver"]
for c in FO:
    v = T[c].values; ok = ~np.isnan(v)
    print(f"  fossa {c}: AUROC {A(y[ok], v[ok]):.3f}")
    R.append(dict(group="H2_fossa_content", subset=c, n=int(ok.sum()), n_pos=int(y[ok].sum()), auroc_E3_late=A(y[ok], v[ok])))
# ---------- H3 label noise
print("\n[H3] label-noise candidates")
cands = {
  "neg: ch_cat=not_mentioned": (y == 0) & (T.ch_cat == "not_mentioned"),
  "neg: report kw cholecystectomy/surgically absent in GB sentence": (y == 0) & (T.kw_chole == 1),
  "neg: 'GB not visualized' wording": (y == 0) & ((T.kw_not_vis == 1) | (T.sec_not_vis == 1)),
  "neg: GB mask<100 & no GB-present/contracted/stone wording": (y == 0) & miss & (T.kw_gb_present == 0) & (T.kw_contracted == 0) & (T.y_gs != 1),
  "neg: GB mask<100 & clip+": (y == 0) & miss & clip,
  "pos: no cholecystectomy wording": (y == 1) & (T.kw_chole == 0),
  "pos: partial/subtotal/remnant wording": (y == 1) & (T.kw_partial == 1),
  "pos: uncertain wording": (y == 1) & (T.kw_uncertain == 1),
  "pos: ch_cat probably_yes": (y == 1) & (T.ch_cat == "probably_yes"),
  "pos: gallstones also labelled positive": (y == 1) & (T.y_gs == 1),
  "pos: liver transplant / Whipple etc. in report": (y == 1) & (T.kw_liver_tx_anywhere == 1),
}
pos_s = S[y == 1]; neg_s = S[y == 0]
for k, m in cands.items():
    m = m.values if hasattr(m, "values") else m
    if k.startswith("neg"):
        # AUROC of all positives vs this negative group, and share of negatives ranked above median positive
        sub = (y == 1) | m
        print(f"  {k}: n={m.sum()} | AUROC(all pos vs this) {A(y[sub], S[sub]):.3f} | mean logit {S[m].mean():.2f} | gb_missing {miss[m].mean():.2f}")
        R.append(dict(group="H3_label_noise", subset=k, n=int(m.sum()), auroc_E3_late=A(y[sub], S[sub]), note=f"mean logit {S[m].mean():.2f}; GB mask missing {miss[m].mean():.2f}"))
    else:
        sub = (y == 0) | m
        print(f"  {k}: n={m.sum()} | AUROC(this vs all neg) {A(y[sub], S[sub]):.3f} | mean logit {S[m].mean():.2f} | gb_missing {miss[m].mean():.2f}")
        R.append(dict(group="H3_label_noise", subset=k, n=int(m.sum()), auroc_E3_late=A(y[sub], S[sub]), note=f"mean logit {S[m].mean():.2f}; GB mask missing {miss[m].mean():.2f}"))
# ---------- H5 age
age = T.Age.values.astype(float); ok = ~np.isnan(age)
print(f"\n[H5] age alone AUROC {A(y[ok], age[ok]):.3f}; sex(F) alone {A(y, (T.Gender=='F').astype(int)):.3f}")
R.append(dict(group="H5_age", subset="age alone", n=int(ok.sum()), n_pos=int(y[ok].sum()), auroc_E3_late=A(y[ok], age[ok])))
R.append(dict(group="H5_age", subset="sex=F alone", n=len(T), n_pos=int(y.sum()), auroc_E3_late=A(y, (T.Gender == "F").astype(int))))
for lo, hi in ((0, 40), (40, 60), (60, 75), (75, 200)):
    m = (age >= lo) & (age < hi)
    rec("H5_age_bin", f"{lo}-{hi}", m, note=f"prevalence {y[m].mean():.3f}")
    print(f"  age {lo}-{hi}: n {m.sum()} prev {y[m].mean():.3f} AUROC {A(y[m], S[m]):.4f}")
for g in ("F", "M"):
    rec("subgroup_sex", g, (T.Gender == g).values)
for ph in T.ph.dropna().unique():
    m = (T.ph == ph).values
    if m.sum() >= 30: rec("subgroup_phase", ph, m)
rec("subgroup", "gallstones positive", (T.y_gs == 1).values)
rec("subgroup", "liver_tx/Whipple in report", (T.kw_liver_tx_anywhere == 1).values)
rec("subgroup", "liver mask <1000 vox", (T.liver_vox.fillna(0) < 1000).values)
# ---------- H4 errors at 90% specificity
thr = np.quantile(neg_s, 0.90); T["pred90"] = S >= thr
print(f"\n[H4] 90% spec threshold {thr:.3f}: sens {np.mean(pos_s >= thr):.4f} | E0 sens {np.mean(S0[y==1] >= np.quantile(S0[y==0], 0.9)):.4f}")
fn = (y == 1) & (S < thr); fp = (y == 0) & (S >= thr)
def fn_cat(r):
    if r.kw_chole == 0: return "A 報告無膽囊切除字樣（標註可疑）"
    if r.kw_partial == 1: return "B 部分切除/殘留膽囊"
    if r.gb_vox >= 100 and r.kw_liver_tx_anywhere == 1: return "C 肝移植/Whipple 等術後解剖改變（遮罩落在其他結構）"
    if r.gb_vox >= 100: return "D 有膽囊遮罩（TS 把窩內腸道/液體當膽囊）"
    if r.gb_vox > 0: return "E 遮罩極小(1-99 vox)"
    return "F 遮罩為 0 仍漏判（器官缺失=常數 logit）"
def fp_cat(r):
    if r.kw_chole == 1 or r.ch_cat == "not_mentioned": return "a 標註可疑（報告提到切除／未提及膽囊）"
    if r.gb_vox < 100 and (r.kw_contracted == 1): return "b 膽囊收縮/未擴張 → 遮罩缺失"
    if r.gb_vox < 100 and r.y_gs == 1: return "c 膽囊充滿結石 → 遮罩缺失"
    if r.gb_vox < 100: return "d 其他遮罩缺失（分割失敗）"
    if r.gb_vox < 300: return "e 小遮罩(100-299 vox)"
    return "f 有正常大小遮罩"
T["fn_cat"] = [fn_cat(r) if f else "" for r, f in zip(T.itertuples(), fn)]
T["fp_cat"] = [fp_cat(r) if f else "" for r, f in zip(T.itertuples(), fp)]
allpos = pd.Series([fn_cat(r) for r in T[y == 1].itertuples()]).value_counts(normalize=True)
allneg = pd.Series([fp_cat(r) for r in T[y == 0].itertuples()]).value_counts(normalize=True)
print(f"  FN {fn.sum()}:"); vc = T.fn_cat[fn].value_counts()
for k, v in vc.items():
    print(f"    {k}: {v} ({v/fn.sum():.1%}; base rate among positives {allpos.get(k,0):.1%})")
    R.append(dict(group="H4_FN_at_90spec", subset=k, n=int(v), note=f"{v/fn.sum():.3f} of FN; base {allpos.get(k,0):.3f}"))
print(f"  FP {fp.sum()}:"); vc = T.fp_cat[fp].value_counts()
for k, v in vc.items():
    print(f"    {k}: {v} ({v/fp.sum():.1%}; base rate among negatives {allneg.get(k,0):.1%})")
    R.append(dict(group="H4_FP_at_90spec", subset=k, n=int(v), note=f"{v/fp.sum():.3f} of FP; base {allneg.get(k,0):.3f}"))
# ---------- stacking (same folds as OOF; LR + HGB)
print("\n[stack] train OOF (fold-wise refit on the OOF folds) | val descriptive (fit on all train)")
VOL = ["gb_vox", "gb_ncc", "gb_lcc_frac", "gb_hu_med", "gb_hu_std", "gb_f_fluid", "gb_f_fat", "gb_f_air", "gb_f_dense", "gb_liver_contact", "gb_bowel_contact", "liver_vox"]
FOS = ["f30_f_fat", "f30_f_fluid", "f30_f_soft", "f30_gb_frac", "f30_nonliver"]
CLP = ["f30_n500", "f30_n800", "f30_n1200", "f30_max", "f50_n800", "f50_n1200", "peri_n800", "peri_n1200", "f30_blobs500", "body_metal1500"]
def X(d, cols, base):
    x = d[cols].copy().astype(float)
    for c in cols:
        if c.startswith(("gb_vox", "liver_vox", "f30_n", "f50_n", "peri_n", "body_", "f30_blobs", "f30_nonliver")): x[c] = np.log1p(x[c].clip(lower=0))
    x = x.fillna(T[cols].median() if len(cols) else 0)
    x["gb_missing"] = (d.gb_vox.fillna(0) < 100).astype(int)
    for b in base: x[b] = d[b]
    return x
sets = {"gb_vol_only": (VOL, []), "clip_only": (CLP, []), "vol+fossa+clip": (VOL + FOS + CLP, []),
        "E3_late+vol": (VOL, ["E3_late"]), "E3_late+vol+fossa": (VOL + FOS, ["E3_late"]), "E3_late+vol+fossa+clip": (VOL + FOS + CLP, ["E3_late"]),
        "E0+vol+fossa+clip": (VOL + FOS + CLP, ["E0"]), "E3_late+vol+fossa+clip+age/sex": (VOL + FOS + CLP, ["E3_late", "Age", "F"])}
T["F"] = (T.Gender == "F").astype(int); V["F"] = (V.Gender == "F").astype(int)
T["Age"] = T.Age.fillna(T.Age.median()); V["Age"] = V.Age.fillna(T.Age.median())
folds = T.fold.values
for nm_, (cols, base) in sets.items():
    for mdl in ("LR", "HGB"):
        mk = (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))) if mdl == "LR" else \
             (lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0))
        xt = X(T, cols, base); xv = X(V, cols, base); p = np.zeros(len(T))
        for k in np.unique(folds):
            tr = folds != k; m = mk().fit(xt[tr], y[tr]); p[~tr] = m.predict_proba(xt[~tr])[:, 1]
        pv = mk().fit(xt, y).predict_proba(xv)[:, 1]
        T[f"stack_{nm_}_{mdl}"] = p; V[f"stack_{nm_}_{mdl}"] = pv
        sens90 = np.mean(p[y == 1] >= np.quantile(p[y == 0], 0.9))
        print(f"  {nm_} [{mdl}]: OOF {A(y, p):.4f} (sens@90spec {sens90:.3f}) | val {A(V.y_chole, pv):.4f}")
        R.append(dict(group="stack", subset=f"{nm_} [{mdl}]", n=len(T), n_pos=int(y.sum()), auroc_E3_late=A(y, p),
                      note=f"sens@90spec {sens90:.3f}; val descriptive {A(V.y_chole, pv):.4f}"))
best = "stack_E3_late+vol+fossa+clip_HGB"
pb = T[best].values
print(f"\n  stratum AUROC E3 -> best stack: GB-missing {A(y[miss], S[miss]):.3f} -> {A(y[miss], pb[miss]):.3f}; GB-present {A(y[~miss], S[~miss]):.3f} -> {A(y[~miss], pb[~miss]):.3f}")
R.append(dict(group="stack_strata", subset="GB-missing", n=int(miss.sum()), n_pos=int(y[miss].sum()), auroc_E3_late=A(y[miss], S[miss]), note=f"best stack {A(y[miss], pb[miss]):.4f}"))
R.append(dict(group="stack_strata", subset="GB-present", n=int((~miss).sum()), n_pos=int(y[~miss].sum()), auroc_E3_late=A(y[~miss], S[~miss]), note=f"best stack {A(y[~miss], pb[~miss]):.4f}"))
# ---------- ceiling: drop label-noise candidates
noisy = ((y == 0) & ((T.kw_chole == 1) | (T.ch_cat == "not_mentioned") | (T.kw_not_vis == 1))).values | ((y == 1) & (T.kw_chole == 0)).values
print(f"\n[ceiling] drop {noisy.sum()} label-suspect cases: E3 {A(y, S):.4f} -> {A(y[~noisy], S[~noisy]):.4f}; best stack {A(y, pb):.4f} -> {A(y[~noisy], pb[~noisy]):.4f}")
R.append(dict(group="ceiling", subset="drop label-suspect", n=int((~noisy).sum()), auroc_E3_late=A(y[~noisy], S[~noisy]), note=f"best stack {A(y[~noisy], pb[~noisy]):.4f}; dropped {int(noisy.sum())}"))
# residual errors of best stack at 90% spec (rank-based, how many FP remain with GB missing etc.)
thr2 = np.quantile(pb[y == 0], 0.90); fn2 = (y == 1) & (pb < thr2); fp2 = (y == 0) & (pb >= thr2)
print(f"  best stack @90spec: sens {np.mean(pb[y==1] >= thr2):.3f}; FN {fn2.sum()} (no-chole-wording {np.mean(T.kw_chole[fn2]==0):.2f}, GB-present {np.mean(~miss[fn2]):.2f}); FP {fp2.sum()} (GB-missing {np.mean(miss[fp2]):.2f})")
pd.DataFrame(R).to_csv(D + "results_train_oof.csv", index=False)
keep = ["study_id", "fold", "y_chole", "ch_cat", "y_gs", "Age", "Gender", "ph", "gb_vox", "gb_missing", "gb_hu_med", "gb_f_fluid", "gb_f_fat", "gb_liver_contact",
        "gb_bowel_contact", "f30_n800", "f30_n1200", "f30_max", "peri_n1200", "f30_f_fat", "f30_f_fluid", "kw_chole", "kw_partial", "kw_contracted", "kw_liver_tx_anywhere",
        "E0", "E3_late", best, "stack_E3_late+vol+fossa+clip_LR", "pred90", "fn_cat", "fp_cat"]
T[keep].to_csv(D + "per_case_train_oof.csv", index=False)
V[["study_id", "E0", "E3_late", best, "stack_E3_late+vol+fossa+clip_LR"]].to_csv(D + "stack_val_descriptive.csv", index=False)
print("ANALYZE_DONE")
