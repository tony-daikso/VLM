"""Bowel obstruction: AUROC-cost ranking, label-variant retrains, stomach-unit variant, mask coverage, NG-tube shortcut,
oral contrast. TRAIN OOF only (val descriptive lines are marked). Test never read."""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
A = os.environ.get("ADIR", "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/analysis_bowel_obstruction/")
lab = pd.read_csv(A + "labels_trval.csv")
ft = pd.concat([pd.read_csv(A + f) for f in ("feat_a6.csv", "feat_ct.csv") if os.path.exists(A + f)]).drop_duplicates("study_id")
ft = ft[ft.err == 0]
oof = pd.read_csv(A + "oof_train.csv")
for v in ("dropamb", "ambpos"):
    if os.path.exists(A + f"oof_train_{v}.csv"):
        o2 = pd.read_csv(A + f"oof_train_{v}.csv")[["study_id", "E3_late"]].rename(columns={"E3_late": f"E3_late_{v}"}); oof = oof.merge(o2, on="study_id")
T = lab[lab.split == "train"].merge(oof, on="study_id").merge(ft, on="study_id", how="left")
T = T[T.y_sbo.notna()].reset_index(drop=True)
y = T.y_sbo.values.astype(int); pos, neg = y == 1, y == 0; S = T.E3_late.values
def auc(yy, s):
    yy = np.asarray(yy); s = np.asarray(s, float); m = ~np.isnan(s)
    return roc_auc_score(yy[m], s[m]) if len(set(yy[m])) == 2 else np.nan
out = []
def rec(sec, item, n1, n0, v, note=""):
    out.append(dict(section=sec, item=item, n_pos=int(n1), n_neg=int(n0), value=round(float(v), 4) if v == v else np.nan, note=note))
base = auc(y, S); rec("0_base", "E3_late OOF", pos.sum(), neg.sum(), base)
AMB = neg & ((T.sbo_cat == "probably_no") | (T.rep_aff_ileus == 1) | (T.rep_aff_obstruct == 1) | (T.rep_aff_sb_dilated == 1)).values
# 1 cost: AUROC_all = sum_k frac_k * AUROC(pos_k vs all neg) for a positive partition (same for negatives);
#   cost of a subgroup = frac * (AUROC of the complement - AUROC of the subgroup) = AUROC gain if the subgroup behaved like the rest
def cost_pos(name, pm):
    pm = np.asarray(pm, bool) & pos; rest = pos & ~pm
    a1 = auc(y[pm | neg], S[pm | neg]); a0 = auc(y[rest | neg], S[rest | neg]); f = pm.sum() / pos.sum()
    rec("1_cost_pos", name, pm.sum(), neg.sum(), f * (a0 - a1), f"frac {f:.3f}; AUROC sub {a1:.4f} vs rest {a0:.4f}")
def cost_neg(name, nm):
    nm = np.asarray(nm, bool) & neg; rest = neg & ~nm
    a1 = auc(y[pos | nm], S[pos | nm]); a0 = auc(y[pos | rest], S[pos | rest]); f = nm.sum() / neg.sum()
    rec("1_cost_neg", name, pos.sum(), nm.sum(), f * (a0 - a1), f"frac {f:.3f}; AUROC vs sub {a1:.4f} vs rest {a0:.4f}")
cost_neg("ambiguous neg (probably_no or affirmed ileus/obstruct/dilated SB)", AMB)
cost_neg("  - neg category probably_no", T.sbo_cat == "probably_no")
cost_neg("  - neg affirmed ileus/pseudo-obstruction", T.rep_aff_ileus == 1)
cost_neg("  - neg affirmed dilated small bowel", T.rep_aff_sb_dilated == 1)
cost_neg("  - neg affirmed 'obstruct' sentence", T.rep_aff_obstruct == 1)
cost_neg("neg dilated colon (affirmed), not ambiguous", (T.rep_aff_colon_dilated == 1).values & ~AMB)
cost_neg("neg NG/enteric tube, not ambiguous", (T.rep_ngtube == 1).values & ~AMB)
cost_neg("neg enteritis/colitis/wall thickening, not ambiguous", ((T.rep_gastroenteritis == 1) | (T.rep_bowel_wall == 1)).values & ~AMB)
cost_neg("neg post-op/ostomy, not ambiguous", ((T.rep_postop == 1) | (T.rep_ostomy == 1)).values & ~AMB)
cost_neg("neg ascites label, not ambiguous", (T.y_asc == 1).values & ~AMB)
cost_neg("neg free-air label, not ambiguous", (T.y_fa == 1).values & ~AMB)
cost_neg("neg hernia, not ambiguous", (T.kw_hernia == 1).values & ~AMB)
PA = (T.kw_partial == 1).values; HG = (T.kw_highgrade == 1).values
cost_pos("pos probably_yes", T.sbo_cat == "probably_yes")
cost_pos("pos partial/low-grade/early", PA)
cost_pos("pos resolving/improving", T.kw_resolving == 1)
cost_pos("pos LBO / both / colonic", (T.sbo_type.isin(["LBO", "both"]) | (T.kw_lbo == 1)).values)
cost_pos("pos gastric outlet / duodenal / afferent limb", T.kw_gastric_outlet == 1)
cost_pos("pos type not_stated", T.sbo_type == "not_stated")
cost_pos("pos closed-loop/volvulus/internal hernia", T.kw_closedloop == 1)
cost_pos("pos intussusception", T.kw_intussusception == 1)
cost_pos("pos NG tube in report", T.rep_ngtube == 1)
cost_pos("pos ileus or no-transition wording", (T.kw_ileus == 1) | (T.kw_no_transition == 1))
# 2 label-variant retrains (OOF AUROC always on original labels)
for c in ["E3_late", "E3_late_dropamb", "E3_late_ambpos"]:
    if c not in T: continue
    s = T[c].values
    rec("2_label_variant", f"{c}: all", pos.sum(), neg.sum(), auc(y, s))
    rec("2_label_variant", f"{c}: excluding ambiguous neg from evaluation", pos.sum(), (neg & ~AMB).sum(), auc(y[~AMB], s[~AMB]))
    rec("2_label_variant", f"{c}: pos vs ambiguous neg only", pos.sum(), AMB.sum(), auc(y[pos | AMB], s[pos | AMB]))
    rec("2_label_variant", f"{c}: ambiguous neg vs clean neg (higher = model ranks ambiguous neg like positives)", 0, neg.sum(), auc(AMB[neg].astype(int), s[neg]))
# 3 stomach-unit variant per subgroup
for nm, mk in [("gastric outlet / duodenal / afferent limb", T.kw_gastric_outlet == 1), ("type=SBO", T.sbo_type == "SBO"),
               ("LBO/both", T.sbo_type.isin(["LBO", "both"])), ("partial/low-grade", T.kw_partial == 1), ("all", pd.Series(True, index=T.index))]:
    mk = mk.values & pos | neg
    rec("3_stomach_unit", f"{nm}: E3_late / E3_late_st", (mk & pos).sum(), neg.sum(), auc(y[mk], T.E3_late_st.values[mk]) - auc(y[mk], S[mk]),
        f"E3_late {auc(y[mk], S[mk]):.4f}; E3_late_st {auc(y[mk], T.E3_late_st.values[mk]):.4f}")
# 4 mask coverage: fluid next to (<=6 mm) but outside the small-bowel mask, relative to fluid inside the mask
H = T.sb_ml.notna().values
if H.sum():
    T["miss_frac"] = T.unl_fluid_adjsb_ml / (T.sb_fluid_ml + T.unl_fluid_adjsb_ml + 1e-6)
    na = (T.y_asc != 1).values & (T.rep_ascites == 0).values
    for nm, m in [("all", H), ("no ascites (label and report)", H & na)]:
        rec("4_mask_cov", f"{nm}: unlabeled fluid within 6 mm of SB mask / (that + fluid in mask), median pos / neg", (m & pos).sum(), (m & neg).sum(),
            T.miss_frac[m & pos].median(), f"neg {T.miss_frac[m & neg].median():.3f}; AUROC as score {auc(y[m], T.miss_frac.values[m]):.3f}; unl_fluid_adjsb_ml AUROC {auc(y[m], T.unl_fluid_adjsb_ml.values[m]):.3f}")
    rec("4_mask_cov", "pos with miss_frac > 0.25 (>=1/4 of peri-SB fluid outside the mask)", (H & pos).sum(), 0, (T.miss_frac[H & pos] > .25).mean(), f"neg {(T.miss_frac[H & neg] > .25).mean():.3f}")
    mm = H & pos; q = T.miss_frac[mm].quantile([.5]).values[0]
    for nm, m2 in [("pos miss_frac <= median", mm & (T.miss_frac <= q).values), ("pos miss_frac > median", mm & (T.miss_frac > q).values)]:
        rec("4_mask_cov", f"{nm}: E3 AUROC vs all neg", m2.sum(), (H & neg).sum(), auc(y[m2 | (H & neg)], S[m2 | (H & neg)]))
    rec("4_mask_cov", "pos SB mask volume <300 ml (mask failure)", (H & pos).sum(), 0, (T.sb_ml[H & pos] < 300).mean(), f"neg {(T.sb_ml[H & neg] < 300).mean():.3f}")
    rec("4_mask_cov", "pos unlabeled fluid blobs >=25 mm wide (missed fluid-filled loop) > 0", (H & pos).sum(), 0, (T.unl_fluid_ge25_ml[H & pos] > 0).mean(),
        f"neg {(T.unl_fluid_ge25_ml[H & neg] > 0).mean():.3f}; AUROC E3 for pos with blob vs all neg {auc(y[(H & pos & (T.unl_fluid_ge25_ml > 0).values) | (H & neg)], S[(H & pos & (T.unl_fluid_ge25_ml > 0).values) | (H & neg)]):.4f}")
# 5 NG-tube shortcut: among negatives, is the higher score for NG-tube cases explained by dilated bowel (real physiology)?
NG = (T.rep_ngtube == 1).values
rec("5_ng", "neg-only AUROC E3 NG vs not", 0, neg.sum(), auc(NG[neg].astype(int), S[neg]))
rec("5_ng", "neg-only AUROC E3 NG vs not, excluding ambiguous neg", 0, (neg & ~AMB).sum(), auc(NG[neg & ~AMB].astype(int), S[neg & ~AMB]))
if H.sum():
    m = neg & H
    rec("5_ng", "neg: median sb_dil25_frac NG / no NG", 0, m.sum(), T.sb_dil25_frac[m & NG].median(), f"no NG {T.sb_dil25_frac[m & ~NG].median():.3f}")
    T["dbin"] = pd.qcut(T.sb_dil25_frac.rank(method="first"), 5, labels=False)
    r = []
    for b in range(5):
        mb = m & (T.dbin == b).values
        if (mb & NG).sum() >= 10: r.append(auc(NG[mb].astype(int), S[mb]))
    rec("5_ng", "neg-only AUROC E3 NG vs not, within sb_dil25_frac quintiles (mean)", 0, m.sum(), np.nanmean(r), f"per-bin {np.round(r, 3).tolist()}")
    rec("5_ng", "esoph_hi_vox>0 (tube proxy) in neg: E3 AUROC for proxy", 0, m.sum(), auc((T.esoph_hi_vox[m] > 0).astype(int), S[m]))
# 6 oral contrast in SB mask
if H.sum():
    oc = H & (T.sb_dense_ml > 20).values
    rec("6_contrast", "frac with positive oral contrast in SB (>20 ml >150 HU) pos / neg", (oc & pos).sum(), (oc & neg).sum(), (oc & pos).sum() / (H & pos).sum(), f"neg {(oc & neg).sum() / (H & neg).sum():.3f}")
    rec("6_contrast", "E3 AUROC within oral-contrast cases", (oc & pos).sum(), (oc & neg).sum(), auc(y[oc], S[oc]))
    rec("6_contrast", "E3 AUROC within no-oral-contrast cases", (H & ~oc & pos).sum(), (H & ~oc & neg).sum(), auc(y[H & ~oc], S[H & ~oc]))
res = pd.DataFrame(out); res.to_csv(A + "extra_train_oof.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 170); pd.set_option("display.max_rows", 300)
print(res.to_string(index=False))
