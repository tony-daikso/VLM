"""External validation of OWL E3_late (expB, 3 mm) on CG10k abdominal CT.

Inputs  (data/CT/CG/10000_data/):
  owl_infer/probs_main.csv   main-series predictions (seed-mean columns + per-seed)
  owl_infer/probs_delay.csv  full-abdomen delay-phase predictions for delay_full_available exams
  disease14_qwen.csv         Qwen report-derived labels (<d>_cat)
  nifti_3mm/series_selection.csv  phase / tier / QC flags
Outputs (owl_infer/): eval_*.csv + report.md
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, rankdata

ROOT = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data")
OUT = ROOT / "owl_infer"
NBOOT = 1000
SEED = 0
MIN_POS = 30

DIS = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "gallstones",
       "biliary_ductal_dilation", "hydronephrosis", "abdominal_aortic_aneurysm", "atherosclerosis",
       "osteopenia", "fracture", "bowel_obstruction", "appendicitis", "ascites", "free_air"]
ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊切除術後", "splenomegaly": "脾腫大",
      "gallstones": "膽結石", "biliary_ductal_dilation": "膽管擴張", "hydronephrosis": "腎水腫",
      "abdominal_aortic_aneurysm": "腹主動脈瘤", "atherosclerosis": "動脈粥狀硬化", "osteopenia": "骨質減少",
      "fracture": "骨折", "bowel_obstruction": "腸阻塞", "appendicitis": "闌尾炎", "ascites": "腹水",
      "free_air": "腹腔游離氣體"}
MERLIN = {"hepatic_steatosis": .819, "surgically_absent_gallbladder": .956, "splenomegaly": .980,
          "gallstones": .783, "biliary_ductal_dilation": .834, "hydronephrosis": .897,
          "abdominal_aortic_aneurysm": .975, "atherosclerosis": .888, "osteopenia": .881, "fracture": .824,
          "bowel_obstruction": .950, "appendicitis": .853, "ascites": .883, "free_air": .861}
POS = {"yes", "probably_yes"}
NEG_EXPLICIT = {"no", "probably_no"}


# ---------------------------------------------------------------- metrics
def auc(y, s):
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_idx(n, rng):
    return rng.integers(0, n, size=(NBOOT, n))


def auc_ci(y, s, idx):
    vals = np.array([auc(y[i], s[i]) for i in idx])
    vals = vals[~np.isnan(vals)]
    return np.percentile(vals, [2.5, 97.5])


def _structural(y, s):
    x, z = s[y], s[~y]
    m, n = len(x), len(z)
    allv = np.concatenate([x, z])
    r_all = rankdata(allv)
    rx = rankdata(x)
    rz = rankdata(z)
    a = (r_all[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (r_all[:m] - rx) / n
    v01 = 1 - (r_all[m:] - rz) / m
    return a, v10, v01


def delong_paired(y, s1, s2):
    y = np.asarray(y, bool)
    a1, v10a, v01a = _structural(y, s1)
    a2, v10b, v01b = _structural(y, s2)
    m, n = y.sum(), (~y).sum()
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    d = a1 - a2
    se = np.sqrt(var)
    p = 2 * norm.sf(abs(d) / se) if se > 0 else np.nan
    return a1, a2, d, d - 1.96 * se, d + 1.96 * se, p


# ---------------------------------------------------------------- data
def load():
    m = pd.read_csv(OUT / "probs_main.csv")
    d = pd.read_csv(OUT / "probs_delay.csv")
    lab = pd.read_csv(ROOT / "disease14_qwen.csv", encoding="utf-8-sig")
    sel = pd.read_csv(ROOT / "nifti_3mm/series_selection.csv")
    for df in (m, d):  # seed mean (recompute from per-seed columns)
        for k in DIS:
            sc = [c for c in df.columns if c.startswith(k + "__seed")]
            if sc:
                df[k] = df[sc].mean(axis=1)
    issues = []
    issues.append(f"probs_main {len(m)} 列（唯一 {m.study_id.nunique()}）；probs_delay {len(d)} 列（唯一 {d.study_id.nunique()}）")
    issues.append(f"標籤 {len(lab)} 筆；main 中無標籤 {len(set(m.study_id) - set(lab.EXAMID))}；"
                  f"delay 不在 main {len(set(d.study_id) - set(m.study_id))}")
    sel = sel[sel.status == "ok"]
    fl = sel["flags"].fillna("")
    sel = sel.assign(chest=fl.str.contains("chest_series"), thick=fl.str.contains("thick_slices"),
                     limz=fl.str.contains("limited_z"), delayfull=fl.str.contains("delay_full_available"))
    issues.append(f"main 中無 series_selection(ok) {len(set(m.study_id) - set(sel.EXAMID))}；"
                  f"delay_full_available 旗標 {sel.delayfull.sum()}，與 delay 檔交集 "
                  f"{len(set(sel.EXAMID[sel.delayfull]) & set(d.study_id))}")
    base = m[["study_id"] + DIS].rename(columns={"study_id": "EXAMID"})
    base = base.merge(lab[["EXAMID"] + [k + "_cat" for k in DIS]], on="EXAMID", how="inner")
    base = base.merge(sel[["EXAMID", "phase", "tier", "chest", "thick", "limz", "delayfull"]], on="EXAMID", how="left")
    dl = d[["study_id"] + DIS].rename(columns={"study_id": "EXAMID"}).set_index("EXAMID")
    return base, dl, issues


def stratum(row):
    if row.phase == "portal":
        return "portal" if row.tier == 1 else "portal_limited(tier3)"
    if row.phase == "cplus":
        return "C+未標期" if row.tier == 2 else "C+_limited(tier3)"
    if row.phase == "noncontrast":
        return "C-"
    return "other"


def eval_set(df, scores, mode, rng, tag):
    """mode A: neg=all non-positive; mode B: neg=explicit no/probably_no only."""
    rows = []
    for k in DIS:
        cat = df[k + "_cat"].astype(str)
        y = cat.isin(POS).values
        keep = np.ones(len(df), bool) if mode == "A" else (y | cat.isin(NEG_EXPLICIT).values)
        yy, ss = y[keep], scores[k].values[keep]
        n, npos = len(yy), int(yy.sum())
        a = auc(yy, ss)
        lo = hi = np.nan
        if 0 < npos < n:
            lo, hi = auc_ci(yy, ss, boot_idx(n, rng))
        rows.append(dict(set=tag, disease=k, 疾病=ZH[k], N=n, n_pos=npos, AUROC=a, CI_lo=lo, CI_hi=hi,
                         note="樣本少，僅供參考" if npos < MIN_POS else ""))
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    base, dl, issues = load()
    base["stratum"] = base.apply(stratum, axis=1)
    preds_main = base[DIS]

    # A / B
    A = eval_set(base, preds_main, "A", rng, "A_main_all")
    B = eval_set(base, preds_main, "B", rng, "B_explicit")
    # C: exclude chest_series + thick_slices
    qc = base[~(base.chest.fillna(False) | base.thick.fillna(False))].reset_index(drop=True)
    CA = eval_set(qc, qc[DIS], "A", rng, "C_qc_A")
    CB = eval_set(qc, qc[DIS], "B", rng, "C_qc_B")
    allres = pd.concat([A, B, CA, CB])
    allres.to_csv(OUT / "eval_ABC_long.csv", index=False)

    def fmt(r):
        s = f"{r.AUROC:.3f} ({r.CI_lo:.3f}–{r.CI_hi:.3f})"
        return s + ("*" if r.n_pos < MIN_POS else "")
    wide = pd.DataFrame({"疾病": [ZH[k] for k in DIS]})
    for t, lab in [("A_main_all", "A"), ("B_explicit", "B"), ("C_qc_A", "C-A"), ("C_qc_B", "C-B")]:
        sub = allres[allres.set == t].set_index("disease").loc[DIS]
        wide[f"{lab} N/陽性"] = [f"{n}/{p}" for n, p in zip(sub.N, sub.n_pos)]
        wide[f"{lab} AUROC (95%CI)"] = [fmt(r) for r in sub.itertuples()]
    wide["Merlin val"] = [MERLIN[k] for k in DIS]
    wide["A − Merlin"] = (allres[allres.set == "A_main_all"].set_index("disease").loc[DIS].AUROC.values
                          - np.array([MERLIN[k] for k in DIS])).round(3)
    wide.to_csv(OUT / "eval_ABC_table.csv", index=False)

    # D: delay comparison
    ids = [i for i in base.EXAMID if i in dl.index]
    sub = base.set_index("EXAMID").loc[ids].reset_index()
    pd_delay = dl.loc[ids].reset_index(drop=True)
    drows = []
    for k in DIS:
        y = sub[k + "_cat"].astype(str).isin(POS).values
        npos = int(y.sum())
        if npos == 0 or npos == len(y):
            continue
        a1, a2, dd, lo, hi, p = delong_paired(y, pd_delay[k].values, sub[k].values)
        idx = boot_idx(len(y), rng)
        bd = np.array([auc(y[i], pd_delay[k].values[i]) - auc(y[i], sub[k].values[i]) for i in idx])
        bd = bd[~np.isnan(bd)]
        drows.append(dict(disease=k, 疾病=ZH[k], N=len(y), n_pos=npos, AUROC_main_liver=a2, AUROC_delay=a1,
                          diff_delay_minus_main=dd, DeLong_lo=lo, DeLong_hi=hi, DeLong_p=p,
                          boot_lo=np.percentile(bd, 2.5), boot_hi=np.percentile(bd, 97.5),
                          note="樣本少，僅供參考" if npos < MIN_POS else ""))
    D = pd.DataFrame(drows)
    # hybrid over full set
    hyb = base[DIS].copy()
    mask = base.EXAMID.isin(dl.index).values
    hyb.loc[mask, DIS] = dl.loc[base.EXAMID[mask], DIS].values
    H = eval_set(base, hyb, "A", rng, "hybrid_A")
    hrows = []
    for k in DIS:
        y = base[k + "_cat"].astype(str).isin(POS).values
        a1, a2, dd, lo, hi, p = delong_paired(y, hyb[k].values, base[k].values)
        hrows.append(dict(disease=k, 疾病=ZH[k], AUROC_main=a2, AUROC_hybrid=a1, diff=dd, lo=lo, hi=hi, p=p))
    HD = pd.DataFrame(hrows).merge(H[["disease", "CI_lo", "CI_hi", "N", "n_pos"]], on="disease")
    D.to_csv(OUT / "eval_D_delay_vs_main_810.csv", index=False)
    HD.to_csv(OUT / "eval_D_hybrid_full.csv", index=False)

    # E: strata (mode A, main preds)
    erows = []
    for st, g in base.groupby("stratum"):
        g = g.reset_index(drop=True)
        e = eval_set(g, g[DIS], "A", rng, st)
        erows.append(e)
    E = pd.concat(erows)
    E.to_csv(OUT / "eval_E_strata.csv", index=False)
    Ew = E.pivot(index="disease", columns="set", values="AUROC").loc[DIS]
    Ep = E.pivot(index="disease", columns="set", values="n_pos").loc[DIS]
    Nst = base.stratum.value_counts()

    # calibration shift
    cal = []
    for k in DIS:
        y = base[k + "_cat"].astype(str).isin(POS).values
        cal.append(dict(disease=k, 疾病=ZH[k], mean_prob_main=base[k].mean(), mean_prob_hybrid=hyb[k].mean(),
                        prevalence_A=y.mean(),
                        prevalence_B=y.sum() / (y | base[k + "_cat"].astype(str).isin(NEG_EXPLICIT)).sum()))
    CAL = pd.DataFrame(cal)
    CAL["ratio_main/prev"] = CAL.mean_prob_main / CAL.prevalence_A
    CAL.to_csv(OUT / "eval_calibration.csv", index=False)

    # ------------------------------------------------------------ report.md
    L = ["# OWL E3_late (expB, 3 mm) — CG10k 外部驗證", "",
         f"- 評估例數：{len(base)}（main）；延遲期 {len(ids)}；bootstrap {NBOOT} 次（以 exam 為單位，seed={SEED}）",
         "- 標籤：Qwen 報告抽取，陽性 = yes/probably_yes；A 陰性 = 其餘全部；B 陰性 = 僅 no/probably_no",
         f"- C：排除 chest_series / thick_slices 後 N={len(qc)}",
         f"- `*` = 陽性 < {MIN_POS}，樣本少，僅供參考", "", "## 資料檢查", ""]
    L += [f"- {s}" for s in issues] + ["", "## A / B / C 主表", "", wide.to_markdown(index=False), "",
                                        "## D. 延遲期比較（delay_full_available 810 例，A 定義）", "",
                                        D.drop(columns="disease").round(3).to_markdown(index=False), "",
                                        "### 混合版（810 例用 delay，其餘 main）vs 純 main，全體", "",
                                        HD.drop(columns="disease").round(4).to_markdown(index=False), "",
                                        "## E. 依 phase/tier 分層（A 定義，AUROC；括號為陽性數）", "",
                                        "N：" + "、".join(f"{k} {v}" for k, v in Nst.items()), ""]
    Et = pd.DataFrame({"疾病": [ZH[k] for k in DIS]})
    for c in Ew.columns:
        Et[c] = [f"{a:.3f} ({int(p)})" + ("*" if p < MIN_POS else "") if not np.isnan(a) else "–"
                 for a, p in zip(Ew[c], Ep[c])]
    L += [Et.to_markdown(index=False), "", "## 校準偏移（平均預測機率 vs 盛行率）", "",
          CAL.drop(columns="disease").round(3).to_markdown(index=False), ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
