"""CG10k OWL E3_late: contrast-enhanced (C+) vs non-contrast (C-) evaluation (section F of owl_infer/report.md).

F0  existing main predictions split by the phase of the selected series: C+ (portal + unlabelled C+) vs C- (the
    152 exams without any C+ series -> selection bias, small sample)
F1  paired, same exams: main C+ prediction vs prediction on the non-contrast series of the SAME exam
    (probs_nc.csv, nifti_3mm_nc/series_selection_nc.csv). Per disease paired DeLong + paired bootstrap of the
    AUROC difference (C- minus C+).  F1a all pairs; F1b strict = C- z >= 300 mm and not chest-only, C+ main not
    chest_series / thick_slices / limited_z (same coverage on both sides).
F2  C- overall: all exams with a C- prediction (nc series of C+ exams + the 152 main-C- exams) vs Merlin val.
Labels: Qwen report labels, definition A (positive = yes/probably_yes, negative = everything else).
usage: python eval_owl_cg10k_nc.py [--quick]   (--quick: F0 only, no probs_nc.csv needed)
"""
import argparse

import numpy as np
import pandas as pd

import eval_owl_cg10k as E

MARK0, MARK1 = "<!-- F_contrast begin -->", "<!-- F_contrast end -->"


def fmt(a, lo, hi, npos):
    if npos == 0 or np.isnan(a):
        return "–"
    return f"{a:.3f} ({lo:.3f}–{hi:.3f})" + ("*" if npos < E.MIN_POS else "")


def wide(res, sets, labels):
    t = pd.DataFrame({"疾病": [E.ZH[k] for k in E.DIS]})
    for st, lab in zip(sets, labels):
        s = res[res.set == st].set_index("disease").reindex(E.DIS)
        t[f"{lab} N/陽性"] = [f"{int(n)}/{int(p)}" for n, p in zip(s.N, s.n_pos)]
        t[f"{lab} AUROC (95%CI)"] = [fmt(r.AUROC, r.CI_lo, r.CI_hi, r.n_pos) for r in s.itertuples()]
    t["Merlin val"] = [E.MERLIN[k] for k in E.DIS]
    return t


def paired(df, rng, tag):
    rows = []
    for k in E.DIS:
        y = df[k + "_cat"].astype(str).isin(E.POS).values
        npos = int(y.sum())
        if npos == 0 or npos == len(y):
            rows.append(dict(set=tag, disease=k, 疾病=E.ZH[k], N=len(y), n_pos=npos))
            continue
        sp, sn = df[k].values, df[k + "_nc"].values
        a_nc, a_cp, d, lo, hi, p = E.delong_paired(y, sn, sp)
        idx = E.boot_idx(len(y), rng)
        bd = np.array([E.auc(y[i], sn[i]) - E.auc(y[i], sp[i]) for i in idx])
        bd = bd[~np.isnan(bd)]
        rows.append(dict(set=tag, disease=k, 疾病=E.ZH[k], N=len(y), n_pos=npos, AUROC_Cplus=a_cp, AUROC_Cminus=a_nc,
                         diff_Cminus_minus_Cplus=d, DeLong_lo=lo, DeLong_hi=hi, DeLong_p=p,
                         boot_lo=np.percentile(bd, 2.5), boot_hi=np.percentile(bd, 97.5),
                         Merlin_val=E.MERLIN[k], note="樣本少，僅供參考" if npos < E.MIN_POS else ""))
    return pd.DataFrame(rows)


def ptable(P):
    t = pd.DataFrame({"疾病": P.疾病, "N/陽性": [f"{n}/{p}" for n, p in zip(P.N, P.n_pos)]})
    t["C+ AUROC"] = P.AUROC_Cplus.map(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
    t["C- AUROC"] = P.AUROC_Cminus.map(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
    t["差 (C−) − (C+) [DeLong 95%CI]"] = [
        f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("*" if n < E.MIN_POS else "") if pd.notna(d) else "–"
        for d, lo, hi, n in zip(P.diff_Cminus_minus_Cplus, P.DeLong_lo, P.DeLong_hi, P.n_pos)]
    t["bootstrap 95%CI"] = [f"[{lo:+.3f}, {hi:+.3f}]" if pd.notna(lo) else "–" for lo, hi in zip(P.boot_lo, P.boot_hi)]
    t["p"] = P.DeLong_p.map(lambda x: f"{x:.3g}" if pd.notna(x) else "–")
    t["Merlin val"] = [E.MERLIN[k] for k in P.disease]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(E.SEED)
    base, _, _ = E.load()
    base["grp"] = np.where(base.phase.isin(["portal", "cplus"]), "C+",
                           np.where(base.phase == "noncontrast", "C-", "other"))
    L = [MARK0, "## F. 顯影 C+ vs 無顯影 C-", "",
         "Merlin 訓練資料幾乎都是顯影（portal venous）；CG 檢查 95% 為 with and without contrast。", ""]

    # F0
    r0 = pd.concat([E.eval_set(g.reset_index(drop=True), g.reset_index(drop=True)[E.DIS], "A", rng, st)
                    for st, g in base.groupby("grp") if st in ("C+", "C-")])
    r0.to_csv(E.OUT / "eval_F0_contrast_strata.csv", index=False)
    n = base.grp.value_counts()
    L += ["### F0. 現有 main 預測依選中 series 分層（A 定義）", "",
          f"C+ = portal {int((base.phase == 'portal').sum())} + 未標期 C+ {int((base.phase == 'cplus').sum())} = {n.get('C+', 0)}；"
          f"C- = {n.get('C-', 0)}（只有整個檢查沒有 C+ series 才選 C-，**選擇偏差、小樣本**，僅供參考）；"
          f"phase unknown {n.get('other', 0)} 例不列入。`*` = 陽性 < {E.MIN_POS}。", "",
          wide(r0, ["C+", "C-"], ["C+", "C-"]).to_markdown(index=False), ""]

    if not a.quick:
        nc = pd.read_csv(E.OUT / "probs_nc.csv")
        for k in E.DIS:
            sc_ = [c for c in nc.columns if c.startswith(k + "__seed")]
            if sc_:
                nc[k] = nc[sc_].mean(axis=1)
        nc = nc[["study_id"] + E.DIS].rename(columns={"study_id": "EXAMID"})
        nc = nc.rename(columns={k: k + "_nc" for k in E.DIS})
        seln = pd.read_csv(E.ROOT / "nifti_3mm_nc/series_selection_nc.csv")
        seln = seln[seln.status == "ok"]
        fl = seln["flags"].fillna("")
        seln = seln.assign(nc_limz=fl.str.contains("limited_z"), nc_chest=fl.str.contains("chest_series"),
                           nc_desc=seln.SeriesDescription)[["EXAMID", "nc_limz", "nc_chest", "nc_desc"]]
        P = base[base.grp == "C+"].merge(nc, on="EXAMID").merge(seln, on="EXAMID", how="left").reset_index(drop=True)
        strict = P[~P.nc_limz.fillna(True) & ~P.nc_chest.fillna(True) & ~P.chest.fillna(False)
                   & ~P.thick.fillna(False) & ~P.limz.fillna(False)].reset_index(drop=True)
        F1a, F1b = paired(P, rng, "F1a_all_pairs"), paired(strict, rng, "F1b_strict_pairs")
        pd.concat([F1a, F1b]).to_csv(E.OUT / "eval_F1_paired_cplus_vs_cminus.csv", index=False)
        n_nc = len(nc)
        n_conv = len(seln)
        L += ["### F1. 同一檢查配對：C+（main）vs C-（同檢查的無顯影 series）", "",
              f"- 非顯影 series 轉檔 ok {n_conv}；有 probs_nc 的 {n_nc}；與 main C+ 配對 N={len(P)}"
              f"（C- z≥300 mm {int((~P.nc_limz.fillna(True)).sum())}，200–300 mm limited_z {int(P.nc_limz.fillna(False).sum())}；"
              f"C- 為胸部描述 chest_series {int(P.nc_chest.fillna(False).sum())}）",
              "- 差 = AUROC(C−) − AUROC(C+)，負值 = 無顯影較差；DeLong 配對 + 配對 bootstrap（exam 為單位，"
              f"{E.NBOOT} 次）。`*` = 陽性 < {E.MIN_POS}", "",
              f"#### F1a. 全部配對（N={len(P)}）", "", ptable(F1a).to_markdown(index=False), "",
              f"#### F1b. 嚴格配對：兩側覆蓋相同（C- z≥300 mm 且非胸部描述；C+ 非 chest/thick/limited_z），N={len(strict)}", "",
              ptable(F1b).to_markdown(index=False), ""]
        # F2: C- overall
        c2 = P[["EXAMID"] + [k + "_cat" for k in E.DIS] + [k + "_nc" for k in E.DIS]].rename(
            columns={k + "_nc": k for k in E.DIS})
        c1 = base[base.grp == "C-"][["EXAMID"] + [k + "_cat" for k in E.DIS] + E.DIS]
        call = pd.concat([c2, c1], ignore_index=True)
        cp_paired = P[["EXAMID"] + [k + "_cat" for k in E.DIS] + E.DIS]
        r2 = pd.concat([E.eval_set(call, call[E.DIS], "A", rng, "C-_all"),
                        E.eval_set(cp_paired, cp_paired[E.DIS], "A", rng, "C+_paired"),
                        E.eval_set(base[base.grp == "C+"].reset_index(drop=True),
                                   base[base.grp == "C+"].reset_index(drop=True)[E.DIS], "A", rng, "C+_all")])
        r2.to_csv(E.OUT / "eval_F2_cminus_overall.csv", index=False)
        L += [f"### F2. C- 全體（nc series {len(c2)} + main 即為 C- 的 {len(c1)} = {len(call)}）vs C+ 全體 vs Merlin val", "",
              wide(r2, ["C-_all", "C+_paired", "C+_all"], ["C- 全體", "C+（配對子集）", "C+ 全體"]).to_markdown(index=False), ""]
    L += [MARK1]
    sec = "\n".join(L)
    rp = E.OUT / "report.md"
    txt = rp.read_text(encoding="utf-8") if rp.exists() else ""
    if MARK0 in txt and MARK1 in txt:
        txt = txt[:txt.index(MARK0)] + sec + txt[txt.index(MARK1) + len(MARK1):]
    else:
        txt = txt.rstrip() + "\n\n" + sec + "\n"
    rp.write_text(txt, encoding="utf-8")
    print(sec)


if __name__ == "__main__":
    main()
