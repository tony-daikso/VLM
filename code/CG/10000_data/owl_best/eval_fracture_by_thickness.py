"""CG10k fracture external validation stratified by the ORIGINAL slice thickness of the main (C+) series converted to NIfTI.
Diagnostic only: nothing is fit or tuned on CG.

Thickness: nifti_3mm/series_selection.csv (select_convert.py per-exam record: the exact SeriesInstanceUID converted, its
SliceThickness, and the median z gap of the source slices in src_pixel_spacing), cross-checked with dicom_index/series.csv
(SliceThickness, z_gap_med) through SeriesInstanceUID. Effective spacing = z_gap_med (identical to SliceThickness here).

Labels / cohort: eval_best_cg10k.load + labels, definition B (pos yes/probably_yes, neg only no/probably_no); subgroups =
eval_best_cg10k.SUBS (site positives vs ALL fracture negatives; other positives excluded). Same exams for every model
(as eval_fracture_newbest.py).
Models: old C3 (= fracture '__best' in probs_best_main.csv), new a6x10w2_tv7 (probs_fracture_newbest_main.csv), E3_late.
Components (fx_thickness_inputs/, small CSVs copied from A6000 by extract_a6000.py, read-only there):
  cg_new_components.csv = A6000 fracture_a6x10w2_tv7/integ/cg_main_scores.csv (stacker inputs as fed + site logits + case)
  cg_c3_members.csv     = CG C3 members assembled exactly as cg10k_best/cg_frac.stack() (+ C3 stack, = cg_scores_main)
  merlin_frame.csv      = Merlin train (OOF inputs) / val frame of integrate.py, no study ids (+ saved-model val site logits)
Statistics: 1,000 exam-level bootstrap (seed 0); paired new-C3 differences use the same resamples (+ paired DeLong);
stratum contrasts (e.g. <5 mm minus 5 mm) are unpaired (independent bootstraps per stratum), two-sided bootstrap p.
'*' = < 30 positives or < 30 negatives. Score shift = (CG mean - Merlin train mean) / Merlin train SD, in NEGATIVES.
Outputs: owl_infer/best/eval_fracture_by_thickness.csv (long) and eval_fracture_by_thickness.md"""
import io, sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import eval_best_cg10k as eb
from eval_owl_cg10k import auc, delong_paired, DIS, ZH, SEED, NBOOT

ROOT = eb.ROOT; BD = eb.BD; INP = BD / "fx_thickness_inputs"
MIN_N = 30
FX_ROWS = ["fracture", "fracture:sub_compression", "fracture:vertebra", "fracture:site_rib", "fracture:only_rib",
           "fracture:site_pelvis", "fracture:site_hip_femur", "fracture:site_other", "fracture:pelvic_ring"]
NAME = {**eb.RZH, "fracture": "骨折（全部）", "fracture:pelvic_ring": "骨折子群－骨盆/薦骨/髖股骨（= Merlin t_pel 定義）"}
# component -> (model, site head, resolution of the input it reads)
NEW = {"J_add": ("vert", "3+1.5 mm（Exp J add：vert3/rib3/pelvis＋v15）"), "L_v15": ("vert", "1.5 mm"), "L_v3": ("vert", "3 mm"),
       "LJ_v15": ("vert", "1.5 mm"), "LJ_v3": ("vert", "3 mm"), "I_v": ("vert", "3 mm（expI 骨單元）"), "FT_sp": ("vert", "3 mm"),
       "FT_sp15": ("vert", "1.5 mm"), "PE": ("vert", "1.5 mm（pe15）"), "PE2": ("vert", "1.5 mm（pe15）"),
       "L_r": ("rib", "1.5 mm（rib15）"), "L_rinst": ("rib", "1.5 mm（rib15）"), "LJ_r": ("rib", "3 mm（rib3）"), "I_r": ("rib", "3 mm（expI）"),
       "FT_rib": ("rib", "1.5 mm（rib15 框）"), "RVRS": ("rib", "1.5 mm（rs15/lev28/a6）"),
       "U_pel": ("pel", "3 mm"), "U_org_pel": ("pel", "3 mm"), "U_org_out": ("pel", "3 mm"), "L_p": ("pel", "3 mm"), "LJ_p": ("pel", "3 mm"),
       "I_p": ("pel", "3 mm（expI）"), "P15FT": ("pel", "1.5 mm（pv15）"),
       "site_vert": ("vert", "椎體頭 logit"), "site_rib": ("rib", "肋骨頭 logit"), "site_pel": ("pel", "骨盆頭 logit"), "fracture": ("all", "case = max(3 頭)")}
C3 = {"expI": ("all", "3 mm（expI 骨單元）", "m_expI"), "J_add": ("vert", "3+1.5 mm", "m_J_add"), "L": ("all", "3+1.5 mm（Exp L）", "m_L"),
      "U_org_pel": ("pel", "3 mm", "m_U_org_pel"), "U_pel": ("pel", "3 mm", "m_U_pel"), "U_org_out": ("pel", "3 mm", "m_U_org_out"),
      "C3_stack": ("all", "C3 LR stack", "C3_stack")}
SITE_LABEL = {"vert": "fracture:vertebra", "rib": "fracture:site_rib", "pel": "fracture:pelvic_ring", "all": "fracture"}
HEADZ = {"vert": "椎體", "rib": "肋骨", "pel": "骨盆"}


def tbin(t):
    if not np.isfinite(t): return "unknown"
    if t <= 3.0: return "≤3 mm"
    if t < 4.5: return "3.75 mm"
    if t <= 5.0: return "5 mm"
    return "≥7.5 mm"


def load():
    df, _ = eb.load("main")
    nw = pd.read_csv(BD / "probs_fracture_newbest_main.csv").rename(columns={"study_id": "EXAMID", "fracture": "fx_new"})
    df = df.merge(nw[["EXAMID", "fx_new"]], on="EXAMID", how="left")
    nc = pd.read_csv(INP / "cg_new_components.csv").rename(columns={"study_id": "EXAMID"})
    nc = nc.rename(columns={c: "n_" + c for c in nc.columns if c != "EXAMID"})
    c3 = pd.read_csv(INP / "cg_c3_members.csv").rename(columns={"study_id": "EXAMID"})
    df = df.merge(nc, on="EXAMID", how="left").merge(c3, on="EXAMID", how="left")
    sel = pd.read_csv(ROOT / "nifti_3mm/series_selection.csv"); sel = sel[sel.status == "ok"].copy()
    sel["gap_sel"] = sel.src_pixel_spacing.str.split("x").str[2].astype(float)
    ser = pd.read_csv(ROOT / "dicom_index/series.csv", usecols=["SeriesInstanceUID", "SliceThickness", "z_gap_med"])
    sel = sel.merge(ser.drop_duplicates("SeriesInstanceUID"), on="SeriesInstanceUID", how="left")
    raw = open(ROOT / "deliver_10k_reports_v1(in).csv", "rb").read().decode("utf-8", errors="replace")
    site = pd.read_csv(io.StringIO(raw)); site.columns = ["PID", "EXAMID", "EDATE", "LOC", "scanner", "Report"]
    site["LOC"] = "LOC" + site.LOC.str.replace("?", "", regex=False).str.strip("()")     # site names are literally '??(n)' in the file
    sel = sel.merge(site[["EXAMID", "LOC", "scanner"]], on="EXAMID", how="left")
    df = df.merge(sel[["EXAMID", "slice_thickness", "gap_sel", "SliceThickness", "z_gap_med", "SeriesDescription", "LOC", "scanner"]],
                  on="EXAMID", how="left")
    eff = df.z_gap_med.where(df.z_gap_med.notna(), df.gap_sel)
    df["eff_mm"] = eff.where(eff.notna(), df.slice_thickness)
    df["tbin"] = df.eff_mm.map(tbin)
    df["chest"] = df.chest.fillna(False).astype(bool)
    # thinnest AXIAL series per exam (user definition 11-10): ImageType contains 'AXIAL', min(SliceThickness) over the exam
    ax = pd.read_csv(ROOT / "dicom_index/series.csv", usecols=["EXAMID", "ImageType", "SliceThickness", "z_gap_med", "SeriesDescription"])
    ax = ax[ax.ImageType.fillna("").str.contains("AXIAL")]
    th = ax.groupby("EXAMID").SliceThickness.min().rename("thin_mm")
    contig = ax[(ax.SliceThickness <= 1.25) & (ax.z_gap_med <= 1.5)].EXAMID.unique()
    hrct = ax[(ax.SliceThickness <= 1.25) & ax.SeriesDescription.fillna("").str.contains("HRCT", case=False)].EXAMID.unique()
    df = df.merge(th, left_on="EXAMID", right_index=True, how="left")
    df["thin_bin"] = df.thin_mm.map(thin_bin)
    df["thin_contig"] = df.EXAMID.isin(contig); df["thin_hrct"] = df.EXAMID.isin(hrct)
    return df, sel


THIN_ORDER = ["最薄 5 mm", "最薄 3 mm", "最薄 2.5 mm", "最薄 3.75 mm", "最薄 1–1.25 mm", "最薄 其他", "最薄 無 axial"]


def thin_bin(t):
    if not np.isfinite(t): return "最薄 無 axial"
    for v, b in ((5.0, "最薄 5 mm"), (3.0, "最薄 3 mm"), (2.5, "最薄 2.5 mm"), (3.75, "最薄 3.75 mm")):
        if abs(t - v) < 1e-3: return b
    if 1.0 <= t <= 1.25: return "最薄 1–1.25 mm"
    return "最薄 其他"   # 0.5 / 0.625 / 2 / 7.5 / 10 mm


def label(d, k):
    if k == "fracture:pelvic_ring":
        base = eb.labels(d, "fracture", "B"); s = d.fx_sites.fillna("").str.split("|")
        m = s.apply(lambda x: any(t in x for t in ("pelvis", "sacrum", "hip_femur"))).values
        y = base.copy(); y[(base == 1) & ~m] = np.nan; return y
    return eb.labels(d, k, "B")


def fx_cohort(d, k):
    """y and the 3 model scores on the common exams (as eval_fracture_newbest)"""
    y = label(d, k); ok = ~np.isnan(y) & d.fx_new.notna().values & d["fracture__best"].notna().values & d["fracture"].notna().values
    return y[ok].astype(bool), {"new": d.fx_new.values[ok], "C3": d["fracture__best"].values[ok], "E3": d["fracture"].values[ok]}, ok


def q(v): return np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).any() else (np.nan, np.nan)


def pval(d): return min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))


def boot_models(y, S, rng):
    n = len(y)
    if n == 0 or y.sum() == 0 or y.sum() == n: return {k: np.nan for k in S}, {k: np.full(NBOOT, np.nan) for k in S}
    idx = rng.integers(0, n, size=(NBOOT, n))
    return {k: auc(y, s) for k, s in S.items()}, {k: np.array([auc(y[i], s[i]) for i in idx]) for k, s in S.items()}


# --------------------------------------------------------------------------------------------- 1+2. models by stratum
def model_table(df, strata, rng):
    L, D = [], {}
    for k in FX_ROWS:
        for g, m in strata:
            d = df[m.values].reset_index(drop=True); y, S, _ = fx_cohort(d, k)
            A, B = boot_models(y, S, rng); D[(k, g)] = B
            r = dict(part="model", row=k, 疾病=NAME[k], stratum=g, N=len(y), n_pos=int(y.sum()), n_neg=int((~y).sum()))
            r["small"] = r["n_pos"] < MIN_N or r["n_neg"] < MIN_N
            for mo in S:
                r[f"AUROC_{mo}"] = A[mo]; r[f"{mo}_lo"], r[f"{mo}_hi"] = q(B[mo])
            if np.isfinite(A["new"]):
                dd = B["new"] - B["C3"]; r["d_new_C3"] = A["new"] - A["C3"]; r["d_new_C3_lo"], r["d_new_C3_hi"] = q(dd)
                r["p_new_C3_DeLong"] = delong_paired(y, S["new"], S["C3"])[5]
            L.append(r)
    return L, D


def contrast_rows(R, D, pairs):
    out = []
    for k in FX_ROWS:
        for a, b in pairs:
            if (k, a) not in D or (k, b) not in D: continue
            ra = R[(R.row == k) & (R.stratum == a)].iloc[0]; rb = R[(R.row == k) & (R.stratum == b)].iloc[0]
            r = dict(part="contrast", row=k, 疾病=NAME[k], stratum=f"Δ {a} − {b}", N=ra.N + rb.N, n_pos=ra.n_pos + rb.n_pos,
                     n_neg=ra.n_neg + rb.n_neg, small=bool(ra.small or rb.small))
            for mo in ("new", "C3", "E3"):
                d = D[(k, a)][mo] - D[(k, b)][mo]
                if np.all(np.isnan(d)): continue
                r[f"AUROC_{mo}"] = ra[f"AUROC_{mo}"] - rb[f"AUROC_{mo}"]; r[f"{mo}_lo"], r[f"{mo}_hi"] = q(d); r[f"p_{mo}"] = pval(d)
            dd = (D[(k, a)]["new"] - D[(k, a)]["C3"]) - (D[(k, b)]["new"] - D[(k, b)]["C3"])   # difference-in-differences
            if not np.all(np.isnan(dd)):
                r["d_new_C3"] = ra.d_new_C3 - rb.d_new_C3; r["d_new_C3_lo"], r["d_new_C3_hi"] = q(dd); r["p_did"] = pval(dd)
            out.append(r)
    return out


# --------------------------------------------------------------------------------------------- 3. components
def comp_list():
    L = [("new", c, h, res, "n_" + c) for c, (h, res) in NEW.items()]
    L += [("C3", c, h, res, col) for c, (h, res, col) in C3.items()]
    return L


def component_table(df, mf, strata, rng):
    tr = mf[mf.split == "train"]; va = mf[mf.split == "val"]
    trn, van = tr[tr.y == 0], va[va.y == 0]
    mcol = lambda model, c: {"fracture": "new_stored", "C3_stack": "C3_stack"}.get(c, c) if model == "new" or c == "C3_stack" else c
    Lt = []
    for model, c, h, res, col in comp_list():
        mc = mcol(model, c)
        mu, sd = tr[mc].mean(), tr[mc].std(); mun, sdn = trn[mc].mean(), trn[mc].std(); medn = trn[mc].median()
        sl = SITE_LABEL[h]
        tsite = {"vert": "t_vert", "rib": "t_rib", "pel": "t_pel", "all": "y"}[h]
        vs = va[va[tsite].notna()]
        base = dict(part="component", model=model, component=c, head=h, resolution=res,
                    merlin_train_auc=auc(tr.y.values.astype(bool), tr[mc].values), merlin_val_auc=auc(va.y.values.astype(bool), va[mc].values),
                    merlin_val_auc_site=auc(vs[tsite].values.astype(bool), vs[mc].values),
                    merlin_train_neg_mean=mun, merlin_train_neg_sd=sdn, merlin_train_neg_median=medn,
                    merlin_val_neg_z=(van[mc].mean() - mun) / sdn)
        for g, m in strata:
            d = df[m.values].reset_index(drop=True)
            y, _, ok = fx_cohort(d, "fracture"); s = d[col].values[ok]; okk = ~np.isnan(s); y, s = y[okk], s[okk]
            A, B = boot_models(y, {"x": s}, rng)
            ys, _, oks = fx_cohort(d, sl); ss = d[col].values[oks]; o2 = ~np.isnan(ss)
            neg = s[~y]
            r = dict(base, stratum=g, N=len(y), n_pos=int(y.sum()), n_neg=int((~y).sum()), small=(y.sum() < MIN_N) or ((~y).sum() < MIN_N),
                     AUROC_any=A["x"], any_lo=q(B["x"])[0], any_hi=q(B["x"])[1],
                     AUROC_site=auc(ys[o2], ss[o2]) if o2.sum() and 0 < ys[o2].sum() < o2.sum() else np.nan, n_pos_site=int(ys[o2].sum()),
                     neg_z=(neg.mean() - mun) / sdn, neg_z_se=neg.std() / np.sqrt(len(neg)) / sdn if len(neg) > 1 else np.nan,
                     neg_median_shift_sd=(np.median(neg) - medn) / sdn if len(neg) else np.nan, neg_sd_ratio=neg.std() / sdn if len(neg) > 1 else np.nan,
                     pos_z=(s[y].mean() - mun) / sdn if y.sum() else np.nan)
            Lt.append(r)
    return pd.DataFrame(Lt)


# --------------------------------------------------------------------------------------------- 4. which head wins the max
def head_wins(df, mf, strata):
    out = []
    def frac(V, lab):
        w = np.argmax(V, 1); return {f"{lab}_{s}": float(np.mean(w == i)) for i, s in enumerate(("vert", "rib", "pel"))}
    for nm, F, cols in (("Merlin train（OOF 輸入，頭為 in-sample）", mf[mf.split == "train"], ["site_vert", "site_rib", "site_pel"]),
                        ("Merlin val（saved models）", mf[mf.split == "val"], ["valsaved_site_vert", "valsaved_site_rib", "valsaved_site_pel"])):
        r = dict(part="headwin", stratum=nm, N=len(F), n_pos=int((F.y == 1).sum()), n_neg=int((F.y == 0).sum()))
        r.update(frac(F[cols].values, "all")); r.update(frac(F.loc[F.y == 0, cols].values, "neg")); r.update(frac(F.loc[F.y == 1, cols].values, "pos"))
        out.append(r)
    for g, m in strata:
        d = df[m.values].reset_index(drop=True); y = eb.labels(d, "fracture", "B"); cols = ["n_site_vert", "n_site_rib", "n_site_pel"]
        ok = d[cols].notna().all(1).values
        r = dict(part="headwin", stratum="CG " + g, N=int(ok.sum()), n_pos=int((y[ok] == 1).sum()), n_neg=int((y[ok] == 0).sum()))
        V = d.loc[ok, cols].values; yy = y[ok]
        r.update(frac(V, "all"))
        if (yy == 0).any(): r.update(frac(V[yy == 0], "neg"))
        if (yy == 1).any(): r.update(frac(V[yy == 1], "pos"))
        out.append(r)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------- 4b. pelvis 1.5 mm ROI (P15FT)
P15_MED = -2.566798448562622   # stacker.json medians_train["P15FT"]: value given to exams without a pelvis ROI (integrate.py)


def pelvis_roi(df, mf, strata):
    tr = mf[(mf.split == "train") & (mf.y == 0)]; mu, sd = tr.P15FT.mean(), tr.P15FT.std()
    trp = tr[(tr.P15FT - P15_MED).abs() > 1e-3]; mup, sdp = trp.P15FT.mean(), trp.P15FT.std()
    out = [dict(part="pelvis_roi", stratum="Merlin train 陰性", N=len(tr), p15_imputed=float(((tr.P15FT - P15_MED).abs() <= 1e-3).mean()),
                p15_neg_z_roi=(trp.P15FT.mean() - mup) / sdp, p15_neg_median_roi=trp.P15FT.median())]
    for g, m in strata:
        d = df[m.values]; y = eb.labels(d, "fracture", "B"); x = d.n_P15FT.values; imp = np.abs(x - P15_MED) < 1e-6
        neg = (y == 0) & ~imp & ~np.isnan(x)
        pel = d[["n_site_vert", "n_site_rib", "n_site_pel"]].values.argmax(1) == 2
        out.append(dict(part="pelvis_roi", stratum="CG " + g, N=len(d), p15_imputed=float(imp.mean()), n_neg_roi=int(neg.sum()),
                        p15_neg_z_roi=(x[neg].mean() - mup) / sdp if neg.any() else np.nan, p15_neg_median_roi=np.median(x[neg]) if neg.any() else np.nan,
                        pel_wins_neg_roi=float(pel[neg].mean()) if neg.any() else np.nan, pel_wins_neg_noroi=float(pel[(y == 0) & imp].mean()) if ((y == 0) & imp).any() else np.nan))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------- appendix: 14 diseases
def appendix(df, strata, rng):
    L = []
    for k in DIS + ["osteopenia:hu110", "osteopenia:hu160"]:
        f = k.split(":")[0]
        for g, m in strata:
            d = df[m.values].reset_index(drop=True); y = eb.labels(d, k, "B")
            sb = d[f + "__best"].values.astype(float); se = d[f].values.astype(float)
            ok = ~np.isnan(y) & ~np.isnan(sb); yy = y[ok].astype(bool)
            A, B = boot_models(yy, {"best": sb[ok], "E3": se[ok]}, rng)
            r = dict(part="appendix14", row=k, 疾病=eb.RZH[k], stratum=g, N=len(yy), n_pos=int(yy.sum()), n_neg=int((~yy).sum()))
            r["small"] = r["n_pos"] < MIN_N or r["n_neg"] < MIN_N
            for mo in A: r[f"AUROC_{mo}"] = A[mo]; r[f"{mo}_lo"], r[f"{mo}_hi"] = q(B[mo])
            L.append(r)
    return pd.DataFrame(L)


# --------------------------------------------------------------------------------------------- formatting
def ci(r, mo, delta=False):
    a = r.get(f"AUROC_{mo}")
    if a is None or not np.isfinite(a): return "–"
    st = "*" if r["small"] else ""
    if delta:
        p = r.get(f"p_{mo}"); return f"{a:+.3f} [{r[mo + '_lo']:+.3f}, {r[mo + '_hi']:+.3f}] p={p:.2g}{st}"
    return f"{a:.3f} ({r[mo + '_lo']:.3f}–{r[mo + '_hi']:.3f}){st}"


def dnc(r):
    a = r.get("d_new_C3")
    if a is None or not np.isfinite(a): return "–"
    p = r.get("p_did") if str(r["stratum"]).startswith("Δ") else r.get("p_new_C3_DeLong")
    return f"{a:+.3f} [{r['d_new_C3_lo']:+.3f}, {r['d_new_C3_hi']:+.3f}] p={p:.2g}" + ("*" if r["small"] else "")


def conclusions(R, C, CT, HW, PV):
    g = lambda st, k="fracture": R[(R.row == k) & (R.stratum == st)].iloc[0]
    c = lambda st, k="fracture": C[(C.row == k) & (C.stratum == st)].iloc[0]
    hw = lambda st: HW[HW.stratum == st].iloc[0]; pv = lambda st: PV[PV.stratum == st].iloc[0]
    f = lambda r: f"新 {r.AUROC_new:.3f} / C3 {r.AUROC_C3:.3f} / E3 {r.AUROC_E3:.3f}（{r.n_pos}+/{r.n_neg}−{'，*' if r.small else ''}）"
    a, b5, x = g("<5 mm"), g("5 mm"), c("Δ <5 mm − 5 mm")
    nc5, ch5, nc3 = g("5 mm 非 chest 序列"), g("5 mm chest 序列"), g("<5 mm 非 chest 序列")
    return [
        f"- **厚度幾乎沒有變異可分析**：主序列 93% 是 5 mm；<5 mm 在骨折 B cohort 只有 {a.N} 例（陽性 {a.n_pos}），而且其中多數是 chest 序列（LOC3 GE Revolution/Hitachi 的 Chest C+ 3.75 mm）。"
        "所以「厚度」和「chest 序列／骨盆是否在視野內／院區機台」幾乎完全混在一起。",
        f"- 整體骨折：<5 mm {f(a)}；5 mm {f(b5)}。<5 − 5 的差：新 {x.AUROC_new:+.3f} [{x.new_lo:+.3f}, {x.new_hi:+.3f}]、C3 {x.AUROC_C3:+.3f} [{x.C3_lo:+.3f}, {x.C3_hi:+.3f}]，"
        f"(新−C3) 層間差 {x.d_new_C3:+.3f} [{x.d_new_C3_lo:+.3f}, {x.d_new_C3_hi:+.3f}] → **看不出 AUROC 隨厚度變化**（CI 很寬，檢定力不足）。子群在 <5 mm 層陽性都 < 30，不解讀。",
        f"- 真正分得開的是 **chest 序列 vs 非 chest**（同為 5 mm）：非 chest {f(nc5)}；chest 序列 {f(ch5)}。新模型輸 C3 只發生在非 chest（腹部／骨盆在視野內）的 5 mm 序列。",
        f"- 機制（元件層）：新模型的骨盆頭在 CG 嚴重偏移。P15FT（1.5 mm 骨盆 ROI）在有 ROI 的陰性病例 z = {pv('CG 5 mm 非 chest 序列').p15_neg_z_roi:+.1f}（5 mm 非 chest），"
        f"骨盆頭在陰性中決定 max 的比例 {hw('CG 5 mm 非 chest 序列').neg_pel:.0%}（Merlin val {hw('Merlin val（saved models）').neg_pel:.0%}）；"
        f"chest 序列常缺骨盆 ROI（填中位數），骨盆頭勝率只有 {hw('CG 5 mm chest 序列').neg_pel:.0%}，此時新模型反而贏 C3。3 mm 的骨盆元件偏移小得多（5 mm 非 chest 陰性：U_pel/U_org_pel z 約 +0.7～+0.9，L_p/I_p/LJ_p ≈ 0）。"
        "肋骨元件不論 1.5 mm（RVRS/FT_rib/L_r，z +1.7～+2.2）或 3 mm（LJ_r/I_r，z +1.3～+1.5）都偏移、單獨 AUROC 掉到 .55–.63，但肋骨頭只在 20–28% 的陰性中勝出；"
        "椎體元件 1.5 mm 比 3 mm 偏移略大（L_v15 +0.9 vs L_v3 +0.6、FT_sp15 +0.5 vs FT_sp +0.1），但單獨 AUROC 幾乎一樣，椎體頭仍是 CG 最好的頭。",
        f"- 1.5 mm 元件是否在厚片退化更多：唯一乾淨一點的比較是非 chest 的 <5 mm（{nc3.N} 例、{nc3.n_pos}+，*）vs 5 mm：P15FT 有 ROI 陰性 z {pv('CG <5 mm 非 chest 序列').p15_neg_z_roi:+.1f} vs {pv('CG 5 mm 非 chest 序列').p15_neg_z_roi:+.1f}、"
        f"骨盆頭勝率 {pv('CG <5 mm 非 chest 序列').pel_wins_neg_roi:.0%} vs {pv('CG 5 mm 非 chest 序列').pel_wins_neg_roi:.0%}——方向符合「越厚偏移越大」，但 <5 mm 非 chest 只有 {pv('CG <5 mm 非 chest 序列').n_neg_roi:.0f} 個有 ROI 陰性（多為 LOC3 Hitachi/GE 的 'C+ 3.75mm'、'Body C+ 3.75mm'，少數 LOC8 2.5–3 mm），**只能當假說**。"
        "在 GE Revolution＋Hitachi 內比 3.75 vs 5 mm 的 Δz 受 chest 序列（骨盆不在視野）主導，不能當厚度效應。",
        "- 結論：CG 骨折的退化（尤其新模型）主要和「骨盆 1.5 mm 頭在含骨盆的 CG 序列上整體偏高、搶走 max」有關；厚度本身在 CG 內無法與序列種類／院區分開，目前資料無法證明厚度效應。"]


def thinnest_section(df, mf):
    """Section A: strata = thinnest AXIAL series of the exam (any series, not necessarily the one converted)."""
    rng = np.random.default_rng(SEED)   # own stream: section B numbers are unchanged by adding this section
    bins = [b for b in THIN_ORDER if (df.thin_bin == b).any()]
    t5 = df.thin_bin == "最薄 5 mm"
    ST = [(b, df.thin_bin == b) for b in bins] + [("最薄 5 mm・主序列非 chest", t5 & ~df.chest), ("最薄 5 mm・主序列 chest", t5 & df.chest),
          ("最薄 ≤3 mm（3/2.5/1–1.25）", df.thin_mm <= 3.0 + 1e-6), ("最薄 1–1.25 mm・連續（z gap ≤1.5）", (df.thin_bin == "最薄 1–1.25 mm") & df.thin_contig),
          ("最薄 1–1.25 mm・有間隙", (df.thin_bin == "最薄 1–1.25 mm") & ~df.thin_contig)]
    L, D = model_table(df, ST, rng); R = pd.DataFrame(L)
    C = pd.DataFrame(contrast_rows(R, D, [(b, "最薄 5 mm") for b in bins if b != "最薄 5 mm"] + [("最薄 ≤3 mm（3/2.5/1–1.25）", "最薄 5 mm")]))
    HW = head_wins(df, mf, ST); PV = pelvis_roi(df, mf, ST)
    yfx = eb.labels(df, "fracture", "B")
    X = pd.crosstab(df.thin_bin, df.tbin).reindex(index=bins, columns=["≤3 mm", "3.75 mm", "5 mm", "≥7.5 mm"]).fillna(0).astype(int)
    XB = pd.crosstab(df.thin_bin[~np.isnan(yfx)], df.tbin[~np.isnan(yfx)]).reindex(index=bins, columns=X.columns).fillna(0).astype(int)
    X["評估集合計"] = X.sum(1); X["骨折 B cohort"] = XB.sum(1)
    X["骨折 B 陽性/陰性"] = [f"{int(((yfx == 1) & (df.thin_bin == b)).sum())}/{int(((yfx == 0) & (df.thin_bin == b)).sum())}" for b in bins]
    X["chest 主序列"] = [int((df.chest & (df.thin_bin == b)).sum()) for b in bins]
    Xo = X.reset_index().rename(columns={"thin_bin": "最薄 axial"}); Xo.insert(0, "part", "thin_x_main")
    m1 = df.thin_bin == "最薄 1–1.25 mm"
    oth = df.loc[df.thin_bin == "最薄 其他", "thin_mm"].value_counts().sort_index()
    f = lambda r: f"新 {r.AUROC_new:.3f} / C3 {r.AUROC_C3:.3f}"
    def mtab(rows, strata):
        W = []
        for k in rows:
            for g in strata:
                x = R[(R.row == k) & (R.stratum == g)]
                if not len(x): continue
                x = x.iloc[0]
                W.append({"子群": NAME[k], "最薄 axial 層": g, "N/陽性/陰性": f"{x.N}/{x.n_pos}/{x.n_neg}", "新 a6x10w2_tv7": ci(x, "new"), "舊 C3": ci(x, "C3"),
                          "E3_late": ci(x, "E3"), "Δ 新−C3 [配對 boot CI] DeLong p": dnc(x)})
        return pd.DataFrame(W)
    def ctab(rows):
        W = []
        for k in rows:
            for _, x in C[C.row == k].iterrows():
                W.append({"子群": NAME[k], "比較": x.stratum.replace("Δ ", ""), "新": ci(x, "new", True), "C3": ci(x, "C3", True), "E3_late": ci(x, "E3", True),
                          "(新−C3) 的層間差 [boot CI] p": dnc(x)})
        return pd.DataFrame(W)
    hp = HW.copy(); hp["stratum"] = hp.stratum.str.replace("^CG ", "", regex=True)
    pv = PV.copy(); pv["stratum"] = pv.stratum.str.replace("^CG ", "", regex=True)
    HP = hp[["stratum", "N", "n_pos", "n_neg", "neg_vert", "neg_rib", "neg_pel", "pos_pel"]].merge(
        pv[["stratum", "p15_imputed", "n_neg_roi", "p15_neg_z_roi", "pel_wins_neg_roi"]], on="stratum", how="left")
    for c in ("neg_vert", "neg_rib", "neg_pel", "pos_pel", "p15_imputed", "pel_wins_neg_roi"): HP[c] = HP[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else "–")
    HP["p15_neg_z_roi"] = HP.p15_neg_z_roi.map(lambda v: f"{v:+.2f}" if pd.notna(v) else "–")
    HP = HP.rename(columns={"stratum": "資料/層", "neg_vert": "陰性·椎體頭勝", "neg_rib": "陰性·肋骨頭勝", "neg_pel": "陰性·骨盆頭勝", "pos_pel": "陽性·骨盆頭勝",
                            "p15_imputed": "缺骨盆 ROI 比例", "n_neg_roi": "有 ROI 陰性 N", "p15_neg_z_roi": "有 ROI 陰性 P15FT z", "pel_wins_neg_roi": "有 ROI 陰性·骨盆頭勝"})
    LOCX = pd.crosstab(df.thin_bin, [df.LOC, df.scanner]).reindex(index=bins).fillna(0).astype(int)
    LOCX.columns = [f"{l} {sc}" for l, sc in LOCX.columns]
    ax = pd.read_csv(ROOT / "dicom_index/series.csv", usecols=["EXAMID", "ImageType", "SliceThickness"])
    t_all = ax[ax.ImageType.fillna("").str.contains("AXIAL")].groupby("EXAMID").SliceThickness.min()
    meta = pd.read_csv(ROOT / "report_meta_qwen.csv", usecols=["EXAMID", "abdomen"]).drop_duplicates("EXAMID").set_index("EXAMID").abdomen
    ab3 = int(((t_all.map(thin_bin) == "最薄 3 mm") & (meta.reindex(t_all.index) == 1)).sum())
    r5, r1, r3 = (R[(R.row == "fracture") & (R.stratum == g)].iloc[0] for g in ("最薄 5 mm", "最薄 1–1.25 mm", "最薄 ≤3 mm（3/2.5/1–1.25）"))
    c3 = C[(C.row == "fracture") & (C.stratum == "Δ 最薄 ≤3 mm（3/2.5/1–1.25） − 最薄 5 mm")].iloc[0]
    md = ["## A. 依每例「最薄 axial 序列」分層（使用者 11-10 指定）", "",
          "- 定義：`dicom_index/series.csv` 中 ImageType 含 'AXIAL' 的序列，每個 EXAMID 取 min(SliceThickness)（與 10k 全體表相同：5 mm 5490 / 3 mm 1799 / 2.5 mm 983 / 3.75 mm 851 / 1–1.25 mm 781 / 其他 39 / 無 axial 57）。"
          f"「其他」在評估集為 {', '.join(f'{k:g} mm×{v}' for k, v in oth.items())}（10k 全體另有 1 例 0.5 mm 也歸「其他」，所以全體是 39 不是 38）。",
          "- **注意：最薄序列多半不是實際轉成 NIfTI 的主序列**（主序列 93% 是 5 mm，見下方交叉表），所以這個分層反映的是「該檢查是否另有薄切重建」（掃描協定／院區），不是模型看到的影像厚度。",
          f"- 1–1.25 mm 層：評估集 {int(m1.sum())} 例，其中 {int((m1 & df.thin_hrct).sum())} 例的 ≤1.25 mm 序列是 HRCT（1.25/5、1/5、1/3 有間隙的肺窗序列），"
          f"只有 {int((m1 & df.thin_contig).sum())} 例有 z_gap_med ≤ 1.5 mm 的連續薄切（10k 全體 31 例）。",
          f"- 重點：整體骨折 最薄 5 mm {f(r5)}（{r5.n_pos}+/{r5.n_neg}−）、最薄 1–1.25 mm {f(r1)}（{r1.n_pos}+/{r1.n_neg}−）、最薄 ≤3 mm 合併 {f(r3)}（{r3.n_pos}+/{r3.n_neg}−）；"
          f"≤3 − 5 的差：新 {c3.AUROC_new:+.3f} [{c3.new_lo:+.3f}, {c3.new_hi:+.3f}] p={c3.p_new:.2g}、C3 {c3.AUROC_C3:+.3f} [{c3.C3_lo:+.3f}, {c3.C3_hi:+.3f}] p={c3.p_C3:.2g}。",
          "- 統計同下方 B 節（1,000 次 bootstrap，seed 0，此節用自己的亂數串；新−C3 配對；層間差非配對；`*` = 陽性或陰性 < 30）。", "",
          "### A1. 骨折整體 × 最薄 axial", "", mtab(["fracture"], [g for g, _ in ST]).to_markdown(index=False), "",
          "### A2. 各層 − 最薄 5 mm", "", ctab(["fracture"]).to_markdown(index=False), "",
          "### A3. 子群 × 最薄 axial", "",
          mtab(["fracture:vertebra", "fracture:sub_compression", "fracture:only_rib", "fracture:site_rib", "fracture:pelvic_ring", "fracture:site_other"], bins).to_markdown(index=False), "",
          "### A4. 最薄 axial × 主序列厚度（評估集 6,970 例主 C+；N 欄 = 骨折 B cohort）", "", X.to_markdown(), "",
          "- 「chest 主序列」欄 = 該層中實際轉成 NIfTI 的主序列是 chest 序列（`chest_series` flag）的例數。1–1.25 mm（HRCT）層有 "
          f"{int((df.chest & m1).sum())}/{int(m1.sum())} 例主序列是 chest 序列（胸腹一起掃），所以此層的結果和 B 節「chest 序列」的結果高度重疊。", "",
          "### A4b. 最薄 axial × 院區／機台（評估集）", "", LOCX.to_markdown(), "",
          f"- 10k 全體中最薄 3 mm 的 1799 例只有 {ab3} 例是腹部檢查（report_meta abdomen=1），其餘不在評估集，所以評估集裡這一層很小。", "",
          "### A5. 部位頭勝出與 P15FT 偏移（陰性；z 以 Merlin train 有 ROI 陰性計）", "", HP.to_markdown(index=False), "", "---", ""]
    return {"R": R, "C": C, "HW": HW, "PV": PV, "X": Xo, "md": md}


def main():
    rng = np.random.default_rng(SEED)
    df, sel = load()
    mf = pd.read_csv(INP / "merlin_frame.csv")
    allm = pd.Series(True, index=df.index); lt5 = df.eff_mm < 4.9
    gh = df.scanner.isin(["GE Revolution CT", "Hitachi SCENARIA"])
    ST = [("全部", allm), ("≤3 mm", df.tbin == "≤3 mm"), ("3.75 mm", df.tbin == "3.75 mm"), ("<5 mm", lt5), ("5 mm", df.tbin == "5 mm"),
          ("≥7.5 mm", df.tbin == "≥7.5 mm"), ("5 mm 非 chest 序列", (df.tbin == "5 mm") & ~df.chest), ("5 mm chest 序列", (df.tbin == "5 mm") & df.chest),
          ("<5 mm chest 序列", lt5 & df.chest), ("<5 mm 非 chest 序列", lt5 & ~df.chest), ("GE Rev+Hitachi 3.75 mm", gh & (df.tbin == "3.75 mm")), ("GE Rev+Hitachi 5 mm", gh & (df.tbin == "5 mm"))]
    L, D = model_table(df, ST, rng); R = pd.DataFrame(L)
    C = pd.DataFrame(contrast_rows(R, D, [("<5 mm", "5 mm"), ("3.75 mm", "5 mm"), ("<5 mm chest 序列", "5 mm chest 序列"), ("<5 mm 非 chest 序列", "5 mm 非 chest 序列"),
                                          ("GE Rev+Hitachi 3.75 mm", "GE Rev+Hitachi 5 mm")]))
    SD = dict(ST)
    CST = [(g, SD[g]) for g in ("全部", "3.75 mm", "<5 mm", "5 mm", "5 mm 非 chest 序列", "5 mm chest 序列", "<5 mm 非 chest 序列",
                                "GE Rev+Hitachi 3.75 mm", "GE Rev+Hitachi 5 mm")]
    CT = component_table(df, mf, CST, rng)
    HW = head_wins(df, mf, CST)
    PV = pelvis_roi(df, mf, CST)
    AP = appendix(df, [ST[0], ST[1], ST[2], ST[3], ST[4], ST[5]], rng)
    TH = thinnest_section(df, mf)
    pd.concat([TH["R"].assign(section="thinnest_axial"), TH["C"].assign(section="thinnest_axial"), TH["HW"].assign(section="thinnest_axial"),
               TH["PV"].assign(section="thinnest_axial"), TH["X"].assign(section="thinnest_axial"),
               R, C, CT, HW, PV, AP], ignore_index=True).to_csv(BD / "eval_fracture_by_thickness.csv", index=False)

    # ---------------- sanity vs existing tables (point estimates and N are deterministic)
    T = pd.read_csv(BD / "eval_best_main_table.csv"); F = pd.read_csv(BD / "eval_fracture_newbest_table.csv")
    san = []
    for k in FX_ROWS[:-1]:
        r = R[(R.row == k) & (R.stratum == "全部")].iloc[0]; f = F[F.疾病 == eb.RZH[k]].iloc[0]
        g = lambda s: float(str(s).split()[0].rstrip("*"))
        san.append(dict(項目=eb.RZH[k], N_既有=f["B N/陽性/陰性"], N_本次=f"{r.N}/{r.n_pos}/{r.n_neg}", 新_既有=g(f["B 新模型 a6x10w2_tv7"]), 新_本次=round(r.AUROC_new, 3),
                        C3_既有=g(f["B 舊模型 C3"]), C3_本次=round(r.AUROC_C3, 3), E3_既有=g(f["B E3_late"]), E3_本次=round(r.AUROC_E3, 3)))
    for k in DIS:
        r = AP[(AP.row == k) & (AP.stratum == "全部")].iloc[0]; t = T[T.疾病 == ZH[k]].iloc[0]
        san.append(dict(項目=ZH[k] + "（附錄）", N_既有=str(t["B N/陽性/陰性"]).split("（")[0], N_本次=f"{r.N}/{r.n_pos}/{r.n_neg}",
                        新_既有=float(str(t["B 最佳"]).split()[0].rstrip("*")), 新_本次=round(r.AUROC_best, 3),
                        E3_既有=float(str(t["B E3_late"]).split()[0].rstrip("*")), E3_本次=round(r.AUROC_E3, 3)))
    S = pd.DataFrame(san)
    S["一致"] = [(a == b) and all(not np.isfinite(x) or abs(x - y) < 6e-4 for x, y in ((r.get("新_既有", np.nan), r.get("新_本次", np.nan)),
                 (r.get("C3_既有", np.nan), r.get("C3_本次", np.nan)), (r.get("E3_既有", np.nan), r.get("E3_本次", np.nan))))
                 for a, b, (_, r) in zip(S.N_既有, S.N_本次, S.iterrows())]

    # ---------------- descriptives
    fine = df.eff_mm.value_counts().sort_index()
    dis1 = int(((df.slice_thickness - df.z_gap_med).abs() > 0.05).sum()); dis2 = int(((df.slice_thickness - df.gap_sel).abs() > 0.05).sum())
    nomatch = int(df.z_gap_med.isna().sum())
    bins = ["≤3 mm", "3.75 mm", "5 mm", "≥7.5 mm"]
    yfx = eb.labels(df, "fracture", "B")
    tab_sc = pd.crosstab([df.LOC, df.scanner], df.tbin)[bins]
    tab_sc["骨折 B 陽性/陰性"] = [f"{int(((yfx == 1) & m).sum())}/{int(((yfx == 0) & m).sum())}" for m in
                              [((df.LOC == l) & (df.scanner == s)).values for l, s in tab_sc.index]]
    tab_ch = pd.crosstab(df.chest.map({True: "chest_series", False: "非 chest"}), df.tbin)[bins]
    desc_top = {b: df[df.tbin == b].SeriesDescription.str.replace(r"\s+", " ", regex=True).value_counts().head(3) for b in bins if b != "5 mm"}

    # ---------------- markdown
    def mtab(rows, strata):
        W = []
        for k in rows:
            for g in strata:
                x = R[(R.row == k) & (R.stratum == g)]
                if not len(x): continue
                x = x.iloc[0]
                W.append({"子群": NAME[k], "層": g, "N/陽性/陰性": f"{x.N}/{x.n_pos}/{x.n_neg}", "新 a6x10w2_tv7": ci(x, "new"), "舊 C3": ci(x, "C3"),
                          "E3_late": ci(x, "E3"), "Δ 新−C3 [配對 boot CI] DeLong p": dnc(x)})
        return pd.DataFrame(W)

    def ctab(rows, pairs):
        W = []
        for k in rows:
            for a, b in pairs:
                x = C[(C.row == k) & (C.stratum == f"Δ {a} − {b}")]
                if not len(x): continue
                x = x.iloc[0]
                W.append({"子群": NAME[k], "比較": f"{a} − {b}", "新": ci(x, "new", True), "C3": ci(x, "C3", True), "E3_late": ci(x, "E3", True),
                          "(新−C3) 的層間差 [boot CI] p": dnc(x)})
        return pd.DataFrame(W)

    main_st = ["全部", "≤3 mm", "3.75 mm", "<5 mm", "5 mm", "≥7.5 mm"]
    md = ["# CG10k 骨折：依原始切片厚度分層（診斷用，未在 CG 上調參）", ""] + TH["md"] + [
          "# B. 依主序列（實際轉成 NIfTI 的 C+ 序列）厚度分層", "",
          "## 先講結論", ""] + conclusions(R, C, CT, HW, PV) + ["",
          "## 方法", "",
          "- **主序列厚度**：`nifti_3mm/series_selection.csv` 是 `select_convert.py` 逐例記錄合併的表，直接記下實際轉成 NIfTI 的 SeriesInstanceUID、SliceThickness、"
          "來源切片 z 間距中位數（src_pixel_spacing 第 3 項）。以 SeriesInstanceUID 對 `dicom_index/series.csv`："
          f"對不上 {nomatch} 例；SliceThickness 與 z_gap_med 差 > 0.05 mm：{dis1} 例（selection 記錄：{dis2} 例）→ 有效間距 = 厚度，沒有 gap/重疊重建。",
          f"- 評估集（{len(df)} 例主序列）間距分布：{ {float(k): int(v) for k, v in fine.items()} }。**93% 是 5 mm**；≤3 mm（1/2.5/3 mm 合併）、≥7.5 mm（7.5/10 合併）都很少。無 unknown。",
          "- 標籤、cohort、模型分數同 `eval_fracture_newbest_table.csv`（定義 B；子群 = 該部位陽性 vs 全部骨折陰性）。1,000 次 exam 層級 bootstrap（seed 0）；"
          "新−C3 為同一批重抽的配對差＋配對 DeLong p；層間差（如 <5 mm − 5 mm）為兩層各自 bootstrap 的非配對差，p 為 bootstrap 雙尾；"
          "「(新−C3) 的層間差」= difference-in-differences。`*` = 陽性或陰性 < 30，**不宜解讀**。",
          "- 元件分數：A6000 `fracture_a6x10w2_tv7/integ/cg_main_scores.csv`（stacker 實際輸入、三個部位頭 logit、case = max）與依 `cg10k_best/cg_frac.py stack()` 原樣組出的 C3 成員"
          "（C3 stack 重算與 `cg_scores_main.csv` 差 < 1e-15）；Merlin 參考 = integrate.py 的 train（OOF 輸入）/ val frame（重算 val AUROC .8973 與存檔一致）。"
          "複製到 `owl_infer/best/fx_thickness_inputs/`（小 CSV；A6000 端只讀、未用 GPU）。",
          "", "## 1. 骨折整體：新模型 vs 舊 C3 vs E3_late", "", mtab(["fracture"], main_st + ["5 mm 非 chest 序列", "5 mm chest 序列", "<5 mm chest 序列",
                                                                                 "GE Rev+Hitachi 3.75 mm", "GE Rev+Hitachi 5 mm"]).to_markdown(index=False), "",
          "### 層間比較（趨勢檢定）", "", ctab(["fracture"], [("<5 mm", "5 mm"), ("3.75 mm", "5 mm"), ("<5 mm chest 序列", "5 mm chest 序列"),
                                                      ("GE Rev+Hitachi 3.75 mm", "GE Rev+Hitachi 5 mm")]).to_markdown(index=False), "",
          "## 2. 子群 × 厚度", "", mtab(FX_ROWS[1:], ["全部", "<5 mm", "5 mm"]).to_markdown(index=False), "",
          ctab(FX_ROWS[1:], [("<5 mm", "5 mm")]).to_markdown(index=False), "",
          "- <5 mm 層各子群陽性都 < 30（椎體 20、只有肋骨 9、骨盆 2、髖股骨 0）→ 子群的厚度比較**無檢定力**，只能看 5 mm 層。", ""]
    # components: wide table AUROC by stratum + shift
    CW = []
    for (model, c), g in CT.groupby(["model", "component"], sort=False):
        x = g.set_index("stratum"); b = g.iloc[0]
        w = {"模型": model, "元件": c, "頭": HEADZ.get(b["head"], "全部"), "解析度": b.resolution,
             "Merlin val AUROC（任何骨折）": f"{b.merlin_val_auc:.3f}", "Merlin val AUROC（該部位）": f"{b.merlin_val_auc_site:.3f}"}
        for s in ("全部", "3.75 mm", "5 mm", "5 mm 非 chest 序列", "<5 mm 非 chest 序列", "5 mm chest 序列"):
            r = x.loc[s]; w[f"CG {s} AUROC"] = f"{r.AUROC_any:.3f}" + ("*" if r.small else "")
        w["CG 全部 AUROC（該部位）"] = f"{x.loc['全部'].AUROC_site:.3f}"
        CW.append(w)
    CW = pd.DataFrame(CW)
    SW = []
    for (model, c), g in CT.groupby(["model", "component"], sort=False):
        x = g.set_index("stratum"); b = g.iloc[0]
        w = {"模型": model, "元件": c, "解析度": b.resolution, "Merlin val 陰性 z": f"{b.merlin_val_neg_z:+.2f}"}
        for s in ("全部", "3.75 mm", "5 mm", "<5 mm 非 chest 序列", "5 mm 非 chest 序列", "5 mm chest 序列", "GE Rev+Hitachi 3.75 mm", "GE Rev+Hitachi 5 mm"):
            r = x.loc[s]; w[f"CG {s} 陰性 z"] = f"{r.neg_z:+.2f}"
        r5, r3 = x.loc["GE Rev+Hitachi 5 mm"], x.loc["GE Rev+Hitachi 3.75 mm"]
        dz = r5.neg_z - r3.neg_z; se = np.hypot(r5.neg_z_se, r3.neg_z_se)
        w["Δz 5−3.75（GE Rev+Hitachi）[95% CI]"] = f"{dz:+.2f} [{dz - 1.96 * se:+.2f}, {dz + 1.96 * se:+.2f}]"
        r5, r3 = x.loc["5 mm 非 chest 序列"], x.loc["<5 mm 非 chest 序列"]
        dz = r5.neg_z - r3.neg_z; se = np.hypot(r5.neg_z_se, r3.neg_z_se)
        w["Δz 5−<5（非 chest）[95% CI]"] = f"{dz:+.2f} [{dz - 1.96 * se:+.2f}, {dz + 1.96 * se:+.2f}]"
        w["CG 全部 陰性 SD 比"] = f"{x.loc['全部'].neg_sd_ratio:.2f}"; w["CG 全部 陰性中位數偏移(SD)"] = f"{x.loc['全部'].neg_median_shift_sd:+.2f}"
        SW.append(w)
    SW = pd.DataFrame(SW)
    md += ["## 3. 元件層級", "",
           "### 3a. 各元件單獨的 AUROC（CG 任何骨折，定義 B；`*` 該層陽性或陰性 < 30）", "",
           "- 「該部位」標籤：椎體頭元件 → 骨折－椎體；肋骨頭 → 部位含肋骨；骨盆頭 → 骨盆/薦骨/髖股骨（= Merlin t_pel）。Merlin val 用 t_vert/t_rib/t_pel。", "",
           CW.to_markdown(index=False), "",
           "### 3b. 分數偏移（陰性病例；z = (CG 平均 − Merlin train 陰性平均) / Merlin train 陰性 SD）", "",
           "- Merlin val 陰性 z 是 Merlin 內部的參考（理想 ≈ 0）。Δz 5−3.75 只在同時有 3.75 與 5 mm 的 GE Revolution＋Hitachi（LOC3）內比，去掉院區/機台混雜（但 3.75 mm 多為 chest 序列）。CI 為常態近似。", "",
           SW.to_markdown(index=False), ""]
    HWt = HW.copy()
    for c in [c for c in HWt.columns if c.split("_")[0] in ("all", "neg", "pos")]: HWt[c] = HWt[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else "–")
    HWt = HWt.rename(columns={"stratum": "資料/層", **{f"{p}_{s}": f"{pz}·{HEADZ[s]}頭勝" for p, pz in (("all", "全部"), ("neg", "陰性"), ("pos", "陽性"))
                                                     for s in ("vert", "rib", "pel")}})
    md += ["## 4. 哪個部位頭決定 max（新模型 case = max(椎體, 肋骨, 骨盆 logit)）", "",
           HWt.drop(columns=["part"]).to_markdown(index=False), "",
           "### 4b. 骨盆 1.5 mm ROI（P15FT）：缺 ROI 比例與有 ROI 陰性的偏移", "",
           "- 無骨盆 ROI 的病例 P15FT 被填 train 中位數（integrate.py 規則）。z 以 Merlin train 有 ROI 陰性的平均/SD 計。", "",
           PV.drop(columns=["part"]).rename(columns={"stratum": "資料/層", "p15_imputed": "缺 ROI（填中位數）比例", "n_neg_roi": "有 ROI 陰性 N",
                                                     "p15_neg_z_roi": "有 ROI 陰性 P15FT z", "p15_neg_median_roi": "有 ROI 陰性 P15FT 中位數",
                                                     "pel_wins_neg_roi": "有 ROI 陰性·骨盆頭勝", "pel_wins_neg_noroi": "缺 ROI 陰性·骨盆頭勝"}).to_markdown(index=False, floatfmt=".2f"), "",
           "## 混雜：厚度幾乎由「chest 序列」與院區/機台決定", "",
           "- 3.75 mm 與 ≤3 mm 主序列多數是 **Chest C+** 序列（胸腹一起掃時涵蓋到腹部的胸部序列，`chest_series` flag，C-B 會排除）；≥7.5 mm 多為 CHEST C+ 10 mm。", "",
           tab_ch.to_markdown(), "", "各層最常見描述：" + "；".join(f"{b}：" + "、".join(f"{i} ({v})" for i, v in s.items()) for b, s in desc_top.items()), "",
           "- 院區 × 機台 × 厚度（原始 csv 的 LOC 名稱已是 '??'，以括號數字代稱）：3.75 mm 只出現在 LOC3（GE Revolution CT、Hitachi SCENARIA）；≤3 mm 與 10 mm 主要在 LOC8（Canon、Toshiba）。", "",
           tab_sc.to_markdown(), "",
           "## 附錄：14 病（每病最佳 / E3_late，定義 B）× 厚度", ""]
    AW = []
    for k in DIS + ["osteopenia:hu110", "osteopenia:hu160"]:
        w = {"疾病": eb.RZH[k]}
        for g in main_st:
            x = AP[(AP.row == k) & (AP.stratum == g)].iloc[0]
            w[g] = f"{ci(x, 'best')} / E3 {x.AUROC_E3:.3f}<br>{x.N}/{x.n_pos}/{x.n_neg}" if np.isfinite(x.AUROC_best) else f"–<br>{x.N}/{x.n_pos}/{x.n_neg}"
        AW.append(w)
    md += [pd.DataFrame(AW).to_markdown(index=False), "", "## Sanity check（全部層 vs 既有表）", "", S.to_markdown(index=False), ""]
    (BD / "eval_fracture_by_thickness.md").write_text("\n".join(md), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(S.to_string()); print(mtab(["fracture"], [s for s, _ in ST]).to_string())
    print(ctab(FX_ROWS, [("<5 mm", "5 mm"), ("3.75 mm", "5 mm"), ("GE Rev+Hitachi 3.75 mm", "GE Rev+Hitachi 5 mm")]).to_string())
    print(mtab(FX_ROWS[1:], ["全部", "<5 mm", "5 mm"]).to_string())
    print(CW.to_string()); print(SW.to_string()); print(HWt.to_string()); print(tab_sc); print(tab_ch)


if __name__ == "__main__":
    main()
