"""CG10k F3: C+ (main series) vs C- (nc = non-contrast series of the same C+ exam), per-disease best OWL model vs E3_late.
Paired on exams that have both a C+ main series and an nc series. Definition B (primary; neg = no/probably_no only).
Per row: AUROC on C+ and on C- for best and E3_late, Δ = C- − C+ (paired bootstrap 1000, exam level, seed 0),
ΔΔ = Δ_best − Δ_E3, and best − E3 on C- (paired DeLong + bootstrap).
Best scores of BOTH phases come from the same refit run (cg10k_nc sandbox: main + nc rows in one run); the formal
probs_best_main.csv is only compared (agreement), never replaced.
Osteopenia image label: L1 (L2) trabecular HU from each phase's own 1.5 mm seg (cg_l1hu.py, same rules); HU agreement.
usage: eval_best_nc.py [--report]"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_owl_cg10k as ev
from eval_owl_cg10k import DIS, POS, NEG_EXPLICIT, auc, delong_paired, MIN_POS
import eval_best_cg10k as eb

ROOT = ev.ROOT; OUT = ev.OUT; BD = OUT / "best"; NR = BD / "nc_run"
TAG = "<!-- F3_BEST_NC -->"; TAG_END = "<!-- F3_BEST_NC end -->"
ROWS = DIS + ["hepatic_steatosis:diffuse", "hepatic_steatosis:diffuse_mild", "hepatic_steatosis:diffuse_moderate",
              "hepatic_steatosis:diffuse_severe", "fracture:compression", "fracture:only_rib", "fracture:site_pelvis",
              "fracture:acute_no", "osteopenia:hu110", "osteopenia:hu160"]


def e3(s):
    d = pd.read_csv(OUT / f"probs_{s}.csv")
    for k in DIS:
        sc = [c for c in d.columns if c.startswith(k + "__seed")]
        if sc: d[k] = d[sc].mean(axis=1)
    return d[["study_id"] + DIS].rename(columns={"study_id": "EXAMID"})


def best(s):
    d = pd.read_csv(NR / f"probs_best_{s}.csv")
    return d[["study_id"] + DIS].rename(columns={"study_id": "EXAMID", **{k: k + "__best" for k in DIS}})


def hu(s):
    f = (NR if s == "nc" else ROOT) / f"osteo_hu_label_{s}.csv"
    o = pd.read_csv(f).rename(columns={"study_id": "EXAMID"})
    return o[["EXAMID", "hu", "level_used", "y_hu110", "y_hu160", "excl_reason"]]


def load():
    base, _ = eb.load("main")                       # labels / subgroup fields / phase of the main series
    keep = [c for c in base.columns if not (c in DIS or c.endswith("__best") or c in ("hu", "level_used", "y_hu110", "y_hu160", "excl_reason"))]
    base = base[keep]
    P = {}
    for s in ("main", "nc"):
        d = base.merge(e3(s), on="EXAMID").merge(best(s), on="EXAMID").merge(hu(s), on="EXAMID", how="left")
        P[s] = d
    ids = sorted(set(P["main"].EXAMID[P["main"].phase.isin(["portal", "cplus"])]) & set(P["nc"].EXAMID))
    for s in P: P[s] = P[s].set_index("EXAMID").loc[ids].reset_index()
    return P, ids


def ylab(d, k):
    if k.startswith("osteopenia:hu"):
        return d["y_" + k.split(":")[1]].values.astype(float)
    if k in eb.SUBS or k.startswith(("hepatic_steatosis:", "fracture:")):
        kk = {"fracture:only_rib": "fracture:only_rib", "fracture:site_pelvis": "fracture:site_pelvis", "fracture:acute_no": "fracture:acute_no"}.get(k, k)
        return eb.labels(d, kk, "B")
    return eb.labels(d, k, "B")


def evaluate(P, rng, hu_label=None):
    rows = []
    a, b = P["main"], P["nc"]
    for k in ROWS:
        f = k.split(":")[0]
        if k.startswith("osteopenia:hu"):
            if hu_label is None: continue            # image-label rows only in the per-phase-label passes
            src = {"C+": a, "C-": b}[hu_label]; y = ylab(src, k)
        else:
            y = ylab(a, k)
        S = {(m, ph): d[f + ("__best" if m == "best" else "")].values for m in ("best", "E3") for ph, d in (("C+", a), ("C-", b))}
        ok = ~np.isnan(y)
        for v in S.values(): ok &= ~np.isnan(v)
        yy = y[ok].astype(bool); S = {kk: v[ok] for kk, v in S.items()}
        n, npos = len(yy), int(yy.sum())
        if npos == 0 or npos == n: continue
        idx = rng.integers(0, n, size=(ev.NBOOT, n))
        A = {kk: auc(yy, v) for kk, v in S.items()}
        BS = {kk: np.array([auc(yy[i], v[i]) for i in idx]) for kk, v in S.items()}
        dB = BS[("best", "C-")] - BS[("best", "C+")]; dE = BS[("E3", "C-")] - BS[("E3", "C+")]; dd = dB - dE
        cm = BS[("best", "C-")] - BS[("E3", "C-")]
        _, _, _, _, _, p_nc = delong_paired(yy, S[("best", "C-")], S[("E3", "C-")])
        _, _, _, _, _, p_b = delong_paired(yy, S[("best", "C-")], S[("best", "C+")])
        _, _, _, _, _, p_e = delong_paired(yy, S[("E3", "C-")], S[("E3", "C+")])
        q = lambda x: (np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5))
        r = dict(row=k + (f"@{hu_label}" if hu_label else ""), N=n, n_pos=npos, n_neg=n - npos, small=npos < MIN_POS,
                 best_Cp=A[("best", "C+")], best_Cm=A[("best", "C-")], E3_Cp=A[("E3", "C+")], E3_Cm=A[("E3", "C-")],
                 d_best=A[("best", "C-")] - A[("best", "C+")], d_best_lo=q(dB)[0], d_best_hi=q(dB)[1], p_best=p_b,
                 d_E3=A[("E3", "C-")] - A[("E3", "C+")], d_E3_lo=q(dE)[0], d_E3_hi=q(dE)[1], p_E3=p_e,
                 dd=(A[("best", "C-")] - A[("best", "C+")]) - (A[("E3", "C-")] - A[("E3", "C+")]), dd_lo=q(dd)[0], dd_hi=q(dd)[1],
                 Cm_best_minus_E3=A[("best", "C-")] - A[("E3", "C-")], Cm_lo=q(cm)[0], Cm_hi=q(cm)[1], Cm_p=p_nc)
        rows.append(r)
    return pd.DataFrame(rows)


def hu_agreement(P):
    a = P["main"][["EXAMID", "hu", "level_used", "y_hu110", "y_hu160"]]; b = P["nc"][["EXAMID", "hu", "level_used", "y_hu110", "y_hu160"]]
    m = a.merge(b, on="EXAMID", suffixes=("_cp", "_cm")).dropna(subset=["hu_cp", "hu_cm"])
    same = m[m.level_used_cp == m.level_used_cm]
    d = same.hu_cp - same.hu_cm
    sl, ic = np.polyfit(same.hu_cm, same.hu_cp, 1)
    out = dict(n_pairs=len(m), n_same_level=len(same), mean_diff_Cp_minus_Cm=d.mean(), sd_diff=d.std(), median_diff=d.median(),
               loa_lo=d.mean() - 1.96 * d.std(), loa_hi=d.mean() + 1.96 * d.std(), pearson=np.corrcoef(same.hu_cp, same.hu_cm)[0, 1],
               fit_slope=sl, fit_intercept=ic, Cp_at_Cm110=sl * 110 + ic, Cp_at_Cm160=sl * 160 + ic,
               Cm_equiv_of_Cp110=(110 - ic) / sl, Cm_equiv_of_Cp160=(160 - ic) / sl)
    for t in ("110", "160"):
        x, y = same[f"y_hu{t}_cp"].astype(bool), same[f"y_hu{t}_cm"].astype(bool)
        po = (x == y).mean(); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
        out.update({f"prev{t}_Cp": x.mean(), f"prev{t}_Cm": y.mean(), f"agree{t}": po, f"kappa{t}": (po - pe) / (1 - pe),
                    f"Cp_pos_Cm_neg{t}": int((x & ~y).sum()), f"Cp_neg_Cm_pos{t}": int((~x & y).sum())})
    return out, same


def main(report=False):
    rng = np.random.default_rng(ev.SEED)
    P, ids = load()
    R = pd.concat([evaluate(P, rng)[lambda d: ~d.row.str.startswith("osteopenia:hu")]] +
                  [evaluate(P, rng, hl)[lambda d: d.row.str.startswith("osteopenia:hu")] for hl in ("C+", "C-")], ignore_index=True)
    R.to_csv(BD / "eval_F3_best_nc_paired.csv", index=False)
    H, same = hu_agreement(P)
    pd.DataFrame([H]).to_csv(BD / "eval_F3_osteo_hu_agreement.csv", index=False)
    # agreement of the sandbox main best scores with the formal probs_best_main.csv
    fm = pd.read_csv(BD / "probs_best_main.csv").set_index("study_id"); sm = pd.read_csv(NR / "probs_best_main.csv").set_index("study_id")
    ii = fm.index.intersection(sm.index)
    ag = pd.DataFrame([dict(finding=k, n=len(ii), max_abs_diff=float(np.nanmax(np.abs(fm.loc[ii, k] - sm.loc[ii, k]))),
                            corr=float(np.corrcoef(fm.loc[ii, k].fillna(0), sm.loc[ii, k].fillna(0))[0, 1])) for k in DIS])
    ag.to_csv(BD / "eval_F3_sandbox_vs_formal_main.csv", index=False)
    if report: write_report(R, H, ag, len(ids))
    print(len(ids), "paired exams"); print(R.round(3).to_string()); print(H); print(ag.to_string())
    return R, H, ag, len(ids)


def write_report(R, H, ag, n):
    """F3 section (between TAG and TAG_END) in owl_infer/report.md; replaced on rerun, rest of the report untouched."""
    zh = {**eb.RZH, **ev.ZH}
    f3 = lambda x: "" if pd.isna(x) else f"{x:.3f}"
    sg = lambda x: "" if pd.isna(x) else f"{x:+.3f}"
    pv = lambda p: "" if pd.isna(p) else ("<.001" if p < .001 else f"{p:.3f}")
    T = []
    for r in R.itertuples():
        base, _, hl = r.row.partition("@")
        name = zh.get(base, base) + (f"（標籤取自 {hl}）" if hl else "") + ("*" if r.small else "")
        T.append({"疾病": name, "N/陽性/陰性": f"{r.N}/{r.n_pos}/{r.n_neg}",
                  "最佳 C+": f3(r.best_Cp), "最佳 C-": f3(r.best_Cm),
                  "Δ最佳(C-−C+) [CI] p": f"{sg(r.d_best)} [{sg(r.d_best_lo)}, {sg(r.d_best_hi)}] {pv(r.p_best)}",
                  "E3 C+": f3(r.E3_Cp), "E3 C-": f3(r.E3_Cm),
                  "ΔE3(C-−C+) [CI] p": f"{sg(r.d_E3)} [{sg(r.d_E3_lo)}, {sg(r.d_E3_hi)}] {pv(r.p_E3)}",
                  "ΔΔ(最佳−E3) [CI]": f"{sg(r.dd)} [{sg(r.dd_lo)}, {sg(r.dd_hi)}]",
                  "C- 最佳−E3 [CI] p": f"{sg(r.Cm_best_minus_E3)} [{sg(r.Cm_lo)}, {sg(r.Cm_hi)}] {pv(r.Cm_p)}"})
    L = [TAG, "", "## F3 每病最佳設定在 C-（無顯影）上的延伸（同檢查 C+ vs C- 配對，B 定義）", "",
         f"- 配對檢查數 {n}：同一檢查同時有 C+ main series 與 nc（C-）series。主要定義 B（陰性只限報告明確 no/probably_no）。",
         "- 最佳設定的 C+ 與 C- 分數來自同一次 refit（cg10k_nc sandbox，main + nc 列一起推論，CG 列從不參與訓練）；E3_late 為同一 expB E3_late。",
         "- Δ = C- − C+，1,000 次 exam 層級配對 bootstrap（seed 0）95% CI；p = 配對 DeLong。ΔΔ = Δ最佳 − ΔE3。`*` 陽性 < 30。",
         "- 骨質減少影像標籤：各期別用自己的 1.5 mm seg 算 L1（L2 替代）小樑骨 HU，閾值未依期別校正。", "",
         pd.DataFrame(T).to_markdown(index=False), "",
         "### 骨質減少 L1 HU：C+ vs C- 一致性", "", pd.DataFrame([H]).T.rename(columns={0: "值"}).round(3).to_markdown(), "",
         "### sandbox main 最佳分數 vs 正式 probs_best_main.csv", "",
         ag.assign(疾病=ag.finding.map(ev.ZH))[["疾病", "n", "max_abs_diff", "corr"]].round(4).to_markdown(index=False), "", TAG_END]
    rp = OUT / "report.md"; txt = rp.read_text(encoding="utf-8")
    if TAG in txt and TAG_END in txt:
        txt = txt[:txt.index(TAG)].rstrip() + "\n" + txt[txt.index(TAG_END) + len(TAG_END):].lstrip("\n")
    rp.write_text(txt.rstrip() + "\n\n" + "\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main("--report" in sys.argv)
