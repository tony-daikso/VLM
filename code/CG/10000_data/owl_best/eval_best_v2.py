"""CG10k external validation, best_v2: NEW Merlin best gallstones model (gallstones15_ft L2_h15cnn, Merlin val .936) vs the
old CG best (實驗 N：G5c＋膽囊單元 U = N_U, val .881) and E3_late (expB). Fracture is pending (see report).
Same conventions as eval_best_cg10k.py / eval_best_nc.py: definition B (pos yes/probably_yes; neg only no/probably_no;
not_mentioned/unclear excluded), C-B = B without chest_series / thick_slices series, 1,000 exam-level bootstrap (seed 0)
paired 95% CI, paired DeLong p, '*' positives < 30.
Inputs: owl_infer/best_v2/probs_v2_gallstones.csv (study_id, set main|main_sandbox|nc, gs_v2), owl_infer/best/probs_best_main.csv
(old best, formal), owl_infer/best/nc_run/probs_best_{main,nc}.csv (old best from the C+/C- sandbox refit), owl_infer/probs_{main,nc}.csv (E3).
Outputs: owl_infer/best_v2/eval_best_v2_main_table.csv, eval_best_v2_main_long.csv, eval_best_v2_nc_paired.csv"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_owl_cg10k as ev
from eval_owl_cg10k import auc, delong_paired, MIN_POS
import eval_best_cg10k as eb

OUT = ev.OUT; BD = OUT / "best"; V2 = OUT / "best_v2"; NR = BD / "nc_run"
ROWS = ["gallstones"]
METHOD = {"gallstones": "gs15_ft：N_U＋1.5 mm 膽囊單元頭＋TS encoder stage 2–4 微調頭（第二層 LR，tier0）"}
MERLIN = {"gallstones": (.783, .881, .936)}   # E3_late, old CG best (Exp N), new best


def v2(s):
    d = pd.read_csv(V2 / "probs_v2_gallstones.csv"); d = d[d.set == s]
    return d[["study_id", "gs_v2"]].rename(columns={"study_id": "EXAMID", "gs_v2": "gallstones__v2"})


def stats(yy, S, rng, pairs):
    """S: name -> score; returns AUROCs, bootstrap CIs, paired deltas for pairs [(a, b)]"""
    n = len(yy); idx = rng.integers(0, n, size=(ev.NBOOT, n))
    A = {k: auc(yy, v) for k, v in S.items()}; B = {k: np.array([auc(yy[i], v[i]) for i in idx]) for k, v in S.items()}
    r = {}
    for k in S:
        r[f"{k}"] = A[k]; r[f"{k}_lo"], r[f"{k}_hi"] = np.nanpercentile(B[k], [2.5, 97.5])
    for a, b in pairs:
        d = B[a] - B[b]; _, _, _, _, _, p = delong_paired(yy, S[a], S[b])
        r[f"d_{a}_{b}"] = A[a] - A[b]; r[f"d_{a}_{b}_lo"], r[f"d_{a}_{b}_hi"] = np.nanpercentile(d, [2.5, 97.5]); r[f"p_{a}_{b}"] = p
    return r


def main_table():
    rng = np.random.default_rng(ev.SEED)
    df, _ = eb.load("main"); df = df.merge(v2("main"), on="EXAMID", how="left")
    qc = df[~(df.chest.fillna(False) | df.thick.fillna(False))].reset_index(drop=True)
    L = []
    for tag, d in (("B", df), ("C-B", qc)):
        for k in ROWS:
            y = eb.labels(d, k, "B"); S = {"new": d[k + "__v2"].values, "old": d[k + "__best"].values, "E3": d[k].values}
            ok = ~np.isnan(y)
            for v in S.values(): ok &= ~np.isnan(v)
            yy = y[ok].astype(bool); S = {kk: v[ok] for kk, v in S.items()}
            r = stats(yy, S, rng, [("new", "old"), ("new", "E3"), ("old", "E3")])
            L.append(dict(set=tag, row=k, N=len(yy), n_pos=int(yy.sum()), n_neg=int((~yy).sum()), small=yy.sum() < MIN_POS,
                          n_missing_v2=int(np.isnan(d[k + "__v2"].values[~np.isnan(y)]).sum()), **r))
    R = pd.DataFrame(L); R.to_csv(V2 / "eval_best_v2_main_long.csv", index=False)
    f3 = lambda r, k: f"{r[k]:.3f} ({r[k + '_lo']:.3f}–{r[k + '_hi']:.3f})" + ("*" if r["small"] else "")
    fd = lambda r, a, b: f"{r[f'd_{a}_{b}']:+.3f} [{r[f'd_{a}_{b}_lo']:+.3f}, {r[f'd_{a}_{b}_hi']:+.3f}] p={(f"{r[f'p_{a}_{b}']:.3g}" if r[f'p_{a}_{b}'] > 0 else '<1e-16')}"
    W = []
    for k in ROWS:
        w = {"疾病": ev.ZH[k], "CG 使用方法": METHOD[k]}
        for t in ("B", "C-B"):
            r = R[(R.set == t) & (R.row == k)].iloc[0]
            w[f"{t} N/陽性/陰性"] = f"{r.N}/{r.n_pos}/{r.n_neg}"; w[f"{t} 最佳"] = f3(r, "new"); w[f"{t} E3_late"] = f3(r, "E3")
            w[f"{t} Δ(最佳−E3) [boot CI] p"] = fd(r, "new", "E3"); w[f"{t} 舊最佳(v1)"] = f3(r, "old")
            w[f"{t} Δ(最佳−舊最佳) [boot CI] p"] = fd(r, "new", "old")
        e3, old, new = MERLIN[k]
        w["Merlin val：E3_late → 舊 CG 模型 → 新 CG 模型（正式最佳）"] = f"{e3:.3f} → {old:.3f} → {new:.3f} ({new:.3f})"
        W.append(w)
    W = pd.DataFrame(W); W.to_csv(V2 / "eval_best_v2_main_table.csv", index=False)
    return W, R


def nc_paired():
    rng = np.random.default_rng(ev.SEED)
    base, _ = eb.load("main")
    keep = [c for c in base.columns if not (c in ev.DIS or c.endswith("__best") or c in ("hu", "level_used", "y_hu110", "y_hu160", "excl_reason"))]
    base = base[keep]; P = {}
    for s, vs in (("main", "main_sandbox"), ("nc", "nc")):
        e3 = pd.read_csv(OUT / f"probs_{s}.csv")
        for k in ROWS:
            sc = [c for c in e3.columns if c.startswith(k + "__seed")]
            if sc: e3[k] = e3[sc].mean(axis=1)
        old = pd.read_csv(NR / f"probs_best_{s}.csv")[["study_id"] + ROWS].rename(columns={"study_id": "EXAMID", **{k: k + "__best" for k in ROWS}})
        P[s] = base.merge(e3[["study_id"] + ROWS].rename(columns={"study_id": "EXAMID"}), on="EXAMID").merge(old, on="EXAMID").merge(v2(vs), on="EXAMID", how="left")
    ids = sorted(set(P["main"].EXAMID[P["main"].phase.isin(["portal", "cplus"])]) & set(P["nc"].EXAMID))
    for s in P: P[s] = P[s].set_index("EXAMID").loc[ids].reset_index()
    L = []
    for k in ROWS:
        y = eb.labels(P["main"], k, "B")
        S = {f"{m}_{ph}": d[k + suf].values for m, suf in (("new", "__v2"), ("old", "__best"), ("E3", "")) for ph, d in (("Cp", P["main"]), ("Cm", P["nc"]))}
        ok = ~np.isnan(y)
        for v in S.values(): ok &= ~np.isnan(v)
        yy = y[ok].astype(bool); S = {kk: v[ok] for kk, v in S.items()}
        r = stats(yy, S, rng, [("new_Cm", "new_Cp"), ("old_Cm", "old_Cp"), ("E3_Cm", "E3_Cp"), ("new_Cm", "old_Cm"), ("new_Cm", "E3_Cm"), ("new_Cp", "old_Cp")])
        L.append(dict(row=k, n_paired_exams=len(ids), N=len(yy), n_pos=int(yy.sum()), n_neg=int((~yy).sum()), small=yy.sum() < MIN_POS, **r))
    R = pd.DataFrame(L); R.to_csv(V2 / "eval_best_v2_nc_paired.csv", index=False)
    return R


if __name__ == "__main__":
    W, R = main_table(); N = nc_paired()
    print(W.T.to_string()); print(R.T.to_string()); print(N.T.to_string())
