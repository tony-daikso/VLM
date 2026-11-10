"""CG10k external validation of the NEW official fracture model a6x10w2_tv7 (Merlin val .897) vs the old CG fracture model
(骨折最終版 C3 LR, CG B .817) and E3_late, on the SAME exams. Labels / subgroups / bootstrap (1000, seed 0, exam level) /
paired DeLong are those of eval_best_cg10k.py (definition B primary; A appendix; C-B = B without chest_series/thick_slices).
Input: owl_infer/best/probs_fracture_newbest_main.csv (study_id, fracture = new model case logit = max of 3 site logits).
Outputs: owl_infer/best/eval_fracture_newbest_{long,table,table_appendixA}.csv"""
import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import eval_best_cg10k as eb
from eval_owl_cg10k import auc, boot_idx, delong_paired, MIN_POS, SEED

BD = eb.BD
ROWS = ["fracture", "fracture:compression", "fracture:vertebra"] + [k for k in eb.SUBS if k.startswith("fracture:")]


def evaluate(df, tag, mode, rng):
    out = []
    for k in ROWS:
        y = eb.labels(df, k, mode); ok = ~np.isnan(y) & df.new.notna().values & df["fracture__best"].notna().values
        yy = y[ok].astype(bool); sn = df.new.values[ok]; so = df["fracture__best"].values[ok]; se = df["fracture"].values[ok]
        n, npos = len(yy), int(yy.sum())
        if npos == 0 or npos == n: continue
        idx = boot_idx(n, rng)
        an, ao, d_no, _, _, p_no = delong_paired(yy, sn, so)
        _, ae, d_ne, _, _, p_ne = delong_paired(yy, sn, se)
        bn = np.array([auc(yy[i], sn[i]) for i in idx]); bo = np.array([auc(yy[i], so[i]) for i in idx]); be = np.array([auc(yy[i], se[i]) for i in idx])
        q = lambda v: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))
        out.append(dict(set=tag, row=k, 疾病=eb.RZH[k], N=n, n_pos=npos, n_neg=n - npos, AUROC_new=an, new_lo=q(bn)[0], new_hi=q(bn)[1],
                        AUROC_old=ao, old_lo=q(bo)[0], old_hi=q(bo)[1], AUROC_E3=ae, E3_lo=q(be)[0], E3_hi=q(be)[1],
                        d_new_old=d_no, d_new_old_lo=q(bn - bo)[0], d_new_old_hi=q(bn - bo)[1], p_new_old=p_no,
                        d_new_E3=d_ne, d_new_E3_lo=q(bn - be)[0], d_new_E3_hi=q(bn - be)[1], p_new_E3=p_ne, small=npos < MIN_POS))
    return pd.DataFrame(out)


def main():
    df, _ = eb.load("main")
    nw = pd.read_csv(BD / "probs_fracture_newbest_main.csv").rename(columns={"study_id": "EXAMID", "fracture": "new"})
    df = df.merge(nw[["EXAMID", "new"]], on="EXAMID", how="left")
    print("exams", len(df), "with new score", int(df.new.notna().sum()))
    rng = np.random.default_rng(SEED)
    qc = df[~(df.chest.fillna(False) | df.thick.fillna(False))].reset_index(drop=True)
    R = pd.concat([evaluate(df, "B", "B", rng), evaluate(df, "A", "A", rng), evaluate(qc, "C-B", "B", rng)])
    R.to_csv(BD / "eval_fracture_newbest_long.csv", index=False)
    f3 = lambda a, l, h: f"{a:.3f} ({l:.3f}–{h:.3f})"
    for sets, name in ((("B", "C-B"), "table"), (("A",), "table_appendixA")):
        W = []
        for k in ROWS:
            r = {"疾病": eb.RZH[k]}
            for t in sets:
                x = R[(R.set == t) & (R.row == k)]
                if not len(x): continue
                x = x.iloc[0]; st = "*" if x.small else ""
                r[f"{t} N/陽性/陰性"] = f"{x.N}/{x.n_pos}/{x.n_neg}"
                r[f"{t} 新模型 a6x10w2_tv7"] = f3(x.AUROC_new, x.new_lo, x.new_hi) + st
                r[f"{t} 舊模型 C3"] = f3(x.AUROC_old, x.old_lo, x.old_hi) + st
                r[f"{t} E3_late"] = f3(x.AUROC_E3, x.E3_lo, x.E3_hi) + st
                r[f"{t} Δ(新−舊) [boot CI] p"] = f"{x.d_new_old:+.3f} [{x.d_new_old_lo:+.3f}, {x.d_new_old_hi:+.3f}] p={x.p_new_old:.3g}"
                r[f"{t} Δ(新−E3) [boot CI] p"] = f"{x.d_new_E3:+.3f} [{x.d_new_E3_lo:+.3f}, {x.d_new_E3_hi:+.3f}] p={x.p_new_E3:.3g}"
            W.append(r)
        pd.DataFrame(W).to_csv(BD / f"eval_fracture_newbest_{name}.csv", index=False)
    print(pd.read_csv(BD / "eval_fracture_newbest_table.csv").to_markdown(index=False))


if __name__ == "__main__":
    main()
