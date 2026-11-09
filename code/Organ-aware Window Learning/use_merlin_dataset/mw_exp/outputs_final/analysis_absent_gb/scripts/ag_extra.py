"""AUROC-loss decomposition by (positive stratum x negative stratum) and val bootstrap CI for the stack (descriptive)."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as A
D = sys.argv[1]
T = pd.read_csv(D + "per_case_train_oof.csv"); V = pd.read_csv(D + "stack_val_descriptive.csv")
lab = pd.read_csv(D + "labels_trval.csv")[["study_id", "y_chole"]]; V = V.merge(lab, on="study_id"); V = V[V.y_chole.notna()]
BEST = "stack_E3_late+vol+fossa+clip_HGB"
y = T.y_chole.values.astype(int)
def pstrat(r):
    if r.kw_chole == 0: return "陽性:報告無切除字樣"
    if r.gb_vox >= 100: return "陽性:有膽囊遮罩(>=100)"
    return "陽性:膽囊遮罩缺失"
def nstrat(r):
    if r.gb_vox >= 100: return "陰性:有膽囊遮罩"
    if r.kw_contracted == 1: return "陰性:遮罩缺失-膽囊收縮"
    if r.y_gs == 1: return "陰性:遮罩缺失-充滿結石/膽結石"
    return "陰性:遮罩缺失-其他(分割失敗/未提及)"
ps = np.array([pstrat(r) if yy else "" for r, yy in zip(T.itertuples(), y)])
ns = np.array([nstrat(r) if not yy else "" for r, yy in zip(T.itertuples(), y)])
rows = []
for col in ("E3_late", BEST):
    s = T[col].values; P = s[y == 1]; N = s[y == 0]; pl = ps[y == 1]; nl = ns[y == 0]
    tot = len(P) * len(N)
    for a in np.unique(pl):
        for b in np.unique(nl):
            p = P[pl == a]; n = N[nl == b]
            # discordant + 0.5 tie pairs
            ns_ = np.sort(n); lo = np.searchsorted(ns_, p, "left"); hi = np.searchsorted(ns_, p, "right")
            bad = ((len(n) - hi) + 0.5 * (hi - lo)).sum()
            rows.append(dict(score=col, pos_stratum=a, neg_stratum=b, n_pos=len(p), n_neg=len(n), auroc_loss=bad / tot))
d = pd.DataFrame(rows)
for col in ("E3_late", BEST):
    x = d[d.score == col].sort_values("auroc_loss", ascending=False)
    print(f"\n{col}: total loss {x.auroc_loss.sum():.4f} (AUROC {1 - x.auroc_loss.sum():.4f})")
    print(x[x.auroc_loss > 0.0003].to_string(index=False))
d.to_csv(D + "auroc_loss_decomposition.csv", index=False)
# val descriptive bootstrap
rng = np.random.default_rng(0); yv = V.y_chole.values.astype(int); base = A(yv, V.E3_late); st = A(yv, V[BEST])
bs = []
for _ in range(1000):
    i = rng.integers(0, len(V), len(V))
    if 0 < yv[i].sum() < len(i): bs.append(A(yv[i], V[BEST].values[i]) - A(yv[i], V.E3_late.values[i]))
print(f"\nVAL (descriptive): E3_late {base:.4f} -> stack {st:.4f}  delta {st-base:+.4f} 95%CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]")
# age-matched-ish: AUROC within age bins averaged (val)
lab2 = pd.read_csv(D + "labels_trval.csv")[["study_id", "Age"]]; V2 = V.merge(lab2, on="study_id")
V2["ab"] = pd.cut(V2.Age, [0, 40, 50, 60, 70, 80, 200])
for c in ("E3_late", BEST):
    w = [(len(g), A(g.y_chole, g[c])) for _, g in V2.groupby("ab") if 0 < g.y_chole.sum() < len(g)]
    print(f"val age-stratified AUROC {c}: {sum(n*a for n,a in w)/sum(n for n,_ in w):.4f}")
