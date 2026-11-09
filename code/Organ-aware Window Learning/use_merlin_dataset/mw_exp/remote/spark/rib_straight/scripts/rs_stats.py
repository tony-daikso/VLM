"""rib_straight tile build stats (no labels read). usage: rs_stats.py BASE -> BASE/tile_stats.json"""
import os, sys, json, numpy as np
from multiprocessing import Pool
B = sys.argv[1]; TD = B + "/tiles"


def one(f):
    with np.load(os.path.join(TD, f)) as z:
        t = z["tab"]; o = z["oor"]; return t, o, int(z["hu"].shape[0])


if __name__ == "__main__":
    fs = [f for f in os.listdir(TD) if f.endswith(".npz") and ".tmp" not in f]
    with Pool(12) as p: R = p.map(one, fs, chunksize=50)
    st = {}
    for l in open(B + "/status.tsv"):
        q = l.rstrip("\n").split("\t"); st[q[0]] = q[1]
    nr = np.array([len(t) for t, _, _ in R]); rows = np.array([r for _, _, r in R])
    T = np.concatenate([t for t, _, _ in R]); O = np.concatenate([o for _, o, _ in R])
    lvl = (T[:, 0] - 1) % 12 + 1
    res = dict(n_cases=len(fs), status={s: sum(1 for v in st.values() if v.split()[0] == s) for s in set(v.split()[0] for v in st.values())},
               ribs_per_case=dict(mean=float(nr.mean()), median=float(np.median(nr)), p5=float(np.percentile(nr, 5)), min=int(nr.min()), max=int(nr.max())),
               rows_per_case=dict(mean=float(rows.mean()), p95=float(np.percentile(rows, 95)), max=int(rows.max())),
               chunks_per_case_est=float(np.mean([sum(1 + max(0, -(-(L - 64) // 48)) for L in t[:, 2]) for t, _, _ in R])),
               n_ribs=len(T), true_len_mm_by_level={int(l): float(T[lvl == l, 4].mean() * 1.5) for l in range(1, 13)},
               ribs_by_level={int(l): int((lvl == l).sum()) for l in range(1, 13)},
               oor_mean=float(O.mean()), oor_gt_0_3=float((O > 0.3).mean()), left=int((T[:, 0] <= 12).sum()), right=int((T[:, 0] > 12).sum()),
               mb_total=float(sum(os.path.getsize(os.path.join(TD, f)) for f in fs) / 2 ** 20))
    json.dump(res, open(B + "/tile_stats.json", "w"), indent=1); print(json.dumps(res, indent=1))
