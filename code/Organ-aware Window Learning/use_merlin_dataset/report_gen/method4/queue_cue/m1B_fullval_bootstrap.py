"""Paired case-level bootstrap of macro F1 (owl, strict) differences between evaluate.py runs (aggregates only).
usage: python m1B_fullval_bootstrap.py A=per_case_categories.jsonl B=... [C=...]   (first one is the reference arm)"""
import json, sys
import numpy as np
D14 = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "hydronephrosis", "abdominal_aortic_aneurysm",
       "atherosclerosis", "osteopenia", "bowel_obstruction", "appendicitis", "ascites", "free_air", "gallstones",
       "biliary_ductal_dilation", "fracture"]
POS = {"yes", "probably_yes"}
arms = []
for a in sys.argv[1:]:
    n, p = a.split("=", 1)
    arms.append((n, {r["case_id"]: r for r in map(json.loads, open(p))}))
ids = sorted(set.intersection(*[set(m) for _, m in arms]))
def mats(m):
    ref = np.full((len(ids), 14), np.nan); gen = np.zeros((len(ids), 14))
    for i, c in enumerate(ids):
        for j, d in enumerate(D14):
            r, g = m[c][d]
            ref[i, j] = np.nan if r == "unclear" else float(r in POS)
            gen[i, j] = float(g in POS)
    return ref, gen
M = [(n, *mats(m)) for n, m in arms]
assert all(np.array_equal(np.isnan(M[0][1]), np.isnan(x[1])) for x in M), "reference labels differ"
def macro(ref, gen, idx, drop=()):
    r, g = ref[idx], gen[idx]; f = []
    for j in range(14):
        if D14[j] in drop: continue
        v = ~np.isnan(r[:, j]); tp = np.sum((r[v, j] == 1) & (g[v, j] == 1)); fp = np.sum((r[v, j] == 0) & (g[v, j] == 1))
        fn = np.sum((r[v, j] == 1) & (g[v, j] == 0)); f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return np.mean(f)
rng = np.random.default_rng(20261112); B = 2000; full = np.arange(len(ids))
boots = [rng.integers(0, len(ids), len(ids)) for _ in range(B)]
print(f"cases {len(ids)}, bootstrap {B}")
for drop in [(), ("ascites",)]:
    base = M[0]
    for n, ref, gen in M[1:]:
        d0 = macro(base[1], base[2], full, drop) - macro(ref, gen, full, drop)
        ds = [macro(base[1], base[2], b, drop) - macro(ref, gen, b, drop) for b in boots]
        lo, hi = np.percentile(ds, [2.5, 97.5])
        print(f"{'excl ascites ' if drop else ''}{base[0]} - {n}: macro F1 diff {d0:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
