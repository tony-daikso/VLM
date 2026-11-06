"""Step 6: test AUROC, bootstrap CIs, paired differences, per-unit win rate, gate heatmap.

Predictions: the mean probability over the 3 seeds (seed ensemble). The per-seed mean AUROC is also reported.
Bootstrap: 1000 resamples by patient if the split file has a 'patient_id' column, otherwise by study.
Findings with < min_test_pos test positives are excluded from the main analysis and listed separately.
Subsets: control (control != none), low_contrast (non-control, low_contrast), other.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import load_cfg, log_progress
from probe import auroc

PAIRS = [("E3", "E0"), ("E4", "E0"), ("E3", "E4")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-test-pos", type=int, default=10, help="min positives in the evaluated split")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="val while tuning the finding map; test ONLY once, after everything is locked")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    res = cfg["out"] / "results"
    exps = sorted(p.stem for p in res.glob("E*.npz"))
    exps = sorted(exps, key=lambda e: (e[:2], e != "E0"))
    R = {e: np.load(res / f"{e}.npz", allow_pickle=True) for e in exps}
    z = R[exps[0]]
    findings, ids, S = list(z["findings"]), z["ids"], z["split"]
    te = S == a.split
    lab = pd.read_csv(cfg["labels_file"]).set_index("study_id")
    Y = lab.loc[ids[te], findings].values.astype(float)
    spl = pd.read_csv(cfg["split_file"]).set_index("study_id")
    groups = spl.loc[ids[te], "patient_id"].values if "patient_id" in spl.columns else ids[te]
    boot_unit = "patient" if "patient_id" in spl.columns else "study (no patient id available)"
    prob = {e: R[e]["prob"][:, te].mean(0) for e in exps}          # seed ensemble
    prob_seed = {e: R[e]["prob"][:, te] for e in exps}

    fm = cfg["finding_map"]
    npos = np.nansum(Y, 0).astype(int)
    included = [f for i, f in enumerate(findings) if npos[i] >= a.min_test_pos and (Y[:, i] == 0).sum() > 0]
    excluded = {f: int(npos[i]) for i, f in enumerate(findings) if f not in included}
    subset = {
        "control": [f for f in included if fm[f]["control"] != "none"],
        "low_contrast": [f for f in included if fm[f]["control"] == "none" and fm[f]["low_contrast"]],
        "other": [f for f in included if fm[f]["control"] == "none" and not fm[f]["low_contrast"]],
    }
    subset["main_all"] = subset["low_contrast"] + subset["other"]
    fi = {f: findings.index(f) for f in findings}

    # point estimates
    per_f = {e: {f: auroc(Y[:, fi[f]], prob[e][:, fi[f]]) for f in findings} for e in exps}
    per_f_seedmean = {e: {f: float(np.nanmean([auroc(Y[:, fi[f]], ps[:, fi[f]]) for ps in prob_seed[e]]))
                          for f in findings} for e in exps}

    # bootstrap over groups
    rng = np.random.default_rng(cfg["seed"])
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(ug))]
    boot = {e: {f: [] for f in included} for e in exps}
    for _ in range(a.n_boot):
        gi = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([members[g] for g in gi])
        for f in included:
            y = Y[idx, fi[f]]
            for e in exps:
                boot[e][f].append(auroc(y, prob[e][idx, fi[f]]))
    boot = {e: {f: np.array(v) for f, v in d.items()} for e, d in boot.items()}

    def ci(arr):
        arr = arr[~np.isnan(arr)]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if len(arr) else [np.nan, np.nan]

    table = {}
    for e in exps:
        table[e] = {}
        for s, fs in subset.items():
            if not fs:
                continue
            pt = float(np.nanmean([per_f[e][f] for f in fs]))
            bs = np.nanmean(np.stack([boot[e][f] for f in fs]), 0)
            table[e][s] = {"auroc": pt, "ci95": ci(bs), "n_findings": len(fs)}
    diffs = {}
    for x, y in PAIRS:
        if x in exps and y in exps:
            diffs[f"{x}-{y}"] = {}
            for s, fs in subset.items():
                if not fs:
                    continue
                d = np.nanmean(np.stack([boot[x][f] - boot[y][f] for f in fs]), 0)
                pt = float(np.nanmean([per_f[x][f] - per_f[y][f] for f in fs]))
                c = ci(d)
                diffs[f"{x}-{y}"][s] = {"diff": pt, "ci95": c, "ci_excludes_0": bool(c[0] > 0 or c[1] < 0)}

    win = {}
    if "E3" in exps:
        for f in subset["main_all"]:
            for u in fm[f]["organs"]:
                win.setdefault(u, []).append(per_f["E3"][f] > per_f["E0"][f])
        win = {u: {"win_rate": float(np.mean(v)), "n": len(v)} for u, v in win.items()}

    if "E3" in exps and R["E3"]["gate"].size:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = R["E3"]["gate"].mean(0)
        units, wins = list(R["E3"]["units"]), list(R["E3"]["windows"])
        pd.DataFrame(g, index=units, columns=wins).to_csv(cfg["out"] / "results" / "gate_weights.csv")
        fig, ax = plt.subplots(figsize=(1.2 * len(wins) + 2, 0.45 * len(units) + 1.5))
        im = ax.imshow(g, cmap="viridis", vmin=0, vmax=max(0.5, g.max()))
        ax.set_xticks(range(len(wins)), wins, rotation=30, ha="right"); ax.set_yticks(range(len(units)), units)
        for i in range(len(units)):
            for j in range(len(wins)):
                ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if g[i, j] < 0.6 * max(0.5, g.max()) else "black")
        fig.colorbar(im, ax=ax, fraction=0.04); ax.set_title("E3 gate weights (mean over seeds)")
        fig.tight_layout(); fig.savefig(cfg["out"] / "results" / "gate_weights.png", dpi=120); plt.close(fig)

    summary = {"split": a.split, "n_eval": int(te.sum()), "bootstrap_unit": boot_unit, "n_boot": a.n_boot,
               "included": included, "excluded_lt_min_pos": excluded, "subsets": subset,
               "test_pos": {f: int(npos[i]) for i, f in enumerate(findings)},
               "per_finding_auroc": per_f, "per_finding_auroc_seedmean": per_f_seedmean,
               "subset_table": table, "paired_diffs": diffs, "win_rate_E3_vs_E0": win}
    (res / f"summary_{a.split}.json").write_text(json.dumps(summary, indent=1, default=float))

    # markdown tables
    L = [f"split = {a.split}; N = {int(te.sum())}; bootstrap unit = {boot_unit}; {a.n_boot} resamples\n",
         "| exp | " + " | ".join(s for s in subset if subset[s]) + " |", "|---" * (1 + sum(bool(v) for v in subset.values())) + "|"]
    for e in exps:
        L.append(f"| {e} | " + " | ".join(f"{t['auroc']:.3f} [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]"
                                          for s, t in table[e].items()) + " |")
    L += ["", "| pair | subset | diff | 95% CI | CI excludes 0 |", "|---|---|---|---|---|"]
    for p, d in diffs.items():
        for s, t in d.items():
            L.append(f"| {p} | {s} | {t['diff']:+.3f} | [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | {t['ci_excludes_0']} |")
    L += ["", "| finding | test pos | " + " | ".join(exps) + " |", "|---" * (2 + len(exps)) + "|"]
    for f in findings:
        mark = "" if f in included else " (excl.)"
        L.append(f"| {f}{mark} | {npos[fi[f]]} | " + " | ".join(f"{per_f[e][f]:.3f}" for e in exps) + " |")
    (res / f"tables_{a.split}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    log_progress(f"evaluate.py: {len(exps)} experiments, {len(included)} findings included, "
                 f"{len(excluded)} excluded (<{a.min_test_pos} test pos) -> {res}/summary_{a.split}.json")


if __name__ == "__main__":
    main()
