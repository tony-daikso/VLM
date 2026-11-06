"""Leave-one-window-out importance for the late-fusion model (E3_late), on the validation split only.

For each window w: retrain E3_late without w (same chosen lr/wd, 3 seeds, same standardization) and report
drop_w(finding) = AUROC_full - AUROC_without_w. Aggregated per organ unit over the findings mapped to it.
This replaces reading the E3_late gate weights, which are not identifiable (each window's head can rescale itself).
Outputs: results/<subdir>/window_ablation_val.{csv,md}
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from common import load_cfg
from probe import auroc, load_data, train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-subdir", required=True, help="where E3_late.npz / E3_late_grid.json live")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    dev = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    ids, E, P, Y, S, units, wins, findings = load_data(cfg)
    if a.standardize:  # identical to probe.py --standardize
        tr = torch.from_numpy(S == "train")
        rows = E[tr][P[tr] > 0]
        mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6)
        E = ((E - mu) / sd) * (P[..., None, None] > 0)
    res = cfg["out"] / "results" / a.results_subdir
    hp = json.loads((res / "E3_late_grid.json").read_text())["chosen"]
    va = S == "val"
    full = np.load(res / "E3_late.npz", allow_pickle=True)
    assert list(full["ids"]) == list(ids), "E3_late.npz was made on a different case set"
    p_full = full["prob"].mean(0)
    auc_full = {f: auroc(Y[va][:, i], p_full[va][:, i]) for i, f in enumerate(findings)}

    rows_out = []
    for wi, w in enumerate(wins):
        keep = [j for j in range(len(wins)) if j != wi]
        data = (ids, E[:, :, keep], P, Y, S, units, [wins[j] for j in keep], findings)
        probs = [train_one(cfg, {"mode": "late"}, data, hp["lr"], hp["wd"], cfg["seed"] + s, dev)[1]
                 for s in range(a.seeds)]
        p = np.mean(probs, 0)
        for i, f in enumerate(findings):
            au = auroc(Y[va][:, i], p[va][:, i])
            rows_out.append({"window": w, "finding": f, "auroc_without": au, "drop": auc_full[f] - au})
        print(f"without {w}: mean drop {np.nanmean([r['drop'] for r in rows_out if r['window'] == w]):+.4f}", flush=True)

    df = pd.DataFrame(rows_out)
    df.to_csv(res / "window_ablation_val.csv", index=False)
    tab = df.pivot(index="finding", columns="window", values="drop").reindex(index=findings, columns=wins)
    fm = cfg["finding_map"]
    org = {}
    for f in findings:
        for u in fm[f]["organs"]:
            org.setdefault(u, []).append(tab.loc[f])
    otab = pd.DataFrame({u: pd.concat(v, axis=1).mean(1) for u, v in org.items()}).T[wins]
    L = ["# Leave-one-window-out (E3_late, val): AUROC drop when the window is removed (+ = window helps)\n",
         "## per finding\n", "| finding | full | " + " | ".join(wins) + " |", "|---" * (len(wins) + 2) + "|"]
    for f in findings:
        L.append(f"| {f} | {auc_full[f]:.3f} | " + " | ".join(f"{tab.loc[f, w]:+.3f}" for w in wins) + " |")
    L += ["\n## per organ unit (mean over its findings)\n", "| unit | " + " | ".join(wins) + " |", "|---" * (len(wins) + 1) + "|"]
    for u in otab.index:
        L.append(f"| {u} | " + " | ".join(f"{otab.loc[u, w]:+.3f}" for w in wins) + " |")
    (res / "window_ablation_val.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
