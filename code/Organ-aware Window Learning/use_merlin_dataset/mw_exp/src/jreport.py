"""Experiment J report: val AUROC (any fracture / compression / mild compression) with paired bootstrap CIs vs the
references, age-matched AUROC, Spearman(score, age | negatives), per-site AUROC. Val only; test never read.

Inputs are OOF-format npz files (ids, split, logit) or final_stack scores.npz (ids, split, stack); several files
per model (comma / glob) are averaged. Rows: --model name=spec (first --ref is the primary comparison).
Train OOF AUROC is shown for information (selection happens elsewhere, on train only).
usage: python jreport.py --ref bone_ens=... --ref E3_late=... --model J_add='.../oof_v15_add_s*.npz' --out DIR
       [--ids-file restrict val to these ids (interim tier0)]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from age_shortcut import matched_auc  # noqa: E402
from final_stack import R, load_labels_trval, split_trval, to_md  # noqa: E402
from fracture_eval_oof import boot_diff  # noqa: E402
from probe import auroc  # noqa: E402

MILD = R / "mw_exp/outputs_fracture/analysis_compression/comp_mild_trval.csv"


def load_any(spec):
    files = sorted({f for g in spec.split(",") for f in glob.glob(g)})
    assert files, spec
    ss = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        assert set(np.unique(z["split"])) <= {"train", "val"}
        v = z["logit"] if "logit" in z else z["stack"]
        ss.append(pd.Series(np.asarray(v, float), index=[str(i) for i in z["ids"]]))
    return pd.concat(ss, axis=1, sort=False).mean(1), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--title", default="實驗 J：1.5 mm 椎體 encoder")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spl = split_trval()
    lab = load_labels_trval(set(spl.index))
    lab = lab[~lab["y"].isna()]
    age = pd.read_csv(R.parent / "merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")["Age"]
    mild = pd.read_csv(MILD).set_index("id")["mild"]
    df = lab.join(spl.rename("split"), how="inner").join(age.rename("age"))
    df["mild"] = df.index.map(mild).fillna(False).astype(bool) & df.compression.astype(bool) & (df.y == 1)
    names = []
    for kind, specs in (("ref", a.ref), ("model", a.model)):
        for sp in specs:
            n, p = sp.split("=", 1)
            s, k = load_any(p)
            df[n] = s
            names.append((n, kind, k))
    cols = [n for n, _, _ in names]
    tr = df[df.split == "train"]
    va = df[(df.split == "val")].dropna(subset=cols + ["age"])
    if a.ids_file:
        keep = set(pd.read_csv(a.ids_file).study_id) if a.ids_file.endswith(".csv") else set(open(a.ids_file).read().split())
        va = va[va.index.isin(keep)]
    y = va.y.values.astype(int); agev = va.age.values.astype(float); neg = y == 0
    groups = {
        "任何骨折": y == 1,
        "椎體壓迫性": (y == 1) & va.compression.values.astype(bool),
        "輕度壓迫": va.mild.values,
        "椎體(部位)": (y == 1) & va.sites.apply(lambda s: "vertebra" in s).values,
        "肋骨": (y == 1) & va.sites.apply(lambda s: "rib" in s).values,
        "骨盆/薦椎/髖": (y == 1) & va.sites.apply(lambda s: any(x in s for x in ("pelvis", "sacrum", "hip_femur"))).values,
        "急性(含混合)": (y == 1) & va.acute.isin(["yes", "mixed"]).values,
        "陳舊": (y == 1) & (va.acute == "no").values,
    }
    refs = [n for n, k, _ in names if k == "ref"]
    rows, drows = [], []
    for n, kind, k in names:
        p = va[n].values
        trm = tr[n].notna() & tr.y.notna()
        r = {"模型": n, "檔數": k, "train OOF AUROC": auroc(tr.y[trm].values, tr[n][trm].values) if trm.sum() > 100 else np.nan}
        for g, m in groups.items():
            sel = m | neg
            r[g] = auroc(m[sel].astype(float), p[sel])
        r["年齡配對(任何)"] = matched_auc(y, p, agev, 2.0)
        sel = groups["椎體壓迫性"] | neg
        r["年齡配對(壓迫)"] = matched_auc(groups["椎體壓迫性"][sel].astype(int), p[sel], agev[sel], 2.0)
        r["Spearman(分數,年齡|陰性)"] = spearmanr(p[neg], agev[neg])[0]
        rows.append(r)
        for rf in refs:
            if rf == n:
                continue
            d = {"模型": n, "vs": rf}
            for g in ("任何骨折", "椎體壓迫性", "輕度壓迫"):
                sel = groups[g] | neg
                yy = groups[g][sel].astype(int)
                b = boot_diff(yy, p[sel], va[rf].values[sel], agev[sel], 2.0, a.n_boot)
                d[f"Δ{g}"] = f"{b['auc'][0]:+.3f} [{b['auc'][1]:+.3f}, {b['auc'][2]:+.3f}]"
                d[f"Δ年齡配對({g})"] = f"{b['matched'][0]:+.3f} [{b['matched'][1]:+.3f}, {b['matched'][2]:+.3f}]"
            drows.append(d)
    res, dres = pd.DataFrame(rows), pd.DataFrame(drows)
    res.to_csv(out / "jreport.csv", index=False); dres.to_csv(out / "jreport_diff.csv", index=False)
    hdr = (f"# {a.title}（val N={len(va)}：骨折 {int(y.sum())}、陰性 {int(neg.sum())}；"
           + "、".join(f"{g} {int(m.sum())}" for g, m in groups.items() if g != "任何骨折") + "）\n\n")
    txt = hdr + "## AUROC（各群陽性 vs 全部陰性）\n\n" + to_md(res) + "\n\n## 配對 bootstrap ΔAUROC（95% CI）\n\n" + to_md(dres) + "\n"
    (out / "jreport.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
