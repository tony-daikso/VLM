"""Experiment L report: val AUROC for any fracture / rib / rib-only / acute & old rib / other sites, paired bootstrap
CIs vs the reference(s), age-matched AUROC, Spearman(score, age | negatives), and rib localization (top-scoring rib
unit vs the report's side / level) on val and train OOF rib positives. Val only for reporting; test never read.

usage: python rib15_eval.py --ref J_add='.../oof_v15_add_s*.npz' --model L='.../oof_L_*_s*.npz' --out DIR
       [--ids-file tier0 csv: restrict val+train to those ids] [--common: evaluate on rows scored by every model]
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

MW = R / "mw_exp"
SD = {"L": {"left"}, "R": {"right"}, "B": {"left", "right"}}


def load_any(spec):
    files = sorted({f for g in spec.split(",") for f in glob.glob(g)})
    assert files, spec
    ss, inst = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        assert set(np.unique(z["split"])) <= {"train", "val"}
        v = z["logit"] if "logit" in z else z["stack"]
        idx = [str(i) for i in z["ids"]]
        ss.append(pd.Series(np.asarray(v, float), index=idx))
        if "inst" in z and z["inst"].ndim == 2 and z["inst"].shape[1] == 18 and len(z["inst_units"]) == 18:
            inst.append(pd.DataFrame(z["inst"], index=idx, columns=[str(u).split("_", 1)[1] for u in z["inst_units"]]))
    I = None
    if inst:
        I = sum(inst) / len(inst)
    return pd.concat(ss, axis=1, sort=False).mean(1), len(files), I


def unit_side_lv(u):
    p = u.split("_")            # rib_left_7 / rib_left_1_4
    return p[1], (list(range(1, 5)) if len(p) == 4 else [int(p[2])])


def localize(I, rp, ids):
    """top unit per case vs report. Returns dict of hit rates + random-unit expectation."""
    cols = list(I.columns)
    meta = [unit_side_lv(c) for c in cols]
    sh, sr, lh, lr_ = [], [], [], []
    for cid in ids:
        if cid not in rp.index or cid not in I.index:
            continue
        r = rp.loc[cid]
        v = I.loc[cid].values
        pres = ~np.isnan(v)
        if not pres.any():
            continue
        top = int(np.nanargmax(v))
        side = r.side if isinstance(r.side, str) else "?"
        if side in ("L", "R"):
            ok = [meta[j][0] in SD[side] for j in range(len(cols))]
            sh.append(ok[top]); sr.append(np.mean([ok[j] for j in np.where(pres)[0]]))
        if isinstance(r.rib_levels, str) and r.rib_levels:
            rep = [(s_, int(t[1:])) for t in r.rib_levels.split(";") for s_ in SD.get(t[0], set())]
            def hit(j):
                s, lvs = meta[j]
                return any(s == s2 and min(abs(l - l2) for l in lvs) <= 1 for s2, l2 in rep)
            lh.append(hit(top)); lr_.append(np.mean([hit(j) for j in np.where(pres)[0]]))
    return {"側別命中": np.mean(sh) if sh else np.nan, "側別(隨機)": np.mean(sr) if sr else np.nan, "n側別": len(sh),
            "節次±1命中": np.mean(lh) if lh else np.nan, "節次±1(隨機)": np.mean(lr_) if lr_ else np.nan, "n節次": len(lh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--common", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--title", default="實驗 L：1.5 mm 肋骨")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spl = split_trval()
    lab = load_labels_trval(set(spl.index))
    lab = lab[~lab["y"].isna()]
    age = pd.read_csv(R.parent / "merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")["Age"]
    df = lab.join(spl.rename("split"), how="inner").join(age.rename("age"))
    rp = pd.read_csv(MW / "outputs_fracture/analysis_rib/rib_positives_parsed.csv").set_index("id")
    names, insts = [], {}
    for kind, specs in (("ref", a.ref), ("model", a.model)):
        for sp in specs:
            n, p = sp.split("=", 1)
            s, k, I = load_any(p)
            df[n] = s
            names.append((n, kind, k))
            if I is not None:
                insts[n] = I
    cols = [n for n, _, _ in names]
    if a.ids_file:
        keep = set(pd.read_csv(a.ids_file).query("tier == 0").study_id) if a.ids_file.endswith(".csv") else set(open(a.ids_file).read().split())
        df = df[df.index.isin(keep)]
    if a.common:
        df = df.dropna(subset=cols)
    tr = df[df.split == "train"]
    va = df[df.split == "val"].dropna(subset=cols + ["age"])
    y = va.y.values.astype(int); agev = va.age.values.astype(float); neg = y == 0
    sites = va.sites.apply(lambda s: list(s) if s is not None else [])
    rib = (y == 1) & sites.apply(lambda s: "rib" in s).values
    groups = {
        "任何骨折": y == 1,
        "肋骨": rib,
        "只有肋骨": rib & sites.apply(lambda s: set(s) == {"rib"}).values,
        "肋骨急性(含混合)": rib & va.acute.isin(["yes", "mixed"]).values,
        "肋骨陳舊": rib & (va.acute == "no").values,
        "椎體壓迫性": (y == 1) & va.compression.values.astype(bool),
        "椎體(部位)": (y == 1) & sites.apply(lambda s: "vertebra" in s).values,
        "骨盆/薦椎/髖": (y == 1) & sites.apply(lambda s: any(x in s for x in ("pelvis", "sacrum", "hip_femur"))).values,
        "急性(含混合)": (y == 1) & va.acute.isin(["yes", "mixed"]).values,
        "陳舊": (y == 1) & (va.acute == "no").values,
    }
    refs = [n for n, k, _ in names if k == "ref"]
    rows, drows, lrows = [], [], []
    trrib = tr[(tr.y == 1) & tr.sites.apply(lambda s: "rib" in (s if s is not None else []))].index
    for n, kind, k in names:
        p = va[n].values
        trm = tr[n].notna() & tr.y.notna()
        r = {"模型": n, "檔數": k, "train OOF": auroc(tr.y[trm].values, tr[n][trm].values) if trm.sum() > 100 else np.nan}
        trneg = tr[(tr.y == 0) & tr[n].notna()]
        trr = tr.loc[trrib][n].dropna()
        r["train OOF 肋骨"] = auroc(np.r_[np.ones(len(trr)), np.zeros(len(trneg))], np.r_[trr.values, trneg[n].values]) if len(trr) > 20 else np.nan
        for g, m in groups.items():
            sel = m | neg
            r[g] = auroc(m[sel].astype(float), p[sel])
        r["年齡配對(任何)"] = matched_auc(y, p, agev, 2.0)
        for g in ("肋骨", "只有肋骨"):
            sel = groups[g] | neg
            r[f"年齡配對({g})"] = matched_auc(groups[g][sel].astype(int), p[sel], agev[sel], 2.0)
        r["Spearman(分數,年齡|陰性)"] = spearmanr(p[neg], agev[neg])[0]
        rows.append(r)
        for rf in refs:
            if rf == n:
                continue
            d = {"模型": n, "vs": rf}
            for g in ("任何骨折", "肋骨", "只有肋骨", "椎體壓迫性", "骨盆/薦椎/髖"):
                sel = groups[g] | neg
                b = boot_diff(groups[g][sel].astype(int), p[sel], va[rf].values[sel], agev[sel], 2.0, a.n_boot)
                d[f"Δ{g}"] = f"{b['auc'][0]:+.3f} [{b['auc'][1]:+.3f}, {b['auc'][2]:+.3f}]"
                if g in ("任何骨折", "肋骨"):
                    d[f"Δ年齡配對({g})"] = f"{b['matched'][0]:+.3f} [{b['matched'][1]:+.3f}, {b['matched'][2]:+.3f}]"
            drows.append(d)
        if n in insts:
            for part, idl in (("val", va.index[rib]), ("train OOF", trrib)):
                lrows.append({"模型": n, "資料": part, **localize(insts[n], rp, list(idl))})
    res, dres, lres = pd.DataFrame(rows), pd.DataFrame(drows), pd.DataFrame(lrows)
    res.to_csv(out / "lreport.csv", index=False); dres.to_csv(out / "lreport_diff.csv", index=False)
    lres.to_csv(out / "lreport_loc.csv", index=False)
    hdr = (f"# {a.title}（val N={len(va)}：骨折 {int(y.sum())}、陰性 {int(neg.sum())}；"
           + "、".join(f"{g} {int(m.sum())}" for g, m in groups.items() if g != "任何骨折") + f"；train N={len(tr)}）\n\n")
    txt = (hdr + "## AUROC（各群陽性 vs 全部陰性）\n\n" + to_md(res) + "\n\n## 配對 bootstrap ΔAUROC（95% CI）\n\n"
           + to_md(dres) + "\n\n## 定位（最高分肋骨單元 vs 報告；側別只算單側病例；隨機 = 在存在單元中隨機挑）\n\n"
           + (to_md(lres) if len(lres) else "（無逐根 instance 分數）") + "\n")
    (out / "lreport.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
