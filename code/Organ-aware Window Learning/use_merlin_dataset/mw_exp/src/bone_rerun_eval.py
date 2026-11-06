"""全骨骼重跑報告：probe.py 視窗實驗 / trap / INLP / window ablation 在骨骼單元 (bone_probe.py --stage modes) vs 舊腰椎單元。

VAL only. Labels are read with bone_probe.load_labels (a label line is parsed only after its study_id matched a val id;
test lines are never parsed). Old probe npz files hold predictions for every split; only their val rows are indexed.
Per experiment: val AUROC, paired bootstrap ΔAUROC (+Δ age-matched) vs bone E0 (same std setting) and vs the old
lumbar-only counterpart (on the common val ids), age-matched AUROC (±2 y), Spearman(score, age | negatives),
per-site AUROC (site positives vs all negatives). Also: age-only baseline, current best (Exp F bone_ens) reference,
leave-one-window-out tables.
usage: python bone_rerun_eval.py [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from age_shortcut import matched_auc  # noqa: E402
from bone_probe import LABELS, load_labels, trval_ids  # noqa: E402
from fracture_eval_oof import boot_diff  # noqa: E402
from probe import auroc  # noqa: E402

R = Path.home() / "Desktop/Tony/VLM/OWL"
MW = R / "mw_exp"
RR = MW / "outputs_fracture/bones/rerun"
FIN, INT = MW / "outputs_final/results", MW / "outputs_interim/results"
OLD = {  # (new exp, std setting) -> old lumbar_spine probe npz (14 findings, fracture column)
    ("E0", "std"): FIN / "std/E0", ("E1_soft", "std"): FIN / "std/E1_soft_tissue", ("E1_bone", "std"): FIN / "std/E1_bone",
    ("E2", "std"): FIN / "std/E2", ("E3", "std"): FIN / "std/E3", ("E3_late", "std"): FIN / "std/E3_late",
    ("E4", "std"): FIN / "std/E4", ("E0_rf", "std"): FIN / "std/E0_rf",
    ("E0_trap", "std"): FIN / "std/E0_trap", ("E3_late_trap", "std"): FIN / "std/E3_late_trap",
    ("E0_trap2", "std"): FIN / "std/E0_trap2", ("E3_late_trap2", "std"): FIN / "std/E3_late_trap2",
    ("E0_inlp", "std"): INT / "inlp/E0", ("E3_late_inlp", "std"): INT / "inlp/E3_late",
    ("E0_trap2_inlp", "std"): INT / "inlp/E0_trap2", ("E3_late_trap2_inlp", "std"): INT / "inlp/E3_late_trap2",
    ("E0", "nostd"): INT / "E0", ("E1_soft", "nostd"): INT / "E1_soft_tissue", ("E1_bone", "nostd"): INT / "E1_bone",
    ("E2", "nostd"): INT / "E2", ("E3", "nostd"): INT / "E3", ("E4", "nostd"): INT / "E4",
}
ORDER = ["E0", "E1_soft", "E1_bone", "E2", "E3", "E3_late", "E4", "E0_rf",
         "E0_trap", "E0_trap2", "E2_trap", "E2_trap2", "E3_late_trap", "E3_late_trap2",
         "E0_inlp", "E2_inlp", "E3_late_inlp", "E0_trap2_inlp", "E2_trap2_inlp", "E3_late_trap2_inlp"]
SITES = ["椎體壓迫性", "肋骨", "骨盆/薦椎/髖", "急性(含混合)", "陳舊"]


def old_val(path):
    z = np.load(f"{path}.npz", allow_pickle=True)
    vm = z["split"] == "val"                                   # only val rows are indexed
    p = z["prob"][:, vm, list(z["findings"]).index("fracture")].mean(0)
    return pd.Series(p, index=np.array(z["ids"])[vm])


def new_val(sub, e):
    f = RR / sub / f"{e if e != 'E4' else 'E1_bone'}.npz"     # E4 preset == E1_bone (all bone units -> bone window)
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    return pd.Series(z["logit"], index=list(z["ids"]))


def ci(b, k="auc"):
    return f"{b[k][0]:+.3f} [{b[k][1]:+.3f}, {b[k][2]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    spl = trval_ids()
    vids = sorted(spl.index[spl == "val"])
    lab = load_labels(LABELS, vids)
    age = pd.read_csv(R.parent / "merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")["Age"]
    base = new_val("std", "E0")
    df = lab.loc[lab.index.isin(base.index)].join(age.rename("age"))
    df = df[~df["y"].isna() & ~df["age"].isna()]
    y = df["y"].values.astype(int)
    grp = {"椎體壓迫性": df["compression"].values.astype(bool),
           "肋骨": df["sites"].apply(lambda s: "rib" in s).values,
           "骨盆/薦椎/髖": df["sites"].apply(lambda s: any(x in s for x in ("pelvis", "sacrum", "hip_femur"))).values,
           "急性(含混合)": df["acute"].isin(["yes", "mixed"]).values, "陳舊": (df["acute"] == "no").values}
    grp = {k: (y == 1) & v for k, v in grp.items()}

    def metrics(p, idx=None):
        m = np.ones(len(df), bool) if idx is None else idx
        yy, pp, ag = y[m], np.asarray(p, float)[m], df["age"].values[m]
        r = {"n": int(m.sum()), "AUROC": auroc(yy.astype(float), pp), "年齡配對": matched_auc(yy, pp, ag, 2.0),
             "Spearman": spearmanr(pp[yy == 0], ag[yy == 0])[0]}
        for g, gm in grp.items():
            sel = (gm[m] | (yy == 0))
            r[g] = auroc(gm[m][sel].astype(float), pp[sel])
        return r

    agev = df["age"].values
    ids = df.index
    rows, ovn = [], []
    bones_e0 = {s: new_val(s, "E0") for s in ("std", "nostd")}
    for sub in ("std", "nostd"):
        b0 = bones_e0[sub]
        if b0 is None:
            continue
        b0 = b0.reindex(ids).values
        for e in ORDER:
            s = new_val(sub, e)
            if s is None:
                continue
            p = s.reindex(ids).values
            assert not np.isnan(p).any(), (sub, e)
            r = {"實驗": e, "設定": sub, **metrics(p)}
            js = RR / sub / f"{e if e != 'E4' else 'E1_bone'}.json"
            info = json.loads(js.read_text())
            r["HP(val 選)"] = f"lr={info['hp']['lr']:g} wd={info['hp']['wd']:g}"
            if info.get("inlp_r2"):
                r["INLP val R²(年齡)"] = f"{info['inlp_r2'][0]:.3f}→{info['inlp_r2'][1]:.3f}"
            if not (e == "E0"):
                bd = boot_diff(y, p, b0, agev, 2.0, a.n_boot)
                r["Δ vs 骨骼E0"], r["Δ年齡配對 vs 骨骼E0"] = ci(bd), ci(bd, "matched")
            if (e, sub) in OLD and Path(f"{OLD[(e, sub)]}.npz").exists():
                o = old_val(OLD[(e, sub)]).reindex(ids)
                m = ~o.isna().values
                po = np.log(o.clip(1e-6, 1 - 1e-6) / (1 - o.clip(1e-6, 1 - 1e-6))).values
                mo, mn = metrics(po, m), metrics(p, m)
                bo = boot_diff(y[m], p[m], po[m], agev[m], 2.0, a.n_boot)
                ovn.append({"實驗": e, "設定": sub, "共同 val N": int(m.sum()),
                            "舊 腰椎 AUROC": mo["AUROC"], "新 全骨骼 AUROC": mn["AUROC"], "Δ新−舊 [95% CI]": ci(bo),
                            "舊 年齡配對": mo["年齡配對"], "新 年齡配對": mn["年齡配對"], "Δ年齡配對": ci(bo, "matched"),
                            "舊 Spearman": mo["Spearman"], "新 Spearman": mn["Spearman"],
                            **{f"舊 {g}": mo[g] for g in SITES}, **{f"新 {g}": mn[g] for g in SITES},
                            "舊來源": str(OLD[(e, sub)].relative_to(MW))})
            rows.append(r)
    # references
    ref = []
    ref.append({"實驗": "僅年齡（age-only）", "設定": "-", **metrics(agev)})
    z = np.load(MW / "outputs_fracture/bones/final_F1/groups/bone_ens.npz", allow_pickle=True)
    assert set(np.unique(z["split"])) <= {"train", "val"}
    ens = pd.Series(z["logit"], index=list(z["ids"])).reindex(ids).values
    ref.append({"實驗": "目前最佳：骨骼 10-seed 集成 (final_F1，train OOF 選)", "設定": "std", **metrics(ens)})
    for e in ("E0", "E3_late"):
        o = old_val(FIN / f"expB/{e}").reindex(ids)
        po = np.log(o.clip(1e-6, 1 - 1e-6) / (1 - o.clip(1e-6, 1 - 1e-6))).values
        ref.append({"實驗": f"probe.py Exp B {e}（17 骨骼單元，5 視窗含替代）", "設定": "std", **metrics(po)})
    best_cmp = []
    for r in rows:
        if r["設定"] == "std":
            p = new_val("std", r["實驗"]).reindex(ids).values
            bd = boot_diff(y, p, ens, agev, 2.0, a.n_boot)
            best_cmp.append({"實驗": r["實驗"], "AUROC": r["AUROC"], "Δ vs 最佳 .826": ci(bd), "Δ年齡配對 vs 最佳": ci(bd, "matched")})

    # window ablation
    wab = []
    for basee in ("E3_late", "E2"):
        full = new_val("std", basee)
        if full is None:
            continue
        pf = full.reindex(ids).values
        for wn in ("baseline", "soft", "bone"):
            s = new_val("std", f"{basee}_without_{wn}")
            if s is None:
                continue
            p = s.reindex(ids).values
            bd = boot_diff(y, pf, p, agev, 2.0, a.n_boot)
            mm = metrics(p)
            wab.append({"模型": basee, "拿掉視窗": wn, "AUROC(拿掉後)": mm["AUROC"],
                        "下降 = 完整−拿掉 [95% CI]": ci(bd), "年齡配對下降": ci(bd, "matched"),
                        **{g: mm[g] for g in SITES}})

    out = RR / "report"
    out.mkdir(exist_ok=True)
    def md(d):
        d = pd.DataFrame(d)
        f = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
        L = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
        return "\n".join(L + ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)])
    hdr = (f"# 全骨骼重跑：先前實驗對照（val N={len(df)}，骨折 {int(y.sum())}，陰性 {int((y == 0).sum())}）\n\n"
           "群組人數：" + "，".join(f"{g} {int(m.sum())}" for g, m in grp.items()) + "\n\n")
    txt = (hdr + "## 新（全骨骼 site MIL）\n\n" + md(rows) + "\n\n## 舊（腰椎）vs 新（全骨骼）\n\n" + md(ovn)
           + "\n\n## 參考\n\n" + md(ref) + "\n\n## 對目前最佳 .826 的配對差\n\n" + md(best_cmp)
           + "\n\n## Window ablation（leave-one-window-out，同 HP，3 seeds）\n\n" + md(wab) + "\n")
    (out / "rerun_report.md").write_text(txt)
    for n, d in (("new", rows), ("old_vs_new", ovn), ("ref", ref), ("vs_best", best_cmp), ("wablate", wab)):
        pd.DataFrame(d).to_csv(out / f"{n}.csv", index=False)
    print(txt)


if __name__ == "__main__":
    main()
