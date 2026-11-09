"""Site-wise fracture scoring (依部位給分). Train/val only; test never read. Selection = train 5-fold CV only.

Site scores (all inputs are member OOF logits: train = member 5-fold OOF, val = fold-model mean):
  vertebra : J_add (10 seed) [+ site heads: L v15/v3 site logits, L-Jadd v15/v3 site logits, expI vertebra site logit]
  rib      : L rib15 site logit [+ L per-rib instance max, L-Jadd rib3 site logit, expI rib site logit]
  pelvis   : U_pel [+ U_org_pel, U_org_out, L pelvis site, L-Jadd pelvis site, expI pelvis site]
Site head = LR (unweighted -> calibrated probability) on the site's inputs, fit on the SITE label
  (site positives = 1, fracture-free = 0, fractures elsewhere excluded). Overall combiners:
  a max(p_s)  b noisy-OR  c logsumexp(site logits)  (+any: add calibrated LR(6 C3 members) as a 4th component)
  d LR(site logits + C3 stack) / d0 LR(site logits) on the overall label.
Nesting: outer StratifiedKFold(5, rs=0) on train (same folds as the C3 stacker); every level-1 model (site heads,
C3 stack) is refit inside each outer-train; level-2 LR (d) is fit on inner 5-fold OOF level-1 scores of the outer-train.
Val: level-1 fit on all train; level-2 fit on outer-CV OOF level-1 scores of train.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"
sys.path.insert(0, str(MW / "src"))
from age_shortcut import matched_auc  # noqa: E402
from fracture_eval_oof import boot_diff  # noqa: E402
from probe import auroc  # noqa: E402

R = MW.parent
FR = MW / "outputs_fracture"
OUT = FR / "sitewise"
POS, NEG = {"yes", "probably_yes"}, {"no", "probably_no", "not_mentioned"}
NBOOT = 2000
SUB = {
    "椎體壓迫性": lambda d: d["compression"],
    "肋骨": lambda d: d["sites"].apply(lambda s: "rib" in s),
    "只有肋骨": lambda d: d["sites"].apply(lambda s: len(s) > 0 and all("rib" in x for x in s)),
    "骨盆/薦椎/髖": lambda d: d["sites"].apply(lambda s: any(x in s for x in ("pelvis", "sacrum", "hip_femur"))),
    "急性(含混合)": lambda d: d["acute"].isin(["yes", "mixed"]),
    "陳舊": lambda d: d["acute"] == "no",
}
RIGHT = {"椎體壓迫性": "vert", "肋骨": "rib", "只有肋骨": "rib", "骨盆/薦椎/髖": "pel"}
SNAME = {"vert": "椎體分數", "rib": "肋骨分數", "pel": "骨盆分數"}


def split_trval():
    spl = pd.read_csv(R / "data/split.csv").set_index("study_id")["split"]
    return spl[spl.isin(["train", "val"])]                       # test ids dropped first


def load_labels_trval(keep):
    rx, rows = re.compile(r'"study_id"\s*:\s*"([^"]+)"'), {}
    for line in open(R / "data/labels_disease14_all.jsonl"):
        m = rx.search(line)
        if not m or m.group(1) not in keep:                     # only train/val lines are parsed
            continue
        r = json.loads(line)
        a = r["answer"]["fracture"]
        d = a.get("detail", {}) or {}
        cat = a["category"]
        rows[r["study_id"]] = {"y": 1.0 if cat in POS else 0.0 if cat in NEG else np.nan,
                               "compression": d.get("vertebral_compression") == "yes",
                               "sites": d.get("sites", []) or [], "acute": d.get("acute", "not_stated")}
    return pd.DataFrame.from_dict(rows, orient="index")


def npz_series(files, key="logit", col=None, fn=None):
    S = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        assert set(np.unique(z["split"])) <= {"train", "val"}, f
        v = np.asarray(z[key], float)
        if col is not None:
            v = v[:, col]
        if fn is not None:
            v = fn(v)
        S.append(pd.Series(v, index=[str(i) for i in z["ids"]]))
    return pd.concat(S, axis=1).mean(1)


def zs(s, trmask):
    return (s - s[trmask].mean()) / s[trmask].std()


def build_frame():
    spl = split_trval()
    lab = load_labels_trval(set(spl.index))
    lab = lab[~lab["y"].isna()]
    age = pd.read_csv(R.parent / "merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")["Age"]
    df = lab.join(spl.rename("split"), how="inner").join(age.rename("age"))
    y = df["y"]
    def site_t(f):
        s = df.apply(f, axis=1)
        return np.where(y == 0, 0.0, np.where((y == 1) & s, 1.0, np.nan))
    df["t_vert"] = site_t(lambda r: "vertebra" in r["sites"])
    df["t_rib"] = site_t(lambda r: "rib" in r["sites"])
    df["t_pel"] = site_t(lambda r: any(x in r["sites"] for x in ("pelvis", "sacrum", "hip_femur")))
    J10 = sorted(glob.glob(str(MW / "outputs_vert15/final_eval/oof/full/oof_v15_add_s*.npz")))
    Lf = sorted(glob.glob(str(FR / "expL/oof/full/oof_L_L_repl-rl03-lrc_s*.npz")))
    LJ = sorted(glob.glob(str(FR / "expL/oof/full/oof_L_Jadd_s*.npz")))
    M = FR / "final_combo/members"
    assert len(J10) == 10 and len(Lf) == 5 and len(LJ) == 5
    for f in Lf + LJ:
        assert list(np.load(f, allow_pickle=True)["site_grp"]) == ["v", "p", "v", "r"]
    cols = {
        "expI": npz_series([FR / "bones/expI/groups/ens_greedy_all.npz"]),
        "J_add": npz_series(J10), "L": npz_series(Lf),
        "U_org_pel": npz_series([M / "U_org_pel.npz"]), "U_pel": npz_series([M / "U_pel.npz"]),
        "U_org_out": npz_series([M / "U_org_out.npz"]),
        "L_v3": npz_series(Lf, "site_logit", 0), "L_p": npz_series(Lf, "site_logit", 1),
        "L_v15": npz_series(Lf, "site_logit", 2), "L_r": npz_series(Lf, "site_logit", 3),
        "L_rinst": npz_series(Lf, "inst", fn=lambda a: np.nanmax(np.where(np.isnan(a), -np.inf, a), 1)),
        "LJ_v3": npz_series(LJ, "site_logit", 0), "LJ_p": npz_series(LJ, "site_logit", 1),
        "LJ_v15": npz_series(LJ, "site_logit", 2), "LJ_r": npz_series(LJ, "site_logit", 3),
    }
    for n, s in cols.items():
        df = df.join(s.replace([np.inf, -np.inf], np.nan).rename(n))
    # expI site logits: greedy bone groups (E3late10 is not a bone member), seed-mean, train-z, averaged
    sel = [g for g in json.load(open(FR / "bones/expI/groups/ens_greedy_all.json"))["greedy"] if g != "E3late10"]
    trm = df["split"] == "train"
    for k, nm in enumerate(["I_v", "I_r", "I_p"]):
        parts = []
        for g in sel:
            fs = sorted(glob.glob(str(FR / f"bones/expI/members/oof_bone_E_{g}_s*.npz"))); assert fs, g
            s = npz_series(fs, "site_logit", k).reindex(df.index)
            parts.append(zs(s, trm))
        df[nm] = pd.concat(parts, axis=1).mean(1)
    return df, sel


C3M = ["expI", "J_add", "L", "U_org_pel", "U_pel", "U_org_out"]
FEATS = {
    "S1": {"vert": ["J_add"], "rib": ["L_r"], "pel": ["U_pel"]},
    "S2": {"vert": ["J_add", "L_v15", "L_v3", "LJ_v15", "LJ_v3", "I_v"],
           "rib": ["L_r", "L_rinst", "LJ_r", "I_r"],
           "pel": ["U_pel", "U_org_pel", "U_org_out", "L_p", "LJ_p", "I_p"]},
}
SITES = ["vert", "rib", "pel"]


def lr_fit(X, y, balanced):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=3000, class_weight="balanced" if balanced else None).fit(sc.transform(X), y)
    return lambda Z: m.decision_function(sc.transform(Z))


def level1(fs, A, B):
    """fit on frame A, return level-1 logits for frame B: site heads (calibrated), C3 stack (balanced), C3 calibrated."""
    o = {}
    for s in SITES:
        m = ~A["t_" + s].isna()
        o[s] = lr_fit(A.loc[m, FEATS[fs][s]].values, A.loc[m, "t_" + s].values, False)(B[FEATS[fs][s]].values)
    o["c3"] = lr_fit(A[C3M].values, A["y"].values, True)(B[C3M].values)
    o["anyp"] = lr_fit(A[C3M].values, A["y"].values, False)(B[C3M].values)
    return pd.DataFrame(o, index=B.index)


def combine(kind, L1, fit=None):
    S = L1[SITES].values
    if kind.endswith("+any"):
        S = np.c_[S, L1["anyp"].values]
    k = kind.replace("+any", "")
    if k == "max":
        return S.max(1)                                              # max of calibrated logits == max prob
    if k == "noisyor":
        return -np.sum(np.log1p(-expit(S).clip(0, 1 - 1e-12)), 1)    # monotone in 1-prod(1-p)
    if k == "lse":
        return logsumexp(S, 1)
    cols = SITES + (["c3"] if k == "lrc3" else [])
    return fit(L1[cols].values)


def l2_fit(kind, L1, y):
    if kind not in ("lrc3", "lr"):
        return None
    cols = SITES + (["c3"] if kind == "lrc3" else [])
    return lr_fit(L1[cols].values, y, True)


def run_cv(tr, fs, kinds):
    outer = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, tr["y"]))
    oof = {k: np.zeros(len(tr)) for k in kinds}
    L1oof = pd.DataFrame(index=tr.index, columns=SITES + ["c3", "anyp"], dtype=float)
    for a, b in outer:
        A, Bf = tr.iloc[a], tr.iloc[b]
        L1b = level1(fs, A, Bf)
        L1oof.iloc[b] = L1b.values
        need2 = any(k in ("lr", "lrc3") for k in kinds)
        if need2:
            inner = pd.DataFrame(index=A.index, columns=L1b.columns, dtype=float)
            for ia, ib in StratifiedKFold(5, shuffle=True, random_state=1).split(A, A["y"]):
                inner.iloc[ib] = level1(fs, A.iloc[ia], A.iloc[ib]).values
        for k in kinds:
            f2 = l2_fit(k, inner, A["y"].values) if k in ("lr", "lrc3") else None
            oof[k][b] = combine(k, L1b, f2)
    return oof, L1oof


def boot_ci(y, p, n=NBOOT, seed=0):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = p[pos][:, None] - p[neg][None, :]
    K = (d > 0) + 0.5 * (d == 0)
    rng = np.random.default_rng(seed); N = len(y); v = []
    for _ in range(n):
        c = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
        cp, cn = c[pos], c[neg]
        v.append(cp @ K @ cn / (cp.sum() * cn.sum()))
    return auroc(y.astype(float), p), *np.percentile(v, [2.5, 97.5])


def sens_at_spec(y, p, spec=0.9):
    thr = np.quantile(p[y == 0], spec)
    return float((p[y == 1] > thr).mean())


def f3(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def fd(b):
    return f"{b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}]"


def to_md(df):
    f = lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    L = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(L + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, sel = build_frame()
    need = sorted({c for fs in FEATS.values() for v in fs.values() for c in v} | set(C3M))
    tr = df[df.split == "train"].dropna(subset=need).copy()
    va = df[df.split == "val"].dropna(subset=need + ["age"]).copy()
    print("train", len(tr), "val", len(va), "expI greedy bone groups", sel, flush=True)
    # sanity: expI site column order + raw site-input AUROC on site targets (train)
    diag = []
    for c in need:
        diag.append({"輸入": c, **{f"train AUROC({s})": auroc(tr["t_" + s].values, tr[c].values) for s in SITES},
                     "train AUROC(整體)": auroc(tr["y"].values, tr[c].values)})
    diag = pd.DataFrame(diag); print(to_md(diag), flush=True)

    kinds = ["max", "noisyor", "lse", "max+any", "noisyor+any", "lse+any", "lr", "lrc3"]
    res, store = [], {}
    # C3 reference recomputed with the same folds (should reproduce .8585 / .854)
    for fs in FEATS:
        oof, L1oof = run_cv(tr, fs, kinds)
        L1va = level1(fs, tr, va)
        site_cv = {s: auroc(tr["t_" + s].values, L1oof[s].values) for s in SITES}
        for k in kinds:
            f2 = l2_fit(k, L1oof, tr["y"].values)
            pv = combine(k, L1va, f2)
            cvauc = auroc(tr["y"].values, oof[k])
            store[(fs, k)] = (pv, oof[k])
            res.append({"site 輸入": fs, "合併": k, "train CV": cvauc, "val": auroc(va["y"].values, pv),
                        **{f"site CV {s}": site_cv[s] for s in SITES}})
            print(fs, k, round(cvauc, 4), round(res[-1]["val"], 4), flush=True)
        store[(fs, "L1oof")], store[(fs, "L1va")] = L1oof, L1va
    c3cv = auroc(tr["y"].values, store[("S1", "L1oof")]["c3"].values)
    c3v = store[("S1", "L1va")]["c3"].values
    res = pd.DataFrame(res)
    best = res.loc[res["train CV"].idxmax()]
    bfs, bk = best["site 輸入"], best["合併"]
    print("SELECTED", bfs, bk, best["train CV"], "C3 CV", c3cv, flush=True)

    y = va["y"].values.astype(int); agev = va["age"].values.astype(float); neg = y == 0
    groups = {g: (y == 1) & np.asarray(f(va), bool) for g, f in SUB.items()}
    jv = va["J_add"].values
    # main table: every config, overall stats with paired deltas
    rows = []
    cands = [("C3（參考，重算）", "—", c3v, c3cv), ("J add 10 seed（參考）", "—", jv, auroc(tr["y"].values, tr["J_add"].values))]
    cands += [(f"{r['site 輸入']} · {r['合併']}", r["site 輸入"], store[(r["site 輸入"], r["合併"])][0], r["train CV"])
              for _, r in res.iterrows()]
    for name, fs, p, cv in cands:
        r = {"設定": name, "train CV": cv, "val [95% CI]": f3(boot_ci(y, p)), "年齡配對(±2y)": matched_auc(y, p, agev, 2.0),
             "陰性 Spearman(分數,年齡)": spearmanr(p[neg], agev[neg])[0], "90%特異度敏感度": sens_at_spec(y, p)}
        for base, tag in [(c3v, "C3"), (jv, "J add")]:
            if np.array_equal(p, base):
                r[f"Δ vs {tag}"] = r[f"Δ年齡配對 vs {tag}"] = "—"; continue
            b = boot_diff(y, p, base, agev, 2.0, NBOOT)
            r[f"Δ vs {tag}"], r[f"Δ年齡配對 vs {tag}"] = fd(b["auc"]), fd(b["matched"])
        for g, m in groups.items():
            s_ = m | neg
            r[g] = auroc(m[s_].astype(float), p[s_])
        rows.append(r)
    main_t = pd.DataFrame(rows)
    main_t.to_csv(OUT / "report_all.csv", index=False)

    # subgroup CIs for selected, C3, J add (+ paired Δ vs C3)
    sel_p = store[(bfs, bk)][0]
    sub_rows = []
    for name, p in [(f"選出：{bfs} · {bk}", sel_p), ("C3", c3v), ("J add 10 seed", jv)]:
        r = {"模型": name}
        for g, m in groups.items():
            s_ = m | neg
            yy = m[s_].astype(int); pp = p[s_]
            r[g] = f3(boot_ci(yy, pp))
        sub_rows.append(r)
    r = {"模型": "Δ 選出 − C3（配對）"}
    for g, m in groups.items():
        s_ = m | neg
        b = boot_diff(m[s_].astype(int), sel_p[s_], c3v[s_], agev[s_], 2.0, NBOOT)
        r[g] = fd(b["auc"])
    sub_rows.append(r)
    sub_t = pd.DataFrame(sub_rows)

    # diagnostic: every site score (selected config's level-1, and S1/S2) on every subgroup, with CIs
    diag_rows = []
    for fs in FEATS:
        L1va = store[(fs, "L1va")]
        for s in SITES:
            r = {"site 分數": f"{fs} {SNAME[s]}", "val AUROC(整體骨折)": f3(boot_ci(y, L1va[s].values)),
                 "val AUROC(該部位標籤)": auroc(va["t_" + s].values, L1va[s].values)}
            for g, m in groups.items():
                s_ = m | neg
                v = f3(boot_ci(m[s_].astype(int), L1va[s].values[s_]))
                r[g] = ("**" + v + "**") if RIGHT.get(g) == s else v
            diag_rows.append(r)
    diag_t = pd.DataFrame(diag_rows)

    np.savez(OUT / "scores.npz", ids=np.r_[tr.index.values, va.index.values].astype(str),
             split=np.r_[["train"] * len(tr), ["val"] * len(va)], y=np.r_[tr["y"].values, va["y"].values],
             selected=np.r_[store[(bfs, bk)][1], sel_p], c3=np.r_[store[("S1", "L1oof")]["c3"].values, c3v],
             **{f"{fs}_{s}": np.r_[store[(fs, "L1oof")][s].values, store[(fs, "L1va")][s].values]
                for fs in FEATS for s in SITES}, selected_name=f"{bfs}·{bk}")
    newbest = best["train CV"] > c3cv
    (OUT / "config.json").write_text(json.dumps({
        "feats": FEATS, "c3_members": C3M, "kinds": kinds, "selected": [bfs, bk], "selected_train_cv": float(best["train CV"]),
        "c3_train_cv": float(c3cv), "selected_val": float(auroc(y.astype(float), sel_p)), "new_best_by_train_cv": bool(newbest),
        "n_train": len(tr), "n_val": len(va), "expI_site_groups": sel}, indent=1, ensure_ascii=False))

    md = [f"# 骨折依部位給分（site-wise scoring）— 2026-11-06\n",
          "規則：只用 train/val（標籤先依 split.csv 過濾 train/val 再解析）；所有選擇只依 train 5-fold CV（外層 folds 同 C3 stacker；"
          "site head／C3 stack 在每個外層 train 內重 fit，第二層 LR 用外層 train 內的 inner 5-fold OOF 訓練，無洩漏）；val 只報告；test 未讀取。",
          f"val 共同集 N={len(va)}（骨折 {int(y.sum())}、陰性 {int(neg.sum())}）；train N={len(tr)}。"
          "群組人數：" + "，".join(f"{g} {int(m.sum())}" for g, m in groups.items()) + "。",
          f"Bootstrap {NBOOT} 次；Δ 為配對 bootstrap。子群組 AUROC＝該群陽性 vs 全部陰性。\n",
          "## 部位分數定義",
          "- site head＝LR（不加權→機率校正）以「部位標籤」訓練：該部位骨折＝1、無骨折＝0、其他部位骨折排除。部位：椎體＝sites 含 vertebra；"
          "肋骨＝rib；骨盆＝pelvis/sacrum/hip_femur（與既有報告的子群組同源：labels_disease14_all.jsonl 的 fracture.detail.sites／vertebral_compression）。",
          "- S1（單一分數）：椎體＝J_add、肋骨＝L 的 rib15 site logit、骨盆＝U_pel。",
          "- S2（多輸入 site head）：椎體＝J_add＋L v15/v3 site＋L-Jadd v15/v3 site＋expI 椎體 site；肋骨＝L rib site＋L 單肋 instance max＋L-Jadd rib3 site＋expI 肋骨 site；"
          f"骨盆＝U_pel＋U_org_pel＋U_org_out＋L 骨盆 site＋L-Jadd 骨盆 site＋expI 骨盆 site。（expI site＝greedy 骨骼群組 {sel} 的 site logit，train z 後平均）",
          "- 合併：max＝校正後部位機率取最大；noisyor＝1−Π(1−p)；lse＝部位 logit 的 logsumexp；+any＝再加一個以整體標籤校正的 LR(C3 六成員) 作第 4 個分量；"
          "lr＝LR(三個部位 logit)；lrc3＝LR(三個部位 logit＋C3 stack)。\n",
          f"## 結果（依 train CV 選出：**{bfs} · {bk}**，train CV {best['train CV']:.4f} vs C3 {c3cv:.4f}）\n",
          to_md(main_t.drop(columns=list(SUB))), "\n### 各設定子群組（點估計）\n",
          to_md(main_t[["設定"] + list(SUB)]),
          "\n## 子群組 95% CI（選出模型 vs C3 vs J add）\n", to_md(sub_t),
          "\n## 診斷：各部位分數在各子群組（粗體＝對應部位的「正確」分數）\n", to_md(diag_t),
          "\n## 輸入診斷（train，原始 OOF logit 對各部位標籤）\n", to_md(diag)]
    (OUT / "report_auto.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
