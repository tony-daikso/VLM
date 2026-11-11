#!/usr/bin/env python3
"""Agent G: how far can the proxy evaluation (proxy_eval.py, mt_lib rules) be trusted?  Compares, for the same
generation file, the proxy per_case_categories / metrics with the official Qwen evaluation.
Aggregate numbers only (no ids, no text).

  * cell agreement (case x finding, generated side): 6-way category, strict positive (yes|probably_yes vs rest),
    3-way (positive / explicit negative / other incl. not_mentioned, unclear); Cohen kappa of strict positive
  * per group: macro sens / spec / F1 (owl, strict), critical-error and omission rate, Qwen vs proxy, and the rank
    order of the groups (Kendall tau, pairwise order agreement)
  * per finding: positive-rate bias (proxy - Qwen), F1 difference, strict-positive agreement, pooled over groups
usage: python proxy_calib.py --pair NAME=<qwen_dir>,<proxy_dir> [...] [--md]
"""
import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize import summarize  # noqa: E402

ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊切除術後", "splenomegaly": "脾腫大",
      "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤", "atherosclerosis": "動脈粥狀硬化",
      "osteopenia": "骨質減少", "bowel_obstruction": "腸阻塞", "appendicitis": "闌尾炎", "ascites": "腹水",
      "free_air": "腹腔游離氣體", "gallstones": "膽結石", "biliary_ductal_dilation": "膽管擴張", "fracture": "骨折"}
D = list(ZH)
POS = {"yes", "probably_yes"}
NEG = {"no", "probably_no"}


def tri(c):
    return "P" if c in POS else ("N" if c in NEG else "O")


def load_pc(p):
    return {r["case_id"]: r for r in map(json.loads, open(Path(p) / "per_case_categories.jsonl"))}


def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def kendall(x, y):
    c = d = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        s = (x[i] - x[j]) * (y[i] - y[j])
        c += s > 0
        d += s < 0
    return (c - d) / (c + d) if c + d else float("nan"), c, c + d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    groups, per_d = [], defaultdict(lambda: defaultdict(list))
    cell = defaultdict(lambda: [0, 0])
    for spec in a.pair:
        name, dirs = spec.split("=", 1)
        qd, pd = dirs.split(",")
        Q, P = load_pc(qd), load_pc(pd)
        ids = sorted(set(Q) & set(P))
        sq = summarize(json.load(open(Path(qd) / "metrics.json")))
        sp = summarize(json.load(open(Path(pd) / "metrics.json")))
        g = {"name": name, "n": len(ids)}
        a6 = ab = a3 = 0
        qa, pa = [], []
        for c in ids:
            for d in D:
                q, p = Q[c][d][1], P[c][d][1]
                a6 += q == p
                ab += (q in POS) == (p in POS)
                a3 += tri(q) == tri(p)
                qa.append(q in POS)
                pa.append(p in POS)
                r = Q[c][d][0]
                yref = 1 if r in POS else (0 if r in NEG or r == "not_mentioned" else None)
                pd_ = per_d[d]
                pd_["agree_bin"].append((q in POS) == (p in POS))
                pd_["agree3"].append(tri(q) == tri(p))
                pd_["q_pos"].append(q in POS)
                pd_["p_pos"].append(p in POS)
                pd_["q_neg"].append(q in NEG)
                pd_["p_neg"].append(p in NEG)
                pd_["q_nm"].append(q == "not_mentioned")
                pd_["p_nm"].append(p == "not_mentioned")
                cell[(tri(q), tri(p))][0] += 1
                if yref == 1:
                    pd_["q_crit"].append(q in NEG)
                    pd_["p_crit"].append(p in NEG)
        nc = len(ids) * len(D)
        g.update(agree6=a6 / nc, agree_bin=ab / nc, agree3=a3 / nc, kappa=kappa(qa, pa))
        for k in ("sens", "spec", "f1", "mf1"):
            g[f"q_{k}"], g[f"p_{k}"] = sq[k], sp[k]
        for k in ("crit", "omit"):
            g[f"q_{k}"], g[f"p_{k}"] = sq[k][0], sp[k][0]
        for k in ("attr:hydronephrosis:side", "attr:fracture:levels", "attr:fracture:sites", "attr:bowel_obstruction:type"):
            g[f"q_{k}"], g[f"p_{k}"] = (sq.get(k) or (None,))[0], (sp.get(k) or (None,))[0]
        g["q_per"], g["p_per"] = sq["_per"], sp["_per"]
        groups.append(g)

    L = []
    L += ["### 逐格一致率（生成端，病例 × 14 類）", "",
          "| 組別 | 例數 | 6 類完全一致 | 嚴格陽性一致 | kappa（嚴格陽性） | 3 類一致（陽／明確陰／其他） |", "|---|---|---|---|---|---|"]
    for g in groups:
        L.append(f"| {g['name']} | {g['n']} | {g['agree6']:.3f} | {g['agree_bin']:.3f} | {g['kappa']:.3f} | {g['agree3']:.3f} |")
    L += ["", "### 各組彙總指標：Qwen vs 代理（owl 定義、嚴格）", "",
          "| 組別 | macro F1 Qwen | 代理 | 差 | macro 敏感度 Qwen／代理 | macro 特異度 Qwen／代理 | 關鍵錯誤率 Qwen／代理 | 漏寫率 Qwen／代理 | 腎水腫側別 Qwen／代理 | 骨折節段 Qwen／代理 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    for g in groups:
        L.append(f"| {g['name']} | {g['q_f1']:.3f} | {g['p_f1']:.3f} | {g['p_f1'] - g['q_f1']:+.3f} | {g['q_sens']:.3f}／{g['p_sens']:.3f} | "
                 f"{g['q_spec']:.3f}／{g['p_spec']:.3f} | {g['q_crit']:.3f}／{g['p_crit']:.3f} | {g['q_omit']:.3f}／{g['p_omit']:.3f} | "
                 f"{f(g['q_attr:hydronephrosis:side'])}／{f(g['p_attr:hydronephrosis:side'])} | {f(g['q_attr:fracture:levels'])}／{f(g['p_attr:fracture:levels'])} |")
    L.append("")
    for k, lab in (("f1", "macro F1"), ("sens", "macro 敏感度"), ("crit", "關鍵錯誤率"), ("mf1", "micro F1")):
        t, c, n = kendall([g[f"q_{k}"] for g in groups], [g[f"p_{k}"] for g in groups])
        qo = " > ".join(x["name"] for x in sorted(groups, key=lambda g: -g[f"q_{k}"]))
        po = " > ".join(x["name"] for x in sorted(groups, key=lambda g: -g[f"p_{k}"]))
        L.append(f"- {lab}：Kendall tau {t:.2f}（兩兩順序一致 {c}/{n}）；Qwen 排序 {qo}；代理排序 {po}")
    # pairwise differences: does the proxy reproduce Qwen's group-vs-group F1 differences?
    diffs = [(g1["name"], g2["name"], g1["q_f1"] - g2["q_f1"], g1["p_f1"] - g2["p_f1"]) for g1, g2 in itertools.combinations(groups, 2)]
    if diffs:
        import statistics
        err = [abs(p - q) for _, _, q, p in diffs]
        sign = sum((q > 0) == (p > 0) for _, _, q, p in diffs if abs(q) >= 0.01)
        big = sum(1 for _, _, q, p in diffs if abs(q) >= 0.01)
        L.append(f"- 組間 macro F1 差（{len(diffs)} 對）：代理差與 Qwen 差的絕對誤差 中位數 {statistics.median(err):.3f}、最大 {max(err):.3f}；"
                 f"Qwen 差 ≥ 0.01 的 {big} 對中方向一致 {sign} 對")
    L += ["", "### 各類偏差（所有組別合併；陽性率 = 生成被判為陽性的比例）", "",
          "| 疾病 | 嚴格陽性一致 | 3 類一致 | 陽性率 Qwen | 陽性率 代理 | 偏差 | 明確陰性率 Qwen／代理 | 沒提到 Qwen／代理 | 關鍵錯誤（原陽性中）Qwen／代理 | F1 差（代理−Qwen，各組平均） |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    mean = lambda x: sum(x) / len(x) if x else float("nan")  # noqa: E731
    for d in D:
        pd_ = per_d[d]
        fd = mean([g["p_per"][ZH[d]][2] - g["q_per"][ZH[d]][2] for g in groups
                   if g["p_per"][ZH[d]][2] == g["p_per"][ZH[d]][2] and g["q_per"][ZH[d]][2] == g["q_per"][ZH[d]][2]])
        L.append(f"| {ZH[d]} | {mean(pd_['agree_bin']):.3f} | {mean(pd_['agree3']):.3f} | {mean(pd_['q_pos']):.3f} | {mean(pd_['p_pos']):.3f} | "
                 f"{mean(pd_['p_pos']) - mean(pd_['q_pos']):+.3f} | {mean(pd_['q_neg']):.3f}／{mean(pd_['p_neg']):.3f} | "
                 f"{mean(pd_['q_nm']):.3f}／{mean(pd_['p_nm']):.3f} | {mean(pd_['q_crit']):.3f}／{mean(pd_['p_crit']):.3f} | {fd:+.3f} |")
    tot = sum(v[0] for v in cell.values())
    L += ["", "### 3 類混淆（列 = Qwen，欄 = 代理；所有組別合併的格數比例）", "", "| Qwen＼代理 | 陽 | 明確陰 | 其他 |", "|---|---|---|---|"]
    for q in "PNO":
        L.append(f"| {dict(P='陽', N='明確陰', O='其他')[q]} | " + " | ".join(f"{cell[(q, p)][0] / tot:.3f}" for p in "PNO") + " |")
    txt = "\n".join(L)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
        json.dump([{k: v for k, v in g.items() if not k.endswith("_per")} for g in groups],
                  open(Path(a.out).with_suffix(".json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
