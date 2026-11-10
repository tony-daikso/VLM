"""Report-location agreement of OWL E3_late unit attributions (Mac, CPU).

in : work/attr.npz (attr_units.py), disease14_qwen.jsonl, probs_main.csv
out: work/loc_summary.csv (rates), work/loc_cases_<finding>.csv (per case), work/unit_rank_long.csv
"""
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data")
W = Path(__file__).resolve().parent / "work"
VERT = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
LEVEL = {**{f"T{i}": i for i in range(1, 13)}, **{f"L{i}": 12 + i for i in range(1, 6)}}
UNIT_CAT = {**{v: "vertebra" for v in VERT}, "ribs_left": "rib", "ribs_right": "rib", "sacrum": "pelvis",
            "hip_left": "pelvis", "hip_right": "pelvis", "femur_left": "hip_femur", "femur_right": "hip_femur"}
# report site -> unit categories that count as a hit
SITE_OK = {"vertebra": {"vertebra"}, "rib": {"rib"}, "pelvis": {"pelvis"}, "sacrum": {"pelvis"},
           "hip_femur": {"hip_femur", "pelvis"}}
HW = re.compile(r"vertebroplasty|kyphoplasty|screw|fixation|s/p|nail|plate|prosthe|arthroplasty|cement", re.I)


def parse_levels(text):
    """'T11-L1', 'L1-3', 'T11/12', 'L2, L3' -> set of level indices (T1=1..L5=17)."""
    out = set()
    pat = re.compile(r"\b([TL])\s?(\d{1,2})(?:\s*(?:-|~|/|to)\s*([TL])?\s?(\d{1,2}))?")
    for m in pat.finditer(text):
        a = f"{m.group(1)}{int(m.group(2))}"
        if a not in LEVEL:
            continue
        if m.group(4):
            b = f"{m.group(3) or m.group(1)}{int(m.group(4))}"
            if b in LEVEL and LEVEL[b] >= LEVEL[a] and LEVEL[b] - LEVEL[a] <= 8:
                out |= set(range(LEVEL[a], LEVEL[b] + 1))
                continue
        out.add(LEVEL[a])
    return out


def side_of(text):
    t = text.lower()
    l, r = bool(re.search(r"\bleft\b|\blt\b", t)), bool(re.search(r"\bright\b|\brt\b", t))
    if re.search(r"bilateral|both", t):
        return "both"
    return "left" if l and not r else "right" if r and not l else ("both" if l and r else None)


def p_hit_random(n, m, k):
    """P(at least one of m good units among k drawn without replacement from n)."""
    if n <= 0 or m <= 0:
        return 0.0
    k = min(k, n)
    return 1 - math.comb(n - m, k) / math.comb(n, k) if n - m >= k else 1.0


def load():
    z = np.load(W / "attr.npz")
    A = {k: z[k] for k in z.files}
    lab = {}
    for l in open(ROOT / "disease14_qwen.jsonl"):
        x = json.loads(l)
        if x.get("parse_ok"):
            lab[x["study_id"]] = x["answer"]
    return A, lab


def main():
    A, lab = load()
    ids, units, F = list(A["ids"]), list(A["units"]), list(A["findings"])
    fi = {f: i for i, f in enumerate(F)}
    ui = {u: i for i, u in enumerate(units)}
    rows = []

    # ---------- fracture ----------
    f = fi["fracture"]
    recs = []
    for n, cid in enumerate(ids):
        a = lab[cid]["fracture"]
        if a["category"] != "yes":
            continue
        quotes = " || ".join(a["quotes"])
        sites = set(a["detail"].get("sites") or [])
        ok_cats = set().union(*[SITE_OK[s] for s in sites if s in SITE_OK]) if sites else set()
        L, Ld = A["mil_L"][n], A["mil_data"][n]
        pres = [u for u in UNIT_CAT if not np.isnan(L[ui[u]])]
        if not pres:
            continue
        order = sorted(pres, key=lambda u: -L[ui[u]])
        order_d = sorted(pres, key=lambda u: -Ld[ui[u]])
        good = [u for u in pres if UNIT_CAT[u] in ok_cats]
        levels = parse_levels(quotes) if "vertebra" in sites else set()
        vpres = [u for u in VERT if u in pres]
        vord = sorted(vpres, key=lambda u: -L[ui[u]])
        vgood = [u for u in vpres if any(abs(LEVEL[u] - l) <= 1 for l in levels)]
        r = dict(id=cid, prob=float(A["prob"][n, f]), sites="|".join(sorted(sites)), quote=quotes,
                 hardware=bool(HW.search(quotes)), n_units=len(pres), n_good=len(good),
                 top1=order[0], top2=order[1] if len(order) > 1 else "", top3=order[2] if len(order) > 2 else "",
                 top1_L=float(L[ui[order[0]]]), top1_data=order_d[0],
                 cat_top1=UNIT_CAT[order[0]] in ok_cats, cat_top3=any(UNIT_CAT[u] in ok_cats for u in order[:3]),
                 cat_top1_data=UNIT_CAT[order_d[0]] in ok_cats,
                 rand_top1=p_hit_random(len(pres), len(good), 1), rand_top3=p_hit_random(len(pres), len(good), 3),
                 levels=",".join(k for k, v in LEVEL.items() if v in levels),
                 lvl_eval=bool(vgood) and len(vpres) >= 3,
                 lvl_top1=vord[0] in vgood if vgood else np.nan,
                 lvl_top3=any(u in vgood for u in vord[:3]) if vgood else np.nan,
                 lvl_rand1=p_hit_random(len(vpres), len(vgood), 1), lvl_rand3=p_hit_random(len(vpres), len(vgood), 3),
                 vert_top=vord[0] if vord else "",
                 rib_L=float(L[ui["ribs_left"]]), rib_R=float(L[ui["ribs_right"]]))
        # rib side
        if "rib" in sites:
            rq = " ".join(q for q in a["quotes"] if re.search(r"rib", q, re.I))
            sd = side_of(rq)
            r["rib_side"] = sd
            if sd in ("left", "right") and all(u in pres for u in ("ribs_left", "ribs_right")):
                r["rib_side_hit"] = (L[ui["ribs_left"]] > L[ui["ribs_right"]]) == (sd == "left")
        recs.append(r)
    fr = pd.DataFrame(recs)
    fr = fr[fr.n_good > 0]                       # report site covered by the bone units
    fr.to_csv(W / "loc_cases_fracture.csv", index=False)
    thr = fr.prob.median()
    for name, d in (("全部陽性", fr), (f"高分(機率≥陽性中位數 {thr:.2f})", fr[fr.prob >= thr]),
                    ("高分且無術後植入物", fr[(fr.prob >= thr) & ~fr.hardware])):
        rows.append(dict(疾病="骨折", 子集=name, 指標="部位類別 top-1（MIL 單元分數）", n=len(d),
                         模型=d.cat_top1.mean(), 隨機=d.rand_top1.mean()))
        rows.append(dict(疾病="骨折", 子集=name, 指標="部位類別 top-3", n=len(d), 模型=d.cat_top3.mean(), 隨機=d.rand_top3.mean()))
        rows.append(dict(疾病="骨折", 子集=name, 指標="部位類別 top-1（僅資料項，去 unit bias）", n=len(d),
                         模型=d.cat_top1_data.mean(), 隨機=d.rand_top1.mean()))
        dv = d[d.lvl_eval]
        rows.append(dict(疾病="骨折", 子集=name, 指標="椎體節數 ±1 top-1（椎體 site 內）", n=len(dv),
                         模型=dv.lvl_top1.astype(float).mean(), 隨機=dv.lvl_rand1.mean()))
        rows.append(dict(疾病="骨折", 子集=name, 指標="椎體節數 ±1 top-3（椎體 site 內）", n=len(dv),
                         模型=dv.lvl_top3.astype(float).mean(), 隨機=dv.lvl_rand3.mean()))
        if "rib_side_hit" in d:
            dr = d[d.rib_side_hit.notna()]
            rows.append(dict(疾病="骨折", 子集=name, 指標="肋骨側別（左右肋骨單元分數比較）", n=len(dr),
                             模型=dr.rib_side_hit.astype(float).mean(), 隨機=0.5))
    # stronger baselines: best fixed vertebra guess (chosen on the same data = optimistic), rib side bias
    for name, d in (("全部陽性", fr), ("高分", fr[fr.prob >= thr])):
        dv = d[d.lvl_eval]
        lv = [{LEVEL[x] for x in str(s).split(",") if x in LEVEL} for s in dv.levels]
        best = max(((u, np.mean([any(abs(LEVEL[u] - l) <= 1 for l in L) for L in lv])) for u in VERT), key=lambda x: x[1])
        rows.append(dict(疾病="骨折", 子集=name, 指標=f"椎體節數 ±1：固定猜 {best[0]} 的基準", n=len(dv),
                         模型=dv.lvl_top1.astype(float).mean(), 隨機=best[1]))
        rb = d[d.sites.str.contains("rib")]
        rows.append(dict(疾病="骨折", 子集=name + "，報告含肋骨", 指標="ribs_right 分數 > ribs_left 的比例（側別偏差）",
                         n=len(rb), 模型=(rb.rib_R > rb.rib_L).mean(),
                         隨機=0.5))
    # per-site breakdown (high-score)
    d = fr[fr.prob >= thr]
    for s in ["vertebra", "rib", "pelvis", "hip_femur"]:
        ds = d[d.sites == s]
        if len(ds):
            rows.append(dict(疾病="骨折", 子集=f"高分，報告僅 {s}", 指標="部位類別 top-1", n=len(ds),
                             模型=ds.cat_top1.mean(), 隨機=ds.rand_top1.mean()))

    # ---------- hydronephrosis ----------
    f = fi["hydronephrosis"]
    recs = []
    for n, cid in enumerate(ids):
        a = lab[cid]["hydronephrosis"]
        if a["category"] != "yes":
            continue
        side = a["detail"].get("side")
        oz, oa = A["occ_zero"][n, f], A["occ_abs"][n, f]
        kl, kr = ui["kidney_left"], ui["kidney_right"]
        if np.isnan(oz[kl]) or np.isnan(oz[kr]):
            continue
        recs.append(dict(id=cid, prob=float(A["prob"][n, f]), side=side, quote=" || ".join(a["quotes"]),
                         c_left=float(oz[kl]), c_right=float(oz[kr]), abs_left=float(oa[kl]), abs_right=float(oa[kr]),
                         top="left" if oz[kl] > oz[kr] else "right", top_abs="left" if oa[kl] > oa[kr] else "right"))
    hy = pd.DataFrame(recs)
    hy.to_csv(W / "loc_cases_hydronephrosis.csv", index=False)
    hs = hy[hy.side.isin(["left", "right"])]
    thr = hs.prob.median()
    for name, d in (("單側陽性", hs), (f"單側高分(≥{thr:.2f})", hs[hs.prob >= thr])):
        rows.append(dict(疾病="腎水腫", 子集=name, 指標="側別 top-1（線性拆解＝occlusion→0）", n=len(d),
                         模型=(d.top == d.side).mean(), 隨機=0.5))
        rows.append(dict(疾病="腎水腫", 子集=name, 指標="側別 top-1（標記為不存在 occlusion）", n=len(d),
                         模型=(d.top_abs == d.side).mean(), 隨機=0.5))
    for s in ["left", "right"]:
        d = hs[hs.side == s]
        rows.append(dict(疾病="腎水腫", 子集=f"報告={s}", 指標="側別 top-1", n=len(d), 模型=(d.top == s).mean(), 隨機=0.5))
    bl = hy[hy.side == "bilateral"]
    rows.append(dict(疾病="腎水腫", 子集="雙側（參考）", 指標="top=左腎比例", n=len(bl), 模型=(bl.top == "left").mean(), 隨機=np.nan))

    # ---------- single/two-unit findings ----------
    for fn, zh in (("abdominal_aortic_aneurysm", "腹主動脈瘤"), ("gallstones", "膽結石"),
                   ("splenomegaly", "脾腫大"), ("ascites", "腹水")):
        f = fi[fn]
        fu = [u for u in units if not np.isnan(A["occ_zero"][:, f, ui[u]]).all()]
        recs = []
        for n, cid in enumerate(ids):
            a = lab[cid][fn]
            if a["category"] != "yes":
                continue
            oz = A["occ_zero"][n, f]
            lin = A["lin"][n, f]                                 # [U,W]
            pres = [u for u in fu if not np.isnan(oz[ui[u]])]
            if not pres:
                continue
            top = max(pres, key=lambda u: oz[ui[u]])
            data = float(np.nansum(oz))
            win = np.nansum(lin, 0)
            recs.append(dict(id=cid, prob=float(A["prob"][n, f]), logit=float(A["logit"][n, f]), top=top,
                             data_term=data, const=float(A["logit"][n, f]) - data,
                             **{f"c_{u}": float(oz[ui[u]]) for u in fu},
                             **{f"win_{w}": float(win[j]) for j, w in enumerate(A["windows"])},
                             quote=" || ".join(a["quotes"])))
        d = pd.DataFrame(recs)
        d.to_csv(W / f"loc_cases_{fn}.csv", index=False)
        thr = d.prob.median()
        dh = d[d.prob >= thr]
        if len(fu) == 1:
            rows.append(dict(疾病=zh, 子集=f"高分(≥{thr:.2f})", 指標=f"top 單元＝{fu[0]}（此病只接 1 個單元，必然吻合）",
                             n=len(dh), 模型=1.0, 隨機=1.0))
        else:
            for u in fu:
                rows.append(dict(疾病=zh, 子集=f"高分(≥{thr:.2f})", 指標=f"top 單元＝{u} 的比例", n=len(dh),
                                 模型=(dh.top == u).mean(), 隨機=1 / len(fu)))
            q = dh.quote.str.lower()
            for kw, u in (("perihepatic|hepatic|liver|morison", "liver"), ("perisplenic|splen", "spleen")):
                dd = dh[q.str.contains(kw)]
                rows.append(dict(疾病=zh, 子集=f"高分且報告提及 {kw.split('|')[0]}", 指標=f"top 單元＝{u}", n=len(dd),
                                 模型=(dd.top == u).mean() if len(dd) else np.nan, 隨機=1 / len(fu)))
        rows.append(dict(疾病=zh, 子集=f"高分(≥{thr:.2f})", 指標="資料項占 logit 的平均（相對 train 平均器官）",
                         n=len(dh), 模型=dh.data_term.mean(), 隨機=np.nan))
        wcols = [c for c in d if c.startswith("win_")]
        top_w = dh[wcols].mean().idxmax()
        rows.append(dict(疾病=zh, 子集=f"高分(≥{thr:.2f})", 指標=f"貢獻最大的視窗={top_w[4:]}", n=len(dh),
                         模型=dh[top_w].mean(), 隨機=np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(W / "loc_summary.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(S.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
