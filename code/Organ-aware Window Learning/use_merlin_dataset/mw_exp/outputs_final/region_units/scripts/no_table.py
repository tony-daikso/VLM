"""Exp O: markdown result table from UNIT/stack/results.json (+ probe log lines and crop method counts). usage: no_table.py UNIT"""
import sys, json, glob, collections, numpy as np
U = sys.argv[1]; R = json.load(open(f"{U}/stack/results.json")); sel = R["selected"]
f3 = lambda x: f"{x:.3f}"
def d(r, k): return f"{r[k]['mean']:+.3f} [{r[k]['ci'][0]:+.3f}, {r[k]['ci'][1]:+.3f}]" if k in r else "–"
subs = sorted({k[4:-4] for r in R["results"] for k in r if k.startswith("sub_") and k.endswith("_val") and not k.endswith("_d_val") and not k.endswith("_n_val")})
print(f"train {R['results'][0]['n_tr']} 例／陽性 {R['results'][0]['pos_tr']}；val {R['results'][0]['n_val']} 例／陽性 {R['results'][0]['pos_val']}\n")
print("| 模型 | 堆疊器 | train OOF | Δ OOF [CI] | val [CI] | Δ val [CI] | 年齡配對 train／val | Δ 年齡配對 val [CI] | Spearman（陰性） | 敏感度@90%特異度 val |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in R["results"]:
    b = "**" if r["model"] == sel else ""
    print(f"| {b}{r['model']}{b} | {r['cand']} | {f3(r['oof'])} | {d(r,'d_oof')} | {f3(r['val'])} [{r['val_ci'][0]:.3f}, {r['val_ci'][1]:.3f}] | {d(r,'d_val')} | {f3(r['oof_age'])}／{f3(r['val_age'])} | {d(r,'d_val_age')} | {r['spearman_neg_val']:+.3f} | {100*r['sens90_val']:.1f}% |")
if subs:
    print("\n**子群**（子群陽性 vs 全部陰性）\n")
    print("| 模型 | " + " | ".join(f"{s}：train OOF／val（val 陽性 n）／Δ val [CI]" for s in subs) + " |")
    print("|---|" + "---|" * len(subs))
    for r in R["results"]:
        cells = []
        for s in subs:
            dd = r.get(f"sub_{s}_d_val"); ds = f"{dd['mean']:+.3f} [{dd['ci'][0]:+.3f}, {dd['ci'][1]:+.3f}]" if dd else "–"
            cells.append(f"{f3(r[f'sub_{s}_oof'])}／{f3(r[f'sub_{s}_val'])}（{r[f'sub_{s}_n_val']}）／{ds}")
        print(f"| {r['model']} | " + " | ".join(cells) + " |")
print("\n**probe（單元頭本身）**\n")
for f in sorted(glob.glob(f"{U}/logs*/probe*.log")):
    for l in open(f):
        if "train OOF" in l: print("- " + l.strip())
for f in sorted(glob.glob(f"{U}/logs*/probe*.log"))[:1]:
    txt = open(f).read(); i = txt.find("presence")
    if i >= 0: print("\n區域出現率（crop × 區域）：\n```\n" + txt[i + 10:i + 400].split("\n\n")[0] + "\n```")
cnt = collections.Counter()
for f in glob.glob(f"{U}/crops_info*.jsonl"):
    for l in open(f):
        s = json.loads(l)["status"]; cnt[s] += 1
print("\n定位方式（train＋val，A6000＋CT-rate）：" + "、".join(f"{k} {v}" for k, v in cnt.most_common(12)))
