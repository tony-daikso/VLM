#!/usr/bin/env python3
"""Agent M: per-attribute diagnosis on val600 (read-only, CPU, no Qwen calls; aggregates only)."""
import csv, hashlib, json, re, sys, statistics
from collections import Counter, defaultdict
TONY = "/home/advantech/Desktop/Tony/"
M4 = TONY + "VLM/OWL/report_gen/m4/"
sys.path.insert(0, M4 + "code_f/eval")
import evaluate as E  # noqa
POS = E.POS
ZH, AZ = E.ZH, E.ATTR_ZH
D14 = TONY + "VLM/merlin_disease14"
d14 = E.load_d14(D14)
PM = E.md5(D14 + "/prompt.md")
meta = {r["study id"]: r for r in csv.DictReader(open(D14 + "/metadata.csv"))}


def rj(p):
    return [json.loads(l) for l in open(p)]


ids600 = set(open(M4 + "data_sub/val_qwen_subset_ids.txt").read().split())
train_ids = {r["case_id"] for r in rj(M4 + "improve/data_unit_pil/train.jsonl")}
val_ids = {r["case_id"] for r in rj(M4 + "improve/data_unit_pil/val.jsonl")}
labels, lab_tr, lab_val = {}, {}, {}
for l in open(TONY + "VLM/OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(l)
    if r.get("split") == "test":
        continue
    a = r["answer"] if isinstance(r["answer"], dict) else None
    if a is None:
        continue
    if r["split"] == "val" and r["study_id"] in ids600:
        labels[r["study_id"]] = a
    if r["split"] == "val" and r["study_id"] in val_ids:
        lab_val[r["study_id"]] = a
    if r["split"] == "train" and r["study_id"] in train_ids:
        lab_tr[r["study_id"]] = a
tgt = {r["case_id"]: r for r in rj(M4 + "inputs/targets_v2/targets_val.jsonl") if r["case_id"] in ids600}
cache = {}
for p in ("eval/qwen_extract_cache.jsonl", "eval/qwen_extract_cache_F.jsonl"):
    for l in open(M4 + p):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("parse_ok") or r["key"] not in cache:
            cache[r["key"]] = r

ARMS = {"A+": ("gen_f/T_unit_gt_pil_val600.jsonl", "eval/results_f/T_unit_gt_pil_val600", "improve/data/val600_unit_pil.jsonl"),
        "T_unit_gt": ("gen_f/T_unit_gt_val600.jsonl", "eval/results_f/T_unit_gt_val600", "improve/data/val600_unit_v2.jsonl"),
        "T_gt": ("gen_cue/T_gt/val600.jsonl", "eval/results_cue/T_gt_val600", "data_sub/val_sub600.jsonl"),
        "U_gt": ("gen_cue/U_gt/val600.jsonl", "eval/results_cue/U_gt_val600", "data_sub/val_sub600.jsonl")}
GEN, EXT, DAT, MET = {}, {}, {}, {}
for arm, (g, res, dat) in ARMS.items():
    m = json.load(open(M4 + res + "/metrics.json"))
    MET[arm] = m
    q = m["qwen"]
    GEN[arm], EXT[arm] = {}, {}
    for r in rj(M4 + g):
        c = r["case_id"]
        if c not in labels:
            continue
        txt, fd = E.gen_text(r)
        GEN[arm][c] = (txt, fd)
        user = f"{d14.patient_line(meta.get(c))}\n\n# Report\n\n{txt}"
        key = hashlib.sha256(json.dumps([q["prompt_md5"], q["model_tag"], q["max_tokens"], user]).encode()).hexdigest()
        hit = cache.get(key)
        if hit is not None and hit.get("parse_ok"):
            EXT[arm][c] = hit["answer"]
    DAT[arm] = {r["case_id"]: r for r in rj(M4 + dat)}
    print(f"<!-- {arm}: gen {len(GEN[arm])}, cached extraction {len(EXT[arm])} -->")

UC = json.load(open(M4 + "improve/data_unit_pil/unit_cue_pil.json"))
THR = UC["thresholds"]
LEVELS = UC["levels"]
ORD3 = E.ORD_GRADE
ORDA = ["trace", "small", "moderate", "large"]
HI = {"moderate", "severe", "large"}


def sent(arm, c, d):
    fd = GEN[arm][c][1] or {}
    s = fd.get(d, "")
    if isinstance(s, list):
        s = " ".join(x for x in s if x)
    return (s or "").lower()


def tsent(c, d):
    return (E.target_sentences(tgt[c]).get(d) or "").lower() if c in tgt else ""


def both(arm, d):
    return [c for c in EXT[arm] if labels[c][d]["category"] in POS and EXT[arm][c][d]["category"] in POS]


def genpos(arm, d):
    return [c for c in EXT[arm] if EXT[arm][c][d]["category"] in POS]


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else f"{a}/0"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------- train statistics (majority classes, usable data)
def train_dist(lab, d, f):
    pos = [c for c in lab if lab[c][d]["category"] in POS]
    vals = [lab[c][d]["detail"].get(f) for c in pos]
    st = [v for v in vals if E.stated(v)]
    return len(pos), st


TRD = {}
for d, f, kind, arg in E.ATTRS:
    if kind == "levels":
        continue
    TRD[(d, f)] = train_dist(lab_tr, d, f)


def majority(d, f, among=None):
    st = TRD[(d, f)][1]
    if among:
        st = [v for v in st if v in among]
    return Counter(st).most_common(1)[0][0] if st else None


# ---------------- prompt-derived rule values (A+ prompt; fixed thresholds from unit_cue_pil.json)
def prompt_info(arm, c):
    r = DAT[arm][c]
    p = r["prompt"]
    o = {}
    us = r.get("unit_scores") or {}
    o["kid"] = us.get("kid")
    o["vert"] = us.get("vert")
    gl = [x for x in p.split("\n") if x.startswith("Grade probe scores")]
    o["grade"] = {}
    if gl:
        for nm, v in re.findall(r"(hepatic steatosis|ascites|atherosclerosis) [a-z ]+? (\d\.\d\d)", gl[0]):
            o["grade"][nm] = float(v)
    m = re.search(r"abdominal aortic max diameter (\d+(?:\.\d+)?) cm", p)
    o["aaa"] = float(m.group(1)) if m else None
    m = re.search(r"spleen craniocaudal length (\d+(?:\.\d+)?) cm", p)
    o["spl"] = float(m.group(1)) if m else None
    return o


GKEY = {"hepatic_steatosis": ("hepatic steatosis", "hs_grade", ORD3), "ascites": ("ascites", "asc_amount", ORDA),
        "atherosclerosis": ("atherosclerosis", "ath_grade", ORD3)}


def rule(arm, c, d, f):
    """-> (has_rule, value or None)"""
    if arm not in ("A+",):
        if (d, f) in (("abdominal_aortic_aneurysm", "diameter_cm"), ("splenomegaly", "length_cm")):
            o = prompt_info(arm, c)
            return True, o["aaa"] if d.startswith("abd") else o["spl"]
        return False, None
    o = prompt_info(arm, c)
    if (d, f) == ("hydronephrosis", "side"):
        k = o["kid"] or [None, None]
        r_, l_ = [(x is not None and x >= THR["kidney"] - 1e-9) for x in k]
        return True, "bilateral" if r_ and l_ else "right" if r_ else "left" if l_ else None
    if f in ("grade", "amount") and d in GKEY:
        nm, key, order = GKEY[d]
        if nm not in o["grade"]:
            return True, None
        hi = o["grade"][nm] >= THR[key] - 1e-9
        return True, majority(d, f, [v for v in order if (v in HI) == hi])
    if (d, f) == ("abdominal_aortic_aneurysm", "diameter_cm"):
        return True, o["aaa"]
    if (d, f) == ("splenomegaly", "length_cm"):
        return True, o["spl"]
    if d == "fracture" and f in ("vertebral_compression", "sites"):
        v = [x for x in (o["vert"] or []) if x is not None]
        anyv = any(x >= THR["vertebra"] - 1e-9 for x in v)
        if f == "vertebral_compression":
            return True, "yes" if anyv else "no"
        return True, ["vertebra"] if anyv else None
    return False, None


INFO = {("hydronephrosis", "side"): "單元分數：左右腎各一個腎水腫分數（門檻 .29）",
        ("hydronephrosis", "grade"): "無（hy arm 排隊中）",
        ("hepatic_steatosis", "grade"): "分級分數（中度以上機率，門檻 .32；疾病 cue 陽性才顯示）＋肝/脾 HU 量測",
        ("splenomegaly", "length_cm"): "量測：脾臟頭尾徑（cm）＋脾體積",
        ("abdominal_aortic_aneurysm", "diameter_cm"): "量測：腹主動脈最大徑（cm）",
        ("abdominal_aortic_aneurysm", "repaired"): "無",
        ("biliary_ductal_dilation", "site"): "無", ("biliary_ductal_dilation", "cbd_mm"): "無",
        ("ascites", "amount"): "分級分數（中量以上機率，門檻 .37；疾病 cue 陽性才顯示）",
        ("ascites", "hemorrhagic"): "無", ("bowel_obstruction", "type"): "無",
        ("appendicitis", "perforated"): "無", ("appendicitis", "appendicolith"): "無",
        ("free_air", "postoperative"): "無", ("gallstones", "calcified"): "無",
        ("atherosclerosis", "grade"): "分級分數（中度以上機率，門檻 .45；疾病 cue 陽性才顯示）＋主髂動脈壁鈣化量測",
        ("atherosclerosis", "aortoiliac"): "間接：鈣化量測只量主髂動脈壁（無對應規則）",
        ("fracture", "sites"): "只有椎體：T8–L5 單元分數（門檻 .24）；肋骨/骨盆/薦骨/髖部無",
        ("fracture", "vertebral_compression"): "單元分數：T8–L5 任一節 ≥ .24",
        ("fracture", "acute"): "無", ("fracture", "levels"): "單元分數：T8–L5（門檻 .24）"}

out = []
P = out.append


def attr_rows(arm):
    rows = {}
    for d, f, kind, arg in E.ATTRS:
        if kind == "levels":
            continue
        b = both(arm, d)
        el, gs, ex, wi, hall = [], 0, 0, 0, 0
        cat = Counter()
        errs = []
        for c in b:
            rv = labels[c][d]["detail"].get(f)
            gv = EXT[arm][c][d]["detail"].get(f)
            if not E.stated(rv):
                hall += int(E.stated(gv))
                continue
            el.append(c)
            gs += int(E.stated(gv))
            e, w = E.attr_compare(kind, arg, rv, gv)
            ex += int(e)
            wi += int(w)
            if not E.stated(gv):
                cat["未寫"] += 1
            elif e:
                cat["正確"] += 1
            elif kind == "ord" and rv in arg and gv in arg:
                dd = abs(arg.index(rv) - arg.index(gv))
                cat["差一級" if dd == 1 else "差兩級以上"] += 1
            elif kind == "num":
                er = abs(float(rv) - float(gv))
                errs.append(er)
                cat["數值錯（≤2×容許）" if er <= 2 * arg + 1e-9 else "數值錯（>2×容許）"] += 1
            elif kind == "set":
                rs, gset = set(rv), set(gv)
                cat["少寫（真子集）" if gset < rs else "多寫（真超集）" if gset > rs else "部分重疊" if rs & gset else "完全不同"] += 1
            else:
                cat["寫錯"] += 1
        rows[(d, f)] = dict(both=len(b), el=el, gs=gs, ex=ex, wi=wi, hall=hall, cat=cat, errs=errs)
    return rows


AR = {arm: attr_rows(arm) for arm in ARMS}
# ---- reproduce evaluator
bad = 0
for arm in ARMS:
    for r in MET[arm]["attributes"]:
        if r["kind"] == "levels":
            continue
        x = AR[arm][(r["disease"], r["field"])]
        if (x["both"], len(x["el"]), x["gs"]) != (r["both_pos"], r["ref_stated"], r["gen_stated"]) or \
                (len(x["el"]) and abs(x["ex"] / len(x["el"]) - r["acc"]) > 1e-9):
            bad += 1
            print("<!-- MISMATCH", arm, r["disease"], r["field"], x["both"], len(x["el"]), x["gs"], x["ex"], r["both_pos"], r["ref_stated"], r["gen_stated"], r["acc"], "-->")
P(f"重現檢查：4 個 arm × 20 個屬性（不含節段）與 evaluator metrics.json 的 n／準確率不一致的格數 = {bad}。\n")

# ---- table 1: accuracy across arms
P("## 1. 各 arm 屬性準確率（val600，Qwen 抽取；分母 = 參考與生成皆陽性且參考有寫該屬性；生成沒寫算錯）\n")
P("| 疾病 | 屬性 | 定義 | A+ n（皆陽性/參考有寫/生成有寫） | A+ | T_unit_gt | T_gt | U_gt | A+ 有寫時準確率 |")
P("|---|---|---|---|---|---|---|---|---|")
KD = {"cat": "類別完全相同", "ord": "分級完全相同（另計差一級內）", "num": "數值差在容許內", "set": "部位集合完全相同（另計有交集）"}
for d, f, kind, arg in E.ATTRS:
    if kind == "levels":
        continue
    x = AR["A+"][(d, f)]
    cells = []
    for arm in ARMS:
        y = AR[arm][(d, f)]
        cells.append(pct(y["ex"], len(y["el"])))
    P(f"| {ZH[d]} | {AZ[f]} | {KD[kind]} | {x['both']}/{len(x['el'])}/{x['gs']} | " + " | ".join(cells) + f" | {pct(x['ex'], x['gs'])} |")
P("| 骨折 | 椎體節段 | 節段集合完全相同（regex） | 54/43/32 | 14/43 (32.6%) | 12/43 (27.9%) | 1/44 (2.3%) | 1/43 (2.3%) | 14/32 (43.8%) |（沿用 level_gap 診斷）")

# ---- table 2: error breakdown + stated rate per arm
P("\n## 2. A+ 錯誤拆解（分母同上）\n")
P("| 疾病 | 屬性 | n | 正確 | 未寫 | 寫錯/差一級/數值 | 參考值分布 | 生成值分布（有寫者） | 參考沒寫而生成寫了 |")
P("|---|---|---|---|---|---|---|---|---|")


def dist_str(vals, kind):
    if kind == "num":
        v = sorted(float(x) for x in vals)
        return f"n={len(v)}，中位數 {statistics.median(v):.1f}" if v else "—"
    if kind == "set":
        k = Counter(x for s in vals for x in s)
    else:
        k = Counter(vals)
    return "、".join(f"{a} {b}" for a, b in k.most_common()) or "—"


for d, f, kind, arg in E.ATTRS:
    if kind == "levels":
        continue
    x = AR["A+"][(d, f)]
    el = x["el"]
    rvs = [labels[c][d]["detail"].get(f) for c in el]
    gvs = [EXT["A+"][c][d]["detail"].get(f) for c in el]
    gvs = [v for v in gvs if E.stated(v)]
    oth = "；".join(f"{k} {v}" for k, v in x["cat"].items() if k not in ("正確", "未寫")) or "—"
    if x["errs"]:
        oth += f"（錯誤者絕對誤差中位數 {statistics.median(x['errs']):.1f}）"
    P(f"| {ZH[d]} | {AZ[f]} | {len(el)} | {x['cat']['正確']} | {x['cat']['未寫']} | {oth} | {dist_str(rvs, kind)} | {dist_str(gvs, kind)} | {x['hall']}/{x['both'] - len(el)} |")

P("\n### 2b. 各 arm「生成有寫該屬性」的比例（分母 = 參考有寫）與有寫時的準確率\n")
P("| 疾病 | 屬性 | A+ 有寫 / 有寫時對 | T_unit_gt | T_gt | U_gt |")
P("|---|---|---|---|---|---|")
for d, f, kind, arg in E.ATTRS:
    if kind == "levels":
        continue
    cells = []
    for arm in ARMS:
        y = AR[arm][(d, f)]
        cells.append(f"{pct(y['gs'], len(y['el']))} / {pct(y['ex'], y['gs'])}")
    P(f"| {ZH[d]} | {AZ[f]} | " + " | ".join(cells) + " |")

# ---- table 3: prompt information, ceiling, LLM agreement
P("\n## 3. prompt 內的資訊、固定規則上限、LLM 是否照用（A+）\n")
P("規則：門檻全部取自 unit_cue_pil.json（腎 .29、椎體 .24、脂肪肝分級 .32、腹水量 .37、動脈硬化分級 .45）；分級分數過門檻 → train 中「中度以上」的多數類，未過 → 「輕度側」的多數類；"
  "量測直接照抄。「常數基準」= 一律輸出 train 多數類。全部在 A+ 的同一批分母病例上計算，沒有在 val 上調任何東西。\n")
P("| 疾病 | 屬性 | prompt 資訊 | n | A+ 報告 | 規則上限（同分母） | 規則有值的例數 | 常數基準（train 多數類） | 二分（輕 vs 中度以上）：報告 / 規則 | LLM 與規則一致（生成陽性且兩者皆有值） |")
P("|---|---|---|---|---|---|---|---|---|---|")
for d, f, kind, arg in E.ATTRS:
    if kind == "levels":
        continue
    x = AR["A+"][(d, f)]
    el = x["el"]
    has = rule("A+", el[0], d, f)[0] if el else False
    rc = rs = 0
    for c in el:
        _, v = rule("A+", c, d, f)
        if E.stated(v):
            rs += 1
            rc += int(E.attr_compare(kind, arg, labels[c][d]["detail"].get(f), v)[0])
    maj = majority(d, f) if kind in ("cat", "ord") else None
    if kind == "set":
        maj = Counter(tuple(sorted(v)) for v in TRD[(d, f)][1]).most_common(1)[0][0]
        mc = sum(set(labels[c][d]["detail"].get(f)) == set(maj) for c in el)
        majs = f"{'+'.join(maj)}：{pct(mc, len(el))}"
    elif maj is not None:
        mc = sum(labels[c][d]["detail"].get(f) == maj for c in el)
        majs = f"{maj}：{pct(mc, len(el))}"
    else:
        majs = "—"
    bins = "—"
    if kind == "ord":
        gb = [c for c in el if E.stated(EXT["A+"][c][d]["detail"].get(f))]
        g_ok = sum((EXT["A+"][c][d]["detail"].get(f) in HI) == (labels[c][d]["detail"].get(f) in HI) for c in gb)
        rb = [c for c in el if E.stated(rule("A+", c, d, f)[1])]
        r_ok = sum((rule("A+", c, d, f)[1] in HI) == (labels[c][d]["detail"].get(f) in HI) for c in rb)
        bins = f"{pct(g_ok, len(el))} / " + (pct(r_ok, len(el)) if has else "—")
    # LLM agreement with rule on all gen-positive cases
    ag = "—"
    if has:
        gp = genpos("A+", d)
        n = k = ns = 0
        for c in gp:
            _, v = rule("A+", c, d, f)
            gv = EXT["A+"][c][d]["detail"].get(f)
            if E.stated(v):
                n += 1
                if E.stated(gv):
                    ns += 1
                    if kind == "num":
                        k += int(abs(float(gv) - float(v)) <= 0.051)
                    elif kind == "set":
                        k += int("vertebra" in gv)
                    else:
                        k += int(gv == v)
        ag = f"規則有值 {n}；報告有寫 {ns}；一致 {pct(k, n)}"
    P(f"| {ZH[d]} | {AZ[f]} | {INFO[(d, f)]} | {len(el)} | {pct(x['ex'], len(el))} | {pct(rc, len(el)) if has else '無規則'} | {rs if has else '—'} | {majs} | {bins} | {ag} |")
P("| 骨折 | 椎體節段 | 單元分數：T8–L5（門檻 .24） | 43 | 14/43 (32.6%) | 過門檻集合 12/43 (27.9%)；空集合退回 top-1 18/43 (41.9%)；top-1 命中 34/43 | 30 | — | — | cue 有節段時報告完全照抄 31/43 (72.1%)、含 top-1 39/43 |（沿用 level_gap）")

# ---- detail blocks
P("\n## 4. 個別屬性細節\n")


def conf(arm, d, f, cs, order=None, rulev=False):
    k = Counter()
    for c in cs:
        rv = labels[c][d]["detail"].get(f)
        gv = rule(arm, c, d, f)[1] if rulev else EXT[arm][c][d]["detail"].get(f)
        k[(rv, gv if E.stated(gv) else "未寫")] += 1
    rvals = order or sorted({a for a, _ in k})
    gvals = (order or sorted({b for _, b in k if b != "未寫"})) + ["未寫"]
    gvals = [g for g in gvals if any(k[(r, g)] for r in rvals)] or ["未寫"]
    P("| 參考 ＼ " + ("規則" if rulev else "生成") + " | " + " | ".join(gvals) + " |")
    P("|---|" + "---|" * len(gvals))
    for r in rvals:
        if any(k[(r, g)] for g in gvals):
            P(f"| {r} | " + " | ".join(str(k[(r, g)]) for g in gvals) + " |")


# hydronephrosis
P("### 腎水腫\n")
el = AR["A+"][("hydronephrosis", "side")]["el"]
P("側別混淆表（A+，生成）：")
conf("A+", "hydronephrosis", "side", el, ["left", "right", "bilateral", "transplant"])
P("\n側別混淆表（規則：兩腎分數 ≥ .29）：")
conf("A+", "hydronephrosis", "side", el, ["left", "right", "bilateral", "transplant"], rulev=True)
fb = 0
for c in el:
    v = rule("A+", c, "hydronephrosis", "side")[1]
    if v is None:
        k = prompt_info("A+", c)["kid"] or [None, None]
        if k[0] is not None and k[1] is not None:
            v = "right" if k[0] >= k[1] else "left"
    fb += int(v == labels[c]["hydronephrosis"]["detail"].get("side"))
P(f"\n規則變體「兩腎都沒過門檻時取分數較高的一側」：{pct(fb, len(el))}。")
el = AR["A+"][("hydronephrosis", "grade")]["el"]
P("\n程度混淆表（A+，生成；prompt 沒有程度資訊）：")
conf("A+", "hydronephrosis", "grade", el, ORD3)

# steatosis / ascites / atherosclerosis grade
for d, f, order in (("hepatic_steatosis", "grade", ORD3), ("ascites", "amount", ORDA), ("atherosclerosis", "grade", ORD3)):
    nm, key, _ = GKEY[d]
    P(f"\n### {ZH[d]}{AZ[f]}\n")
    x = AR["A+"][(d, f)]
    el = x["el"]
    P(f"皆陽性 {x['both']} 例，參考有寫 {len(el)}，生成有寫 {x['gs']}。生成混淆表（A+）：")
    conf("A+", d, f, el, order)
    P("\n規則混淆表（分級分數，門檻 " + str(THR[key]) + "）：")
    conf("A+", d, f, el, order, rulev=True)
    shown = [c for c in el if nm in prompt_info("A+", c)["grade"]]
    P(f"\n分母 {len(el)} 例中 prompt 有顯示分級分數 {len(shown)} 例。")
    # among all gen-positive: stated rate by cue bin
    gp = genpos("A+", d)
    bins = Counter(); st = Counter(); hi = Counter()
    for c in gp:
        g = prompt_info("A+", c)["grade"].get(nm)
        b = "未顯示" if g is None else ("≥門檻" if g >= THR[key] - 1e-9 else "<門檻")
        bins[b] += 1
        gv = EXT["A+"][c][d]["detail"].get(f)
        if E.stated(gv):
            st[b] += 1
            hi[b] += int(gv in HI)
    P("生成陽性的全部病例（不看參考）依分級分數分組：" + "；".join(f"{b}：{bins[b]} 例，報告有寫分級 {st[b]}，其中寫中度以上 {hi[b]}" for b in ("≥門檻", "<門檻", "未顯示") if bins[b]) + "。")
    # reference side: stated rate by cue bin among both-positive
    b2 = both("A+", d)
    rb = Counter(); rs = Counter(); rh = Counter()
    for c in b2:
        g = prompt_info("A+", c)["grade"].get(nm)
        b = "未顯示" if g is None else ("≥門檻" if g >= THR[key] - 1e-9 else "<門檻")
        rb[b] += 1
        rv = labels[c][d]["detail"].get(f)
        if E.stated(rv):
            rs[b] += 1
            rh[b] += int(rv in HI)
    P("皆陽性病例的參考端：" + "；".join(f"{b}：{rb[b]} 例，參考有寫分級 {rs[b]}，其中中度以上 {rh[b]}" for b in ("≥門檻", "<門檻", "未顯示") if rb[b]) + "。")

# severity vocabulary in sentences (atherosclerosis, steatosis, ascites)
MILD = r"\b(mild|mildly|minimal|minimally|minor|slight|slightly|scattered|trace|small|tiny)\b"
MOD = r"\b(moderate|moderately)\b"
SEV = r"\b(severe|severely|extensive|marked|markedly|advanced|heavy|heavily|dense|densely|diffuse|significant|prominent|large|massive|profound)\b"


def sev_class(s):
    a, b, c = bool(re.search(MILD, s)), bool(re.search(MOD, s)), bool(re.search(SEV, s))
    return "無程度詞" if not (a or b or c) else "+".join(n for n, t in (("輕度類詞", a), ("中度詞", b), ("重度類詞", c)) if t)


P("\n### 程度用詞（regex，不經 Qwen）：參考目標句 vs 生成句，皆陽性病例\n")
P("輕度類詞 = mild/minimal/minor/slight/scattered/trace/small/tiny；中度詞 = moderate；重度類詞 = severe/extensive/marked/advanced/heavy/dense/diffuse/significant/prominent/large/massive/profound。\n")
P("| 疾病 | arm | 皆陽性 | 參考目標句用詞分布 | 生成句用詞分布 | 參考 Qwen 分級分布 | 生成 Qwen 分級分布 |")
P("|---|---|---|---|---|---|---|")
for d, f in (("atherosclerosis", "grade"), ("hepatic_steatosis", "grade"), ("ascites", "amount"), ("hydronephrosis", "grade")):
    for arm in ARMS:
        b = both(arm, d)
        rk = Counter(sev_class(tsent(c, d)) for c in b)
        gk = Counter(sev_class(sent(arm, c, d)) for c in b)
        rq = Counter(labels[c][d]["detail"].get(f) for c in b)
        gq = Counter(EXT[arm][c][d]["detail"].get(f) for c in b)
        fm = lambda k: "、".join(f"{a} {n}" for a, n in k.most_common())  # noqa
        P(f"| {ZH[d]} | {arm} | {len(b)} | {fm(rk)} | {fm(gk)} | {fm(rq)} | {fm(gq)} |")
# cross: Qwen grade vs severity word class, reference side (atherosclerosis), val600 both-pos A+
d = "atherosclerosis"
b = both("A+", d)
k = Counter((labels[c][d]["detail"].get("grade"), sev_class(tsent(c, d))) for c in b)
P("\n動脈粥狀硬化，參考端「Qwen 分級 × 目標句用詞」（A+ 皆陽性）：" + "；".join(f"{a}／{w} {n}" for (a, w), n in sorted(k.items(), key=lambda t: -t[1])) + "。")
k = Counter((EXT["A+"][c][d]["detail"].get("grade"), sev_class(sent("A+", c, d))) for c in b)
P("動脈粥狀硬化，生成端「Qwen 分級 × 生成句用詞」（A+ 皆陽性）：" + "；".join(f"{a}／{w} {n}" for (a, w), n in sorted(k.items(), key=lambda t: -t[1])) + "。")
# does target sentence exist / empty
P(f"動脈粥狀硬化皆陽性 {len(b)} 例中，targets_v2 有動脈粥狀硬化目標句 {sum(bool(tsent(c, d)) for c in b)} 例；生成有該句 {sum(bool(sent('A+', c, d)) for c in b)} 例。")
# severity word given grade cue bin (generated)
kk = defaultdict(Counter)
for c in genpos("A+", d):
    g = prompt_info("A+", c)["grade"].get("atherosclerosis")
    bn = "未顯示" if g is None else ("≥.45" if g >= .45 - 1e-9 else "<.45")
    kk[bn][sev_class(sent("A+", c, d))] += 1
P("動脈粥狀硬化生成陽性病例，生成句用詞依分級分數分組：" + "；".join(f"{bn}：" + "、".join(f"{w} {n}" for w, n in v.most_common()) for bn, v in kk.items()) + "。")
kk = defaultdict(Counter)
for c in b:
    g = prompt_info("A+", c)["grade"].get("atherosclerosis")
    bn = "未顯示" if g is None else ("≥.45" if g >= .45 - 1e-9 else "<.45")
    kk[bn][sev_class(tsent(c, d))] += 1
P("同上，參考目標句（皆陽性）：" + "；".join(f"{bn}：" + "、".join(f"{w} {n}" for w, n in v.most_common()) for bn, v in kk.items()) + "。")

# sentence diversity
P("\n### 句型多樣性（陽性病例的句子，正規化後去重；只有計數）\n")
P("| 疾病 | 參考陽性例數 | 參考目標句：相異句數 / 最常見句占比 | A+ 生成陽性例數 | A+ 生成句：相異句數 / 最常見句占比 | T_gt 生成句：相異句數 / 最常見句占比 |")
P("|---|---|---|---|---|---|")
norm = lambda s: re.sub(r"\d+(\.\d+)?", "#", re.sub(r"\s+", " ", s.strip()))  # noqa
for d in E.DISEASES:
    rp = [c for c in labels if labels[c][d]["category"] in POS and tsent(c, d)]
    rk = Counter(norm(tsent(c, d)) for c in rp)
    cells = []
    for arm in ("A+", "T_gt"):
        gp = [c for c in genpos(arm, d) if sent(arm, c, d)]
        gk = Counter(norm(sent(arm, c, d)) for c in gp)
        cells.append((len(gp), f"{len(gk)} / {100 * gk.most_common(1)[0][1] / len(gp):.0f}%" if gp else "—"))
    P(f"| {ZH[d]} | {len(rp)} | {len(rk)} / {100 * rk.most_common(1)[0][1] / len(rp):.0f}% | {cells[0][0]} | {cells[0][1]} | {cells[1][1]} |")

# bowel obstruction
P("\n### 腸阻塞類型\n")
d = "bowel_obstruction"
P("| arm | 皆陽性 | 參考有寫 | 生成有寫（Qwen） | 正確 | 生成句含 small bowel/SBO | 生成句含 large bowel/colon/LBO | 兩者皆無 | 生成 Qwen 類型分布 |")
P("|---|---|---|---|---|---|---|---|---|")
SB = r"small[- ]bowel|\bsbo\b|jejun|ile(um|al)"
LB = r"large[- ]bowel|\blbo\b|colon|colonic|sigmoid|cecal|cecum"
for arm in ARMS:
    x = AR[arm][(d, "type")]
    el = x["el"]
    s = sum(bool(re.search(SB, sent(arm, c, d))) for c in el)
    l_ = sum(bool(re.search(LB, sent(arm, c, d))) for c in el)
    n0 = sum(not re.search(SB, sent(arm, c, d)) and not re.search(LB, sent(arm, c, d)) for c in el)
    gq = Counter(EXT[arm][c][d]["detail"].get("type") for c in el)
    P(f"| {arm} | {x['both']} | {len(el)} | {x['gs']} | {x['ex']} | {s} | {l_} | {n0} | " + "、".join(f"{a} {n}" for a, n in gq.most_common()) + " |")
el = AR["A+"][(d, "type")]["el"]
P("\n參考類型分布（A+ 分母）：" + dist_str([labels[c][d]["detail"].get("type") for c in el], "cat") + f"。參考目標句含 small bowel 類詞 {sum(bool(re.search(SB, tsent(c, d))) for c in el)}、含 large bowel 類詞 {sum(bool(re.search(LB, tsent(c, d))) for c in el)}、皆無 {sum(not re.search(SB, tsent(c, d)) and not re.search(LB, tsent(c, d)) for c in el)}。")
P("A+ 混淆表：")
conf("A+", d, "type", el, ["SBO", "LBO", "both"])
P("\nT_gt 混淆表：")
conf("T_gt", d, "type", AR["T_gt"][(d, "type")]["el"], ["SBO", "LBO", "both"])
# stated-by-Qwen vs regex
for arm in ("A+", "T_gt"):
    el = AR[arm][(d, "type")]["el"]
    k = Counter((bool(re.search(SB + "|" + LB, sent(arm, c, d))), E.stated(EXT[arm][c][d]["detail"].get("type"))) for c in el)
    P(f"\n{arm}：生成句有腸段詞 × Qwen 判有寫類型：有詞且有寫 {k[(True, True)]}、有詞但 Qwen 判未寫 {k[(True, False)]}、無詞但 Qwen 判有寫 {k[(False, True)]}、無詞且未寫 {k[(False, False)]}。")
# same-case comparison T_gt vs A+
common = set(AR["A+"][(d, "type")]["el"]) & set(AR["T_gt"][(d, "type")]["el"])
k = Counter((E.stated(EXT["T_gt"][c][d]["detail"].get("type")), E.stated(EXT["A+"][c][d]["detail"].get("type"))) for c in common)
P(f"\n同一批 {len(common)} 例：T_gt 有寫且 A+ 有寫 {k[(True, True)]}、T_gt 有寫而 A+ 沒寫 {k[(True, False)]}、T_gt 沒寫而 A+ 有寫 {k[(False, True)]}、都沒寫 {k[(False, False)]}。")

# numeric: AAA, spleen
for d, f, tol, key in (("abdominal_aortic_aneurysm", "diameter_cm", 0.3, "aaa"), ("splenomegaly", "length_cm", 1.0, "spl")):
    P(f"\n### {ZH[d]}{AZ[f]}\n")
    P("| arm | 皆陽性 | 參考有寫 | 生成有寫 | 報告在容許內 | prompt 量測有值 | 量測在容許內（照抄上限） | 報告數值 = prompt 量測（±0.05） | 報告≠量測時：報告對 / 量測對 | |量測−參考| 中位數 | |報告−參考| 中位數 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        x = AR[arm][(d, f)]
        el = x["el"]
        pm_ = {c: prompt_info(arm, c)[key] for c in el}
        rv = {c: fnum(labels[c][d]["detail"].get(f)) for c in el}
        gv = {c: fnum(EXT[arm][c][d]["detail"].get(f)) for c in el}
        hasm = [c for c in el if pm_[c] is not None]
        ceil = sum(abs(pm_[c] - rv[c]) <= tol + 1e-9 for c in hasm)
        gst = [c for c in hasm if gv[c] is not None]
        cp = sum(abs(gv[c] - pm_[c]) <= 0.051 for c in gst)
        nc = [c for c in gst if abs(gv[c] - pm_[c]) > 0.051]
        nc_g = sum(abs(gv[c] - rv[c]) <= tol + 1e-9 for c in nc)
        nc_m = sum(abs(pm_[c] - rv[c]) <= tol + 1e-9 for c in nc)
        md = statistics.median([abs(pm_[c] - rv[c]) for c in hasm]) if hasm else float("nan")
        gd = statistics.median([abs(gv[c] - rv[c]) for c in gst]) if gst else float("nan")
        P(f"| {arm} | {x['both']} | {len(el)} | {x['gs']} | {pct(x['ex'], len(el))} | {len(hasm)} | {pct(ceil, len(el))} | {pct(cp, len(gst))} | {nc_g} / {nc_m}（{len(nc)} 例） | {md:.2f} | {gd:.2f} |")
    # signed bias of measurement vs reference (A+ denominators)
    el = AR["A+"][(d, f)]["el"]
    df = [prompt_info("A+", c)[key] - fnum(labels[c][d]["detail"].get(f)) for c in el if prompt_info("A+", c)[key] is not None]
    if df:
        P(f"\n量測 − 參考（A+ 分母，{len(df)} 例）：中位數 {statistics.median(df):+.2f}，量測偏小 {sum(x < -tol for x in df)} 例、偏大 {sum(x > tol for x in df)} 例、容許內 {sum(abs(x) <= tol + 1e-9 for x in df)} 例。")
    # all ref-pos with stated value regardless of gen
    rp = [c for c in labels if labels[c][d]["category"] in POS and fnum(labels[c][d]["detail"].get(f)) is not None and c in DAT["A+"]]
    hm = [c for c in rp if prompt_info("A+", c)[key] is not None]
    P(f"不限生成是否陽性：參考陽性且有寫數值 {len(rp)} 例，量測在容許內 {pct(sum(abs(prompt_info('A+', c)[key] - fnum(labels[c][d]['detail'].get(f))) <= tol + 1e-9 for c in hm), len(rp))}。")
    # same cases across arms
    common = set.intersection(*[set(AR[a][(d, f)]["el"]) for a in ARMS])
    P(f"四個 arm 共同分母 {len(common)} 例上的正確數：" + "、".join(f"{a} {sum(E.attr_compare('num', tol, labels[c][d]['detail'].get(f), EXT[a][c][d]['detail'].get(f))[0] for c in common)}" for a in ARMS) + f"；量測照抄上限 {sum(prompt_info('A+', c)[key] is not None and abs(prompt_info('A+', c)[key] - fnum(labels[c][d]['detail'].get(f))) <= tol + 1e-9 for c in common)}。")
    if d.startswith("abd"):
        for arm in ARMS:
            el = AR[arm][(d, "repaired")]["el"]
            P(f"{arm} 已修補：參考分布 " + dist_str([labels[c][d]["detail"].get("repaired") for c in el], "cat") + "；生成分布 " + dist_str([EXT[arm][c][d]["detail"].get("repaired") for c in el], "cat") + "。")

# fracture acute / sites / compression
P("\n### 骨折：急性/陳舊、部位、椎體壓迫性\n")
d = "fracture"
for arm in ARMS:
    el = AR[arm][(d, "acute")]["el"]
    P(f"\n{arm} 急性/陳舊混淆表（參考有寫 {len(el)}）：")
    conf(arm, d, "acute", el, ["yes", "no", "mixed"])
AC_OLD = r"\b(old|chronic|healed|remote|age[- ]indeterminate|subacute|stable|unchanged|prior)\b"
AC_NEW = r"\b(acute|new|recent|nondisplaced|displaced|comminuted)\b"
P("\n| arm | 皆陽性 | 參考目標句含陳舊類詞 | 參考目標句含急性類詞 | 生成句含陳舊類詞 | 生成句含急性類詞 |")
P("|---|---|---|---|---|---|")
for arm in ARMS:
    b = both(arm, d)
    P(f"| {arm} | {len(b)} | {sum(bool(re.search(AC_OLD, tsent(c, d))) for c in b)} | {sum(bool(re.search(AC_NEW, tsent(c, d))) for c in b)} | {sum(bool(re.search(AC_OLD, sent(arm, c, d))) for c in b)} | {sum(bool(re.search(AC_NEW, sent(arm, c, d))) for c in b)} |")
P("\n部位（逐部位，分母 = 皆陽性且參考有寫部位）：\n")
P("| arm | 部位 | 參考有 | 生成有 | 兩者皆有 | 召回 | 精確 |")
P("|---|---|---|---|---|---|---|")
SZ = {"vertebra": "椎體", "rib": "肋骨", "sacrum": "薦骨", "pelvis": "骨盆", "hip_femur": "髖/股骨", "other": "其他"}
for arm in ("A+", "T_gt"):
    el = AR[arm][(d, "sites")]["el"]
    for s in SZ:
        r_ = [c for c in el if s in (labels[c][d]["detail"].get("sites") or [])]
        g_ = [c for c in el if s in (EXT[arm][c][d]["detail"].get("sites") or [])]
        tp = len(set(r_) & set(g_))
        P(f"| {arm} | {SZ[s]} | {len(r_)} | {len(g_)} | {tp} | {pct(tp, len(r_))} | {pct(tp, len(g_))} |")
el = AR["A+"][(d, "sites")]["el"]
k = Counter("+".join(sorted(labels[c][d]["detail"].get("sites"))) for c in el)
P("\n參考部位集合分布（A+ 分母）：" + "、".join(f"{a} {n}" for a, n in k.most_common()) + "。")
k = Counter("+".join(sorted(EXT["A+"][c][d]["detail"].get("sites") or ["未寫"])) for c in el)
P("A+ 生成部位集合分布：" + "、".join(f"{a} {n}" for a, n in k.most_common()) + "。")
nv = [c for c in el if "vertebra" not in labels[c][d]["detail"].get("sites")]
P(f"參考不含椎體的 {len(nv)} 例：A+ 生成集合完全正確 {sum(set(EXT['A+'][c][d]['detail'].get('sites') or []) == set(labels[c][d]['detail'].get('sites')) for c in nv)}；生成寫了椎體 {sum('vertebra' in (EXT['A+'][c][d]['detail'].get('sites') or []) for c in nv)}；prompt 有椎體分數過門檻 {sum(rule('A+', c, d, 'sites')[1] is not None for c in nv)}。")
vv = [c for c in el if set(labels[c][d]["detail"].get("sites")) == {"vertebra"}]
P(f"參考只有椎體的 {len(vv)} 例：A+ 完全正確 {sum(set(EXT['A+'][c][d]['detail'].get('sites') or []) == {'vertebra'} for c in vv)}；prompt 有椎體分數過門檻 {sum(rule('A+', c, d, 'sites')[1] is not None for c in vv)}。")
P("\n椎體壓迫性混淆表（A+ 生成）：")
el = AR["A+"][(d, "vertebral_compression")]["el"]
conf("A+", d, "vertebral_compression", el, ["yes", "no"])
P("\n椎體壓迫性混淆表（規則：任一節 ≥ .24）：")
conf("A+", d, "vertebral_compression", el, ["yes", "no"], rulev=True)
k = Counter((rule("A+", c, d, "vertebral_compression")[1], EXT["A+"][c][d]["detail"].get("vertebral_compression")) for c in genpos("A+", d))
P("\n生成骨折陽性的全部病例，規則 × 生成：" + "；".join(f"規則 {a}／生成 {b} {n}" for (a, b), n in sorted(k.items(), key=lambda t: -t[1])) + "。")

# remaining yes/no attributes: confusion for A+
P("\n### 其餘屬性的混淆表（A+ 生成）\n")
for d, f in (("abdominal_aortic_aneurysm", "repaired"), ("biliary_ductal_dilation", "site"), ("ascites", "hemorrhagic"), ("appendicitis", "perforated"),
             ("appendicitis", "appendicolith"), ("free_air", "postoperative"), ("gallstones", "calcified"), ("atherosclerosis", "aortoiliac")):
    el = AR["A+"][(d, f)]["el"]
    P(f"\n{ZH[d]}／{AZ[f]}（n={len(el)}）：")
    conf("A+", d, f, el)
# CBD: gen sentence has number?
d = "biliary_ductal_dilation"
b = both("A+", d)
P(f"\n膽管擴張皆陽性 {len(b)} 例：參考目標句含 mm/cm 數值 {sum(bool(re.search(r'\d+(\.\d+)?\s*(mm|cm)', tsent(c, d))) for c in b)}；A+ 生成句含數值 {sum(bool(re.search(r'\d+(\.\d+)?\s*(mm|cm)', sent('A+', c, d))) for c in b)}；"
  f"參考目標句含 intrahepatic {sum('intrahepatic' in tsent(c, d) for c in b)}、含 extrahepatic/common bile duct/CBD {sum(bool(re.search(r'extrahepatic|common bile duct|common duct|cbd', tsent(c, d))) for c in b)}；"
  f"A+ 生成句含 intrahepatic {sum('intrahepatic' in sent('A+', c, d) for c in b)}、含 extrahepatic/common bile duct/CBD {sum(bool(re.search(r'extrahepatic|common bile duct|common duct|cbd', sent('A+', c, d))) for c in b)}。")

# ---- 5 usable training data
P("\n## 5. 可用訓練資料（train 15,306 例；Qwen 參考標註；疾病陽性且參考有寫該屬性）與 full val 可評估例數\n")
P("| 疾病 | 屬性 | train 疾病陽性 | train 有寫屬性 | train 類別分布 | full val 疾病陽性 | full val 有寫屬性 |")
P("|---|---|---|---|---|---|---|")
for d, f, kind, arg in E.ATTRS:
    if kind == "levels":
        continue
    npos, st = TRD[(d, f)]
    vp, vst = train_dist(lab_val, d, f)
    if kind == "num":
        v = sorted(float(x) for x in st)
        ds = f"中位數 {statistics.median(v):.1f}，四分位 {v[len(v) // 4]:.1f}–{v[3 * len(v) // 4]:.1f}" if v else "—"
    else:
        ds = dist_str(st, kind)
    P(f"| {ZH[d]} | {AZ[f]} | {npos} | {len(st)} | {ds} | {vp} | {len(vst)} |")
P(f"\ntrain 標註病例 {len(lab_tr)}、full val 標註病例 {len(lab_val)}、val600 {len(labels)}。")
print("\n".join(out))
