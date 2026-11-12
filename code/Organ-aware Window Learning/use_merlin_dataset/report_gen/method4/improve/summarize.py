#!/usr/bin/env python3
"""evaluate.py metrics.json -> one summary column (aggregate numbers only, no ids / text), the rows used in
queue_cue/RESULTS_interim.md: macro sens / spec / F1 and micro F1 (owl definition, strict), critical-error and
omission rate, natural FN / FP cue concordance, and the attribute accuracies (both-positive cases).
usage: python summarize.py <name>=<metrics.json> [...] [--per-disease] [--attr-first] [--hy-first]
--hy-first (agent I, 11-12, hydronephrosis grade cue arm): hydronephrosis grade and side rows first, then --attr-first."""
import json
import math
import sys

ROWS = [("macro 敏感度", "sens"), ("macro 特異度", "spec"), ("macro F1（owl）", "f1"), ("micro F1（owl）", "mf1"),
        ("關鍵錯誤率", "crit"), ("漏寫率", "omit"), ("關鍵錯誤＋漏寫", "critomit"),
        ("自然 FN 提示一致率", "natfn"), ("自然 FP 提示一致率", "natfp"),
        ("腎水腫側別", "attr:hydronephrosis:side"), ("骨折椎體節段", "attr:fracture:levels"),
        ("骨折部位", "attr:fracture:sites"), ("腸阻塞類型", "attr:bowel_obstruction:type"),
        ("ROUGE-L", "rouge"), ("BERTScore F", "bert")]
# --attr-first (agent G, for the attribute-focused unit-cue experiments): these attribute rows first, then the rest
ATTR_FIRST = [("腎水腫側別", "attr:hydronephrosis:side"), ("骨折椎體節段", "attr:fracture:levels"),
              ("骨折部位", "attr:fracture:sites"), ("腹主動脈瘤直徑（±0.3 cm）", "attr:abdominal_aortic_aneurysm:diameter_cm"),
              ("骨折椎體壓迫性", "attr:fracture:vertebral_compression"), ("腎水腫程度", "attr:hydronephrosis:grade"),
              # agent H (11-12, A+ grade cue): grade attributes (exact grade; both-positive cases, ref grade stated)
              ("脂肪肝程度", "attr:hepatic_steatosis:grade"), ("腹水量", "attr:ascites:amount"),
              ("動脈粥狀硬化程度", "attr:atherosclerosis:grade")]


def summarize(m):
    ow = {p["disease"]: p for p in m["presence"] if p["definition"] == "owl" and p["mode"] == "strict"}
    pr = [p for d, p in ow.items() if not d.startswith("ALL")]
    s = {"sens": ow["ALL_macro"]["sens"], "spec": ow["ALL_macro"]["spec"], "f1": ow["ALL_macro"]["F1"],
         "mf1": ow["ALL_micro"]["F1"]}
    al = next(c for c in m["critical"] if c["disease"] == "ALL")
    s["crit"] = (al["critical_rate"], al["critical"], al["ref_pos"])
    s["omit"] = (al["omission_rate"], al["omitted"], al["ref_pos"])
    s["critomit"] = ((al["critical"] + al["omitted"]) / al["ref_pos"], al["critical"] + al["omitted"], al["ref_pos"])
    ct = m.get("cue_test") or {}
    for k, key in (("natfn", "nat_fn"), ("natfp", "nat_fp")):
        n = sum(r[key].get("n", 0) for r in ct.get("rows", []) if key in r and not str(r.get("disease", "")).startswith("ALL"))
        c = sum(r[key].get("consistent", 0) for r in ct.get("rows", []) if key in r and not str(r.get("disease", "")).startswith("ALL"))
        s[k] = (c / n, c, n) if n else None
    for a in m["attributes"]:
        s[f"attr:{a['disease']}:{a['field']}"] = (a["acc"], round(a["acc"] * a["ref_stated"]) if a["acc"] == a["acc"] else 0,
                                                 a["ref_stated"])
    st = m.get("style") or {}
    s["rouge"], s["bert"] = st.get("ROUGE-L"), st.get("BERTScore_F")
    s["_per"] = {p["疾病"]: (p["sens"], p["spec"], p["F1"]) for p in pr}
    s["_crit"] = {c["疾病"]: (c["critical"], c["omitted"], c["ref_pos"]) for c in m["critical"] if c["disease"] != "ALL"}
    return s


def fmt(v, key):
    if v is None:
        return "—"
    if isinstance(v, tuple):
        r, k, n = v
        if r is None or (isinstance(r, float) and math.isnan(r)):
            return "—"
        return f"{100 * r:.1f}%（{k}/{n}）"
    return f"{v:.3f}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cols = [(a.split("=", 1)[0], summarize(json.load(open(a.split("=", 1)[1])))) for a in args]
    print("| 指標 | " + " | ".join(n for n, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    rows = ATTR_FIRST + [r for r in ROWS if r[1] not in {k for _, k in ATTR_FIRST}] if "--attr-first" in sys.argv else ROWS
    if "--hy-first" in sys.argv:
        hy = ["attr:hydronephrosis:grade", "attr:hydronephrosis:side"]
        rows = ATTR_FIRST + [r for r in ROWS if r[1] not in {k for _, k in ATTR_FIRST}]
        rows = sorted(rows, key=lambda r: hy.index(r[1]) if r[1] in hy else len(hy))      # stable: the rest keeps its order
    for label, key in rows:
        print(f"| {label} | " + " | ".join(fmt(s.get(key), key) for _, s in cols) + " |")
    if "--per-disease" in sys.argv:
        print("\n| 疾病 | " + " | ".join(f"{n} 敏感度/特異度/F1/關鍵錯誤" for n, _ in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for d in cols[0][1]["_per"]:
            print(f"| {d} | " + " | ".join(
                "{:.2f}/{:.2f}/{:.2f}/{}/{}".format(*s["_per"][d], s["_crit"][d][0], s["_crit"][d][2]) for _, s in cols) + " |")


if __name__ == "__main__":
    main()
