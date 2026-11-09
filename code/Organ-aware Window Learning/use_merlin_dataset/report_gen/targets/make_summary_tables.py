#!/usr/bin/env python3
"""stats.json (from build_targets.py) -> markdown tables (Traditional Chinese disease names) for targets_summary.md.
usage: python make_summary_tables.py <stats.json>  > tables.md"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_lib as L

S = json.load(open(sys.argv[1]))
P = lambda a, b: f"{100 * a / b:.1f}%" if b else "-"
out = []
w = out.append

w("### A. 各 split 數量\n")
w("| split | 例數 | 目標行數平均 | 目標字元平均 | 空目標（14 類都沒提到） | 找不到器官段落標題的報告 |")
w("|---|---|---|---|---|---|")
for sp in ("train", "val", "test"):
    s = S[sp]
    w(f"| {sp} | {s['n']:,} | {s['lines_mean']:.1f} | {s['chars_mean']:.0f} | {s['empty_target']} | {s['unparsed_headers']} |")

for sp in ("train", "val"):
    s = S[sp]
    n = s["n"]
    w(f"\n### B. 最終狀態分布與和 labels_disease14 的一致率（{sp}，{n:,} 例）\n")
    w("一致率定義：規則狀態（只用規則，沒有 Qwen 補）對 labels 類別（yes/probably_yes→陽性、unclear→不確定、probably_no/no→陰性、not_mentioned）。"
      "「提及一致」= 兩邊都提到或都沒提到；「陽陰一致」= 兩邊都判陽性或陰性時方向相同的比例；「嚴格→寬鬆差」= 規則判沒提到、labels 判陰性（定義差異，見正文）。\n")
    w("| 疾病 | 陽性 | 不確定 | 陰性 | 沒提到 | 四類一致 | 提及一致 | 陽陰一致（兩邊都有判） | 規則漏掉的陽性/不確定（由 Qwen quotes 補） | 規則沒提到但 labels 陰性 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for d in L.DISEASES:
        st = s["status"][d]
        pr = {tuple(k.split("|")): v for k, v in s["pair"][d].items()}
        exact = sum(v for (r, l), v in pr.items() if r == l)
        ment = sum(v for (r, l), v in pr.items() if (r == "not_mentioned") == (l == "not_mentioned"))
        both = sum(v for (r, l), v in pr.items() if r in ("pos", "neg") and l in ("pos", "neg"))
        bagree = sum(v for (r, l), v in pr.items() if r in ("pos", "neg") and r == l)
        miss = sum(v for (r, l), v in pr.items() if r == "not_mentioned" and l in ("pos", "uncertain"))
        strict = sum(v for (r, l), v in pr.items() if r == "not_mentioned" and l == "neg")
        w(f"| {L.ZH[d]} | {st.get('pos', 0):,} | {st.get('uncertain', 0):,} | {st.get('neg', 0):,} | {st.get('not_mentioned', 0):,} | "
          f"{P(exact, n)} | {P(ment, n)} | {P(bagree, both)}（{both:,}） | {miss} | {strict:,} |")

    w(f"\n### C. 句子來源（{sp}）\n")
    w("| 疾病 | rule | rule+qwen_label | qwen_label | 衝突（規則與 labels 狀態不同，採 labels） | 規則 other（交 labels 決定） | 只有 impression 有句子 | quote 找不到 | 清理後變空 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for d in L.DISEASES:
        so, fl = s["source"][d], s["flags"][d]
        w(f"| {L.ZH[d]} | {so.get('rule', 0):,} | {so.get('rule+qwen_label', 0):,} | {so.get('qwen_label', 0):,} | {fl.get('conflict', 0):,} | "
          f"{fl.get('rule_other', 0):,} | {fl.get('impression_only', 0):,} | {fl.get('quote_not_located', 0)} | {fl.get('clean_empty', 0)} |")

    w(f"\n### D. 屬性覆蓋率（{sp}，陽性＋不確定的例子；括號內含只在原句、清理後被拿掉的尺寸）\n")
    w("| 疾病 | 陽性+不確定 | 屬性有寫的比例 |")
    w("|---|---|---|")
    for d in L.DISEASES:
        nn = s["attr_n"].get(d, 0)
        a = s["attr"][d]
        parts = []
        for k in L.ATTR_KEYS[d]:
            x, y = a.get(k, 0), a.get(k + "_incl_raw", 0)
            parts.append(f"{k} {P(x, nn)}" + (f"（{P(y, nn)}）" if y != x else ""))
        w(f"| {L.ZH[d]} | {nn:,} | {'；'.join(parts) or '（無屬性）'} |")

    w(f"\n### E. 屬性和 labels detail 的一致率（{sp}，陽性＋不確定、labels 有寫該屬性的例子）\n")
    w("| 疾病 | 屬性 | labels 有寫 | 兩邊都有寫 | 一致（數值 ±1 mm） |")
    w("|---|---|---|---|---|")
    for d in L.DISEASES:
        ag = s["attr_agree"][d]
        for k in sorted({x.rsplit("_", 2)[0] if x.endswith("_label_stated") else None for x in ag} - {None}):
            ls_, b, g = ag.get(k + "_label_stated", 0), ag.get(k + "_both", 0), ag.get(k + "_agree", 0)
            w(f"| {L.ZH[d]} | {k} | {ls_:,} | {b:,}（{P(b, ls_)}） | {g:,}（{P(g, b)}） |")

for sp in ("train", "val", "test"):
    s = S[sp]
    w(f"\n### F. 清理統計（{sp}，以句為單位；一句可能同時算到多類）\n")
    keys = ["seim", "cmp_clause", "keep_word", "change_word_stripped", "change_clause_dropped", "change_sentence_dropped",
            "size_removed", "size_kept", "typo", "broken_dropped", "qwen_rewrite", "qwen_empty", "degraded_no_qwen"]
    names = ["se/im 刪除", "比較子句刪除", "stable 類字詞刪除", "變化字詞刪除（new/increased…）", "變化子句刪除", "整句刪除（無法只刪子句）",
             "無測量依據尺寸刪除", "主動脈/脾臟尺寸保留", "錯字修正", "刪完變殘句而丟掉", "Qwen 改寫", "Qwen 判定無可保留內容", "規則清理不佳（陽性，未送 Qwen）"]
    w("| 疾病 | " + " | ".join(names) + " |")
    w("|---|" + "---|" * len(names))
    for d in L.DISEASES:
        c = s["clean"][d]
        w(f"| {L.ZH[d]} | " + " | ".join(f"{c.get(k, 0):,}" for k in keys) + " |")
    ct = s["clean_total"]
    w("| **全部（不重複句）** | " + " | ".join(f"{ct.get(k, 0):,}" for k in keys) + " |")
print("\n".join(out))
