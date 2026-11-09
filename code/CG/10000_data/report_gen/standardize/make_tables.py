#!/usr/bin/env python3
"""Markdown tables (Chinese disease names) for standardize_report.md from check_std14.py outputs.

Usage: python make_tables.py --prefix full/full > full/tables.md
"""
import argparse
import json

import pandas as pd

ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊切除術後", "splenomegaly": "脾腫大",
      "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤", "atherosclerosis": "動脈粥狀硬化",
      "osteopenia": "骨質減少", "bowel_obstruction": "腸阻塞", "appendicitis": "闌尾炎", "ascites": "腹水",
      "free_air": "腹腔游離氣體", "gallstones": "膽結石", "biliary_ductal_dilation": "膽管擴張", "fracture": "骨折"}
ST = {"positive": "陽性", "uncertain": "不確定", "negative": "陰性", "not_mentioned": "未提及"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    ag = pd.read_csv(a.prefix + "_agreement.csv")
    rows = pd.read_csv(a.prefix + "_rows.csv")
    summ = json.load(open(a.prefix + "_summary.json"))
    print("### 各類一致率（標準化 status vs disease14_qwen.csv）\n")
    print("| 疾病 | 四類一致率 | 二元一致率 | 原標籤陽性數 | 標準化陽性數 | 兩者皆陽性 | 原陽性→標準化非陽性 | 標準化陽性→原非陽性 | 有無提及一致率 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for _, r in ag.iterrows():
        print(f"| {ZH[r.disease]} | {r.agree4:.3f} | {r.agree_bin:.4f} | {r.pos_d14} | {r.pos_std} | {r.pos_both} | "
              f"{r.d14pos_std_not} | {r.std_pos_d14_not} | {r.mentioned_agree:.3f} |")
    r4 = (rows.std_status == rows.d14_status).mean()
    rb = ((rows.std_status == "positive") == (rows.d14_status == "positive")).mean()
    print(f"| **全部** | **{r4:.3f}** | **{rb:.4f}** | {ag.pos_d14.sum()} | {ag.pos_std.sum()} | {ag.pos_both.sum()} | "
          f"{ag.d14pos_std_not.sum()} | {ag.std_pos_d14_not.sum()} | |\n")
    print("### 混淆矩陣（列 = 原標籤，欄 = 標準化）\n")
    ct = pd.crosstab(rows.d14_status.map(ST), rows.std_status.map(ST))
    order = [ST[k] for k in ST if ST[k] in ct.index]
    ct = ct.reindex(index=order, columns=[ST[k] for k in ST if ST[k] in ct.columns], fill_value=0)
    print("| 原 \\ 標準化 | " + " | ".join(ct.columns) + " |")
    print("|---" * (len(ct.columns) + 1) + "|")
    for i, r in ct.iterrows():
        print(f"| {i} | " + " | ".join(str(v) for v in r.values) + " |")
    print("\n### 院區別\n")
    print("| 院區 | 報告數 | 四類一致率 | 二元一致率 |")
    print("|---|---|---|---|")
    for loc, g in rows.groupby("loc"):
        print(f"| {loc} | {g.exam_id.nunique()} | {(g.std_status == g.d14_status).mean():.3f} | "
              f"{((g.std_status == 'positive') == (g.d14_status == 'positive')).mean():.4f} |")
    print("\n### 標準化結果分布\n")
    print("| 疾病 | 陽性 | 不確定 | 陰性 | 未提及 | 有比較前次 (stable/new/progression/regression) |")
    print("|---|---|---|---|---|---|")
    for d, g in rows.groupby("disease", sort=False):
        vc = g.std_status.value_counts()
        cm = g[g.std_status != "not_mentioned"].comparison.value_counts()
        cs = "/".join(str(cm.get(k, 0)) for k in ["stable", "new", "progression", "regression"])
        print(f"| {ZH[d]} | {vc.get('positive', 0)} | {vc.get('uncertain', 0)} | {vc.get('negative', 0)} | "
              f"{vc.get('not_mentioned', 0)} | {cs} |")
    print("\n### 保留率（程式檢查）\n")
    for k in ["n", "parse_ok", "src_total", "src_verbatim", "src_verbatim_rate", "size_norm", "size_norm_ok",
              "size_norm_in_report_rate", "size_src", "size_kept", "size_kept_rate", "seim_src", "seim_kept",
              "seim_kept_rate", "seim_norm", "seim_norm_ok", "seim_norm_in_report_rate", "temporal_left",
              "no_source", "empty_norm"]:
        v = summ.get(k)
        print(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")


if __name__ == "__main__":
    main()
