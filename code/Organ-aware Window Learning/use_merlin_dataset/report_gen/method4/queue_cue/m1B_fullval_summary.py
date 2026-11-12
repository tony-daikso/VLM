"""Aggregate-only side-by-side table from evaluate.py metrics.json files (no text, no ids).
usage: python m1B_fullval_summary.py name1=path/metrics.json [name2=path/metrics.json ...]
Prints: macro sens / spec / F1 and micro F1 (owl, strict), critical / omission rates, style metrics, per-class F1 (owl strict).
"""
import json
import sys

ms = [(a.split("=", 1)[0], json.load(open(a.split("=", 1)[1]))) for a in sys.argv[1:]]


def pres(m, d):
    return next(r for r in m["presence"] if r["definition"] == "owl" and r["mode"] == "strict" and r["disease"] == d)


def crit(m, d="ALL"):
    return next(r for r in m["critical"] if r["disease"] == d)


hdr = "| 指標 | " + " | ".join(n for n, _ in ms) + " |"
print(hdr + "\n|---|" + "---|" * len(ms))
rows = [("病例數", lambda m: str(m["n_cases"])),
        ("Qwen 反抽失敗", lambda m: str(m["qwen"]["n_failed_or_unlabelled"])),
        ("macro 敏感度", lambda m: f'{pres(m, "ALL_macro")["sens"]:.3f}'),
        ("macro 特異度", lambda m: f'{pres(m, "ALL_macro")["spec"]:.3f}'),
        ("macro F1（owl）", lambda m: f'{pres(m, "ALL_macro")["F1"]:.3f}'),
        ("micro F1（owl）", lambda m: f'{pres(m, "ALL_micro")["F1"]:.3f}'),
        ("關鍵錯誤率", lambda m: f'{crit(m)["critical_rate"]:.1%}（{crit(m)["critical"]}/{crit(m)["ref_pos"]}）'),
        ("漏寫率", lambda m: f'{crit(m)["omission_rate"]:.1%}（{crit(m)["omitted"]}/{crit(m)["ref_pos"]}）'),
        ("關鍵錯誤＋漏寫", lambda m: f'{(crit(m)["critical"] + crit(m)["omitted"]) / crit(m)["ref_pos"]:.1%}'),
        ("BLEU-4", lambda m: f'{m["style"]["BLEU-4"]:.3f}' if "style" in m else "—"),
        ("ROUGE-L", lambda m: f'{m["style"]["ROUGE-L"]:.3f}' if "style" in m else "—"),
        ("BERTScore F", lambda m: f'{m["style"]["BERTScore_F"]:.3f}' if "BERTScore_F" in m.get("style", {}) else "—")]
for name, f in rows:
    print(f"| {name} | " + " | ".join(f(m) for _, m in ms) + " |")
print()
print("| 疾病（F1，owl 嚴格） | " + " | ".join(n for n, _ in ms) + " |\n|---|" + "---|" * len(ms))
for r in ms[0][1]["presence"]:
    if r["definition"] == "owl" and r["mode"] == "strict" and not r["disease"].startswith("ALL"):
        d = r["disease"]
        print(f'| {r["疾病"]} | ' + " | ".join(f'{pres(m, d)["F1"]:.3f}' for _, m in ms) + " |")
print()
print("| 疾病（漏寫率） | " + " | ".join(n for n, _ in ms) + " |\n|---|" + "---|" * len(ms))
for r in ms[0][1]["critical"]:
    if r["disease"] != "ALL":
        d = r["disease"]
        print(f'| {r["疾病"]} | ' + " | ".join(f'{crit(m, d)["omission_rate"]:.1%}' for _, m in ms) + " |")
