#!/usr/bin/env python3
"""Programmatic checks + status agreement vs disease14_qwen.csv for std14 outputs; optional side-by-side dump.

Usage: python check_std14.py --jsonl std14.jsonl [--side side_by_side.txt] [--prefix out/val_v1]
Writes <prefix>_summary.json, <prefix>_agreement.csv, <prefix>_disagreements.csv, <prefix>_issues.txt.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import pandas as pd

from std14_lib import CAT2STATUS, DISEASES, check_one, postproc, render

DATA = "/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data"
OUT = os.path.join(DATA, "report_gen", "standardize")


def load(path):
    best = {}
    for l in open(path):
        try:
            j = json.loads(l)
        except Exception:  # noqa: BLE001
            continue
        if j.get("parse_ok") or j["exam_id"] not in best:
            best[j["exam_id"]] = j
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, nargs="+")
    ap.add_argument("--side")
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()

    rep = json.load(open(os.path.join(OUT, "reports_std_in.json")))
    lab = pd.read_csv(os.path.join(DATA, "disease14_qwen.csv")).set_index("EXAMID")
    recs = {}
    for p in a.jsonl:
        recs.update(load(p))
    tot, issues, rows = Counter(), [], []
    n_ok = sum(bool(r.get("parse_ok")) for r in recs.values())
    side = open(a.side, "w") if a.side else None
    for eid, r in sorted(recs.items(), key=lambda kv: kv[1]["row"]):
        if not r.get("parse_ok"):
            issues.append(f"{eid}: PARSE FAIL finish={r.get('finish')}")
            continue
        ans = postproc(r["answer"])
        c, iss = check_one(rep[eid]["report"], ans)
        tot.update(c)
        issues += [f"{eid} [{r['loc']}] {x}" for x in iss]
        for d in DISEASES:
            cat = lab.at[eid, d + "_cat"] if eid in lab.index else None
            rows.append(dict(exam_id=eid, loc=r["loc"], disease=d, d14_cat=cat,
                             d14_status=CAT2STATUS.get(cat), std_status=ans["diseases"][d]["status"],
                             comparison=ans["diseases"][d]["comparison"],
                             source=" || ".join(ans["diseases"][d]["source_sentences"]),
                             normalized=ans["diseases"][d]["normalized_sentence"]))
        if side:
            side.write(f"\n{'=' * 100}\n{eid}  {r['loc']}\n{'-' * 40} ORIGINAL\n{rep[eid]['report'].strip()}\n"
                       f"{'-' * 40} RENDERED\n{render(ans)}\n{'-' * 40} PER DISEASE\n")
            for d in DISEASES:
                x = ans["diseases"][d]
                if x["status"] == "not_mentioned":
                    continue
                cat = lab.at[eid, d + "_cat"] if eid in lab.index else "?"
                flag = "" if CAT2STATUS.get(cat) == x["status"] else f"   <<< d14={cat}"
                side.write(f"  {d}: {x['status']} / cmp={x['comparison']} / {json.dumps(x['attributes'])}{flag}\n"
                           f"     src: {x['source_sentences']}\n     norm: {x['normalized_sentence']}\n")
            miss = [d for d in DISEASES if ans["diseases"][d]["status"] == "not_mentioned"
                    and lab.at[eid, d + "_cat"] != "not_mentioned"]
            if miss:
                side.write(f"  not_mentioned here but d14 says: {[(d, lab.at[eid, d + '_cat']) for d in miss]}\n")
    df = pd.DataFrame(rows)
    df.to_csv(a.prefix + "_rows.csv", index=False)
    # agreement per disease
    ag = []
    for d, g in df.groupby("disease", sort=False):
        g = g[g.d14_status.notna()]
        b_std = g.std_status.eq("positive")
        b_d14 = g.d14_status.eq("positive")
        ag.append(dict(disease=d, n=len(g), agree4=round((g.std_status == g.d14_status).mean(), 4),
                       agree_bin=round((b_std == b_d14).mean(), 4),
                       pos_d14=int(b_d14.sum()), pos_std=int(b_std.sum()),
                       pos_both=int((b_std & b_d14).sum()),
                       d14pos_std_not=int((b_d14 & ~b_std).sum()), std_pos_d14_not=int((b_std & ~b_d14).sum()),
                       mentioned_agree=round(((g.std_status == "not_mentioned") ==
                                              (g.d14_status == "not_mentioned")).mean(), 4)))
    ag = pd.DataFrame(ag)
    ag.to_csv(a.prefix + "_agreement.csv", index=False)
    df[df.std_status != df.d14_status].to_csv(a.prefix + "_disagreements.csv", index=False)
    conf = pd.crosstab(df.d14_status, df.std_status)
    summ = dict(n=len(recs), parse_ok=n_ok,
                src_verbatim_rate=tot["src_verbatim"] / max(tot["src_total"], 1),
                size_kept_rate=tot["size_kept"] / max(tot["size_src"], 1),
                size_norm_in_report_rate=tot["size_norm_ok"] / max(tot["size_norm"], 1),
                seim_kept_rate=tot["seim_kept"] / max(tot["seim_src"], 1),
                seim_norm_in_report_rate=tot["seim_norm_ok"] / max(tot["seim_norm"], 1),
                **dict(tot))
    json.dump(summ, open(a.prefix + "_summary.json", "w"), indent=1)
    open(a.prefix + "_issues.txt", "w").write("\n".join(issues) + "\n")
    print(json.dumps(summ, indent=1))
    print(ag.to_string(index=False))
    print(conf)
    kinds = Counter(i.split("] ", 1)[-1].split(":")[1].strip().split(" ")[0] if "] " in i else "parse" for i in issues)
    print(len(issues), "issues", kinds.most_common(10))
    _ = defaultdict


if __name__ == "__main__":
    main()
