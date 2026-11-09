#!/usr/bin/env python3
"""Phase 3 target statistics on the (partial or final) std14 output.

usage: python p3_target_stats.py --std <reports_std.jsonl | std14 shard jsonl ...>
Writes into data/.../report_gen/phase3/:
  typo_dictionary.csv     corpus-frequency typo candidates in target sentences (typo, fix, counts, auto, reviewed)
  target_stats.json       counters (temporal stripped/dropped, imp_only, typos, out-of-scope content, ...)
  attr_presence.csv       per disease: n positive/uncertain, % with each attribute stated, comparison mix
  temporal_examples.csv   sample of stripped / dropped / fallback sentences
  style_phrasings.csv     per LOC x disease x polarity: top doctor phrasings and their share
  template_families.csv   per LOC: report layout signature families (radiologist-style proxy)
  loc_scanner.csv         LOC x scanner crosstab (all 10k)
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "standardize"))
from p3_targets_lib import DISEASES, IMP_HDR, build_target, load_typo_fix  # noqa: E402

ROOT = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data")
STD = ROOT / "report_gen/standardize"
P3 = ROOT / "report_gen/phase3"
WORD = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def load_std(paths):
    from std14_lib import postproc
    best = {}
    for p in paths:
        for line in open(p):
            try:
                j = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "answer" in j:                       # raw shard row
                if not (j.get("parse_ok") or j["exam_id"] not in best):
                    continue
                if not j.get("parse_ok"):
                    continue
                best[j["exam_id"]] = postproc(j["answer"])["diseases"]
            else:                                   # finalized reports_std.jsonl row (already postproc'd)
                best[j["exam_id"]] = j["diseases"]
    return best


def edits1(w):
    letters = "abcdefghijklmnopqrstuvwxyz"
    sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    return ({a + b[1:] for a, b in sp if b} | {a + b[1] + b[0] + b[2:] for a, b in sp if len(b) > 1}
            | {a + c + b[1:] for a, b in sp if b for c in letters} | {a + c + b for a, b in sp for c in letters})


def build_typo_dict(rep, std):
    freq = collections.Counter(w.lower() for r in rep.values() for w in WORD.findall(r["report"]))
    cand = collections.Counter()
    for e, ds in std.items():
        for d in DISEASES:
            for s in ds[d]["source_sentences"] if ds[d]["status"] != "not_mentioned" else []:
                for w in WORD.findall(s):
                    lw = w.lower()
                    if len(lw) >= 4 and freq[lw] <= 5:
                        cand[lw] += 1
    rows = []
    for w, n in cand.items():
        best = max(((freq[x], x) for x in edits1(w) if freq[x] >= 30 and x != w), default=None)
        if best:
            ratio = best[0] / max(freq[w], 1)
            plural = w in (best[1] + "s", best[1] + "es") or best[1] in (w + "s", w + "es")
            auto = int(ratio >= 50 and len(w) >= 5 and not plural)
            rows.append(dict(typo=w, fix=best[1], n_in_targets=n, corpus_typo=freq[w], corpus_fix=best[0],
                             auto=auto, reviewed=auto))
    return pd.DataFrame(rows).sort_values("n_in_targets", ascending=False)


def layout_signature(text):
    """coarse report-layout signature: header keywords present + bullet style."""
    t = text
    keys = []
    for k, pat in [("Indications", r"(?im)^\s*(indications?|clinical information|brief hx|history)\s*[:：]"),
                   ("Techniques", r"(?im)^\s*techniques?\s*[:：]"), ("Comparison", r"(?im)^\s*comparison"),
                   ("Findings", r"(?im)^\s*(imaging\s+)?findings?\s*[:：]"), ("INDEX", r"<\s*INDEX"),
                   ("Major/Other", r"(?im)^\s*(major|other)\s+findings"), ("Organ-sections", r"(?im)^\s*(liver|kidneys?|bones?|chest|abdomen|neck)\s*[:：]"),
                   ("Shows", r"(?i)\bshows\s*:"), ("CANCER STATUS", r"CANCER STATUS"), ("Staging", r"(?i)tumor invasion|AJCC"),
                   ("IMP", IMP_HDR.pattern)]:
        if re.search(pat, t):
            keys.append(k)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    num = sum(bool(re.match(r"^\(?\d{1,2}[\.\)]", ln)) for ln in lines)
    star = sum(ln.startswith(("*", "-", "•")) for ln in lines)
    style = "numbered" if num >= 3 else "bullets" if star >= 3 else "plain"
    return "+".join(keys) + "|" + style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std", nargs="+", required=True)
    a = ap.parse_args()
    P3.mkdir(parents=True, exist_ok=True)
    rep = json.load(open(STD / "reports_std_in.json"))
    std = load_std(a.std)
    print("std reports", len(std))

    td = build_typo_dict(rep, std)
    td.to_csv(P3 / "typo_dictionary.csv", index=False)
    fix = load_typo_fix(P3 / "typo_dictionary.csv")

    C = collections.Counter()
    lines_by_loc = collections.defaultdict(list)
    ex_rows, phr = [], collections.Counter()
    n_lines_typo = 0
    typo_words = set(td.typo)
    for e, ds in std.items():
        r = rep[e]
        t = build_target(r["report"], ds, fix)
        C.update(t["counters"])
        C["exams"] += 1
        C["exams_empty_target"] += int(not t["lines"])
        lines_by_loc[r["LOC"]].append(len(t["lines"]))
        for ln in t["lines"]:
            C["lines"] += 1
            C["lines_oos"] += ln["oos"]
            C["lines_size"] += ln["size"]
            C["lines_seim"] += ln["seim"]
            C["lines_multi_disease"] += len(ln["diseases"]) > 1
            has_typo = any(w.lower() in typo_words for w in WORD.findall(ln["orig"] or ""))
            n_lines_typo += has_typo
            if ln["action"] != "keep":
                ex_rows.append(dict(exam_id=e, loc=r["LOC"], action=ln["action"], orig=ln["orig"], target=ln["text"]))
            for d in ln["diseases"]:
                st = ds[d]["status"]
                pol = "pos" if st in ("positive", "uncertain") else "neg"
                phr[(r["LOC"], d, pol, ln["text"].strip().rstrip(".").strip().lower())] += 1
        for d in DISEASES:   # dropped sentences
            pass
    # dropped examples
    for e, ds in list(std.items())[:3000]:
        for d in DISEASES:
            for s in ds[d]["source_sentences"] if ds[d]["status"] != "not_mentioned" else []:
                from p3_targets_lib import strip_temporal, clean_bullet
                tt, act = strip_temporal(clean_bullet(s))
                if act == "drop":
                    ex_rows.append(dict(exam_id=e, loc=rep[e]["LOC"], action="drop", orig=s, target=""))
    C["lines_with_typo_candidate"] = n_lines_typo
    ex = pd.DataFrame(ex_rows).drop_duplicates(["action", "orig"])
    ex.groupby("action").head(40).to_csv(P3 / "temporal_examples.csv", index=False)

    # attribute presence
    rows = []
    for d in DISEASES:
        xs = [ds[d] for ds in std.values() if ds[d]["status"] in ("positive", "uncertain")]
        r = {"disease": d, "n_pos_unc": len(xs),
             "n_neg": sum(ds[d]["status"] == "negative" for ds in std.values()),
             "n_not_mentioned": sum(ds[d]["status"] == "not_mentioned" for ds in std.values())}
        keys = list(xs[0]["attributes"].keys()) if xs else []
        for k in keys:
            vals = [x["attributes"].get(k) for x in xs]
            stated = [v for v in vals if v not in (None, "", "not_stated", [], 0) and v != "not_stated"]
            r[f"{k}_stated_pct"] = round(100 * len(stated) / max(len(xs), 1), 1)
            vc = collections.Counter(str(v) if not isinstance(v, (int, float)) else "number" for v in stated)
            r[f"{k}_values"] = "; ".join(f"{v}:{n}" for v, n in vc.most_common(6))
        cmpc = collections.Counter(x["comparison"] for x in xs)
        r["comparison"] = "; ".join(f"{k}:{v}" for k, v in cmpc.most_common())
        rows.append(r)
    pd.DataFrame(rows).to_csv(P3 / "attr_presence.csv", index=False)

    # style phrasings: share of top-3 phrasings per LOC x disease x polarity
    srows = []
    grp = collections.defaultdict(list)
    for (loc, d, pol, txt), n in phr.items():
        grp[(loc, d, pol)].append((n, txt))
    for (loc, d, pol), lst in sorted(grp.items()):
        lst.sort(reverse=True)
        tot = sum(n for n, _ in lst)
        srows.append(dict(loc=loc, disease=d, polarity=pol, n=tot, n_distinct=len(lst),
                          top1_share=round(lst[0][0] / tot, 3), top3_share=round(sum(n for n, _ in lst[:3]) / tot, 3),
                          top1=lst[0][1], top2=lst[1][1] if len(lst) > 1 else "", top3=lst[2][1] if len(lst) > 2 else ""))
    pd.DataFrame(srows).to_csv(P3 / "style_phrasings.csv", index=False)

    # layout families (all 10k reports) + scanner crosstab
    raw = pd.read_csv(ROOT / "deliver_10k_reports_v1(in).csv", encoding="cp1252")
    raw.columns = ["PID", "EXAMID", "EDATE", "LOC", "scanner", "Report"]
    raw["sig"] = raw.Report.astype(str).map(layout_signature)
    fam = raw.groupby(["LOC", "sig"]).size().rename("n").reset_index().sort_values(["LOC", "n"], ascending=[True, False])
    fam["share"] = fam.n / fam.groupby("LOC").n.transform("sum")
    fam.to_csv(P3 / "template_families.csv", index=False)
    pd.crosstab(raw.LOC, raw.scanner).to_csv(P3 / "loc_scanner.csv")

    stats = dict(C)
    stats["lines_per_exam_by_loc"] = {k: round(sum(v) / len(v), 2) for k, v in lines_by_loc.items()}
    stats["n_typo_candidates"] = int(len(td))
    stats["n_typo_auto"] = int(td.auto.sum()) if len(td) else 0
    json.dump(stats, open(P3 / "target_stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    print(pd.DataFrame(rows)[["disease", "n_pos_unc", "n_neg"] + [c for c in pd.DataFrame(rows).columns if c.endswith("_pct")]].to_string())
    top = fam.groupby("LOC").head(4)
    print(top.to_string())


if __name__ == "__main__":
    main()
