#!/usr/bin/env python3
"""Typo dictionary for target sentences, built from train+val report word frequencies only.

A word is a candidate typo when it occurs <= MAX_TYPO times in the train+val corpus, is >= 5 letters, and is one edit
away from a word that occurs >= MIN_FIX times and >= RATIO times more often (plural / hyphen variants excluded).
Output (data side only, never in the code folder): <data>/typo_dictionary.csv with columns
typo, fix, corpus_typo, corpus_fix, auto. Manual review is recorded in <data>/typo_review.csv (typo, keep 0/1);
rows with keep=0 are not applied.
usage: python build_typo_dict.py --reports <reports.json> --out <data_dir>
"""
import argparse, collections, csv, json, re

MAX_TYPO, MIN_FIX, RATIO = 3, 30, 50
LET = "abcdefghijklmnopqrstuvwxyz"


def edits1(w):
    sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    out = set()
    for a, b in sp:
        if b:
            out.add(a + b[1:])
            for c in LET:
                out.add(a + c + b[1:])
        if len(b) > 1:
            out.add(a + b[1] + b[0] + b[2:])
        for c in LET:
            out.add(a + c + b)
    out.discard(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reps = json.load(open(a.reports))
    freq = collections.Counter()
    for v in reps.values():
        if v["split"] in ("train", "val"):
            freq.update(w.lower() for w in re.findall(r"[A-Za-z]+", v["report"]))
    rows = []
    for w, n in freq.items():
        if n > MAX_TYPO or len(w) < 5:
            continue
        best = None
        for e in edits1(w):
            m = freq.get(e, 0)
            if m >= MIN_FIX and m >= RATIO * n and (best is None or m > freq[best]):
                best = e
        if not best:
            continue
        if best in (w + "s", w[:-1]) or w in (best + "s", best[:-1]) or best + "es" == w or w + "es" == best:
            continue
        rows.append(dict(typo=w, fix=best, corpus_typo=n, corpus_fix=freq[best], auto=1))
    rows.sort(key=lambda r: -r["corpus_fix"])
    with open(f"{a.out}/typo_dictionary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(len(rows), "candidates")


if __name__ == "__main__":
    main()
