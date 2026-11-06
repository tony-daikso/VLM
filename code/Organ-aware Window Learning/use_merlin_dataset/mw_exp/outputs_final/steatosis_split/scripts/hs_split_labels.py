"""脂肪肝拆分標籤：瀰漫性脂肪肝 (y_diffuse) vs 局部脂肪浸潤 (y_focal).

Deterministic rules on report text + the existing Qwen hepatic_steatosis label (never model scores).
Rules are documented in label_rules.md (same folder). Locked 2026-11-05 before any test use.

Split safety:
  * split.csv is read first; only study_ids whose split is in --splits are kept.
  * labels jsonl: the study_id is regex-extracted from each raw line and checked against the allowed set
    BEFORE the line is json-decoded.
  * reports.json (one big dict): a key-gated scanner json-decodes only values whose key is allowed; other
    values are skipped by a bracket/string tokenizer without being decoded or stored.
  * 'test' is refused unless --allow-test is given (only after the user locks the test run).
Output: one row per study with keyword flags and labels only (no report text).

Usage:
  python hs_split_labels.py --out labels_trval_split.csv [--splits train,val]
  python hs_split_labels.py --out labels_test_split.csv --splits test --allow-test      # later, once locked
"""
import argparse, json, re, sys
import numpy as np, pandas as pd

R = "/home/advantech/Desktop/Tony/VLM/"
SPLIT = R + "OWL/data/split.csv"
LABELS = R + "OWL/data/labels_disease14_all.jsonl"
REPORTS = R + "merlin_disease14/reports.json"
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}

# ---------------------------------------------------------------- text rules (fixed)
LIVER_CTX = re.compile(r"liver|hepat|steato|falciform|gallbladder fossa|porta hepatis|periportal|ligamentum|fissure|"
                       r"segment (?:4|iv)|caudate|quadrate|(?:left|right) lobe|fatty infiltrat|fatty replace|fat deposition|focal fat", re.I)
OTHER_ORGAN = re.compile(r"pancrea|kidney|renal|adrenal|bone|marrow|vertebra|muscle|omentum|mesenter|hernia|bowel|colon|"
                         r"breast|lipoma|myelolipoma|angiomyolipoma|epiploic|ileum|jejun|duoden|stomach|append|cecum|strand", re.I)
NONLIVER_SPAN = re.compile(r"strand|ileum|jejun|duoden|bowel|colon|cecum|append|mesenter|omentum|pancrea|kidney|renal|hernia", re.I)
HISTORY = re.compile(r"history of|\bindication|clinical (?:history|information|indication)|reason for (?:exam|study)|"
                     r"evaluate for|\beval for|rule out|\br/o\b|^\s*(?:history|clinical|indication|reason|comparison|technique)\b", re.I)
FAT = r"(?:steato\w*|fatty|fat|hypoattenuat\w*|hypodens\w*|(?:decreased|diminished|low|lower|reduced)(?: in)? (?:attenuation|density))"
SPARING = re.compile(r"\bspar(?:ing|ed)\b", re.I)
# focal fatty infiltration / deposition (phrase spans)
FOCAL = re.compile(
    r"(?:\bfocal(?:ly)?|\bgeographic|\bmultifocal|\bnodular|\bregional|\bpatchy)\W+(?:\w+\W+){0,3}?" + FAT +
    r"|\b(?:area|region|zone|focus|foci)s?\s+of\s+(?:\w+\W+){0,2}?" + FAT +
    r"|\b(?:fat|fatty|steatosis|infiltration|deposition)\W+(?:\w+\W+){0,9}?(?:falciform|gallbladder fossa|porta hepatis|"
    r"periportal|fissure|ligamentum\w*|segment (?:4|iv)\w*)"
    r"|\b(?:falciform|gallbladder fossa|porta hepatis|periportal|fissure|ligamentum\w*|segment (?:4|iv)\w*)\W+(?:\w+\W+){0,6}?"
    r"(?:fat|fatty|steatosis|infiltration|deposition)", re.I)
# diffuse / generalized steatosis (checked AFTER removing focal phrase spans from the sentence)
DIFFUSE = re.compile(
    r"steato\w*|fatty (?:liver|change|infiltrat\w*|replacement|deposition|liver disease|in (?:appearance|attenuation))"
    r"|fatty[- ]infiltrat\w*|fatty[- ]replaced|(?:liver|hepatic parenchyma)\W+(?:is|appears)\W+(?:\w+\W+){0,2}?fatty"
    r"|\b(?:diffuse(?:ly)?|overall|generalized)\W+(?:\w+\W+){0,5}?" + FAT +
    r"|(?:decreased|diminished|low|lower|reduced) (?:attenuation|density)\W+(?:\w+\W+){0,3}?(?:liver|hepatic)"
    r"|(?:liver|hepatic)\W+(?:\w+\W+){0,4}?(?:hypoattenuat\w*|hypodens\w*|(?:decreased|diminished|low|lower|reduced)(?: in)? (?:attenuation|density))"
    r"|(?:hypoattenuat\w*|hypodens\w*) liver|\bnafld\b|\bnash\b", re.I)
DIFFUSE_WORD = re.compile(r"\bdiffuse(?:ly)?\b|\bgeneralized\b|\bthroughout\b", re.I)
NEGRX = re.compile(r"\b(?:no|not|without|negative for|free of|resolved|resolution of|absence of|rather than|no evidence of)\b", re.I)
CLAUSE = re.compile(r"[;:,]|\bbut\b|\bhowever\b|\bwith\b|\bwhich\b", re.I)
UNCERTAIN = re.compile(r"possible|possibly|may represent|may reflect|could (?:represent|reflect)|cannot (?:be )?exclude|"
                       r"questionable|equivocal|suspect|likely|probable|probably|suggest|versus|\bvs\.?\b|\bor\b", re.I)
FATWORD = re.compile(r"steato|fatty|\bfat\b|\bnafld\b|\bnash\b", re.I)
LESION = re.compile(r"lesion|cyst|mass|metasta|nodul|hemangioma|tumou?r|abscess|lacerat|infarct|biloma|hematoma|collection|"
                    r"too small|subcentimeter|\b\d+(?:\.\d+)? ?(?:mm|cm)\b", re.I)
DIFFUSE_CTX = re.compile(r"diffuse|overall|generali[sz]ed|throughout|entire|(?:liver|parenchyma) (?:is|appears)", re.I)
LANDMARK = re.compile(r"falciform|gallbladder fossa|porta hepatis|periportal|fissure|ligamentum|segment (?:4|iv)", re.I)


def negated(sent, start):
    pre = sent[max(0, start - 80):start]
    parts = CLAUSE.split(pre)
    return bool(NEGRX.search(parts[-1] if parts else pre))


def sentences(txt):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+|\n+|(?<=\.)(?=[A-Z])", txt or "") if s.strip()]


def classify(sents, need_ctx=True):
    """Returns flags over affirmed (non-negated, non-history) liver-fat sentences."""
    f = dict(n_fat_sent=0, d_txt=0, f_txt=0, sparing=0, f_landmark=0, d_diffuse_word=0, neg_only=0, hist_only=0,
             unc_d=0, unc_f=0, n_neg_hits=0)
    any_aff, any_neg, any_hist = False, False, False
    for s in sents:
        if (need_ctx and not LIVER_CTX.search(s)) or not re.search(FAT + r"|spar", s, re.I):
            continue
        if OTHER_ORGAN.search(s) and not re.search(r"liver|hepat|steato", s, re.I):
            continue
        if HISTORY.search(s):
            any_hist = True; continue
        if not FATWORD.search(s) and not SPARING.search(s) and LESION.search(s):
            continue                         # low attenuation describing a lesion, not fat
        f["n_fat_sent"] += 1
        rest = s; got = False
        for m in FOCAL.finditer(s):
            if negated(s, m.start()):
                f["n_neg_hits"] += 1; any_neg = True; continue
            if NONLIVER_SPAN.search(s[m.start():m.end() + 25]):
                continue                     # e.g. "fat stranding", "focal fatty infiltration of the terminal ileum"
            if re.search(r"spar", s[m.start():m.end() + 25], re.I):
                rest = rest.replace(m.group(0), " "); continue   # focal (fatty) SPARING = diffuse fat, not focal fat
            if not FATWORD.search(m.group(0)) and not FATWORD.search(s):
                continue                     # focal low attenuation without any fat wording
            f["f_txt"] = 1; got = True
            f["f_landmark"] |= int(bool(LANDMARK.search(m.group(0)) or LANDMARK.search(s)))
            f["unc_f"] |= int(bool(UNCERTAIN.search(s)))
            rest = rest.replace(m.group(0), " ")
        for m in SPARING.finditer(s):
            if negated(s, m.start()):
                continue
            f["sparing"] = 1; f["d_txt"] = 1; got = True
        for m in DIFFUSE.finditer(rest):
            if negated(rest, m.start()):
                f["n_neg_hits"] += 1; any_neg = True; continue
            if not FATWORD.search(m.group(0)) and (LESION.search(s) or (not DIFFUSE_CTX.search(s)
                                                                   and re.search(r"focus|foci|area|region", s, re.I))):
                continue                     # "hepatic hypodensity" without fat wording must be diffuse and lesion-free
            f["d_txt"] = 1; got = True
            f["unc_d"] |= int(bool(UNCERTAIN.search(s)))
            f["d_diffuse_word"] |= int(bool(DIFFUSE_WORD.search(s)))
        any_aff |= got
    f["neg_only"] = int(any_neg and not any_aff)
    f["hist_only"] = int(any_hist and not any_aff)
    return f


# ---------------------------------------------------------------- split-gated readers
def read_labels(okids):
    rx = re.compile(r'"study_id"\s*:\s*"([^"]+)"')
    out = {}
    with open(LABELS) as fh:
        for line in fh:
            m = rx.search(line)
            if not m or m.group(1) not in okids:
                continue                      # not decoded
            r = json.loads(line)
            g = r["answer"]["hepatic_steatosis"]
            out[r["study_id"]] = (r.get("split"), g["category"], " ".join(g.get("quotes") or []),
                                  (g.get("detail") or {}).get("grade", ""))
    return out


def read_reports(okids):
    s = open(REPORTS, encoding="utf-8").read()
    dec = json.JSONDecoder(); STR = re.compile(r'"(?:[^"\\]|\\.)*"'); TOK = re.compile(r'"(?:[^"\\]|\\.)*"|[{}\[\]]')
    WS = re.compile(r"[\s,]*"); out = {}
    p = WS.match(s, s.index("{") + 1).end()
    while s[p] != "}":
        m = STR.match(s, p); key = json.loads(m.group(0))
        p = re.compile(r"\s*:\s*").match(s, m.end()).end()
        if key in okids:
            v, p = dec.raw_decode(s, p)
            out[key] = v.get("report", "") if isinstance(v, dict) else str(v)
        elif s[p] in "{[":                    # skip without decoding
            depth = 0
            for t in TOK.finditer(s, p):
                c = t.group(0)
                if c in "{[":
                    depth += 1
                elif c in "}]":
                    depth -= 1
                    if depth == 0:
                        p = t.end(); break
        elif s[p] == '"':
            p = STR.match(s, p).end()
        else:
            p = re.compile(r"[^,}]*").match(s, p).end()
        p = WS.match(s, p).end()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()
    ok = set(a.splits.split(","))
    if "test" in ok and not a.allow_test:
        sys.exit("test split refused: pass --allow-test only after the test run is locked")
    spl = pd.read_csv(SPLIT).set_index("study_id")["split"]
    okids = set(spl[spl.isin(ok)].index)
    lab = read_labels(okids)
    rep = read_reports(okids)
    rows = []
    for sid, (sp, cat, quote, grade) in lab.items():
        if sp not in ok or spl.get(sid) not in ok:
            continue
        txt = rep.get(sid, "")
        fr = classify(sentences(txt))
        fq = classify(sentences(quote), need_ctx=False)   # quotes are hepatic-steatosis evidence by construction
        d = dict(study_id=sid, split=spl[sid], hs_cat=cat, grade=grade, has_report=int(bool(txt)), quote_empty=int(not quote.strip()))
        d.update({k: v for k, v in fr.items()})
        d.update({"q_" + k: fq[k] for k in ("d_txt", "f_txt", "sparing")})
        # union of report + label quote (quote is report text selected by the labeller)
        D = int(fr["d_txt"] or fq["d_txt"]); Fo = int(fr["f_txt"] or fq["f_txt"])
        d["D"], d["F"] = D, Fo
        P, N = cat in POS, cat in NEG
        d["y_hs"] = 1.0 if P else 0.0 if N else np.nan
        sub = ("both" if D and Fo else "diffuse" if D else "focal" if Fo else "unassigned") if P else ("neg" if N else "unclear")
        d["subtype"] = sub
        # ---- main definition
        d["y_diffuse"] = 1.0 if sub in ("diffuse", "both") else 0.0 if (N or sub == "focal") else np.nan
        d["y_focal"] = 1.0 if sub in ("focal", "both") else 0.0 if (N or sub == "diffuse") else np.nan
        # ---- sensitivity variants (not the main definition)
        d["y_diffuse_excl"] = 1.0 if sub in ("diffuse", "both") else 0.0 if N else np.nan       # focal-only dropped
        d["y_focal_excl"] = 1.0 if sub in ("focal", "both") else 0.0 if N else np.nan           # diffuse-only dropped
        d["neg_focal_text"] = int(N and fr["f_txt"] == 1)                                       # orig negative w/ affirmed focal fat
        d["y_focal_negclean"] = np.nan if d["neg_focal_text"] else d["y_focal"]
        d["neg_diffuse_text"] = int(N and fr["d_txt"] == 1)
        rows.append(d)
    df = pd.DataFrame(rows).sort_values(["split", "study_id"])
    df.to_csv(a.out, index=False)
    print("rows", len(df), df.groupby("split").size().to_dict())
    print(pd.crosstab([df.split, df.hs_cat], df.subtype))
    for c in ["y_diffuse", "y_focal", "y_diffuse_excl", "y_focal_excl", "y_focal_negclean"]:
        print(c, df.groupby("split")[c].agg(["sum", "count"]).to_dict())


if __name__ == "__main__":
    main()
