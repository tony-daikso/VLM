"""膽泥 (gallbladder sludge) label: deterministic rules on report text + the existing Qwen gallstones label (never model scores).
Decision: user 2026-11-06 (LABEL_DECISIONS_PENDING.md item 11): separate finding 膽泥. 膽結石 definition unchanged.
Rules documented in ../label_rules.md. Locked 2026-11-06 before any test use.

Split safety (same readers as steatosis_split/scripts/hs_split_labels.py):
  * split.csv first; only study_ids whose split is in --splits are kept.
  * labels jsonl: study_id regex-extracted from the raw line and checked BEFORE json decoding.
  * reports.json: key-gated scanner, values of other ids are skipped without decoding.
  * 'test' refused unless --allow-test.
Output: one row per study with flags + labels only (no report text). --audit N prints short matched spans (train only).
Usage: python sl_labels.py --out labels_trval_sludge.csv [--splits train,val] [--audit 40]
"""
import argparse, json, re, sys
import numpy as np, pandas as pd

R = "/home/advantech/Desktop/Tony/VLM/"
SPLIT = R + "OWL/data/split.csv"
LABELS = R + "OWL/data/labels_disease14_all.jsonl"
REPORTS = R + "merlin_disease14/reports.json"
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}

SLUDGE = re.compile(r"\bsludge\w*|\bmicrolithiasis\b", re.I)
GBCTX = re.compile(r"gall ?bladder|\bgb\b|cholecyst|cholelith|gallstone|biliary sludge|sludge", re.I)
DUCT = re.compile(r"biliary tree|biliary duct|intrahepatic|extrahepatic|common bile duct|common duct|\bcbd\b|cystic duct|hepatic duct|bile ducts?\b|choledoch|ampulla", re.I)
GBWORD = re.compile(r"gall ?bladder|\bgb\b|cholecyst|cholelith|gallstone", re.I)
NONGB = re.compile(r"kidney|renal|ureter|urinary|\bbladder\b(?<!gall bladder)|pyelo|nephro|calyx|calyc", re.I)
HISTORY = re.compile(r"history of|\bindication|clinical (?:history|information|indication)|reason for (?:exam|study)|"
                     r"^\s*(?:history|clinical|indication|reason|comparison|technique)\b", re.I)
HYPO = re.compile(r"evaluate for|eval(?:uation)? (?:for|of)|assess(?:ment)? (?:for|of)|to exclude|rule out|\br/o\b|"
                  r"if (?:clinically|there is)|could be (?:obtained|performed)|recommend", re.I)
NEGRX = re.compile(r"\b(?:no|not|without|negative for|free of|resolved|resolution of|absence of|rather than|no evidence of|nor)\b", re.I)
POSTNEG = re.compile(r"^\W*(?:\w+\W+){0,4}?(?:(?:has|have) resolved|resolved|no longer (?:seen|visualized|present|identified|evident)|"
                     r"(?:is|are) not (?:seen|identified|visualized|present)|(?:is|are) absent)", re.I)
CLAUSE = re.compile(r"[;:]|\bbut\b|\bhowever\b|\bwhich\b|\bwith\b|\.\s", re.I)
UNCERTAIN = re.compile(r"possible|possibly|may represent|may reflect|could (?:represent|reflect)|cannot (?:be )?exclude|"
                       r"(?:cannot|can not) be excluded|not excluded|questionable|equivocal|suspect|likely|probable|probably|"
                       r"if any|suggest|versus|\bvs\.?\b|\bor\b|and/or|\bvs\b", re.I)
STONEW = re.compile(r"stone|cholelith|calcul|gallstone", re.I)
VICARIOUS = re.compile(r"vicarious|excreted contrast|contrast (?:excretion|within the gallbladder)", re.I)
DEBRIS = re.compile(r"debris|hyperdense bile|layering (?:hyper)?dense|dense bile|milk of calcium", re.I)


def sentences(txt):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+|\n+|(?<=\.)(?=[A-Z])", txt or "") if s.strip()]


NOTCHANGE = re.compile(r"\bnot\s+(?:(?:significantly|substantially|markedly|appreciably)\s+)?(?:changed|increased|decreased|enlarged|distended)\b", re.I)


def negated(s, m):
    pre = NOTCHANGE.sub(" ", s[max(0, m.start() - 120):m.start()])
    parts = CLAUSE.split(pre); seg = parts[-1] if parts else pre
    if NEGRX.search(seg):
        # "no stones; sludge" handled by clause split; "no stones but sludge" by 'but'. Allow "no ... , sludge" style:
        # if a comma separates the negator from the term and the text after the comma has no 'or/and/nor', not negated.
        last = max(seg.rfind(","), -1)
        if last >= 0 and not NEGRX.search(seg[last:]) and not re.search(r"\b(?:or|nor|and)\b", seg[last:], re.I):
            return False
        return True
    return bool(POSTNEG.match(s[m.end():m.end() + 60]))


def hypothetical(s, m):
    pre = s[max(0, m.start() - 60):m.start()]
    parts = CLAUSE.split(pre)
    return bool(HYPO.search(parts[-1] if parts else pre))


def classify(sents, need_ctx=True):
    f = dict(sl_txt=0, sl_neg=0, sl_hist=0, sl_hypo=0, sl_duct_only=0, sl_unc=0, sl_tumefactive=0, sl_ball=0,
             sl_microlith=0, sl_stone_sent=0, kw_vicarious=0, kw_debris=0)
    spans = []
    for s in sents:
        if VICARIOUS.search(s) and GBWORD.search(s): f["kw_vicarious"] = 1
        if DEBRIS.search(s) and GBWORD.search(s) and not NONGB.search(s): f["kw_debris"] = 1
        for m in SLUDGE.finditer(s):
            if need_ctx and not GBCTX.search(s): continue
            if NONGB.search(s) and not GBWORD.search(s): continue
            sp_ = s[max(0, m.start() - 60):m.end() + 40]
            if HISTORY.search(s): f["sl_hist"] = 1; spans.append("[HIST] " + sp_); continue
            if negated(s, m): f["sl_neg"] = 1; spans.append("[NEG] " + sp_); continue
            if hypothetical(s, m): f["sl_hypo"] = 1; spans.append("[HYPO] " + sp_); continue
            if DUCT.search(s) and not GBWORD.search(s):
                f["sl_duct_only"] = 1; spans.append("[DUCT] " + sp_); continue
            f["sl_txt"] = 1
            if m.group(0).lower().startswith("microlith"): f["sl_microlith"] = 1
            f["sl_tumefactive"] |= int(bool(re.search(r"tumefactive|tumefactant|sludge ball|ball of sludge", s, re.I)))
            f["sl_ball"] |= int(bool(re.search(r"sludge ball|ball of sludge|balls? of sludge", s, re.I)))
            f["sl_unc"] |= int(bool(UNCERTAIN.search(s)))
            f["sl_stone_sent"] |= int(bool(STONEW.search(s)))
            spans.append(sp_)
    return f, spans


# ---------------------------------------------------------------- split-gated readers (copied from hs_split_labels.py)
def read_labels(okids):
    rx = re.compile(r'"study_id"\s*:\s*"([^"]+)"'); out = {}
    with open(LABELS) as fh:
        for line in fh:
            m = rx.search(line)
            if not m or m.group(1) not in okids:
                continue
            r = json.loads(line); a = r["answer"]; g = a["gallstones"]
            out[r["study_id"]] = (r.get("split"), g["category"], " ".join(g.get("quotes") or []),
                                  a["surgically_absent_gallbladder"]["category"])
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
        elif s[p] in "{[":
            depth = 0
            for t in TOK.finditer(s, p):
                c = t.group(0)
                if c in "{[": depth += 1
                elif c in "}]":
                    depth -= 1
                    if depth == 0: p = t.end(); break
        elif s[p] == '"':
            p = STR.match(s, p).end()
        else:
            p = re.compile(r"[^,}]*").match(s, p).end()
        p = WS.match(s, p).end()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--splits", default="train,val")
    ap.add_argument("--allow-test", action="store_true"); ap.add_argument("--audit", type=int, default=0)
    ap.add_argument("--audit-seed", type=int, default=0)
    a = ap.parse_args(); ok = set(a.splits.split(","))
    if "test" in ok and not a.allow_test:
        sys.exit("test split refused: pass --allow-test only after the test run is locked")
    spl = pd.read_csv(SPLIT).set_index("study_id")["split"]
    okids = set(spl[spl.isin(ok)].index)
    lab = read_labels(okids); rep = read_reports(okids)
    rows, aud = [], {}
    for sid, (sp, cat, quote, chole) in lab.items():
        if sp not in ok or spl.get(sid) not in ok: continue
        txt = rep.get(sid, "")
        fr, sr = classify(sentences(txt)); fq, sq = classify(sentences(quote), need_ctx=False)
        d = dict(study_id=sid, split=spl[sid], gs_cat=cat, chole_cat=chole, has_report=int(bool(txt)))
        d.update(fr); d.update({"q_sl_txt": fq["sl_txt"], "q_sl_neg": fq["sl_neg"]})
        S = int(fr["sl_txt"] or fq["sl_txt"]); d["SL"] = S
        d["y_gs"] = 1.0 if cat in POS else 0.0 if cat in NEG else np.nan        # unchanged official gallstone label
        d["y_sludge"] = float(S) if (txt or quote.strip()) else np.nan           # 膽泥 (with or without stones)
        d["sl_group"] = ("none" if not S else "sludge_stones" if d["y_gs"] == 1 else "sludge_only" if d["y_gs"] == 0 else "sludge_gs_unclear")
        d["y_sludge_only"] = (1.0 if d["sl_group"] == "sludge_only" else np.nan if np.isnan(d["y_gs"]) or np.isnan(d["y_sludge"]) else 0.0)
        # 3-class target: 0 neither, 1 sludge-only, 2 stones (+/- sludge); NaN if gallstone unclear
        d["y3"] = np.nan if np.isnan(d["y_gs"]) else 2.0 if d["y_gs"] == 1 else 1.0 if S else 0.0
        # secondary gallstone view: sludge-only cases removed from the negatives
        d["y_gs_excl_sludgeonly"] = np.nan if d["sl_group"] == "sludge_only" else d["y_gs"]
        rows.append(d)
        if spl[sid] == "train" and (sr or sq or fr["sl_neg"] or fr["sl_hypo"] or fr["sl_duct_only"]):
            aud[sid] = (d["sl_group"], cat, fr, (sr + sq)[:3])
    df = pd.DataFrame(rows).sort_values(["split", "study_id"])
    df.to_csv(a.out, index=False)
    print("rows", len(df), df.groupby("split").size().to_dict())
    print(pd.crosstab([df.split, df.gs_cat], df.sl_group))
    for c in ["y_sludge", "y_sludge_only", "y_gs", "y_gs_excl_sludgeonly"]:
        print(c, df.groupby("split")[c].agg(["sum", "count"]).to_dict())
    for c in ["sl_neg", "sl_hist", "sl_hypo", "sl_duct_only", "sl_unc", "sl_tumefactive", "sl_ball", "sl_microlith", "kw_vicarious", "kw_debris"]:
        print(c, df.groupby("split")[c].sum().to_dict(), "| among SL=1:", df[df.SL == 1].groupby("split")[c].sum().to_dict())
    if a.audit:
        rng = np.random.default_rng(a.audit_seed); keys = list(aud)
        spec = [k for k in keys if aud[k][2]["sl_neg"] or aud[k][2]["sl_hypo"] or aud[k][2]["sl_duct_only"] or aud[k][2]["sl_hist"]]
        for k in spec + list(rng.choice(keys, min(a.audit, len(keys)), replace=False)):
            g, cat, fr, sp = aud[k]
            fl = {x: v for x, v in fr.items() if v}
            print(f"AUDIT {g:14s} gs={cat:13s} {fl} || " + " ## ".join(x.replace("\n", " ") for x in sp))


if __name__ == "__main__":
    main()
