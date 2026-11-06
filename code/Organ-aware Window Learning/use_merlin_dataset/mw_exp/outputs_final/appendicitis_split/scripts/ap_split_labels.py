"""闌尾炎拆分標籤：確定的急性闌尾炎 (y_ap_def) vs 可疑／治療中／追蹤／術後 (y_ap_unc).

Deterministic rules on report text + the existing Qwen appendicitis label (never model scores).
Rules are documented in label_rules.md (same output folder). Locked 2026-11-05 before any test use.

Split safety (same scanner as steatosis_split/hs_split_labels.py):
  * split.csv is read first; only study_ids whose split is in --splits are kept.
  * labels jsonl: study_id is regex-extracted from each raw line and checked BEFORE the line is json-decoded.
  * reports.json: a key-gated scanner json-decodes only values whose key is allowed; other values are skipped
    by a bracket/string tokenizer without being decoded or stored.
  * 'test' is refused unless --allow-test is given (only after the user locks the test run).
Output: one row per study, keyword flags + labels only (no report text).

Usage:
  python ap_split_labels.py --out labels_trval_split.csv [--splits train,val]
  python ap_split_labels.py --out labels_test_split.csv --splits test --allow-test      # later, once locked
"""
import argparse, json, re, sys
import numpy as np, pandas as pd

R = "/home/advantech/Desktop/Tony/VLM/"
SPLIT = R + "OWL/data/split.csv"
LABELS = R + "OWL/data/labels_disease14_all.jsonl"
REPORTS = R + "merlin_disease14/reports.json"
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
SOFT_UNC = 0.5          # suggested soft target for the uncertain/follow-up group (exploratory only)

# ---------------------------------------------------------------- text rules (fixed)
# sentences considered: appendix / RLQ / cecal context, appendectomy, or RLQ collection/drain context
CTX = re.compile(r"append|periappend|cecum|cecal|ileocec|right lower quadrant|\bRLQ\b|terminal ileum|"
                 r"right (?:lower )?(?:abdomen|pelvis|paracolic)", re.I)
RULES = {
    # post-operative: appendectomy already done / surgical bed / stump
    "t_postop": r"post-?operative|post-?surgical|(?:status post|s/p|after|following|since|recent|prior|previous|interval)\s+"
                r"(?:\w+\s+){0,2}appendectomy|appendectomy (?:bed|site|clips?)|post[- ]?appendectomy|(?:surgical|operative) bed|"
                r"appendiceal stump|stump appendicitis|surgically (?:absent|removed)|appendix (?:has been |was )?removed",
    # drain / percutaneous drainage already performed (recommendations like 'amenable to drainage' excluded below)
    "t_drain": r"drain(?:age)? (?:catheter|tube)|pigtail|\bdrains?\b(?! ?able)|(?:status post|s/p|after|following|with|interval)\s+"
               r"(?:\w+\s+){0,2}(?:percutaneous\s+)?drain(?:age)?|percutaneous(?:ly)? drained|catheter (?:in|within) (?:the )?(?:collection|abscess)",
    # medical treatment
    "t_treat": r"antibiotic|\babx\b|non-?operative(?:ly)?|conservative(?:ly)? (?:management|managed|treat)|treated (?:with|for|medically|nonoperatively)|"
               r"being treated|on treatment|under treatment|medical(?:ly)? (?:management|managed|treated)",
    # follow-up / evolution vs prior study
    "t_followup": r"(?<!following )(?<!after )(?<!upon )(?<!until )(?<!before )(?:resolv|resolution)|improv|interval (?:decrease|improvement|resolution|regression|evolution|organization|increase|worsening|progression|enlargement|"
                  r"development of (?:\w+\s+){0,2}(?:abscess|collection))|decreas(?:ed|ing|e) in (?:size|extent|caliber|inflammat)|"
                  r"(?:smaller|less (?:prominent|conspicuous|pronounced|extensive)) (?:than|compared|when compared)|redemonstrat|"
                  r"again (?:seen|noted|demonstrated|identified|visualized)|\bpersist|\bresidual(?!\s+(?:\w+\s+)?(?:contrast|barium|oral))|evolving|\bhealing|"
                  r"worsen|progress(?:ed|ion|ing)|(?:has|have) (?:organized|evolved)|"
                  r"previously (?:demonstrated|identified|reported|diagnosed|measur\w+)|unchanged|stable|"
                  r"(?:known|recent(?:ly)?|prior|previous|treated|ongoing) (?:\w+\s+){0,2}appendicitis|"
                  r"history of (?:\w+\s+){0,3}appendicitis|follow[- ]?up (?:of|for|on) (?:\w+\s+){0,2}appendicitis",
}
RX = {k: re.compile(v, re.I) for k, v in RULES.items()}
RECOMMEND = re.compile(r"amenable|could be|may be|consider|candidate|recommend|if clinically|for possible|potential(?:ly)?", re.I)
NEGRX = re.compile(r"\b(?:no|not|without|negative for|free of|absence of|no evidence of)\b", re.I)
CLAUSE = re.compile(r"[;:,]|\bbut\b|\bhowever\b|\bwhich\b", re.I)
ABSC = re.compile(r"abscess|collection|phlegmon", re.I)
APPX_DONE = re.compile(r"appendectomy|surgically (?:absent|removed)|appendix (?:has been |was )?removed", re.I)
NEGPHRASE = re.compile(r"no (?:ct |radiographic |definite |convincing )?(?:evidence|findings|signs?) (?:of|to suggest|suggestive of) (?:acute )?appendicitis|"
                       r"no (?:acute )?appendicitis|normal(?:[- ]appearing)? appendix|appendix (?:is |appears )?(?:normal|unremarkable)|"
                       r"without (?:evidence of )?appendicitis|unremarkable appendix|appendicitis is not", re.I)


def sentences(txt):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+|\n+|(?<=\.)(?=[A-Z])", txt or "") if s.strip()]


def negated(sent, start):
    pre = sent[max(0, start - 60):start]
    parts = CLAUSE.split(pre)
    return bool(NEGRX.search(parts[-1] if parts else pre))


def flags(sents):
    f = {k: 0 for k in RULES}; f["t_abscess_postappx"] = 0; f["n_ctx_sent"] = 0
    for s in sents:
        ctx = bool(CTX.search(s)) or bool(re.search(r"drain|abscess|collection", s, re.I) and re.search(r"right|append|cec|pelv", s, re.I))
        if not ctx:
            continue
        f["n_ctx_sent"] += 1
        for k, rx in RX.items():
            for m in rx.finditer(s):
                if k == "t_drain" and RECOMMEND.search(s[max(0, m.start() - 40):m.start()]):
                    continue
                # negation guard only for drain / treatment / appendectomy (follow-up words keep meaning when negated,
                # e.g. 'no residual abscess' still implies a prior episode)
                if k in ("t_drain", "t_treat", "t_postop") and negated(s, m.start()):
                    continue
                f[k] = 1; f.setdefault("_hits", []).append(m.group(0).lower()); break
        if ABSC.search(s) and APPX_DONE.search(s) and not negated(s, APPX_DONE.search(s).start()):
            f["t_abscess_postappx"] = 1
    return f


def read_labels(okids):
    rx = re.compile(r'"study_id"\s*:\s*"([^"]+)"')
    out = {}
    with open(LABELS) as fh:
        for line in fh:
            m = rx.search(line)
            if not m or m.group(1) not in okids:
                continue                      # not decoded
            r = json.loads(line)
            g = r["answer"]["appendicitis"]; det = g.get("detail") or {}
            out[r["study_id"]] = (r.get("split"), g["category"], " ".join(g.get("quotes") or []),
                                  det.get("perforated", "na"), det.get("appendicolith", "na"))
    return out


def read_reports(okids):
    s = open(REPORTS, encoding="utf-8").read()
    dec = json.JSONDecoder(); STR = re.compile(r'"(?:[^"\\]|\\.)*"'); TOK = re.compile(r'"(?:[^"\\]|\\.)*"|[{}\[\]]')
    WS = re.compile(r"[\s,]*"); COL = re.compile(r"\s*:\s*"); out = {}
    p = WS.match(s, s.index("{") + 1).end()
    while s[p] != "}":
        m = STR.match(s, p); key = json.loads(m.group(0))
        p = COL.match(s, m.end()).end()
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
    for sid, (sp, cat, quote, perf, colith) in lab.items():
        if sp not in ok or spl.get(sid) not in ok:
            continue
        txt = rep.get(sid, "")
        fr = flags(sentences(txt)); fq = flags(sentences(quote))
        d = dict(study_id=sid, split=spl[sid], ap_cat=cat, ap_perf=perf, ap_colith=colith, has_report=int(bool(txt)))
        for k in list(RULES) + ["t_abscess_postappx"]:
            d[k] = int(fr[k] or fq[k])
        d["n_ctx_sent"] = fr["n_ctx_sent"]
        d["kw_negphrase"] = int(bool(NEGPHRASE.search(txt + " " + quote)))
        tf = int(d["t_postop"] or d["t_drain"] or d["t_treat"] or d["t_followup"] or d["t_abscess_postappx"])
        d["treat_followup_any"] = tf
        P, N = cat in POS, cat in NEG
        d["y_ap"] = 1.0 if P else 0.0 if N else np.nan                     # original combined label (reference)
        if cat == "yes" and not tf:
            grp = "definite"
        elif P:
            grp = "uncertain_followup"
        else:
            grp = "neg" if N else "unclear"
        d["ap_group"] = grp
        d["unc_reason"] = ("probably_yes" if cat == "probably_yes" else "") + \
                          ("+" if cat == "probably_yes" and tf else "") + \
                          ("+".join(k[2:] for k in ("t_postop", "t_drain", "t_treat", "t_followup", "t_abscess_postappx") if d[k]) if tf and P else "")
        # ---- main definitions (masking)
        d["y_ap_def"] = 1.0 if grp == "definite" else 0.0 if grp == "neg" else np.nan
        d["y_ap_unc"] = 1.0 if grp == "uncertain_followup" else 0.0 if grp == "neg" else np.nan
        # ---- suggested soft targets for later soft-label training of the definite finding (NOT the main result)
        d["y_ap_soft"] = 1.0 if grp == "definite" else SOFT_UNC if grp == "uncertain_followup" else 0.0 if grp == "neg" else np.nan
        rows.append(d)
    df = pd.DataFrame(rows).sort_values(["split", "study_id"])
    df.to_csv(a.out, index=False)
    print("rows", len(df), df.groupby("split").size().to_dict())
    print(pd.crosstab(df.ap_group, df.split).to_string())
    print(pd.crosstab(df.ap_cat, [df.split, df.ap_group]).to_string())
    for s in sorted(ok):
        t = df[(df.split == s) & (df.y_ap == 1)]
        print(s, "positives flag rates:", t[[k for k in RULES] + ["t_abscess_postappx"]].mean().round(3).to_dict())
        print(s, "unc reasons:", t[t.ap_group == "uncertain_followup"].unc_reason.value_counts().to_dict())


if __name__ == "__main__":
    main()
