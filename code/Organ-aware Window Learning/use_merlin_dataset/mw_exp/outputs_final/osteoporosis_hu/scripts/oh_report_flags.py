"""影像骨質疏鬆 (HU-defined): report-text flags used ONLY for the exclusion / fallback rules (never as the label).

Flags per study (train/val only unless --allow-test):
  cement   : vertebroplasty / kyphoplasty / vertebral augmentation, or cement / PMMA in a sentence with spine context
             (vertebra / spine / lumbar / thoracic / T# / L#; excludes cemented hip arthroplasty), not negated
             (history sentences are KEPT: cement stays in the bone)
  L1_frac / L2_frac : sentence names the level (L1, L-1, first lumbar; or a range like T12-L2 / L1-L3 that spans it)
             and fracture / compression / collapse / wedg* / height loss, not negated
  L1_scl / L2_scl : level named + sclerotic / sclerosis / blastic / bone island / enostosis / dense, not negated
  hardware : pedicle screw / spinal fusion / rods / laminectomy hardware (reported only, not a rule)
Split safety: split.csv read first; reports.json values are json-decoded only for allowed keys
(same key-gated scanner as steatosis_split/scripts/hs_split_labels.py). Output has flags only, no text.
usage: python oh_report_flags.py --out report_flags_trval.csv [--splits train,val] [--allow-test]"""
import argparse, json, re, sys
import pandas as pd

R = "/home/advantech/Desktop/Tony/VLM/"
SPLIT = R + "OWL/data/split.csv"
REPORTS = R + "merlin_disease14/reports.json"

CEMENT = re.compile(r"vertebroplast\w*|kyphoplast\w*|vertebral augmentation|vertebral body augmentation", re.I)
# generic cement only counts in a spine sentence (excludes cemented hip/knee arthroplasty)
CEMENT_GEN = re.compile(r"\bcement\w*|\bpmma\b|methylmethacrylate", re.I)
SPINE_CTX = re.compile(r"vertebra\w*|spine|spinal|lumbar|thoracic|\b[TL]\s?-?\d{1,2}\b", re.I)
FRAC = re.compile(r"fractur\w*|compress\w*|collaps\w*|wedg\w*|height loss|loss of (?:\w+\s+){0,2}height|"
                  r"plana|burst", re.I)
SCL = re.compile(r"scleros\w*|sclerotic|blastic|bone island|enostos\w*|\bdense\b|increased (?:density|attenuation)", re.I)
HW = re.compile(r"pedicle screw|spinal fusion|posterior fusion|fusion hardware|\brods?\b|interbody|laminectomy", re.I)
NEGRX = re.compile(r"\b(?:no|not|without|negative for|free of|resolved|absence of|no evidence of)\b", re.I)
CLAUSE = re.compile(r"[;:,]|\bbut\b|\bhowever\b|\bwhich\b", re.I)
LEVEL_TOK = re.compile(r"\b([TL])\s?-?\s?(\d{1,2})\b(?:\s*(?:-|–|to|through)\s*([TL])?\s?-?\s?(\d{1,2})\b)?", re.I)
WORD_LEVEL = {"L1": re.compile(r"first lumbar|\bL1\b", re.I), "L2": re.compile(r"second lumbar|\bL2\b", re.I)}
ORDER = [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]


def negated(sent, start):
    pre = sent[max(0, start - 80):start]
    parts = CLAUSE.split(pre)
    return bool(NEGRX.search(parts[-1] if parts else pre))


def sentences(txt):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+|\n+|(?<=\.)(?=[A-Z])", txt or "") if s.strip()]


def levels_in(sent):
    out = set()
    for m in LEVEL_TOK.finditer(sent):
        a = f"{m.group(1).upper()}{int(m.group(2))}"
        if a not in ORDER:
            continue
        if m.group(4):
            b = f"{(m.group(3) or m.group(1)).upper()}{int(m.group(4))}"
            if b in ORDER:
                i, j = sorted((ORDER.index(a), ORDER.index(b)))
                out.update(ORDER[i:j + 1]); continue
        out.add(a)
    for lv, rx in WORD_LEVEL.items():
        if rx.search(sent):
            out.add(lv)
    return out


def hit(sent, rx):
    return any(not negated(sent, m.start()) for m in rx.finditer(sent))


def flags(txt):
    f = dict(cement=0, L1_frac=0, L2_frac=0, L1_scl=0, L2_scl=0, hardware=0)
    for s in sentences(txt):
        if hit(s, CEMENT) or (SPINE_CTX.search(s) and hit(s, CEMENT_GEN)):
            f["cement"] = 1
        if hit(s, HW):
            f["hardware"] = 1
        lv = levels_in(s)
        for L in ("L1", "L2"):
            if L in lv:
                if hit(s, FRAC):
                    f[f"{L}_frac"] = 1
                if hit(s, SCL):
                    f[f"{L}_scl"] = 1
    return f


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
    rep = read_reports(okids)
    rows = [dict(study_id=k, split=spl[k], has_report=int(bool(rep.get(k))), **flags(rep.get(k, ""))) for k in sorted(okids)]
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    print(df.groupby("split")[["has_report", "cement", "L1_frac", "L2_frac", "L1_scl", "L2_scl", "hardware"]].sum())


if __name__ == "__main__":
    main()
