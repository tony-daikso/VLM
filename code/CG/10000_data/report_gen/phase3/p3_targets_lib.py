#!/usr/bin/env python3
"""Phase 3 target construction: the doctors' ORIGINAL sentences about the 14 diseases, in report order.

Rules (2026-11-09, style-imitation goal):
- Source = std14 `source_sentences` (verbatim copies chosen by Qwen), positives AND negatives, any status except
  not_mentioned. Sentences are located in the original report; duplicates (same sentence for 2 diseases, or the
  Impression repeating a Findings sentence) are kept once, at their first (Findings) position.
- Impression-only sentences (disease has no sentence before the IMP header) are kept (flag `imp_only`).
- Temporal / comparison wording (the model cannot know it without prior imaging):
    * keep-status words (stable, stationary, unchanged, persistent, again seen, still, known, redemonstrated)
      and comparison clauses ("compared with ...", "than previous study", "since 2021-10-21") are STRIPPED;
    * change-describing sentences (increased, decreased, progression, regression, improved, resolved, new,
      a -> b) are DROPPED; if that removes the only sentence of a disease that is positive/uncertain, the std14
      normalized_sentence (already temporal-free) is used instead (flag `fallback_norm`).
- Typos: doctor wording is kept; only clear misspellings in a reviewed dictionary (TYPO_FIX, built by
  p3_target_stats.py from corpus frequencies) are corrected.
- se/im image references ("(Se:6, Im:45)", "se601-im37") are removed (unknowable for the model).
- Bullets / leading numbers are removed; one sentence per line.
"""
import re

DISEASES = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "hydronephrosis",
            "abdominal_aortic_aneurysm", "atherosclerosis", "osteopenia", "bowel_obstruction", "appendicitis",
            "ascites", "free_air", "gallstones", "biliary_ductal_dilation", "fracture"]
BULLET = re.compile(r"^\s*(?:[\*\-•#?]+|\(?\d{1,2}[\.\)](?!\d))\s*")
IMP_HDR = re.compile(r"(?im)^[ \t\*]*(?:imp(?:ression)?s?|conclusions?|summary)\s*[:：]")

KEEP_STATUS = r"(?:stable|stationary|unchanged|persistent(?:ly)?|persisted|still|known|again(?:\s+(?:seen|noted|shown|demonstrated))?|redemonstrat\w*|re-demonstrat\w*|similar)"
CMP_CLAUSE = re.compile(r"[,;]?\s*\(?\s*(?:as\s+)?(?:compared|comparing|in\s+comparison)\s+(?:with|to)\b[^,;.)]*\)?"
                        r"|[,;]?\s*\b(?:than|from|since)\s+(?:the\s+)?(?:previous|prior|last|preceding|former)\b[^,;.]*"
                        r"|[,;]?\s*\b(?:than|from|since)\s+(?:\d{2,4}[/\-.]\d{1,2}[/\-.]\d{1,4}|\d{4})\b[^,;.]*"
                        r"|[,;]?\s*\b(?:in|on)\s+(?:the\s+)?(?:previous|prior|last)\s+(?:study|studies|exam\w*|CT|MRI?|image\w*|film)\b[^,;.]*"
                        r"|[,;]?\s*\b(?:status|condition)\s+quo\b", re.I)
DESCR = r"(?!\s+(?:of\s+)?(?:the\s+)?(?:\w+\s+)?(?:attenuation|density|densities|height|enhancement|opacit\w*|echogenicity|signal|uptake|lucency|stranding|vascularity|fat|wall|thickness|calibe?r|size\s+of\s+the\s+(?:liver|spleen)))"
CHANGE = re.compile(r"\b(?:(?:increas\w*|decreas\w*|reduc\w*)" + DESCR + r"|progress\w*|regress\w*|improv\w*|resolv\w*|resolution|disappear\w*|"
                    r"new|newly|enlarging|shrink\w*|(?:smaller|larger|bigger|more|less)\s+than\s+(?:the\s+)?(?:previous|prior|before|last)|"
                    r"less(?:er)?\s+(?:extent|amount)|more\s+(?:extent|amount)|interval\s+(?!change)\w+|worse\w*|aggravat\w*|subsid\w*|no\s+longer)\b|->|→", re.I)
KEEP_PHRASE = re.compile(r"[,;]?\s*\b(?:with(?:out)?\s+)?(?:no\s+(?:significant\s+|obvious\s+|definite\s+)?interval\s+change|"
                         r"(?:in\s+)?(?:a\s+)?(?:stable|stationary)\s+(?:condition|status)(?:\s+of(?:\s+the)?)?)\b", re.I)
KEEP_RE = re.compile(r"\b" + KEEP_STATUS + r"\b[,]?\s*", re.I)
TEMPORAL_ANY = re.compile(KEEP_STATUS + r"|" + CHANGE.pattern + r"|compar\w*|previous|prior\b", re.I)
OUT_OF_SCOPE = re.compile(r"\b(?:tumou?r|mass(?:es)?|nodule\w*|metasta\w*|cancer|carcinoma|lymph\s*nodes?|LNs?|cyst\w*|"
                          r"lesion\w*|hemangioma\w*|HCC|lymphadenopathy)\b", re.I)
SIZE = re.compile(r"\d+(?:\.\d+)?\s*(?:x\s*\d+(?:\.\d+)?\s*)*(?:mm|cm)\b", re.I)
SEIM = re.compile(r"\bse(?:ries)?\s*[:#.]?\s*\d+[\s,/\-]*im(?:age)?\s*[:#.]?\s*\d+", re.I)
SEIM_STRIP = re.compile(r"\s*[\(\[]\s*(?:se(?:ries)?|im(?:age)?)\s*[:#.]?\s*\d+[^\)\]]*[\)\]]"
                        r"|\s*\(?\bse(?:ries)?\s*[:#.]?\s*\d+[\s,/\-]*im(?:age)?\s*[:#.]?\s*\d+(?:\s*[,~\-]\s*\d+)*\)?;?", re.I)
TYPO_FIX = {}  # filled from phase3 typo_dictionary.csv (reviewed=1 rows) by load_typo_fix()


def load_typo_fix(path):
    import csv
    d = {}
    try:
        for r in csv.DictReader(open(path)):
            if str(r.get("reviewed", "1")) == "1":
                d[r["typo"]] = r["fix"]
    except FileNotFoundError:
        pass
    return d


def loose_map(s):
    """lower-case alphanumerics of s and the index of each kept char in s."""
    keep, idx = [], []
    for i, ch in enumerate(s):
        c = ch.lower()
        if c.isascii() and c.isalnum():
            keep.append(c)
            idx.append(i)
    return "".join(keep), idx


def loose(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def clean_bullet(s):
    return BULLET.sub("", s).strip()


DKW = {"hepatic_steatosis": r"fat|steato|liver|hepat", "surgically_absent_gallbladder": r"gall|\\bGB|cholecyst",
       "splenomegaly": r"splen", "hydronephrosis": r"hydro|kidney|renal|uropath|urotract|pelvi|calyc|calic|ureter",
       "abdominal_aortic_aneurysm": r"aort|aneurysm|\\bAAA|EVAR|endograft|stent", "atherosclerosis": r"atheroscl|calcif|plaque|aort|arter|vascul",
       "osteopenia": r"osteop|bone\\s+(?:mineral\\s+)?density|demineral|osteoporo|bone", "bowel_obstruction": r"obstruct|ileus|bowel|GI\\s+tract|transition|intestin|\\bSBO|colon|gastrointestinal",
       "appendicitis": r"append", "ascites": r"ascites|fluid|effusion|peritone", "free_air": r"\\bair|pneumoperit|\\bgas",
       "gallstones": r"stone|lith|gall|\\bGB", "biliary_ductal_dilation": r"bil|\\bIHD|\\bCBD|duct|cholang",
       "fracture": r"fract|bony\\s+(?:destruct|patholog|abnormal|lesion)|compress|plasty|osseous\\s+(?:destruct|abnormal)|skelet"}
DKW = {k: re.compile(v, re.I) for k, v in DKW.items()}
SUBSPLIT = re.compile(r"(?<=[a-z0-9)\]][.;])\s*(?=[A-Z][a-z])|(?<=[.;])\s+(?=[A-Z(])")


def relevant_part(s, diseases):
    """multi-sentence source item -> keep only the sub-sentences that mention one of its diseases."""
    parts = [p for p in SUBSPLIT.split(s) if p.strip()]
    if len(parts) <= 1:
        return s, False
    keep = [p for p in parts if any(DKW[d].search(p) for d in diseases)]
    if not keep or len(keep) == len(parts):
        return s, False
    return " ".join(keep), True


def strip_temporal(s):
    """-> (text, action) with action in keep | stripped | drop."""
    if CHANGE.search(s):
        return s, "drop"
    t = CMP_CLAUSE.sub("", s)
    t = KEEP_PHRASE.sub("", t)
    t = KEEP_RE.sub("", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s*[,;]\s*([.;])", r"\1", t)
    t = re.sub(r"\s+([.,;])", r"\1", t).strip(" ,;")
    t = re.sub(r"^(?:is|are|was|were)\s+", "", t, flags=re.I)
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    if loose(t) == loose(s):
        return s, "keep"
    if len(loose(t)) < 4:
        return s, "drop"
    return t, "stripped"


def fix_typos(s, fix):
    if not fix:
        return s, 0
    n = 0

    def rep(m):
        nonlocal n
        w = m.group(0)
        f = fix.get(w.lower())
        if f is None:
            return w
        n += 1
        return f.capitalize() if w[0].isupper() else f
    return re.sub(r"[A-Za-z][A-Za-z\-]+", rep, s), n


def build_target(report, diseases, typo_fix=None):
    """report: original text; diseases: std14 answer['diseases'] (after postproc).
    Returns dict(target=str, lines=[...], info=[per-line dicts], per_disease={d: [...line idx]}, counters)."""
    L, idx = loose_map(report)
    m = IMP_HDR.search(report)
    imp_pos = m.start() if m else len(report)
    cand = {}   # loose key -> dict
    c = dict(src=0, located=0, not_located=0, imp_dup=0, imp_only=0, temporal_stripped=0,
             temporal_drop=0, fallback_norm=0, typo_fixed=0, lost_disease=0, subsentence_trimmed=0, seim_stripped=0)
    for d in DISEASES:
        x = diseases[d]
        if x["status"] == "not_mentioned":
            continue
        for s in x["source_sentences"]:
            c["src"] += 1
            s0 = clean_bullet(s)
            k = loose(s0)
            if not k:
                continue
            p = L.find(k)
            if p < 0:
                c["not_located"] += 1
                pos = 10 ** 9
            else:
                c["located"] += 1
                pos = idx[p]
                # prefer an occurrence in Findings if the same text also occurs earlier
            sec = "findings" if pos < imp_pos else "impression"
            if k in cand:
                cand[k]["diseases"].add(d)
                continue
            cand[k] = dict(text=s0, pos=pos, sec=sec, diseases={d})
    # Impression copies of a Findings sentence: same loose key already covered (find returns first = Findings)
    have_find = {d for v in cand.values() if v["sec"] == "findings" for d in v["diseases"]}
    lines = []
    for k, v in sorted(cand.items(), key=lambda kv: kv[1]["pos"]):
        if v["sec"] == "impression":
            if v["diseases"] <= have_find:
                c["imp_dup"] += 1
                continue
            c["imp_only"] += 1
        t0, trimmed = relevant_part(v["text"], v["diseases"])
        c["subsentence_trimmed"] += trimmed
        t, act = strip_temporal(t0)
        if act != "keep":
            c["temporal_" + act] += 1
        if act == "drop":
            v["dropped"] = True
            continue
        t2 = SEIM_STRIP.sub("", t).strip()
        if t2 != t:
            c["seim_stripped"] += 1
            t = re.sub(r"\s+([.,;])", r"\1", t2)
        t, nf = fix_typos(t, typo_fix)
        c["typo_fixed"] += nf
        lines.append(dict(text=t, orig=v["text"], sec=v["sec"], diseases=sorted(v["diseases"]), action=act, pos=v["pos"],
                          oos=bool(OUT_OF_SCOPE.search(t)), size=bool(SIZE.search(t)), seim=bool(SEIM.search(t))))
    covered = {d for ln in lines for d in ln["diseases"]}
    for d in DISEASES:   # disease lost all its sentences through dropping -> fallback to normalized (positives)
        x = diseases[d]
        if x["status"] in ("positive", "uncertain") and d not in covered and x["source_sentences"]:
            ns = (x.get("normalized_sentence") or "").strip()
            if ns:
                c["fallback_norm"] += 1
                pos = min([v["pos"] for v in cand.values() if d in v["diseases"]] or [10 ** 9])
                lines.append(dict(text=ns, orig="", sec="fallback", diseases=[d], action="fallback_norm",
                                  oos=False, size=bool(SIZE.search(ns)), seim=bool(SEIM.search(ns)), pos=pos))
            else:
                c["lost_disease"] += 1
    lines.sort(key=lambda ln: ln["pos"])
    target = "\n".join(ln["text"] for ln in lines)   # doctor punctuation kept as written
    return dict(target=target, lines=lines, counters=c)
