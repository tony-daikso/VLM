"""
Rule-based replacement for report_parsing.py (Step 2: anatomy-level description extraction).

Input : merlin_report_mention(_strict).json  (output of check_organ_mention_rules.py)
Output: merlin_report_organ_report_*.json    (same format as the official
        merlin_report_organ_report_v1.json: {pid: {"report": ..., organ: description}})

For every organ marked "yes" in Step 1, the description is built from the report's own
sentences, in report order and without duplicates:
  1. sentences inside the organ's own section (e.g. "Spleen: Normal." -> "Normal."),
     where a section header belongs to an organ if the organ's Step 1 regex matches the
     header text (so "Kidneys, ureters, and bladder" -> kidney + bladder);
  2. any other sentence (other sections, IMPRESSION) that names the organ, using the
     same EXPLICIT regex as Step 1.

Usage:
    python report_parsing_rules.py \
        --mention /path/to/merlin_report_mention_strict.json \
        --output  /path/to/merlin_report_organ_report_strict.json
"""
import argparse
import json
import re
from collections import Counter

from check_organ_mention_rules import EXPLICIT, ORGANS

# Candidate "Header: " in a well-punctuated position (report start, after ". " or ": ").
# Used only to learn the header vocabulary; matching then uses the vocabulary alone, because some
# reports drop the period before a header ("Adrenal glands: Normal Kidneys and ureters: Normal").
HEADER_CANDIDATE = re.compile(r"(?:^|(?<=\.\s)|(?<=:\s))([A-Z][A-Za-z ,/&\-]{2,45}):\s")
HEADER = None  # set by build_header_regex()
# Headers that start a non-organ block; sentences after them only count via keywords
RESET_HEADERS = re.compile(r"^(findings|impression|summary|abdomen|pelvis|abdomen and pelvis|abdomen/pelvis)$", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(<])")
LIST_MARK = re.compile(r"^\d{1,2}\.$")  # "1." tokens that open the numbered IMPRESSION list
PATTERNS = {o: re.compile(p, re.IGNORECASE) for o, p in EXPLICIT.items()}


def build_header_regex(reports, min_count=20):
    """Learn section headers that appear >= min_count times, then match them anywhere (longest first)."""
    global HEADER
    counts = Counter(m.group(1).strip().lower() for r in reports for m in HEADER_CANDIDATE.finditer(r))
    vocab = {h for h, n in counts.items() if n >= min_count}
    # Drop fused candidates like "normal kidneys and ureters" / "appears normal pancreas":
    # previous-section text glued to a real header when the period is missing.
    fused = re.compile(r"^.*\b(normal|unremarkable|appears?|negative|clear|bowel|absent|ascites|ends)\s+(.+)$")
    vocab = {h for h in vocab if not ((m := fused.match(h)) and m.group(2) in vocab)}
    vocab -= {"for example", "index lesions as follows"}  # colons inside sentences, not section headers
    vocab = sorted(vocab, key=len, reverse=True)
    HEADER = re.compile(r"(?<![A-Za-z/])(" + "|".join(re.escape(h) for h in vocab) + r"):\s", re.IGNORECASE)
    return vocab


VASCULAR = {"aorta", "iliac artery", "iliac vena", "inferior vena cava", "portal vein", "pulmonary artery"}
BOWEL = {"small bowel", "large bowel"}
# Section kinds that restrict keyword attribution (tuned against 300 LLM-extracted reports, md section 10)
KIND_LYMPH = re.compile(r"lymph", re.I)                       # node locations ("peripancreatic nodes") -> no organ
KIND_VASC = re.compile(r"vascul|cardiovascular|aorta", re.I)  # vessel names ("splenic artery") -> vessels only
KIND_BOWEL = re.compile(r"bowel|gastrointestinal", re.I)      # generic "No bowel obstruction" -> small + large bowel
KIND_MSK = re.compile(r"musculoskeletal|bones", re.I)         # generic "No suspicious osseous lesion" -> vertebrae
IMPRESSION_HEADERS = re.compile(r"^(impression|summary)$", re.I)
VERTEBRAE = {"cervical vertebrae", "thoracic vertebrae", "lumbar vertebrae"}

# Options chosen by evaluating against 300 LLM-extracted reports (md section 10)
OPTS = {
    "dedup_impression": 0.3,   # drop an IMPRESSION sentence if >= this share of its words already appear in the organ's findings text
    "msk_generic": False,      # attribute organ-less Musculoskeletal sentences to mentioned vertebrae
    # when the organ has its own section, ignore keyword hits in OTHER findings sections
    # (e.g. "gallbladder fossa" under Liver); None = off, "findings" = findings only, "all" = also IMPRESSION
    "own_section_priority": "all",
}
STOPWORDS = set("a an the of and or with without in on at to for is are was were be been there this that these those "
                "which by as from no not than likely may".split())


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def header_info(header):
    """(organs owning the section, section kind, starts IMPRESSION?)."""
    h = header.strip()
    if RESET_HEADERS.match(h):
        return set(), None, bool(IMPRESSION_HEADERS.match(h))
    kind = ("lymph" if KIND_LYMPH.search(h) else "vasc" if KIND_VASC.search(h) else
            "bowel" if KIND_BOWEL.search(h) else "msk" if KIND_MSK.search(h) else None)
    return {o for o, pat in PATTERNS.items() if pat.search(h)}, kind, False


def split_sections(report):
    """Return [(organs_of_section, section_kind, in_impression, sentence), ...] in report order."""
    out = []
    pos, section, kind, imp = 0, set(), None, False
    headers = list(HEADER.finditer(report))
    bounds = [(m.start(), m.end(), header_info(m.group(1))) for m in headers] + [(len(report), len(report), None)]
    for start, end, info in bounds:
        chunk = report[pos:start].strip()
        for sent in SENT_SPLIT.split(chunk) if chunk else []:
            sent = sent.strip()
            if not sent:
                continue
            if LIST_MARK.match(sent):  # numbered impression starts -> leave the current organ section
                section, kind, imp = set(), None, True
                continue
            out.append((section, kind, imp, sent))
        if info is not None:
            section, kind, starts_imp = info
            imp = imp or starts_imp
        pos = end
    return out


def restricted(organ, kind):
    """Keyword hits in lymph-node sections (and vascular sections, for non-vessel organs) are usually locations."""
    return kind == "lymph" or (kind == "vasc" and organ not in VASCULAR)


def extract(report, organs):
    sents = split_sections(report)
    desc = {}
    for organ in organs:
        picked, fallback = [], []  # (report order index, in_impression, sentence)
        has_specific = False
        for i, (section, kind, imp, sent) in enumerate(sents):
            organless = not any(p.search(sent) for p in PATTERNS.values())
            generic = organless and ((organ in BOWEL and kind == "bowel") or
                                     (OPTS["msk_generic"] and organ in VERTEBRAE and kind == "msk"))
            if organ in section or generic:
                picked.append((i, imp, sent))
                has_specific |= organ in section
            elif PATTERNS[organ].search(sent):
                if restricted(organ, kind):
                    fallback.append((i, imp, sent))
                else:
                    picked.append((i, imp, sent))
                    has_specific = True
        own = any(organ in sents[i][0] for i, _, _ in picked)
        mode = OPTS["own_section_priority"]
        if own and mode:
            picked = [(i, imp, s) for i, imp, s in picked
                      if organ in sents[i][0] or (imp and mode == "findings")]
        # restricted-section hits are used only when no specific sentence describes the organ
        # (e.g. "esophageal varices" only under Vasculature); generic sentences do not count as specific
        chosen = picked if has_specific else sorted(picked + fallback)
        # IMPRESSION often restates a finding: keep it only if it adds new words to this organ's findings text
        thr = OPTS["dedup_impression"]
        if thr is not None:
            findings_words = set().union(*(words(s) for _, imp, s in chosen if not imp))
            if findings_words:
                chosen = [(i, imp, s) for i, imp, s in chosen
                          if not imp or len(words(s) & findings_words) < thr * max(len(words(s)), 1)]
        seen, out = set(), []
        for _, _, sent in chosen:
            if sent.lower() not in seen:  # also drops ALL-CAPS repeats
                seen.add(sent.lower())
                out.append(sent)
        desc[organ] = " ".join(out)
    return desc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mention", default="./merlin_report_mention_strict.json")
    parser.add_argument("--output", default="./merlin_report_organ_report_strict.json")
    args = parser.parse_args()

    mention = json.load(open(args.mention))
    vocab = build_header_regex([v["report"] for v in mention.values()])
    print(f"learned {len(vocab)} section headers")
    new_info, empty = {}, Counter()
    for pid, v in mention.items():
        organs = [o for o in ORGANS if v["mention"][o] == "yes"]
        desc = extract(v["report"], organs)
        new_info[pid] = {"report": v["report"]}
        for organ in organs:
            if desc[organ]:
                new_info[pid][organ] = desc[organ]
            else:
                empty[organ] += 1  # mentioned but no sentence found: leave it out (-> "normal." in training)
    json.dump(new_info, open(args.output, "w"), ensure_ascii=False, indent=4)

    n_desc = sum(len(v) - 1 for v in new_info.values())
    lens = sorted(len(d) for v in new_info.values() for k, d in v.items() if k != "report")
    print(f"processed {len(new_info)} reports, {n_desc} organ descriptions, "
          f"median {lens[len(lens) // 2]} chars, p95 {lens[int(len(lens) * .95)]} chars")
    print(f"mentioned but empty: {sum(empty.values())} {dict(empty.most_common())}")
    print(f"saved to {args.output}")
