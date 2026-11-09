#!/usr/bin/env python3
"""Merlin report_gen step 2: build per-case, per-disease training targets (doctor's original sentences).

Inputs : disease14/reports.json, disease14/labels_disease14.jsonl, <data>/typo_review.csv
Outputs: <out>/targets_{train,val,test}.jsonl, <out>/stats.json, <out>/sample_val50.jsonl
Header vocabulary and typo dictionary come from train+val only; test is processed with the same frozen rules.

Decision per (case, disease):
  rule status == label status            -> rule sentences, source=rule
  rule not_mentioned, label pos/uncertain -> sentences located from the existing disease14 Qwen quotes, source=qwen_label
  rule not_mentioned, label neg          -> not_mentioned (strict definition); label-only sentences kept in `label_only_neg`
  rule pos/unc/neg != label pos/unc/neg  -> status = label, sentences = rule + quote-located, source=rule+qwen_label, flag conflict
  rule 'other' (about the finding, no clear status) -> status from label (if label nm: uncertain), flag rule_other
  rule mentioned, label not_mentioned    -> rule status kept, flag label_nm
No new Qwen requests are sent by this script.
"""
import argparse, collections, csv, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_lib as L

DATA = "/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin"


def loose(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def locate_quotes(quotes, items, vocab):
    """label quotes -> item indices (quote inside a sentence, or sentence inside a multi-sentence quote)."""
    out = []
    for q in quotes:
        q2, hsec = q, None
        m = re.match(r"^\s*([A-Z][A-Za-z/,&\-() ]{1,60}):\s*", q)
        if m and m.group(1).strip().lower() in vocab:
            q2, hsec = q[m.end():], m.group(1).strip().lower()
        parts = [p for p in re.split(r"\.\.\.|…", q2) if len(loose(p)) >= 6]
        for p in parts:
            lp = loose(p)
            pool = [it for it in items if it["sec"].lower() == hsec] if hsec else items
            hit = [it["i"] for it in pool if lp in loose(it["text"])]
            if not hit:
                hit = [it["i"] for it in items if len(loose(it["text"])) >= 10 and loose(it["text"]) in lp]
            out += hit[:1] if hit and lp in loose(items[hit[0]]["text"]) else hit
    return sorted(set(out))


def with_header(it, text):
    if it["first"] and it["sec"] and it["part"] == "findings":
        return f"{it['sec']}: {text}"
    return text


def new_log():
    d = collections.defaultdict(int)
    d["typo_list"] = []
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qwen-fixes", default=None, help="jsonl {case_id, i, clean} from qwen_rewrite.py (data side)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reps = json.load(open(f"{DATA}/disease14/reports.json"))
    labs = {}
    for l in open(f"{DATA}/disease14/labels_disease14.jsonl"):
        x = json.loads(l)
        labs[x["study_id"]] = x["answer"]
    vocab = L.build_header_vocab([v["report"] for v in reps.values() if v["split"] in ("train", "val")])
    typo_fix = {r["typo"]: r["fix"] for r in csv.DictReader(open(f"{DATA}/report_gen/targets/typo_review.csv")) if r["keep"] == "1"}

    S = {sp: dict(n=0, status=collections.defaultdict(collections.Counter), rule=collections.defaultdict(collections.Counter),
                  pair=collections.defaultdict(collections.Counter), source=collections.defaultdict(collections.Counter),
                  flags=collections.defaultdict(collections.Counter), attr=collections.defaultdict(collections.Counter),
                  attr_n=collections.Counter(), attr_agree=collections.defaultdict(collections.Counter),
                  clean=collections.defaultdict(collections.Counter), clean_total=collections.Counter(),
                  typos=collections.Counter(), lines=[], chars=[], empty_target=0, unparsed_headers=0)
         for sp in ("train", "val", "test")}
    outs = {sp: open(f"{a.out}/targets_{sp}.jsonl", "w") for sp in ("train", "val", "test")}
    qfix = {}
    if a.qwen_fixes and os.path.exists(a.qwen_fixes):
        for l in open(a.qwen_fixes):
            x = json.loads(l)
            qfix[(x["case_id"], x["i"])] = x.get("clean")
    qreq = open(f"{a.out}/qwen_rewrite_requests.jsonl", "w")
    nq = collections.Counter()
    ids = sorted(reps)
    if a.limit:
        ids = ids[:a.limit]
    for k in ids:
        sp = reps[k]["split"]
        st = S[sp]
        st["n"] += 1
        items = L.parse_report(reps[k]["report"], vocab)
        if not any(it["sec"] for it in items):
            st["unparsed_headers"] += 1
        got = L.extract(items)
        rec = dict(case_id=k, split=sp, diseases={})
        used_items = collections.defaultdict(set)    # item index -> diseases using it (target)
        dinfo = {}
        for d in L.DISEASES:
            rule_stat = L.combine([s for _, s in got[d]])
            lab = labs[k][d]
            lab_stat = L.CAT2STATUS[lab["category"]]
            rule_idx = [i for i, _ in got[d]]
            flags = []
            src = "rule"
            label_only_neg = []
            if rule_stat == "not_mentioned":
                if lab_stat in ("pos", "uncertain"):
                    idx = locate_quotes(lab["quotes"], items, vocab)
                    status, src = lab_stat, "qwen_label"
                    if not idx:
                        flags.append("quote_not_located")
                else:
                    idx, status, src = [], "not_mentioned", "none"
                    if lab_stat == "neg":
                        label_only_neg = locate_quotes(lab["quotes"], items, vocab)
                        flags.append("label_neg_rule_nm")
            elif rule_stat == "other" and lab_stat == "not_mentioned":
                idx, status, src = [], "not_mentioned", "none"
                flags.append("rule_other_label_nm")
            elif rule_stat == "other":
                status = lab_stat
                idx = sorted(set(rule_idx) | set(locate_quotes(lab["quotes"], items, vocab)))
                src = "rule+qwen_label" if set(idx) - set(rule_idx) else "rule"
                flags.append("rule_other")
            elif lab_stat == "not_mentioned":
                status, idx = rule_stat, rule_idx
                flags.append("label_nm")
            elif rule_stat != lab_stat:
                status = lab_stat
                qi = locate_quotes(lab["quotes"], items, vocab)
                idx = sorted(set(rule_idx) | set(qi))
                src = "rule+qwen_label" if set(qi) - set(rule_idx) else "rule"
                flags.append("conflict")
            else:
                status, idx = rule_stat, rule_idx
                qi = locate_quotes(lab["quotes"], items, vocab)
                if set(qi) - set(rule_idx):
                    if status in ("pos", "uncertain"):   # positives: add the sentences Qwen quoted that the rule missed
                        idx = sorted(set(idx) | set(qi))
                        src = "rule+qwen_label"
                        flags.append("label_extra_sentence_added")
                    else:
                        flags.append("label_extra_sentence")
            find = [i for i in idx if items[i]["part"] == "findings"]
            imp = [i for i in idx if items[i]["part"] == "impression"]
            use = find if find else imp
            if not find and imp:
                flags.append("impression_only")
            sstat = {i: s for i, s in got[d]}
            if status in ("pos", "uncertain") and find and imp and \
                    not any(sstat.get(i, "quote") in ("pos", "uncertain", "quote", "other") for i in find):
                # findings only hold negative sentences for a positive finding -> add the impression's positive sentences
                add = [i for i in imp if sstat.get(i, "quote") in ("pos", "uncertain", "quote", "other")]
                if add:
                    use = sorted(set(use) | set(add))
                    imp = [i for i in imp if i not in add]
                    flags.append("impression_pos_added")
            for i in use:
                used_items[i].add(d)
            dinfo[d] = dict(status=status, rule=rule_stat, lab=lab, lab_stat=lab_stat, use=use, imp=imp if find else [],
                            src=src if status != "not_mentioned" else "none", flags=flags, label_only_neg=label_only_neg,
                            sent_stat={i: s for i, s in got[d]})
            st["rule"][d][rule_stat] += 1
            st["pair"][d][(rule_stat, lab_stat)] += 1
        # clean every used sentence once
        cleaned = {}
        qwen_used = set()
        for i, ds in used_items.items():
            log = new_log()
            c = L.clean_sentence(items[i]["text"], ds, typo_fix, log)
            if any(dinfo[d]["status"] in ("pos", "uncertain") for d in ds) and L.degraded(items[i]["text"], c):
                nq[sp] += 1
                qreq.write(json.dumps(dict(case_id=k, split=sp, i=i, raw=items[i]["text"], rule_clean=c, diseases=sorted(ds)), ensure_ascii=False) + "\n")
                if (k, i) in qfix:
                    c = qfix[(k, i)] or None
                    if c:   # deterministic se/im + unmeasured-size pass on the Qwen output (kept only if it stays clean)
                        c2 = L._tidy(L.strip_sizes(L.strip_seim(c, new_log()), ds, new_log()))
                        if c2 != c and not L.degraded(c, c2):
                            c = c2
                    qwen_used.add(i)
                    log["qwen_rewrite"] += 1
                    if not c:
                        log["qwen_empty"] += 1
                else:
                    log["degraded_no_qwen"] += 1
            cleaned[i] = (c, log)
            for key, v in log.items():
                if key == "typo_list":
                    for t in v:
                        st["typos"][t] += 1
                elif v:
                    st["clean_total"][key] += 1
                    for d in ds:
                        st["clean"][d][key] += 1
        lines = []
        for i in sorted(used_items):
            c, _ = cleaned[i]
            if c:
                lines.append(dict(i=i, sec=items[i]["sec"], part=items[i]["part"], raw=with_header(items[i], items[i]["text"]),
                                  clean=with_header(items[i], c), diseases=sorted(used_items[i])))
        for d in L.DISEASES:
            x = dinfo[d]
            sents = []
            for i in x["use"]:
                c, _ = cleaned[i]
                sents.append(dict(i=i, part=items[i]["part"], sec=items[i]["sec"], raw=with_header(items[i], items[i]["text"]),
                                  clean=with_header(items[i], c) if c else None, rule_status=x["sent_stat"].get(i, "quote"),
                                  clean_by="qwen" if i in qwen_used else "rule"))
            raw = " ".join(s["raw"] for s in sents)
            clean = " ".join(s["clean"] for s in sents if s["clean"])
            if sents and not clean:
                x["flags"].append("clean_empty")
            status = x["status"]
            ctext = [s["clean"] for s in sents if s["clean"]]
            attrs = L.attributes(d, ctext, status) if status != "not_mentioned" else {}
            attrs_raw = L.attributes(d, [s["raw"] for s in sents], status) if status != "not_mentioned" else {}
            raw_only = {kk: v for kk, v in attrs_raw.items() if L.stated(v) and not L.stated(attrs.get(kk))}
            rec["diseases"][d] = dict(status=status, status_rule=x["rule"], label_category=x["lab"]["category"],
                                      sentence_raw=raw, sentence_clean=clean, attributes=attrs, attributes_raw_only=raw_only,
                                      source=x["src"], flags=x["flags"], sentences=sents,
                                      impression_raw=[items[i]["text"] for i in x["imp"]],
                                      label_only_neg=[items[i]["text"] for i in x["label_only_neg"]])
            st["status"][d][status] += 1
            st["source"][d][x["src"]] += 1
            for f in x["flags"]:
                st["flags"][d][f] += 1
            if status in ("pos", "uncertain"):
                st["attr_n"][d] += 1
                for kk in L.ATTR_KEYS[d]:
                    if L.stated(attrs.get(kk)):
                        st["attr"][d][kk] += 1
                    if L.stated(attrs.get(kk)) or L.stated(raw_only.get(kk)):
                        st["attr"][d][kk + "_incl_raw"] += 1
                # agreement with the disease14 label detail (both stated)
                det = x["lab"].get("detail", {}) or {}
                A = {**attrs, **raw_only}
                cmp = []
                if d == "hydronephrosis":
                    cmp = [("side", A.get("side"), det.get("side")), ("grade", A.get("grade"), det.get("grade"))]
                elif d == "abdominal_aortic_aneurysm":
                    lv = det.get("diameter_cm")
                    cmp = [("diameter_mm", A.get("diameter_mm"), round(lv * 10, 1) if isinstance(lv, (int, float)) else None),
                           ("repaired", A.get("repaired"), det.get("repaired") if det.get("repaired") == "yes" else None)]
                elif d == "biliary_ductal_dilation":
                    cmp = [("site", A.get("site"), det.get("site")), ("cbd_mm", A.get("cbd_mm"), det.get("cbd_mm"))]
                elif d == "ascites":
                    cmp = [("amount", A.get("amount"), det.get("amount"))]
                elif d == "hepatic_steatosis":
                    cmp = [("grade", A.get("grade"), det.get("grade"))]
                elif d == "bowel_obstruction":
                    cmp = [("type", A.get("type"), det.get("type"))]
                elif d == "splenomegaly":
                    lv = det.get("length_cm")
                    cmp = [("length_mm", A.get("length_mm"), round(lv * 10, 1) if isinstance(lv, (int, float)) else None)]
                elif d == "fracture":
                    cmp = [("vertebral_compression", A.get("vertebral_compression"), det.get("vertebral_compression") if det.get("vertebral_compression") == "yes" else None)]
                elif d == "appendicitis":
                    cmp = [("perforated", A.get("perforated"), det.get("perforated")), ("appendicolith", A.get("appendicolith"), det.get("appendicolith"))]
                elif d == "free_air":
                    cmp = [("postoperative", A.get("postoperative"), det.get("postoperative") if det.get("postoperative") == "yes" else None)]
                for name, mine, theirs in cmp:
                    if L.stated(theirs) and theirs != "not_stated":
                        key = f"{name}"
                        st["attr_agree"][d][key + "_label_stated"] += 1
                        if L.stated(mine):
                            st["attr_agree"][d][key + "_both"] += 1
                            ok = (abs(float(mine) - float(theirs)) <= 1.0) if isinstance(theirs, (int, float)) and not isinstance(theirs, bool) else (str(mine) == str(theirs))
                            st["attr_agree"][d][key + "_agree"] += int(ok)
        rec["target_lines"] = lines
        rec["target_text"] = "\n".join(ln["clean"] for ln in lines)
        st["lines"].append(len(lines))
        st["chars"].append(len(rec["target_text"]))
        st["empty_target"] += int(not lines)
        outs[sp].write(json.dumps(rec, ensure_ascii=False) + "\n")
    for f in outs.values():
        f.close()
    qreq.close()
    print("qwen rewrite candidates:", dict(nq), "fixes applied:", len(qfix))
    # stats
    js = {}
    for sp, st in S.items():
        if not st["n"]:
            continue
        n = st["n"]
        js[sp] = dict(n=n, empty_target=st["empty_target"], unparsed_headers=st["unparsed_headers"],
                      lines_mean=sum(st["lines"]) / n, chars_mean=sum(st["chars"]) / n,
                      status={d: dict(st["status"][d]) for d in L.DISEASES}, rule={d: dict(st["rule"][d]) for d in L.DISEASES},
                      pair={d: {f"{r}|{l}": v for (r, l), v in st["pair"][d].items()} for d in L.DISEASES},
                      source={d: dict(st["source"][d]) for d in L.DISEASES}, flags={d: dict(st["flags"][d]) for d in L.DISEASES},
                      attr_n=dict(st["attr_n"]), attr={d: dict(st["attr"][d]) for d in L.DISEASES},
                      attr_agree={d: dict(st["attr_agree"][d]) for d in L.DISEASES},
                      clean={d: dict(st["clean"][d]) for d in L.DISEASES}, clean_total=dict(st["clean_total"]),
                      typos={f"{x}>{y}": v for (x, y), v in st["typos"].most_common()})
    json.dump(js, open(f"{a.out}/stats.json", "w"), ensure_ascii=False, indent=1)
    # val sample for manual audit
    vals = [json.loads(l) for l in open(f"{a.out}/targets_val.jsonl")]
    random.seed(20261109)
    smp = random.sample(vals, min(50, len(vals)))
    with open(f"{a.out}/sample_val50.jsonl", "w") as f:
        for r in smp:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({sp: dict(n=v["n"], empty=v["empty_target"]) for sp, v in js.items()}))


if __name__ == "__main__":
    main()
