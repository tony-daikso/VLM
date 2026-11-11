#!/usr/bin/env python3
"""Agent G: fast CPU proxy evaluation of method-4 generations (no Qwen, ~1 min for val600).

The 14 finding categories and the detail fields of each generated report are extracted with the agent-B report rules
(targets/mt_lib.py: keyword spans, NegEx scope, hedges, section groups, attribute rules) instead of the Qwen
re-extraction; everything else is evaluate.py itself (imported, its extract_all is replaced), so the reference
labels, the owl / B definitions, strict / lenient, critical error, omission, attributes, error-cue test and
BLEU / ROUGE-L are computed by exactly the same code as the official Qwen evaluation.  BERTScore is skipped by default.

Generated text: the {disease: sentence} dict of infer.py (the same rendered text Qwen sees; a sentence that serves
several findings appears once). Each sentence gets the organ section group(s) of the finding tag(s) it was written
for, so mt_lib's section-aware rules work as on a real report.
Rule status -> category: pos -> yes, uncertain -> unclear, neg -> no, other -> unclear (about the finding, no clear
status), not_mentioned -> not_mentioned.

ONLY a screening tool: official numbers stay with the Qwen evaluation (see proxy_calib.py for its agreement).
usage (A6000): python proxy_eval.py --gen gen_f/X.jsonl --name X --out-root eval/results_proxy [evaluate.py options]
Results: <out-root>/<name>/{metrics.json, report.md, per_case_categories.jsonl}; metrics.json carries "proxy": "mt_lib".
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent.parent / "eval", HERE.parent / "eval"):        # repo layout / A6000 code_f layout
    if (p / "evaluate.py").exists():
        sys.path.insert(0, str(p))
        break
for p in (HERE.parent.parent / "targets", HERE.parent / "targets"):
    if (p / "mt_lib.py").exists():
        sys.path.insert(0, str(p))
        break
import evaluate as E  # noqa: E402
import mt_lib as L    # noqa: E402

GRP = {"hepatic_steatosis": {"liver"}, "biliary_ductal_dilation": {"biliary", "liver"}, "gallstones": {"gb"},
       "surgically_absent_gallbladder": {"gb"}, "splenomegaly": {"spleen"}, "hydronephrosis": {"kidney"},
       "bowel_obstruction": {"gi"}, "appendicitis": {"gi"}, "ascites": {"peritoneum"}, "free_air": {"peritoneum"},
       "abdominal_aortic_aneurysm": {"vessels"}, "atherosclerosis": {"vessels"}, "osteopenia": {"msk"},
       "fracture": {"msk"}}
HDR = re.compile(r"^\s*([A-Z][A-Za-z/,&\-() ]{1,60}):\s*")
# Where the Qwen re-extraction (judge) reads generated text differently from the agent-B reference rules (checked on the
# 8 calibration groups, proxy_calib.py): GI section "Normal." -> appendicitis no; small free fluid in the abdomen AND
# pelvis -> ascites yes (pelvis only stays below the cut: Qwen itself splits ~60/40 on that identical sentence);
# focal fatty infiltration / deposition -> steatosis yes; pars defect / spondylolysis and a deformity of a named bone -> fracture yes; AAA < 3.0 cm -> probably_no.
QWEN_ASC_POS = re.compile(r"\bsmall\s+(?:amount|volume)\s+of\s+(?:free\s+)?fluid\b[^.;]{0,30}\babdomen\b", re.I)
QWEN_STEAT_POS = re.compile(r"\bfocal\s+(?:fatty\s+)?(?:fat\s+)?(?:infiltration|deposition)", re.I)
QWEN_FX_POS = re.compile(r"\bpars\s+(?:interarticularis\s+)?(?:defect|fracture|break)\w*|spondylolys\w*|"
                         r"\bdeformit\w*\s+of\s+(?:the\s+)?(?:\w+\s+){0,3}(?:ram[iu]s?|rib|ribs|pubic|sacrum|acetabul\w*|femur|femoral|ilium|iliac|ischi\w*|coccyx)",
                         re.I)
STATUS2CAT = {"pos": "yes", "uncertain": "unclear", "neg": "no", "other": "unclear", "not_mentioned": "not_mentioned"}


def yn(v):
    return v if v in ("yes", "no") else "not_stated"


def detail(d, a, texts=()):
    """mt_lib attribute dict -> the Qwen detail schema of run_disease14.DETAIL (positive findings: a yes/no field the
    text does not support is "no", as Qwen fills it; ileum / jejunum named -> SBO)."""
    t = " ".join(texts)
    mm2cm = lambda v: None if v is None else round(v / 10.0, 2)   # noqa: E731
    if d == "hydronephrosis":
        return {"grade": a.get("grade", "not_stated"), "side": a.get("side", "not_stated")}
    if d == "free_air":
        return {"postoperative": yn(a.get("postoperative"))}
    if d == "bowel_obstruction":
        ty = a.get("type", "not_stated")
        if ty == "not_stated" and a and re.search(r"\b(?:ileum|jejunum)\b", t, re.I) and not re.search(r"colon|large\s+bowel|sigmoid|cecum|rectum", t, re.I):
            ty = "SBO"
        return {"type": ty}
    if d == "abdominal_aortic_aneurysm":
        return {"diameter_cm": mm2cm(a.get("diameter_mm")), "repaired": "yes" if a.get("repaired") == "yes" else ("no" if a else "not_stated")}
    if d == "biliary_ductal_dilation":
        return {"site": a.get("site", "not_stated"), "cbd_mm": a.get("cbd_mm")}
    if d == "ascites":
        return {"amount": a.get("amount", "not_stated"), "hemorrhagic": yn(a.get("hemorrhagic"))}
    if d == "appendicitis":
        return {"appendicolith": yn(a.get("appendicolith")), "perforated": yn(a.get("perforated"))}
    if d == "hepatic_steatosis":
        return {"grade": a.get("grade", "not_stated")}
    if d == "fracture":
        acu = {"acute": "yes", "old": "no", "mixed": "mixed"}.get(a.get("acuity"), "not_stated")
        return {"vertebral_compression": "yes" if a.get("vertebral_compression") == "yes" else ("no" if a.get("sites") else "not_stated"), "sites": a.get("sites", []), "acute": acu}
    if d == "atherosclerosis":
        loc = set(a.get("location") or [])
        return {"aortoiliac": "yes" if loc & {"aorta", "aortoiliac", "iliac"} else "not_stated",
                "grade": a.get("grade", "not_stated")}
    if d == "splenomegaly":
        return {"length_cm": mm2cm(a.get("length_mm"))}
    if d == "gallstones":
        return {"calcified": yn(a.get("calcified"))}
    return {}


def rule_answer(fd, text):
    """generated findings dict (or plain text) -> Qwen-style answer {disease: {category, detail, quotes}}."""
    groups = []                                  # (sentence, groups) in rendered order, identical sentences once
    if fd:
        idx = {}
        keys = [d for d in E.RENDER_ORDER if d in fd] + [d for d in fd if d not in E.RENDER_ORDER]
        for d in keys:
            s = fd.get(d)
            if isinstance(s, list):
                s = " ".join(x for x in s if x)
            s = (s or "").strip()
            if not s:
                continue
            if s in idx:
                groups[idx[s]][1].update(GRP.get(d, {"other"}))
            else:
                idx[s] = len(groups)
                groups.append((s, set(GRP.get(d, {"other"}))))
    else:
        groups = [(ln, {"other"}) for ln in (text or "").splitlines() if ln.strip()]
    items = []
    for s, g in groups:
        m = HDR.match(s)                          # Merlin organ-section header written by the model ("Kidneys: Normal.")
        if m:
            hg = L.header_groups(m.group(1))
            s = s[m.end():]
            g = (hg | g) if hg != {"other"} else g
        for j, ss in enumerate(L.split_sentences(s)):
            items.append(dict(part="findings", sec="", grps=set(g), text=ss, first=(j == 0), i=len(items)))
    got = L.extract(items)
    for it in items:                              # Qwen-reading adjustments (see QWEN_* above)
        t = it["text"]
        if "gi" in it["grps"] and it["first"] and L.BLANKET_SENT.search(t):
            got["appendicitis"].append((it["i"], "neg_blanket"))
        if QWEN_ASC_POS.search(t) and not L.negated(t, QWEN_ASC_POS.search(t).start()):
            got["ascites"].append((it["i"], "pos"))
        if "liver" in it["grps"] and QWEN_STEAT_POS.search(t) and not L.negated(t, QWEN_STEAT_POS.search(t).start()):
            got["hepatic_steatosis"].append((it["i"], "pos"))
        m = QWEN_FX_POS.search(t)
        if m and ("msk" in it["grps"] or re.search(r"\b[TLS]\d|vertebr|rib|pelvi|pubic|ram[iu]|sacr|femo|acetab", t, re.I)) \
                and not L.negated(t, m.start()):
            got["fracture"].append((it["i"], "uncertain" if L.HEDGE_UNCERTAIN.search(t) else "pos"))
    ans = {}
    for d in E.DISEASES:
        st = L.combine([s for _, s in got[d]])
        texts = [items[i]["text"] for i, _ in dict.fromkeys(got[d])]
        st_attr = st if st in ("pos", "uncertain", "neg") else "neg"
        a = L.attributes(d, texts, st_attr) if texts else {}
        cat = STATUS2CAT[st]
        if d == "abdominal_aortic_aneurysm" and cat == "yes" and a.get("diameter_mm") and a["diameter_mm"] < 30:
            cat = "probably_no"                    # Qwen: an "aneurysm" under 3 cm is below the cut
        ans[d] = {"category": cat, "detail": detail(d, a, texts) if st in ("pos", "uncertain") else detail(d, {}), "quotes": texts[:2]}
    return ans


def main():
    argv = sys.argv[1:]
    gen = argv[argv.index("--gen") + 1]
    rows = {E.case_id_of(r): r for r in E.read_jsonl(gen)}

    def extract_all(items, d14, meta, args, prompt_md5):
        out = {}
        for cid, text in items:
            r = rows[cid]
            fd = r["findings"] if isinstance(r.get("findings"), dict) else None
            out[cid] = rule_answer(fd, text)
        print(f"[proxy] {len(out)} texts extracted with mt_lib rules (no Qwen)", flush=True)
        return out

    class _D14:                                   # evaluate.main only needs the module to exist
        pass
    E.extract_all = extract_all
    E.load_d14 = lambda d: _D14()
    if "--url" not in argv:
        argv += ["--url", "none://proxy"]
    sys.argv = [sys.argv[0]] + argv
    E.main()
    name = argv[argv.index("--name") + 1]
    root = argv[argv.index("--out-root") + 1] if "--out-root" in argv else str(E.DATA_EVAL / "results")
    mp = Path(root) / name / "metrics.json"
    m = json.loads(mp.read_text())
    m["proxy"] = "mt_lib"
    m["qwen"] = {**m.get("qwen", {}), "note": "PROXY: categories from targets/mt_lib.py rules, no Qwen"}
    mp.write_text(json.dumps(m, ensure_ascii=False, indent=1, default=float))
    rp = Path(root) / name / "report.md"
    rp.write_text("> **代理評估（mt_lib 規則抽取，未用 Qwen）：只供快速篩選，正式數字以 Qwen 評估為準。**\n\n" + rp.read_text())


if __name__ == "__main__":
    main()
