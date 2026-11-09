#!/usr/bin/env python3
"""Shared evaluator for Merlin 14-disease findings generation (method 1 templates, method 4 LLM, ablations).

Same Qwen recipe as the reference labels: the generated text is re-labelled with
merlin/disease14/run_disease14.py (PROMPT = prompt.md, schema(), ask(), patient_line()) -- imported, not copied,
and the md5 of both files is written into metrics.json.

Inputs
  --gen       generated jsonl, one row per case:
                {"case_id": ..., "findings": {disease: sentence, ...}}   (sentences rendered in organ order, de-duplicated)
             or {"case_id": ..., "text": "..."}                          (also accepted: report / draft / generated)
  --labels    labels_disease14.jsonl (reference categories + detail, same Qwen run that made the OWL labels)
  --targets   agent B targets_{split}.jsonl (reference doctor sentences; for BLEU / ROUGE-L / BERTScore and the
              vertebral-level attribute). Optional: without it only the Qwen metrics are computed.
  --meta      metadata.csv (patient line of the Qwen user message, exactly as for the labels)

Metrics (see README.md)
  * presence: Qwen re-extracted category of the generated text vs reference labels, per class + micro/macro:
      sens / spec / PPV / F1; strict (gen yes|probably_yes) and lenient (+unclear);
      reference definitions: "owl" (pos yes|probably_yes; neg no|probably_no|not_mentioned; unclear excluded) = the
      binarisation the OWL classifiers were trained on, and "B" (neg only no|probably_no) = CG eval_phase1.
  * critical error: reference positive, generated explicitly negative (no|probably_no); plus omission
      (reference positive, generated not_mentioned). Rates over reference positives, per class + overall.
  * attributes: on cases positive in both, detail fields of the re-extraction vs the reference detail
      (side, grade, AAA diameter, spleen length, CBD, ascites amount, SBO/LBO, fracture sites ...) and vertebral
      levels parsed from the fracture sentences.
  * style: corpus BLEU-1..4, mean ROUGE-L F1, BERTScore P/R/F (own implementation of Zhang et al. 2020 =
      bert_score defaults for English: roberta-large layer 17, no idf, no baseline rescale).

Qwen requests are cached (--cache, keyed by prompt md5 + model + user message), so re-running is free.
Concurrency is capped at MAX_JOBS = 8 (shared endpoints; use <=4 when another project is using the server).
The test split is refused unless --allow-test (only after the user unlocks test).

usage (Mac, Qwen through an ssh tunnel: ssh -N -L 18003:127.0.0.1:8003 sparks):
  python evaluate.py --gen $MERLIN_DATA/report_gen/method1/val_drafts.jsonl --name method1 \
      --targets $MERLIN_DATA/report_gen/targets/targets_val.jsonl \
      --url http://127.0.0.1:18003/v1 --jobs 8
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# data root (labels, reports, results, Qwen cache: DUA content stays under data/, never in the code folder / git)
MERLIN = Path(os.environ.get("MERLIN_DATA", "/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin"))
D14_DEFAULT = MERLIN / "disease14"
DATA_EVAL = MERLIN / "report_gen" / "eval"

DISEASES = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "hydronephrosis",
            "abdominal_aortic_aneurysm", "atherosclerosis", "osteopenia", "bowel_obstruction", "appendicitis",
            "ascites", "free_air", "gallstones", "biliary_ductal_dilation", "fracture"]
ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊切除術後", "splenomegaly": "脾腫大",
      "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤", "atherosclerosis": "動脈粥狀硬化",
      "osteopenia": "骨質減少", "bowel_obstruction": "腸阻塞", "appendicitis": "闌尾炎", "ascites": "腹水",
      "free_air": "腹腔游離氣體", "gallstones": "膽結石", "biliary_ductal_dilation": "膽管擴張", "fracture": "骨折"}
# organ order of a Merlin FINDINGS section (same as disease14/make_report_sft.py), used to render dict findings
RENDER_ORDER = ["hepatic_steatosis", "biliary_ductal_dilation", "gallstones", "surgically_absent_gallbladder",
                "splenomegaly", "hydronephrosis", "bowel_obstruction", "appendicitis", "ascites", "free_air",
                "abdominal_aortic_aneurysm", "atherosclerosis", "osteopenia", "fracture"]
POS = {"yes", "probably_yes"}
NEG_EXPLICIT = {"no", "probably_no"}
CATS = ["yes", "probably_yes", "unclear", "probably_no", "no", "not_mentioned"]
MAX_JOBS = 8  # shared endpoints: <=4 while CG used them; 8 allowed on Spark merlin-qwen-fp8 (11-09)

# attribute checks: (disease, field, kind, arg)
#   cat: exact; ord: exact + within one step (arg = order); num: |diff| <= arg; set: exact set; levels: regex
ORD_GRADE = ["mild", "moderate", "severe"]
ATTRS = [
    ("hydronephrosis", "side", "cat", None),
    ("hydronephrosis", "grade", "ord", ORD_GRADE),
    ("hepatic_steatosis", "grade", "ord", ORD_GRADE),
    ("splenomegaly", "length_cm", "num", 1.0),
    ("abdominal_aortic_aneurysm", "diameter_cm", "num", 0.3),
    ("abdominal_aortic_aneurysm", "repaired", "cat", None),
    ("biliary_ductal_dilation", "site", "cat", None),
    ("biliary_ductal_dilation", "cbd_mm", "num", 2.0),
    ("ascites", "amount", "ord", ["trace", "small", "moderate", "large"]),
    ("ascites", "hemorrhagic", "cat", None),
    ("bowel_obstruction", "type", "cat", None),
    ("appendicitis", "perforated", "cat", None),
    ("appendicitis", "appendicolith", "cat", None),
    ("free_air", "postoperative", "cat", None),
    ("gallstones", "calcified", "cat", None),
    ("atherosclerosis", "grade", "ord", ORD_GRADE),
    ("atherosclerosis", "aortoiliac", "cat", None),
    ("fracture", "sites", "set", None),
    ("fracture", "vertebral_compression", "cat", None),
    ("fracture", "acute", "cat", None),
    ("fracture", "levels", "levels", None),
]
ATTR_ZH = {"side": "側別", "grade": "程度", "length_cm": "脾臟長度(±1 cm)", "diameter_cm": "直徑(±0.3 cm)",
           "repaired": "已修補", "site": "部位", "cbd_mm": "總膽管徑(±2 mm)", "amount": "量", "hemorrhagic": "出血性",
           "type": "類型(SBO/LBO)", "perforated": "穿孔", "appendicolith": "糞石", "postoperative": "術後",
           "calcified": "鈣化", "aortoiliac": "主髂動脈", "sites": "骨折部位", "vertebral_compression": "椎體壓迫性",
           "acute": "急性/陳舊", "levels": "椎體節段"}


# --------------------------------------------------------------------------------------------- io helpers
def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def read_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_d14(d14_dir):
    """Import run_disease14.py (it reads prompt.md next to itself) -> module with PROMPT, schema, ask, patient_line."""
    spec = importlib.util.spec_from_file_location("run_disease14", str(Path(d14_dir) / "run_disease14.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert list(mod.DETAIL) and set(mod.FINDINGS) == set(DISEASES), "disease14 schema mismatch"
    return mod


def case_id_of(r):
    for k in ("case_id", "study_id", "id"):
        if k in r:
            return str(r[k])
    raise KeyError(f"no case_id in row keys {list(r)[:8]}")


def render_findings(findings: dict) -> str:
    """dict {disease: sentence} -> text, organ order, identical sentences only once (one sentence may cover several)."""
    seen, lines = set(), []
    keys = [d for d in RENDER_ORDER if d in findings] + [d for d in findings if d not in RENDER_ORDER]
    for d in keys:
        s = findings.get(d)
        if isinstance(s, list):
            s = " ".join(x for x in s if x)
        s = (s or "").strip()
        if s and s not in seen:
            seen.add(s)
            lines.append(s)
    return "\n".join(lines)


def gen_text(r):
    if isinstance(r.get("findings"), dict):
        return render_findings(r["findings"]), r["findings"]
    for k in ("text", "report", "draft", "generated", "findings"):
        if isinstance(r.get(k), str):
            return r[k].strip(), None
    raise KeyError(f"generated row {case_id_of(r)} has neither findings dict nor text")


def target_sentences(r):
    """Agent B target row -> {disease: sentence_clean}. Accepts the per-disease dict at the top level or under
    diseases / findings / per_disease / targets."""
    cont = None
    for k in ("diseases", "findings", "per_disease", "targets"):
        if isinstance(r.get(k), dict):
            cont = r[k]
            break
    if cont is None:
        cont = {d: r[d] for d in DISEASES if d in r}
    out = {}
    for d in DISEASES:
        v = cont.get(d)
        if isinstance(v, dict):
            s = v.get("sentence_clean") or v.get("sentence") or ""
        elif isinstance(v, str):
            s = v
        else:
            s = ""
        if isinstance(s, list):
            s = " ".join(x for x in s if x)
        if s and s.strip():
            out[d] = s.strip()
    return out


def target_text(r):
    for k in ("target_text", "target"):
        if isinstance(r.get(k), str) and r[k].strip():
            return r[k].strip()
    return render_findings(target_sentences(r))


# --------------------------------------------------------------------------------------------- Qwen re-extraction
class QwenCache:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.d = {}
        if self.path.exists():           # tolerant: several evaluate.py runs may append to the same cache
            with open(self.path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("parse_ok") or r["key"] not in self.d:
                        self.d[r["key"]] = r
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, k):
        return self.d.get(k)

    def put(self, row):
        with self.lock:
            self.d[row["key"]] = row
            with open(self.path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_all(items, d14, meta, args, prompt_md5):
    """items: [(case_id, text)] -> {case_id: answer dict or None}. Cached; at most MAX_JOBS concurrent requests."""
    cache = QwenCache(args.cache)
    jobs = min(args.jobs, MAX_JOBS)
    ns = argparse.Namespace(url=args.url, max_tokens=args.max_tokens)
    out, todo = {}, []
    for cid, text in items:
        user = f"{d14.patient_line(meta.get(cid))}\n\n# Report\n\n{text}"
        key = hashlib.sha256(json.dumps([prompt_md5, args.qwen_model_tag, args.max_tokens, user]).encode()).hexdigest()
        hit = cache.get(key)
        if hit is not None and hit.get("parse_ok"):
            out[cid] = hit["answer"]
        else:
            todo.append((cid, text, key))
    print(f"[qwen] {len(items)} texts: {len(items) - len(todo)} cached, {len(todo)} to request (jobs={jobs})",
          flush=True)
    if todo and args.no_request:
        print("[qwen] --no-request: uncached texts are left unscored", flush=True)
        todo = []
    t0, n = time.time(), [0]

    def one(job):
        cid, text, key = job
        try:
            r = d14.ask(cid, text, meta.get(cid), ns)
            ok = r["finish"] == "stop" and isinstance(r["answer"], dict)
            row = {"key": key, "case_id": cid, "parse_ok": ok, "answer": r["answer"], "usage": r["usage"],
                   "finish": r["finish"], "sec": r["sec"]}
        except Exception as e:  # noqa: BLE001  logged; a rerun retries (failures are not cached as ok)
            row = {"key": key, "case_id": cid, "parse_ok": False, "error": str(e)[:300]}
        cache.put(row)
        if row["parse_ok"]:
            out[cid] = row["answer"]
        n[0] += 1
        if n[0] % 10 == 0 or n[0] == len(todo):
            el = time.time() - t0
            print(f"[qwen] {n[0]}/{len(todo)}  {el / n[0]:.1f}s/text (wall, {jobs} jobs)", flush=True)

    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(one, todo))
    return out


# --------------------------------------------------------------------------------------------- metrics
def ref_label(cat, definition):
    if cat in POS:
        return 1.0
    if cat in NEG_EXPLICIT:
        return 0.0
    if definition == "owl" and cat == "not_mentioned":
        return 0.0
    return float("nan")


def prf(pred, y):
    keep = ~np.isnan(y)
    p, yy = pred[keep], y[keep] == 1
    tp, fp = int((p & yy).sum()), int((p & ~yy).sum())
    fn, tn = int((~p & yy).sum()), int((~p & ~yy).sum())
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return dict(N=int(keep.sum()), n_pos=int(yy.sum()), TP=tp, FP=fp, FN=fn, TN=tn, sens=div(tp, tp + fn),
                spec=div(tn, tn + fp), PPV=div(tp, tp + fp), F1=div(2 * tp, 2 * tp + fp + fn))


LEVEL_RE = re.compile(r"\b([TLSC])\s?(\d{1,2})(?:\s*(?:-|–|to|through)\s*([TLSC])?\s?(\d{1,2}))?\b")
SPINE = [f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)] + \
        [f"S{i}" for i in range(1, 6)]


def vertebral_levels(text):
    """'L1 and T12 compression fractures', 'T11-L1', 'L2-4' -> {'T11','T12','L1',...}. se/im numbers never match
    (they have no T/L/S/C prefix)."""
    out = set()
    for a, i, b, j in LEVEL_RE.findall(text or ""):
        s = f"{a}{int(i)}"
        if s not in SPINE:
            continue
        if not j:
            out.add(s)
            continue
        e = f"{b or a}{int(j)}"
        if e in SPINE and SPINE.index(e) >= SPINE.index(s) and SPINE.index(e) - SPINE.index(s) <= 12:
            out.update(SPINE[SPINE.index(s):SPINE.index(e) + 1])
        else:
            out.add(s)
    return out


def stated(v):
    if v is None or v == "not_stated":
        return False
    if isinstance(v, (list, set)) and not v:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return True


def attr_compare(kind, arg, rv, gv):
    """-> (exact, within) for a stated reference value; gen not stated counts as wrong."""
    if not stated(gv):
        return False, False
    if kind in ("cat",):
        return rv == gv, rv == gv
    if kind == "ord":
        if rv not in arg or gv not in arg:
            return rv == gv, rv == gv
        d = abs(arg.index(rv) - arg.index(gv))
        return d == 0, d <= 1
    if kind == "num":
        try:
            ok = abs(float(rv) - float(gv)) <= arg + 1e-9
        except (TypeError, ValueError):
            ok = False
        return ok, ok
    if kind in ("set", "levels"):
        rs, gs = set(rv), set(gv)
        return rs == gs, bool(rs & gs)
    raise ValueError(kind)


# ---------------------------------------------------------------- BLEU / ROUGE-L (pure python, lower-cased words)
TOK_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?|[^\sa-z0-9]")


def tok(s):
    return TOK_RE.findall((s or "").lower())


def ngrams(t, n):
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    """Papineni et al. corpus BLEU (single reference), returns BLEU-1..4 with their own brevity penalty."""
    match, total = [0] * max_n, [0] * max_n
    hl = rl = 0
    for h, r in zip(hyps, refs):
        hl, rl = hl + len(h), rl + len(r)
        for n in range(1, max_n + 1):
            hc, rc = ngrams(h, n), ngrams(r, n)
            match[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            total[n - 1] += max(len(h) - n + 1, 0)
    if hl == 0:
        return {f"BLEU-{n}": 0.0 for n in range(1, max_n + 1)}
    bp = 1.0 if hl > rl else math.exp(1 - rl / hl)
    out = {}
    for n in range(1, max_n + 1):
        ps = [match[k] / total[k] if total[k] and match[k] else 0.0 for k in range(n)]
        out[f"BLEU-{n}"] = bp * math.exp(sum(math.log(p) for p in ps) / n) if all(ps) else 0.0
    return out


def lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(h, r):
    l_ = lcs(h, r)
    if not l_:
        return 0.0
    p, rc = l_ / len(h), l_ / len(r)
    return 2 * p * rc / (p + rc)


def bertscore(cands, refs, model_path, layer=17, batch=16, device="cpu", threads=4):
    """BERTScore P/R/F per pair (greedy cosine matching of layer-`layer` token embeddings, special tokens dropped,
    no idf, no rescale) = bert_score defaults for lang='en' when model_path is roberta-large and layer 17."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(threads)
    tk = AutoTokenizer.from_pretrained(model_path)
    m = AutoModel.from_pretrained(model_path).to(device).eval()
    is_roberta = "roberta" in type(m).__name__.lower()

    def emb(texts):
        texts = [(" " + t.strip()) if is_roberta else t.strip() for t in texts]
        enc = tk(texts, padding=True, truncation=True, max_length=510, return_tensors="pt",
                 return_special_tokens_mask=True).to(device)
        sp = enc.pop("special_tokens_mask").bool()
        with torch.no_grad():
            hs = m(**enc, output_hidden_states=True).hidden_states[layer]
        hs = torch.nn.functional.normalize(hs.float(), dim=-1)
        keep = enc["attention_mask"].bool() & ~sp
        return [hs[i][keep[i]] for i in range(len(texts))]

    P, R, F = [], [], []
    for i in range(0, len(cands), batch):
        ce, re_ = emb(cands[i:i + batch]), emb(refs[i:i + batch])
        for c, r in zip(ce, re_):
            if len(c) == 0 or len(r) == 0:
                P.append(0.0), R.append(0.0), F.append(0.0)
                continue
            s = c @ r.T
            p, rr = s.max(1).values.mean().item(), s.max(0).values.mean().item()
            P.append(p), R.append(rr), F.append(2 * p * rr / (p + rr) if p + rr > 0 else 0.0)
    return np.array(P), np.array(R), np.array(F)


# --------------------------------------------------------------------------------------------- main
def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--name", required=True, help="system name; results go to <out-root>/<name>/")
    ap.add_argument("--labels", default=str(D14_DEFAULT / "labels_disease14.jsonl"))
    ap.add_argument("--meta", default=str(D14_DEFAULT / "metadata.csv"))
    ap.add_argument("--targets", default="", help="agent B targets_{split}.jsonl (optional)")
    ap.add_argument("--disease14-dir", default=str(D14_DEFAULT), help="dir with run_disease14.py + prompt.md")
    ap.add_argument("--split", default="val")
    ap.add_argument("--allow-test", action="store_true", help="ONLY after the user unlocked the test split")
    ap.add_argument("--ids", default="", help="optional id list (one per line) to restrict to")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--url", default="http://127.0.0.1:18003/v1", help="Spark merlin-qwen-fp8 via ssh -L 18003:127.0.0.1:8003 sparks")
    ap.add_argument("--jobs", type=int, default=2, help=f"concurrent Qwen requests (capped at {MAX_JOBS})")
    ap.add_argument("--max-tokens", type=int, default=6000, help="same as the label run")
    ap.add_argument("--qwen-model-tag", default="Qwen3.8-27B-FP8", help="part of the cache key")
    ap.add_argument("--cache", default=str(DATA_EVAL / "cache" / "qwen_extract_cache.jsonl"))
    ap.add_argument("--no-qwen", action="store_true", help="text metrics only")
    ap.add_argument("--no-request", action="store_true", help="use cached extractions only, send nothing")
    ap.add_argument("--bertscore-model", default="", help="local dir (e.g. A6000 ~/Desktop/Tony/models/roberta-large)")
    ap.add_argument("--bertscore-layer", type=int, default=17)
    ap.add_argument("--bertscore-threads", type=int, default=4)
    ap.add_argument("--out-root", default=str(DATA_EVAL / "results"))
    a = ap.parse_args()

    out = Path(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=True)
    if a.split == "test" and not a.allow_test:
        sys.exit("test split is locked; use --allow-test only after the user unlocked it")

    # ---- reference labels
    labels, lab_split = {}, {}
    for r in read_jsonl(a.labels):
        labels[r["study_id"]] = r["answer"]
        lab_split[r["study_id"]] = r.get("split")
    gen_rows = read_jsonl(a.gen)
    ids_keep = set(l.strip() for l in open(a.ids) if l.strip()) if a.ids else None
    rows, dropped = [], Counter()
    for r in gen_rows:
        cid = case_id_of(r)
        sp = lab_split.get(cid)
        if sp == "test" and not a.allow_test:
            dropped["test (locked)"] += 1
            continue
        if sp != a.split:
            dropped[f"split={sp}"] += 1
            continue
        if ids_keep is not None and cid not in ids_keep:
            dropped["not in --ids"] += 1
            continue
        rows.append(r)
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(rows)} generated cases in split {a.split}; dropped {dict(dropped)}", flush=True)
    gen = {}
    for r in rows:
        txt, fd = gen_text(r)
        gen[case_id_of(r)] = (txt, fd)
    ids = list(gen)

    tgt = {}
    if a.targets:
        for r in read_jsonl(a.targets):
            tgt[case_id_of(r)] = r

    metrics = {"name": a.name, "gen_file": str(a.gen), "split": a.split, "n_cases": len(ids),
               "dropped": dict(dropped), "time": time.strftime("%Y-%m-%d %H:%M")}

    # ---- Qwen re-extraction
    ext = {}
    if not a.no_qwen:
        d14 = load_d14(a.disease14_dir)
        pm = md5(Path(a.disease14_dir) / "prompt.md")
        metrics["qwen"] = {"prompt_md5": pm, "run_disease14_md5": md5(Path(a.disease14_dir) / "run_disease14.py"),
                           "url": a.url, "model_tag": a.qwen_model_tag, "max_tokens": a.max_tokens,
                           "temperature": 0, "thinking": False}
        meta = {r["study id"]: r for r in csv.DictReader(open(a.meta))}
        ext = extract_all([(c, gen[c][0]) for c in ids], d14, meta, a, pm)
        ok_ids = [c for c in ids if c in ext and c in labels]
        metrics["qwen"]["n_extracted_ok"] = len(ok_ids)
        metrics["qwen"]["n_failed_or_unlabelled"] = len(ids) - len(ok_ids)

        per_case = []
        for c in ok_ids:
            per_case.append({"case_id": c, **{d: [labels[c][d]["category"], ext[c][d]["category"]] for d in DISEASES}})
        with open(out / "per_case_categories.jsonl", "w") as f:  # [reference, generated]; no report text
            for r in per_case:
                f.write(json.dumps(r) + "\n")

        # presence + critical errors
        pres, crit = [], []
        agree6 = {}
        for definition in ("owl", "B"):
            allp = {m: [np.zeros(0, bool), np.zeros(0)] for m in ("strict", "lenient")}
            for d in DISEASES:
                rc = np.array([labels[c][d]["category"] for c in ok_ids])
                gc = np.array([ext[c][d]["category"] for c in ok_ids])
                y = np.array([ref_label(x, definition) for x in rc])
                for mode, s in (("strict", POS), ("lenient", POS | {"unclear"})):
                    p = np.isin(gc, list(s))
                    pres.append(dict(definition=definition, mode=mode, disease=d, 疾病=ZH[d], **prf(p, y)))
                    allp[mode][0] = np.concatenate([allp[mode][0], p])
                    allp[mode][1] = np.concatenate([allp[mode][1], y])
            for mode in ("strict", "lenient"):
                pres.append(dict(definition=definition, mode=mode, disease="ALL_micro", 疾病="合計（micro）",
                                 **prf(*allp[mode])))
                sub = [r for r in pres if r["definition"] == definition and r["mode"] == mode and r["disease"] in DISEASES]
                pres.append(dict(definition=definition, mode=mode, disease="ALL_macro", 疾病="平均（macro）",
                                 **{k: float(np.nanmean([r[k] for r in sub])) for k in ("sens", "spec", "PPV", "F1")}))
        tp = tc = to = 0
        for d in DISEASES:
            rc = np.array([labels[c][d]["category"] for c in ok_ids])
            gc = np.array([ext[c][d]["category"] for c in ok_ids])
            rpos = np.isin(rc, list(POS))
            c_ = int((rpos & np.isin(gc, list(NEG_EXPLICIT))).sum())
            o_ = int((rpos & (gc == "not_mentioned")).sum())
            npos = int(rpos.sum())
            tp, tc, to = tp + npos, tc + c_, to + o_
            crit.append(dict(disease=d, 疾病=ZH[d], ref_pos=npos, critical=c_,
                             critical_rate=c_ / npos if npos else float("nan"), omitted=o_,
                             omission_rate=o_ / npos if npos else float("nan")))
            agree6[d] = float(np.mean(rc == gc)) if len(rc) else float("nan")
        crit.append(dict(disease="ALL", 疾病="合計", ref_pos=tp, critical=tc, critical_rate=tc / tp if tp else float("nan"),
                         omitted=to, omission_rate=to / tp if tp else float("nan")))
        metrics["presence"] = pres
        metrics["critical"] = crit
        metrics["category_agreement_6way"] = agree6
        allr = [labels[c][d]["category"] for c in ok_ids for d in DISEASES]
        allg = [ext[c][d]["category"] for c in ok_ids for d in DISEASES]
        metrics["category_agreement_6way_all"] = float(np.mean([x == y for x, y in zip(allr, allg)])) if allr else float("nan")
        bin_agree = {}
        for d in DISEASES:
            yr = np.array([ref_label(labels[c][d]["category"], "owl") for c in ok_ids])
            yg = np.array([ref_label(ext[c][d]["category"], "owl") for c in ok_ids])
            k = ~np.isnan(yr) & ~np.isnan(yg)
            bin_agree[d] = float(np.mean(yr[k] == yg[k])) if k.any() else float("nan")
        metrics["binary_agreement_owl"] = bin_agree

        # attributes
        arows = []
        for d, field, kind, arg in ATTRS:
            n_elig = n_gen_stated = n_exact = n_within = n_halluc = n_both = 0
            for c in ok_ids:
                if labels[c][d]["category"] not in POS or ext[c][d]["category"] not in POS:
                    continue
                n_both += 1
                if kind == "levels":
                    tsent = target_sentences(tgt[c]).get(d) if c in tgt else None
                    rv = vertebral_levels(tsent if tsent else " ".join(labels[c][d].get("quotes", [])))
                    fd = gen[c][1]
                    gv = vertebral_levels(fd.get(d, "") if fd else " ".join(ext[c][d].get("quotes", [])))
                else:
                    rv = labels[c][d]["detail"].get(field)
                    gv = ext[c][d]["detail"].get(field)
                if not stated(rv):
                    n_halluc += int(stated(gv))
                    continue
                n_elig += 1
                n_gen_stated += int(stated(gv))
                e, w = attr_compare(kind, arg, rv, gv)
                n_exact += int(e)
                n_within += int(w)
            arows.append(dict(disease=d, 疾病=ZH[d], field=field, 屬性=ATTR_ZH[field], kind=kind, both_pos=n_both,
                              ref_stated=n_elig, gen_stated=n_gen_stated,
                              acc=n_exact / n_elig if n_elig else float("nan"),
                              acc_if_stated=n_exact / n_gen_stated if n_gen_stated else float("nan"),
                              within=n_within / n_elig if n_elig else float("nan"),
                              gen_only_stated=n_halluc,
                              gen_only_rate=n_halluc / (n_both - n_elig) if n_both - n_elig else float("nan")))
        metrics["attributes"] = arows

    # ---- style metrics (needs targets)
    if tgt:
        pair_ids = [c for c in ids if c in tgt and target_text(tgt[c])]
        hyp_txt = [gen[c][0] for c in pair_ids]
        ref_txt = [target_text(tgt[c]) for c in pair_ids]
        H, R = [tok(x) for x in hyp_txt], [tok(x) for x in ref_txt]
        style = {"n_pairs": len(pair_ids), "n_no_target": len(ids) - len(pair_ids), **corpus_bleu(H, R),
                 "ROUGE-L": float(np.mean([rouge_l(h, r) for h, r in zip(H, R)])) if pair_ids else float("nan"),
                 "mean_words_gen": float(np.mean([len(h) for h in H])) if H else float("nan"),
                 "mean_words_ref": float(np.mean([len(r) for r in R])) if R else float("nan")}
        if a.bertscore_model and pair_ids:
            t0 = time.time()
            P, Rr, F = bertscore(hyp_txt, ref_txt, a.bertscore_model, a.bertscore_layer,
                                 threads=a.bertscore_threads)
            style.update({"BERTScore_P": float(P.mean()), "BERTScore_R": float(Rr.mean()),
                          "BERTScore_F": float(F.mean()),
                          "BERTScore_model": f"{a.bertscore_model} layer {a.bertscore_layer}, no idf, no rescale",
                          "BERTScore_sec": round(time.time() - t0, 1)})
        # per disease, sentence level (dict-format generations only)
        per_d = []
        for d in DISEASES:
            hp, rp = [], []
            for c in pair_ids:
                fd = gen[c][1]
                rs = target_sentences(tgt[c]).get(d)
                if fd is not None and rs and (fd.get(d) or "").strip():
                    hp.append(tok(fd[d])), rp.append(tok(rs))
            if hp:
                b = corpus_bleu(hp, rp)
                per_d.append(dict(disease=d, 疾病=ZH[d], n=len(hp), **{"BLEU-4": b["BLEU-4"]},
                                  **{"ROUGE-L": float(np.mean([rouge_l(h, r) for h, r in zip(hp, rp)]))}))
        style["per_disease"] = per_d
        metrics["style"] = style

    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=float))
    write_md(out / "report.md", metrics)
    print(f"wrote {out}/metrics.json, report.md", flush=True)


def write_md(path, m):
    L = [f"# 評估結果：{m['name']}", "",
         f"- 生成檔：`{m['gen_file']}`；split：{m['split']}；病例 {m['n_cases']}；排除 {m['dropped'] or '無'}；時間 {m['time']}"]
    if "qwen" in m:
        q = m["qwen"]
        L += [f"- Qwen 反抽：prompt.md md5 `{q['prompt_md5']}`、run_disease14.py md5 `{q['run_disease14_md5']}`，"
              f"成功 {q['n_extracted_ok']} 例、失敗或無標籤 {q['n_failed_or_unlabelled']} 例", ""]
        for definition, title in (("owl", "參考標籤定義 owl（陰性含 not_mentioned，unclear 排除）"),
                                  ("B", "參考標籤定義 B（陰性只算 no / probably_no）")):
            L += [f"## 疾病有無：{title}", "",
                  "| 疾病 | N | 陽性 | 敏感度 | 特異度 | PPV | F1 | 敏感度（寬鬆） | F1（寬鬆） |",
                  "|---|---|---|---|---|---|---|---|---|"]
            S = {r["disease"]: r for r in m["presence"] if r["definition"] == definition and r["mode"] == "strict"}
            Ln = {r["disease"]: r for r in m["presence"] if r["definition"] == definition and r["mode"] == "lenient"}
            for d in DISEASES + ["ALL_micro", "ALL_macro"]:
                s, l_ = S[d], Ln[d]
                L.append(f"| {s['疾病']} | {s.get('N', '')} | {s.get('n_pos', '')} | {fmt(s['sens'])} | {fmt(s['spec'])} | "
                         f"{fmt(s['PPV'])} | {fmt(s['F1'])} | {fmt(l_['sens'])} | {fmt(l_['F1'])} |")
            L += ["", "嚴格：生成反抽為 yes / probably_yes 才算陽性；寬鬆：再加 unclear。", ""]
        L += ["## 關鍵錯誤（原報告陽性、生成寫陰性）", "", "| 疾病 | 原報告陽性 | 關鍵錯誤 | 關鍵錯誤率 | 漏寫（沒提到） | 漏寫率 |",
              "|---|---|---|---|---|---|"]
        for r in m["critical"]:
            L.append(f"| {r['疾病']} | {r['ref_pos']} | {r['critical']} | {fmt(r['critical_rate'])} | {r['omitted']} | "
                     f"{fmt(r['omission_rate'])} |")
        L += ["", "## 屬性正確率（原報告與生成皆陽性的病例）", "",
              "| 疾病 | 屬性 | 皆陽性 | 參考有寫 | 生成有寫 | 正確率 | 生成有寫時正確率 | 差一級內／部分重疊 | 參考沒寫但生成寫了 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in m["attributes"]:
            L.append(f"| {r['疾病']} | {r['屬性']} | {r['both_pos']} | {r['ref_stated']} | {r['gen_stated']} | {fmt(r['acc'])} | "
                     f"{fmt(r['acc_if_stated'])} | {fmt(r['within'])} | {r['gen_only_stated']} |")
        L += ["", "正確率分母 = 參考有寫該屬性的病例（生成沒寫算錯）。椎體節段由骨折句以規則抽出（T/L/S/C + 數字，含範圍）。", "",
              "## Qwen 反抽與參考標籤的一致率（6 類別完全一致／二元一致）", "", "| 疾病 | 6 類一致 | 二元一致（owl 定義） |", "|---|---|---|"]
        for d in DISEASES:
            L.append(f"| {ZH[d]} | {fmt(m['category_agreement_6way'][d])} | {fmt(m['binary_agreement_owl'][d])} |")
        L += [f"| 全部（病例×疾病） | {fmt(m.get('category_agreement_6way_all'))} | |", ""]
    if "style" in m:
        s = m["style"]
        L += ["## 風格相似度（和醫師原句目標比較）", "",
              f"- 配對 {s['n_pairs']} 例（無目標句 {s['n_no_target']} 例）；平均字數 生成 {fmt(s['mean_words_gen'], 1)} / 目標 {fmt(s['mean_words_ref'], 1)}",
              f"- BLEU-1/2/3/4：{fmt(s['BLEU-1'])} / {fmt(s['BLEU-2'])} / {fmt(s['BLEU-3'])} / {fmt(s['BLEU-4'])}（corpus，小寫字詞）",
              f"- ROUGE-L（F1，逐例平均）：{fmt(s['ROUGE-L'])}"]
        if "BERTScore_F" in s:
            L.append(f"- BERTScore P/R/F：{fmt(s['BERTScore_P'])} / {fmt(s['BERTScore_R'])} / {fmt(s['BERTScore_F'])}（{s['BERTScore_model']}）")
        if s.get("per_disease"):
            L += ["", "| 疾病 | 句數 | BLEU-4 | ROUGE-L |", "|---|---|---|---|"]
            for r in s["per_disease"]:
                L.append(f"| {r['疾病']} | {r['n']} | {fmt(r['BLEU-4'])} | {fmt(r['ROUGE-L'])} |")
        L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
