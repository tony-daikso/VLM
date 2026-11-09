#!/usr/bin/env python3
"""P2-1: extract in-scope lesions (liver, kidney, peritoneum, lymph node) from each CG10k report with Qwen on vLLM.

Same client recipe as disease14_shards/run_cg10k.py (guided JSON schema, thinking off, temperature 0), with the
lesion prompt (lesion_prompt.md) and schema below. Resumable: ids with a parse_ok line in --out are skipped;
failed parses / HTTP errors are retried on the next run.

Usage: python run_lesions.py --reports reports_cg10k.json --out lesions_qwen.jsonl \
           [--ids ids.txt] [--url http://127.0.0.1:8004/v1] [--jobs 48] [--max-tokens 4000] [--limit 50]
"""
import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = open(os.path.join(HERE, "lesion_prompt.md")).read()


def enum(*v):
    return {"type": "string", "enum": list(v)}


def strict(obj):
    return {"type": "object", "properties": obj, "required": list(obj), "additionalProperties": False}


CHAR = ["cyst", "complex_cyst", "hemangioma", "metastasis", "hcc", "cholangiocarcinoma", "renal_cell_carcinoma",
        "lymphoma", "other_malignancy", "carcinomatosis", "abscess", "angiomyolipoma", "fnh_adenoma",
        "treated_lesion", "calcified_granuloma", "enlarged_lymph_node", "small_non_enlarged", "other_benign",
        "indeterminate"]
SEGS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]

MEAS = strict({"dims_mm": {"type": "array", "items": {"type": "number"}, "maxItems": 3},
               "qualifier": enum("exact", "up_to", "about", "less_than", "greater_than"),
               "timepoint": enum("current", "prior"),
               "series": {"type": ["integer", "null"]},
               "image": {"type": ["string", "null"]}})
LESION = strict({
    "organ": enum("liver", "kidney", "peritoneum", "lymph_node"),
    "side": enum("left", "right", "bilateral", "right_lobe", "left_lobe", "both_lobes", "caudate",
                 "not_stated", "not_applicable"),
    "liver_segments": {"type": "array", "items": enum(*SEGS), "maxItems": 8},
    "location": {"type": "string"},
    "ln_region": enum("abdominal", "pelvic", "inguinal", "thoracic", "neck", "axillary", "multiple_regions",
                      "not_stated", "not_applicable"),
    "count": enum("single", "multiple", "not_stated"),
    "number": {"type": ["integer", "null"]},
    "measurements": {"type": "array", "items": MEAS, "maxItems": 8},
    "character": enum(*CHAR),
    "malignancy": enum("malignant", "suspicious", "benign", "indeterminate"),
    "comparison": enum("new", "stable", "progression", "regression", "mixed", "resolved", "not_compared"),
    "in_impression": {"type": "boolean"},
    "quote": {"type": "string"},
})
SCHEMA = strict({"abdomen_covered": {"type": "boolean"},
                 "comparison_available": {"type": "boolean"},
                 "lesions": {"type": "array", "items": LESION, "maxItems": 40}})


def ask(report, a):
    body = {"model": "qwen", "temperature": 0, "max_tokens": a.max_tokens,
            "messages": [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": f"# Report\n\n{report}"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "lesions", "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    raw = r["choices"][0]["message"]["content"]
    try:
        ans = json.loads(raw)
        ok = isinstance(ans.get("lesions"), list)
    except Exception:  # noqa: BLE001
        ans, ok = None, False
    return {"answer": ans, "parse_ok": ok, "raw": None if ok else raw, "usage": r["usage"],
            "finish": r["choices"][0]["finish_reason"], "sec": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, help="{EXAMID: {report, PID, row}}")
    ap.add_argument("--ids", help="EXAMIDs in processing order (default: all, by row)")
    ap.add_argument("--url", default="http://127.0.0.1:8004/v1")
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    reports = json.load(open(a.reports))
    ids = [l.strip() for l in open(a.ids) if l.strip()] if a.ids else sorted(reports, key=lambda k: reports[k]["row"])
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                j = json.loads(l)
            except Exception:  # noqa: BLE001  half-written last line after a kill
                continue
            if j.get("parse_ok"):
                done.add(j["study_id"])
    todo = [s for s in ids if s not in done][: a.limit or None]
    print(f"[{time.strftime('%m-%d %H:%M')}] {len(todo)} to do, {len(done)} done", flush=True)

    t0, n, bad = time.time(), [0], [0]

    def one(sid):
        rec = reports[sid]
        try:
            r = ask(rec["report"], a)
        except Exception as e:  # noqa: BLE001  log and keep going; rerun picks it up
            print(f"{sid} FAILED {str(e)[:300]}", flush=True)
            return
        with open(a.out, "a") as f:
            f.write(json.dumps({"study_id": sid, "PID": rec["PID"], "row": rec["row"], **r}, ensure_ascii=False) + "\n")
        n[0] += 1
        bad[0] += not r["parse_ok"]
        if n[0] % 50 == 0 or n[0] == len(todo):
            el = time.time() - t0
            print(f"[{time.strftime('%m-%d %H:%M')}] {n[0]}/{len(todo)}  {n[0] / el * 60:.1f} rep/min  "
                  f"parse_fail {bad[0]}  eta {(len(todo) - n[0]) * el / n[0] / 60:.0f} min", flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
