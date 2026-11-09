#!/usr/bin/env python3
"""Report standardisation (PLAN §4.5, narrowed to the 14 target diseases) with Qwen on a vLLM endpoint.

For each CG report, per disease: source_sentences (verbatim), status, normalized_sentence, attributes,
comparison; plus clinical_info / comparison_ref. Same client recipe as disease14_shards/run_cg10k.py
(guided JSON, thinking off, temperature 0, urllib + ThreadPool). Resumable: ids with a parse_ok line in
--out are skipped; failed parses / HTTP errors are retried on the next run.

Usage: python std14.py --reports reports_std_in.json --out std14_<host>.jsonl [--ids ids.txt]
           [--url http://127.0.0.1:8000/v1] [--jobs 12] [--max-tokens 3000] [--limit 20]
"""
import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = open(os.path.join(HERE, "std14_prompt.md")).read()

YN = ["yes", "no", "not_stated"]
GRADE = ["mild", "moderate", "severe", "not_stated"]
NUM = {"type": ["number", "null"]}
STR = {"type": "string"}


def enum(*v):
    return {"type": "string", "enum": list(v)}


ATTR = {
    "hepatic_steatosis": {"grade": enum(*GRADE), "distribution": enum("diffuse", "focal", "not_stated")},
    "surgically_absent_gallbladder": {},
    "splenomegaly": {"grade": enum(*GRADE), "length_mm": NUM},
    "hydronephrosis": {"side": enum("left", "right", "bilateral", "transplant", "not_stated"),
                       "grade": enum(*GRADE), "cause": STR},
    "abdominal_aortic_aneurysm": {"diameter_mm": NUM, "location": STR, "repaired": enum(*YN)},
    "atherosclerosis": {"location": STR, "grade": enum(*GRADE)},
    "osteopenia": {"grade": enum(*GRADE)},
    "bowel_obstruction": {"type": enum("SBO", "LBO", "both", "not_stated"), "partial": enum(*YN), "transition": STR},
    "appendicitis": {"perforated": enum(*YN), "appendicolith": enum(*YN)},
    "ascites": {"amount": enum("trace", "small", "moderate", "large", "not_stated"), "location": STR},
    "free_air": {"postoperative": enum(*YN)},
    "gallstones": {"number": enum("single", "multiple", "not_stated"), "size_mm": NUM},
    "biliary_ductal_dilation": {"site": enum("intrahepatic", "extrahepatic", "both", "not_stated"), "cbd_mm": NUM,
                                "cause": STR},
    "fracture": {"sites": {"type": "array", "items": STR, "maxItems": 8},
                 "vertebral_compression": enum(*YN), "acuity": enum("acute", "old", "mixed", "not_stated")},
}
DISEASES = list(ATTR)
STATUS = ["positive", "uncertain", "negative"]  # not_mentioned = null in the model output
COMPARISON = ["stable", "new", "progression", "regression", "none"]


def strict(obj):
    return {"type": "object", "properties": obj, "required": list(obj), "additionalProperties": False}


def schema():
    dis = strict({d: {"anyOf": [strict({"source_sentences": {"type": "array", "items": STR, "minItems": 1, "maxItems": 4},
                                        "status": enum(*STATUS),
                                        "normalized_sentence": STR,
                                        "attributes": strict(a),
                                        "comparison": enum(*COMPARISON)}),
                                {"type": "null"}]}
                  for d, a in ATTR.items()})
    return strict({"clinical_info": STR, "comparison_ref": STR, "diseases": dis})


def empty_attr(a):
    return {k: ([] if v.get("type") == "array" else "" if v.get("type") == "string" and "enum" not in v
                else None if v is NUM else "not_stated") for k, v in a.items()}


def expand(ans):
    """null (not mentioned) -> full not_mentioned record, so every output has the same shape."""
    for d, a in ATTR.items():
        if ans["diseases"].get(d) is None:
            ans["diseases"][d] = {"source_sentences": [], "status": "not_mentioned", "normalized_sentence": "",
                                  "attributes": empty_attr(a), "comparison": "none"}
    return ans


def ask(report, a):
    body = {"model": "qwen", "temperature": 0, "max_tokens": a.max_tokens,
            "messages": [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": f"# Report\n\n{report}"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "std14", "schema": schema()}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    raw = r["choices"][0]["message"]["content"]
    try:
        ans = json.loads(raw)
        ok = all(d in ans["diseases"] for d in DISEASES)
        ans = expand(ans)
    except Exception:  # noqa: BLE001
        ans, ok = None, False
    return {"answer": ans, "parse_ok": ok, "raw": None if ok else raw, "usage": r["usage"],
            "finish": r["choices"][0]["finish_reason"], "sec": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, help="{EXAMID: {report, PID, LOC, EDATE, row}}")
    ap.add_argument("--ids", help="EXAMIDs in processing order (default: all, by row)")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=3000)
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
                done.add(j["exam_id"])
    todo = [s for s in ids if s not in done][: a.limit or None]
    print(f"{len(todo)} to do, {len(done)} done", flush=True)

    t0, n, bad = time.time(), [0], [0]

    def one(sid):
        rec = reports[sid]
        try:
            r = ask(rec["report"], a)
        except Exception as e:  # noqa: BLE001  log and keep going; rerun picks it up
            print(f"{sid} FAILED {str(e)[:300]}", flush=True)
            return
        with open(a.out, "a") as f:
            f.write(json.dumps({"exam_id": sid, "pid": rec["PID"], "loc": rec["LOC"], "row": rec["row"], **r},
                               ensure_ascii=False) + "\n")
        n[0] += 1
        bad[0] += not r["parse_ok"]
        if n[0] % 20 == 0 or n[0] == len(todo):
            el = time.time() - t0
            print(f"[{time.strftime('%m-%d %H:%M')}] {n[0]}/{len(todo)}  {el / n[0]:.2f}s/report  "
                  f"parse_fail {bad[0]}  eta {(len(todo) - n[0]) * el / n[0] / 60:.0f} min", flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
