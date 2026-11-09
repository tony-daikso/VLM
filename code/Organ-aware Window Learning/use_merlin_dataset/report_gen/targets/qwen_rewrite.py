#!/usr/bin/env python3
"""Qwen fallback for target sentences the rules could not clean (comparison wording / sizes / se-im removal left a
broken or meaning-losing fragment). Minimal-edit rewrite, doctor's wording kept. Only sentences of positive /
uncertain findings are sent (listed by build_targets.py in qwen_rewrite_requests.jsonl).

Endpoint: OpenAI-compatible vLLM (served-model-name "qwen"), e.g. http://127.0.0.1:18003/v1 through an ssh tunnel.
No credentials are needed or stored. Resumable: requests already present in --out are skipped.
usage: python qwen_rewrite.py --req <requests.jsonl> --out <fixes.jsonl> [--url ...] [--workers 6]
"""
import argparse, concurrent.futures as cf, json, os, threading, time, urllib.request

PROMPT = """You edit ONE sentence taken from an abdominal CT report. The edited sentence becomes a training target for a
model that sees ONLY the current CT: it has no prior imaging and no image coordinates.

Make the smallest possible edit:
1. Remove comparison with prior imaging: stable, unchanged, again seen, redemonstrated, new, newly, interval,
   increased / decreased / improved / worsened / progressed / resolving (as change over time), "compared with ...",
   "since <date>", "previously X", "now X". When the sentence describes a change, keep the CURRENT state in the
   report's own words (a change ending in "now <grade>" becomes "<grade> <finding>").
2. Remove series / image numbers and date placeholders.
3. Remove measured sizes (mm / cm) EXCEPT the abdominal aortic diameter and the spleen size, which are kept.
4. Change nothing else: keep every other word, the order, side, grade, vertebral level, location, cause and the
   hedge words (likely, possible, may ...). Do not add findings, do not summarise, do not fix style.
5. Return a grammatical sentence (or the same few sentences). If nothing about the current exam remains, return "".

Return JSON {"sentence": "..."}."""
SCHEMA = {"type": "object", "properties": {"sentence": {"type": "string"}}, "required": ["sentence"]}


def ask(url, raw, timeout=120):
    body = dict(model="qwen", temperature=0, max_tokens=400,
                messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": raw}],
                response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": SCHEMA}},
                chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        js = json.loads(r.read())
    return json.loads(js["choices"][0]["message"]["content"])["sentence"].strip(), js.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--req", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18003/v1")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            x = json.loads(l)
            done.add((x["case_id"], x["i"]))
    todo = [json.loads(l) for l in open(a.req)]
    todo = [x for x in todo if (x["case_id"], x["i"]) not in done]
    lock = threading.Lock()
    f = open(a.out, "a")
    n = dict(sent=0, ok=0, err=0, prompt_tokens=0, completion_tokens=0)
    t0 = time.time()

    def work(x):
        for attempt in range(3):
            try:
                s, u = ask(a.url, x["raw"])
                with lock:
                    n["sent"] += 1
                    n["ok"] += 1
                    n["prompt_tokens"] += u.get("prompt_tokens", 0)
                    n["completion_tokens"] += u.get("completion_tokens", 0)
                    f.write(json.dumps(dict(case_id=x["case_id"], i=x["i"], split=x["split"], clean=s or None), ensure_ascii=False) + "\n")
                    f.flush()
                return
            except Exception as e:  # noqa
                with lock:
                    n["sent"] += 1
                    n["err"] += 1
                time.sleep(5 * (attempt + 1))

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for k, _ in enumerate(ex.map(work, todo)):
            if k % 200 == 0:
                print(f"{k}/{len(todo)} {n} {time.time() - t0:.0f}s", flush=True)
    f.close()
    print("DONE", json.dumps(n), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
