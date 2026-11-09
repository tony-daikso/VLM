#!/usr/bin/env python3
"""v3: Qwen sentence selection for disputed (case, disease) pairs (rule/label conflicts and rule 'other').

For each request the model sees the full report, the target finding with its definition, the final status (decided by
the disease14 labels) and the labeler's quotes, and returns the report's own sentences about that finding that support
the status, copied verbatim. Spirit of the CG std14 prompt, adapted to Merlin (organ-section reports, strict
not-mentioned definition). Sentences that are not verbatim substrings of the report are dropped by build_targets.py
(and counted here).

Endpoint: OpenAI-compatible vLLM, served-model-name "qwen" (e.g. http://127.0.0.1:18003/v1 via ssh tunnel). No
credentials. Resumable: requests already in --out are skipped.
usage: python qwen_select.py --req <qwen_select_requests.jsonl> --reports <reports.json> --out <qwen_select.jsonl>
       [--ids-file <subset ids>] [--workers 8]
"""
import argparse, concurrent.futures as cf, json, os, re, threading, time, urllib.request

DEF = {
    "hepatic_steatosis": "fatty liver / hepatic steatosis / fatty infiltration, diffuse or focal (decreased liver attenuation described as fat counts).",
    "surgically_absent_gallbladder": "prior cholecystectomy; gallbladder surgically absent (surgical clips in the gallbladder fossa, gallbladder remnant after surgery). A gallbladder that is described (normal, distended, wall thickening, stones) is present: negative.",
    "splenomegaly": "enlarged spleen. Statements only about splenic lesions say nothing about spleen size.",
    "hydronephrosis": "dilatation of the renal pelvis / calyces, any grade, either kidney (hydroureteronephrosis, transplant kidney). An extrarenal pelvis or minimal pelviectasis without caliectasis is below the cut (negative).",
    "abdominal_aortic_aneurysm": "abdominal aorta 3.0 cm or more, or called an aneurysm, including a treated aneurysm with an endograft. Ectatic or 2.5-2.9 cm is below the cut. Aneurysms of other arteries (iliac, splenic, thoracic) alone are negative.",
    "atherosclerosis": "atherosclerosis / calcified plaque / vascular calcification of the aorta or its non-coronary branches. Coronary calcification alone does not count.",
    "osteopenia": "diffuse low bone density / osteopenia / osteoporosis / demineralisation.",
    "bowel_obstruction": "small or large bowel obstruction, complete or partial (dilated loops with a transition point). Ileus, pseudo-obstruction or dilated loops without a transition point are below the cut (negative).",
    "appendicitis": "acute appendicitis, including perforated. A normal appendix, appendectomy, mucocele or an appendix not seen without secondary signs is negative.",
    "ascites": "free fluid in the peritoneal cavity. Trace / small physiologic pelvic fluid is below the cut. Loculated collections, abscesses, cysts or haematomas alone are not ascites. Pleural / pericardial effusion is not ascites.",
    "free_air": "pneumoperitoneum, i.e. free gas in the peritoneal cavity outside the bowel (postoperative free air counts). Gas inside an abscess, collection or fistula is not free air.",
    "gallstones": "cholelithiasis (stones in the gallbladder). Sludge alone is below the cut. A surgically absent gallbladder means no gallstones.",
    "biliary_ductal_dilation": "intra- or extrahepatic biliary duct dilatation. Mild prominence called normal for age or due to cholecystectomy is below the cut. Pancreatic duct dilatation alone does not count.",
    "fracture": "any fracture (acute, subacute or old; vertebral compression fracture; vertebroplasty for fracture).",
}
STATUS_TXT = {"pos": "PRESENT", "uncertain": "UNCERTAIN (hedged / possible)", "neg": "NEGATIVE (absent, normal, or below the definition's cut)"}

PROMPT = """You are an expert abdominal radiologist. You read one CT abdomen/pelvis report written with organ-section headers
(e.g. "Liver and biliary tree: ...", "Spleen: ...") followed by a numbered impression.

Task: for ONE target finding, return the report's own sentences about that finding that support the given status.
The status was already decided by a reference labeler; do not change it, only select sentences.

Rules:
- Copy each sentence character for character from the report (same words, spelling, punctuation). Never rewrite,
  shorten, merge or summarise. Copy a whole sentence, not a fragment. If the sentence is the first one of a section
  and is only a blanket statement (e.g. "Normal."), copy it together with its header ("Spleen: Normal.").
- Select only sentences that state something about the target finding itself, or a blanket normal statement of the
  organ section that covers it (e.g. "Kidneys and ureters: Normal." for hydronephrosis, "Peritoneal cavity: No free
  fluid or free air." for ascites and free air). Do NOT select sentences that only describe something else (lesions,
  cysts, degenerative change, vessel patency, other organs), even if the labeler quoted them.
- The sentences must agree with the status: for PRESENT or UNCERTAIN select the sentences describing the finding;
  for NEGATIVE select the sentences saying it is absent, normal or below the cut.
- Prefer the findings section; add an impression sentence only if it says something about the finding that the
  findings section does not. At most 4 sentences, in report order.
- If no sentence of the report fits, return an empty list.

Return JSON {"sentences": [...]}."""
SCHEMA = {"type": "object", "properties": {"sentences": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
          "required": ["sentences"]}


def loose(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def ask(url, user, timeout=180):
    body = dict(model="qwen", temperature=0, max_tokens=800,
                messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": SCHEMA}},
                chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        js = json.loads(r.read())
    return json.loads(js["choices"][0]["message"]["content"])["sentences"], js.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--req", required=True)
    ap.add_argument("--reports", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-file", default=None, help="optional: only these 'case_id<TAB>disease' pairs")
    ap.add_argument("--url", default="http://127.0.0.1:18003/v1")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    reps = json.load(open(a.reports))
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            x = json.loads(l)
            done.add((x["case_id"], x["disease"]))
    todo = [json.loads(l) for l in open(a.req)]
    if a.ids_file:
        keep = {tuple(l.rstrip("\n").split("\t")) for l in open(a.ids_file) if l.strip()}
        todo = [x for x in todo if (x["case_id"], x["disease"]) in keep]
    todo = [x for x in todo if (x["case_id"], x["disease"]) not in done]
    lock = threading.Lock()
    f = open(a.out, "a")
    n = dict(sent=0, ok=0, err=0, sentences=0, not_verbatim=0, empty=0, prompt_tokens=0, completion_tokens=0)
    t0 = time.time()

    def work(x):
        rep = reps[x["case_id"]]["report"]
        user = (f"# Target finding: {x['disease']}\nDefinition: {DEF[x['disease']]}\n"
                f"Status (fixed): {STATUS_TXT[x['status']]}\nLabeler quotes: {json.dumps(x['quotes'], ensure_ascii=False)}\n\n# Report\n\n{rep}")
        for attempt in range(3):
            try:
                sents, u = ask(a.url, user)
                lr = loose(rep)
                nv = sum(1 for s in sents if loose(re.sub(r"^\s*[A-Z][A-Za-z/,&\-() ]{1,60}:\s*", "", s)) not in lr and loose(s) not in lr)
                with lock:
                    n["sent"] += 1
                    n["ok"] += 1
                    n["sentences"] += len(sents)
                    n["not_verbatim"] += nv
                    n["empty"] += int(not sents)
                    n["prompt_tokens"] += u.get("prompt_tokens", 0)
                    n["completion_tokens"] += u.get("completion_tokens", 0)
                    f.write(json.dumps(dict(case_id=x["case_id"], disease=x["disease"], split=x["split"], status=x["status"],
                                            sentences=sents, not_verbatim=nv), ensure_ascii=False) + "\n")
                    f.flush()
                return
            except Exception:  # noqa
                with lock:
                    n["sent"] += 1
                    n["err"] += 1
                time.sleep(5 * (attempt + 1))

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for k, _ in enumerate(ex.map(work, todo)):
            if k % 500 == 0:
                print(f"{k}/{len(todo)} {n} {time.time() - t0:.0f}s", flush=True)
    f.close()
    print("DONE", json.dumps(n), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
