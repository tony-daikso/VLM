"""Single-call version of radar_llm_preprocess.py.

The per-organ version asks ~38 questions per report (26 for Step 1, then Steps
2 and 3 per mentioned organ). On one L40S that is ~6 s / report, ~70 days for
the 950k CG reports. Here the three steps are asked in ONE call per report:
the instructions and anatomical knowledge are those of the official prompts,
plus STRICT_RULES (strict mention criterion and a few fixes), and the answer is constrained to a JSON schema (vLLM structured outputs):

    {"<organ>": {"description": "...", "status": "normal" | "abnormal"}, ...}

containing only the organs the report mentions (Step 1 = yes).

Writes the same three files as the official scripts:
    cg_report_mention{tag}.json       {pid: {report, mention: {organ: yes/no}}}
    cg_report_organ_report{tag}.json  {pid: {report, organ: description}}
    cg_report_organ_normal{tag}.json  {pid: {report, organ: normal/abnormal}}
Progress goes to cg_report_combined{tag}.jsonl, so the run can be resumed.
"""

import argparse
import json
import os
import re
import time

from radar_llm_preprocess import FAIL, KNOWLEDGE, ORGANS, flatten, load_progress

# Added on top of the official knowledge (see claude_check/: Qwen vs Claude on
# 200 reports). "Strict" = the report has to name the organ itself, as in
# RADAR_report_preprocess.md section 7.
STRICT_RULES = """Rules for deciding whether an organ is mentioned (strict):
- Answer "yes" only if the report explicitly names the organ, one of its parts or sub-structures, or a finding that is by definition located in it (e.g. hepatic, renal, colonic, splenomegaly, hydronephrosis, cholecystitis).
- General region or system terms do not count for a specific organ: e.g. "gastrointestinal tract", "bowel", "bowel loops", "vasculature", "vessels", "musculoskeletal", "bones", "osseous structures", "spine" without a level, "lower thorax", "urinary tract", "obstructive uropathy".
- An organ used only as a location reference does not count: e.g. "para-aortic / periaortic lymph nodes" is not the aorta, "ureteral stone at the L4 level" is not the lumbar vertebrae, "mass adjacent to the inferior vena cava" is not the inferior vena cava, unless the report describes the organ itself (e.g. invasion, compression, thrombus).
- A hiatal (hiatus) hernia is a description relating to the stomach.
- The pericardium and the coronary arteries are part of the heart.
- Portal hypertension, portosystemic collaterals or shunts (e.g. splenorenal shunt) and varices of the portal system are descriptions relating to the portal vein, and make it abnormal even if the portal vein is patent."""

PROMPT = """CT report:
{report}

{knowledge}

{rules}

For each of the following organs: {organs}

1. Determine whether the given CT report mentions the organ ("yes" or "no"). Answer every organ in the order listed.
2. For each mentioned organ, extract the description information related to that specific anatomy. Please follow these guidelines:
- Precise extraction: Extract only the description relevant to the organ directly from the report.
- Focus on the affected site: If the report mentions a specific part of the organ, make sure it is included in the extracted information.
- Concise and clear: Directly extract content from the report, avoiding unnecessary explanations or background information.
- Even if the organ has multiple distinct parts or bilateral characteristics, treat it as a whole and return only one comprehensive description for the organ.
3. For each mentioned organ, determine whether it is "normal" or "abnormal". Do not add diagnoses or summaries.

Answer in JSON, one entry per organ in the order listed. Use "no" for an organ that is not mentioned:
{{"<organ>": "no", "<organ>": {{"description": "...", "status": "normal" or "abnormal"}}, ...}}"""

SCHEMA = {
    "type": "object",
    "properties": {
        o: {"anyOf": [
            {"const": "no"},
            {"type": "object",
             "properties": {"description": {"type": "string"},
                            "status": {"type": "string", "enum": ["normal", "abnormal"]}},
             "required": ["description", "status"],
             "additionalProperties": False},
        ]}
        for o in ORGANS
    },
    "required": ORGANS,
    "additionalProperties": False,
}


def build_prompt(report, strict=True):
    return PROMPT.format(report=report, knowledge=KNOWLEDGE,
                         rules=STRICT_RULES if strict else "", organs=", ".join(ORGANS))


META = re.compile(r"\((note|nb)\b|the report (does not|doesn't|lists|mentions|states)|"
                  r"not (specifically )?(mentioned|described) in the report", re.I)


def is_meta(desc):
    """Model commentary instead of report text, e.g. "(Note: the report lists 'Colon' but ...)".
    Rare (~0.02% of organs); the organ itself is usually a false mention."""
    return bool(META.search(desc))


def parse_answer(text):
    """Returns {organ: {"description", "status"}} or None."""
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    out = {}
    for o, v in d.items():
        if o not in ORGANS:
            return None
        if v == "no":
            continue
        if not isinstance(v, dict):
            return None
        desc = str(v.get("description", "")).strip()
        status = v.get("status")
        if status not in ("normal", "abnormal"):
            return None
        out[o] = {"description": desc or f"{FAIL}：空描述", "status": status}
    return out


def generate(llm, prompts, max_tokens, temperature):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    sp = SamplingParams(temperature=temperature, top_p=0.8 if temperature else 1.0,
                        max_tokens=max_tokens, seed=0,
                        structured_outputs=StructuredOutputsParams(
                            json=SCHEMA, disable_any_whitespace=True))
    convs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(convs, sp, use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False})
    return [o.outputs[0].text for o in outs]


def finalize(progress_path, names):
    done = load_progress(progress_path)
    mention, parse, normal = {}, {}, {}
    n_fail = n_meta = 0
    for pid, r in done.items():
        res = r["result"]
        if res is not None:  # drop organs whose "description" is model commentary
            n_meta += sum(is_meta(v["description"]) for v in res.values())
            res = {o: v for o, v in res.items() if not is_meta(v["description"])}
        if res is None:  # unusable answer after retry: record like the official scripts
            n_fail += 1
            mention[pid] = {"report": r["report"],
                            "mention": {o: f"{FAIL}：{r.get('raw', '')[:200]}" for o in ORGANS}}
            continue
        mention[pid] = {"report": r["report"],
                        "mention": {o: "yes" if o in res else "no" for o in ORGANS}}
        parse[pid] = {"report": r["report"], **{o: v["description"] for o, v in res.items()}}
        normal[pid] = {"report": r["report"], **{o: v["status"] for o, v in res.items()}}
    for key, obj in (("mention", mention), ("parse", parse), ("normal", normal)):
        json.dump(obj, open(names[key], "w"), ensure_ascii=False)
        print(f"{len(obj)} reports -> {names[key]}")
    print(f"failed reports: {n_fail}, organs dropped as model commentary: {n_meta}")


def make_llm(args):
    from vllm import LLM
    return LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=0,
        speculative_config=({"method": "mtp", "num_speculative_tokens": args.mtp}
                            if args.mtp else None),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/datadrive/VLM/data/CT/CG/radar_preprocess")
    ap.add_argument("--data", help="cg_report.json (default: <outdir>/cg_report.json)")
    ap.add_argument("--types", nargs="*", help="only these report types, processed in this order, e.g. abdomen chest brain")
    ap.add_argument("--limit", type=int, help="only the first N reports (pilot runs)")
    ap.add_argument("--pids", help="json list of pids to run (instead of --types/--limit)")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _pilot")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--lenient", action="store_true",
                    help="official knowledge only, without STRICT_RULES")
    ap.add_argument("--chunk", type=int, default=2000, help="reports per vLLM batch")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--model", default="/root/models/Qwen3.8-27B-FP8")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--mtp", type=int, default=2, help="MTP speculative tokens (0 = off)")
    ap.add_argument("--gpu-mem", type=float, default=0.94)
    args = ap.parse_args()

    d = args.outdir
    data_path = args.data or f"{d}/cg_report.json"
    names = {"mention": f"{d}/cg_report_mention{args.tag}.json",
             "parse": f"{d}/cg_report_organ_report{args.tag}.json",
             "normal": f"{d}/cg_report_organ_normal{args.tag}.json"}
    progress_path = f"{d}/cg_report_combined{args.tag}.jsonl"

    if args.finalize_only:
        finalize(progress_path, names)
        return

    data = json.load(open(data_path))
    if args.pids:
        pids = json.load(open(args.pids))
    else:
        pids = [p for p, v in data.items()
                if v["split"] == "train" and isinstance(v["report"], str)
                and (not args.types or v["type"] in args.types)]
        if args.types:  # run in the order the types are given
            pids.sort(key=lambda p: args.types.index(data[p]["type"]))
        if args.limit:
            pids = pids[:args.limit]

    done = load_progress(progress_path)
    todo = [p for p in pids if p not in done]
    print(f"[combined] {len(pids)} reports, {len(done)} done, {len(todo)} to do", flush=True)

    if todo:
        llm = make_llm(args)
        t0 = time.time()
        n_done = 0
        with open(progress_path, "a") as fout:
            # a crash in the middle of a write leaves a partial last line; start on a
            # new line so the next record is not glued to it
            if fout.tell() > 0:
                with open(progress_path, "rb") as f:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":
                        fout.write("\n")
            for c in range(0, len(todo), args.chunk):
                chunk = todo[c:c + args.chunk]
                reports = [flatten(data[p]["report"]) for p in chunk]
                prompts = [build_prompt(r, not args.lenient) for r in reports]
                texts = generate(llm, prompts, args.max_tokens, 0.0)
                answers = [parse_answer(t) for t in texts]

                bad = [i for i, a in enumerate(answers) if a is None]
                if bad:
                    retry = generate(llm, [prompts[i] for i in bad], args.max_tokens, 0.7)
                    for i, t in zip(bad, retry):
                        answers[i] = parse_answer(t)

                for pid, rep, a, t in zip(chunk, reports, answers, texts):
                    rec = {"pid": pid, "report": rep, "result": a}
                    if a is None:
                        rec["raw"] = t
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                n_done += len(chunk)
                dt = time.time() - t0
                eta = dt / n_done * (len(todo) - n_done)
                n_bad = sum(a is None for a in answers)
                print(f"[combined] {len(done) + n_done}/{len(pids)}  {len(bad)} retried, "
                      f"{n_bad} failed  |  {n_done / dt:.2f} reports/s, "
                      f"ETA {eta / 3600:.1f} h", flush=True)

    finalize(progress_path, names)


if __name__ == "__main__":
    main()
