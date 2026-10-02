"""RADAR report preprocess for our CG CT reports, using a local LLM through vLLM.

Same three steps as the official RADAR_train/preprocess_code (see
data/CT/merlin/RADAR_report_preprocess.md), with the DashScope calls replaced
by vLLM offline batch inference:

    prepare : ct_report_1001.csv          -> cg_report.json
    mention : Step 1, check_organ_mention -> cg_report_mention.json
    parse   : Step 2, report_parsing        -> cg_report_organ_report.json
    normal  : Step 3, report_parsing_normal -> cg_report_organ_normal.json

Output JSON formats are the same as the official ones. Progress is appended to
<output>.jsonl, so an interrupted run resumes where it stopped; the .json is
written at the end (or with `--finalize-only`).

Differences from the official scripts:
- The prompt text is the same, but the CT report comes first (`--official-order`
  restores the original order). With the report as a shared prefix, vLLM
  prefix caching reuses it across the 26 organ questions of the same report.
- Greedy decoding (temperature 0) instead of the API default.
- Failed answers are retried once with sampling before being stored as
  "未处理成功...".
"""

import argparse
import json
import os
import re
import time

ORGANS = [
    "adrenal gland", "aorta", "large bowel", "duodenum", "esophagus",
    "gallbladder", "heart", "iliac artery", "iliac vena", "inferior vena cava",
    "kidney", "liver", "lung", "pancreas", "portal vein", "pulmonary artery",
    "rib", "sacrum", "small bowel", "spleen", "stomach", "trachea", "bladder",
    "cervical vertebrae", "thoracic vertebrae", "lumbar vertebrae",
]

KNOWLEDGE = """Supplementary anatomical knowledge:
- The large bowel includes the cecum, colon, rectum, and anal canal. The cecum includes the appendix, so information related to the appendix should also be categorized under the large intestine.
- The small bowel includes the jejunum, and ileum.
- The splenic vein is part of the portal venous system.
- C1 to C7 refer to the cervical vertebrae.
- T1 to T12 refer to the thoracic vertebrae.
- L1 to L5 refer to the lumbar vertebrae.
- Pleural effusion is considered a description relating to the lungs."""

INSTRUCTION = {
    "mention": """Please determine whether the given CT report mentions the organ ({organ}).
Simply answer "yes" or "no". Do not add diagnoses or summaries.""",
    "parse": """From the given CT report, extract the description information related to the specific anatomy ({organ}). Please follow these guidelines:
- Precise extraction: Extract only the description relevant to {organ} directly from the report.
- Focus on the affected site: If the report mentions a specific part of {organ}, make sure it is included in the extracted information.
- Concise and clear: Directly extract content from the report, avoiding unnecessary explanations or background information.
- Formatting requirement: Provide the information in the format “{organ}: description”. Ensure that {organ} is used consistently as the prefix for the entry. Even if the organ has multiple distinct parts or bilateral characteristics, treat it as a whole and return only one comprehensive description for {organ}.""",
    "normal": """From the given CT report, determine whether the specified anatomy ({organ}) is normal or abnormal.
Please answer directly with "normal" or "abnormal". Do not add diagnoses or summaries.""",
}

MAX_TOKENS = {"mention": 4, "parse": 512, "normal": 4}
FAIL = "未处理成功"


def flatten(report):
    """Same normalisation as the official scripts (one line, single spaces)."""
    return re.sub(r"\s{2,}", " ", report.strip().replace("\n", " "))


def build_prompt(step, report, organ, official_order):
    inst = INSTRUCTION[step].format(organ=organ)
    if step == "mention":
        rep = f"CT report: ({report})"
    else:
        rep = f"CT report:\n{report}"
    if official_order:
        return f"{inst}\n\n{KNOWLEDGE}\n\n{rep}"
    return f"{rep}\n\n{KNOWLEDGE}\n\n{inst}"


def parse_answer(step, text, organ):
    """Returns the stored value, or None if the answer is not usable."""
    text = text.strip().replace("\n", " ").strip()
    if step == "mention":
        t = text.lower().strip(" .\"'*")
        return t if t in ("yes", "no") else None
    if step == "normal":
        t = text.lower().strip(" \"'*")
        if t.startswith("abnormal"):
            return "abnormal"
        if t.startswith("normal"):
            return "normal"
        return None
    # parse: "{organ}: description"
    t = text.replace("：", ":").replace("**", "").strip(" \"“”")
    segs = t.split(":")
    if len(segs) < 2 or segs[0].strip().lower() != organ:
        return None
    desc = ":".join(segs[1:]).strip()
    return desc or None


# ---------------------------------------------------------------- prepare

def split_impression(report):
    m = re.search(r"\**\s*impression\s*:?\s*\**\s*:?", report, flags=re.I)
    if m is None:
        return report.strip(), ""
    return report[:m.start()].strip(), report[m.end():].strip()


def prepare(args):
    import pandas as pd
    df = pd.read_csv(args.csv)
    data = {}
    for i, row in enumerate(df.itertuples(index=False)):
        report = str(row.report)
        findings, impression = split_impression(report)
        data[f"CG{i:07d}"] = {
            "report": report, "findings": findings, "impression": impression,
            "split": "train", "type": row.type,
            "year": int(row.year), "month": int(row.month),
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump(data, open(args.output, "w"), ensure_ascii=False)
    print(f"{len(data)} reports -> {args.output}")


# ---------------------------------------------------------------- LLM steps

def load_progress(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # partial last line after a crash
                    continue
                done[r["pid"]] = r
    return done


def finalize(step, progress_path, output):
    done = load_progress(progress_path)
    out = {}
    for pid, r in done.items():
        if step == "mention":
            out[pid] = {"report": r["report"], "mention": r["result"]}
        else:
            out[pid] = {"report": r["report"], **r["result"]}
    json.dump(out, open(output, "w"), ensure_ascii=False)
    n_fail = sum(1 for r in done.values() for v in r["result"].values()
                 if isinstance(v, str) and v.startswith(FAIL))
    print(f"{len(out)} reports -> {output}  (failed answers: {n_fail})")


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
    )


def generate(llm, prompts, max_tokens, temperature):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, top_p=0.8 if temperature else 1.0,
                        max_tokens=max_tokens, seed=0)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(convs, sp, use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False})
    return [o.outputs[0].text for o in outs]


def run_step(step, args, llm=None):
    progress_path = args.output + ".jsonl"
    if args.finalize_only:
        finalize(step, progress_path, args.output)
        return llm

    data = json.load(open(args.data))
    pids = [p for p, v in data.items()
            if v["split"] == "train" and isinstance(v["report"], str)
            and (not args.types or v["type"] in args.types)]
    if args.limit:
        pids = pids[:args.limit]

    mention = None
    if step != "mention":
        mention = {r["pid"]: r["result"]
                   for r in load_progress(args.mention + ".jsonl").values()}

    done = load_progress(progress_path)
    todo = [p for p in pids if p not in done]
    if step != "mention":
        todo = [p for p in todo if p in mention]
    print(f"[{step}] {len(pids)} reports, {len(done)} done, {len(todo)} to do")
    if not todo:
        finalize(step, progress_path, args.output)
        return llm

    if llm is None:
        llm = make_llm(args)

    t0 = time.time()
    n_done = 0
    with open(progress_path, "a") as fout:
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            jobs = []  # (pid, organ)
            reports = {}
            for pid in chunk:
                reports[pid] = flatten(data[pid]["report"])
                organs = ORGANS if step == "mention" else \
                    [o for o, s in mention[pid].items() if s == "yes"]
                jobs += [(pid, o) for o in organs]

            prompts = [build_prompt(step, reports[p], o, args.official_order)
                       for p, o in jobs]
            texts = generate(llm, prompts, MAX_TOKENS[step], 0.0)
            answers = [parse_answer(step, t, o) for t, (_, o) in zip(texts, jobs)]

            # one sampled retry for unusable answers
            bad = [i for i, a in enumerate(answers) if a is None]
            if bad:
                retry = generate(llm, [prompts[i] for i in bad], MAX_TOKENS[step], 0.7)
                for i, t in zip(bad, retry):
                    a = parse_answer(step, t, jobs[i][1])
                    if a is None:
                        a = f"{FAIL}：{texts[i].strip()[:200]}"
                    answers[i] = a

            results = {pid: {} for pid in chunk}
            for (pid, o), a in zip(jobs, answers):
                results[pid][o] = a
            for pid in chunk:
                fout.write(json.dumps({"pid": pid, "report": reports[pid],
                                       "result": results[pid]},
                                      ensure_ascii=False) + "\n")
            fout.flush()

            n_done += len(chunk)
            dt = time.time() - t0
            eta = dt / n_done * (len(todo) - n_done)
            print(f"[{step}] {len(done) + n_done}/{len(pids)}  {len(jobs)} prompts, "
                  f"{len(bad)} retried  |  {n_done / dt:.2f} reports/s, "
                  f"ETA {eta / 3600:.1f} h", flush=True)

    finalize(step, progress_path, args.output)
    return llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "mention", "parse", "normal", "all"])
    ap.add_argument("--csv", default="/datadrive/VLM/data/CT/CG/ct_report_1001.csv")
    ap.add_argument("--outdir", default="/datadrive/VLM/data/CT/CG/radar_preprocess")
    ap.add_argument("--data", help="cg_report.json (default: <outdir>/cg_report.json)")
    ap.add_argument("--output", help="output json of a single step")
    ap.add_argument("--types", nargs="*", help="only these report types, e.g. abdomen chest")
    ap.add_argument("--limit", type=int, help="only the first N reports (pilot runs)")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _pilot")
    ap.add_argument("--official-order", action="store_true",
                    help="put the CT report at the end of the prompt, as in the official code")
    ap.add_argument("--finalize-only", action="store_true",
                    help="only rebuild the .json from the .jsonl progress file")
    ap.add_argument("--chunk", type=int, default=2000, help="reports per vLLM batch")
    ap.add_argument("--model", default="/root/models/Qwen3.8-27B-FP8")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=160)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    args = ap.parse_args()

    d = args.outdir
    args.data = args.data or f"{d}/cg_report.json"
    names = {"mention": f"{d}/cg_report_mention{args.tag}.json",
             "parse": f"{d}/cg_report_organ_report{args.tag}.json",
             "normal": f"{d}/cg_report_organ_normal{args.tag}.json"}
    args.mention = names["mention"]

    if args.step == "prepare":
        args.output = args.output or args.data
        prepare(args)
        return

    steps = ["mention", "parse", "normal"] if args.step == "all" else [args.step]
    llm = None
    for s in steps:
        args.output = names[s] if (args.step == "all" or not args.output) else args.output
        llm = run_step(s, args, llm)


if __name__ == "__main__":
    main()
