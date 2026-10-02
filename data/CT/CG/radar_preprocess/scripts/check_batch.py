"""Sanity-check each finished batch of cg_report_combined.jsonl.

Checks the lines added since the last run (state in logs/check_batch.state) in
blocks of --batch lines, prints one summary line per block plus WARN lines,
and appends everything to logs/check_batch.log.

    python3 check_batch.py            # check new batches
    python3 check_batch.py --all      # re-check everything from the start
"""

import argparse
import sys
import collections
import json
import os
import re

D = "/datadrive/VLM/data/CT/CG/radar_preprocess"
PROGRESS = f"{D}/cg_report_combined.jsonl"
STATE = f"{D}/logs/check_batch.state"
LOG = f"{D}/logs/check_batch.log"

# expected ranges, from the 200-report strict pilot (abdomen 6.6, chest 5.1 organs / report)
LIMITS = {
    "failed": 0.01,          # result is null after retry
    "avg_yes": (3.0, 10.0),  # mentioned organs per report
    "zero_yes": 0.05,        # reports with no organ mentioned (abdomen/chest)
    "abnormal": (0.15, 0.70),  # share of mentioned organs that are abnormal
    "not_in_report": 0.03,   # descriptions whose words are mostly not in the report
    "placeholder": 0.005,    # "not mentioned" / "no findings" style descriptions
    "empty_desc": 0.005,
    "meta": 0.002,           # model's own commentary, e.g. "(Note: the report ...)"
}
PLACEHOLDER = re.compile(r"not (specifically )?(mentioned|described|reported)( in the report)?\.?$|"
                         r"no (description|information|findings? (mentioned|reported))|^n/?a$", re.I)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_llm_preprocess_combined import is_meta  # noqa: E402


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def check(recs, first_line):
    n = len(recs)
    failed = sum(r["result"] is None for r in recs)
    ok = [r for r in recs if r["result"] is not None]
    yes = [len(r["result"]) for r in ok]
    organs = [(r, o, v) for r in ok for o, v in r["result"].items()]
    abnormal = sum(v["status"] == "abnormal" for _, _, v in organs)
    empty = sum(not v["description"].strip() or v["description"].startswith("未处理成功")
                for _, _, v in organs)
    placeholder = [(r, o, v) for r, o, v in organs if PLACEHOLDER.search(v["description"])]
    meta = [(r, o, v) for r, o, v in organs if is_meta(v["description"])]
    not_in = []
    for r, o, v in organs:
        w = words(v["description"])
        if w and len(w & words(r["report"])) / len(w) < 0.6:
            not_in.append((r, o, v))

    m = max(len(organs), 1)
    stats = {
        "failed": failed / n,
        "avg_yes": sum(yes) / max(len(yes), 1),
        "zero_yes": sum(y == 0 for y in yes) / max(len(yes), 1),
        "abnormal": abnormal / m,
        "not_in_report": len(not_in) / m,
        "placeholder": len(placeholder) / m,
        "empty_desc": empty / m,
        "meta": len(meta) / m,
    }
    warns = []
    for k, lim in LIMITS.items():
        v = stats[k]
        bad = not (lim[0] <= v <= lim[1]) if isinstance(lim, tuple) else v > lim
        if bad:
            warns.append(f"{k}={v:.3f} (limit {lim})")

    top = collections.Counter(o for _, o, _ in organs).most_common(5)
    line = (f"lines {first_line + 1}-{first_line + n}: failed {failed}, yes/report {stats['avg_yes']:.2f}, "
            f"0-organ {stats['zero_yes']:.1%}, abnormal {stats['abnormal']:.1%}, "
            f"not-in-report {len(not_in)}, placeholder {len(placeholder)}, meta {len(meta)}, empty {empty} | "
            f"top: {', '.join(f'{o} {c}' for o, c in top)}")
    examples = []
    if warns:
        for tag, lst in (("not-in-report", not_in), ("placeholder", placeholder), ("meta", meta)):
            for r, o, v in lst[:3]:
                examples.append(f"   e.g. {tag} {r['pid']} {o}: {v['description'][:120]}")
        for r in [r for r in recs if r["result"] is None][:3]:
            examples.append(f"   e.g. failed {r['pid']}: {r.get('raw', '')[:120]}")
    return line, warns, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    done = 0 if args.all or not os.path.exists(STATE) else int(open(STATE).read().strip() or 0)
    lines = []
    with open(PROGRESS) as f:
        for line in f:
            if not line.endswith("\n"):  # still being written
                break
            lines.append(line)
    out = []
    while len(lines) - done >= args.batch:
        recs = []
        for line in lines[done:done + args.batch]:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        line, warns, examples = check(recs, done)
        bad_json = args.batch - len(recs)
        if bad_json:
            warns.append(f"{bad_json} unreadable lines")
        out.append(("WARN " if warns else "OK   ") + line)
        out += [f"   WARN {w}" for w in warns] + examples
        done += args.batch
    if out:
        with open(LOG, "a") as f:
            f.write("\n".join(out) + "\n")
        print("\n".join(out), flush=True)
    open(STATE, "w").write(str(done))


if __name__ == "__main__":
    main()
