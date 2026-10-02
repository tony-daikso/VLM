# 在 300 份 LLM 抽取樣本上，比較 report_parsing_rules.OPTS 不同設定的平均 F1
import json, glob, os, re, sys, itertools
from collections import Counter

H = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(os.path.dirname(H))
sys.path.insert(0, f"{D}/scripts")
import report_parsing_rules as R


def norm(s):
    s = re.sub(r"^[a-z ]+:\s*", "", s.strip(), flags=re.I) if s else ""
    s = re.sub(r"\((?:series\s*)?[\d<>A-Z/;,\- ]*\d[\d<>A-Z/;,\- ]*\)", " ", s)
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", s.lower())


def f1(a, b):
    ca, cb = Counter(a), Counter(b)
    if not a and not b:
        return 1.0
    inter = sum((ca & cb).values())
    p, r = inter / max(len(b), 1), inter / max(len(a), 1)
    return 2 * p * r / (p + r) if p + r else 0.0


inp, llm = {}, {}
for f in glob.glob(f"{H}/in/batch_[0-9].json"): inp.update(json.load(open(f)))
for f in glob.glob(f"{H}/out/llm_[0-9].json"): llm.update(json.load(open(f)))
mention = json.load(open(f"{D}/merlin_report_mention_strict.json"))
R.build_header_regex([v["report"] for v in mention.values()])  # same header vocab as the full run


def evaluate():
    scores, per = [], {}
    for p, v in inp.items():
        yes = [o for o in v["organs"] if mention[p]["mention"][o] == "yes"]
        d = R.extract(mention[p]["report"], yes)
        for o in v["organs"]:
            s = f1(norm(llm[p].get(o, "") or ""), norm(d.get(o, "")))
            scores.append(s); per.setdefault(o, []).append(s)
    return sum(scores) / len(scores), {o: sum(x) / len(x) for o, x in per.items()}


if __name__ == "__main__":
    results = []
    for thr, msk in itertools.product([None, 0.5, 0.6, 0.7, 0.8, 0.9], [False, True]):
        R.OPTS.update(dedup_impression=thr, msk_generic=msk)
        m, _ = evaluate()
        results.append((m, thr, msk))
        print(f"dedup_impression={str(thr):4s} msk_generic={msk!s:5s} -> mean F1 {m * 100:.2f}%")
    best = max(results)
    print(f"\nbest: dedup_impression={best[1]} msk_generic={best[2]} ({best[0] * 100:.2f}%)")
