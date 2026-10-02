# 合併全量 LLM Step 1 結果 → 官方格式 merlin_report_mention_llm.json
# 來源：full/out/llm_*.json（15,009 份）＋ llm_check/llm_[0-5].json（先前抽查的 300 份）
import json, glob, os

H = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(os.path.dirname(H))  # data/CT/merlin
ORGANS = ["adrenal gland", "aorta", "large bowel", "duodenum", "esophagus", "gallbladder", "heart",
          "iliac artery", "iliac vena", "inferior vena cava", "kidney", "liver", "lung", "pancreas",
          "portal vein", "pulmonary artery", "rib", "sacrum", "small bowel", "spleen", "stomach",
          "trachea", "bladder", "cervical vertebrae", "thoracic vertebrae", "lumbar vertebrae"]

llm = {}
for f in sorted(glob.glob(f"{H}/out/llm_*.json")) + sorted(glob.glob(f"{os.path.dirname(H)}/llm_[0-9].json")):
    llm.update(json.load(open(f)))

rules = json.load(open(f"{D}/merlin_report_mention_strict.json"))  # 取 normalize 後的 report 與完整 ID 清單
missing = [p for p in rules if p not in llm]
assert not missing, f"{len(missing)} reports have no LLM label yet, e.g. {missing[:5]}"

out = {p: {"report": rules[p]["report"],
           "mention": {o: ("yes" if o in llm[p] else "no") for o in ORGANS}} for p in rules}
json.dump(out, open(f"{D}/merlin_report_mention_llm.json", "w"), ensure_ascii=False, indent=4)

n = len(out)
print(f"saved {n} reports -> merlin_report_mention_llm.json")
for name, path in [("strict", "merlin_report_mention_strict.json"), ("lenient", "merlin_report_mention.json")]:
    R = json.load(open(f"{D}/{path}"))
    agree = sum(R[p]["mention"][o] == out[p]["mention"][o] for p in out for o in ORGANS)
    same = sum(R[p]["mention"] == out[p]["mention"] for p in out)
    print(f"LLM vs {name:7s}: cell agreement {agree / (n * 26) * 100:.2f}%, identical reports {same}/{n}")
print(f"{'organ':20s} {'LLM':>7s}")
for o in ORGANS:
    k = sum(v["mention"][o] == "yes" for v in out.values())
    print(f"{o:20s} {k / n * 100:6.1f}%")
