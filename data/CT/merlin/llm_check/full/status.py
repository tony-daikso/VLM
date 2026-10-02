# 檢查每批輸出是否完整、合法；印出尚未完成或需重跑的批次
import json, glob, os
H = os.path.dirname(os.path.abspath(__file__))
ORG = {"adrenal gland","aorta","large bowel","duodenum","esophagus","gallbladder","heart","iliac artery","iliac vena",
       "inferior vena cava","kidney","liver","lung","pancreas","portal vein","pulmonary artery","rib","sacrum","small bowel",
       "spleen","stomach","trachea","bladder","cervical vertebrae","thoracic vertebrae","lumbar vertebrae"}
ok, bad, todo = [], [], []
for f in sorted(glob.glob(f"{H}/in/batch_*.json")):
    b = os.path.basename(f)[6:9]; o = f"{H}/out/llm_{b}.json"
    if not os.path.exists(o): todo.append(b); continue
    try:
        ids = set(json.load(open(f))); out = json.load(open(o))
        assert set(out) == ids, "id mismatch"
        assert all(isinstance(v, list) and set(v) <= ORG for v in out.values()), "bad organ"
        ok.append(b)
    except Exception as e:
        bad.append((b, str(e)))
print(f"done {len(ok)} | bad {len(bad)} {bad} | todo {len(todo)}")
print("todo:", " ".join(todo))
